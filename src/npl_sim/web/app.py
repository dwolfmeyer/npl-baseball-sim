"""The NPL web app (FastAPI + Jinja): managers' lineup cards and game results.

Run it with `npl-sim web`. Rosters come from the NPL app database when
`NPL_DATABASE_URL` is set, otherwise from a Django dump (`NPL_ROSTER_DUMP`,
default the current Ducksnorts and McFlys MLB rosters). Position eligibility
and hands come from `data/eligibility/<season>.json` (`npl-sim eligibility`).
"""

from __future__ import annotations

import logging
import os
import random
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from npl_sim.engine.box import box_score
from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.game import simulate_game
from npl_sim.engine.manager import AGGRESSION, BUNT_STRATEGIES, STYLES
from npl_sim.engine.playlog import game_log
from npl_sim.league import load_league_file, team_slots
from npl_sim.playing_time import load_mlb_week
from npl_sim.reporting import gameday
from npl_sim.reporting.week import render_week, standings
from npl_sim.rosters import eligibility
from npl_sim.rosters.eligibility import Eligibility
from npl_sim.rosters.npl_app import NplPlayer
from npl_sim.rosters.sources import NPL_DATABASE_URL_ENV, players_from_database, players_from_dump
from npl_sim.season_stats import (
    EXPANDED,
    RaceRow,
    division_standings,
    expanded_records,
    format_gb,
    format_wcgb,
    league_standings,
    wild_card_race,
)
from npl_sim.simweek import round_robin, sim_week
from npl_sim.stats_table import Query as StatsQuery
from npl_sim.stats_table import build as build_stats_table
from npl_sim.web import lineup, stats
from npl_sim.web.store import Store, slug

ROSTER_DUMP_ENV = "NPL_ROSTER_DUMP"
DEFAULT_DUMP = Path("tests/fixtures/npl_2026_test_teams.json")
SEASON = 2026
SEASON_START = date(SEASON, 3, 30)  # a Monday; the sim weeks of the season to date follow it
log = logging.getLogger(__name__)

TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")
TEMPLATES.env.globals.update(
    gameday_css=gameday.CSS,
    gameday_js=gameday.JS,
    positions=lineup.POSITIONS,
    sides=lineup.SIDES,
    roles=lineup.ROLES,
    usage=lineup.USAGE,
    matchups=lineup.MATCHUPS,
    stints=lineup.STINTS,
    pen_groups=[*lineup.ROLES.items(), ("", "No role")],
    active_limit=lineup.ACTIVE_LIMIT,
    rotation_size=lineup.ROTATION_SIZE,
    pitch_limits=lineup.PITCH_LIMITS,
    tto_hooks=lineup.TTO_HOOKS,
    run_hooks=lineup.RUN_HOOKS,
    leverage_hooks=lineup.LEVERAGE_HOOKS,
    leverage_innings=lineup.LEVERAGE_INNINGS,
    defense_rules=lineup.DEFENSE_RULES,
    sub_rows=lineup.DEFENSIVE_SUB_ROWS,
    sub_innings=lineup.SUB_INNINGS,
    styles=STYLES,
    pinch_innings=lineup.PINCH_INNINGS,
    pinch_leverage=lineup.PINCH_LEVERAGE,
    pinch_gains=lineup.PINCH_GAINS,
    pinch_max=lineup.PINCH_MAX,
    bunt_strategies=BUNT_STRATEGIES,
    steal_levels=AGGRESSION,
    extra_base_levels=lineup.EXTRA_BASE,
)


def load_players() -> list[NplPlayer]:
    if os.environ.get(NPL_DATABASE_URL_ENV):
        return players_from_database()
    return players_from_dump(os.environ.get(ROSTER_DUMP_ENV) or DEFAULT_DUMP)


def abbr(team: str) -> str:
    return team.split()[-1][:3].upper()


def create_app(
    store: Store | None = None,
    players: list[NplPlayer] | None = None,
    elig: Eligibility | None = None,
    clean_root: Path | str | None = None,
) -> FastAPI:
    app = FastAPI(title="NPL Sim")
    store = store or Store()
    players = players if players is not None else load_players()
    elig = elig if elig is not None else eligibility.load(SEASON)
    alignment = load_league_file().for_season(SEASON)
    slots = team_slots(alignment)
    rosters: dict[str, list[NplPlayer]] = {s.team: [] for s in slots}
    for p in players:
        if p.team in rosters and p.is_active_mlb and p.mlb_id is not None:
            rosters[p.team].append(p)
    by_slug = {slug(t): t for t in rosters}
    clean_root = clean_root or os.environ.get(stats.CLEAN_ROOT_ENV) or stats.DEFAULT_CLEAN_ROOT
    splits = stats.SplitsCache(clean_root, SEASON)
    talents = stats.TalentCache(clean_root, SEASON)
    hands = {
        str(p.mlb_id): rec.bats
        for r in rosters.values()
        for p in r
        if (rec := elig.players.get(p.mlb_id)) and rec.bats
    }

    def team_for(team_slug: str) -> str:
        if team_slug not in by_slug or not rosters[by_slug[team_slug]]:
            raise HTTPException(404, "No roster for that team")
        return by_slug[team_slug]

    def card_for(team: str) -> tuple[lineup.LineupCard, bool]:
        saved = store.load_card(team)
        return (saved, True) if saved else (lineup.default_card(rosters[team], elig), False)

    def check(card: lineup.LineupCard, team: str) -> lineup.Check:
        return lineup.validate(card, rosters[team], elig)

    def card_page(request: Request, team: str, card, check, saved: bool, status: int = 200):
        roster = rosters[team]
        hitters = [p for p in roster if not lineup.is_pitcher(p)]
        arms = [p for p in roster if lineup.is_pitcher(p)]
        fields = {}
        lines = splits.get()
        talent = talents.get(hands)
        for h in hitters:
            positions, from_games = lineup.fielding(h, elig)
            fields[h.mlb_id] = {
                "positions": [p for p in lineup.POSITIONS if p in positions],
                "from_games": from_games,
                "games": elig.games_text(h.mlb_id),
                "lines": {side: lines.line(h.mlb_id, side) for side in lineup.SIDES},
                "grades": stats.grades(lineup.glove_positions(h, elig), h.mlb_id, talent),
            }
        return TEMPLATES.TemplateResponse(
            request,
            "partials/card.html" if request.headers.get("HX-Request") else "lineup.html",
            {
                "team": team,
                "team_slug": slug(team),
                "card": card,
                "check": check,
                "saved": saved,
                "saved_at": store.card_saved_at(team),
                "hitters": hitters,
                "arms": arms,
                "names": {a.mlb_id: a.name for a in arms},
                "fields": fields,
                "stats_source": lines.source,
                "active": set(card.active),
                "rotation": [pid for pid in card.rotation if pid is not None],
            },
            status_code=status,
        )

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        teams = [
            {
                "slot": s,
                "slug": slug(s.team),
                "players": len(rosters[s.team]),
                "saved_at": store.card_saved_at(s.team),
            }
            for s in slots
        ]
        return TEMPLATES.TemplateResponse(
            request,
            "home.html",
            {"teams": teams, "games": store.recent_games(), "weeks": store.recent_weeks()},
        )

    @app.get("/teams/{team_slug}/lineup", response_class=HTMLResponse)
    def lineup_card(request: Request, team_slug: str, saved: int = 0):
        team = team_for(team_slug)
        card, _ = card_for(team)
        return card_page(request, team, card, check(card, team), bool(saved))

    @app.post("/teams/{team_slug}/lineup/check", response_class=HTMLResponse)
    async def check_card(request: Request, team_slug: str):
        team = team_for(team_slug)
        card = lineup.LineupCard.from_form(await request.form())
        return card_page(request, team, card, check(card, team), False)

    @app.post("/teams/{team_slug}/lineup/autofill", response_class=HTMLResponse)
    async def autofill_card(request: Request, team_slug: str):
        team = team_for(team_slug)
        card = lineup.LineupCard.from_form(await request.form())
        card = lineup.autofill(card, rosters[team], elig)
        return card_page(request, team, card, check(card, team), False)

    @app.post("/teams/{team_slug}/lineup")
    async def save_card(request: Request, team_slug: str):
        team = team_for(team_slug)
        card = lineup.LineupCard.from_form(await request.form())
        result = check(card, team)
        if not result.ok:
            return card_page(request, team, card, result, False, status=422)
        store.save_card(team, card)
        return RedirectResponse(f"/teams/{team_slug}/lineup?saved=1", status_code=303)

    playable = [t for t in rosters if rosters[t]]

    @app.get("/games/new", response_class=HTMLResponse)
    def new_game(request: Request):
        away = next((t for t in playable if "Ducksnorts" in t), playable[0])
        home = next((t for t in playable if t != away), away)
        return TEMPLATES.TemplateResponse(
            request, "new_game.html", {"teams": playable, "away": away, "home": home}
        )

    @app.post("/games")
    async def play_game(request: Request):
        form = await request.form()
        away, home = str(form.get("away", "")), str(form.get("home", ""))
        if away not in playable or home not in playable or away == home:
            raise HTTPException(400, "Pick two different teams with rosters")
        slot = {side: int(str(form.get(f"{side}_slot") or 1)) - 1 for side in ("away", "home")}
        seed_text = str(form.get("seed") or "").strip()
        seed = int(seed_text) if seed_text.isdigit() else random.randrange(1, 10**6)

        cards = {}
        for side, team in (("away", away), ("home", home)):
            card, _ = card_for(team)
            result = check(card, team)
            if not result.ok:
                raise HTTPException(400, f"{team}'s lineup card has errors: {result.errors[0]}")
            cards[side] = card
        # The listed starter (or opener) sets the other team's batting order.
        starters = {}
        for side, team in (("away", away), ("home", home)):
            by_id = {p.mlb_id: p for p in rosters[team]}
            slots = [(i, pid) for i, pid in enumerate(cards[side].rotation) if pid is not None]
            _, pid = slots[slot[side] % len(slots)]
            starters[side] = by_id[pid]
        teams = {
            side: lineup.team_card(
                team,
                abbr(team),
                cards[side],
                rosters[team],
                slot[side],
                (starters["home" if side == "away" else "away"].throws or "R"),
                elig,
            )
            for side, team in (("away", away), ("home", home))
        }
        events = simulate_game(teams["away"], teams["home"], DEFAULT_ENV, seed)
        box = box_score(events)
        game_id = store.save_game(
            {
                "away": away,
                "home": home,
                "seed": seed,
                "starters": {s: starters[s].name for s in starters},
                "runs": {"away": box.runs[away], "home": box.runs[home]},
                "events": events,
            }
        )
        return RedirectResponse(f"/games/{game_id}", status_code=303)

    @app.post("/weeks")
    async def play_week(request: Request):
        """Sim a test week between every team with a roster, from their lineup
        cards (saved, or the default), at each player's season pace."""
        form = await request.form()
        seed_text = str(form.get("seed") or "").strip()
        seed = int(seed_text) if seed_text.isdigit() else random.randrange(1, 10**6)
        teams = sorted(playable)
        cards = {}
        for team in teams:
            card, _ = card_for(team)
            result = check(card, team)
            if not result.ok:
                raise HTTPException(400, f"{team}'s lineup card has errors: {result.errors[0]}")
            cards[team] = card
        week_rosters = {t: rosters[t] for t in teams}
        talent = talents.get(hands)
        ability = (
            f"Ability: the talent blend for {SEASON} week {talents.week}."
            if talent is not None
            else f"Ability: league average (the talent blend didn't load: {talents.error})."
            if talents.error
            else "Ability: league average (no clean MLB data on this machine)."
        )
        # The sample week is the latest week of clean data (week 27 of 2026 once the
        # season is in): its real PA, starts and relief outs set who can play, and
        # the talent blend above rates them. Without it, a typical week at season pace.
        mlb, playing = None, f"a typical week at each player's {SEASON} pace"
        if talents.week is not None:
            try:
                mlb = load_mlb_week(clean_root, SEASON, talents.week)
                playing = f"the real MLB {SEASON} week {talents.week}"
            except Exception as exc:  # no batting/pitching tables for that week
                log.warning("MLB week %s usage failed: %s", talents.week, exc)
        result = sim_week(cards, week_rosters, elig, round_robin(teams), mlb, seed, talent=talent)
        game_ids = []
        for g in result.games:
            events = g["events"]
            pitchers = {
                side: {p["id"]: p["name"] for p in events[0][side]["pitchers"]}
                for side in ("away", "home")
            }
            starters = {
                side: pitchers[side].get(g["moves"][g[side]].starter, "")
                for side in ("away", "home")
            }
            final = events[-1]
            game_ids.append(
                store.save_game(
                    {
                        "away": g["away"],
                        "home": g["home"],
                        "seed": seed,
                        "starters": starters,
                        "runs": {"away": final["away"], "home": final["home"]},
                        "events": events,
                    }
                )
            )
        # Fold the week into the season to date, in date order, under full names.
        names = {str(p.mlb_id): p.name for r in week_rosters.values() for p in r}
        season, weeks_played = store.load_season()
        week_start = SEASON_START + timedelta(weeks=weeks_played)
        for g in sorted(result.games, key=lambda g: g["day"]):
            season.add_game(week_start + timedelta(days=g["day"] - 1), box_score(g["events"]))
        for totals in (season.batters, season.pitchers):
            for pid, line in totals.items():
                line.name = names.get(pid, line.name)
        store.save_season(season, weeks_played + 1)
        page = render_week(
            result,
            "NPL test week",
            f"Playing time: {playing}. {ability} Seed {seed}.",
            link=lambda i: f"/games/{game_ids[i]}",
            names=names,
        )
        page = page.replace("<main>", '<main><p class="sub"><a href="/">Back to teams</a></p>', 1)
        week_id = store.save_week(
            {
                "seed": seed,
                "teams": teams,
                "games": game_ids,
                "standings": [list(row) for row in standings(result)],
                "page": page,
            }
        )
        return RedirectResponse(f"/weeks/{week_id}", status_code=303)

    @app.get("/weeks/{week_id}", response_class=HTMLResponse)
    def week_page(week_id: str):
        week = store.load_week(week_id)
        if week is None:
            raise HTTPException(404, "No such week")
        return HTMLResponse(week["page"])

    def season_page(request: Request, template: str, heading: str, **context):
        season, weeks = store.load_season()
        return TEMPLATES.TemplateResponse(
            request,
            template,
            {
                "heading": heading,
                "weeks": weeks,
                "games": season.games,
                "season": season,
                "abbr": abbr,
                **context,
            },
        )

    team_league = {t: lg.name for lg in alignment.leagues for d in lg.divisions for t in d.teams}
    league_names = [lg.name for lg in alignment.leagues]
    last_seed = len(alignment.leagues[0].divisions) + alignment.wild_cards

    @app.get("/standings", response_class=HTMLResponse)
    def standings_page(request: Request, view: str = "division", expanded: int = 0):
        season, _ = store.load_season()
        view = view if view in ("division", "wildcard", "league") else "division"
        race = {name: rows for name, rows in wild_card_race(season, alignment)}
        wcgb = {r.team: r.wcgb for rows in race.values() for r in rows}
        seed = {r.team: r.seed for rows in race.values() for r in rows}

        def line(row):  # a standings row with its wild card position
            return RaceRow(row.team, row.record, row.gb, wcgb.get(row.team), seed.get(row.team))

        if view == "wildcard":
            tables = [(f"{name} wild card", rows) for name, rows in race.items()]
        elif view == "league":
            tables = [
                (f"{name} standings", [line(r) for r in rows])
                for name, rows in league_standings(season, alignment)
            ]
        else:
            tables = [
                (f"{lg} {div}", [line(r) for r in rows])
                for lg, div, rows in division_standings(season, alignment)
            ]
        return season_page(
            request,
            "standings.html",
            "Standings",
            view=view,
            expanded=bool(expanded),
            tables=tables,
            wild_cards=alignment.wild_cards,
            last_seed=last_seed,
            extra=expanded_records(season, alignment) if expanded else {},
            expanded_columns=EXPANDED,
            format_gb=format_gb,
            format_wcgb=format_wcgb,
        )

    @app.get("/stats", response_class=HTMLResponse)
    def stats_page(
        request: Request,
        kind: str = "hitting",
        group: str = "players",
        league: str = "",
        team: str = "",
        qualified: int = 1,
        sort: str = "",
        dir: str = "",
        page: int = 1,
    ):
        season, _ = store.load_season()
        query = StatsQuery(
            kind=kind if kind in ("hitting", "pitching") else "hitting",
            group="teams" if group == "teams" else "players",
            league=league if league in league_names else "",
            team=team if team in team_league else "",
            qualified=bool(qualified),
            sort=sort,
            descending={"asc": False, "desc": True}.get(dir),
            page=page,
        )
        table = build_stats_table(season, query, team_league)
        params = {
            "kind": query.kind,
            "group": query.group,
            "league": query.league,
            "team": query.team,
            "qualified": int(query.qualified),
            "sort": sort,
            "dir": dir,
            "page": page,
        }

        def link(**changes) -> str:
            merged = {k: v for k, v in {**params, **changes}.items() if v != ""}
            return "/stats?" + urlencode(merged)

        teams = sorted(t for t in team_league if not query.league or team_league[t] == query.league)
        return season_page(
            request,
            "stats.html",
            "Stats",
            q=query,
            table=table,
            link=link,
            league_names=league_names,
            team_names=teams,
        )

    @app.get("/leaders")
    @app.get("/season")
    def old_stats_pages():
        return RedirectResponse("/stats", status_code=307)

    @app.post("/season/reset")
    def reset_season():
        store.reset_season()
        return RedirectResponse("/standings", status_code=303)

    @app.get("/games/{game_id}", response_class=HTMLResponse)
    def game_page(request: Request, game_id: str):
        game = store.load_game(game_id)
        if game is None:
            raise HTTPException(404, "No such game")
        box = box_score(game["events"])
        return TEMPLATES.TemplateResponse(
            request,
            "game.html",
            {"game": game, "box": box, "body": gameday.game_body(box, game_log(game["events"]))},
        )

    return app
