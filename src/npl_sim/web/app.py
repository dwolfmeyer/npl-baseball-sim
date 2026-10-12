"""The NPL web app (FastAPI + Jinja): managers' lineup cards and game results.

Run it with `npl-sim web`. Rosters come from the NPL app database when
`NPL_DATABASE_URL` is set, otherwise from a Django dump (`NPL_ROSTER_DUMP`,
default the current Ducksnorts and McFlys MLB rosters). Position eligibility
and hands come from `data/eligibility/<season>.json` (`npl-sim eligibility`).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from npl_sim.engine.box import box_score
from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.game import simulate_game
from npl_sim.engine.playlog import game_log
from npl_sim.league import load_league_file, team_slots
from npl_sim.reporting import gameday
from npl_sim.rosters import eligibility
from npl_sim.rosters.eligibility import Eligibility
from npl_sim.rosters.npl_app import NplPlayer
from npl_sim.rosters.sources import NPL_DATABASE_URL_ENV, players_from_database, players_from_dump
from npl_sim.web import lineup
from npl_sim.web.store import Store, slug

ROSTER_DUMP_ENV = "NPL_ROSTER_DUMP"
DEFAULT_DUMP = Path("tests/fixtures/npl_2026_test_teams.json")
SEASON = 2026

TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")
TEMPLATES.env.globals.update(
    gameday_css=gameday.CSS,
    gameday_js=gameday.JS,
    positions=lineup.POSITIONS,
    sides=lineup.SIDES,
    roles=lineup.ROLES,
    pen_groups=[*lineup.ROLES.items(), ("", "No role")],
    active_limit=lineup.ACTIVE_LIMIT,
    rotation_size=lineup.ROTATION_SIZE,
    pitch_limits=lineup.PITCH_LIMITS,
    tto_hooks=lineup.TTO_HOOKS,
    defense_rules=lineup.DEFENSE_RULES,
    sub_rows=lineup.DEFENSIVE_SUB_ROWS,
    sub_innings=lineup.SUB_INNINGS,
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
) -> FastAPI:
    app = FastAPI(title="NPL Sim")
    store = store or Store()
    players = players if players is not None else load_players()
    elig = elig if elig is not None else eligibility.load(SEASON)
    slots = team_slots(load_league_file().for_season(SEASON))
    rosters: dict[str, list[NplPlayer]] = {s.team: [] for s in slots}
    for p in players:
        if p.team in rosters and p.is_active_mlb and p.mlb_id is not None:
            rosters[p.team].append(p)
    by_slug = {slug(t): t for t in rosters}

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
        for h in hitters:
            positions, from_games = lineup.fielding(h, elig)
            fields[h.mlb_id] = {
                "positions": [p for p in lineup.POSITIONS if p in positions],
                "from_games": from_games,
                "games": elig.games_text(h.mlb_id),
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
            request, "home.html", {"teams": teams, "games": store.recent_games()}
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
