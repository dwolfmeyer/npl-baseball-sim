"""Sim a week of games between teams' saved lineup cards, for testing.

Dave (2026-10-12): with the weekly playing-time budget in place, sim a full
week between the test teams (Ducksnorts, McFlys, Frozen Ropes). Each team's
lineup card becomes a `WeekTeam`: its batting orders vs LHP and RHP, the
rotation with each slot's plan (pitch limit, hooks, opener and piggyback),
the pen, bench and fielding from position eligibility. The week then runs
under `engine.workload.play_week`, with each player's real MLB week when the
pipeline's clean tables are at hand, or a sample full-time week otherwise.

With `talent` (talent.load.WeekTalent) each player plays at his blended
ability for the week (Milestone 7); without it everyone is league average
by handedness, which tests the machinery, not the rosters' talent.
"""

from __future__ import annotations

import random
from dataclasses import replace

from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.sample import hitter
from npl_sim.engine.workload import Carry, MlbWeek, WeekResult, WeekTeam, play_week
from npl_sim.rosters.eligibility import Eligibility
from npl_sim.rosters.npl_app import NplPlayer
from npl_sim.web import lineup

DAYS = 7


def abbr(team: str) -> str:
    return team.split()[-1][:3].upper()


def week_team(
    team: str, card: lineup.LineupCard, roster: list[NplPlayer], elig: Eligibility | None
) -> WeekTeam:
    """One team's week from its lineup card."""
    slots = [i for i, pid in enumerate(card.rotation) if pid is not None]
    if not slots:
        raise ValueError(f"{team}'s lineup card has no rotation")

    def build(slot: int, vs: str):
        return lineup.team_card(team, abbr(team), card, roster, slot, vs, elig)

    lineups = {vs: build(slots[0], vs).lineup for vs in ("L", "R")}
    slot_cards = {}
    for i in slots:
        c = build(i, "R")
        slot_cards[c.starter.id] = c
    base = build(slots[0], "R")
    starting = {h.id for lineup_ in lineups.values() for h, _ in lineup_}
    by_id = {str(p.mlb_id): p for p in roster}
    bench_ids = [
        str(pid)
        for pid in card.active
        if str(pid) in by_id and not lineup.is_pitcher(by_id[str(pid)]) and str(pid) not in starting
    ]
    hitters = {h.id: h for lineup_ in lineups.values() for h, _ in lineup_}
    bench = []
    for pid in bench_ids:
        # Build the bench hitter the way team_card builds lineup hitters.
        if pid not in hitters:
            hitters[pid] = _hitter(by_id[pid], elig)
        bench.append(hitters[pid])
    fields = {
        str(p.mlb_id): lineup.fielding(p, elig)[0]
        for p in roster
        if not lineup.is_pitcher(p) and p.mlb_id is not None
    }

    # Each hitter knows the positions he's experienced at, so playing him
    # anywhere else costs runs on the defensive spectrum (engine/fielding.py).
    def glove(h):
        return replace(h, positions={pos: 0.0 for pos in fields.get(h.id, ())})

    def gloved(c):
        return replace(
            c,
            lineup=[(glove(h), pos) for h, pos in c.lineup],
            defensive_subs=tuple(replace(d, player=glove(d.player)) for d in c.defensive_subs),
        )

    return WeekTeam(
        card=gloved(base),
        lineups={vs: [(glove(h), pos) for h, pos in o] for vs, o in lineups.items()},
        rotation=[c.starter for c in slot_cards.values()],
        bench=[glove(h) for h in bench],
        fields=fields,
        slot_cards={k: gloved(c) for k, c in slot_cards.items()},
    )


def _hitter(p: NplPlayer, elig: Eligibility | None):
    """A bench hitter's card, built the way team_card builds lineup hitters."""
    rec = elig.players.get(p.mlb_id) if elig else None
    bats = rec.bats if rec and rec.bats in ("L", "R", "S") else "R"
    return replace(hitter(str(p.mlb_id), bats=bats), name=p.name)


def round_robin(teams: list[str], days: int = DAYS) -> list[tuple[int, str, str]]:
    """A round robin for a test week: each day the teams pair up (one sits
    when the count is odd), rotating partners and home field."""
    names = list(teams) + ([None] if len(teams) % 2 else [])
    n = len(names)
    games = []
    for day in range(days):
        r = day % (n - 1)
        order = [names[0]] + names[1:][r:] + names[1:][:r]
        for i in range(n // 2):
            a, b = order[i], order[n - 1 - i]
            if a is None or b is None:
                continue
            away, home = (a, b) if (day // (n - 1) + i) % 2 else (b, a)
            games.append((day, away, home))
    return games


def sample_mlb_week(teams: dict[str, WeekTeam]) -> MlbWeek:
    """A normal full week when real data isn't at hand: starters in the
    lineup about 26 PA, bench 8, the first two rotation spots two starts and
    the rest one, relievers about three innings."""
    pa, starts, relief = {}, {}, {}
    for t in teams.values():
        for lineup_ in t.lineups.values():
            for h, _ in lineup_:
                pa[h.id] = 26
        for h in t.bench:
            pa.setdefault(h.id, 8)
        for i, p in enumerate(t.rotation):
            starts[p.id] = 2 if i < 2 else 1
        for c in [t.card, *t.slot_cards.values()]:
            for p in c.bullpen:
                if p.id not in starts:
                    relief[p.id] = 9
    return MlbWeek(pa=pa, starts=starts, relief_outs=relief)


SEASON_WEEKS = 26.5
PA_PER_GAME = 4.2


def typical_mlb_week(
    rosters: dict[str, list[NplPlayer]], elig: Eligibility, season: int, seed: int
) -> MlbWeek:
    """A plausible MLB week for each rostered player from his season pace in
    the eligibility data: games played x 4.2 PA, starts, and relief outs, each
    divided over the season's weeks and rounded at random (a pitcher who
    starts 1.2 times a week gets two starts in some weeks). Players with no
    games this season don't play."""
    rng = random.Random(seed)

    def draw(per_week: float) -> int:
        whole = int(per_week)
        return whole + (rng.random() < per_week - whole)

    pa, starts, relief = {}, {}, {}
    for roster in rosters.values():
        for p in roster:
            rec = elig.players.get(p.mlb_id)
            if rec is None:
                continue
            pid = str(p.mlb_id)
            line = rec.pitching.get(season, {})
            if line.get("g"):
                gs, g, outs = line.get("gs", 0), line["g"], line.get("outs", 0)
                starts[pid] = draw(gs / SEASON_WEEKS)
                relief_share = (g - gs) / g if gs else 1.0
                relief[pid] = draw(outs * relief_share / SEASON_WEEKS)
            games = sum(rec.games.get(season, {}).values())
            if games and not lineup.is_pitcher(p):
                pa[pid] = round(draw(min(7.0, games / SEASON_WEEKS)) * PA_PER_GAME)
    return MlbWeek(
        pa={k: v for k, v in pa.items() if v},
        starts={k: v for k, v in starts.items() if v},
        relief_outs={k: v for k, v in relief.items() if v},
    )


def sim_week(
    cards: dict[str, lineup.LineupCard],
    rosters: dict[str, list[NplPlayer]],
    elig: Eligibility | None,
    schedule: list[tuple],
    mlb_week: MlbWeek | None,
    seed: int,
    carry: dict[str, Carry] | None = None,
    env=DEFAULT_ENV,
    talent=None,
    audit: bool = False,
) -> WeekResult:
    """Play a week between saved lineup cards (keyed by team name).
    `mlb_week` None plays a typical week from each player's season pace
    (default_week). `talent` (a talent.load.WeekTalent) gives every player
    his blended ability for the week and the league its real environment;
    None keeps everyone league average. Each game's events are
    simulate_game's, so box scores and game pages render as usual. `audit`
    keeps each PA's inputs for npl_sim/audit.py."""
    teams = {name: week_team(name, cards[name], rosters[name], elig) for name in cards}
    if talent is not None:
        teams = {name: talent.team(t) for name, t in teams.items()}
        env = talent.league.env()
    mlb = mlb_week if mlb_week is not None else default_week(teams, rosters, elig, seed)
    return play_week(teams, schedule, mlb, env, seed, carry, audit)


def default_week(
    teams: dict[str, WeekTeam],
    rosters: dict[str, list[NplPlayer]],
    elig: Eligibility | None,
    seed: int,
) -> MlbWeek:
    """The week to play when no real MLB week is given: each player's season
    pace where the eligibility data has him, a full-time week otherwise."""
    sample = sample_mlb_week(teams)
    if elig is None:
        return sample
    typical = typical_mlb_week(rosters, elig, elig.season, seed)
    known = {str(p.mlb_id) for r in rosters.values() for p in r if p.mlb_id in elig.players}

    def merge(pace: dict, full: dict) -> dict:
        return {**{k: v for k, v in full.items() if k not in known}, **pace}

    return MlbWeek(
        pa=merge(typical.pa, sample.pa),
        starts=merge(typical.starts, sample.starts),
        relief_outs=merge(typical.relief_outs, sample.relief_outs),
    )
