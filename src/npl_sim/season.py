"""Sim a whole season week by week, for testing the schedule and the engine.

Dave (2026-10-12): with every team's roster loaded, run full-season sims
to test the schedule and the outputs. Each sim week plays that week's
scheduled games between the teams' lineup cards through `simweek.sim_week`
(the weekly playing-time budget), and carries unused starts, the rotation
turn and pitchers' rest into the next week. Each game's events are reduced
to a summary as it goes, so a season fits in memory.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta

from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.workload import Carry, MlbWeek, WeekResult
from npl_sim.rosters.eligibility import Eligibility
from npl_sim.rosters.npl_app import NplPlayer
from npl_sim.schedule.calendar import Week
from npl_sim.schedule.packer import ScheduledSeries
from npl_sim.simweek import sim_week, typical_mlb_week
from npl_sim.web import lineup

# A full-time MLB week for players the eligibility data doesn't cover yet:
# regulars about 26 PA, bench 8, relievers 3 innings, and each of five
# starters 6.2 / 5 starts (MLB teams play about 6.2 games a week), drawn so
# a starter makes two starts in about a quarter of weeks.
REGULAR_PA, BENCH_PA, RELIEF_OUTS = 26, 8, 9
STARTS_PER_WEEK = 6.2 / 5


@dataclass
class GameSummary:
    date: date
    away: str
    home: str
    runs: dict[str, int]  # by team
    innings: int
    starters: dict[str, str]  # team -> pitcher id
    bullpen_game: dict[str, bool]
    standins: int  # (AAA) stand-ins used, both teams


@dataclass
class SeasonResult:
    games: list[GameSummary] = field(default_factory=list)
    totals: Counter = field(default_factory=Counter)  # league-wide PA outcomes and pitches
    carry: dict[str, Carry] = field(default_factory=dict)

    def standings(self) -> list[tuple[str, int, int, int, int]]:
        """(team, W, L, runs scored, runs allowed), best record first."""
        rows: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for g in self.games:
            for team, other in ((g.away, g.home), (g.home, g.away)):
                r = rows[team]
                r[0 if g.runs[team] > g.runs[other] else 1] += 1
                r[2] += g.runs[team]
                r[3] += g.runs[other]
        table = [(t, *r) for t, r in rows.items()]
        return sorted(table, key=lambda r: (-r[1] / max(1, r[1] + r[2]), r[4] - r[3]))

    def starter_rest(self) -> Counter:
        """Days between a pitcher's starts (bullpen games left out), counted."""
        last: dict[str, date] = {}
        gaps: Counter = Counter()
        for g in sorted(self.games, key=lambda g: g.date):
            for team, pid in g.starters.items():
                if g.bullpen_game[team]:
                    continue
                if pid in last:
                    gaps[(g.date - last[pid]).days] += 1
                last[pid] = g.date
        return gaps


def default_cards(
    rosters: dict[str, list[NplPlayer]], elig: Eligibility | None
) -> dict[str, lineup.LineupCard]:
    """Each team's default lineup card, with the rotation and the top of the
    pen chosen by this season's MLB usage where it's known: the five arms
    with the most starts start, and the pen is ordered by saves plus holds
    (so the closer is the arm with the most). Arms with no usage data keep
    their roster order after those with it."""

    def usage(p: NplPlayer) -> tuple[int, int]:
        rec = elig.players.get(p.mlb_id) if elig else None
        line = rec.pitching.get(elig.season, {}) if rec else {}
        return line.get("gs", 0), line.get("sv", 0) + line.get("hld", 0)

    cards = {}
    for team, roster in rosters.items():
        arms = [p for p in roster if lineup.is_pitcher(p)]
        starters = sorted(arms, key=lambda p: -usage(p)[0])[: lineup.ROTATION_SIZE]
        pen = sorted((p for p in arms if p not in starters), key=lambda p: -usage(p)[1])
        hitters = [p for p in roster if not lineup.is_pitcher(p)]
        cards[team] = lineup.default_card(hitters + starters + pen, elig)
    return cards


def pace_week(
    cards: dict[str, lineup.LineupCard],
    rosters: dict[str, list[NplPlayer]],
    elig: Eligibility | None,
    seed: int,
) -> MlbWeek:
    """A typical MLB week: each player's season pace where the eligibility
    data has him (`typical_mlb_week`), a full-time week by his card role
    where it doesn't."""
    rng = random.Random(seed)
    known = {p.mlb_id for r in rosters.values() for p in r if elig and p.mlb_id in elig.players}
    pa: dict[str, int] = {}
    starts: dict[str, int] = {}
    relief: dict[str, int] = {}
    for card in cards.values():
        in_order = {pid for order in card.orders.values() for pid, _ in order if pid}
        rotation = {pid for pid in card.rotation if pid}
        by_id = {p.mlb_id: p for r in rosters.values() for p in r}
        for pid in card.active:
            if pid in known or pid not in by_id:
                continue
            if pid in rotation:
                whole = int(STARTS_PER_WEEK)
                starts[str(pid)] = whole + (rng.random() < STARTS_PER_WEEK - whole)
            elif lineup.is_pitcher(by_id[pid]):
                relief[str(pid)] = RELIEF_OUTS
            else:
                pa[str(pid)] = REGULAR_PA if pid in in_order else BENCH_PA
    if elig is not None:
        typical = typical_mlb_week(rosters, elig, elig.season, seed)
        pa.update(typical.pa)
        starts.update(typical.starts)
        relief.update(typical.relief_outs)
    return MlbWeek(pa=pa, starts=starts, relief_outs=relief)


def week_games(schedule: list[ScheduledSeries], week: Week) -> list[tuple[int, str, str, int]]:
    """(day 0-6, away, home, series id) for every game of `week`."""
    return sorted(
        ((d - week.start).days, item.series.away, item.series.home, n)
        for n, item in enumerate(schedule)
        if item.week == week.number
        for d in item.dates
    )


def summarize(result: WeekResult, week: Week, season: SeasonResult) -> None:
    for g in result.games:
        events = g["events"]
        final = events[-1]
        moves = g["moves"]
        season.games.append(
            GameSummary(
                date=week.start + timedelta(days=g["day"]),
                away=g["away"],
                home=g["home"],
                runs={g["away"]: final["away"], g["home"]: final["home"]},
                innings=max(e["inning"] for e in events if "inning" in e),
                starters={t: m.starter for t, m in moves.items()},
                bullpen_game={t: m.bullpen_game for t, m in moves.items()},
                standins=sum(
                    sum(r["standin"] for r in m.replaced) + m.standin_pitcher
                    for m in moves.values()
                ),
            )
        )
        t = season.totals
        t["games"] += 1
        t["runs"] += final["away"] + final["home"]
        for e in events:
            if e["type"] == "pa":
                t["PA"] += 1
                t[e["result"]] += 1
                t["pitches"] += e["pitches"]
                t["thrown_out"] += len(e.get("thrown_out", ()))
            elif e["type"] in (
                "stolen_base",
                "caught_stealing",
                "wild_pitch",
                "passed_ball",
                "balk",
            ):
                t[e["type"]] += 1
                t["pitches"] += e.get("pitches", 0)  # a PA cut short


def sim_season(
    cards: dict[str, lineup.LineupCard],
    rosters: dict[str, list[NplPlayer]],
    elig: Eligibility | None,
    schedule: list[ScheduledSeries],
    weeks: list[Week],
    seed: int,
    env=DEFAULT_ENV,
    on_week: Callable[[Week, WeekResult], None] | None = None,
) -> SeasonResult:
    """Play every week of `schedule` in order, carrying each team's rest and
    unused starts forward. Playing time is each player's typical week at his
    season pace (`pace_week`)."""
    season = SeasonResult()
    carry: dict[str, Carry] = {}
    for week in weeks:
        games = week_games(schedule, week)
        if not games:
            continue
        week_seed = seed * 100 + week.number
        mlb = pace_week(cards, rosters, elig, week_seed)
        result = sim_week(cards, rosters, elig, games, mlb, week_seed, carry, env)
        summarize(result, week, season)
        carry = result.carry
        if on_week is not None:
            on_week(week, result)
    season.carry = carry
    return season
