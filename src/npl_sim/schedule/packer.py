"""Pack series into Monday-Sunday weeks with a constraint solver (CP-SAT).

Each series is played on consecutive days inside one sim week, starting on
a day MLB series usually start (see SERIES_STARTS). Constraints:
- every series is played exactly once
- a team plays at most one game per day
- the same two teams never meet in back-to-back weeks
- in each full week a team plays at least 4 games
- no team is entirely home, or entirely away, for three straight weeks
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from ortools.sat.python import cp_model

from npl_sim.schedule.calendar import Week
from npl_sim.schedule.matchups import Series

# Weekday offsets a series may start on (0 = Monday).
SERIES_STARTS = {2: (0, 1, 2, 3, 5), 3: (0, 1, 3, 4), 4: (0, 3)}
MIN_GAMES_FULL_WEEK = 4


@dataclass(frozen=True)
class ScheduledSeries:
    series: Series
    week: int
    dates: tuple[date, ...]


class ScheduleInfeasible(RuntimeError):
    pass


def _placements(series: Series, week: Week) -> list[tuple[date, ...]]:
    days = set(week.game_days)
    out = []
    for offset in SERIES_STARTS[series.games]:
        dates = tuple(week.start + timedelta(days=offset + i) for i in range(series.games))
        if all(d in days for d in dates):
            out.append(dates)
    return out


def pack(
    series: list[Series], weeks: list[Week], seed: int = 0, time_limit_s: float = 120.0
) -> list[ScheduledSeries]:
    model = cp_model.CpModel()
    teams = sorted({t for s in series for t in (s.home, s.away)})
    choices: dict[int, list[tuple[cp_model.IntVar, Week, tuple[date, ...]]]] = defaultdict(list)
    by_team_day: dict[tuple[str, date], list] = defaultdict(list)
    by_pair_week: dict[tuple[frozenset[str], int], list] = defaultdict(list)
    games_team_week: dict[tuple[str, int], list] = defaultdict(list)
    home_team_week: dict[tuple[str, int], list] = defaultdict(list)
    away_team_week: dict[tuple[str, int], list] = defaultdict(list)

    for i, s in enumerate(series):
        for week in weeks:
            for dates in _placements(s, week):
                v = model.new_bool_var(f"s{i}w{week.number}d{dates[0].day}")
                choices[i].append((v, week, dates))
                for d in dates:
                    by_team_day[(s.home, d)].append(v)
                    by_team_day[(s.away, d)].append(v)
                by_pair_week[(s.pair, week.number)].append(v)
                games_team_week[(s.home, week.number)].append((v, s.games))
                games_team_week[(s.away, week.number)].append((v, s.games))
                home_team_week[(s.home, week.number)].append(v)
                away_team_week[(s.away, week.number)].append(v)
        if not choices[i]:
            raise ScheduleInfeasible(f"no week can hold {s}")
        model.add_exactly_one(v for v, _, _ in choices[i])

    for vs in by_team_day.values():
        model.add_at_most_one(vs)

    pairs = {s.pair for s in series}
    for pair in pairs:
        for w in weeks[:-1]:
            model.add_at_most_one(
                by_pair_week[(pair, w.number)] + by_pair_week[(pair, w.number + 1)]
            )

    for team in teams:
        for w in weeks:
            if len(w.game_days) == 7:
                model.add(
                    sum(v * g for v, g in games_team_week[(team, w.number)]) >= MIN_GAMES_FULL_WEEK
                )
        for a, b, c in zip(weeks, weeks[1:], weeks[2:], strict=False):
            window = (a.number, b.number, c.number)
            model.add(sum(v for n in window for v in home_team_week[(team, n)]) >= 1)
            model.add(sum(v for n in window for v in away_team_week[(team, n)]) >= 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.random_seed = seed
    solver.parameters.num_workers = 8
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise ScheduleInfeasible(f"solver status {solver.status_name(status)}")

    out = []
    for i, s in enumerate(series):
        for v, week, dates in choices[i]:
            if solver.boolean_value(v):
                out.append(ScheduledSeries(series=s, week=week.number, dates=dates))
                break
    return sorted(out, key=lambda x: (x.dates[0], x.series.home))
