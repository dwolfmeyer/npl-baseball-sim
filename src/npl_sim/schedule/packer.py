"""Pack series into Monday-Sunday weeks with a constraint solver (CP-SAT).

Dave (2026-10-12): weekly sims need a schedule shaped like MLB's, with 2-
and 3-game series and off and travel days, because starters need four
days' rest and relievers rest after heavy outings.

The season is cut into rounds, the way MLB's week splits into a midweek
and a weekend series. Every team plays one series per round:
- a 6-game week has two 3-day rounds and an off day (Tue-Thu and Fri-Sun,
  or Mon-Wed and Fri-Sun)
- a 7-game week has a 3-day round and two 2-day rounds (Mon-Wed, Thu-Fri,
  Sat-Sun, or Mon-Tue, Wed-Thu, Fri-Sun); 7-game weeks are spread out and
  never back to back, so no team plays more than 16 days in a row
- short weeks (Opening Day, the All-Star break) split their days into 3- and
  2-day rounds
A 2-game series in a 3-day round leaves its team a travel day (the round's
first day when it ends on Sunday, its last day otherwise). The number of
7-game weeks is set so that each team's series exactly fill the rounds.

Every round is one perfect matching of the league. The pairs split into
matchings by kind (division, own league, interleague) and series count, as
in a round robin, and the solver (CP-SAT) then works in two stages:
1. which matching plays each round, so that the same two teams never meet
   in the same or back-to-back weeks, and a 2-day round only goes to a
   matching whose pairs all have a 2-game series left
2. which of a pair's series is played in each of its rounds (3-game series
   in 3-day rounds), with no homestand or road trip longer than
   MAX_STRAIGHT_SERIES series and at least MIN_GAMES_FULL_WEEK games in
   every full week
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from ortools.sat.python import cp_model

from npl_sim.schedule.calendar import Week
from npl_sim.schedule.matchups import Series

# Rounds of a full week as (first weekday, days); 0 = Monday.
SIX_GAME_WEEKS = (((1, 3), (4, 3)), ((0, 3), (4, 3)))
SEVEN_GAME_WEEKS = (((0, 3), (3, 2), (5, 2)), ((0, 2), (2, 2), (4, 3)))
MAX_STRAIGHT_SERIES = 4
MAX_ATTEMPTS = 5
MIN_GAMES_FULL_WEEK = 5


@dataclass(frozen=True)
class ScheduledSeries:
    series: Series
    week: int
    dates: tuple[date, ...]


@dataclass(frozen=True)
class Round:
    week: int
    days: tuple[date, ...]


class ScheduleInfeasible(RuntimeError):
    pass


def _split_run(days: list[date]) -> list[tuple[date, ...]]:
    """Consecutive game days of a short week as 3- and 2-day rounds."""
    n = len(days)
    sizes = {1: [], 2: [2], 3: [3], 4: [2, 2], 5: [3, 2], 6: [3, 3]}.get(n)
    if sizes is None:
        sizes = [3] * (n // 3) + ([2] if n % 3 == 2 else [])
        if n % 3 == 1:
            sizes[-1:] = [2, 2]
    out, i = [], 0
    for s in sizes:
        out.append(tuple(days[i : i + s]))
        i += s
    return out


def _short_week_rounds(week: Week) -> list[Round]:
    runs: list[list[date]] = []
    for d in week.game_days:
        if runs and (d - runs[-1][-1]).days == 1:
            runs[-1].append(d)
        else:
            runs.append([d])
    return [Round(week.number, r) for run in runs for r in _split_run(run)]


def plan_rounds(weeks: list[Week], series_per_team: int) -> list[Round]:
    """The season's rounds, with enough 7-game weeks for every team to play
    `series_per_team` series."""
    full = [w for w in weeks if len(w.game_days) == 7]
    short = {w.number: _short_week_rounds(w) for w in weeks if len(w.game_days) < 7}
    sevens = series_per_team - sum(len(r) for r in short.values()) - 2 * len(full)
    if not 0 <= sevens <= (len(full) + 1) // 2:
        raise ScheduleInfeasible(
            f"{series_per_team} series per team can't fill {len(full)} full weeks"
        )
    # Spread the 7-game weeks evenly; with at most half the weeks, none touch.
    seven_at = {(2 * i + 1) * len(full) // (2 * sevens) for i in range(sevens)}
    rounds: list[Round] = []
    n6 = n7 = 0
    for w in weeks:
        if w.number in short:
            rounds += short[w.number]
            continue
        k = full.index(w)
        if k in seven_at:
            template, n7 = SEVEN_GAME_WEEKS[n7 % 2], n7 + 1
        else:
            template, n6 = SIX_GAME_WEEKS[n6 % 2], n6 + 1
        for first, length in template:
            days = tuple(w.start + timedelta(days=first + i) for i in range(length))
            rounds.append(Round(w.number, days))
    return rounds


def series_dates(s: Series, r: Round) -> tuple[date, ...]:
    """A series' dates in its round; a short series leaves a travel day."""
    if s.games == len(r.days):
        return r.days
    return r.days[-s.games :] if r.days[-1].weekday() == 6 else r.days[: s.games]


def pack(
    series: list[Series], weeks: list[Week], seed: int = 0, time_limit_s: float = 120.0
) -> list[ScheduledSeries]:
    teams = sorted({t for s in series for t in (s.home, s.away)})
    per_team = Counter(t for s in series for t in (s.home, s.away))
    if len(set(per_team.values())) != 1:
        raise ScheduleInfeasible(f"teams play different numbers of series: {per_team}")
    rounds = plan_rounds(weeks, per_team[teams[0]])
    families = matching_families(series, teams, seed)
    for attempt in range(MAX_ATTEMPTS):
        skeleton = _assign_families(families, rounds, seed + attempt, time_limit_s)
        out = _assign_series(series, families, skeleton, rounds, teams, time_limit_s)
        if out is not None:
            return sorted(out, key=lambda x: (x.dates[0], x.series.home))
    raise ScheduleInfeasible(f"no home/away order fits after {MAX_ATTEMPTS} skeletons")


@dataclass(frozen=True)
class Family:
    """A perfect matching of the league whose pairs each meet `times` times."""

    pairs: tuple[frozenset[str], ...]
    times: int
    two_day_max: int  # rounds of 2 days it can fill: every pair needs a 2-game series


def matching_families(series: list[Series], teams: list[str], seed: int = 0) -> list[Family]:
    """Split the season's pairs into perfect matchings, so every round can be
    one matching: pairs are grouped by kind (division and rival together) and
    how many series they play, and each group, where every team has the
    same number of opponents, is split into matchings."""
    by_pair: dict[frozenset[str], list[Series]] = defaultdict(list)
    for s in series:
        by_pair[s.pair].append(s)
    groups: dict[tuple[str, int], list[frozenset[str]]] = defaultdict(list)
    for pair, ss in by_pair.items():
        kind = "division" if ss[0].kind in ("rival", "division") else ss[0].kind
        groups[(kind, len(ss))].append(pair)
    families = []
    for (_kind, times), pairs in sorted(groups.items()):
        for matching in _one_factorization(sorted(pairs, key=sorted), teams, seed):
            two_day = min(sum(s.games == 2 for s in by_pair[p]) for p in matching)
            families.append(Family(tuple(matching), times, two_day))
    return families


def _one_factorization(
    pairs: list[frozenset[str]], teams: list[str], seed: int
) -> list[list[frozenset[str]]]:
    degree = Counter(t for p in pairs for t in p)
    if set(degree) != set(teams) or len(set(degree.values())) != 1:
        raise ScheduleInfeasible("a group of pairs doesn't give every team the same opponents")
    k = degree[teams[0]]
    model = cp_model.CpModel()
    color = {(i, c): model.new_bool_var(f"e{i}c{c}") for i in range(len(pairs)) for c in range(k)}
    for i in range(len(pairs)):
        model.add_exactly_one(color[(i, c)] for c in range(k))
    for t in teams:
        mine = [i for i, p in enumerate(pairs) if t in p]
        for c in range(k):
            model.add_exactly_one(color[(i, c)] for i in mine)
    solver = cp_model.CpSolver()
    solver.parameters.random_seed = seed
    solver.parameters.max_time_in_seconds = 60.0
    if solver.solve(model) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise ScheduleInfeasible("a group of pairs can't be split into rounds")
    return [
        [p for i, p in enumerate(pairs) if solver.boolean_value(color[(i, c)])] for c in range(k)
    ]


def _assign_families(
    families: list[Family], rounds: list[Round], seed: int, time_limit_s: float
) -> list[int]:
    """Stage one: which family plays each round."""
    model = cp_model.CpModel()
    y = {
        (f, r): model.new_bool_var(f"f{f}r{r}")
        for f in range(len(families))
        for r in range(len(rounds))
    }
    for r in range(len(rounds)):
        model.add_exactly_one(y[(f, r)] for f in range(len(families)))
    weeks = sorted({rnd.week for rnd in rounds})
    for f, fam in enumerate(families):
        model.add(sum(y[(f, r)] for r in range(len(rounds))) == fam.times)
        two_day = [y[(f, r)] for r, rnd in enumerate(rounds) if len(rnd.days) == 2]
        model.add(sum(two_day) <= fam.two_day_max)
        for a, b in zip(weeks, weeks[1:], strict=False):
            model.add_at_most_one(y[(f, r)] for r, rnd in enumerate(rounds) if rnd.week in (a, b))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.random_seed = seed
    solver.parameters.num_workers = 8
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise ScheduleInfeasible(f"rounds: solver status {solver.status_name(status)}")
    return [
        next(f for f in range(len(families)) if solver.boolean_value(y[(f, r)]))
        for r in range(len(rounds))
    ]


def _assign_series(
    series: list[Series],
    families: list[Family],
    skeleton: list[int],
    rounds: list[Round],
    teams: list[str],
    time_limit_s: float,
) -> list[ScheduledSeries] | None:
    """Stage two: which of a pair's series is played in each of its family's
    rounds, keeping homestands and road trips to MAX_STRAIGHT_SERIES series."""
    family_of = {p: f for f, fam in enumerate(families) for p in fam.pairs}
    rounds_of: dict[int, list[int]] = defaultdict(list)
    for r, f in enumerate(skeleton):
        rounds_of[f].append(r)
    model = cp_model.CpModel()
    x: dict[tuple[int, int], cp_model.IntVar] = {}
    by_pair_round: dict[tuple[frozenset[str], int], list] = defaultdict(list)
    home: dict[tuple[str, int], list] = defaultdict(list)
    week_games: dict[tuple[str, int], list] = defaultdict(list)
    for i, s in enumerate(series):
        options = []
        for r in rounds_of[family_of[s.pair]]:
            if s.games > len(rounds[r].days):
                continue
            v = model.new_bool_var(f"s{i}r{r}")
            x[(i, r)] = v
            options.append(v)
            by_pair_round[(s.pair, r)].append(v)
            home[(s.home, r)].append(v)
            for t in (s.home, s.away):
                week_games[(t, rounds[r].week)].append(v * s.games)
        model.add_exactly_one(options)
    for vs in by_pair_round.values():
        model.add_exactly_one(vs)
    round_days = Counter()
    for rnd in rounds:
        round_days[rnd.week] += len(rnd.days)
    full = {w for w, n in round_days.items() if n >= 6}  # not Opening Day or All-Star weeks
    for (_team, w), games in week_games.items():
        if w in full:
            model.add(sum(games) >= MIN_GAMES_FULL_WEEK)
    span = MAX_STRAIGHT_SERIES + 1
    for t in teams:
        at_home = [sum(home[(t, r)]) for r in range(len(rounds))]
        for r in range(len(rounds) - MAX_STRAIGHT_SERIES):
            window = sum(at_home[r : r + span])
            model.add(window >= 1)
            model.add(window <= MAX_STRAIGHT_SERIES)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 8
    if solver.solve(model) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    out = []
    for (i, r), v in x.items():
        if solver.boolean_value(v):
            s, rnd = series[i], rounds[r]
            out.append(ScheduledSeries(series=s, week=rnd.week, dates=series_dates(s, rnd)))
    return out
