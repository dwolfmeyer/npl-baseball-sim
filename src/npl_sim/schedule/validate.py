"""Check a packed season against the schedule rules. Returns every problem found."""

from __future__ import annotations

from collections import Counter, defaultdict

from npl_sim.league import Alignment
from npl_sim.schedule.calendar import Week
from npl_sim.schedule.packer import ScheduledSeries

GAMES = 162
HOME = 81


def validate(schedule: list[ScheduledSeries], alignment: Alignment, weeks: list[Week]) -> list[str]:
    problems: list[str] = []
    league_of, div_of, rival_of = {}, {}, {}
    for lg in alignment.leagues:
        for div in lg.divisions:
            for t in div.teams:
                league_of[t], div_of[t] = lg.name, div.name
            for a, b in div.rival_pairs():
                rival_of[a], rival_of[b] = b, a
    teams = sorted(league_of)
    week_by_number = {w.number: w for w in weeks}

    games, home = Counter(), Counter()
    pair_games: Counter[frozenset[str]] = Counter()
    busy: dict[tuple[str, object], int] = defaultdict(int)
    for item in schedule:
        s = item.series
        if len(item.dates) != s.games:
            problems.append(f"{s} has {len(item.dates)} dates")
        week = week_by_number.get(item.week)
        for d in item.dates:
            if week is None or d not in week.game_days:
                problems.append(f"{s.away} at {s.home} on {d} is outside week {item.week}")
            busy[(s.home, d)] += 1
            busy[(s.away, d)] += 1
        games[s.home] += s.games
        games[s.away] += s.games
        home[s.home] += s.games
        pair_games[s.pair] += s.games

    for (team, d), n in sorted(busy.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        if n > 1:
            problems.append(f"{team} plays {n} games on {d}")

    for t in teams:
        if games[t] != GAMES:
            problems.append(f"{t} plays {games[t]} games, not {GAMES}")
        if home[t] != HOME:
            problems.append(f"{t} has {home[t]} home games, not {HOME}")
        three_game = 0
        for u in teams:
            if u == t:
                continue
            n = pair_games[frozenset((t, u))]
            if league_of[u] != league_of[t]:
                if n == 3:
                    three_game += 1
                elif n != 4:
                    problems.append(f"{t} vs {u}: {n} interleague games, not 3 or 4")
            elif rival_of[t] == u:
                if n != 18:
                    problems.append(f"{t} vs rival {u}: {n} games, not 18")
            elif div_of[u] == div_of[t]:
                if n != 17:
                    problems.append(f"{t} vs {u}: {n} division games, not 17")
            elif n != 8:
                problems.append(f"{t} vs {u}: {n} league games, not 8")
        if three_game != 2:
            problems.append(f"{t} has {three_game} 3-game interleague opponents, not 2")
    return problems
