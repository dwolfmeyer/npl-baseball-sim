"""Win expectancy (WE): the home team's chance of winning from any game state,
and the win probability added (WPA) by each play.

Built the way Tango's tables in "The Book" are: a Markov model over half
innings. `_RUNS` holds, for each base-out state, the chance the batting team
scores 0, 1, ... 10+ more runs before the half ends; it was counted from 20,000
simulated games between league-average teams (`python -m npl_sim.engine.winprob`
re-counts it after an engine change). Chaining half innings gives WE for every
inning, half, score, base and out. Like Tango's, it assumes two equal teams in a
neutral park. Extra innings start with a runner on second (`LeagueEnv.auto_runner`).

`_RUNS` excludes bottom halves from the 9th on, which a walk-off can cut short.
"""

from __future__ import annotations

from collections import defaultdict
from functools import cache

from npl_sim.engine.leverage import STATES, state

MAX_RUNS = 10  # the last column is "10 or more"
MAX_LEAD = 30  # beyond this the game is decided
LAST_INNING = 30  # extra innings past this count as a coin flip

# One row per base-out state: the eight runner states with 0 outs, then 1, then
# 2. Each row: chance of 0..9 and 10+ more runs this half inning.
_RUNS = """
0.7343 0.1382 0.0686 0.0331 0.0149 0.0063 0.0027 0.0011 0.0005 0.0002 0.0001
0.6095 0.1435 0.1238 0.0657 0.0318 0.0141 0.0066 0.0029 0.0013 0.0005 0.0004
0.4405 0.2920 0.1371 0.0703 0.0333 0.0147 0.0070 0.0029 0.0011 0.0006 0.0004
0.2210 0.5050 0.1379 0.0706 0.0365 0.0163 0.0096 0.0010 0.0019 0.0000 0.0000
0.4175 0.1936 0.1334 0.1261 0.0683 0.0323 0.0160 0.0067 0.0035 0.0015 0.0009
0.2259 0.3852 0.1436 0.1219 0.0642 0.0312 0.0167 0.0074 0.0027 0.0007 0.0005
0.2362 0.2328 0.2647 0.1364 0.0696 0.0319 0.0204 0.0051 0.0024 0.0000 0.0007
0.1764 0.2471 0.1868 0.1335 0.1266 0.0665 0.0333 0.0157 0.0078 0.0037 0.0026
0.8345 0.0963 0.0412 0.0171 0.0067 0.0027 0.0009 0.0004 0.0001 0.0000 0.0000
0.7411 0.0995 0.0911 0.0413 0.0168 0.0065 0.0022 0.0010 0.0004 0.0001 0.0000
0.6074 0.2284 0.0946 0.0411 0.0177 0.0064 0.0028 0.0010 0.0003 0.0002 0.0001
0.4240 0.4133 0.0932 0.0399 0.0178 0.0073 0.0026 0.0015 0.0005 0.0000 0.0000
0.5865 0.1572 0.0933 0.0933 0.0413 0.0169 0.0071 0.0028 0.0010 0.0004 0.0002
0.4548 0.2785 0.1029 0.0942 0.0423 0.0162 0.0064 0.0028 0.0013 0.0004 0.0002
0.4065 0.1967 0.2272 0.0966 0.0424 0.0197 0.0079 0.0016 0.0011 0.0001 0.0001
0.3883 0.2001 0.1580 0.0916 0.0916 0.0414 0.0166 0.0074 0.0032 0.0012 0.0005
0.9251 0.0503 0.0164 0.0054 0.0018 0.0007 0.0002 0.0001 0.0000 0.0000 0.0000
0.8727 0.0511 0.0506 0.0175 0.0055 0.0019 0.0004 0.0002 0.0001 0.0000 0.0000
0.7681 0.1558 0.0514 0.0168 0.0053 0.0014 0.0008 0.0003 0.0001 0.0000 0.0000
0.7556 0.1662 0.0514 0.0175 0.0064 0.0018 0.0004 0.0004 0.0001 0.0001 0.0000
0.7550 0.1153 0.0530 0.0518 0.0167 0.0053 0.0018 0.0008 0.0002 0.0001 0.0000
0.7435 0.1296 0.0513 0.0518 0.0150 0.0062 0.0016 0.0006 0.0004 0.0001 0.0000
0.7404 0.0255 0.1571 0.0512 0.0170 0.0064 0.0015 0.0007 0.0002 0.0000 0.0000
0.6685 0.0859 0.1169 0.0525 0.0506 0.0168 0.0058 0.0019 0.0008 0.0003 0.0000
"""


@cache
def runs_table() -> dict[tuple[str, int], list[float]]:
    rows = [[float(x) for x in line.split()] for line in _RUNS.strip().splitlines()]
    if len(rows) != 24:
        raise ValueError(f"run table has {len(rows)} rows")
    return {(STATES[i % 8], i // 8): row for i, row in enumerate(rows)}


def _tail(dist: list[float], k: int) -> float:
    """P(at least k runs)."""
    return sum(dist[max(k, 0) :]) if k <= MAX_RUNS else 0.0


@cache
def _start(inning: int, half: str, lead: int, auto_runner: bool) -> float:
    """Home WE at the start of a half inning, `lead` = home minus away."""
    if lead > MAX_LEAD:
        return 1.0
    if lead < -MAX_LEAD:
        return 0.0
    if inning > LAST_INNING:
        return 0.5 if lead == 0 else float(lead > 0)
    if half == "bottom" and inning >= 9 and lead > 0:
        return 1.0  # the home half isn't played
    if half == "top" and inning > 9 and lead != 0:
        return float(lead > 0)
    base = "_2_" if inning > 9 and auto_runner else "___"
    return _rest(inning, half, lead, base, 0, auto_runner)


def _rest(inning: int, half: str, lead: int, bases: str, outs: int, auto_runner: bool) -> float:
    dist = runs_table()[(bases, outs)]
    if half == "top":
        return sum(p * _start(inning, "bottom", lead - r, auto_runner) for r, p in enumerate(dist))
    if inning >= 9:
        # Walk-off: scoring enough to lead ends it; tying sends it on.
        win = _tail(dist, 1 - lead)
        tie = dist[-lead] if 0 <= -lead < MAX_RUNS else 0.0
        return win + tie * _start(inning + 1, "top", 0, auto_runner)
    return sum(p * _start(inning + 1, "top", lead + r, auto_runner) for r, p in enumerate(dist))


def win_expectancy(
    inning: int, half: str, home_lead: int, bases: list, outs: int, auto_runner: bool = True
) -> float:
    """Home team's chance of winning, before the next play (`bases`: three
    entries, None = empty)."""
    if outs >= 3:
        if half == "top":
            return _start(inning, "bottom", home_lead, auto_runner)
        return _start(inning + 1, "top", home_lead, auto_runner)
    if half == "bottom" and inning >= 9 and home_lead > 0:
        return 1.0
    return _rest(inning, half, home_lead, state(bases), outs, auto_runner)


def count_runs(games: list[list[dict]]) -> dict[tuple[str, int], list[float]]:
    """Runs to the end of the half inning from each base-out state, counted
    from simulated event logs."""
    counts: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0] * (MAX_RUNS + 1))
    for events in games:
        half: list[tuple[tuple[str, int], int]] = []  # (state, runs so far) per PA
        runs = 0
        for e in events:
            if e["type"] == "pa":
                half.append(((state(e["bases_before"]), e["outs_before"]), runs))
                runs += len(e["scored"])
            elif e["type"] in ("wild_pitch", "passed_ball", "balk"):
                runs += len(e["scored"])
            elif e["type"] == "half_end":
                if not (e["half"] == "bottom" and e["inning"] >= 9):
                    for key, before in half:
                        counts[key][min(runs - before, MAX_RUNS)] += 1
                half, runs = [], 0
    out = {}
    for outs in range(3):
        for s in STATES:
            row = counts[(s, outs)]
            n = sum(row) or 1
            out[(s, outs)] = [c / n for c in row]
    return out


def main(games: int = 20000) -> None:  # pragma: no cover - regenerates _RUNS
    from npl_sim.engine.defaults import DEFAULT_ENV
    from npl_sim.engine.game import simulate_game
    from npl_sim.engine.sample import average_team

    away, home = average_team("Away"), average_team("Home")
    table = count_runs([simulate_game(away, home, DEFAULT_ENV, 90000 + i) for i in range(games)])
    for outs in range(3):
        for s in STATES:
            print(" ".join(f"{p:.4f}" for p in table[(s, outs)]))


if __name__ == "__main__":  # pragma: no cover
    main()
