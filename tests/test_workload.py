from collections import Counter

import pandas as pd

from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.sample import average_week_team, busy_week
from npl_sim.engine.workload import (
    RELIEF_SLACK,
    Carry,
    MlbWeek,
    WeekState,
    play_week,
)
from npl_sim.playing_time import mlb_week

AWAY, HOME = average_week_team("Away"), average_week_team("Home")
TEAMS = {"Away": AWAY, "Home": HOME}
# Seven games in a week, the home team at home for all of them.
GAMES = [(day, "Away", "Home") for day in range(7)]


def week(**changes) -> MlbWeek:
    usage = busy_week(AWAY, HOME)
    for kind, values in changes.items():
        usage[kind] = {**usage[kind], **values}
    return MlbWeek(**usage)


def pa_by_batter(result, team="Home"):
    return Counter(
        e["batter"]
        for g in result.games
        for e in g["events"]
        if e["type"] == "pa" and e["team"] == team
    )


def test_budgets_and_carry():
    state = WeekState.open(
        MlbWeek(pa={"a": 20, "b": 0}, starts={"s": 1}, relief_outs={"r": 7}),
        Carry(starts={"s": 2, "t": 1}),
    )
    assert state.pa_left == {"a": 25}  # 20 x 1.25; no MLB PA, no budget
    assert state.starts_left == {"s": 2, "t": 1}  # at most one start carries
    assert state.outs_left == {"r": 11}  # 7 x 1.5, rounded up
    state.starts_left["t"] = 0
    assert state.close().starts == {"s": 1}


def test_regulars_play_every_day_and_part_timers_run_out():
    usage = week(pa={"Home-1B": 8, "Home-C": 0})
    result = play_week(TEAMS, GAMES, usage, DEFAULT_ENV, seed=1)
    pa = pa_by_batter(result)
    assert pa["Home-SS"] >= 26  # a regular plays the whole week
    assert pa["Home-C"] == 0  # no MLB PA: doesn't play
    assert pa["Home-1B"] <= 10 + 4  # 8 x 1.25, plus at most the PA of his last game
    moves = [g["moves"]["Home"] for g in result.games]
    catcher_subs = [r for m in moves for r in m.replaced if r["out"] == "Home-C"]
    assert len(catcher_subs) == 7
    # The bench catcher fills in until his own PA run out, then a stand-in.
    assert catcher_subs[0]["into"] == "Home-BN1" and not catcher_subs[0]["standin"]
    assert catcher_subs[-1]["standin"]


def test_rotation_follows_mlb_starts_and_carries_unused_ones():
    usage = week(starts={"Home-SP3": 0, "Home-SP1": 2, "Home-SP2": 2})
    result = play_week(TEAMS, GAMES, usage, DEFAULT_ENV, seed=2)
    starters = [g["moves"]["Home"].starter for g in result.games]
    assert "Home-SP3" not in starters  # no MLB start this week
    counts = Counter(starters)
    assert all(counts[p] <= 2 for p in counts)
    # Four days off between starts.
    for p in counts:
        days = [g["day"] for g in result.games if g["moves"]["Home"].starter == p]
        assert all(b - a > 4 for a, b in zip(days, days[1:], strict=False))
    carried = result.carry["Home"].starts
    assert all(n == 1 for n in carried.values())


def test_bullpen_game_when_no_starter_can_go():
    usage = week(starts={f"Home-SP{i}": 0 for i in range(1, 6)})
    result = play_week(TEAMS, GAMES[:1], usage, DEFAULT_ENV, seed=3)
    moves = result.games[0]["moves"]["Home"]
    assert moves.bullpen_game and moves.starter.startswith("Home-RP")


def relief_lines(result, team_prefix="Home-RP"):
    outs, days = Counter(), {}
    for g in result.games:
        for e in g["events"]:
            if e["type"] == "pa" and e["pitcher"].startswith(team_prefix):
                outs[e["pitcher"]] += e["outs"]
                days.setdefault(e["pitcher"], set()).add(g["day"])
    return outs, days


def test_relievers_stay_inside_budget_and_rest():
    usage = week(relief_outs={"Home-RP1": 3})  # one MLB inning: 5 outs this week
    for seed in range(6):
        result = play_week(TEAMS, GAMES, usage, DEFAULT_ENV, seed=seed)
        outs, days = relief_lines(result)
        assert outs["Home-RP1"] <= 3 * RELIEF_SLACK + 3  # can finish the out he's on
        for d in days.values():
            assert not any({x, x + 1, x + 2} <= d for x in d)  # never three days running


def test_mlb_week_from_clean_tables():
    batting = pd.DataFrame({"mlbam_id": [1, 1, 2], "pa": [4, 5, 0]})
    pitching = pd.DataFrame({"mlbam_id": [9, 9, 8], "gs": [1, 0, 0], "outs": [18, 3, 4]})
    usage = mlb_week(batting, pitching)
    assert usage.pa == {"1": 9}
    assert usage.starts == {"9": 1}
    assert usage.relief_outs == {"9": 3, "8": 4}


def test_standins_hit_at_scoresheet_penalty_levels():
    from dataclasses import replace

    from npl_sim.engine.game import simulate_game
    from npl_sim.engine.replacement import standin_hitter
    from npl_sim.engine.sample import average_team

    away = average_team("A")
    lineup = list(away.lineup)
    for i, pos in ((0, "C"), (3, "SS"), (5, "LF")):
        lineup[i] = (standin_hitter(f"AAA-{pos}", pos), pos)
    away = replace(away, lineup=lineup)
    ab, hits = Counter(), Counter()
    for seed in range(1500):  # enough at bats that noise stays inside the bands
        for e in simulate_game(away, average_team("H"), DEFAULT_ENV, seed):
            if (
                e["type"] == "pa"
                and e["batter"].startswith("AAA")
                and e["result"] not in ("BB", "HBP", "SF")
                and not e.get("bunt")  # weak stand-ins sometimes sacrifice
            ):
                ab[e["batter"]] += 1
                hits[e["batter"]] += e["result"] in ("1B", "2B", "3B", "HR")
    avg = {k: hits[k] / ab[k] for k in ab}
    assert 0.11 < avg["AAA-C"] < 0.16  # ~.135
    assert 0.14 < avg["AAA-SS"] < 0.20  # ~.170
    assert 0.16 < avg["AAA-LF"] < 0.21  # ~.185
