from dataclasses import replace

import pytest

from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.game import simulate_game
from npl_sim.engine.hooks import StarterHook
from npl_sim.engine.sample import average_team, pitcher

NO_HOOK = StarterHook(max_runs=None, leverage=None)


def test_reason_order_and_off_switches():
    hook = StarterHook(times_through=3, max_runs=5, leverage=2.0, leverage_from_inning=7)
    assert hook.reason(10, 2, 4, 3.0) is None  # big spot, but too early
    assert hook.reason(10, 5, 2, 0.5) == "runs"
    assert hook.reason(27, 0, 7, 0.5) == "times_through"
    assert hook.reason(20, 1, 7, 2.0) == "leverage"
    assert NO_HOOK.reason(40, 12, 9, 9.0) is None
    with pytest.raises(ValueError):
        StarterHook(max_runs=0)


def starter_lines(events, team, starter):
    """Runs charged to the starter before each PA he pitched, and the PAs."""
    charged, before = 0, []
    for e in events:
        if e["type"] == "pa" and e["team"] != team and e["pitcher"] == starter:
            before.append(charged)
            charged += sum(1 for s in e["scored"] if s["pitcher"] == starter)
    return before


def test_runs_hook_pulls_a_starter_who_is_getting_hit():
    # A batting-practice starter with a high pitch limit: only the runs hook
    # gets him out.
    bad = pitcher("Home-SP", limit=200, scale={"hr": 0.12, "k": 0.08})
    home = replace(average_team("Home"), starter=bad, hook=StarterHook(max_runs=4, leverage=None))
    pulled_for_runs = 0
    for seed in range(60):
        events = simulate_game(average_team("Away"), home, DEFAULT_ENV, seed)
        assert all(r < 4 for r in starter_lines(events, "Home", "Home-SP"))
        changes = [e for e in events if e["type"] == "pitching_change" and e["team"] == "Home"]
        pulled_for_runs += bool(changes) and changes[0]["reason"] == "runs"  # or a complete game
    assert pulled_for_runs > 40


def test_times_through_hook():
    home = replace(average_team("Home"), hook=replace(NO_HOOK, times_through=2))
    for seed in range(40):
        events = simulate_game(average_team("Away"), home, DEFAULT_ENV, seed)
        assert len(starter_lines(events, "Home", "Home-SP")) <= 18


def test_leverage_hook_only_late_and_in_big_spots():
    hook = StarterHook(max_runs=None, leverage=1.8, leverage_from_inning=6)
    home = replace(average_team("Home"), hook=hook)
    seen = 0
    for seed in range(150):
        events = simulate_game(average_team("Away"), home, DEFAULT_ENV, seed)
        for e in events:
            if e["type"] == "pitching_change" and e["team"] == "Home" and e["reason"] == "leverage":
                assert e["out"] == "Home-SP" and e["inning"] >= 6 and e["leverage"] >= 1.8
                seen += 1
    assert seen > 10


def test_no_hook_falls_back_to_pitch_count():
    home = replace(average_team("Home"), hook=NO_HOOK)
    events = simulate_game(average_team("Away"), home, DEFAULT_ENV, 3)
    first = next(e for e in events if e["type"] == "pitching_change" and e["team"] == "Home")
    assert first["reason"] == "pitch_count"


def test_an_opener_never_goes_past_one_inning():
    # A great pitcher with a huge pitch limit and no hook would run deep; as an
    # opener he is out after one inning, clean or not.
    ace = pitcher("Home-SP", limit=200, scale={"k": 0.4, "bb": 0.02, "hr": 0.005})
    base = replace(average_team("Home"), starter=ace, hook=NO_HOOK)
    opener = replace(base, opener=True)
    away = average_team("Away")
    clean = early = 0
    for seed in range(80):
        starter_outs = faced = 0
        for e in simulate_game(away, opener, DEFAULT_ENV, seed):
            if e["type"] == "pa" and e["team"] == away.name and e["pitcher"] == "Home-SP":
                starter_outs += e["outs"]
                faced += 1
        assert starter_outs <= 3  # the outs of one inning at most
        clean += starter_outs == 3 and faced == 3
        early += starter_outs < 3
    assert clean  # the clean inning that would otherwise have kept him in
    # Without the opener flag the same pitcher goes deep.
    deep = [
        sum(
            e["outs"]
            for e in simulate_game(away, base, DEFAULT_ENV, s)
            if e["type"] == "pa" and e["pitcher"] == "Home-SP" and e["team"] == away.name
        )
        for s in range(10)
    ]
    assert min(deep) > 9


def test_an_opener_can_still_be_pulled_before_he_finishes_the_inning():
    wild = pitcher("Home-SP", limit=200, scale={"hr": 0.15, "k": 0.05})
    home = replace(
        average_team("Home"),
        starter=wild,
        opener=True,
        hook=StarterHook(max_runs=3, leverage=None),
    )
    away = average_team("Away")
    mid = 0
    for seed in range(60):
        for e in simulate_game(away, home, DEFAULT_ENV, seed):
            if e["type"] == "pitching_change" and e["out"] == "Home-SP" and e["inning"] == 1:
                mid += e["outs"] < 3
    assert mid


def pitches_by_pitcher(events, team):
    counts = {}
    for e in events:
        if e["type"] == "pa" and e["team"] != team:
            counts[e["pitcher"]] = counts.get(e["pitcher"], 0) + e["pitches"]
    return counts


def test_no_pitcher_goes_past_120_pitches_even_with_a_huge_limit():
    ace = pitcher("Home-SP", limit=400, scale={"hr": 0.005, "k": 0.45, "bb": 0.03, "hit": 0.6})
    home = replace(average_team("Home"), starter=ace, hook=NO_HOOK)
    longest = 0
    for seed in range(60):
        events = simulate_game(average_team("Away"), home, DEFAULT_ENV, seed)
        counts = pitches_by_pitcher(events, "Home")
        longest = max(longest, *counts.values())
        # The cap is checked before each PA, so the last one can run a few over.
        assert all(n <= 130 for n in counts.values())
    assert longest >= 100


def test_a_reliever_is_pulled_after_seven_runs_and_a_standin_covers_an_empty_pen():
    bad = {"hr": 0.15, "k": 0.05, "bb": 0.2}
    arms = [pitcher(f"Home-RP{i}", limit=200, scale=bad) for i in range(1, 3)]
    starter = pitcher("Home-SP", limit=3, scale=bad)
    home = replace(average_team("Home"), starter=starter, bullpen=arms, hook=NO_HOOK)
    saw_standin = False
    for seed in range(40):
        events = simulate_game(average_team("Away"), home, DEFAULT_ENV, seed)
        runs = {}
        for e in events:
            if e["type"] == "pa" and e["team"] != "Home":
                p = e["pitcher"]
                if p != "Home-SP":
                    assert runs.get(p, 0) < 7  # pulled before facing another batter
                runs[p] = runs.get(p, 0) + sum(1 for s in e["scored"] if s["pitcher"] == p)
                saw_standin |= p.startswith("AAA-")
    assert saw_standin


def test_standin_arms_keep_a_readable_short_name():
    from npl_sim.engine.names import Roster

    home = average_team("Home")
    start = {
        "type": "game_start",
        "away": {
            "name": "A",
            "abbr": "AWY",
            "lineup": [],
            "pitchers": [{"id": "a1", "name": "Al Ace", "throws": "R"}],
            "starter": "a1",
        },
        "home": {
            "name": home.name,
            "abbr": "HOM",
            "lineup": [],
            "pitchers": [{"id": f"AAA-HOM-R{i}", "name": "(AAA) P", "throws": "R"} for i in (1, 2)],
            "starter": "AAA-HOM-R1",
        },
    }
    roster = Roster([start])
    assert roster.short["AAA-HOM-R1"] == "(AAA) P" and roster.short["a1"] == "Ace"
