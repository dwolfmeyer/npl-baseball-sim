from dataclasses import replace

import pytest

from npl_sim.engine.bullpen import ReliefUsage, choose, leverage, situation
from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.game import simulate_game
from npl_sim.engine.sample import average_team, hitter, pitcher

EMPTY, ON_2ND, LOADED = [None] * 3, [None, "r", None], ["r", "r", "r"]


def test_leverage_is_high_late_and_close():
    save_spot = leverage(9, "top", 1, EMPTY, 0)
    early_tie = leverage(3, "top", 0, EMPTY, 0)
    blowout = leverage(9, "top", 8, EMPTY, 0)
    assert save_spot > 1.8 > early_tie > blowout
    assert leverage(7, "top", 0, LOADED, 1) > leverage(7, "top", 0, EMPTY, 1)
    # Straight from Tango's table, from the fielding team's side.
    assert leverage(9, "top", 1, EMPTY, 0) == 2.9  # home up 1, top of the 9th
    assert leverage(9, "bottom", 1, LOADED, 2) == 10.9  # visitors up 1, bases loaded
    assert leverage(1, "top", 0, EMPTY, 0) == 0.9


def test_situation_tiers():
    assert situation(8, 6, 0.1, starter_out=False) == "mopup"
    assert situation(3, -1, 0.9, starter_out=True) == "early"
    assert situation(9, 1, 2.2, starter_out=False) == "high"
    assert situation(6, 2, 1.0, starter_out=False) == "medium"
    assert situation(6, 4, 0.4, starter_out=False) == "low"


def test_usage_validation_and_stint_length():
    with pytest.raises(ValueError):
        ReliefUsage(role="closer")
    with pytest.raises(ValueError):
        ReliefUsage(vs="S")
    assert ReliefUsage().outs == 3
    assert ReliefUsage(role="long").outs == 9
    assert ReliefUsage(role="high", max_outs=4).outs == 4


RIGHTY, LEFTY = hitter("R-bat", bats="R"), hitter("L-bat", bats="L")
A, B, C, D = (pitcher(x, t) for x, t in (("A", "R"), ("B", "R"), ("C", "L"), ("D", "L")))


def test_choose_follows_tier_then_matchup_then_owner_order():
    usage = {"A": ReliefUsage("low"), "B": ReliefUsage("high"), "C": ReliefUsage("high")}
    pen = [A, B, C]
    assert choose(pen, usage, "high", False, RIGHTY).id == "B"  # righty on righty
    assert choose(pen, usage, "high", False, LEFTY).id == "C"  # lefty on lefty
    assert choose(pen, usage, "low", False, RIGHTY).id == "A"
    assert choose(pen, {}, "high", False, RIGHTY).id == "A"  # no tags: owner's order
    assert choose([], usage, "high", False, RIGHTY) is None


def test_start_of_inning_only_and_specialists():
    usage = {
        "B": ReliefUsage("high", start_of_inning_only=True),
        "D": ReliefUsage("high", vs="L"),
        "A": ReliefUsage("medium"),
    }
    pen = [A, B, D]
    assert choose(pen, usage, "high", True, RIGHTY).id == "A"  # B waits for a clean inning
    assert choose(pen, usage, "high", False, RIGHTY).id == "B"
    assert choose(pen, usage, "high", True, LEFTY).id == "D"  # the specialist's spot
    assert choose([D], usage, "high", True, RIGHTY).id == "D"  # last arm standing


def tagged_home(**extra):
    home = average_team("Home")
    names = [p.id for p in home.bullpen]  # RP1..RP7
    usage = {
        names[0]: ReliefUsage("high"),
        names[1]: ReliefUsage("high", start_of_inning_only=True),
        names[2]: ReliefUsage("medium"),
        names[3]: ReliefUsage("low"),
        names[4]: ReliefUsage("mopup"),
        names[5]: ReliefUsage("long"),
        names[6]: ReliefUsage("medium", vs="L"),
    }
    return replace(home, relief=usage, **extra), names


def entries(events, team="Home"):
    return [e for e in events if e["type"] == "pitching_change" and e["team"] == team]


def test_engine_uses_designations_without_a_closer():
    home, names = tagged_home()
    away = average_team("Away")
    roles = {pid: u.role for pid, u in home.relief.items()}
    high_late, mopup_blowout = [], []
    for seed in range(300):
        events = simulate_game(away, home, DEFAULT_ENV, seed)
        for e in entries(events):
            assert e["situation"] in ("high", "medium", "low", "mopup", "early")
            if e["into"] == names[1]:
                assert e["outs"] == 0 and e["on_base"] == 0  # start of an inning only
            if e["situation"] == "high":
                high_late.append(roles[e["into"]])
            if e["situation"] == "mopup":
                mopup_blowout.append(roles[e["into"]])
    assert high_late and high_late.count("high") / len(high_late) > 0.6
    assert mopup_blowout and mopup_blowout.count("mopup") / len(mopup_blowout) > 0.4


def test_relievers_go_one_inning_by_default():
    home, names = tagged_home()
    events = simulate_game(average_team("Away"), home, DEFAULT_ENV, 11)
    outs = {}
    current = None
    for e in events:
        if e["type"] == "pitching_change" and e["team"] == "Home":
            current = e["into"]
        elif e["type"] == "pa" and e["half"] == "top" and current in names[:4]:
            outs[current] = outs.get(current, 0) + e["outs"]
    assert outs and all(n <= 6 for n in outs.values())


def test_follow_chain_comes_first():
    home, names = tagged_home(follow=("Home-RP7",))
    for seed in range(20):
        first = entries(simulate_game(average_team("Away"), home, DEFAULT_ENV, seed))[0]
        assert first["into"] == "Home-RP7"


def test_untagged_bullpen_keeps_listed_order():
    home = average_team("Home")
    events = simulate_game(average_team("Away"), home, DEFAULT_ENV, 4)
    used = [e["into"] for e in entries(events)]
    assert used == [p.id for p in home.bullpen][: len(used)]
