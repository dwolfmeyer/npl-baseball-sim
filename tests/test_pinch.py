from dataclasses import replace

import pytest

from npl_sim.engine.box import box_score
from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.game import simulate_game
from npl_sim.engine.manager import NO_PINCH_HITTING, STYLES, PinchHitting, run_value, style
from npl_sim.engine.model import NEUTRAL_PARK, Defense, pa_probabilities
from npl_sim.engine.playlog import game_log
from npl_sim.engine.sample import average_team, hitter, pitcher

ANY_SPOT = PinchHitting(from_inning=1, min_leverage=0.0, min_gain=0.0)


def lefty_lineup_vs_lefties(pinch, bench, hands="LLLLLLLLL"):
    """Home bats nine lefties against an all-lefty staff, with `bench`."""
    home = replace(average_team("Home", hands), bench=tuple(bench), pinch=pinch)
    away = average_team("Away")
    away = replace(
        away,
        starter=pitcher("Away-SP", "L"),
        bullpen=[pitcher(f"Away-RP{i}", "L", limit=25) for i in range(1, 8)],
    )
    return away, home


def pinch_events(away, home, seeds=range(20)):
    games = [simulate_game(away, home, DEFAULT_ENV, s) for s in seeds]
    return games, [e for g in games for e in g if e["type"] == "pinch_hitter"]


def test_pinch_hits_for_the_platoon_edge_and_logs_it():
    bench = [hitter("Home-R1", bats="R"), hitter("Home-R2", bats="R")]
    away, home = lefty_lineup_vs_lefties(ANY_SPOT, bench)
    games, ph = pinch_events(away, home)
    assert ph and {e["into"] for e in ph} <= {"Home-R1", "Home-R2"}
    for g in games:
        used = [e["into"] for e in g if e["type"] == "pinch_hitter"]
        assert len(used) == len(set(used)) <= 2  # each bench bat once
        box = box_score(g, "Away", "Home")
        assert sum(b.r for b in box.batting["Home"].values()) == box.runs["Home"]
        for pid in used:
            assert box.batting["Home"][pid].position.startswith("PH")
    texts = [p.description for g in games for half in game_log(g).halves for p in half.plays]
    assert any(t.startswith("Offensive Substitution: Pinch-hitter") for t in texts)


def test_rules_hold_the_bench():
    bench = [hitter("Home-R1", bats="R")]
    for rule in (
        NO_PINCH_HITTING,
        replace(ANY_SPOT, from_inning=20),  # never late enough
        replace(ANY_SPOT, min_leverage=50.0),  # never a big enough spot
        replace(ANY_SPOT, min_gain=1.0),  # never a big enough edge
    ):
        away, home = lefty_lineup_vs_lefties(rule, bench)
        assert pinch_events(away, home, range(5))[1] == []
    # Same-handed bench bat: no platoon edge, so no pinch hitter...
    away, home = lefty_lineup_vs_lefties(ANY_SPOT, [hitter("Home-L1", bats="L")])
    assert pinch_events(away, home, range(5))[1] == []
    # ...unless the owner pinch-hits for any edge and he's much the better hitter.
    slugger = hitter("Home-L1", bats="L", scale={"hr": 0.08, "k": 0.12})
    rule = replace(ANY_SPOT, platoon_only=False, min_gain=0.02)
    away, home = lefty_lineup_vs_lefties(rule, [slugger])
    assert pinch_events(away, home, range(5))[1]


def test_never_pinch_hits_away_the_only_catcher():
    bench = [hitter("Home-R1", bats="R")]
    away, home = lefty_lineup_vs_lefties(
        replace(ANY_SPOT, never_for=tuple(f"Home-{p}" for p in ("1B", "2B", "3B", "SS",
                                                               "LF", "CF", "RF", "DH"))),
        bench,
    )  # fmt: skip
    assert pinch_events(away, home, range(5))[1] == []
    catcher = replace(hitter("Home-R2", bats="R"), positions={"C": 0.0})
    away, home = lefty_lineup_vs_lefties(home.pinch, [catcher])
    assert {e["into"] for e in pinch_events(away, home, range(5))[1]} == {"Home-R2"}


def test_manager_styles():
    assert set(STYLES) == {"sabermetric", "aggressive", "balanced", "old_school", "passive"}
    assert not style("passive").pinch.enabled
    assert style("old_school").pinch.from_inning > style("aggressive").pinch.from_inning
    assert style("sabermetric").hook.times_through == 2
    with pytest.raises(ValueError):
        style("chaos")


def test_run_value_prefers_the_better_hitter():
    def value(h, p):
        side = h.side_vs(p.throws)
        probs = pa_probabilities(h.rates_vs(p.throws), p.rates_vs(side), DEFAULT_ENV, side,
                                 p.throws, NEUTRAL_PARK, Defense())  # fmt: skip
        return run_value(probs)

    lhp = pitcher("x", "L")
    assert value(hitter("r", bats="R"), lhp) > value(hitter("l", bats="L"), lhp)
    assert value(hitter("s", bats="R", scale={"hr": 0.06}), lhp) > value(hitter("r"), lhp)


def test_limits_per_game_and_out_of_position():
    bench = [hitter(f"Home-R{i}", bats="R") for i in range(1, 5)]
    away, home = lefty_lineup_vs_lefties(replace(ANY_SPOT, max_per_game=1), bench)
    for g in pinch_events(away, home, range(10))[0]:
        assert sum(e["type"] == "pinch_hitter" for e in g) <= 1
    # A bench bat who only plays first base can't hit for the shortstop.
    first_baseman = replace(hitter("Home-R1", bats="R"), positions={"1B": 0.0})
    rule = replace(
        ANY_SPOT,
        never_for=tuple(f"Home-{p}" for p in ("C", "1B", "2B", "3B", "LF", "CF", "RF", "DH")),
    )
    away, home = lefty_lineup_vs_lefties(rule, [first_baseman])  # fmt: skip
    assert pinch_events(away, home, range(5))[1] == []
    away, home = lefty_lineup_vs_lefties(replace(rule, out_of_position=True), [first_baseman])
    assert pinch_events(away, home, range(5))[1]
