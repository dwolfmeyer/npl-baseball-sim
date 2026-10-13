from dataclasses import replace

import pytest

from npl_sim.engine.box import box_score
from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.fielding import rating_at, team_defense
from npl_sim.engine.game import simulate_game
from npl_sim.engine.model import Defense
from npl_sim.engine.sample import average_team, hitter

SS, LF, FIRST = {"SS": 0.0}, {"LF": 0.0}, {"1B": 0.0}


def test_moving_up_costs_more_than_moving_down():
    assert rating_at(SS, "SS") == 0.0
    assert rating_at({}, "C") == 0.0  # no data: average anywhere
    down, up = rating_at(SS, "2B"), rating_at({"2B": 0.0}, "SS")
    assert 0 < down < 3 and up < -3  # short to second is a free move down; not back up
    assert rating_at(SS, "3B") > 0
    assert rating_at(SS, "LF") < rating_at(SS, "3B") - 2  # infield to outfield hurts
    assert rating_at(LF, "CF") < -8 < 0 < rating_at({"CF": 0.0}, "LF")
    assert rating_at(FIRST, "3B") < -10 and rating_at(FIRST, "C") < -25
    assert rating_at({"SS": 8.0}, "2B") > 0  # a great shortstop is still good at second
    assert rating_at({"SS": 0.0, "LF": 0.0}, "CF") == rating_at(SS, "CF")  # best source


def test_team_defense_moves_hit_rates_on_the_right_balls():
    lf_in_cf = replace(hitter("x"), positions=LF)
    d = team_defense(Defense(), [(lf_in_cf, "CF")])
    assert d.hit_mult["FB"] > d.hit_mult["LD"] > 1.0 and d.hit_mult["GB"] == 1.0
    first_at_short = replace(hitter("y"), positions=FIRST)
    d = team_defense(Defense(), [(first_at_short, "SS")])
    assert d.hit_mult["GB"] > 1.0 and d.hit_mult["FB"] == 1.0
    assert team_defense(Defense(), [(hitter("z"), "SS")]) == Defense()


def test_out_of_position_shows_in_the_box_and_costs_runs():
    def team(swap: bool):
        t = average_team("Home")
        lineup = [(replace(h, positions={p: 0.0} if p != "DH" else {}), p) for h, p in t.lineup]
        if swap:  # shortstop and left fielder trade places
            ss = next(i for i, (_, p) in enumerate(lineup) if p == "SS")
            lf = next(i for i, (_, p) in enumerate(lineup) if p == "LF")
            lineup[ss], lineup[lf] = (lineup[ss][0], "LF"), (lineup[lf][0], "SS")
        return replace(t, lineup=lineup)

    events = simulate_game(average_team("Away"), team(True), DEFAULT_ENV, 1)
    fielding = dict(dict(box_score(events).team_notes["Home"])["Fielding"])
    assert fielding["Out of position"] == "Home-SS (LF), Home-LF (SS)."
    runs = {
        swap: sum(
            simulate_game(average_team("Away"), team(swap), DEFAULT_ENV, s)[-1]["away"]
            for s in range(2000)  # the cost is ~0.1 runs a game, so it takes a big sample
        )
        for swap in (False, True)
    }
    assert runs[True] > runs[False]


def test_zimmerman_pairs_keep_moves_asymmetric():
    # 2B -> 1B costs little; 1B -> 2B costs a lot (Zimmerman, 2015).
    assert rating_at({"2B": 0.0}, "1B") > -2 > -8 > rating_at({"1B": 0.0}, "2B")
    # Center is much harder than the corners; short a little harder than second.
    assert rating_at({"LF": 0.0}, "CF") < -8 and rating_at({"CF": 0.0}, "RF") > 0


def test_catcher_out_from_behind_the_plate_pays():
    catcher = {"C": 0.0}
    assert rating_at(catcher, "3B") < -3 and rating_at(catcher, "LF") < -6
    assert -1 < rating_at(catcher, "1B") <= 0


def test_outfielders_and_shortstops_move_without_penalty():
    from npl_sim.engine.fielding import out_of_position

    # Two elite center fielders: the one in right is better than elite there.
    assert rating_at({"CF": 12.0}, "RF") > 12.0 and rating_at({"CF": 12.0}, "LF") > 12.0
    assert rating_at({"SS": 6.0}, "2B") > 6.0 and rating_at({"SS": 6.0}, "3B") > 6.0
    assert not out_of_position({"CF": 0.0}, "RF") and not out_of_position({"SS": 0.0}, "2B")
    assert out_of_position({"2B": 0.0}, "SS") and out_of_position({"SS": 0.0}, "LF")
    # No experience penalty: an average corner outfielder in center loses only
    # the measured gap, not the gap plus an out-of-position charge.
    assert rating_at({"LF": 0.0}, "CF") > rating_at({"3B": 0.0}, "CF")


def test_a_fielders_rating_is_the_runs_he_saves_in_a_season():
    from npl_sim.engine.defaults import LEAGUE_BIP, LEAGUE_HIT_TABLE
    from npl_sim.engine.fielding import BIP_PER_SEASON, RUNS_PER_HIT_SAVED

    t = average_team("H")
    for pos in ("SS", "CF", "1B", "RF"):
        lineup = [(replace(h, positions={p: 12.0 if p == pos else 0.0}), p) for h, p in t.lineup]
        d = team_defense(Defense(), lineup)
        hits = sum(
            LEAGUE_BIP[b] * sum(LEAGUE_HIT_TABLE[b].values()) * (1 - d.hit_mult.get(b, 1.0))
            for b in LEAGUE_BIP
        )
        assert hits * BIP_PER_SEASON * RUNS_PER_HIT_SAVED == pytest.approx(12.0, rel=0.1)
