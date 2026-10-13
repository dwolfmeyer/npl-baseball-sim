from datetime import date

import pandas as pd
import pytest

from npl_sim.engine.fielding import team_defense
from npl_sim.engine.model import Defense
from npl_sim.engine.sample import hitter
from npl_sim.talent import blend as tb
from npl_sim.talent import fielding as tf
from npl_sim.talent.env import default_league
from npl_sim.talent.events import plate_appearances
from npl_sim.talent.load import WeekTalent


def _balls(n, **fielders):
    row = {"bb_type": "ground_ball", "events": "field_out"}
    row.update({f"fielder_{k}": v for k, v in fielders.items()})
    return pd.DataFrame([row] * n)


def test_bip_on_field_counts_each_position():
    pitches = pd.concat(
        [_balls(30, **{"6": 1, "2": 9}), _balls(10, **{"4": 1, "2": 9})], ignore_index=True
    )
    pitches = pd.concat(
        [pitches, pd.DataFrame([{"bb_type": "fly_ball", "events": "home_run", "fielder_6": 1}])]
    )
    got = {(r.mlbam_id, r.position): r.bip for r in tf.bip_on_field(pitches).itertuples()}
    assert got == {(1, "SS"): 30, (1, "2B"): 10, (9, "C"): 40}  # homers aren't chances


def fg(rows):
    return pd.DataFrame(rows, columns=["mlbam_id", "Pos", "Inn", "DRS", "UZR"])


def test_defense_rating_averages_oaa_drs_and_uzr_over_three_years():
    # A full-time shortstop, +10 DRS and +6 UZR a year, +8 OAA runs (Statcast).
    oaa = pd.DataFrame(
        {"mlbam_id": [1], "primary_pos_formatted": ["SS"], "fielding_runs_prevented": [8]}
    )
    full = fg([[1, "SS", "1350.0", 10, 6], [1, "2B", "45.1", 1, 0]])
    seasons = [(back, full, oaa, None) for back in range(4)]
    r = tf.defense_ratings(seasons)
    # (10 + 6 + 8) / 3 = +8 per 150 games, regressed by 700 of 5400 innings.
    assert r["1"]["SS"] == pytest.approx(8 * 5400 / 6100)
    assert 0 < r["1"]["2B"] < 4.0  # +15 a season in 181 innings, mostly regressed
    # Seasons more than three back don't count.
    assert tf.defense_ratings(seasons + [(4, fg([[1, "SS", "1350", -50, -50]]), None, None)]) == r
    # One season counts less than three.
    assert 0 < tf.defense_ratings(seasons[:1])["1"]["SS"] < r["1"]["SS"]


def test_defense_rating_rates_part_timers_per_150_games():
    # 450 innings at -5 DRS / -3 UZR is -12 per 150 games before regression.
    r = tf.defense_ratings([(1, fg([[2, "LF", "450", -5, -3]]), None, None)])
    assert r["2"]["LF"] == pytest.approx(-12 * 450 / 1150)


def test_statcast_only_uses_balls_in_play_for_innings():
    oaa = pd.DataFrame(
        {"mlbam_id": [3], "primary_pos_formatted": ["CF"], "fielding_runs_prevented": ["6"]}
    )
    field = pd.DataFrame({"mlbam_id": [3], "position": ["CF"], "bip": [3750]})
    r = tf.defense_ratings([(0, None, oaa, field)])
    assert r["3"]["CF"] == pytest.approx(6 * 1350 / 2050)


def test_positions_default_to_average_and_move_free_within_families():
    eligible = {"CF": 0.0, "RF": 0.0, "1B": 0.0}
    got = tf.positions(eligible, {"CF": 12.0})
    assert got["CF"] == 12.0
    assert got["RF"] > 12.0  # an elite center fielder is better than elite in right
    assert got["1B"] == 0.0  # no rating at first: average there
    assert tf.positions({"SS": 0.0, "2B": 0.0}, {"SS": 6.0})["2B"] > 6.0
    assert tf.positions({"LF": 0.0, "CF": 0.0}, {"LF": -3.0})["CF"] == 0.0  # never below average
    assert tf.positions({"2B": 0.0, "3B": 0.0}, None) == {"2B": 0.0, "3B": 0.0}


def test_framing_arms_and_parks():
    field = pd.DataFrame({"mlbam_id": [9], "position": ["C"], "bip": [3000]})
    framing = tf.framing_ratings([(0, pd.DataFrame({"mlbam_id": [9], "rv_tot": [15.0]}), field)])
    assert 10 < framing["9"] < 15  # +15 a season, a little regressed
    arm = tf.arms(
        pd.DataFrame({"mlbam_id": [7, 8], "arm_of": [94.0, 82.0]}),
        pd.DataFrame({"mlbam_id": [9], "pop_2b_sba": [1.85]}),
    )
    assert arm == {"7": 2.0, "8": -1.0, "9": pytest.approx(1.25)}
    parks = tf.park_factors(
        pd.DataFrame(
            {
                "main_team_id": [115, 115],
                "key_bat_side": ["L", "R"],
                "index_hr": ["130", "125"],
                "index_1b": ["110", "110"],
                "index_2b": [120, 120],
                "index_3b": [200, 200],
            }
        )  # fmt: skip
    )
    assert parks[("COL", "L")]["hr"] == pytest.approx(1.3)


def test_park_neutralizes_a_hitters_record():
    rows = [{"batter": 1, "pitcher": 2, "stand": "R", "p_throws": "R", "home_team": "COL",
             "game_date": date(2026, 5, 4), "events": "home_run", "bb_type": "fly_ball",
             "description": "hit_into_play"}] * 20 + [
            {"batter": 1, "pitcher": 2, "stand": "R", "p_throws": "R", "home_team": "COL",
             "game_date": date(2026, 5, 4), "events": "field_out", "bb_type": "fly_ball",
             "description": "hit_into_play"}] * 380  # fmt: skip
    pa = plate_appearances(pd.DataFrame(rows))
    lg = default_league()
    coors = {("COL", "R"): {"hr": 1.3, "1b": 1.1, "2b": 1.2, "3b": 2.0}}
    plain = tb.blend([tb.Season(0, pa, lg)], tb.Season(0, pa.iloc[0:0], lg))["batter:1"]
    neutral = tb.blend([tb.Season(0, pa, lg, parks=coors)], tb.Season(0, pa.iloc[0:0], lg))
    assert neutral["batter:1"].layers["baseline"]["HR"] < plain.layers["baseline"]["HR"]


def test_week_talent_fills_gloves_and_arms():
    wt = WeekTalent(
        league=default_league(),
        range={"1": {"SS": 6.0}},
        framing={"9": 19.0},
        arm={"7": 1.5, "9": 1.0},
    )
    ss = wt.hitter(hitter("1"))
    assert ss.positions == {"SS": 6.0}
    eligible = wt.hitter(tb.hitter_card(hitter("1"), None, default_league()))
    assert eligible.positions["SS"] == 6.0
    two_way = wt.hitter(hitter("1").__class__(**{**hitter("1").__dict__, "positions": {"2B": 0.0}}))
    # Eligible at second only: a shortstop's rating carries over, with credit.
    assert two_way.positions["2B"] > 6.0 and "SS" not in two_way.positions
    c = wt.hitter(hitter("9"))
    assert c.positions == {"C": 19.0} and c.arm == 1.0
    of = wt.hitter(hitter("7"))
    d = team_defense(Defense(), [(c, "C"), (of, "RF"), (hitter("x"), "LF"), (hitter("y"), "CF")])
    assert d.catcher_arm == 1.0 and d.of_arm == pytest.approx(0.5)
    assert d.framing == pytest.approx(0.05)
