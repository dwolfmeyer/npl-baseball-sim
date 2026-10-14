"""Engine realism fixes from the 2026-10-14 engine review (one section per item)."""

from collections import Counter
from dataclasses import replace

import pytest

from npl_sim.calibration import harness as h
from npl_sim.engine.defaults import DEFAULT_ENV, average_split
from npl_sim.engine.game import simulate_game
from npl_sim.engine.model import LEAGUE_FATIGUE, NEUTRAL_PARK, Defense, pa_probabilities

from .test_engine import HITS, probs, total

# --- item 1: fatigue is centered on the league, not added on top of it -------------------


def rates_vs_input(games):
    want = h.input_rates(games)
    pa = Counter(e["result"] for g in games for e in g if e["type"] == "pa")
    n = sum(pa.values())
    return {k: (pa[k] / n) / want[k] for k in ("K", "BB", "HR")}


@pytest.fixture(scope="module")
def no_fatigue_games():
    return h.average_games(1000, env=replace(DEFAULT_ENV, fatigue=False))


def test_average_teams_reproduce_the_input_rates_with_fatigue_off(no_fatigue_games):
    for stat, ratio in rates_vs_input(no_fatigue_games).items():
        assert ratio == pytest.approx(1.0, abs=0.02), (stat, ratio)


def test_average_teams_reproduce_the_input_rates_with_fatigue_on():
    for stat, ratio in rates_vs_input(h.average_games(1000)).items():
        assert ratio == pytest.approx(1.0, abs=0.02), (stat, ratio)


def test_fresh_arms_beat_the_league_and_tired_ones_lose_to_it():
    b = average_split("R", "R")

    def stats(**kw):
        p = pa_probabilities(b, b, DEFAULT_ENV, "R", "R", NEUTRAL_PARK, Defense(), **kw)
        return p["K"], p["BB"], p["HR"]

    fresh, league, tired = stats(fatigue=0.0), stats(fatigue=LEAGUE_FATIGUE), stats(fatigue=1.0)
    assert fresh[0] > league[0] > tired[0]  # strikeouts
    assert fresh[1] < league[1] < tired[1]  # walks
    assert fresh[2] < league[2] < tired[2]  # home runs
    assert stats() == league  # no fatigue given means the league average


# --- item 2: power reallocates hits, it doesn't add them --------------------------------


def hit_profile(**card):
    """(BABIP, extra-base share of hits) for a hitter with `card` changes vs an average pitcher."""
    p = probs(batter=replace(average_split("R", "R"), **card))
    bip = sum(v for k, v in p.items() if ":" in k)
    hits = sum(total(p, h) for h in HITS)
    return hits / bip, (total(p, "2B") + total(p, "3B")) / hits


def test_power_moves_hits_between_singles_and_extra_bases_without_changing_babip():
    base_babip, base_share = hit_profile()
    for power in (1.2, 1.5):
        babip, share = hit_profile(power=power)
        assert abs(babip - base_babip) <= 0.001, (power, babip, base_babip)
        assert share > base_share
    babip, share = hit_profile(power=0.8)
    assert abs(babip - base_babip) <= 0.001
    assert share < base_share


# --- item 3: alignment presets act on the hit types their names say ---------------------


def bip_rates(alignment="normal"):
    """Per ball in play: singles, extra-base hits, all hits; and 2B+3B on grounders/liners."""
    p = probs(alignment=alignment)
    bip = sum(v for k, v in p.items() if ":" in k)
    singles = total(p, "1B")
    xbh = total(p, "2B") + total(p, "3B")
    return {
        "1B": singles / bip,
        "xbh": xbh / bip,
        "hits": (singles + xbh) / bip,
        "gb_xbh": (p["GB:2B"] + p["GB:3B"]) / bip,
        "lc_xbh": (p["LD:2B"] + p["LD:3B"]) / bip,
        "gb_1b": p["GB:1B"] / bip,
    }


def test_no_doubles_cuts_extra_base_hits_and_gives_singles():
    base, deep = bip_rates(), bip_rates("no_doubles")
    assert 0.07 < 1 - deep["xbh"] / base["xbh"] < 0.13  # about 10% fewer 2B and 3B
    assert 0.015 < deep["1B"] / base["1B"] - 1 < 0.05  # shallow flares fall in
    assert abs(deep["hits"] - base["hits"]) < 0.003  # a trade, not a gift or a theft


def test_guard_lines_cuts_doubles_down_the_line_and_costs_singles_up_the_middle():
    base, guard = bip_rates(), bip_rates("guard_lines")
    assert 0.15 < 1 - guard["gb_xbh"] / base["gb_xbh"] < 0.25  # grounders down the line
    assert 0.02 < 1 - guard["lc_xbh"] / base["lc_xbh"] < 0.06  # liners, less so
    assert 1 - guard["xbh"] / base["xbh"] > 0.04  # about 5% of all XBH, the old preset gave 1.3%
    assert guard["gb_1b"] > base["gb_1b"]  # the hole between the guards opens
    # The stated cost: hits allowed rise by a few thousandths per ball in play.
    assert 0.0 < guard["hits"] - base["hits"] < 0.006


def test_infield_in_and_double_play_still_move_every_grounder_hit_alike():
    p0, p1, p2 = probs(), probs(alignment="infield_in"), probs(alignment="double_play")
    for hit in HITS:
        assert p1[f"GB:{hit}"] / p0[f"GB:{hit}"] == pytest.approx(1.12, rel=1e-6)
        assert p2[f"GB:{hit}"] / p0[f"GB:{hit}"] == pytest.approx(1.03, rel=1e-6)
    assert p1["LD:2B"] == pytest.approx(p0["LD:2B"])


# Detail stream (item 4): fielders, counts and strikes draw from detail_rng and must never
# change what happens in the game.
_DETAIL_KEYS = {"fielder", "count", "strikes", "k_type", "direction", "batted_ball"}


def _play_with_detail_seed(seed, detail_seed):
    import random

    from npl_sim.engine.game import GameSim

    from .test_engine import AWAY, HOME

    sim = GameSim(AWAY, HOME, DEFAULT_ENV, seed)
    sim.detail_rng = random.Random(detail_seed)
    return sim.play()


def _without_detail(events):
    return [{k: v for k, v in e.items() if k not in _DETAIL_KEYS} for e in events]


def test_swapping_the_detail_stream_changes_no_event_or_score():
    for seed in range(60):
        a = _play_with_detail_seed(seed, 1)
        b = _play_with_detail_seed(seed, 99)
        assert a[-1] == b[-1]  # final score and line
        assert _without_detail(a) == _without_detail(b)
    assert a != b  # the detail itself does differ


# Steals (item 5): an average lineup has a spread of speeds, not nine zeros.
def test_average_lineup_has_a_deterministic_normal_spread_of_speeds():
    from statistics import fmean, pstdev

    from .test_engine import AWAY

    speeds = sorted(hit.speed for hit, _ in AWAY.lineup)
    assert len(set(speeds)) == 9
    assert fmean(speeds) == pytest.approx(0, abs=1e-9)
    assert 0.9 < pstdev(speeds) < 1.05


def test_average_teams_steal_at_the_league_rate():
    from .test_engine import AWAY, HOME

    att = sb = 0
    games = 600
    for seed in range(games):
        for e in simulate_game(AWAY, HOME, DEFAULT_ENV, seed):
            if e["type"] in ("stolen_base", "caught_stealing"):
                att += 1
                sb += e["type"] == "stolen_base"
    assert att / games / 2 == pytest.approx(0.96, rel=0.10)  # attempts per team-game
    assert sb / games / 2 == pytest.approx(0.74, rel=0.10)  # steals per team-game
