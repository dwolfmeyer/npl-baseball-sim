"""Engine realism fixes from the 2026-10-14 engine review (one section per item)."""

from collections import Counter
from dataclasses import replace

import pytest

from npl_sim.calibration import harness as h
from npl_sim.engine.defaults import DEFAULT_ENV, average_split
from npl_sim.engine.model import LEAGUE_FATIGUE, NEUTRAL_PARK, Defense, pa_probabilities

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
