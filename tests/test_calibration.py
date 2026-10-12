"""Calibration harness gates (Milestone 8). If the drift test fails after an
intended engine change, run `npl-sim calibrate --save-baseline` and commit
tests/baselines/engine.json with the change."""

import pytest

from npl_sim.calibration import harness as h


@pytest.fixture(scope="module")
def games():
    return h.average_games(1000)


def test_box_scores_add_up(games):
    assert h.identities(games) == []


def test_placeholder_run_environment(games):
    off = [c for c in h.check_targets(h.summary(games), h.load_targets("placeholder")) if not c.ok]
    assert off == []


def test_k_spread_is_binomial_and_shrinks_with_sample(games):
    sd50, bin50 = h.k_spread(games, 50)
    sd200, bin200 = h.k_spread(games, 200)
    assert sd50 == pytest.approx(bin50, rel=0.15)
    assert sd200 == pytest.approx(bin200, rel=0.15)
    assert 1.6 < sd50 / sd200 < 2.4  # quadruple the PA, half the spread


def test_archetypes_play_back_their_cards():
    report = h.archetype_report(150)
    for kind, r in report.items():
        sim, exp = r["sim"], r["expected"]
        for k in ("K", "BB", "HR"):
            # Fatigue adds walks and homers late in starts, so allow some drift.
            assert 0.8 < sim[k] / exp[k] < 1.25, (kind, k, sim[k], exp[k])
    assert report["three_true_outcomes"]["sim"]["tto"] > 0.5  # comes out as one
    assert report["slap"]["sim"]["tto"] < 0.22
    hr = {k: r["sim"]["HR"] for k, r in report.items()}
    assert hr["three_true_outcomes"] > hr["power"] > hr["average"] > hr["slap"]


def test_fast_teams_run_more():
    assert h.steals_per_game(2.0) > 2 * h.steals_per_game(0.0) > 0


def test_engine_matches_its_baseline(games):
    d = h.drift(h.load_baseline(), h.fingerprint(games))
    assert d["jsd"] <= h.DRIFT_LIMITS["jsd"], (
        f"PA result mix shifted (JSD {d['jsd']:.5f}); if intended, "
        "run `npl-sim calibrate --save-baseline`"
    )
    assert d["wasserstein"] <= h.DRIFT_LIMITS["wasserstein"], (
        f"runs per game shifted (W1 {d['wasserstein']:.3f}); if intended, "
        "run `npl-sim calibrate --save-baseline`"
    )


def test_drift_catches_a_real_change(games):
    """A 25% bump in home runs is caught; a different seed set isn't."""
    base = h.fingerprint(games)
    other_seeds = h.fingerprint(h.average_games(1000, seed=5000))
    d = h.drift(base, other_seeds)
    assert d["jsd"] <= h.DRIFT_LIMITS["jsd"] and d["wasserstein"] <= h.DRIFT_LIMITS["wasserstein"]

    from dataclasses import replace

    from npl_sim.engine.defaults import DEFAULT_ENV

    juiced = replace(
        DEFAULT_ENV,
        rates={k: replace(r, hr=r.hr * 1.25) for k, r in DEFAULT_ENV.rates.items()},
    )
    shifted = h.drift(base, h.fingerprint(h.average_games(1000, env=juiced)))
    assert (
        shifted["jsd"] > h.DRIFT_LIMITS["jsd"]
        or shifted["wasserstein"] > h.DRIFT_LIMITS["wasserstein"]
    )


def test_distances_and_targets():
    assert h.js_divergence({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == 0
    assert h.js_divergence({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)
    assert h.wasserstein([1, 0, 0], [0, 0, 1]) == pytest.approx(2.0)
    checks = h.check_targets({"r_per_g": 4.6}, {"r_per_g": {"target": 4.48, "tol": 0.03}})
    assert checks[0].ok and checks[0].hi == pytest.approx(4.6144)
    with pytest.raises(KeyError):
        h.load_targets("mlb_1901")
    assert set(h.load_targets("mlb_2026")) == {"r_per_g", "k_pct", "bb_pct"}
