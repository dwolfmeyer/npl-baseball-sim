"""Win expectancy, WPA and the box score's win probability chart."""

import pytest

from npl_sim.engine.box import box_score
from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.game import simulate_game
from npl_sim.engine.playlog import game_log
from npl_sim.engine.sample import average_team
from npl_sim.engine.winprob import runs_table, win_expectancy
from npl_sim.reporting.gameday import render_game

EMPTY = [None, None, None]


def bases(*occupied: int) -> list:
    return ["x" if b in occupied else None for b in (1, 2, 3)]


def test_run_table_rows_are_distributions():
    for row in runs_table().values():
        assert sum(row) == pytest.approx(1.0, abs=0.002)
    # More runners and fewer outs mean more runs.
    mean = {k: sum(r * p for r, p in enumerate(v)) for k, v in runs_table().items()}
    assert mean[("123", 0)] > mean[("___", 0)] > mean[("___", 1)] > mean[("___", 2)]


def test_matches_the_book():
    assert win_expectancy(1, "top", 0, EMPTY, 0) == pytest.approx(0.5, abs=0.01)
    # Tango's Leverage Index example: top 9th, home up 3, first and second, no outs.
    assert win_expectancy(9, "top", 3, bases(1, 2), 0) == pytest.approx(0.841, abs=0.03)
    assert win_expectancy(9, "top", 3, bases(1, 2), 1) == pytest.approx(0.910, abs=0.03)
    assert win_expectancy(7, "top", 1, EMPTY, 0) == pytest.approx(0.70, abs=0.04)
    assert win_expectancy(9, "bottom", 0, EMPTY, 0) == pytest.approx(0.65, abs=0.04)


def test_game_states_order_sensibly():
    leads = [win_expectancy(6, "top", lead, EMPTY, 0) for lead in range(-5, 6)]
    assert leads == sorted(leads)
    assert win_expectancy(9, "bottom", 0, bases(3), 0) > win_expectancy(9, "bottom", 0, EMPTY, 2)
    assert win_expectancy(9, "bottom", 1, EMPTY, 0) == 1.0  # a walk-off already happened
    assert win_expectancy(9, "top", 2, EMPTY, 3) == 1.0  # home leads after the top of the 9th
    assert win_expectancy(9, "bottom", -1, EMPTY, 3) == 0.0
    assert win_expectancy(12, "top", 0, bases(2), 0) == pytest.approx(0.5, abs=0.02)


def test_plays_carry_wpa_that_adds_up():
    away, home = average_team("Away"), average_team("Home")
    for seed in range(60):
        log = game_log(simulate_game(away, home, DEFAULT_ENV, seed))
        curve = log.win_curve()
        final = curve[-1][1]
        a, h = final.score.values()
        assert final.home_we == (1.0 if h > a else 0.0)
        signed = sum(p.wpa if half.half == "bottom" else -p.wpa for half, p in curve)
        assert log.home_we_start + signed == pytest.approx(final.home_we)
        assert all(p.li is not None for _, p in curve if p.kind == "pa")


def test_box_score_shows_the_chart():
    events = simulate_game(average_team("Away"), average_team("Home"), DEFAULT_ENV, 7)
    page = render_game(box_score(events), game_log(events))
    assert "Win probability" in page and '<polyline class="curve"' in page
    assert page.count('class="key"') == 3  # the three biggest plays are marked
    assert "WPA " in page and " · LI " in page
