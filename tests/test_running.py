"""Stealing and taking the extra base (engine/game.py _maybe_steal, _send)
and the real records behind them (talent/running.py)."""

from collections import Counter
from dataclasses import replace

import pandas as pd
import pytest

from npl_sim.engine.box import box_score
from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.game import simulate_game, steal_odds
from npl_sim.engine.manager import Baserunning
from npl_sim.engine.model import Defense
from npl_sim.engine.playlog import format_log, game_log
from npl_sim.engine.sample import average_team, hitter
from npl_sim.talent import running as tr

HOME = average_team("Home")


def runners(**changes):
    """An average away team whose hitters all have `changes` (speed, steal_attempt...)."""
    team = average_team("Away")
    lineup = [(replace(h, **changes), pos) for h, pos in team.lineup]
    return replace(team, lineup=lineup)


def play(away, games=400, home=HOME):
    events = []
    for seed in range(games):
        g = simulate_game(away, home, DEFAULT_ENV, seed)
        box_score(g)  # runs add up with steals and runners thrown out
        events += [e for e in g if e.get("half") == "top"]
    return events


def steals(events):
    c = Counter(
        (e["type"], e["base"]) for e in events if e["type"] in ("stolen_base", "caught_stealing")
    )
    tries = sum(c.values())
    made = c["stolen_base", "2B"] + c["stolen_base", "3B"]
    third = c["stolen_base", "3B"] + c["caught_stealing", "3B"]
    return tries, made / max(tries, 1), third / max(tries, 1)


@pytest.fixture(scope="module")
def league_runners():
    # Runners at the league's attempt and success rates, as real data averages out.
    return play(
        runners(steal_attempt=DEFAULT_ENV.steal_attempt, steal_success=DEFAULT_ENV.steal_success),
        600,
    )


def test_league_runners_steal_like_mlb_today(league_runners):
    tries, success, third = steals(league_runners)
    assert 0.75 < tries / 600 < 1.2  # about 0.96 attempts a team game in 2024-25
    assert 0.72 < success < 0.83
    assert 0.04 < third < 0.18  # about one try in ten is for third


def test_real_record_drives_how_often_and_how_well():
    burner = steals(play(runners(steal_attempt=0.30, steal_success=0.90)))
    station = steals(play(runners(steal_attempt=0.02, steal_success=0.60)))
    assert burner[0] > 8 * station[0]
    assert burner[1] > 0.84 and station[1] < 0.75


def test_speed_fills_in_without_a_record():
    fast, _ = steal_odds(hitter("f", speed=2.0), DEFAULT_ENV)
    slow, ok = steal_odds(hitter("s", speed=-2.0), DEFAULT_ENV)
    assert fast > 10 * slow and ok < DEFAULT_ENV.steal_success


def test_catcher_arm_throws_out_more_runners():
    league = runners(steal_attempt=0.2, steal_success=0.78)
    weak = steals(play(league, home=replace(HOME, defense=Defense(catcher_arm=-2.0))))
    strong = steals(play(league, home=replace(HOME, defense=Defense(catcher_arm=2.0))))
    assert weak[1] - strong[1] > 0.1


def test_owner_aggressiveness_and_red_lights():
    team = runners(steal_attempt=0.15)
    normal = steals(play(team))
    bold = steals(play(replace(team, running=Baserunning(steal=2.0))))
    assert bold[0] > 1.6 * normal[0] and bold[1] < normal[1]  # more tries, more caught
    assert steals(play(replace(team, running=Baserunning(steal=0.0))))[0] == 0
    red = Baserunning(no_steal=tuple(h.id for h, _ in team.lineup[:8]))
    events = play(replace(team, running=red))
    assert {e["runner"] for e in events if e["type"] in ("stolen_base", "caught_stealing")} == {
        team.lineup[8][0].id
    }


def test_no_steals_of_third_with_two_out_or_in_blowouts(league_runners):
    for e in league_runners:
        if e["type"] in ("stolen_base", "caught_stealing") and e["base"] == "3B":
            assert e["outs_after"] - (e["type"] == "caught_stealing") < 2


def thrown_out(events):
    return [t for e in events if e["type"] == "pa" for t in e.get("thrown_out", ())]


def scored_from_second_on_singles(events):
    tries = [
        e for e in events if e["type"] == "pa" and e["result"] == "1B" and e["bases_before"][1]
    ]
    home = sum(any(m["start"] == "2B" and m["end"] == "score" for m in e["runners"]) for e in tries)
    return home / len(tries)


def test_runners_get_thrown_out_and_speed_decides_it():
    norm = play(runners(steal_attempt=0.0))
    outs = thrown_out(norm)
    assert 0.03 < len(outs) / 400 < 0.25  # per team game
    fast = play(runners(steal_attempt=0.0, speed=1.5))
    slow = play(runners(steal_attempt=0.0, speed=-1.5))
    assert scored_from_second_on_singles(fast) > scored_from_second_on_singles(slow) + 0.1
    assert len(thrown_out(fast)) < len(thrown_out(slow))


def test_extra_base_aggressiveness():
    team = runners(steal_attempt=0.0)
    careful = play(replace(team, running=Baserunning(extra_base=0.5)))
    bold = play(replace(team, running=Baserunning(extra_base=2.0)))
    assert scored_from_second_on_singles(bold) > scored_from_second_on_singles(careful)
    assert len(thrown_out(bold)) > 2 * len(thrown_out(careful))


def test_play_by_play_names_steals_of_third_and_runners_thrown_out(league_runners):
    for seed in range(400):
        g = simulate_game(runners(steal_attempt=0.3), HOME, DEFAULT_ENV, seed)
        text = format_log(game_log(g))
        if "steals 3rd base" in text and "thrown out at" in text:
            break
    else:
        pytest.fail("no game with a steal of third and a runner thrown out")
    e = next(e for e in g if e["type"] == "pa" and e.get("thrown_out"))
    out = e["thrown_out"][0]
    assert any(
        m["runner"] == out["runner"] and m.get("out_at") == out["base"] for m in e["runners"]
    )


# --- real records -------------------------------------------------------------


def lines(rows):
    cols = ["mlbam_id", "h", "doubles", "triples", "hr", "bb", "hbp", "sb", "cs"]
    return pd.DataFrame(rows, columns=cols)


def test_steal_ratings_regress_toward_speed():
    league = (0.105, 0.775)
    # 100 times on first: 80 singles + 20 walks; 30 steals, 2 caught.
    batting = lines([(1, 80, 0, 0, 0, 20, 0, 30, 2), (2, 80, 0, 0, 0, 20, 0, 0, 0)])
    rated = tr.steal_ratings([(0, batting)], {"1": 1.5, "2": -1.0}, league)
    att, ok = rated["1"]
    assert 0.2 < att < 0.32 and 0.85 < ok < 0.94  # pulled toward his speed's rate, not all the way
    assert rated["2"][0] < 0.03
    # Last season counts, a little less than this one.
    two = tr.steal_ratings(
        [(0, batting.iloc[1:]), (1, batting.iloc[:1].assign(mlbam_id=2))], {}, league
    )
    assert two["2"][0] > rated["2"][0]


def test_league_rates_need_enough_attempts():
    few = lines([(1, 50, 0, 0, 0, 10, 0, 5, 1)])
    assert tr.league_rates(few) is None
    many = lines([(i, 50, 0, 0, 0, 10, 0, 5, 2) for i in range(60)])
    attempt, success = tr.league_rates(many)
    assert attempt == pytest.approx(7 / 60) and success == pytest.approx(5 / 7)


def test_catcher_arms_from_pop_time_and_caught_stealing_runs():
    pop = pd.DataFrame(
        {"mlbam_id": [5, 6], "pop_2b_sba": [1.85, 2.05], "pop_2b_sba_count": [60, 60]}
    )
    fg = pd.DataFrame(
        {"mlbam_id": [5, 7], "Pos": ["C", "C"], "Inn": [1000.0, 1000.0], "rSB": [4.0, -3.0]}
    )
    arms = tr.catcher_arms([(0, pop, fg)])
    assert arms["5"] > 1.0  # quick pop and throws runners out
    assert arms["6"] < 0  # slow pop, no FanGraphs line
    assert arms["7"] < 0  # no pop time, gives up steals
