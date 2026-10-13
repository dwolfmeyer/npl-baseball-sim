"""The Gameday at-bat pop-up: simulated pitch sequences and the pitcher's mix."""

from dataclasses import replace

import pandas as pd

from npl_sim.engine.box import box_score
from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.game import simulate_game
from npl_sim.engine.playlog import game_log
from npl_sim.engine.sample import average_team
from npl_sim.reporting.gameday import render_game
from npl_sim.talent.load import arsenals

AWAY, HOME = average_team("Away"), average_team("Home")


def test_pitches_fit_the_logged_plate_appearance():
    for seed in range(40):
        events = simulate_game(AWAY, HOME, DEFAULT_ENV, seed)
        pas = [e for e in events if e["type"] == "pa"]
        plays = [p for _, ps in game_log(events).plays() for p in ps if p.kind == "pa"]
        assert len(pas) == len(plays)
        for e, play in zip(pas, plays, strict=True):
            seq = play.at_bat.pitches
            assert len(seq) == e["pitches"]
            for q in seq[:-1]:  # nothing ends the PA early
                assert q.balls <= 3 and q.strikes <= 2 and not q.call.startswith("In play")
            last = seq[-1]
            if e["result"] == "K":
                assert last.strikes == 3 and last.balls == e["count"][0]
            elif e["result"] == "BB":
                assert last.balls == 4 and last.call == "Ball"
            elif e["result"] == "HBP":
                assert last.call == "Hit By Pitch"
            else:
                assert last.call.startswith("In play")
                assert (last.balls, last.strikes) == tuple(e["count"])


def test_sequences_are_stable_and_leave_results_alone():
    a = simulate_game(AWAY, HOME, DEFAULT_ENV, 3)
    b = simulate_game(AWAY, HOME, DEFAULT_ENV, 3)
    assert a == b
    seq = [p.at_bat.pitches for _, ps in game_log(a).plays() for p in ps if p.at_bat]
    again = [p.at_bat.pitches for _, ps in game_log(b).plays() for p in ps if p.at_bat]
    assert seq == again


def test_real_pitch_mix_is_used():
    mix = (("SI", 0.6, 96.0), ("SL", 0.4, 88.0))
    home = replace(HOME, starter=replace(HOME.starter, arsenal=mix))
    events = simulate_game(AWAY, home, DEFAULT_ENV, 1)
    start = next(e for e in events if e["type"] == "game_start")
    assert start["home"]["pitchers"][0]["arsenal"] == [list(m) for m in mix]
    thrown = [
        q
        for half, ps in game_log(events).plays()
        if half.half == "top"
        for p in ps
        if p.at_bat and p.pitcher == home.starter.id
        for q in p.at_bat.pitches
    ]
    assert thrown and {q.code for q in thrown} <= {"SI", "SL"}
    sinkers = [q.mph for q in thrown if q.code == "SI"]
    assert 94 < sum(sinkers) / len(sinkers) < 98


def test_arsenal_from_statcast():
    rows = [("FF", 95.0)] * 60 + [("SL", 86.0)] * 39 + [("EP", 70.0)]  # 1% eephus dropped
    season = pd.DataFrame(
        {"pitcher": 1, "pitch_type": [r[0] for r in rows], "release_speed": [r[1] for r in rows]}
    )
    last = pd.DataFrame({"pitcher": [2] * 10, "pitch_type": "CH", "release_speed": 84.0})
    mix = arsenals(season, last, min_pitches=50)
    assert mix["1"] == (("FF", 0.606, 95.0), ("SL", 0.394, 86.0))
    assert mix["2"] == (("CH", 1.0, 84.0),)
    # Too few pitches this season: keep last season's mix.
    assert arsenals(season.head(20), season, min_pitches=50)["1"][1][0] == "SL"


def test_play_by_play_opens_an_at_bat():
    events = simulate_game(AWAY, HOME, DEFAULT_ENV, 7)
    page = render_game(box_score(events), game_log(events))
    pas = sum(e["type"] == "pa" for e in events)
    assert page.count("data-ab=") == pas and page.count('<template id="ab-') == pas
    assert '<dialog class="ab" id="atbat">' in page and "win probability:" in page
