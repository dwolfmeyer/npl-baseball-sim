"""Gameday-style output: box score notes, decisions, and the filterable play log."""

from collections import Counter

import pytest

from npl_sim.engine.box import box_score, dp_numbers, format_box
from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.game import simulate_game
from npl_sim.engine.playlog import format_log, game_log
from npl_sim.engine.sample import average_team
from npl_sim.reporting.gameday import render_game

AWAY, HOME = average_team("Away"), average_team("Home")


@pytest.fixture(scope="module")
def games():
    return [simulate_game(AWAY, HOME, DEFAULT_ENV, s) for s in range(300)]


def test_every_game_has_one_winner_and_one_loser(games):
    saves = 0
    for events in games:
        box = box_score(events)
        decisions = {
            team: Counter(d for p in box.pitching[team].values() for d in p.decisions)
            for team in (box.away, box.home)
        }
        loser = box.home if box.winner == box.away else box.away
        assert decisions[box.winner]["W"] == 1 and decisions[loser]["W"] == 0
        assert decisions[loser]["L"] == 1 and decisions[box.winner]["L"] == 0
        assert decisions[loser]["S"] == 0 and decisions[box.winner]["S"] <= 1
        saves += decisions[box.winner]["S"]
        for team in (box.away, box.home):
            for p in box.pitching[team].values():
                assert not {"W", "S"} <= set(p.decisions)
                assert not {"H", "S"} <= set(p.decisions)
    assert 0 < saves < len(games)


def test_pitching_notes_are_consistent(games):
    for events in games[:100]:
        box = box_score(events)
        for team in (box.away, box.home):
            for p in box.pitching[team].values():
                assert p.strikes <= p.pitches
                assert p.inherited_scored <= p.inherited
                assert p.er <= p.r
            lob = sum(
                e["left_on_base"]
                for e in events
                if e["type"] == "half_end" and (e["half"] == "top") == (team == box.away)
            )
            assert box.lob[team] == lob


def test_runner_movements_match_the_bases(games):
    order = {"1B": 0, "2B": 1, "3B": 2}
    for events in games[:100]:
        for e in (e for e in events if e["type"] == "pa"):
            on = {m["runner"]: m["end"] for m in e["runners"] if m["end"] in order}
            assert {r for r in e["bases_after"] if r} == set(on)
            for runner, base in on.items():
                assert e["bases_after"][order[base]] == runner
            assert sum(m["end"] == "score" for m in e["runners"]) == len(e["scored"])
            assert sum(m["end"] == "out" for m in e["runners"]) == e["outs"]
            balls, strikes = e["count"]
            assert balls <= 4 and strikes <= 3


def test_scoring_plays_filter_adds_up_to_the_final(games):
    for events in games[:100]:
        log = game_log(events)
        final = events[-1]
        scoring = [p for _, plays in log.plays("scoring") for p in plays]
        assert all(p.scoring for p in scoring)
        assert sum(p.runs for p in scoring) == final["away"] + final["home"]
        last = scoring[-1].score
        assert list(last.values()) == [final["away"], final["home"]]
        every = [p for _, plays in log.plays("all") for p in plays]
        assert len(every) > len(scoring)
        assert all(p.description.endswith(".") and "  " not in p.description for p in every)


def test_play_filter_is_validated():
    with pytest.raises(ValueError):
        game_log(simulate_game(AWAY, HOME, DEFAULT_ENV, 1)).plays("hits")


def test_double_play_chains():
    assert dp_numbers("SS") == "6-4-3"
    assert dp_numbers("2B") == "4-6-3"
    assert dp_numbers("3B") == "5-4-3"


def test_text_and_html_render():
    events = simulate_game(AWAY, HOME, DEFAULT_ENV, 3)
    box, log = box_score(events), game_log(events)
    text = format_box(box)
    assert "Team LOB" in text and "Pitches-strikes" in text and "(W)" in text
    assert format_log(log, "scoring").count("(AWA") == sum(
        p.scoring for _, plays in log.plays("all") for p in plays
    )
    page = render_game(box, log)
    assert 'data-filter="scoring"' in page and 'data-filter="all"' in page
    assert page.count('class="play scoring"') == sum(
        p.scoring for _, plays in log.plays("all") for p in plays
    )
