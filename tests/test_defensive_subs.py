from dataclasses import replace

import pytest

from npl_sim.engine.box import box_score
from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.game import DefensiveSub, simulate_game
from npl_sim.engine.playlog import game_log
from npl_sim.engine.sample import average_team, hitter

AWAY = average_team("Away")
GLOVE = hitter("Glove", bats="R")
GLOVE = replace(GLOVE, name="Gold Glove")


def subs_in(events, team="Home"):
    return [e for e in events if e["type"] == "defensive_sub" and e["team"] == team]


def home_with(*subs):
    return replace(average_team("Home"), defensive_subs=subs)


def test_sub_comes_in_late_only_when_leading():
    home = home_with(DefensiveSub(GLOVE, "LF"))
    for seed in range(60):
        events = simulate_game(AWAY, home, DEFAULT_ENV, seed)
        subs = subs_in(events)
        assert len(subs) <= 1
        for e in subs:
            assert e["inning"] >= 9 and e["half"] == "top"  # home takes the field in the top
            assert e["out"] == "Home-LF" and e["slot"] == 5 and e["moved"] is None
        line = events[-1]["line"]
        if sum(line["home"][:8]) <= sum(line["away"][:8]):  # not leading going to the 9th
            assert not any(e["inning"] == 9 for e in subs)


def test_sub_for_a_named_starter_moves_the_player_at_that_position():
    # Glove plays CF for the LF starter; the CF starter slides to LF.
    sub = DefensiveSub(GLOVE, "CF", from_inning=1, only_when_leading=False, replaces="Home-LF")
    home = home_with(sub)
    events = simulate_game(AWAY, home, DEFAULT_ENV, 7)
    (e,) = subs_in(events)
    assert e["inning"] == 1 and e["out"] == "Home-LF"
    assert e["moved"] == {"player": "Home-CF", "position": "LF"}
    batters = [x["batter"] for x in events if x["type"] == "pa" and x["team"] == "Home"]
    assert "Gold Glove" not in batters and "Glove" in batters and "Home-LF" not in batters
    assert home.lineup[5][0].id == "Home-LF"  # the card itself is untouched

    box = box_score(events)
    order = list(box.batting["Home"])
    assert order[order.index("Home-LF") + 1] == "Glove"
    assert box.batting["Home"]["Glove"].substitute
    assert box.batting["Home"]["Glove"].position == "CF"
    assert box.batting["Home"]["Home-CF"].position == "CF-LF"

    log = game_log(events)
    (play,) = [p for _, ps in log.plays() for p in ps if p.kind == "defensive_sub"]
    assert play.description == (
        "Defensive Substitution: Gold Glove replaces left fielder Home-LF, batting 6th, "
        "playing center field. Defensive switch from center field to left field for Home-CF."
    )
    # After the switch, plays to center field name the sub.
    after = [p for _, ps in log.plays() for p in ps][
        [p.kind for _, ps in log.plays() for p in ps].index("defensive_sub") :
    ]
    assert not any("center fielder Home-CF" in p.description for p in after)
    assert any("center fielder Gold Glove" in p.description for p in after)


def test_each_sub_is_used_once_and_a_removed_starter_is_skipped():
    subs = (
        DefensiveSub(GLOVE, "LF", from_inning=1, only_when_leading=False),
        DefensiveSub(
            hitter("Spare"), "RF", from_inning=1, only_when_leading=False, replaces="Home-LF"
        ),
    )
    events = simulate_game(AWAY, home_with(*subs), DEFAULT_ENV, 3)
    used = subs_in(events)
    assert [e["into"] for e in used] == ["Glove"]  # Home-LF already left, so Spare stays put


def test_sub_needs_a_field_position():
    with pytest.raises(ValueError):
        DefensiveSub(GLOVE, "DH")
