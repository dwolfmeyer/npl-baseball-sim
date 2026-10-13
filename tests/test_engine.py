from collections import Counter
from dataclasses import replace

import pytest

from npl_sim.engine.box import box_score, format_box
from npl_sim.engine.defaults import DEFAULT_ENV, average_split
from npl_sim.engine.game import simulate_game
from npl_sim.engine.model import (
    NEUTRAL_PARK,
    Defense,
    ParkFactors,
    custom_park,
    pa_probabilities,
)
from npl_sim.engine.sample import average_team, hitter

AWAY, HOME = average_team("Away"), average_team("Home")
HITS = ("1B", "2B", "3B")


def probs(bats="R", throws="R", park=NEUTRAL_PARK, defense=None, alignment="normal", batter=None):
    b = batter or average_split(bats, throws)
    return pa_probabilities(
        b,
        average_split(bats, throws),
        DEFAULT_ENV,
        bats,
        throws,
        park,
        defense or Defense(),
        alignment=alignment,
    )


def total(p, suffix):
    return sum(v for k, v in p.items() if k.endswith(suffix))


def test_probabilities_sum_to_one():
    assert sum(probs().values()) == pytest.approx(1.0)


def test_same_seed_same_game():
    assert simulate_game(AWAY, HOME, DEFAULT_ENV, 42) == simulate_game(AWAY, HOME, DEFAULT_ENV, 42)
    assert simulate_game(AWAY, HOME, DEFAULT_ENV, 42) != simulate_game(AWAY, HOME, DEFAULT_ENV, 43)


@pytest.fixture(scope="module")
def season():
    return [simulate_game(AWAY, HOME, DEFAULT_ENV, s) for s in range(800)]


def test_league_average_teams_play_like_mlb(season):
    """Placeholder-env sanity band; Milestone 8 calibrates to a real season."""
    runs = sum(g[-1]["away"] + g[-1]["home"] for g in season) / len(season) / 2
    c = Counter(e["result"] for g in season for e in g if e["type"] == "pa")
    pa = sum(c.values())
    assert 4.0 < runs < 5.0
    assert 0.20 < c["K"] / pa < 0.24
    assert 0.075 < c["BB"] / pa < 0.095
    assert 0.025 < c["HR"] / pa < 0.037


def test_box_score_adds_up(season):
    for events in season[:200]:
        box = box_score(events, "Away", "Home")
        for team, opp in (("Away", "Home"), ("Home", "Away")):
            assert sum(b.r for b in box.batting[team].values()) == box.runs[team]
            assert sum(p.r for p in box.pitching[opp].values()) == box.runs[team]
            assert sum(b.h for b in box.batting[team].values()) == box.hits[team]
            assert sum(box.line[team]) == box.runs[team]
        # The away team always bats full innings.
        assert sum(p.outs for p in box.pitching["Home"].values()) == 3 * len(box.line["Away"])
        assert box.runs["Away"] != box.runs["Home"]


def test_extra_innings_start_with_an_unearned_runner_on_second(season):
    extras = [g for g in season if len(g[-1]["line"]["away"]) > 9]
    assert extras
    for g in extras:
        assert any(e["type"] == "auto_runner" for e in g)
    # Runs scored by automatic runners are unearned.
    auto = {
        (e["inning"], e["half"]): e["runner"]
        for g in extras
        for e in g
        if e["type"] == "auto_runner"
    }
    assert auto


def test_walkoff_ends_the_game_at_once(season):
    for g in season:
        final = g[-1]
        if (
            final["home"] > final["away"]
            and len(final["line"]["home"]) >= 9
            and (len(final["line"]["home"]) == len(final["line"]["away"]))
        ):
            last = [e for e in g if e["type"] == "pa"][-1]
            assert last["half"] == "bottom" and last["scored"]


def test_park_hr_factor_is_applied_once():
    base = probs()["HR"]
    boosted = probs(park=ParkFactors(hr={"L": 1.2, "R": 1.2}))["HR"]
    assert 1.17 < boosted / base <= 1.2


def test_short_right_field_helps_left_handed_power_only():
    short_rf = ParkFactors(hr={"L": 1.25, "R": 1.0})
    assert probs("L", park=short_rf)["HR"] > 1.2 * probs("L")["HR"]
    assert probs("R", park=short_rf)["HR"] == pytest.approx(probs("R")["HR"])


def test_platoon_advantage():
    def on_base(p):
        return total(p, "1B") + total(p, "2B") + total(p, "3B") + p["HR"] + p["BB"] + p["HBP"]

    assert on_base(probs("L", "R")) > on_base(probs("L", "L"))


def test_better_defense_takes_away_hits():
    good = Defense(hit_mult={"GB": 0.95, "LD": 0.97, "FB": 0.95, "PU": 1.0})
    assert sum(total(probs(defense=good), h) for h in HITS) < sum(total(probs(), h) for h in HITS)


def test_infield_in_lets_more_grounders_through():
    assert probs(alignment="infield_in")["GB:1B"] > probs()["GB:1B"]


def test_custom_park_caps():
    custom_park({"L": 1.03, "R": 1.0}, {"L": 1.03, "R": 1.0}, {"L": 1.05, "R": 1.0})
    with pytest.raises(ValueError, match="10%"):
        custom_park({"L": 1.0, "R": 1.0}, {"L": 1.0, "R": 1.0}, {"L": 1.15, "R": 1.0})
    with pytest.raises(ValueError, match="run impact"):
        custom_park({"L": 1.1, "R": 1.0}, {"L": 1.1, "R": 1.0}, {"L": 1.1, "R": 1.0})


def test_better_hitters_score_more():
    strong = average_team("Strong")
    strong.lineup = [
        (
            replace(
                h,
                vs_lhp=replace(h.vs_lhp, hr=h.vs_lhp.hr * 1.4, hit=1.08),
                vs_rhp=replace(h.vs_rhp, hr=h.vs_rhp.hr * 1.4, hit=1.08),
            ),
            pos,
        )
        for h, pos in strong.lineup
    ]
    weak = average_team("Weak")
    games = [simulate_game(weak, strong, DEFAULT_ENV, s) for s in range(300)]

    def per_inning(team):
        return sum(sum(g[-1]["line"][team]) for g in games) / sum(
            len(g[-1]["line"][team]) for g in games
        )

    assert per_inning("home") > 1.15 * per_inning("away")


def test_fast_runners_steal_more():
    fast = average_team("Fast")
    fast.lineup = [(replace(h, speed=1.5), pos) for h, pos in fast.lineup]
    slow = average_team("Slow")
    slow.lineup = [(replace(h, speed=-1.0), pos) for h, pos in slow.lineup]
    sb = Counter()
    for s in range(200):
        for e in simulate_game(fast, slow, DEFAULT_ENV, s):
            if e["type"] == "stolen_base":
                sb[e["half"]] += 1
    assert sb["top"] > 2 * sb["bottom"]


def test_switch_hitter_bats_opposite_the_pitcher():
    s = hitter("S", bats="S")
    assert s.side_vs("R") == "L" and s.side_vs("L") == "R"


def test_format_box_renders():
    text = format_box(box_score(simulate_game(AWAY, HOME, DEFAULT_ENV, 1), "Away", "Home"))
    assert "Away Batters" in text and "Home Pitchers" in text


def test_alignment_rules_pick_the_first_that_fits():
    from npl_sim.engine.alignment import AlignmentRule, choose

    on_third, on_first, empty = [None, None, "r"], ["r", None, None], [None, None, None]
    rules = (AlignmentRule("infield_in", from_inning=7, max_lead=1), AlignmentRule("double_play"))
    assert choose((), 9, 0, on_third, 0) == "normal"  # nothing ticked: straight up
    assert choose(rules, 8, 1, on_third, 1) == "infield_in"
    assert choose(rules, 6, 1, on_third, 1) == "normal"  # too early for infield in
    assert choose(rules, 8, 1, on_third, 3) == "normal"  # lead too big
    assert choose(rules, 8, 2, on_third, 0) == "normal"  # two outs
    assert choose(rules, 3, 0, on_first, 0) == "double_play"
    assert choose(rules, 3, 0, empty, 0) == "normal"
    with pytest.raises(ValueError):
        AlignmentRule("shift")


def test_infield_in_holds_the_runner_on_third():
    from npl_sim.engine.alignment import AlignmentRule

    def scored_on_grounders(rules):
        home = replace(HOME, alignment_rules=rules)
        chances = scored = 0
        for seed in range(400):
            for e in simulate_game(AWAY, home, DEFAULT_ENV, seed):
                if (
                    e["type"] == "pa"
                    and e["half"] == "top"
                    and e["result"] == "GO"
                    and e["bases_before"][2]
                    and e["outs_before"] < 2
                ):
                    chances += 1
                    scored += bool(e["scored"])
        return scored / chances

    straight = scored_on_grounders(())
    infield_in = scored_on_grounders((AlignmentRule("infield_in"),))
    assert infield_in < straight - 0.15


def test_no_rules_plays_straight_up():
    events = simulate_game(AWAY, HOME, DEFAULT_ENV, 5)
    assert not any("alignment" in e for e in events if e["type"] == "pa")


class _NoChances:
    """An rng whose chance events never happen: no double play, no fielder's
    choice, no optional extra base."""

    def random(self) -> float:
        return 0.999


@pytest.mark.parametrize(
    "on, after, runs",
    [
        ((1, 1, 1), (0, 1, 1), 1),  # bases loaded: everyone is forced
        ((1, 1, 0), (0, 1, 1), 0),
        ((1, 0, 1), (0, 1, 1), 0),  # the runner on third isn't forced
        ((0, 1, 0), (0, 1, 0), 0),
        ((0, 0, 1), (0, 0, 1), 0),
    ],
)
def test_groundout_at_first_moves_forced_runners(on, after, runs):
    from npl_sim.engine.game import GameSim, Runner

    sim = GameSim(AWAY, HOME, DEFAULT_ENV, 1)
    sim.rng = _NoChances()
    batter, *runners = (Runner(h, "p") for h, _ in AWAY.lineup[:4])
    sim.bases = [r if o else None for r, o in zip(runners, on, strict=True)]
    scored: list = []
    assert sim._ground_out(batter, scored) == "GO"
    assert sim.outs == 1 and len(scored) == runs
    assert tuple(int(b is not None) for b in sim.bases) == after
