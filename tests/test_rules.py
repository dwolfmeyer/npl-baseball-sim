"""MLB playing rules the engine follows (engine/game.py docstring)."""

from collections import Counter
from dataclasses import replace

import pytest

from npl_sim.engine.box import box_score
from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.game import GameSim, Runner, simulate_game
from npl_sim.engine.manager import NO_BUNTING
from npl_sim.engine.playlog import format_log, game_log
from npl_sim.engine.sample import average_team, hitter

AWAY, HOME = average_team("Away"), average_team("Home")
WEAK = {"k": 0.32, "bb": 0.05, "hr": 0.008}  # a light-hitting bat who bunts


@pytest.fixture(scope="module")
def games():
    return [simulate_game(AWAY, HOME, DEFAULT_ENV, s) for s in range(600)]


class _Always:
    """An rng whose chance events always (0.0) or never (0.999) happen."""

    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


def _sim(bases, inning=1, half="top", value=0.999):
    sim = GameSim(AWAY, HOME, DEFAULT_ENV, 1)
    sim.rng = _Always(value)
    sim.inning, sim.half = inning, half
    offense = AWAY if half == "top" else HOME
    batter, *runners = (Runner(h, "p") for h, _ in offense.lineup[:4])
    sim.bases = [r if on else None for r, on in zip(runners, bases, strict=True)]
    return sim, batter


def test_three_batter_minimum(games):
    for g in games:
        faced: Counter = Counter()
        since_half = False  # a PA already in this half inning
        for e in g:
            if e["type"] == "pa":
                faced[e["pitcher"]] += 1
                since_half = True
            elif e["type"] == "half_end":
                since_half = False
            elif e["type"] == "pitching_change" and since_half:
                assert faced[e["out"]] >= 3, "pulled mid-inning before facing three"


def test_mid_inning_changes_still_happen(games):
    assert sum(
        e["type"] == "pitching_change" and (e["outs"] or e["on_base"]) for g in games for e in g
    )


def test_walkoff_counts_only_the_winning_run():
    sim, batter = _sim((1, 0, 1), inning=9, half="bottom")
    before = list(sim.bases)
    scored: list = []
    assert sim._resolve("LD:2B", batter.hitter, HOME.starter, scored) == "2B"
    assert len(scored) == 1 and sim.sides["home"].runs == 1  # the runner from first doesn't count
    # The winner came from third, so the double goes in the book as a single.
    assert sim._walkoff_hit("2B", batter.hitter, before, scored) == "1B"
    assert sim.bases[0] is not None and sim.bases[0].hitter.id == batter.hitter.id


def test_walkoff_home_run_counts_every_run():
    sim, batter = _sim((1, 1, 1), inning=10, half="bottom")
    sim._homer = False
    scored: list = []
    assert sim._resolve("HR", batter.hitter, HOME.starter, scored) == "HR"
    assert sim.sides["home"].runs == 4


def test_ground_rule_double_gives_runners_two_bases():
    sim, batter = _sim((1, 0, 0), value=0.0)
    scored: list = []
    assert sim._resolve("FB:2B", batter.hitter, HOME.starter, scored) == "2B"
    assert sim._ground_rule and not scored
    assert [r is not None for r in sim.bases] == [False, True, True]


def test_ground_rule_doubles_show_in_the_play_by_play(games):
    doubles = [e for g in games for e in g if e["type"] == "pa" and e["result"] == "2B"]
    share = sum(bool(e.get("ground_rule")) for e in doubles) / len(doubles)
    assert 0.02 < share < 0.10
    g = next(g for g in games if any(e.get("ground_rule") for e in g))
    assert "ground-rule double" in format_log(game_log(g))


@pytest.mark.parametrize(
    "res, bases, after, outs",
    [
        ("SH", (1, 0, 0), (0, 1, 0), 1),
        ("SH", (1, 1, 0), (0, 1, 1), 1),
        ("1B", (1, 0, 0), (1, 1, 0), 0),
        ("FC", (1, 1, 0), (1, 1, 0), 1),  # runner from second out at third
        ("FC", (1, 0, 0), (1, 0, 0), 1),
        ("DP", (1, 0, 0), (0, 0, 0), 2),
        ("PO", (0, 1, 0), (0, 1, 0), 1),
    ],
)
def test_bunt_outcomes(res, bases, after, outs):
    sim, batter = _sim(bases)
    assert sim._resolve(f"BUNT:{res}", batter.hitter, HOME.starter, []) == res
    assert sim.outs == outs
    assert tuple(int(b is not None) for b in sim.bases) == after


def weak_team(name: str, bunt=None):
    team = average_team(name)
    lineup = [(hitter(h.id, h.bats, scale=WEAK), pos) for h, pos in team.lineup]
    return replace(team, lineup=lineup, **({"bunt": bunt} if bunt else {}))


def test_weak_hitters_sacrifice_late_in_close_games():
    away = weak_team("Away")
    bunts, sh = [], 0
    for seed in range(300):
        g = simulate_game(away, HOME, DEFAULT_ENV, seed)
        bunts += [e for e in g if e["type"] == "pa" and e.get("bunt")]
        box = box_score(g)
        for line in box.batting["Away"].values():
            sh += line.sh
            assert line.ab == line.pa - line.bb - line.hbp - line.sf - line.sh
    assert bunts and sh == sum(e["result"] == "SH" for e in bunts)
    for e in bunts:
        assert e["inning"] >= 7 and e["outs_before"] == 0 and e["bases_before"][2] is None
        assert e["result"] in ("SH", "1B", "FC", "PO", "DP", "K")
    log = "\n".join(format_log(game_log(simulate_game(away, HOME, DEFAULT_ENV, s))) for s in [0])
    assert log  # renders


def test_average_hitters_and_owners_who_say_no_never_bunt(games):
    assert not any(e.get("bunt") for g in games for e in g)
    away = weak_team("Away", bunt=NO_BUNTING)
    for seed in range(100):
        assert not any(e.get("bunt") for e in simulate_game(away, HOME, DEFAULT_ENV, seed))


def test_wild_pitches_passed_balls_and_balks(games):
    kinds = Counter(e["type"] for g in games for e in g)
    team_games = 2 * len(games)
    assert 0.2 < kinds["wild_pitch"] / team_games < 0.5
    assert 0.01 < kinds["passed_ball"] / team_games < 0.12
    assert 0 < kinds["balk"] / team_games < 0.08
    for g in games:
        for e in g:
            if e["type"] == "passed_ball":
                assert not any(s["earned"] for s in e["scored"])
            if e["type"] in ("wild_pitch", "passed_ball", "balk"):
                assert any(e["bases_before"])
    g = next(g for g in games if any(e["type"] == "wild_pitch" for e in g))
    box = box_score(g)  # runs add up, with the WP note
    assert any(k == "WP" for k, _ in box.game_notes)
    assert "Wild pitch by pitcher" in format_log(game_log(g))
