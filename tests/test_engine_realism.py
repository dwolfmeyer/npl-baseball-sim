"""Engine realism fixes from the 2026-10-14 engine review (one section per item)."""

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from npl_sim.calibration import harness as h
from npl_sim.engine.defaults import DEFAULT_ENV, average_split
from npl_sim.engine.game import GameSim, simulate_game
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
        assert ratio == pytest.approx(1.0, abs=0.04), (stat, ratio)


def test_average_teams_reproduce_the_input_rates_with_fatigue_on():
    for stat, ratio in rates_vs_input(h.average_games(1000)).items():
        assert ratio == pytest.approx(1.0, abs=0.04), (stat, ratio)


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


# --- item 7: fatigue grows with every batter faced, not only with late pitches -----------


def _side(batters, pitches, starter=True):
    pitcher = SimpleNamespace(id="p1", pitch_limit=100)
    card = SimpleNamespace(starter=SimpleNamespace(id="p1" if starter else "other"), follow=())
    return SimpleNamespace(pitcher=pitcher, batters_faced=batters, pitches=pitches, card=card)


def test_workload_rises_from_the_first_batter_and_not_just_late_in_the_count():
    me = SimpleNamespace(env=DEFAULT_ENV)
    f = [GameSim._fatigue(me, _side(bf, bf * 4)) for bf in (0, 9, 18, 27)]
    assert f[0] < f[1] < f[2] < f[3]  # pitches are far below the limit all the way
    assert GameSim._fatigue(me, _side(0, 0)) == 0.0


def test_a_starter_and_a_reliever_at_their_role_typical_workload_get_the_league_rates():
    b = average_split("R", "R")
    me = SimpleNamespace(env=DEFAULT_ENV)
    for starter in (True, False):
        typical = GameSim._typical(me, _side(0, 0, starter))
        a = pa_probabilities(b, b, DEFAULT_ENV, "R", "R", NEUTRAL_PARK, Defense(), typical, typical=typical)
        c = pa_probabilities(b, b, DEFAULT_ENV, "R", "R", NEUTRAL_PARK, Defense())
        for k in c:
            assert a[k] == pytest.approx(c[k], rel=1e-9)


def test_third_time_through_costs_strikeouts_and_adds_hits_and_homers():
    b = average_split("R", "R")

    def at(f):
        p = pa_probabilities(b, b, DEFAULT_ENV, "R", "R", NEUTRAL_PARK, Defense(), f, typical=0.6)
        hits = sum(v for k, v in p.items() if k.endswith(("1B", "2B", "3B")))
        return p["K"], p["HR"], hits

    first, third = at(0.2), at(1.0)
    assert third[0] < first[0] and third[1] > first[1] and third[2] > first[2]


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


# Earned runs (item 11, rule 9.16): rebuild the inning as if the errors were outs.
def _scoring_sim(outs, phantom):
    from npl_sim.engine.game import GameSim, Runner

    from .test_engine import AWAY, HOME

    sim = GameSim(AWAY, HOME, DEFAULT_ENV, 1)
    sim.outs = outs
    sim._phantom_outs = phantom
    scored: list = []
    sim._score(Runner(AWAY.lineup[0][0], "p"), scored)
    return scored[0]["earned"]


def test_a_run_after_the_error_out_would_have_ended_the_inning_is_unearned():
    assert _scoring_sim(outs=2, phantom=0) is True
    assert _scoring_sim(outs=1, phantom=1) is True  # error out + 1 real = 2 outs, still alive
    assert _scoring_sim(outs=1, phantom=2) is False  # two error outs + 1 real: 3 outs
    assert _scoring_sim(outs=0, phantom=1) is True  # only 1 out so far: inning still alive
    assert _scoring_sim(outs=2, phantom=1) is False


def test_phantom_outs_reset_each_half_inning_and_runs_stay_unearned_once_flagged():
    from npl_sim.engine.game import GameSim

    from .test_engine import AWAY, HOME

    for seed in range(100):
        events = GameSim(AWAY, HOME, DEFAULT_ENV, seed).play()
        errors = 0
        key = None
        for e in events:
            if e["type"] == "half_end":
                errors, key = 0, None
            if e["type"] != "pa":
                continue
            k = (e["inning"], e["half"])
            if k != key:
                errors, key = 0, k
            if e["result"] == "E":
                errors += 1
            for s in e["scored"]:
                # outs before the play + errors so far this half (a prior E) >= 3 means unearned
                if e["outs_before"] + e["outs"] + errors >= 3 and errors:
                    assert not s["earned"], (seed, e)


# Errors (item 10). Retrosheet 2023 to 2025, per team-game: 0.221 reach-on-error, 0.114 errors on
# hits (a throw lets the batter or a runner take a base), 0.079 on ground-ball outs with a
# runner on; 0.52 in all, of which steal and pickoff errors (0.065) and fly-ball errors are
# not modeled.
def _error_games():
    from npl_sim.engine.game import simulate_game

    from .test_engine import AWAY, HOME

    return [simulate_game(AWAY, HOME, DEFAULT_ENV, s) for s in range(1000)]


def test_errors_per_team_game_match_retrosheet_by_kind():
    games = _error_games()
    reach = on_hit = on_out = 0
    for events in games:
        thrown: set = set()
        for e in events:
            if e["type"] == "half_end":
                thrown = set()
            if e["type"] != "pa":
                continue
            reach += e["result"] == "E"
            if e.get("throwing_error"):
                if e["result"] in ("1B", "2B"):
                    on_hit += 1
                    thrown.add(e["batter"])
                    mv = next(m for m in e["runners"] if m["runner"] == e["batter"])
                    assert mv["end"] in ("2B", "3B", "score")
                else:
                    assert e["result"] in ("GO", "FC")
                    on_out += 1
            for s in e["scored"]:
                if s["runner"] in thrown:
                    assert not s["earned"]
    tg = len(games) * 2
    assert reach / tg == pytest.approx(0.221, rel=0.12)
    assert on_hit / tg == pytest.approx(0.114, rel=0.15)
    assert on_out / tg == pytest.approx(0.079, rel=0.20)


# Double plays (item 9): Retrosheet 2023 to 2025, ground balls that aren't hits or errors with a
# runner on first and fewer than two outs: 47.5% DP (RHB 52.8%, LHB 39.7%), 32% force out, 21% out.
def test_ground_balls_with_a_runner_on_first_turn_into_double_plays_like_mlb():
    from collections import Counter

    from npl_sim.engine.game import simulate_game

    from .test_engine import AWAY, HOME

    bats = {h.id: h.bats for team in (AWAY, HOME) for h, _ in team.lineup}
    shares = {"R": Counter(), "L": Counter()}
    for seed in range(1500):
        for e in simulate_game(AWAY, HOME, DEFAULT_ENV, seed):
            if (
                e["type"] == "pa"
                and e["bip"] == "GB"
                and not e.get("bunt")
                and e["result"] in ("DP", "FC", "GO")
                and e["bases_before"][0]
                and not e["bases_before"][2]
                and e["outs_before"] < 2
                and bats.get(e["batter"]) in shares
            ):
                shares[bats[e["batter"]]][e["result"]] += 1
    both = shares["R"] + shares["L"]
    n = sum(both.values())
    assert both["DP"] / n == pytest.approx(0.48, abs=0.03)
    assert both["FC"] / n == pytest.approx(0.32, abs=0.03)
    right = shares["R"]["DP"] / sum(shares["R"].values())
    left = shares["L"]["DP"] / sum(shares["L"].values())
    # About 1,300 left-handed chances, so the gap wanders +-0.03 with the seed (0.07 to 0.12).
    assert right - left == pytest.approx(0.13, abs=0.07)


# Advancement (item 8): Retrosheet 2023 to 2025. A runner on second scores on 60.9% of singles
# (40% with none out, 53% with one, 79% with two); a runner on first, second open, takes third
# or home on 31.7%; a runner on first scores on 39.3% of doubles. Where the ball went matters
# (to right a runner on first reaches third far more often than to left).
def test_runners_advance_on_hits_like_mlb():
    from collections import Counter

    from npl_sim.engine.game import simulate_game

    from .test_engine import AWAY, HOME

    seen: Counter = Counter()
    for seed in range(1500):
        for e in simulate_game(AWAY, HOME, DEFAULT_ENV, seed):
            if e["type"] != "pa" or e["result"] not in ("1B", "2B"):
                continue
            for m in e["runners"]:
                if e["result"] == "1B" and m["start"] == "2B":
                    seen["s2", e["outs_before"]] += 1
                    seen["s2+", e["outs_before"]] += m["end"] == "score"
                if e["result"] == "1B" and m["start"] == "1B" and e["bases_before"][1] is None:
                    seen["s1"] += 1
                    seen["s1+"] += m["end"] in ("3B", "score")
                if e["result"] == "2B" and m["start"] == "1B":
                    seen["d1"] += 1
                    seen["d1+"] += m["end"] == "score"
    for outs, real in ((0, 0.40), (1, 0.53), (2, 0.79)):
        assert seen["s2+", outs] / seen["s2", outs] == pytest.approx(real, abs=0.04)
    assert seen["s1+"] / seen["s1"] == pytest.approx(0.317, abs=0.03)
    assert seen["d1+"] / seen["d1"] == pytest.approx(0.393, abs=0.05)


def test_hits_to_the_outfield_name_the_fielder_the_baserunning_used():
    from npl_sim.engine.game import simulate_game

    from .test_engine import AWAY, HOME

    fields = set()
    for seed in range(60):
        for e in simulate_game(AWAY, HOME, DEFAULT_ENV, seed):
            if e["type"] == "pa" and e["result"] in ("1B", "2B") and e["bip"] in ("LD", "FB"):
                fields.add(e["fielder"])
    assert {"LF", "CF", "RF"} <= fields


# Steal timing (item 12): Retrosheet 2023 to 2025, runner on first with second open. Attempt rate
# relative to the average: tie game late 1.57, down 4+ late 0.24, up 6+ late 0.12; 0.83 with none
# out, 1.12 with two out. Attempts per team-game still match the league (0.91 vs 0.91).
def _timing(inning, lead, outs):
    from npl_sim.engine.game import GameSim

    from .test_engine import AWAY, HOME

    sim = GameSim(AWAY, HOME, DEFAULT_ENV, 1)
    sim.inning, sim.outs = inning, outs
    sim.sides["away"].runs, sim.sides["home"].runs = lead, 0
    return sim._steal_timing(sim.sides["away"])


def test_steal_timing_follows_the_score_the_inning_and_the_outs():
    assert _timing(8, 0, 1) == pytest.approx(1.57)
    assert _timing(8, -4, 1) == pytest.approx(0.24)
    assert _timing(8, 7, 1) == pytest.approx(0.12)
    assert _timing(3, -4, 1) == pytest.approx(0.44)
    assert _timing(3, 0, 0) == pytest.approx(1.11 * 0.83)
    assert _timing(3, 0, 2) > _timing(3, 0, 0)


def test_average_teams_still_attempt_steals_at_the_league_rate_with_timing():
    from npl_sim.engine.game import simulate_game

    from .test_engine import AWAY, HOME

    attempts = games = 0
    for seed in range(800):
        games += 1
        attempts += sum(
            e["type"] in ("stolen_base", "caught_stealing")
            for e in simulate_game(AWAY, HOME, DEFAULT_ENV, seed)
        )
    assert attempts / games / 2 == pytest.approx(0.91, rel=0.07)
