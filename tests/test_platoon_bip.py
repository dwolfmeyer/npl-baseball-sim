"""Item 6: the league's platoon edge on balls in play (Retrosheet 2023 to 2025: a right-handed
batter facing a lefty has a BABIP .2975 against .2909 facing a righty, a left-handed batter
about .289 either way) is in the league environment once, not twice."""

import pandas as pd
import pytest

from npl_sim.engine.model import NEUTRAL_PARK, Defense, pa_probabilities
from npl_sim.talent import blend
from npl_sim.talent.env import league

MIX = {"GB": 0.44, "LD": 0.21, "FB": 0.25, "PU": 0.10}
HIT = {"GB": 0.257, "LD": 0.662, "FB": 0.132, "PU": 0.015}  # hit share by type


def season(edge: dict[tuple[str, str], float]) -> pd.DataFrame:
    """Balls in play for each matchup: the league hit rates times that matchup's edge."""
    rows = []
    for (bats, throws), m in edge.items():
        for outcome, n in (("K", 2200), ("BB", 800), ("HBP", 100), ("HR", 300)):
            rows += [{"stand": bats, "p_throws": throws, "outcome": outcome, "bip": None}] * n
        for bip, share in MIX.items():
            n = int(share * 4000)
            hits = round(n * HIT[bip] * m)
            outcomes = ["1B"] * round(hits * 0.8) + ["2B"] * round(hits * 0.2)
            outcomes += ["GO"] * (n - len(outcomes))
            rows += [
                {"stand": bats, "p_throws": throws, "outcome": o, "bip": bip} for o in outcomes
            ]
    return pd.DataFrame(rows)


EDGE = {("R", "R"): 1.0, ("R", "L"): 1.03, ("L", "R"): 0.99, ("L", "L"): 0.99}


def test_league_measures_each_matchups_hit_level():
    lg = league(season(EDGE))
    hit_r_vs_l = lg.index[("R", "L")][0]
    hit_r_vs_r = lg.index[("R", "R")][0]
    assert hit_r_vs_l / hit_r_vs_r == pytest.approx(1.03, abs=0.01)
    assert lg.env().split("R", "L").hit == pytest.approx(hit_r_vs_l)


def test_an_average_matchup_has_the_matchups_babip_not_its_square():
    lg = league(season(EDGE))
    env = lg.env()
    avg = {x: dict.fromkeys(blend.STATS, 1.0) for x in ("L", "R")}
    batter = blend.rates(avg["L"], lg, ("R", "L"))  # right-handed batter vs a lefty
    pitcher = blend.rates(avg["R"], lg, ("R", "L"))
    p = pa_probabilities(batter, pitcher, env, "R", "L", NEUTRAL_PARK, Defense())

    def babip(probs):
        hits = sum(v for k, v in probs.items() if k[-2:] in ("1B", "2B", "3B") and ":" in k)
        bip = sum(v for k, v in probs.items() if ":" in k)
        return hits / bip

    sub = season({("R", "L"): EDGE[("R", "L")]})
    seen = sub[sub["bip"].notna()]["outcome"].isin(("1B", "2B", "3B")).mean()
    assert babip(p) == pytest.approx(seen, rel=0.01)


def test_a_players_platoon_gap_is_measured_against_the_matchups_league():
    lg = league(season(EDGE))
    data = season(EDGE).assign(batter=1, pitcher=2, home_team="X", outcome_pa=1)
    annotated = blend.annotate(data, lg)
    rhb_vs_l = annotated[(annotated["stand"] == "R") & (annotated["p_throws"] == "L")]
    # An average hitter in an average matchup seen exactly as often as the league: index 1.
    assert rhb_vs_l["o_HIT"].sum() / rhb_vs_l["e_HIT"].sum() == pytest.approx(1.0, abs=0.01)
