"""Base stealing from real MLB records (Dave, 2026-10-13).

- Runners: from the statsapi box-score batting lines (sb, cs) over this
  season to date and the two before, each runner's attempts per time he
  reached first (singles, walks, hit by pitch) and the share he made. Both
  are regressed toward the league's rate for his sprint speed, so a burner
  with ten games of data still runs like a burner and a slow catcher with
  one lucky steal doesn't.
- The league: the same two rates over this season's box scores, so the sim
  steals as often, and as successfully, as MLB does today.
- Catchers: pop time to second (Statcast) and his caught-stealing runs
  (DRS rSB, FanGraphs) over this season and the three before, each turned
  into arm units and averaged (talent/fielding.arms reads pop time alone
  when that's all there is).
"""

from __future__ import annotations

import math

import pandas as pd

from npl_sim.engine.game import SPEED_ATTEMPT, SPEED_SUCCESS
from npl_sim.talent.fielding import ARM_RANGE, POP_AVERAGE, POP_PER_ARM, _innings, _num

SEASON_WEIGHT = {0: 1.0, 1: 0.7, 2: 0.4}  # this season, last, two ago
ATTEMPT_REGRESSION = 60.0  # times on first at the speed-based rate
SUCCESS_REGRESSION = 15.0  # attempts at the speed-based rate
MIN_LEAGUE_ATTEMPTS = 300  # this season's attempts before its rates replace the defaults
POP_REGRESSION = 15.0  # throws to second at average pop time
RSB_PER_ARM = 3.0  # caught-stealing runs per 1350 innings for one arm unit
RSB_REGRESSION = 500.0  # innings
INNINGS_PER_SEASON = 1350.0


def times_on_first(batting: pd.DataFrame) -> pd.Series:
    """Times each line reached first: singles, walks (with intentional) and HBP."""
    singles = batting["h"] - batting["doubles"] - batting["triples"] - batting["hr"]
    return singles + batting["bb"] + batting["hbp"]


def _totals(batting: pd.DataFrame) -> pd.DataFrame:
    need = {"mlbam_id", "h", "doubles", "triples", "hr", "bb", "hbp", "sb", "cs"}
    if batting is None or batting.empty or not need <= set(batting.columns):
        return pd.DataFrame(columns=["opp", "sb", "cs"])
    frame = batting.assign(opp=times_on_first(batting))
    frame = frame.dropna(subset=["mlbam_id"])
    frame["mlbam_id"] = frame["mlbam_id"].astype(int).astype(str)
    return frame.groupby("mlbam_id")[["opp", "sb", "cs"]].sum()


def league_rates(batting: pd.DataFrame) -> tuple[float, float] | None:
    """(attempts per time on first, success rate) across the league's box
    scores, or None until there are enough attempts to trust them."""
    t = _totals(batting)
    attempts = float(t["sb"].sum() + t["cs"].sum()) if not t.empty else 0.0
    opp = float(t["opp"].sum()) if not t.empty else 0.0
    if attempts < MIN_LEAGUE_ATTEMPTS or opp <= 0:
        return None
    return attempts / opp, float(t["sb"].sum()) / attempts


def steal_ratings(
    seasons: list[tuple[int, pd.DataFrame]],
    speed: dict[str, float],
    league: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    """{id: (attempts per time on first, success rate)} from [(seasons back,
    batting lines)], regressed toward the league's rates for his speed."""
    opp: dict[str, float] = {}
    sb: dict[str, float] = {}
    cs: dict[str, float] = {}
    for back, batting in seasons:
        w = SEASON_WEIGHT.get(back)
        if w is None:
            continue
        for pid, r in _totals(batting).iterrows():
            opp[pid] = opp.get(pid, 0.0) + w * float(r["opp"])
            sb[pid] = sb.get(pid, 0.0) + w * float(r["sb"])
            cs[pid] = cs.get(pid, 0.0) + w * float(r["cs"])
    out = {}
    for pid, n in opp.items():
        z = speed.get(pid, 0.0)
        prior_att = league[0] * math.exp(SPEED_ATTEMPT * z - SPEED_ATTEMPT**2 / 2)
        logit = math.log(league[1] / (1 - league[1])) + SPEED_SUCCESS * z
        prior_ok = 1 / (1 + math.exp(-logit))
        tries = sb[pid] + cs[pid]
        attempt = (tries + ATTEMPT_REGRESSION * prior_att) / (n + ATTEMPT_REGRESSION)
        success = (sb[pid] + SUCCESS_REGRESSION * prior_ok) / (tries + SUCCESS_REGRESSION)
        out[pid] = (round(attempt, 4), round(success, 4))
    return out


def catcher_arms(
    seasons: list[tuple[int, pd.DataFrame | None, pd.DataFrame | None]],
) -> dict[str, float]:
    """{catcher id: arm units} from [(seasons back, Statcast pop time board,
    FanGraphs fielding board)]: pop time to second weighted by throws and
    caught-stealing runs (rSB) weighted by innings at catcher, each regressed,
    then averaged when he has both."""
    pop_sum: dict[str, float] = {}
    throws: dict[str, float] = {}
    rsb: dict[str, float] = {}
    innings: dict[str, float] = {}
    for _back, pop, fg in seasons:
        if pop is not None and {"mlbam_id", "pop_2b_sba"} <= set(pop.columns):
            counts = pop["pop_2b_sba_count"] if "pop_2b_sba_count" in pop else 10
            for pid, sec, n in zip(
                pop["mlbam_id"], pop["pop_2b_sba"], pd.Series(counts, index=pop.index), strict=True
            ):
                sec, n = _num(sec), _num(n) or 0.0
                if pd.notna(pid) and sec is not None and n > 0:
                    key = str(int(pid))
                    pop_sum[key] = pop_sum.get(key, 0.0) + sec * n
                    throws[key] = throws.get(key, 0.0) + n
        if fg is not None and not fg.empty and "mlbam_id" in fg:
            cols = {str(c).lower(): c for c in fg.columns}
            pos, inn, runs = cols.get("pos"), cols.get("inn"), cols.get("rsb")
            if pos is None or inn is None or runs is None:
                continue
            for _, r in fg[fg[pos] == "C"].iterrows():
                i, v = _innings(r[inn]), _num(r[runs])
                if pd.notna(r["mlbam_id"]) and i and v is not None:
                    key = str(int(r["mlbam_id"]))
                    rsb[key] = rsb.get(key, 0.0) + v
                    innings[key] = innings.get(key, 0.0) + i
    out = {}
    for key in set(throws) | set(innings):
        parts = []
        if throws.get(key):
            n = throws[key]
            sec = (pop_sum[key] + POP_REGRESSION * POP_AVERAGE) / (n + POP_REGRESSION)
            parts.append((POP_AVERAGE - sec) / POP_PER_ARM)
        if innings.get(key):
            i = innings[key]
            per_season = rsb[key] * INNINGS_PER_SEASON / (i + RSB_REGRESSION)
            parts.append(per_season / RSB_PER_ARM)
        out[key] = max(-ARM_RANGE, min(ARM_RANGE, sum(parts) / len(parts)))
    return out
