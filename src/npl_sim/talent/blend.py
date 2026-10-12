"""The weekly talent blend: each player's per-PA ability for one sim week.

Dave's spec ("Realism over replay"): the week's MLB performance is evidence
about how well a player is playing now, not a line to reproduce. Each
player's rates are built in three layers, all as indexes against the league
(1.0 = league average for the same handedness matchup), so they carry over
to today's run environment whatever season they came from:

1. Baseline: a Marcel-style weighting of this season before the sim week and
   the two seasons before it (5/4/3), each PA measured against its own
   season's league, then regressed to average by each stat's stabilization
   point. Platoon splits come from prior seasons, regressed hard toward the
   league's own platoon gap; this season's splits count only for players
   with no MLB split data (league decision, 2026-10-12).
2. Process: the week's fast-stabilizing skill signals against the player's
   own baseline. Hitters: whiff, chase, barrel, xBA on contact, hard-hit.
   Pitchers: whiff, zone, chase, barrels and xBA allowed, fastball velocity,
   and Stuff+ (each pitch type's Stuff+ regressed by how many he's thrown).
   Each is regressed toward the baseline by its own sample, so 4 PA or an
   inning barely moves anything and a full week of starts moves it more.
3. Results: what actually happened (K, BB, HR, hits, batted-ball types)
   beyond what baseline and process already expect, with three times the
   regression of the baseline, since weekly outcomes carry the most luck.

The engine then plays every PA from these rates with park, handedness,
defense and fatigue applied, so good players produce good results on
average, with variance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from npl_sim.engine.game import Hitter, Pitcher
from npl_sim.engine.model import BIP_TYPES, Rates
from npl_sim.talent.env import HANDS, League

PER_PA = ("K", "BB", "HBP", "HR")
STATS = PER_PA + BIP_TYPES + ("HIT", "XBH")
BIP_PER_PA = 0.68
SEASON_WEIGHTS = (5.0, 4.0, 3.0)  # this season to date, last season, two ago

# Stabilization points (Carleton / Pizza Cutter), in PA for K/BB/HBP/HR and
# in balls in play for the rest: the baseline adds this many league-average
# opportunities before trusting a player's own record.
REGRESSION = {
    "batter": {
        "K": 60, "BB": 120, "HBP": 240, "HR": 170,
        "GB": 80, "LD": 600, "FB": 80, "PU": 250, "HIT": 820, "XBH": 600,
    },
    "pitcher": {
        "K": 70, "BB": 170, "HBP": 640, "HR": 1000,
        "GB": 70, "LD": 650, "FB": 70, "PU": 300, "HIT": 2000, "XBH": 2000,
    },
}  # fmt: skip
RESULTS_REGRESSION = 3.0  # results get three times the baseline's regression
# How far a player's platoon split may stray from the league's, in PA added
# at the league gap (Tango, The Book: about 2,200 for RHB, 1,000 for LHB).
PLATOON_REGRESSION = {"batter": {"R": 2200, "L": 1000, "S": 1500}, "pitcher": 1000}
INDEX_RANGE = (0.25, 4.0)

# Process signals: metric -> (regression in its own units, {stat: elasticity}).
# Each moves a stat by (week metric / baseline metric) ** elasticity.
PROCESS = {
    "batter": {
        "whiff": (200, {"K": 1.0}),
        "chase": (250, {"BB": -0.8, "K": 0.3}),
        "barrel": (60, {"HR": 0.6}),
        "xba": (40, {"HIT": 1.0}),
        "hard": (40, {"XBH": 0.5}),
    },
    "pitcher": {
        "whiff": (250, {"K": 1.0}),
        "zone": (300, {"BB": -1.5}),
        "chase": (200, {"BB": -0.5, "K": 0.3}),
        "barrel": (80, {"HR": 0.6}),
        "xba": (80, {"HIT": 0.7}),
    },
}
VELO_REGRESSION = 15  # fastballs
VELO_EFFECT = {"K": 0.03, "HR": -0.04}  # per mph above his baseline
STUFF_EFFECT = {"K": 0.4, "HR": -0.3, "HIT": -0.15}  # elasticity on Stuff+/100


@dataclass
class Talent:
    """One player's blended indexes for the week: `index[opp_hand][stat]`,
    where opp_hand is the pitcher's hand for a batter, the batter's side for
    a pitcher. `layers` keeps each layer's overall multipliers for reporting."""

    id: str
    role: str
    index: dict[str, dict[str, float]]
    layers: dict[str, dict[str, float]] = field(default_factory=dict)
    sample: dict[str, float] = field(default_factory=dict)


@dataclass
class Season:
    """One season of evidence: `offset` 0 is this season (PAs before the sim
    week), 1 last season, 2 the one before. `flags` are `events.pitch_flags`."""

    offset: int
    pa: pd.DataFrame
    league: League
    flags: pd.DataFrame | None = None
    parks: dict | None = None  # (home team, batter side) -> factors, fielding.park_factors


# --- observed and expected counts ------------------------------------------


def annotate(pa: pd.DataFrame, league: League, parks: dict | None = None) -> pd.DataFrame:
    """Adds o_<stat> (what happened) and e_<stat> (what a league-average
    player would expect in the same PA, in that park when `parks` maps
    (home team, batter side) to factors) for every stat, plus opportunity
    columns n_pa and n_bip."""
    out = pa.copy()
    key = list(zip(out["stand"], out["p_throws"], strict=True))
    bip = out["bip"].notna()
    out["n_pa"] = 1.0
    out["n_bip"] = bip.astype(float)
    for s in PER_PA:
        out[f"o_{s}"] = (out["outcome"] == s).astype(float)
        out[f"e_{s}"] = [league.pa.get(k, {}).get(s, 0.0) for k in key]
    for t in BIP_TYPES:
        out[f"o_{t}"] = (out["bip"] == t).astype(float)
        share = pd.Series([league.bip.get(k, {}).get(t, 0.0) for k in key], index=out.index)
        out[f"e_{t}"] = share * out["n_bip"]
    hit = {t: sum(league.hit[t].values()) for t in BIP_TYPES}
    xbh = {t: league.hit[t]["2B"] + league.hit[t]["3B"] for t in BIP_TYPES}
    out["o_HIT"] = (out["outcome"].isin(("1B", "2B", "3B")) & bip).astype(float)
    out["e_HIT"] = out["bip"].map(hit).fillna(0.0)
    out["o_XBH"] = (out["outcome"].isin(("2B", "3B")) & bip).astype(float)
    out["e_XBH"] = out["bip"].map(xbh).fillna(0.0)
    if parks:
        _park_adjust(out, league, parks)
    return out


def _park_adjust(out: pd.DataFrame, league: League, parks: dict) -> None:
    """Scale each PA's expectations by its park, so a hitter's record is
    measured against what an average hitter would do in the same parks."""
    home = out["home_team"] if "home_team" in out else pd.Series(None, index=out.index)
    f = [parks.get((h, s), {}) for h, s in zip(home, out["stand"], strict=True)]
    hr = pd.Series([x.get("hr", 1.0) for x in f], index=out.index)
    one = pd.Series([x.get("1b", 1.0) for x in f], index=out.index)
    two = pd.Series([x.get("2b", 1.0) for x in f], index=out.index)
    three = pd.Series([x.get("3b", 1.0) for x in f], index=out.index)
    out["e_HR"] *= hr
    p1 = out["bip"].map({t: league.hit[t]["1B"] for t in BIP_TYPES}).fillna(0.0) * one
    p2 = out["bip"].map({t: league.hit[t]["2B"] for t in BIP_TYPES}).fillna(0.0) * two
    p3 = out["bip"].map({t: league.hit[t]["3B"] for t in BIP_TYPES}).fillna(0.0) * three
    out["e_HIT"] = p1 + p2 + p3
    out["e_XBH"] = p2 + p3


def _opportunity(stat: str) -> str:
    return "n_pa" if stat in PER_PA else "n_bip"


def _sums(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    cols = ["n_pa", "n_bip"] + [f"{p}_{s}" for s in STATS for p in ("o", "e")]
    weighted = frame[cols].mul(frame["w"], axis=0)
    weighted[by] = frame[by]
    return weighted.groupby(by)[cols].sum()


def _league_rate(frame: pd.DataFrame) -> dict[str, float]:
    """League rate per opportunity of each stat, for the regression's pseudo-counts."""
    rates = {}
    for s in STATS:
        opp = frame[_opportunity(s)].sum()
        rates[s] = frame[f"e_{s}"].sum() / opp if opp else 0.0
    return rates


# --- the blend ---------------------------------------------------------------


def blend(
    seasons: list[Season],
    week: Season,
    stuff: dict[str, float] | None = None,
    players: dict[str, str] | None = None,
) -> dict[str, Talent]:
    """Each player's talent for the week, keyed by MLBAM id (as a string) and
    role: returns {"batter:<id>": Talent, "pitcher:<id>": Talent}.

    `seasons` hold the evidence before the week; `week` the sim week's PAs
    and pitches, with this season's league. `stuff` maps a pitcher id to his
    Stuff+ already regressed by pitch count (`stuff_plus`). `players` maps id
    -> batting hand for platoon regression (R if unknown)."""
    history = []
    for s in seasons:
        if s.pa.empty or s.offset >= len(SEASON_WEIGHTS):
            continue
        a = annotate(s.pa, s.league, s.parks)
        a["w"] = SEASON_WEIGHTS[s.offset]
        a["current"] = s.offset == 0
        history.append(a)
    week_rows = annotate(week.pa, week.league, week.parks) if not week.pa.empty else None
    if week_rows is not None:
        week_rows["w"] = 1.0
    every = pd.concat(history + ([week_rows] if week_rows is not None else []), ignore_index=True)
    if every.empty:
        return {}
    league_rate = _league_rate(every)
    out: dict[str, Talent] = {}
    for role, me, opp_hand in (("batter", "batter", "p_throws"), ("pitcher", "pitcher", "stand")):
        base = _baseline(history, role, me, opp_hand, league_rate, players or {})
        process = _process(seasons, week, role, me, stuff or {})
        results = _results(week_rows, role, me, opp_hand, base, process, league_rate)
        ids = set(base) | set(process) | set(results)
        for pid in ids:
            idx = {}
            b = base.get(pid) or {h: dict.fromkeys(STATS, 1.0) for h in HANDS}
            p = process.get(pid, {})
            r = results.get(pid, {})
            for h in HANDS:
                idx[h] = {s: _clamp(b[h][s] * p.get(s, 1.0) * r.get(s, 1.0)) for s in STATS}
            key = str(pid)
            out[f"{role}:{key}"] = Talent(
                id=key,
                role=role,
                index=idx,
                layers={
                    "baseline": _mean_hands(b),
                    "process": {s: p.get(s, 1.0) for s in STATS},
                    "results": {s: r.get(s, 1.0) for s in STATS},
                },
            )
    return out


def _baseline(history, role, me, opp_hand, league_rate, hands) -> dict:
    """{id: {opp_hand: {stat: index}}} from the weighted seasons before the week."""
    if not history:
        return {}
    data = pd.concat(history, ignore_index=True)
    reg = REGRESSION[role]
    overall = _sums(data, [me])
    index = {}
    for pid, row in overall.iterrows():
        index[pid] = {
            s: (row[f"o_{s}"] + reg[s] * league_rate[s]) / (row[f"e_{s}"] + reg[s] * league_rate[s])
            for s in STATS
        }
    # Platoon: prior seasons, or this season for a player with no split data.
    split = data[~data["current"]]
    prior = _sums(split, [me, opp_hand]) if not split.empty else None
    current = _sums(data[data["current"]], [me, opp_hand]) if data["current"].any() else None
    out = {}
    for pid, idx in index.items():
        per_hand = {}
        for h in HANDS:
            row = _row(prior, (pid, h))
            if row is None or row["n_pa"] == 0:
                row = _row(current, (pid, h))
            if role == "batter":
                pl = PLATOON_REGRESSION["batter"].get(hands.get(str(pid), "R"), 2200)
            else:
                pl = PLATOON_REGRESSION["pitcher"]
            per_hand[h] = {}
            for s in STATS:
                if row is None:
                    per_hand[h][s] = idx[s]
                    continue
                opps = pl * (1.0 if s in PER_PA else BIP_PER_PA)
                k = opps * league_rate[s] * idx[s]
                gap = (row[f"o_{s}"] + k) / (idx[s] * row[f"e_{s}"] + k)
                per_hand[h][s] = idx[s] * gap
        out[pid] = per_hand
    return out


def _row(frame, key):
    if frame is None:
        return None
    try:
        return frame.loc[key]
    except KeyError:
        return None


def _metrics(flags: pd.DataFrame | None, pa: pd.DataFrame, me: str) -> pd.DataFrame:
    """Per player: numerators and denominators of each process metric."""
    cols = {}
    if flags is not None and not flags.empty:
        g = flags.groupby(me)
        cols["whiff_n"], cols["whiff_d"] = g["whiff"].sum(), g["swing"].sum()
        cols["chase_n"], cols["chase_d"] = g["chase"].sum(), g["outside"].sum()
        cols["zone_n"], cols["zone_d"] = g["zone"].sum(), g["zone"].count()
        fast = flags[flags["pitch_type"].isin({"FF", "SI", "FA"}) & flags["velo"].notna()]
        fg = fast.groupby(me)["velo"]
        cols["velo_n"], cols["velo_d"] = fg.sum(), fg.count()
    if not pa.empty:
        g = pa.groupby(me)
        for m in ("barrel", "xba", "hard"):
            cols[f"{m}_n"], cols[f"{m}_d"] = g[m].sum(), g[m].count()
    return pd.DataFrame(cols).fillna(0.0)


def _process(seasons, week, role, me, stuff) -> dict:
    """{id: {stat: multiplier}} from the week's process signals against the
    player's own (regressed) baseline signals."""
    if week.pa.empty and (week.flags is None or week.flags.empty):
        return {}
    now = _metrics(week.flags, week.pa, me)
    past = now * 0
    for s in seasons:
        if s.offset < len(SEASON_WEIGHTS):
            past = past.add(_metrics(s.flags, s.pa, me), fill_value=0.0)
    past = past.reindex(index=now.index.union(past.index), columns=now.columns, fill_value=0.0)
    past = past.fillna(0.0)
    totals = past.sum() + now.sum()
    out = {}
    for pid in now.index:
        mult = dict.fromkeys(STATS, 1.0)
        for metric, (k, effects) in PROCESS[role].items():
            n, d = f"{metric}_n", f"{metric}_d"
            if n not in now or now.at[pid, d] == 0 or totals[d] == 0:
                continue
            league = totals[n] / totals[d]
            b = (past.at[pid, n] + k * league) / (past.at[pid, d] + k)
            w = (now.at[pid, n] + k * b) / (now.at[pid, d] + k)
            if b <= 0 or w <= 0:
                continue
            for stat, e in effects.items():
                mult[stat] *= (w / b) ** e
        if role == "pitcher":
            if "velo_d" in now and now.at[pid, "velo_d"] and past.at[pid, "velo_d"]:
                n_now = now.at[pid, "velo_d"]
                dv = now.at[pid, "velo_n"] / n_now - past.at[pid, "velo_n"] / past.at[pid, "velo_d"]
                dv *= n_now / (n_now + VELO_REGRESSION)
                for stat, e in VELO_EFFECT.items():
                    mult[stat] *= math.exp(e * dv)
            s = stuff.get(str(pid))
            if s:
                for stat, e in STUFF_EFFECT.items():
                    mult[stat] *= (s / 100.0) ** e
        out[pid] = mult
    if role == "pitcher":
        for pid, s in stuff.items():
            key = _id_like(pid, now.index)
            if key not in out and s:
                out[key] = {st: (s / 100.0) ** STUFF_EFFECT.get(st, 0.0) for st in STATS}
    return out


def _id_like(pid: str, index) -> object:
    """A Stuff+ id (string) in the same type as the pitch data's ids."""
    if len(index) and not isinstance(index[0], str):
        try:
            return int(pid)
        except ValueError:
            return pid
    return pid


def _results(week_rows, role, me, opp_hand, base, process, league_rate) -> dict:
    """{id: {stat: multiplier}}: the week's outcomes beyond what the baseline
    and process layers expect, regressed three times as hard as the baseline."""
    if week_rows is None:
        return {}
    reg = REGRESSION[role]
    sums = _sums(week_rows, [me, opp_hand])
    out = {}
    for pid in sums.index.get_level_values(0).unique():
        b = base.get(pid)
        p = process.get(pid, {})
        observed = dict.fromkeys(STATS, 0.0)
        expected = dict.fromkeys(STATS, 0.0)  # at his baseline x process level
        league = dict.fromkeys(STATS, 0.0)  # at league average
        for h in HANDS:
            row = _row(sums, (pid, h))
            if row is None:
                continue
            for s in STATS:
                level = (b[h][s] if b else 1.0) * p.get(s, 1.0)
                observed[s] += row[f"o_{s}"]
                expected[s] += row[f"e_{s}"] * level
                league[s] += row[f"e_{s}"]
        mult = {}
        for s in STATS:
            if league[s] <= 0:
                mult[s] = 1.0
                continue
            # Pseudo-events at his expected level, so the week pulls toward it.
            k = RESULTS_REGRESSION * reg[s] * league_rate[s] * expected[s] / league[s]
            mult[s] = (observed[s] + k) / (expected[s] + k)
        out[pid] = mult
    return out


def _clamp(x: float) -> float:
    lo, hi = INDEX_RANGE
    return min(hi, max(lo, x))


def _mean_hands(b: dict) -> dict[str, float]:
    return {s: sum(b[h][s] for h in HANDS) / len(HANDS) for s in STATS}


# --- Stuff+ --------------------------------------------------------------------

STUFF_REGRESSION = 150  # pitches of a type before his Stuff+ on it counts in full


def stuff_plus(per_pitch: dict[str, float], counts: dict[str, int], overall=None) -> float | None:
    """A pitcher's Stuff+ with each pitch type's value regressed to 100 by how
    many of it he's thrown, weighted by usage (league decision: FanGraphs'
    per-pitch numbers swing wildly on small samples). Falls back to the
    overall Stuff+ regressed by total pitches when there are no per-pitch
    values."""
    used = {
        t: (v, counts.get(t, 0))
        for t, v in per_pitch.items()
        if v is not None and not math.isnan(v) and counts.get(t, 0) > 0
    }
    if used:
        total = sum(n for _, n in used.values())
        return sum(
            n / total * (100 + (v - 100) * n / (n + STUFF_REGRESSION)) for v, n in used.values()
        )
    if overall is None or math.isnan(overall):
        return None
    n = sum(counts.values())
    return 100 + (overall - 100) * n / (n + STUFF_REGRESSION)


# --- cards ---------------------------------------------------------------------


def rates(index: dict[str, float], league: League, key: tuple[str, str]) -> Rates:
    """Per-PA rates for one matchup from a player's indexes and the league."""
    lg = league.pa[key]
    true = {s: index[s] * lg[s] for s in PER_PA}
    total = sum(true.values())
    if total > 0.9:  # keep balls in play possible
        true = {s: v * 0.9 / total for s, v in true.items()}
    mix = {t: index[t] * league.bip[key][t] for t in BIP_TYPES}
    m = sum(mix.values()) or 1.0
    hit = index["HIT"]
    return Rates(
        k=true["K"],
        bb=true["BB"],
        hbp=true["HBP"],
        hr=true["HR"],
        bip={t: v / m for t, v in mix.items()},
        hit=hit,
        power=index["XBH"] / hit if hit else 1.0,
    )


def hitter_card(h: Hitter, talent: Talent | None, league: League) -> Hitter:
    """`h` with his rates from the blend (league average if no talent)."""
    if talent is None:
        talent = Talent(h.id, "batter", {x: dict.fromkeys(STATS, 1.0) for x in HANDS})

    def vs(throws: str) -> Rates:
        return rates(talent.index[throws], league, (h.side_vs(throws), throws))

    return Hitter(
        h.id, h.name, h.bats, vs_lhp=vs("L"), vs_rhp=vs("R"), speed=h.speed, positions=h.positions
    )


def pitcher_card(p: Pitcher, talent: Talent | None, league: League) -> Pitcher:
    if talent is None:
        talent = Talent(p.id, "pitcher", {x: dict.fromkeys(STATS, 1.0) for x in HANDS})

    def vs(side: str) -> Rates:
        return rates(talent.index[side], league, (side, p.throws))

    return Pitcher(
        p.id, p.name, p.throws, vs_lhb=vs("L"), vs_rhb=vs("R"), pitch_limit=p.pitch_limit
    )
