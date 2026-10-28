"""The league run environment from real plate appearances.

`league_env` turns a season's PAs into the engine's `LeagueEnv`: K, BB, HBP
and HR per PA and the batted-ball mix for each batter-hand / pitcher-hand
matchup, and the chance a ball in play of each type is a 1B, 2B or 3B. The
talent blend measures every player against the same numbers, so a player's
card is "this much better than the league" and the league itself is today's.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.model import BIP_TYPES, LeagueEnv, Rates

HANDS = ("L", "R")
PER_PA = ("K", "BB", "HBP", "HR")
HITS = ("1B", "2B", "3B")
# A season's environment leans on last season's until this many PAs are in.
FULL_SEASON_PA = 60_000


@dataclass(frozen=True)
class League:
    """League rates the blend measures players against: `pa[(bats, throws)]`
    maps K/BB/HBP/HR to a rate per PA, `bip[(bats, throws)]` maps GB/LD/FB/PU
    to a share of balls in play, and `hit[type]` maps 1B/2B/3B to the chance a
    ball in play of that type goes for one."""

    pa: dict[tuple[str, str], dict[str, float]]
    bip: dict[tuple[str, str], dict[str, float]]
    hit: dict[str, dict[str, float]]
    # (steal attempts per time on first, success rate) from this season's box
    # scores (talent/running.py); None = the engine's defaults.
    steal: tuple[float, float] | None = None
    # (hit, power) per (bats, throws): how much more often a ball in play falls in, and how
    # much more of those hits go for extra bases, than the all-matchups `hit` table says.
    # Retrosheet 2023 to 2025: a right-handed batter facing a lefty has a BABIP 2% above the
    # average, a lefty batter facing a righty slightly below. Empty means no difference.
    index: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)

    def env(self) -> LeagueEnv:
        rates = {
            key: Rates(
                k=r["K"],
                bb=r["BB"],
                hbp=r["HBP"],
                hr=r["HR"],
                bip=dict(self.bip[key]),
                hit=self.index.get(key, (1.0, 1.0))[0],
                power=self.index.get(key, (1.0, 1.0))[1],
            )
            for key, r in self.pa.items()
        }
        steal = (
            {"steal_attempt": self.steal[0], "steal_success": self.steal[1]} if self.steal else {}
        )
        return LeagueEnv(rates=rates, hit_table={t: dict(h) for t, h in self.hit.items()}, **steal)


def default_league() -> League:
    """The placeholder environment, for when no real PAs are at hand."""
    env = DEFAULT_ENV
    return League(
        pa={k: {"K": r.k, "BB": r.bb, "HBP": r.hbp, "HR": r.hr} for k, r in env.rates.items()},
        bip={k: dict(r.bip) for k, r in env.rates.items()},
        hit={t: dict(h) for t, h in env.hit_table.items()},
    )


def league(pa: pd.DataFrame, prior: pd.DataFrame | None = None) -> League:
    """League rates from a season's PAs (`events.plate_appearances`). Early
    in a season, `prior` (last season's PAs) fills in up to FULL_SEASON_PA."""
    frames = [pa.assign(w=1.0)]
    if prior is not None and not prior.empty and len(pa) < FULL_SEASON_PA:
        frames.append(prior.assign(w=(FULL_SEASON_PA - len(pa)) / len(prior)))
    data = pd.concat(frames, ignore_index=True)
    if data["w"].sum() == 0:
        return default_league()
    fallback = default_league()
    rates, mix = {}, {}
    for bats in HANDS:
        for throws in HANDS:
            key = (bats, throws)
            sub = data[(data["stand"] == bats) & (data["p_throws"] == throws)]
            total = sub["w"].sum()
            if total < 500:
                rates[key], mix[key] = fallback.pa[key], fallback.bip[key]
                continue
            by = sub.groupby("outcome")["w"].sum()
            rates[key] = {o: float(by.get(o, 0.0)) / total for o in PER_PA}
            mix[key] = _shares(sub[sub["bip"].notna()], "bip", BIP_TYPES)
    hit = {}
    balls = data[data["bip"].notna()]
    for t in BIP_TYPES:
        sub = balls[balls["bip"] == t]
        total = sub["w"].sum()
        if total < 200:
            hit[t] = fallback.hit[t]
            continue
        by = sub.groupby("outcome")["w"].sum()
        hit[t] = {h: float(by.get(h, 0.0)) / total for h in HITS}
    return League(pa=rates, bip=mix, hit=hit, index=_split_index(balls, hit))


def _shares(frame: pd.DataFrame, col: str, keys) -> dict[str, float]:
    by = frame.groupby(col)["w"].sum()
    total = by.sum()
    return {k: float(by.get(k, 0.0)) / total if total else 0.0 for k in keys}


def _split_index(balls: pd.DataFrame, hit: dict) -> dict[tuple[str, str], tuple[float, float]]:
    """(hit, power) for each batter-hand / pitcher-hand matchup: hits and extra-base hits per
    ball in play against what the all-matchups `hit` table expects for the same balls."""
    sums = {t: sum(h.values()) for t, h in hit.items()}
    xbh = {t: h["2B"] + h["3B"] for t, h in hit.items()}
    out = {}
    for bats in HANDS:
        for throws in HANDS:
            sub = balls[(balls["stand"] == bats) & (balls["p_throws"] == throws)]
            weight = sub["w"]
            expected = (sub["bip"].map(sums).astype(float) * weight).sum()
            expected_xbh = (sub["bip"].map(xbh).astype(float) * weight).sum()
            if weight.sum() < 500 or expected <= 0 or expected_xbh <= 0:
                continue
            seen = (sub["outcome"].isin(HITS) * weight).sum()
            seen_xbh = (sub["outcome"].isin(("2B", "3B")) * weight).sum()
            hit_index = seen / expected
            out[(bats, throws)] = (float(hit_index), float(seen_xbh / expected_xbh / hit_index))
    return out
