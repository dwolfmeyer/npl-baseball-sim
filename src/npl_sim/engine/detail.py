"""Descriptive detail for each play: who fielded it, the count, how a strikeout ended.

None of this changes what happens in a game. It draws from its own random stream
so adding or tuning detail never shifts a seeded game's outcomes.
"""

from __future__ import annotations

import random

# Scorebook numbers, used for double-play chains like 6-4-3.
POSITION_NUMBER = {
    "P": 1,
    "C": 2,
    "1B": 3,
    "2B": 4,
    "3B": 5,
    "SS": 6,
    "LF": 7,
    "CF": 8,
    "RF": 9,
}
POSITION_TITLE = {
    "P": "pitcher",
    "C": "catcher",
    "1B": "first baseman",
    "2B": "second baseman",
    "3B": "third baseman",
    "SS": "shortstop",
    "LF": "left fielder",
    "CF": "center fielder",
    "RF": "right fielder",
}

# Weights for a right-handed batter; a lefty mirrors them (pull side flips).
_INFIELD_GB = {"3B": 0.22, "SS": 0.30, "2B": 0.22, "1B": 0.14, "P": 0.12}
_OUTFIELD_GB = {"LF": 0.45, "CF": 0.25, "RF": 0.30}
_OUTFIELD_AIR = {"LF": 0.36, "CF": 0.36, "RF": 0.28}
_INFIELD_LD = {"3B": 0.25, "SS": 0.30, "2B": 0.25, "1B": 0.15, "P": 0.05}
_POPUP = {"3B": 0.22, "SS": 0.20, "2B": 0.18, "1B": 0.22, "C": 0.13, "P": 0.05}
_HR_DIRECTION = {
    "left field": 0.30,
    "left center field": 0.22,
    "center field": 0.14,
    "right center field": 0.16,
    "right field": 0.18,
}
_MIRROR = {"LF": "RF", "RF": "LF", "3B": "1B", "1B": "3B", "SS": "2B", "2B": "SS"}


def _mirror(weights: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, w in weights.items():
        flipped = _MIRROR.get(key, key)
        if key.startswith("left"):
            flipped = "right" + key[4:]
        elif key.startswith("right"):
            flipped = "left" + key[5:]
        out[flipped] = w
    return out


def _pick(rng: random.Random, weights: dict[str, float], side: str) -> str:
    w = weights if side == "R" else _mirror(weights)
    return rng.choices(list(w), weights=list(w.values()))[0]


def fielder(rng: random.Random, result: str, bip: str | None, side: str) -> str | None:
    """Position that fielded the ball, or None when nobody did (K, BB, HBP, HR)."""
    if result in ("K", "BB", "HBP", "HR") or bip is None:
        return None
    if result in ("1B", "2B", "3B"):
        return _pick(rng, _OUTFIELD_GB if bip == "GB" else _OUTFIELD_AIR, side)
    if result in ("SF", "FO"):
        return _pick(rng, _OUTFIELD_AIR, side)
    if result == "LO":
        return _pick(rng, _INFIELD_LD if rng.random() < 0.35 else _OUTFIELD_AIR, side)
    if result == "PO":
        return _pick(rng, _POPUP, side)
    # GO, DP, FC, and errors on grounders.
    if bip == "GB":
        return _pick(rng, _INFIELD_GB, side)
    if bip == "PU":
        return _pick(rng, _POPUP, side)
    return _pick(rng, _OUTFIELD_AIR, side)


def hr_ball(rng: random.Random) -> str:
    return "LD" if rng.random() < 0.15 else "FB"


def hr_direction(rng: random.Random, side: str) -> str:
    return _pick(rng, _HR_DIRECTION, side)


def count(rng: random.Random, kind: str, pitches: int) -> tuple[int, int, int]:
    """The count after the final pitch (balls, strikes), and total strikes thrown.

    `kind` is K, BB, HBP, HR or BIP. A ball in play or a hit batter ends the PA
    on the count it came on. Strikes thrown include fouls and the ball put in play.
    """
    if kind == "K":
        balls = rng.randint(0, min(3, pitches - 3)) if pitches > 3 else 0
        return balls, 3, pitches - balls
    if kind == "BB":
        return 4, min(2, pitches - 4), pitches - 4
    before = pitches - 1
    balls = rng.randint(0, min(3, before)) if before > 0 else 0
    strikes_before = before - balls
    final_is_strike = kind != "HBP"
    return balls, min(2, strikes_before), strikes_before + final_is_strike


def strikeout_kind(rng: random.Random) -> str:
    return "swinging" if rng.random() < 0.75 else "looking"
