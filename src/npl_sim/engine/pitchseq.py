"""A plausible pitch-by-pitch sequence for a plate appearance, for the Gameday
at-bat pop-up.

The engine decides each PA in one draw and records only its pitch count and
final count (`detail.count`). This fills in the pitches between: the order of
balls and strikes, how each strike came (called, swinging, foul), pitch type
and speed from the pitcher's real mix when the talent data has it
(`Pitcher.arsenal`), and a location in or out of the zone that fits the call.
It runs on its own RNG, seeded from the PA itself, after the game is played,
so it never changes a result and the same game always shows the same pitches.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

PITCH_NAME = {
    "FF": "Four-Seam Fastball",
    "FA": "Fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "SV": "Slurve",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "CS": "Slow Curve",
    "CH": "Changeup",
    "FS": "Splitter",
    "FO": "Forkball",
    "SC": "Screwball",
    "KN": "Knuckleball",
    "EP": "Eephus",
}
FASTBALLS = {"FF", "FA", "SI", "FC"}
# A league-average mix (share, mph) for pitchers without Statcast data.
DEFAULT_ARSENAL = (("FF", 0.34, 94.2), ("SI", 0.15, 93.4), ("SL", 0.18, 85.6),
                   ("CH", 0.11, 85.9), ("CU", 0.09, 80.2), ("FC", 0.07, 89.4),
                   ("ST", 0.06, 82.3))  # fmt: skip
# Strike zone, feet: plate half-width plus a ball's radius, and a typical
# hitter's knees to letters. x is from the catcher's view, + toward first base.
ZONE_X, ZONE_LO, ZONE_HI = 0.83, 1.55, 3.45
STRIKE_CALL = (("Called Strike", 0.42), ("Swinging Strike", 0.27), ("Foul", 0.31))


@dataclass(frozen=True)
class Pitch:
    number: int
    call: str  # Ball, Called Strike, Swinging Strike, Foul, In play, Hit By Pitch ...
    balls: int  # count after the pitch
    strikes: int
    code: str  # Statcast pitch type
    mph: float
    x: float  # feet from the middle of the plate, catcher's view
    z: float  # feet above the ground

    @property
    def name(self) -> str:
        return PITCH_NAME.get(self.code, self.code)

    @property
    def kind(self) -> str:
        """ball, strike or play: what the pop-up colors the pitch by."""
        if self.call in ("Ball", "Hit By Pitch"):
            return "ball"
        return "play" if self.call.startswith("In play") else "strike"


def sequence(
    key: str,
    result: str,
    pitches: int,
    count: tuple[int, int],
    arsenal=(),
    batter_side: str = "R",
    k_type: str | None = None,
    runs: int = 0,
) -> list[Pitch]:
    """The pitches of one PA. `count` is the final (balls, strikes) as logged;
    `key` seeds the draw (anything unique to the PA)."""
    rng = random.Random(f"pitchseq:{key}")
    calls = _calls(rng, result, pitches, count, k_type, runs)
    mix = [(c, s, v) for c, s, v in arsenal if s > 0] or list(DEFAULT_ARSENAL)
    out, balls, strikes = [], 0, 0
    for n, call in enumerate(calls, 1):
        code, mph = _pitch(rng, mix, balls, strikes)
        x, z = _location(rng, call, batter_side)
        if call == "Ball":
            balls += 1
        elif call in ("Called Strike", "Swinging Strike", "Foul Bunt") or (
            call == "Foul" and strikes < 2
        ):
            strikes += 1
        out.append(Pitch(n, call, min(balls, 4), min(strikes, 3), code, mph, x, z))
    return out


def _calls(rng, result, pitches, count, k_type, runs) -> list[str]:
    balls, strikes = count
    if result == "K":
        last = {"swinging": "Swinging Strike", "bunt": "Foul Bunt"}.get(k_type, "Called Strike")
        before_b, before_s = balls, 2
    elif result == "BB":
        last, before_b, before_s = "Ball", 3, strikes
    elif result == "HBP":
        last, before_b, before_s = "Hit By Pitch", balls, strikes
    else:
        last, before_b, before_s = _in_play(result, runs), balls, strikes
    extra = max(0, pitches - 1 - before_b - before_s)  # two-strike fouls
    seq = ["B"] * before_b + ["S"] * before_s
    rng.shuffle(seq)
    calls = []
    for token in seq:
        calls.append("Ball" if token == "B" else _strike(rng))
    if extra:
        # Fouls with two strikes can only come after the second strike.
        second = [i for i, c in enumerate(calls) if c != "Ball"]
        start = second[1] + 1 if len(second) >= 2 else len(calls)
        for _ in range(extra):
            calls.insert(rng.randint(start, len(calls)), "Foul")
    return calls + [last]


def _strike(rng: random.Random) -> str:
    r = rng.random()
    for call, p in STRIKE_CALL:
        r -= p
        if r < 0:
            return call
    return STRIKE_CALL[-1][0]


def _in_play(result: str, runs: int) -> str:
    """Gameday's call on the last pitch of a ball in play."""
    if runs:
        return "In play, run(s)"
    return "In play, no out" if result in ("1B", "2B", "3B", "E") else "In play, out(s)"


def _pitch(rng, mix, balls, strikes) -> tuple[str, float]:
    # Behind in the count, more fastballs; two strikes, more breaking balls.
    weights = []
    for code, share, _ in mix:
        w = share
        if code in FASTBALLS:
            w *= 1.35 if balls > strikes else 0.8 if strikes == 2 else 1.0
        elif strikes == 2:
            w *= 1.3
        weights.append(w)
    code, _, mph = rng.choices(mix, weights=weights)[0]
    return code, round(mph + rng.gauss(0, 0.9), 1)


def _location(rng, call: str, side: str) -> tuple[float, float]:
    if call == "Hit By Pitch":
        inside = -1 if side == "R" else 1  # a righty stands on the catcher's left
        return round(inside * rng.uniform(1.25, 1.6), 2), round(rng.uniform(1.8, 3.6), 2)
    in_zone = {
        "Ball": 0.0,
        "Called Strike": 1.0,
        "Swinging Strike": 0.5,
        "Foul": 0.7,
    }.get(call, 0.85)
    if rng.random() < in_zone:
        x = rng.uniform(-ZONE_X + 0.05, ZONE_X - 0.05)
        z = rng.uniform(ZONE_LO + 0.05, ZONE_HI - 0.05)
    else:
        # Off the plate to a side, or above or below the zone.
        if rng.random() < 0.55:
            x = rng.choice((-1, 1)) * rng.uniform(ZONE_X + 0.1, ZONE_X + 0.9)
            z = rng.uniform(ZONE_LO - 0.6, ZONE_HI + 0.5)
        else:
            x = rng.uniform(-ZONE_X - 0.3, ZONE_X + 0.3)
            z = rng.choice((rng.uniform(0.6, ZONE_LO - 0.1), rng.uniform(ZONE_HI + 0.1, 4.3)))
    return round(x, 2), round(z, 2)
