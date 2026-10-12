"""Fielders behind the pitcher: position ratings and the defensive spectrum.

Each hitter may carry `positions`: his fielding value (runs per 600 balls in
play, against the average fielder there) at the positions he is experienced
at (MLB-eligible). Milestone 7 fills these from OAA/DRS; until then they are
0.0, which makes the out-of-position rules below the only effect.

Out of position (Dave, 2026-10-12): use Tango's defensive spectrum (runs per
600 BIP, "Fielding Position Adjustments", The Book blog, 2006):

    C +8, CF +5, SS +4, 2B 0, 3B -1, LF -4, RF -4, 1B -8   (DH -12: no glove)

Zimmerman ("Re-Examining WAR's Defensive Spectrum", FanGraphs, 2015) measured
the change directly for players who played both spots of a pair (UZR,
2002-14), in PAIRS below. Where he measured a move, its gap is the average of
his number and Tango's; otherwise Tango's alone. (His numbers make center much
harder than the corners and short only a little harder than second or third.)

A player at a position he isn't experienced at starts from his best
experienced position, then:
- moving down the spectrum (to an easier spot) keeps only DOWN_CREDIT of the
  gap, since his skills only partly carry over, so 2B to 1B is cheap while
  1B to 2B is costly;
- moving up loses the whole gap;
- either way he pays an experience penalty (Tango measured about 7 runs in
  both infield and outfield, some of it selection bias; EXPERIENCE is 4 so
  a move down stays cheap), plus CROSS_GROUP more when he
  crosses between infield (and catcher) and outfield (not to first base, the
  easiest spot), and CATCHER more to put on the gear. A catcher moving out
  gets no credit for the catcher's spot (FROM_CATCHER).
So a good shortstop plays second or third at little cost, a shortstop in left
costs a few runs, and a left fielder in center or a first baseman at third
costs a lot.
"""

from __future__ import annotations

from dataclasses import replace

from npl_sim.engine.model import Defense

SPECTRUM = {"C": 8, "CF": 5, "SS": 4, "2B": 0, "3B": -1, "LF": -4, "RF": -4, "1B": -8, "DH": -12}
INFIELD = {"1B", "2B", "3B", "SS"}
OUTFIELD = {"LF", "CF", "RF"}
DOWN_CREDIT = 0.5
EXPERIENCE = 4.0
CROSS_GROUP = 5.0
CATCHER = 10.0
# A catcher's value is framing, blocking and his arm, which don't travel, so
# out from behind the plate he moves as if from second base (no range credit).
FROM_CATCHER = SPECTRUM["2B"]

# Zimmerman's pairs: a player's UZR at the first spot minus at the second,
# per 150 games, so moving first -> second changes his value by -d.
PAIRS = {
    ("LF", "RF"): 0.7, ("LF", "CF"): 8.8, ("CF", "RF"): -9.4,
    ("2B", "SS"): 3.0, ("2B", "3B"): -0.1, ("SS", "3B"): -2.8,
    ("1B", "3B"): 6.9, ("LF", "1B"): -5.5, ("RF", "1B"): -9.3,
    ("LF", "3B"): 1.6, ("LF", "2B"): 2.3, ("1B", "2B"): 4.2,
    ("RF", "3B"): 1.1, ("RF", "2B"): 1.5, ("CF", "2B"): -2.1,
    ("LF", "SS"): 5.9, ("CF", "3B"): -4.2, ("1B", "SS"): 11.2,
    ("CF", "1B"): -6.4, ("CF", "SS"): 6.2, ("RF", "SS"): 0.7,
}  # fmt: skip

# Share of each batted-ball type each fielder handles, and the league hit
# rate on that type, to turn runs per 600 BIP into hit probability.
SHARE = {
    "GB": {"3B": 0.27, "SS": 0.30, "2B": 0.28, "1B": 0.15},
    "LD": {"LF": 0.22, "CF": 0.25, "RF": 0.22, "SS": 0.09, "2B": 0.08, "3B": 0.08, "1B": 0.06},
    "FB": {"LF": 0.32, "CF": 0.38, "RF": 0.30},
}
RUNS_PER_HIT_SAVED = 0.75  # a hit turned into an out, in runs
LEAGUE_HIT_RATE = {"GB": 0.257, "LD": 0.662, "FB": 0.132}


def _group(pos: str) -> str:
    if pos in OUTFIELD:
        return "OF"
    if pos in INFIELD or pos == "C":
        return "IF"
    return pos  # DH


def gap(frm: str, to: str) -> float:
    """Runs gained moving `frm` -> `to` before credit and penalties (> 0: an
    easier spot): Tango's spectrum, averaged with Zimmerman's pair if measured."""
    tango = (FROM_CATCHER if frm == "C" else SPECTRUM[frm]) - SPECTRUM[to]
    if (frm, to) in PAIRS:
        return (tango - PAIRS[frm, to]) / 2
    if (to, frm) in PAIRS:
        return (tango + PAIRS[to, frm]) / 2
    return tango


def move(rating: float, frm: str, to: str) -> float:
    """Value at `to` for a player rated `rating` at his experienced spot `frm`."""
    g = gap(frm, to)
    value = rating + (DOWN_CREDIT * g if g > 0 else g) - EXPERIENCE
    if to == "C":
        value -= CATCHER
    elif to != "1B" and frm != "DH" and _group(frm) != _group(to):
        value -= CROSS_GROUP
    return value


def rating_at(positions: dict[str, float], pos: str) -> float:
    """A hitter's fielding value at `pos`, in runs per 600 BIP. Players with
    no position data are treated as average anywhere."""
    if not positions or pos == "DH":
        return 0.0
    if pos in positions:
        return positions[pos]
    return max(move(r, frm, pos) for frm, r in positions.items())


def out_of_position(positions: dict[str, float], pos: str) -> bool:
    return bool(positions) and pos != "DH" and pos not in positions


def team_defense(base: Defense, lineup) -> Defense:
    """`base` adjusted for who plays where. `lineup` is [(hitter, position)];
    hitters carry `positions` (see the module docstring)."""
    runs = {pos: rating_at(getattr(h, "positions", {}), pos) for h, pos in lineup}
    if not any(runs.values()):
        return base
    mult = dict(base.hit_mult)
    for bip, shares in SHARE.items():
        hits = sum(share * runs.get(pos, 0.0) for pos, share in shares.items())
        delta = -hits / (RUNS_PER_HIT_SAVED * 600)  # change in hit probability per BIP
        mult[bip] = mult.get(bip, 1.0) * max(0.5, 1 + delta / LEAGUE_HIT_RATE[bip])
    catcher = runs.get("C", 0.0)
    return replace(
        base,
        hit_mult=mult,
        framing=base.framing + catcher / 1000,
        catcher_arm=base.catcher_arm + catcher / 15,
    )
