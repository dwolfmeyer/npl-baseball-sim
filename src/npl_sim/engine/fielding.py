"""Fielders behind the pitcher: position ratings and the defensive spectrum.

Each hitter may carry `positions`: his fielding value (runs per 150 games,
against the average fielder there) at the positions he is eligible at. The
talent blend fills them from three-year averages of OAA, DRS and UZR
(talent/fielding.py); a player without those is average (0.0) at each
eligible position. On the 20-80 scale each 2.5 runs is one grade (5 points).

Out of position (Dave, 2026-10-12): use Tango's defensive spectrum (runs per
150 games, "Fielding Position Adjustments", The Book blog, 2006):

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

Free moves (Dave, 2026-10-13): an outfielder can play any outfield spot, and a
shortstop second or third, with no experience or out-of-position penalty. His
rating still carries over along the spectrum, so an elite center fielder in
right is better than elite there (two elite center fielders help in center
and right), while a left fielder in center keeps only what Zimmerman measured
corner outfielders lose in center.
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
# rate on that type, to turn runs per 150 games into hit probability.
SHARE = {
    "GB": {"3B": 0.27, "SS": 0.30, "2B": 0.28, "1B": 0.15},
    "LD": {"LF": 0.22, "CF": 0.25, "RF": 0.22, "SS": 0.09, "2B": 0.08, "3B": 0.08, "1B": 0.06},
    "FB": {"LF": 0.32, "CF": 0.38, "RF": 0.30},
}
# A catcher's runs at C are his framing runs per 150 games. Framing f moves
# strikeouts up and walks down by f; one unit of f is worth about 380 runs over
# a catcher's season (~4300 PA at ~0.09 runs per PA), so an elite +15 is 4%.
FRAMING_RUNS = 380.0
FRAMING_CAP = 0.06
RUNS_PER_HIT_SAVED = 0.75  # a hit turned into an out, in runs
LEAGUE_HIT_RATE = {"GB": 0.257, "LD": 0.662, "FB": 0.132}
BIP_PER_SEASON = 3750  # a team's balls in play (no homers) in 150 games
BIP_MIX = {"GB": 0.44, "LD": 0.21, "FB": 0.25}  # league share of balls in play by type


# Share of all balls in play each fielder handles.
CHANCES = {
    pos: sum(BIP_MIX[b] * SHARE[b].get(pos, 0.0) for b in SHARE)
    for pos in {p for shares in SHARE.values() for p in shares}
}


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


def free(frm: str, to: str) -> bool:
    """A move with no experience penalty: outfield to outfield, or short to
    second or third."""
    return (frm in OUTFIELD and to in OUTFIELD) or (frm == "SS" and to in ("2B", "3B"))


def move(rating: float, frm: str, to: str) -> float:
    """Value at `to` for a player rated `rating` at his experienced spot `frm`."""
    g = gap(frm, to)
    carried = rating + (DOWN_CREDIT * g if g > 0 else g)
    if free(frm, to):
        return carried
    value = carried - EXPERIENCE
    if to == "C":
        value -= CATCHER
    elif to != "1B" and frm != "DH" and _group(frm) != _group(to):
        value -= CROSS_GROUP
    return value


def rating_at(positions: dict[str, float], pos: str) -> float:
    """A hitter's fielding value at `pos`, in runs per 150 games. Players with
    no position data are treated as average anywhere."""
    if not positions or pos == "DH":
        return 0.0
    if pos in positions:
        return positions[pos]
    return max(move(r, frm, pos) for frm, r in positions.items())


def out_of_position(positions: dict[str, float], pos: str) -> bool:
    if not positions or pos == "DH" or pos in positions:
        return False
    return not any(free(frm, pos) for frm in positions)


def team_defense(base: Defense, lineup) -> Defense:
    """`base` adjusted for who plays where. `lineup` is [(hitter, position)];
    hitters carry `positions` (see the module docstring)."""
    runs = {pos: rating_at(getattr(h, "positions", {}), pos) for h, pos in lineup}
    if not any(runs.values()) and not any(getattr(h, "arm", 0.0) for h, _ in lineup):
        return base
    mult = dict(base.hit_mult)
    for bip, shares in SHARE.items():
        # Each fielder's runs come off the balls he handles, so +R saves R runs
        # over a season of balls in play, split by his share of each type.
        hits = sum(share * runs.get(pos, 0.0) / CHANCES[pos] for pos, share in shares.items())
        delta = -hits / (RUNS_PER_HIT_SAVED * BIP_PER_SEASON)  # hit probability per BIP
        mult[bip] = mult.get(bip, 1.0) * max(0.5, 1 + delta / LEAGUE_HIT_RATE[bip])
    catcher = runs.get("C", 0.0)
    at = {pos: h for h, pos in lineup}
    c = at.get("C")
    # A real catcher's arm is his pop time; one out of position also can't throw.
    c_arm = getattr(c, "arm", 0.0) if c else 0.0
    if c is not None and out_of_position(getattr(c, "positions", {}), "C"):
        c_arm += catcher / 15
    outfield = [getattr(at[p], "arm", 0.0) for p in ("LF", "CF", "RF") if p in at]
    return replace(
        base,
        hit_mult=mult,
        framing=base.framing + max(-FRAMING_CAP, min(FRAMING_CAP, catcher / FRAMING_RUNS)),
        catcher_arm=base.catcher_arm + c_arm,
        of_arm=base.of_arm + (sum(outfield) / len(outfield) if outfield else 0.0),
    )
