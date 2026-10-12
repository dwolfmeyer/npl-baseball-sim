"""Bullpen usage: owner designations per reliever and leverage-based selection.

Owners tag each reliever with how they want him used (Dave, 2026-10-12):
high or medium leverage, low leverage, mop-up, long relief, or anywhere; start
of an inning only; a same-side matchup specialist; and how many outs a stint
should last. No closer is required: save spots and the 9th are simply high
leverage. When a pitcher comes out, the engine reads the game situation,
works out its leverage, and walks the tiers that fit it, preferring a
platoon edge on the next batter, then the owner's order.

Leverage is Tango's leverage index (1.0 = an average moment) read from his
table by inning, half, score margin and base-out state; see leverage.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from npl_sim.engine import leverage as li_table

ROLES = ("high", "medium", "low", "mopup", "long", "any")
MOPUP_MARGIN = 5  # runs either way

# Tier preference by situation; "any" relievers fill in before reaching up or
# down a tier.
PREFERENCE = {
    "high": ("high", "medium", "any", "low", "long", "mopup"),
    "medium": ("medium", "any", "low", "high", "long", "mopup"),
    "low": ("low", "any", "long", "medium", "mopup", "high"),
    "mopup": ("mopup", "long", "low", "any", "medium", "high"),
    "early": ("long", "low", "any", "medium", "mopup", "high"),
}


@dataclass(frozen=True)
class ReliefUsage:
    """How the owner wants one reliever used."""

    role: str = "any"  # high, medium, low, mopup, long or any
    start_of_inning_only: bool = False  # never brought in mid-inning
    vs: str | None = None  # "L" or "R": matchup specialist, used when that side is up
    max_outs: int | None = None  # stint length; default 3, or 9 for long and mop-up men

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown bullpen role {self.role!r}")
        if self.vs not in (None, "L", "R"):
            raise ValueError("vs must be 'L', 'R' or None")

    @property
    def outs(self) -> int:
        if self.max_outs is not None:
            return self.max_outs
        return 9 if self.role in ("long", "mopup") else 3


def leverage(inning: int, half: str, lead: int, bases: list, outs: int) -> float:
    """Leverage index of the coming PA (Tango's table) for the fielding team,
    `lead` runs ahead."""
    home_lead = lead if half == "top" else -lead  # the home team fields in the top
    return li_table.index(inning, half, home_lead, bases, outs)


def situation(inning: int, lead: int, li: float, starter_out: bool) -> str:
    if abs(lead) >= MOPUP_MARGIN:
        return "mopup"
    if starter_out and inning <= 5:
        return "early"
    if li >= 1.5:
        return "high"
    if li >= 0.85:
        return "medium"
    return "low"


def choose(candidates: list, usage: dict, tier: str, mid_inning: bool, batter=None):
    """Pick the reliever for this spot from `candidates` (the owner's order),
    or None if nobody fits.

    A start-of-inning-only arm is skipped mid-inning. A specialist is used only
    when the next batter hits from his side, unless nobody else is left. Within
    a tier, a same-side matchup on the next batter wins, then the owner's order.
    """

    def plan(p) -> ReliefUsage:
        return usage.get(p.id, ReliefUsage())

    def side(p) -> str | None:
        return batter.side_vs(p.throws) if batter is not None else None

    pool = [p for p in candidates if not (mid_inning and plan(p).start_of_inning_only)]
    if not pool:
        return None
    pool = [p for p in pool if plan(p).vs in (None, side(p))] or pool
    order = {role: i for i, role in enumerate(PREFERENCE[tier])}
    owner = {p.id: i for i, p in enumerate(candidates)}

    def rank(p):
        return (order[plan(p).role], side(p) != p.throws, owner[p.id])

    return min(pool, key=rank)
