"""Pinch hitting and manager styles.

Dave (2026-10-12): adopt pinch hitting with owner-set manager tendencies.
Before each PA the batting team may send up a bench bat when the owner's
`PinchHitting` rules allow it: late enough, a big enough moment (Tango's
leverage index, engine/leverage.py), and a bench hitter who is expected to
do clearly better against the pitcher on the mound. "Better" is the run
value of each hitter's outcome probabilities in this matchup (the same
model that plays the PA, so platoon splits, the talent blend and the park
all count), not just handedness.

A manager style bundles a starter hook (engine/hooks.py) with pinch-hitting
rules; owners pick one and can still change any setting on the card.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from npl_sim.engine.hooks import StarterHook

# Run value of each PA outcome (linear weights, outs = 0), for comparing hitters.
RUN_VALUE = {"BB": 0.69, "HBP": 0.72, "1B": 0.89, "2B": 1.27, "3B": 1.62, "HR": 2.10}


@dataclass(frozen=True)
class PinchHitting:
    """When the owner pinch-hits.

    - `enabled`: off = never.
    - `from_inning`: earliest inning.
    - `min_leverage`: the PA's leverage index must be at least this.
    - `platoon_only`: only for the platoon edge (the bench bat has it against
      this pitcher and the batter due up doesn't).
    - `min_gain`: the bench bat's expected run value per PA must beat the
      batter's by this much (0.015 is about 18 points of wOBA).
    - `max_per_game`: most pinch hitters in one game.
    - `out_of_position`: allow a bench bat who'd have to field a position he
      isn't experienced at (off = only for the DH or at his own positions).
    - `never_for`: player ids the owner never pinch-hits for.
    """

    enabled: bool = True
    from_inning: int = 7
    min_leverage: float = 2.0
    platoon_only: bool = True
    min_gain: float = 0.015
    max_per_game: int = 2
    out_of_position: bool = False
    never_for: tuple[str, ...] = field(default=())


NO_PINCH_HITTING = PinchHitting(enabled=False)


@dataclass(frozen=True)
class ManagerStyle:
    name: str
    label: str
    hook: StarterHook
    pinch: PinchHitting


STYLES = {
    s.name: s
    for s in (
        ManagerStyle(
            "sabermetric",
            "Sabermetric: starter out before a third time through or in a big spot; "
            "pinch-hits for any clear edge in leverage",
            StarterHook(times_through=2, max_runs=4, leverage=1.8, leverage_from_inning=6),
            PinchHitting(from_inning=6, min_leverage=1.8, platoon_only=False, min_gain=0.015),
        ),
        ManagerStyle(
            "aggressive",
            "Aggressive: quick hook, pinch-hits early and often",
            StarterHook(times_through=0, max_runs=4, leverage=1.5, leverage_from_inning=6),
            PinchHitting(
                from_inning=6,
                min_leverage=1.2,
                platoon_only=False,
                min_gain=0.008,
                max_per_game=3,
                out_of_position=True,
            ),
        ),
        ManagerStyle(
            "balanced",
            "Balanced (default): pitch count and game context; pinch-hits late for the "
            "platoon edge in big spots",
            StarterHook(),
            PinchHitting(),
        ),
        ManagerStyle(
            "old_school",
            "Old-school: rides the starter; pinch-hits only for the platoon edge in the "
            "8th or later in a big spot",
            StarterHook(times_through=0, max_runs=6, leverage=None),
            PinchHitting(
                from_inning=8, min_leverage=2.5, platoon_only=True, min_gain=0.020, max_per_game=1
            ),
        ),
        ManagerStyle(
            "passive",
            "Passive: lets starters work and never pinch-hits",
            StarterHook(times_through=0, max_runs=7, leverage=None),
            NO_PINCH_HITTING,
        ),
    )
}
DEFAULT_STYLE = "balanced"


def style(name: str) -> ManagerStyle:
    if name not in STYLES:
        raise ValueError(f"unknown manager style {name!r}; pick one of {', '.join(STYLES)}")
    return STYLES[name]


def run_value(probabilities: dict[str, float]) -> float:
    """Expected runs from one PA's outcome probabilities (pa_probabilities)."""
    total = 0.0
    for outcome, p in probabilities.items():
        result = outcome.split(":")[-1]
        total += p * RUN_VALUE.get(result, 0.0)
    return total
