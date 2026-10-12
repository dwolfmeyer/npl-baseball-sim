"""When a starter comes out: the owner's hook settings.

Dave (2026-10-12): pitch count alone isn't how managers pull starters. A
starter who gives up 8 runs in three innings is gone long before 100
pitches, and late in a close game a manager goes to the pen when the spot
gets big. So besides each pitcher's pitch limit, the owner sets:

- times through the order: pull him before he faces the lineup again;
- runs allowed: pull him once he has been charged with this many runs;
- leverage: from a given inning on, pull him when the spot's leverage index
  (Tango's table, see leverage.py) reaches a threshold.

The defaults apply when an owner sets nothing: out after 6 runs, and from the
7th on in a spot of LI 2.0 or more. Any of them can be turned off with None
(0 for times through). The same hook covers the bulk arm after an opener.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_RUNS = 6
DEFAULT_LEVERAGE = 2.0
DEFAULT_LEVERAGE_INNING = 7


@dataclass(frozen=True)
class StarterHook:
    times_through: int = 0  # pull after this many times through the order; 0 = no limit
    max_runs: int | None = DEFAULT_MAX_RUNS  # runs charged to him; None = no limit
    leverage: float | None = DEFAULT_LEVERAGE  # LI at or above this; None = never
    leverage_from_inning: int = DEFAULT_LEVERAGE_INNING

    def __post_init__(self) -> None:
        if self.times_through < 0:
            raise ValueError("times_through can't be negative")
        if self.max_runs is not None and self.max_runs < 1:
            raise ValueError("max_runs must be at least 1")
        if self.leverage is not None and self.leverage <= 0:
            raise ValueError("leverage threshold must be positive")

    def reason(self, batters_faced: int, runs: int, inning: int, li: float | None) -> str | None:
        """Why he comes out now, or None to leave him in."""
        if self.max_runs is not None and runs >= self.max_runs:
            return "runs"
        if self.times_through and batters_faced >= 9 * self.times_through:
            return "times_through"
        if (
            self.leverage is not None
            and li is not None
            and inning >= self.leverage_from_inning
            and li >= self.leverage
        ):
            return "leverage"
        return None
