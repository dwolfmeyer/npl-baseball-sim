"""Each player's real MLB week, read from the pipeline's weekly clean tables,
in the shape the engine's playing-time budget wants (engine/workload.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from npl_sim.engine.workload import MlbWeek
from npl_sim.season_stats import (
    BAT_COUNTS,
    PITCH_COUNTS,
    BatterTotals,
    PitcherTotals,
)


def mlb_week(batting: pd.DataFrame, pitching: pd.DataFrame) -> MlbWeek:
    """PA per hitter, starts and relief outs per pitcher; ids as strings."""
    pa = batting.groupby("mlbam_id")["pa"].sum()
    starts = pitching.groupby("mlbam_id")["gs"].sum()
    relief = pitching[pitching["gs"] == 0].groupby("mlbam_id")["outs"].sum()
    start_outs = pitching[pitching["gs"] > 0].groupby("mlbam_id")["outs"].sum()
    return MlbWeek(
        pa={str(k): int(v) for k, v in pa.items() if v > 0},
        starts={str(k): int(v) for k, v in starts.items() if v > 0},
        relief_outs={str(k): int(v) for k, v in relief.items() if v > 0},
        start_outs={str(k): int(v) for k, v in start_outs.items() if v > 0},
    )


def load_mlb_week(clean_root: Path, season: int, week: int) -> MlbWeek:
    folder = Path(clean_root) / str(season) / f"week-{week:02d}"
    return mlb_week(
        pd.read_parquet(folder / "batting.parquet", columns=["mlbam_id", "pa"]),
        pd.read_parquet(folder / "pitching.parquet", columns=["mlbam_id", "gs", "outs"]),
    )


@dataclass
class RealStats:
    """What each MLB player really did in a week, in the same shape as the sim's season
    totals (`season_stats`), by player id as a string."""

    batters: dict[str, BatterTotals] = field(default_factory=dict)
    pitchers: dict[str, PitcherTotals] = field(default_factory=dict)


# The clean tables call strikeouts "so"; the sim's totals call them "k".
_RENAME = {"so": "k"}


def real_week_stats(batting: pd.DataFrame, pitching: pd.DataFrame) -> RealStats:
    """A player's counting stats for the week, summed over his games (the clean tables
    have a row per player per game)."""

    def fold(frame: pd.DataFrame, make, counts) -> dict:
        if frame.empty or "mlbam_id" not in frame:
            return {}
        frame = frame.rename(columns=_RENAME)
        have = [c for c in counts if c in frame.columns]
        out = {}
        for pid, row in frame.groupby("mlbam_id")[have].sum().iterrows():
            total = make()
            for c in have:
                setattr(total, c, int(row[c]))
            out[str(pid)] = total
        return out

    batters = fold(batting, BatterTotals, BAT_COUNTS)
    pitchers = fold(pitching, PitcherTotals, (*PITCH_COUNTS, "w", "l", "sv", "hld", "gs"))
    return RealStats(batters, pitchers)


def load_real_week_stats(clean_root: Path, season: int, week: int) -> RealStats:
    folder = Path(clean_root) / str(season) / f"week-{week:02d}"
    return real_week_stats(
        pd.read_parquet(folder / "batting.parquet"),
        pd.read_parquet(folder / "pitching.parquet"),
    )
