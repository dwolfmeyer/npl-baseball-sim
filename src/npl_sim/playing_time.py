"""Each player's real MLB week, read from the pipeline's weekly clean tables,
in the shape the engine's playing-time budget wants (engine/workload.py)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from npl_sim.engine.workload import MlbWeek


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
