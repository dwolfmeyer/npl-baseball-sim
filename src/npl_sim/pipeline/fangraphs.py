"""FanGraphs leaderboards (Stuff+, DRS, standard and advanced lines) and
projections, from the JSON API behind fangraphs.com.

pybaseball 2.2.7's FanGraphs scraper still targets the retired legacy
leaderboard page, so these calls go to the API directly. Rows carry FanGraphs'
own MLBAM ID (`xMLBAMID`); the Chadwick register fills any gaps.

The plan flags FanGraphs' terms for a check before this runs on a schedule.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from npl_sim.pipeline.http import Http

SOURCE = "fangraphs"
LEADERS = "https://www.fangraphs.com/api/leaders/major-league/data"
PROJECTIONS = "https://www.fangraphs.com/api/projections"

# dataset -> FanGraphs `stats` group
LEADERBOARDS = {"batting": "bat", "pitching": "pit", "fielding": "fld"}
PROJECTION_STATS = {"bat": "batting", "pit": "pitching"}


def leaders_params(stats: str, season: int, start: date | None = None, end: date | None = None):
    params = {
        "pos": "all",
        "stats": stats,
        "lg": "all",
        "qual": 0,
        "type": 8,
        "season": season,
        "season1": season,
        "ind": 0,
        "team": 0,
        "pageitems": 10000,
        "pagenum": 1,
        "month": 0,
    }
    if start and end:  # custom date window
        params.update(month=1000, startdate=start.isoformat(), enddate=end.isoformat())
    return params


def fetch_leaders(http: Http, dataset: str, season: int, end: date | None = None):
    """Season-to-date leaderboard through `end` (whole season if None)."""
    start = date(season, 3, 1) if end else None
    params = leaders_params(LEADERBOARDS[dataset], season, start, end)
    return http.get(LEADERS, params), _full(LEADERS, params)


def fetch_projections(http: Http, system: str, stats: str):
    params = {"type": system, "stats": stats, "pos": "all", "team": 0, "players": 0, "lg": "all"}
    return http.get(PROJECTIONS, params), _full(PROJECTIONS, params)


def _full(url: str, params: dict) -> str:
    return url + "?" + "&".join(f"{k}={v}" for k, v in params.items())


def rows(raw: bytes | str | dict | list) -> list[dict]:
    data = json.loads(raw) if isinstance(raw, bytes | str) else raw
    return data["data"] if isinstance(data, dict) else data


def clean(
    raw_rows: list[dict], season: int, as_of: date, fg_to_mlbam: dict[str, int] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rows keyed by MLBAM ID, rows with no MLBAM ID for the review queue)."""
    frame = pd.DataFrame(raw_rows)
    if frame.empty:
        return frame, frame
    # Leaderboard rows wrap names in HTML links; keep the plain-text name.
    for html_col, plain in (("Name", "PlayerName"), ("Team", "TeamNameAbb")):
        if plain in frame.columns:
            frame[html_col] = frame[plain]
    frame.columns = [str(c).strip() for c in frame.columns]
    mlbam = pd.to_numeric(frame.get("xMLBAMID"), errors="coerce")
    if fg_to_mlbam:
        fg_ids = frame.get("playerid", frame.get("playerids"))
        if fg_ids is not None:
            fallback = fg_ids.astype(str).map(fg_to_mlbam)
            mlbam = mlbam.fillna(pd.to_numeric(fallback, errors="coerce"))
    frame.insert(0, "mlbam_id", mlbam.astype("Int64"))
    frame.insert(1, "season", season)
    frame.insert(2, "as_of", as_of)
    has_id = frame["mlbam_id"].notna()
    return frame[has_id].reset_index(drop=True), frame[~has_id].reset_index(drop=True)
