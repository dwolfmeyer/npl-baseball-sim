"""Baseball Savant: pitch-level Statcast and the fielding, running and park
leaderboards.

Pitch data and the leaderboards pybaseball covers go through pybaseball
(>= 2.2.7); its frames are stored raw as returned. Arm strength and park
factors are not in pybaseball, so they are fetched from Savant directly.
"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Callable
from datetime import date

import pandas as pd

from npl_sim.pipeline.http import Http

SOURCE = "savant"
SAVANT = "https://baseballsavant.mlb.com"


def fetch_statcast_day(day: date) -> pd.DataFrame:
    """Every regular-season pitch thrown on `day`."""
    from pybaseball import statcast

    frame = statcast(start_dt=day.isoformat(), end_dt=day.isoformat(), verbose=False)
    return frame if frame is not None else pd.DataFrame()


def statcast_url(day: date) -> str:
    d = day.isoformat()
    return f"{SAVANT}/statcast_search/csv?all=true&type=details&game_date_gt={d}&game_date_lt={d}"


def _pybaseball_boards() -> dict[str, Callable[[int], pd.DataFrame]]:
    import pybaseball as pb

    return {
        "oaa": lambda y: pb.statcast_outs_above_average(y, pos="all", min_att=1),
        "framing": lambda y: pb.statcast_catcher_framing(y, min_called_p=1),
        "poptime": lambda y: pb.statcast_catcher_poptime(y, min_2b_att=1, min_3b_att=0),
        "sprint_speed": lambda y: pb.statcast_sprint_speed(y, min_opp=1),
    }


def arm_strength_url(year: int) -> str:
    return (
        f"{SAVANT}/leaderboard/arm-strength?type=player&year={year}&minThrows=1&pos=&team=&csv=true"
    )


def park_factors_url(year: int, bat_side: str) -> str:
    return (
        f"{SAVANT}/leaderboard/statcast-park-factors?type=year&year={year}"
        f"&batSide={bat_side}&stat=index_wOBA&condition=All&rolling=3"
    )


LEADERBOARDS = ("oaa", "framing", "poptime", "sprint_speed", "arm_strength", "park_factors")


def fetch_leaderboard(http: Http, name: str, year: int) -> tuple[pd.DataFrame, str]:
    """One season-to-date leaderboard as a frame, with the URL it came from."""
    if name == "arm_strength":
        url = arm_strength_url(year)
        return pd.read_csv(io.BytesIO(http.get(url))), url
    if name == "park_factors":
        frames = []
        for side in ("L", "R"):
            url = park_factors_url(year, side)
            frame = parse_park_factors_page(http.get(url).decode("utf-8"))
            frame["bat_side"] = side
            frames.append(frame)
        return pd.concat(frames, ignore_index=True), park_factors_url(year, "{L,R}")
    board = _pybaseball_boards()[name]
    return board(year), f"pybaseball:{name}({year})"


_DATA_VAR = re.compile(r"var\s+data\s*=\s*(\[.*?\]);", re.DOTALL)


def parse_park_factors_page(html: str) -> pd.DataFrame:
    """The park factor leaderboard ships its rows as a `var data = [...]` literal."""
    match = _DATA_VAR.search(html)
    if not match:
        raise ValueError("Savant park factor page changed: no `var data` table found")
    return pd.DataFrame(json.loads(match.group(1)))


# --- clean ------------------------------------------------------------------

# Pitch columns the player model reads; anything else stays in the raw snapshot.
PITCH_COLUMNS = [
    "game_pk", "game_date", "at_bat_number", "pitch_number", "inning", "inning_topbot",
    "home_team", "away_team", "batter", "pitcher", "stand", "p_throws",
    "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b",
    "pitch_type", "release_speed", "release_spin_rate", "release_extension",
    "pfx_x", "pfx_z", "plate_x", "plate_z", "sz_top", "sz_bot", "zone",
    "type", "description", "events", "bb_type", "hit_location",
    "launch_speed", "launch_angle", "launch_speed_angle", "hc_x", "hc_y",
    "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
    "woba_value", "woba_denom", "babip_value", "iso_value", "delta_run_exp",
    "if_fielding_alignment", "of_fielding_alignment",
    "fielder_2", "fielder_3", "fielder_4", "fielder_5", "fielder_6",
    "fielder_7", "fielder_8", "fielder_9",
]  # fmt: skip

_ID_COLUMNS = ["game_pk", "batter", "pitcher", "on_1b", "on_2b", "on_3b"] + [
    f"fielder_{i}" for i in range(2, 10)
]


def clean_pitches(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=PITCH_COLUMNS)
    if "game_type" in raw.columns:
        raw = raw[raw["game_type"] == "R"]
    cols = [c for c in PITCH_COLUMNS if c in raw.columns]
    out = raw[cols].copy()
    for col in _ID_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.date
    return out.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(
        drop=True
    )


# Savant leaderboards name the MLBAM ID differently per board.
_ID_ALIASES = ("player_id", "entity_id", "fielder_id", "catcher_id", "id")


def clean_leaderboard(name: str, raw: pd.DataFrame, season: int, as_of: date) -> pd.DataFrame:
    out = raw.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    if name != "park_factors":
        id_col = next((c for c in _ID_ALIASES if c in out.columns), None)
        if id_col is None:
            raise ValueError(f"Savant {name} leaderboard has no player id column: {list(out)}")
        out = out.rename(columns={id_col: "mlbam_id"})
        out["mlbam_id"] = pd.to_numeric(out["mlbam_id"], errors="coerce").astype("Int64")
        out = out[out["mlbam_id"].notna()]
    out.insert(0, "season", season)
    out.insert(1, "as_of", as_of)
    return out.reset_index(drop=True)
