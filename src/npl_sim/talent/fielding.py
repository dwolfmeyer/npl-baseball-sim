"""Fielders' and parks' real numbers, from the pipeline's Savant leaderboards.

- Range: Statcast's fielding runs prevented (OAA in runs) at the player's
  primary position, per 600 balls in play he was on the field for (counted
  from the pitch data's fielder columns), this season and last weighted 5/4,
  regressed toward average by FIELD_REGRESSION balls in play. Spec: "Rolling
  OAA, DRS and arm, regressed and updated weekly".
- Catchers: framing runs per 600 balls in play caught, regressed the same
  way; pop time to second base sets his arm.
- Outfield arms: Statcast arm strength (mph) at the position.
- Parks: Savant's 3-year park factors by batter side, used to neutralize
  each player's record for the parks he played in (the sim's own parks are
  the NPL's custom ones, applied once per PA in the engine).
"""

from __future__ import annotations

import pandas as pd

FIELDERS = {2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS", 7: "LF", 8: "CF", 9: "RF"}
FIELD_REGRESSION = 1500  # balls in play
FRAMING_REGRESSION = 3000
SEASON_WEIGHTS = (5.0, 4.0)
OTHER_POSITION_CREDIT = 0.5  # share of his primary rating at other experienced spots
POP_AVERAGE, POP_PER_ARM = 1.95, 0.08  # seconds to second base; one arm unit
OF_ARM_AVERAGE, OF_ARM_PER_UNIT = 86.0, 4.0  # mph
ARM_RANGE = 2.0

# MLB team abbreviation (Statcast's home_team) -> MLB team id (Savant's main_team_id).
TEAM_IDS = {
    "LAA": 108, "AZ": 109, "ARI": 109, "BAL": 110, "BOS": 111, "CHC": 112, "CIN": 113,
    "CLE": 114, "COL": 115, "DET": 116, "HOU": 117, "KC": 118, "LAD": 119, "WSH": 120,
    "NYM": 121, "OAK": 133, "ATH": 133, "PIT": 134, "SD": 135, "SEA": 136, "SF": 137,
    "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "MIN": 142, "PHI": 143, "ATL": 144,
    "CWS": 145, "MIA": 146, "NYY": 147, "MIL": 158,
}  # fmt: skip
PARK_INDEX = {"hr": "index_hr", "1b": "index_1b", "2b": "index_2b", "3b": "index_3b"}


def bip_on_field(pitches: pd.DataFrame) -> pd.DataFrame:
    """Balls in play each fielder was on the field for, by position:
    columns mlbam_id, position, bip."""
    cols = [f"fielder_{n}" for n in FIELDERS if f"fielder_{n}" in pitches]
    if pitches.empty or not cols or "bb_type" not in pitches:
        return pd.DataFrame(columns=["mlbam_id", "position", "bip"])
    balls = pitches[pitches["bb_type"].notna() & (pitches.get("events") != "home_run")]
    long = balls[cols].melt(var_name="slot", value_name="mlbam_id").dropna()
    long["position"] = long["slot"].map(lambda c: FIELDERS[int(c.split("_")[1])])
    long["mlbam_id"] = long["mlbam_id"].astype(int)
    return long.groupby(["mlbam_id", "position"]).size().rename("bip").reset_index()


def range_ratings(seasons: list[tuple[int, pd.DataFrame | None, pd.DataFrame]]) -> dict:
    """{id: (position, runs per 600 BIP)} at each player's primary position.
    `seasons` is [(seasons back, oaa board or None, bip_on_field)], 0 = this season."""
    runs: dict[tuple[int, str], float] = {}
    bip: dict[tuple[int, str], float] = {}
    primary: dict[int, str] = {}
    for back, board, field in sorted(seasons, key=lambda s: s[0], reverse=True):
        if board is None or back >= len(SEASON_WEIGHTS):
            continue
        w = SEASON_WEIGHTS[back]
        seen = {(int(r.mlbam_id), r.position): r.bip for r in field.itertuples()}
        for r in board.itertuples():
            pos = str(getattr(r, "primary_pos_formatted", "") or "")
            frp = _num(getattr(r, "fielding_runs_prevented", None))
            if pd.isna(r.mlbam_id) or pos not in FIELDERS.values() or frp is None:
                continue
            key = (int(r.mlbam_id), pos)
            n = seen.get(key, 0)
            if n == 0:
                continue
            runs[key] = runs.get(key, 0.0) + w * frp
            bip[key] = bip.get(key, 0.0) + w * n
            primary[key[0]] = pos  # this season's primary wins (seasons run oldest first)
    out = {}
    for pid, pos in primary.items():
        n = bip.get((pid, pos), 0.0)
        if n:
            rate = runs[(pid, pos)] * 600 / n
            out[str(pid)] = (pos, rate * n / (n + FIELD_REGRESSION * SEASON_WEIGHTS[0]))
    return out


def framing_ratings(seasons: list[tuple[int, pd.DataFrame | None, pd.DataFrame]]) -> dict:
    """{id: framing runs per 600 BIP caught}, regressed."""
    runs: dict[int, float] = {}
    bip: dict[int, float] = {}
    for back, board, field in seasons:
        if board is None or back >= len(SEASON_WEIGHTS) or "rv_tot" not in board:
            continue
        w = SEASON_WEIGHTS[back]
        caught = field[field["position"] == "C"].set_index("mlbam_id")["bip"]
        for pid, rv in zip(board["mlbam_id"], board["rv_tot"], strict=True):
            rv = _num(rv)
            if pd.isna(pid) or rv is None or int(pid) not in caught.index:
                continue
            runs[int(pid)] = runs.get(int(pid), 0.0) + w * rv
            bip[int(pid)] = bip.get(int(pid), 0.0) + w * caught[int(pid)]
    return {
        str(p): runs[p] * 600 / n * n / (n + FRAMING_REGRESSION * SEASON_WEIGHTS[0])
        for p, n in bip.items()
        if n
    }


def arms(arm_board: pd.DataFrame | None, pop_board: pd.DataFrame | None) -> dict[str, float]:
    """{id: arm, about -2..+2}: catchers from pop time to second, everyone
    else from outfield arm strength."""
    out = {}
    if arm_board is not None and "arm_of" in arm_board:
        for pid, mph in zip(arm_board["mlbam_id"], arm_board["arm_of"], strict=True):
            mph = _num(mph)
            if pd.notna(pid) and mph is not None:
                out[str(int(pid))] = _clip((mph - OF_ARM_AVERAGE) / OF_ARM_PER_UNIT)
    if pop_board is not None and "pop_2b_sba" in pop_board:
        for pid, pop in zip(pop_board["mlbam_id"], pop_board["pop_2b_sba"], strict=True):
            pop = _num(pop)
            if pd.notna(pid) and pop is not None:
                out[str(int(pid))] = _clip((POP_AVERAGE - pop) / POP_PER_ARM)
    return out


def positions(eligible: dict[str, float], rating: tuple[str, float] | None, framing=None):
    """A hitter's `positions` for the engine: his primary position's rating,
    half of it at his other experienced spots, and framing at catcher."""
    out = dict(eligible)
    if rating is not None:
        pos, runs = rating
        for p in out:
            out[p] = runs * OTHER_POSITION_CREDIT
        out[pos] = runs
    if framing is not None:
        out["C"] = framing
    return out


def park_factors(board: pd.DataFrame | None) -> dict[tuple[str, str], dict[str, float]]:
    """{(team abbreviation, batter side): {"hr"|"1b"|"2b"|"3b": factor}},
    1.0 = neutral, from Savant's park factors (index 100 = neutral)."""
    if board is None or board.empty:
        return {}
    side_col = next((c for c in ("key_bat_side", "bat_side") if c in board), None)
    team_col = "main_team_id" if "main_team_id" in board else None
    if side_col is None or team_col is None:
        return {}
    abbr = {}
    for a, tid in TEAM_IDS.items():
        abbr.setdefault(tid, []).append(a)
    out = {}
    for _, row in board.iterrows():
        tid, side = _num(row[team_col]), str(row[side_col])[:1].upper()
        if tid is None or side not in ("L", "R"):
            continue
        factors = {}
        for k, col in PARK_INDEX.items():
            v = _num(row.get(col))
            factors[k] = v / 100 if v else 1.0
        for a in abbr.get(int(tid), []):
            out[(a, side)] = factors
    return out


def _clip(x: float) -> float:
    return max(-ARM_RANGE, min(ARM_RANGE, x))


def _num(v) -> float | None:
    try:
        f = float(str(v).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f
