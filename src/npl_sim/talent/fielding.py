"""Fielders' and parks' real numbers, from the pipeline's Savant leaderboards.

- Range (Dave, 2026-10-13): at each position, the average of the fielding
  metrics he has there (Statcast OAA in runs, DRS and UZR from FanGraphs'
  fielding leaderboard) over this season to date and the three before,
  weighted by innings, per 150 games (1350 innings), regressed toward average
  by RANGE_REGRESSION innings. A player without them is average at each
  eligible position (engine/fielding.py). Spec: "Rolling OAA, DRS and arm,
  regressed and updated weekly".
- Catchers: framing runs per 150 games caught, over the same seasons,
  regressed by FRAMING_REGRESSION balls in play; pop time to second sets his
  arm.
- Outfield arms: Statcast arm strength (mph) at the position.
- Parks: Savant's 3-year park factors by batter side, used to neutralize
  each player's record for the parks he played in (the sim's own parks are
  the NPL's custom ones, applied once per PA in the engine).
"""

from __future__ import annotations

import pandas as pd

from npl_sim.engine import fielding as glove

FIELDERS = {2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS", 7: "LF", 8: "CF", 9: "RF"}
SEASONS_BACK = 3  # seasons before this one in the averages
INNINGS_PER_SEASON = 1350.0  # 150 games
BIP_PER_INNING = 25 / 9  # balls in play (no homers) per inning
BIP_PER_SEASON = INNINGS_PER_SEASON * BIP_PER_INNING
RANGE_REGRESSION = 700.0  # innings of league-average fielding added to his record
FRAMING_REGRESSION = 1500.0  # balls in play caught
OAA_RUNS = 0.85  # runs per out above average, for a FanGraphs OAA column in outs
METRICS = ("drs", "uzr")  # FanGraphs fielding columns, in runs
POP_AVERAGE, POP_PER_ARM = 1.95, 0.08  # seconds to second base; one arm unit
OF_ARM_AVERAGE, OF_ARM_PER_UNIT = 86.0, 4.0  # mph
ARM_RANGE = 2.0
RANGE_POSITIONS = ("1B", "2B", "3B", "SS", "LF", "CF", "RF")  # catchers: framing and arm

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


def defense_ratings(seasons: list[tuple]) -> dict[str, dict[str, float]]:
    """{id: {position: runs per 150 games}}. `seasons` is [(seasons back,
    FanGraphs fielding board or None, Statcast OAA board or None, bip_on_field
    or None)], 0 = this season to date; seasons more than SEASONS_BACK back are
    ignored. Each season counts by the innings he played there."""
    runs: dict[tuple[str, str], float] = {}
    innings: dict[tuple[str, str], float] = {}
    for back, fg, oaa, field in seasons:
        if back > SEASONS_BACK:
            continue
        found: dict[tuple[str, str], dict] = {}  # (id, pos) -> {metric: runs, "inn": ...}
        if fg is not None and not fg.empty:
            cols = {str(c).lower(): c for c in fg.columns}
            pos_col, inn_col = cols.get("pos"), cols.get("inn")
            if pos_col is not None and inn_col is not None and "mlbam_id" in fg:
                for _, r in fg.iterrows():
                    pos, inn = str(r[pos_col]), _innings(r[inn_col])
                    if pd.isna(r["mlbam_id"]) or pos not in RANGE_POSITIONS or not inn:
                        continue
                    rec = found.setdefault((str(int(r["mlbam_id"])), pos), {})
                    rec["inn"] = inn
                    for m in METRICS:
                        v = _num(r[cols[m]]) if m in cols else None
                        if v is not None:
                            rec[m] = v
                    if "oaa" in cols and _num(r[cols["oaa"]]) is not None:
                        rec["oaa"] = OAA_RUNS * _num(r[cols["oaa"]])
        if oaa is not None and not oaa.empty:
            seen = {}
            if field is not None:
                seen = {(str(int(r.mlbam_id)), r.position): r.bip for r in field.itertuples()}
            for r in oaa.itertuples():
                pos = str(getattr(r, "primary_pos_formatted", "") or "")
                frp = _num(getattr(r, "fielding_runs_prevented", None))
                if pd.isna(r.mlbam_id) or pos not in RANGE_POSITIONS or frp is None:
                    continue
                key = (str(int(r.mlbam_id)), pos)
                rec = found.setdefault(key, {})
                rec["oaa"] = frp  # Statcast's own runs beat FanGraphs' copy
                if "inn" not in rec and seen.get(key):
                    rec["inn"] = seen[key] / BIP_PER_INNING
        for key, rec in found.items():
            values = [rec[m] for m in (*METRICS, "oaa") if m in rec]
            if values and rec.get("inn"):
                runs[key] = runs.get(key, 0.0) + sum(values) / len(values)
                innings[key] = innings.get(key, 0.0) + rec["inn"]
    out: dict[str, dict[str, float]] = {}
    for (pid, pos), inn in innings.items():
        per_season = runs[(pid, pos)] * INNINGS_PER_SEASON / inn
        out.setdefault(pid, {})[pos] = per_season * inn / (inn + RANGE_REGRESSION)
    return out


def framing_ratings(seasons: list[tuple[int, pd.DataFrame | None, pd.DataFrame]]) -> dict:
    """{id: framing runs per 150 games caught}, regressed."""
    runs: dict[int, float] = {}
    bip: dict[int, float] = {}
    for back, board, field in seasons:
        if board is None or back > SEASONS_BACK or "rv_tot" not in board:
            continue
        caught = field[field["position"] == "C"].set_index("mlbam_id")["bip"]
        for pid, rv in zip(board["mlbam_id"], board["rv_tot"], strict=True):
            rv = _num(rv)
            if pd.isna(pid) or rv is None or int(pid) not in caught.index:
                continue
            runs[int(pid)] = runs.get(int(pid), 0.0) + rv
            bip[int(pid)] = bip.get(int(pid), 0.0) + caught[int(pid)]
    return {
        str(p): runs[p] * BIP_PER_SEASON / n * n / (n + FRAMING_REGRESSION)
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


def positions(eligible: dict[str, float], ratings=None, framing=None) -> dict[str, float]:
    """A hitter's `positions` for the engine: at each eligible position his
    rating there when he has one; else the best free move from a position he
    has a rating at (outfield to outfield, short to second or third), if that
    beats average; else average. Framing at catcher. `ratings` is
    {position: runs per 150 games} (defense_ratings) or an older
    (position, runs) pair."""
    if isinstance(ratings, tuple):
        ratings = {ratings[0]: ratings[1]}
    ratings = ratings or {}
    out = {}
    for pos in eligible or ratings:
        if pos in ratings:
            out[pos] = ratings[pos]
        else:
            moves = [glove.move(r, frm, pos) for frm, r in ratings.items() if glove.free(frm, pos)]
            out[pos] = max([0.0, *moves])
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


def _innings(v) -> float | None:
    """FanGraphs innings, where .1 and .2 are thirds."""
    f = _num(v)
    if f is None:
        return None
    whole = int(f)
    return whole + round((f - whole) * 10) / 3


def _clip(x: float) -> float:
    return max(-ARM_RANGE, min(ARM_RANGE, x))


def _num(v) -> float | None:
    try:
        f = float(str(v).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f
