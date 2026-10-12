"""Statcast pitches reduced to what the talent blend reads: one row per plate
appearance (its outcome in the engine's terms) and per-pitch process flags.

Outcomes follow `engine.model`: K, BB, HBP, HR, or a ball in play typed GB,
LD, FB or PU that went for a 1B, 2B, 3B or an out. Intentional walks, bunts,
catcher's interference and truncated PAs are left out, since the sim doesn't
play them; reaching on an error counts as an out (the engine adds errors on
its own).
"""

from __future__ import annotations

import pandas as pd

OUTCOME = {
    "strikeout": "K",
    "strikeout_double_play": "K",
    "walk": "BB",
    "hit_by_pitch": "HBP",
    "home_run": "HR",
    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "field_out": "OUT",
    "force_out": "OUT",
    "grounded_into_double_play": "OUT",
    "double_play": "OUT",
    "triple_play": "OUT",
    "fielders_choice": "OUT",
    "fielders_choice_out": "OUT",
    "field_error": "OUT",
    "sac_fly": "OUT",
    "sac_fly_double_play": "OUT",
}
BB_TYPE = {"ground_ball": "GB", "line_drive": "LD", "fly_ball": "FB", "popup": "PU"}
# A ball in play with no batted-ball type (rare) is typed by its result.
_FALLBACK_TYPE = {"1B": "GB", "2B": "LD", "3B": "LD", "OUT": "GB"}

SWINGS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "foul_bunt",
    "missed_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
    "bunt_foul_tip",
}
WHIFFS = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}
BARREL = 6  # launch_speed_angle code for a barrel
HARD_HIT_MPH = 95.0
FASTBALLS = {"FF", "SI", "FA"}

PA_COLUMNS = [
    "game_date", "batter", "pitcher", "stand", "p_throws", "home_team", "outcome", "bip",
    "xba", "barrel", "hard",
]  # fmt: skip


def plate_appearances(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per PA the sim plays: batter, pitcher, their hands, `outcome`
    (K/BB/HBP/HR/1B/2B/3B/OUT), `bip` type for balls in play (else None) and
    contact quality (xBA, barrel, hard-hit) where Statcast measured it."""
    if pitches.empty or "events" not in pitches:
        return pd.DataFrame(columns=PA_COLUMNS)
    pa = pitches[pitches["events"].isin(OUTCOME.keys())].copy()
    pa["outcome"] = pa["events"].map(OUTCOME)
    in_play = pa["outcome"].isin(_FALLBACK_TYPE.keys())
    typed = pa.get("bb_type", pd.Series(index=pa.index, dtype=object)).map(BB_TYPE)
    pa["bip"] = typed.where(in_play).fillna(pa["outcome"].map(_FALLBACK_TYPE).where(in_play))
    contact = pa["outcome"].isin({"1B", "2B", "3B", "OUT", "HR"})
    pa["xba"] = _col(pa, "estimated_ba_using_speedangle").where(contact)
    pa["barrel"] = (_col(pa, "launch_speed_angle") == BARREL).astype(float).where(contact)
    speed = _col(pa, "launch_speed")
    pa["hard"] = (speed >= HARD_HIT_MPH).astype(float).where(contact & speed.notna())
    for hand in ("stand", "p_throws"):
        pa[hand] = pa[hand].astype(str)
    if "home_team" not in pa:
        pa["home_team"] = None
    return pa[PA_COLUMNS].reset_index(drop=True)


def pitch_flags(pitches: pd.DataFrame) -> pd.DataFrame:
    """Per pitch: batter, pitcher, swing, whiff, in-zone, chase (a swing out of
    the zone), pitch type and velocity."""
    if pitches.empty:
        return pd.DataFrame(
            columns=["batter", "pitcher", "swing", "whiff", "zone", "chase", "pitch_type", "velo"]
        )
    desc = pitches["description"].astype(str)
    zone = _col(pitches, "zone")
    out = pd.DataFrame(
        {
            "batter": pitches["batter"],
            "pitcher": pitches["pitcher"],
            "swing": desc.isin(SWINGS),
            "whiff": desc.isin(WHIFFS),
            "zone": zone.between(1, 9),
            "pitch_type": pitches.get("pitch_type", pd.Series(index=pitches.index, dtype=object)),
            "velo": _col(pitches, "release_speed"),
        }
    )
    out["chase"] = out["swing"] & ~out["zone"] & zone.notna()
    out["outside"] = ~out["zone"] & zone.notna()
    return out


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(float("nan"), index=frame.index)
    return pd.to_numeric(frame[name], errors="coerce")
