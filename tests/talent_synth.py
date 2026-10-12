"""Synthetic Statcast pitches from players of known talent, for the blend's tests."""

from __future__ import annotations

import random
from datetime import date, timedelta

import pandas as pd

TYPES = {"ground_ball": 0.44, "line_drive": 0.21, "fly_ball": 0.25, "popup": 0.10}
HIT = {  # P(1B, 2B, 3B) by batted-ball type
    "ground_ball": (0.235, 0.020, 0.002),
    "line_drive": (0.490, 0.160, 0.012),
    "fly_ball": (0.045, 0.075, 0.012),
    "popup": (0.015, 0.0, 0.0),
}


def pitches(
    batters: dict[int, dict],
    pitchers: dict[int, dict],
    n_pa: int,
    seed: int = 0,
    start: date = date(2026, 5, 4),
    days: int = 7,
) -> pd.DataFrame:
    """`batters`/`pitchers` map id -> {"hand": "L"/"R", "k": x, "hr": x, "bb": x}
    (multipliers on league rates, default 1). Each PA picks a batter and a
    pitcher at random (`weight` in the dict biases the pick)."""
    rng = random.Random(seed)
    b_ids, p_ids = list(batters), list(pitchers)
    b_w = [batters[b].get("weight", 1.0) for b in b_ids]
    p_w = [pitchers[p].get("weight", 1.0) for p in p_ids]
    rows = []
    for _ in range(n_pa):
        b, p = rng.choices(b_ids, b_w)[0], rng.choices(p_ids, p_w)[0]
        bt, pt = batters[b], pitchers[p]
        stand, throws = bt.get("hand", "R"), pt.get("hand", "R")
        platoon = bt.get("platoon", {}).get(throws, 1.0)
        k = 0.22 * bt.get("k", 1) * pt.get("k", 1) * platoon
        bb = 0.085 * bt.get("bb", 1) * pt.get("bb", 1)
        hr = 0.03 * bt.get("hr", 1) * pt.get("hr", 1)
        whiff = min(0.9, 0.25 * (k / 0.22))
        day = start + timedelta(days=rng.randrange(days))
        base = {"game_date": day, "batter": b, "pitcher": p, "stand": stand, "p_throws": throws}
        for _ in range(2):  # two non-terminal pitches
            zone = rng.choice([5, 5, 12])
            swing = zone == 5 and rng.random() < 0.7 or zone == 12 and rng.random() < 0.3
            desc = (
                ("swinging_strike" if rng.random() < whiff else "foul")
                if swing
                else ("called_strike" if zone == 5 else "ball")
            )
            rows.append({**base, "description": desc, "zone": zone, "events": None,
                         "pitch_type": "FF", "release_speed": pt.get("velo", 94.0)})  # fmt: skip
        r = rng.random()
        row = {**base, "zone": 5, "pitch_type": "FF", "release_speed": pt.get("velo", 94.0)}
        if r < k:
            row.update(events="strikeout", description="swinging_strike")
        elif r < k + bb:
            row.update(events="walk", description="ball", zone=12)
        elif r < k + bb + 0.01:
            row.update(events="hit_by_pitch", description="hit_by_pitch")
        elif r < k + bb + 0.01 + hr:
            row.update(events="home_run", description="hit_into_play", bb_type="fly_ball",
                       launch_speed=105.0, launch_speed_angle=6,
                       estimated_ba_using_speedangle=0.8)  # fmt: skip
        else:
            t = rng.choices(list(TYPES), list(TYPES.values()))[0]
            s, d, tr = HIT[t]
            x = rng.random()
            event = (
                "single"
                if x < s
                else "double"
                if x < s + d
                else "triple"
                if x < s + d + tr
                else "field_out"
            )  # noqa: E501
            row.update(events=event, description="hit_into_play", bb_type=t,
                       launch_speed=rng.gauss(89, 12), launch_speed_angle=rng.choice([1, 2, 3, 4]),
                       estimated_ba_using_speedangle=s + d + tr)  # fmt: skip
        rows.append(row)
    return pd.DataFrame(rows)
