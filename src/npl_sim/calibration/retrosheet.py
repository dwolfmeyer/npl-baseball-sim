"""League tables measured from Retrosheet play-by-play (the `plays.csv` season files).

Run `python -m npl_sim.calibration.retrosheet 2023plays.csv 2024plays.csv 2025plays.csv` to
print the tables that `engine/advancement.py` holds. Regular-season games only.

Advancement: for a runner on a single or double, how often he takes the extra base and how
often he is thrown out, by outs and by where the ball was hit (infield, left, center, right).
Each cell is shrunk toward its situation and outs row with SHRINK pseudo-plays so thin cells
(a double to the infield) stay close to the row.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import pandas as pd

FIELDS = ("IF", "LF", "CF", "RF")
SHRINK = 60
# (situation, hit, base the runner started on, base he reaches without taking more)
SITUATIONS = {
    "single_2nd": ("single", 2, "3B"),
    "single_1st": ("single", 1, "2B"),
    "double_1st": ("double", 1, "3B"),
}


def load(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p, low_memory=False) for p in paths]
    data = pd.concat(frames, ignore_index=True)
    return data[data["gametype"] == "regular"]


def _field(frame: pd.DataFrame) -> pd.Series:
    first = frame["loc"].astype(str).str[0]
    return pd.Series(
        np.select([first == "7", first == "8", first == "9"], ["LF", "CF", "RF"], "IF"),
        index=frame.index,
    )


def _side(frame: pd.DataFrame) -> pd.Series:
    switch = np.where(frame["pithand"] == "R", "L", "R")
    return pd.Series(np.where(frame["bathand"] == "B", switch, frame["bathand"]), index=frame.index)


def hits(data: pd.DataFrame) -> pd.DataFrame:
    """Singles and doubles with where they went and the batter's side."""
    hit = data[(data["pa"] == 1) & ((data["single"] + data["double"]) > 0) & (data["bunt"] == 0)]
    hit = hit.assign(field=_field(hit), side=_side(hit))
    kind = hit["hittype"].astype(str).str[-1].map({"G": "GB", "F": "FB", "L": "LD", "P": "PU"})
    return hit.assign(bip=kind).dropna(subset=["bip"])


def field_shares(hit: pd.DataFrame) -> dict:
    """P(field | result, ball type, batter side). Ground balls are all infield."""
    out: dict = {}
    for (single, bip, side), g in hit.groupby([hit["single"] > 0, "bip", "side"]):
        shares = g["field"].value_counts(normalize=True)
        out[("1B" if single else "2B", bip, side)] = {
            f: round(float(shares.get(f, 0)), 3) for f in FIELDS
        }
    return out


def _fates(hit: pd.DataFrame, hit_name: str, base: int, stop: str) -> pd.DataFrame:
    x = hit[hit[hit_name] == 1]
    if (hit_name, base) == ("single", 1):
        x = x[x["br2_pre"].isna()]  # a runner ahead of him on second blocks the extra base
    ids = x[f"br{base}_pre"]
    on = ids.notna()
    x, ids = x[on], ids[on]
    scored = (x["run1"] == ids) | (x["run2"] == ids) | (x["run3"] == ids) | (x["run_b"] == ids)
    where = {b: x[f"br{b}_post"] == ids for b in (1, 2, 3)}
    end = np.select([scored, where[1], where[2], where[3]], ["score", "1B", "2B", "3B"], "out")
    extra = (end == "score") if stop == "3B" else np.isin(end, ["3B", "score"])
    return pd.DataFrame(
        {
            "outs": x["outs_pre"].astype(int).values,
            "field": x["field"].values,
            "extra": extra,
            "out": end == "out",
        }
    )


def advancement(hit: pd.DataFrame) -> dict:
    """{situation: {(outs, field): (P(extra base), P(thrown out))}}."""
    tables: dict = {}
    for name, (hit_name, base, stop) in SITUATIONS.items():
        f = _fates(hit, hit_name, base, stop)
        cells: dict = defaultdict(tuple)
        for outs in (0, 1, 2):
            row = f[f["outs"] == outs]
            row_extra, row_out = row["extra"].mean(), row["out"].mean()
            for field in FIELDS:
                c = row[row["field"] == field]
                n = len(c)
                extra = (c["extra"].sum() + SHRINK * row_extra) / (n + SHRINK)
                out = (c["out"].sum() + SHRINK * row_out) / (n + SHRINK)
                cells[(outs, field)] = (round(float(extra), 3), round(float(out), 3))
        tables[name] = dict(cells)
    return tables


def main(paths: list[str]) -> None:
    hit = hits(load(paths))
    print("FIELD_SHARES =", field_shares(hit))
    print("ADVANCE =", advancement(hit))


if __name__ == "__main__":
    main(sys.argv[1:])
