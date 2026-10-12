"""FanGraphs-to-MLBAM ID map from the Chadwick register (via pybaseball)."""

from __future__ import annotations

import pandas as pd


def fetch_register() -> pd.DataFrame:
    from pybaseball import chadwick_register

    return chadwick_register()


def fangraphs_to_mlbam(register: pd.DataFrame) -> dict[str, int]:
    reg = register[(register["key_fangraphs"] > 0) & (register["key_mlbam"] > 0)]
    return dict(zip(reg["key_fangraphs"].astype(str), reg["key_mlbam"].astype(int), strict=True))
