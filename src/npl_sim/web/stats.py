"""Hitters' MLB slash lines by pitcher hand, and fielding grades, for the
lineup card.

Slash lines come from the pipeline's clean Statcast tables
(data/clean/<season>/week-NN/pitches.parquet): every plate appearance this
season, split by the pitcher's throwing hand. With no clean data for the
season (the pipeline runs on Dave's machine), the previous season's is used
if present, else the card shows none.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from npl_sim.engine import fielding as glove

CLEAN_ROOT_ENV = "NPL_CLEAN_ROOT"
DEFAULT_CLEAN_ROOT = Path("data/clean")
COLUMNS = ["batter", "p_throws", "events"]
HITS = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
WALKS = {"walk", "intent_walk"}
# Plate appearances that aren't at-bats besides walks and HBP.
NOT_AB = {"sac_fly", "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play", "catcher_interf"}
SAC_FLIES = {"sac_fly", "sac_fly_double_play"}
# Statcast `events` values that don't end a plate appearance.
NOT_PA = {"pickoff_1b", "pickoff_2b", "pickoff_3b", "caught_stealing_2b", "caught_stealing_3b",
          "caught_stealing_home", "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b",
          "pickoff_caught_stealing_home", "stolen_base_2b", "stolen_base_3b",
          "stolen_base_home", "wild_pitch", "passed_ball", "other_advance", "game_advisory",
          "runner_double_play", "truncated_pa", "pickoff_error_1b", "pickoff_error_2b",
          "pickoff_error_3b"}  # fmt: skip
SIDE_HAND = {"rhp": "R", "lhp": "L"}


@dataclass(frozen=True)
class Line:
    pa: int
    ab: int
    hits: int
    bases: int
    walks: int
    hbp: int
    sf: int

    @property
    def avg(self) -> float:
        return self.hits / self.ab if self.ab else 0.0

    @property
    def obp(self) -> float:
        den = self.ab + self.walks + self.hbp + self.sf
        return (self.hits + self.walks + self.hbp) / den if den else 0.0

    @property
    def slg(self) -> float:
        return self.bases / self.ab if self.ab else 0.0

    def text(self) -> str:
        """e.g. ".285/.350/.480"."""
        return "/".join(_rate(x) for x in (self.avg, self.obp, self.slg))


def _rate(x: float) -> str:
    return f"{x:.3f}".removeprefix("0")


@dataclass
class Splits:
    """Slash lines by MLBAM id and pitcher hand ("R"/"L"), and where they came from."""

    lines: dict[int, dict[str, Line]]
    season: int | None = None
    weeks: int = 0

    def line(self, mlb_id: int | None, side: str) -> Line | None:
        return self.lines.get(mlb_id, {}).get(SIDE_HAND.get(side, side))

    @property
    def source(self) -> str:
        if self.season is None:
            return "No MLB stats loaded yet: the pipeline's clean data isn't on this machine."
        weeks = f"{self.weeks} week" + ("" if self.weeks == 1 else "s")
        return f"{self.season} MLB stats, {weeks} of clean data."


def slash_lines(pitches: pd.DataFrame) -> dict[int, dict[str, Line]]:
    """Each batter's slash line vs right- and left-handed pitchers."""
    pa = pitches[pitches["events"].notna() & (pitches["events"] != "")]
    pa = pa[~pa["events"].isin(NOT_PA)]
    out: dict[int, dict[str, Line]] = {}
    for (batter, hand), g in pa.groupby(["batter", "p_throws"]):
        ev = g["events"]
        walks = int(ev.isin(WALKS).sum())
        hbp = int((ev == "hit_by_pitch").sum())
        n = len(ev)
        out.setdefault(int(batter), {})[str(hand)] = Line(
            pa=n,
            ab=n - walks - hbp - int(ev.isin(NOT_AB).sum()),
            hits=int(ev.isin(HITS).sum()),
            bases=int(ev.map(HITS).fillna(0).sum()),
            walks=walks,
            hbp=hbp,
            sf=int(ev.isin(SAC_FLIES).sum()),
        )
    return out


def _week_files(season_dir: Path) -> list[Path]:
    return sorted(season_dir.glob("week-*/pitches.parquet"))


def load_splits(clean_root: Path | str, season: int) -> Splits:
    """This season's splits from the clean tables, else last season's, else none."""
    root = Path(clean_root)
    for year in (season, season - 1):
        files = _week_files(root / str(year))
        if files:
            frames = [pd.read_parquet(f, columns=COLUMNS) for f in files]
            return Splits(slash_lines(pd.concat(frames, ignore_index=True)), year, len(files))
    return Splits({})


class SplitsCache:
    """Reloads the splits when the pipeline writes a new week."""

    def __init__(self, clean_root: Path | str, season: int):
        self.root, self.season = Path(clean_root), season
        self._key: tuple | None = None
        self._splits = Splits({})

    def get(self) -> Splits:
        files = [f for y in (self.season, self.season - 1) for f in _week_files(self.root / str(y))]
        key = tuple((str(f), f.stat().st_mtime_ns) for f in files)
        if key != self._key:
            self._splits, self._key = load_splits(self.root, self.season), key
        return self._splits


def grade(runs: float) -> int:
    """A fielding rating in runs per 600 balls in play on the 20-80 scouting
    scale: average is 50 and each 2.5 runs is 5 points (one grade), so about
    +12 runs is a 75 and -15 or worse a 20."""
    return int(max(20, min(80, 50 + 5 * round(runs / 2.5))))


def grades(positions: dict[str, float]) -> dict[str, int]:
    """His grade at every fielding position, as the sim rates him there."""
    return {p: grade(glove.rating_at(positions, p)) for p in glove.SPECTRUM if p != "DH"}
