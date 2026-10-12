"""Position eligibility: who may play where in the field.

League rule (Dave, 2026-10-12): a player is eligible at a position after 20
MLB games there in the previous season or 10 in the current season. Games
come from MLB box scores (every position a player appeared at in a game, from
the pipeline's weekly `batting` tables), so a late-inning switch counts.

Outfield eligibility is per spot (LF, CF, RF). A player who reaches the
threshold across the outfield without reaching it at any one spot is eligible
at the spot he played most. A player with no qualifying games anywhere (a
call-up, a prospect) falls back to his listed position until he qualifies.
Anyone can DH.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

FIELD_POSITIONS = ("C", "1B", "2B", "3B", "SS", "LF", "CF", "RF")
OUTFIELD = ("LF", "CF", "RF")
PREVIOUS_SEASON_GAMES = 20
CURRENT_SEASON_GAMES = 10

# Listed NPL positions mapped to the field spots they cover when a player has
# no qualifying MLB games yet.
LISTED_FALLBACK = {
    "C": {"C"},
    "CA": {"C"},
    "1B": {"1B"},
    "2B": {"2B"},
    "3B": {"3B"},
    "SS": {"SS"},
    "IF": {"1B", "2B", "3B", "SS"},
    "OF": {"LF", "CF", "RF"},
    "LF": {"LF"},
    "CF": {"CF"},
    "RF": {"RF"},
    "UT": {"1B", "2B", "3B", "SS", "LF", "CF", "RF"},
}


def games_by_position(batting: pd.DataFrame) -> dict[int, Counter]:
    """Games at each field position per player, from box-score batting rows.

    Each row is one player in one game, with `positions` a comma-separated
    list of every position he played in it (e.g. "SS,2B" or "PH,1B").
    """
    out: dict[int, Counter] = {}
    for row in batting[["mlbam_id", "game_pk", "positions"]].drop_duplicates().itertuples():
        spots = {p for p in str(row.positions or "").split(",") if p in FIELD_POSITIONS}
        if spots:
            out.setdefault(int(row.mlbam_id), Counter()).update(spots)
    return out


def load_season_games(clean_root: Path, season: int, through_week: int | None = None) -> dict:
    """Games by position for one season from its weekly clean tables."""
    frames = []
    for week_dir in sorted((clean_root / str(season)).glob("week-*")):
        number = int(week_dir.name.split("-")[1])
        if through_week is not None and number > through_week:
            continue
        path = week_dir / "batting.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path, columns=["mlbam_id", "game_pk", "positions"]))
    if not frames:
        return {}
    return games_by_position(pd.concat(frames, ignore_index=True))


def qualifying(
    previous: Mapping[str, int] | None,
    current: Mapping[str, int] | None,
    previous_needed: int = PREVIOUS_SEASON_GAMES,
    current_needed: int = CURRENT_SEASON_GAMES,
) -> set[str]:
    """Field positions a player qualifies at from games last season and this season."""
    previous, current = previous or {}, current or {}
    spots = {
        pos
        for pos in FIELD_POSITIONS
        if previous.get(pos, 0) >= previous_needed or current.get(pos, 0) >= current_needed
    }
    if not spots & set(OUTFIELD):
        for games, needed in ((previous, previous_needed), (current, current_needed)):
            total = sum(games.get(p, 0) for p in OUTFIELD)
            if total >= needed:
                spots.add(max(OUTFIELD, key=lambda p: games.get(p, 0)))
    return spots


def eligible_positions(
    previous: Mapping[str, int] | None,
    current: Mapping[str, int] | None,
    listed: str | None = None,
) -> set[str]:
    """Every position a player may be written into a lineup at, DH included."""
    spots = qualifying(previous, current)
    if not spots and listed:
        spots = set(LISTED_FALLBACK.get(listed.upper(), set()))
    return spots | {"DH"}


def eligibility_table(
    players: Iterable[tuple[int, str | None]],
    previous_season: Mapping[int, Mapping[str, int]],
    current_season: Mapping[int, Mapping[str, int]],
) -> dict[int, set[str]]:
    """Eligibility for each (mlbam_id, listed position) pair."""
    return {
        pid: eligible_positions(previous_season.get(pid), current_season.get(pid), listed)
        for pid, listed in players
    }
