"""Save a packed season to the database as sim weeks and games."""

from __future__ import annotations

import zlib

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from npl_sim.db.models import Game, Season, SimWeek, Team
from npl_sim.schedule.calendar import Week
from npl_sim.schedule.packer import ScheduledSeries


def game_seed(season: int, day: object, home: str) -> int:
    """Stable per-game seed so a re-sim with the same inputs is identical."""
    return zlib.crc32(f"{season}|{day}|{home}".encode()) & 0x7FFFFFFF


def save_schedule(
    session: Session, season: int, weeks: list[Week], schedule: list[ScheduledSeries]
) -> int:
    """Write weeks and games for `season`. Refuses if the season already has games."""
    existing = session.scalar(
        select(func.count())
        .select_from(Game)
        .join(SimWeek, SimWeek.id == Game.week_id)
        .where(SimWeek.season_year == season)
    )
    if existing:
        raise RuntimeError(f"season {season} already has {existing} games scheduled")

    season_row = session.get(Season, season)
    if season_row is None:
        season_row = Season(year=season)
        session.add(season_row)
    season_row.start_date = min(d for w in weeks for d in w.game_days)
    season_row.end_date = max(d for w in weeks for d in w.game_days)

    week_ids = {}
    for w in weeks:
        row = session.scalar(
            select(SimWeek).where(SimWeek.season_year == season, SimWeek.number == w.number)
        )
        if row is None:
            row = SimWeek(season_year=season, number=w.number, start_date=w.start, end_date=w.end)
            session.add(row)
            session.flush()
        week_ids[w.number] = row.id

    team_ids = {t.name: t.id for t in session.scalars(select(Team))}
    count = 0
    for item in schedule:
        s = item.series
        for d in item.dates:
            session.add(
                Game(
                    week_id=week_ids[item.week],
                    game_date=d,
                    home_team_id=team_ids[s.home],
                    away_team_id=team_ids[s.away],
                    seed=game_seed(season, d, s.home),
                )
            )
            count += 1
    session.flush()
    return count
