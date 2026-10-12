"""Core league schema for v1 milestone 1.

Later milestones add weekly cards, lineups, game events and stats. Rules
kept from the plan:
- Players are keyed by MLBAM ID once matched; (AAA) stand-ins are separate
  rows (`is_standin`) so a fallback never edits a real player's record.
- Division membership is per season (`TeamSeason`), so realignment never
  rewrites history.
"""

from __future__ import annotations

import enum
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class RosterTier(enum.StrEnum):
    """Tiers on each NPL team page. Only MLB_ROSTER is sim-eligible."""

    MLB_ROSTER = "mlb_roster"
    IL_7 = "il_7"
    IL_56 = "il_56"
    IL_EOS = "il_eos"
    OUTRIGHTED = "outrighted"
    NON_ROSTER = "non_roster"
    AAA = "aaa"
    AA = "aa"
    A = "a"
    RESTRICTED = "restricted"
    FOREIGN = "foreign"
    RETIRED = "retired"


class League(Base):
    __tablename__ = "league"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(32), unique=True)
    dh: Mapped[bool] = mapped_column(Boolean, default=True)

    divisions: Mapped[list[Division]] = relationship(back_populates="league")


class Division(Base):
    __tablename__ = "division"
    __table_args__ = (UniqueConstraint("league_id", "name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("league.id"))
    name: Mapped[str] = mapped_column(String(64))

    league: Mapped[League] = relationship(back_populates="divisions")


class Team(Base):
    __tablename__ = "team"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)


class Season(Base):
    __tablename__ = "season"

    year: Mapped[int] = mapped_column(Integer, primary_key=True)
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)


class TeamSeason(Base):
    """A team's division for one season."""

    __tablename__ = "team_season"
    __table_args__ = (UniqueConstraint("season_year", "team_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    season_year: Mapped[int] = mapped_column(ForeignKey("season.year"))
    team_id: Mapped[int] = mapped_column(ForeignKey("team.id"))
    division_id: Mapped[int] = mapped_column(ForeignKey("division.id"))

    team: Mapped[Team] = relationship()
    division: Mapped[Division] = relationship()


class SimWeek(Base):
    """A Monday-Sunday sim week matching the MLB calendar."""

    __tablename__ = "sim_week"
    __table_args__ = (UniqueConstraint("season_year", "number"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    season_year: Mapped[int] = mapped_column(ForeignKey("season.year"))
    number: Mapped[int] = mapped_column(Integer)
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    # 1 hour before the week's first MLB game.
    lineup_lock_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Player(Base):
    __tablename__ = "player"

    id: Mapped[int] = mapped_column(primary_key=True)
    mlbam_id: Mapped[int | None] = mapped_column(Integer, unique=True)
    name: Mapped[str] = mapped_column(String(128))
    position: Mapped[str] = mapped_column(String(16))
    bats: Mapped[str | None] = mapped_column(String(1))
    throws: Mapped[str | None] = mapped_column(String(1))
    mlb_team: Mapped[str | None] = mapped_column(String(8))
    is_standin: Mapped[bool] = mapped_column(Boolean, default=False)


class RosterEntry(Base):
    """A player's NPL team and tier as of one sim week's roster import."""

    __tablename__ = "roster_entry"
    __table_args__ = (UniqueConstraint("week_id", "player_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    week_id: Mapped[int] = mapped_column(ForeignKey("sim_week.id"))
    team_id: Mapped[int] = mapped_column(ForeignKey("team.id"))
    player_id: Mapped[int] = mapped_column(ForeignKey("player.id"))
    tier: Mapped[RosterTier] = mapped_column(
        Enum(RosterTier, name="roster_tier", values_callable=lambda e: [m.value for m in e])
    )

    team: Mapped[Team] = relationship()
    player: Mapped[Player] = relationship()


class Game(Base):
    __tablename__ = "game"

    id: Mapped[int] = mapped_column(primary_key=True)
    week_id: Mapped[int] = mapped_column(ForeignKey("sim_week.id"))
    game_date: Mapped[date] = mapped_column(Date)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("team.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("team.id"))
    # Fixed per game so a re-sim with the same inputs reproduces the box score.
    seed: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="scheduled")
