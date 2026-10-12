import os

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from npl_sim.db.models import Base, Division, Team, TeamSeason
from npl_sim.db.seed import sync_alignment
from npl_sim.league import LeagueFile, load_league_file

# Postgres when DATABASE_URL is set (CI), otherwise in-memory SQLite.
URL = os.environ.get("DATABASE_URL", "sqlite://")


@pytest.fixture
def session():
    engine = create_engine(URL)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s
        s.rollback()
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_sync_alignment_creates_league_structure(session):
    sync_alignment(session, load_league_file(), 2026)
    assert session.scalar(select(func.count()).select_from(Team)) == 24
    assert session.scalar(select(func.count()).select_from(Division)) == 6
    assert session.scalar(select(func.count()).select_from(TeamSeason)) == 24


def test_sync_alignment_is_idempotent(session):
    lf = load_league_file()
    sync_alignment(session, lf, 2026)
    sync_alignment(session, lf, 2026)
    assert session.scalar(select(func.count()).select_from(Team)) == 24
    assert session.scalar(select(func.count()).select_from(TeamSeason)) == 24


def test_realignment_keeps_prior_season_history(session):
    base = load_league_file().alignments[0].model_dump()
    realigned = load_league_file().alignments[0].model_dump()
    realigned["from_season"] = 2028
    al = realigned["leagues"][0]["divisions"]
    al[0]["teams"][0], al[1]["teams"][0] = al[1]["teams"][0], al[0]["teams"][0]
    lf = LeagueFile.model_validate({"alignments": [base, realigned]})

    sync_alignment(session, lf, 2026)
    sync_alignment(session, lf, 2028)

    def division(season):
        return session.scalar(
            select(Division.name)
            .join(TeamSeason, TeamSeason.division_id == Division.id)
            .join(Team, Team.id == TeamSeason.team_id)
            .where(Team.name == "Asbury Park Bosses", TeamSeason.season_year == season)
        )

    assert division(2026) == "Oil Can"
    assert division(2028) == "Splendid Splinter"
