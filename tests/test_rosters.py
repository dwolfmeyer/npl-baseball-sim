from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from npl_sim.db.models import Base, Player, RosterEntry, RosterTier
from npl_sim.db.seed import sync_alignment
from npl_sim.league import load_league_file
from npl_sim.rosters.importer import get_or_create_week, import_rosters
from npl_sim.rosters.npl_app import ROSTER_FLAGS, NplPlayer
from npl_sim.rosters.sources import players_from_database, players_from_dump

from .test_db import URL

# Ducksnorts and McFlys from the NPL app's repo fixture (Jan 2025 rosters).
SAMPLE = Path(__file__).parent / "fixtures" / "npl_sample.json"
TEST_TEAMS = ("Midwest Ducksnorts", "Hill Valley McFlys")


def player(flags=(), mlb_id=1, position="SS"):
    return NplPlayer(
        mlb_id=mlb_id,
        name="Test Player",
        position=position,
        mlb_org="SEA",
        team="Midwest Ducksnorts",
        flags=frozenset(flags),
    )


@pytest.mark.parametrize(
    ("flags", "tier"),
    [
        (["roster_40man"], RosterTier.MLB_ROSTER),
        (["roster_40man", "roster_7dayIL"], RosterTier.IL_7),
        (["roster_40man", "roster_56dayIL"], RosterTier.IL_56),
        (["roster_40man", "roster_eosIL"], RosterTier.IL_EOS),
        (["roster_40man", "roster_tripleA_option"], RosterTier.AAA),
        (["roster_outrighted"], RosterTier.OUTRIGHTED),
        (["roster_nonroster"], RosterTier.NON_ROSTER),
        (["roster_tripleA"], RosterTier.AAA),
        (["roster_doubleA"], RosterTier.AA),
        (["roster_singleA"], RosterTier.A),
        (["roster_restricted"], RosterTier.RESTRICTED),
        ([], None),
    ],
)
def test_tier_follows_npl_app_rules(flags, tier):
    assert player(flags).tier is tier


def test_only_active_40_man_is_mlb_roster():
    assert player(["roster_40man"]).is_active_mlb
    assert not player(["roster_40man", "roster_56dayIL"]).is_active_mlb
    assert not player(["roster_tripleA"]).is_active_mlb


def test_dump_source_reads_both_test_teams():
    players = players_from_dump(SAMPLE)
    assert {p.team for p in players} == set(TEST_TEAMS)
    mlb = [p for p in players if p.tier is RosterTier.MLB_ROSTER]
    assert sum(p.team == "Midwest Ducksnorts" for p in mlb) == 32
    assert sum(p.team == "Hill Valley McFlys" for p in mlb) == 37


@pytest.fixture
def session():
    engine = create_engine(URL)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        sync_alignment(s, load_league_file(), 2026)
        yield s
        s.rollback()
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_import_maps_every_mlb_roster_player_to_an_mlbam_id(session):
    week = get_or_create_week(session, 2026, 1, date(2026, 3, 23))
    report = import_rosters(session, players_from_dump(SAMPLE), week, teams=TEST_TEAMS)

    assert report.unmatched == []
    assert report.unknown_teams == set()
    assert report.tiers["Midwest Ducksnorts"][RosterTier.MLB_ROSTER] == 32
    assert report.tiers["Hill Valley McFlys"][RosterTier.MLB_ROSTER] == 37
    mlb_entries = session.scalars(
        select(RosterEntry).where(RosterEntry.tier == RosterTier.MLB_ROSTER)
    ).all()
    assert len(mlb_entries) == 69
    assert all(e.player.mlbam_id for e in mlb_entries)


def test_reimport_replaces_the_week_and_reuses_players(session):
    week = get_or_create_week(session, 2026, 1, date(2026, 3, 23))
    players = players_from_dump(SAMPLE)
    import_rosters(session, players, week)
    import_rosters(session, players, week)
    assert session.scalar(select(func.count()).select_from(RosterEntry)) == len(players)
    assert session.scalar(select(func.count()).select_from(Player)) == len(players)


def test_unmatched_and_unknown_team_are_reported_not_imported(session):
    week = get_or_create_week(session, 2026, 1, date(2026, 3, 23))
    no_id = player(["roster_40man"], mlb_id=None)
    stranger = NplPlayer(
        mlb_id=2,
        name="X",
        position="C",
        mlb_org=None,
        team="Sound Hydra",
        flags=frozenset({"roster_40man"}),
    )
    report = import_rosters(session, [no_id, stranger], week)
    assert report.unmatched_mlb_roster == [no_id]
    assert report.unknown_teams == {"Sound Hydra"}
    assert session.scalar(select(func.count()).select_from(RosterEntry)) == 0


def test_pitcher_hand_comes_from_position(session):
    week = get_or_create_week(session, 2026, 1, date(2026, 3, 23))
    import_rosters(session, [player(["roster_40man"], mlb_id=9, position="LHP")], week)
    assert session.scalar(select(Player.throws).where(Player.mlbam_id == 9)) == "L"


def test_week_must_start_on_monday(session):
    with pytest.raises(ValueError, match="Monday"):
        get_or_create_week(session, 2026, 1, date(2026, 3, 24))


@pytest.mark.skipif(not URL.startswith("postgresql"), reason="needs Postgres")
def test_database_source_reads_npl_app_tables():
    """Build the NPL app's two tables in a scratch schema and read them back."""
    engine = create_engine(URL)
    flag_cols = ", ".join(f"{f} boolean NOT NULL DEFAULT false" for f in ROSTER_FLAGS)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS npl_app CASCADE; CREATE SCHEMA npl_app"))
        conn.execute(text("CREATE TABLE npl_app.npl_team (id int PRIMARY KEY, full_name text)"))
        conn.execute(
            text(
                "CREATE TABLE npl_app.npl_player (mlb_id varchar PRIMARY KEY, scoresheet_id text, "
                "name text, raw_name text, position text, mlb_org text, team_id int, "
                + flag_cols
                + ")"
            )
        )
        conn.execute(text("INSERT INTO npl_app.npl_team VALUES (98, 'Midwest Ducksnorts')"))
        conn.execute(
            text(
                "INSERT INTO npl_app.npl_player "
                "(mlb_id, name, position, mlb_org, team_id, roster_40man) "
                "VALUES ('660271', 'Shohei Ohtani', 'DH', 'LAD', 98, true), "
                "('-', 'No Id', 'C', 'SEA', 98, true), ('1', 'Free Agent', 'C', NULL, NULL, false)"
            )
        )
    try:
        sep = "&" if "?" in URL else "?"
        players = players_from_database(f"{URL}{sep}options=-csearch_path%3Dnpl_app")
        by_name = {p.name: p for p in players}
        assert set(by_name) == {"Shohei Ohtani", "No Id"}
        assert by_name["Shohei Ohtani"].mlb_id == 660271
        assert by_name["Shohei Ohtani"].tier is RosterTier.MLB_ROSTER
        assert by_name["No Id"].mlb_id is None
    finally:
        with engine.begin() as conn:
            conn.execute(text("DROP SCHEMA npl_app CASCADE"))
        engine.dispose()
