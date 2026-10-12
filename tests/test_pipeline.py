"""Data pipeline tests. Responses are small hand-built payloads in the shape
the MLB Stats API, Savant and FanGraphs return; nothing here touches the
network."""

import json
import os
from datetime import UTC, date, datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from npl_sim.db.models import Base, Player, Season, SimWeek
from npl_sim.pipeline import calendar_check, fangraphs, ids, savant, statsapi
from npl_sim.pipeline.db import set_lineup_locks, sync_players
from npl_sim.pipeline.run import FrozenWeekError, Pipeline, lineup_locks, rates
from npl_sim.pipeline.snapshots import SnapshotStore
from npl_sim.schedule.calendar import load_calendar, sim_weeks, week_for_date

NYY, SF, LAD, CHC = 147, 137, 119, 112


def sched_game(pk, day, away, home, state="Final", score=(3, 2), start="T20:05:00Z"):
    abstract = "Final" if state in ("Final", "Postponed") else "Preview"
    return {
        "gamePk": pk,
        "gameType": "R",
        "officialDate": day,
        "gameDate": f"{day}{start}",
        "status": {"abstractGameState": abstract, "detailedState": state},
        "teams": {
            "away": {"team": {"id": away, "name": f"T{away}"}, "score": score[0]},
            "home": {"team": {"id": home, "name": f"T{home}"}, "score": score[1]},
        },
        "venue": {"id": 2395, "name": "Oracle Park"},
        "doubleHeader": "N",
        "gameNumber": 1,
    }


def schedule(*games):
    by_day = {}
    for g in games:
        by_day.setdefault(g["officialDate"], []).append(g)
    return {"dates": [{"date": d, "games": gs} for d, gs in sorted(by_day.items())]}


def batter(pid, order, pa, h, hr=0, bb=0, so=0, r=0, positions=("RF",)):
    return {
        "person": {"id": pid},
        "battingOrder": order,
        "allPositions": [{"abbreviation": p} for p in positions],
        "stats": {
            "batting": {
                "plateAppearances": pa,
                "atBats": pa - bb,
                "hits": h,
                "homeRuns": hr,
                "baseOnBalls": bb,
                "strikeOuts": so,
                "runs": r,
            }
        },
    }


def pitcher(pid, outs=None, ip=None, so=0, bf=0, gs=0):
    stats = {"strikeOuts": so, "battersFaced": bf, "gamesStarted": gs, "numberOfPitches": bf * 4}
    if outs is not None:
        stats["outs"] = outs
    if ip is not None:
        stats["inningsPitched"] = ip
    return {"person": {"id": pid}, "stats": {"pitching": stats}}


def boxscore(away_id, home_id):
    away = {
        "team": {"id": away_id},
        "batters": [1001, 1002, 1003],
        "pitchers": [1101, 1102],
        "players": {
            "ID1001": batter(1001, "100", 4, 2, hr=1, r=1),
            "ID1002": batter(1002, "200", 4, 1, bb=1, so=1, r=2),
            "ID1003": batter(1003, "201", 1, 0, so=1, positions=("PH", "LF")),
            "ID1101": pitcher(1101, outs=18, so=7, bf=24, gs=1),
            "ID1102": pitcher(1102, ip="2.1", so=2, bf=8),
        },
    }
    home = {
        "team": {"id": home_id},
        "batters": [2001, 2002],
        "pitchers": [2101],
        "players": {
            "ID2001": batter(2001, "100", 5, 1, so=2, r=1),
            "ID2002": batter(2002, "200", 4, 2, hr=1, r=1),
            "ID2101": pitcher(2101, outs=27, so=9, bf=36, gs=1),
        },
    }
    return {"teams": {"away": away, "home": home}}


def people(ids_):
    hands = {1001: ("L", "R"), 1101: ("R", "L")}
    return {
        "people": [
            {
                "id": i,
                "fullName": f"Player {i}",
                "batSide": {"code": hands.get(i, ("R", "R"))[0]},
                "pitchHand": {"code": hands.get(i, ("R", "R"))[1]},
                "primaryPosition": {"abbreviation": "P" if i % 1000 > 100 else "RF"},
            }
            for i in ids_
        ]
    }


# Week 1 of 2026: Opening Night Wed Mar 25, a final on Thu Mar 26, a game
# postponed Fri Mar 27 and made up Sat Mar 28, and a game still to be played
# the next week.
WEEK1 = schedule(
    sched_game(1, "2026-03-25", NYY, SF, start="T23:05:00Z"),
    sched_game(2, "2026-03-26", LAD, CHC, start="T18:20:00Z"),
    sched_game(3, "2026-03-27", CHC, LAD, state="Postponed", score=(None, None)),
    sched_game(3, "2026-03-28", CHC, LAD, score=(5, 4)),
)


class FakeHttp:
    """Answers Stats API and FanGraphs URLs from the payloads above."""

    def __init__(self):
        self.calls = []

    def get(self, url, params=None):
        self.calls.append((url, params))
        if url.endswith("/schedule"):
            return json.dumps(WEEK1).encode()
        if "/boxscore" in url:
            return json.dumps(boxscore(NYY, SF)).encode()
        if url.endswith("/people"):
            return json.dumps(people(map(int, params["personIds"].split(",")))).encode()
        if url.endswith("/transactions"):
            return json.dumps(
                {
                    "transactions": [
                        {
                            "id": 9,
                            "person": {"id": 1003, "fullName": "Player 1003"},
                            "date": "2026-03-27",
                            "typeCode": "SC",
                            "typeDesc": "Status Change",
                            "description": "placed on the 10-day IL",
                        },
                        {"id": 10, "date": "2026-03-27", "typeCode": "CU"},
                    ]
                }
            ).encode()
        if url == fangraphs.LEADERS:
            return json.dumps(
                {"data": [{"xMLBAMID": 1101, "playerid": 5, "PlayerName": "P", "sp_stuff": 112}]}
            ).encode()
        if url == fangraphs.PROJECTIONS:
            return json.dumps([{"xMLBAMID": None, "playerids": "77", "PA": 600}]).encode()
        raise AssertionError(f"unexpected URL {url}")


@pytest.fixture
def fake_savant(monkeypatch):
    def statcast_day(day):
        return pd.DataFrame(
            {
                "game_pk": [2, 2],
                "game_date": [day.isoformat()] * 2,
                "game_type": ["R", "R"],
                "at_bat_number": [1, 1],
                "pitch_number": [2, 1],
                "batter": [1001, 1001],
                "pitcher": [2101, 2101],
                "stand": ["L", "L"],
                "p_throws": ["R", "R"],
                "description": ["hit_into_play", "swinging_strike"],
                "launch_speed": [101.2, None],
                "unused_column": [0, 0],
            }
        )

    def leaderboard(http, name, year):
        if name == "park_factors":
            raise ValueError("page changed")
        return pd.DataFrame({"player_id": [1101], "Value": [3]}), f"test:{name}"

    monkeypatch.setattr(savant, "fetch_statcast_day", statcast_day)
    monkeypatch.setattr(savant, "fetch_leaderboard", leaderboard)
    monkeypatch.setattr(
        ids,
        "fetch_register",
        lambda: pd.DataFrame({"key_fangraphs": [77, -1], "key_mlbam": [2002, 5]}),
    )


@pytest.fixture
def pipe(tmp_path, fake_savant):
    return Pipeline(
        season=2026,
        calendar=load_calendar(2026),
        http=FakeHttp(),
        raw=SnapshotStore(tmp_path / "raw"),
        clean_root=tmp_path / "clean",
    )


def test_parse_schedule_keeps_the_played_listing_of_a_postponed_game():
    games = statsapi.parse_schedule(WEEK1)
    assert [g["game_pk"] for g in games] == [1, 2, 3]
    made_up = games[2]
    assert made_up["official_date"] == date(2026, 3, 28)
    assert made_up["final"] and made_up["home_score"] == 4


def test_parse_boxscore_lines():
    game = {"game_pk": 1, "official_date": date(2026, 3, 25), "venue_id": 2395}
    batting, pitching = statsapi.parse_boxscore(boxscore(NYY, SF), game)
    assert len(batting) == 5 and len(pitching) == 3
    lead = batting[0]
    assert (lead["mlbam_id"], lead["team_id"], lead["opponent_id"]) == (1001, NYY, SF)
    assert lead["started"] and lead["batting_order"] == 1 and not lead["home"]
    ph = next(b for b in batting if b["mlbam_id"] == 1003)
    assert not ph["started"] and ph["positions"] == "PH,LF"
    reliever = next(p for p in pitching if p["mlbam_id"] == 1102)
    assert reliever["outs"] == 7  # 2.1 IP
    assert reliever["appearance"] == 2 and reliever["gs"] == 0


def test_calendar_openers_fold_into_week_one():
    cal = load_calendar(2025)
    weeks = sim_weeks(cal)
    assert weeks[0].start == date(2025, 3, 24)
    assert weeks[0].game_days[0] == date(2025, 3, 27)  # no NPL games on Tokyo days
    assert week_for_date(weeks, date(2025, 3, 18)).number == 1
    assert week_for_date(weeks, date(2025, 9, 28)) is weeks[-1]
    assert week_for_date(weeks, date(2025, 10, 1)) is None
    assert sim_weeks(load_calendar(2026))[0].game_days[0] == date(2026, 3, 26)


def test_derive_and_compare_calendar():
    full_slate = [sched_game(100 + i, "2026-03-26", NYY, SF) for i in range(14)]
    late = [sched_game(200 + i, "2026-03-28", NYY, SF) for i in range(15)]
    raw = schedule(sched_game(1, "2026-03-25", NYY, SF), *full_slate, *late)
    actual = calendar_check.derive_calendar(raw)
    assert actual.openers == [date(2026, 3, 25)]
    assert actual.opening_day == date(2026, 3, 26)
    assert actual.final_day == date(2026, 3, 28)
    assert actual.off_days == [date(2026, 3, 27)]
    config = load_calendar(2026)
    problems = calendar_check.compare(config, actual)
    assert any(p.startswith("final_day") for p in problems)
    assert not any(p.startswith("opening_day") or p.startswith("openers") for p in problems)


def test_run_week_writes_clean_tables_and_freezes(pipe):
    week = pipe.week(1)
    pipe.run_week(week)
    out = pipe.week_dir(week)
    games = pd.read_parquet(out / "games.parquet")
    assert list(games["game_pk"]) == [1, 2, 3]  # Opening Night counts toward week 1
    batting = pd.read_parquet(out / "batting.parquet")
    assert len(batting) == 15 and batting["mlbam_id"].dtype == "Int64"
    totals = pd.read_parquet(out / "league_totals.parquet").iloc[0]
    assert totals["games"] == 3 and totals["pa"] == 3 * 18
    assert totals["r_per_g"] == pytest.approx(3 * 5 / 6)
    tx = pd.read_parquet(out / "transactions.parquet")
    assert list(tx["mlbam_id"]) == [1003]  # row without a person dropped

    pitches = pd.read_parquet(out / "pitches.parquet")
    assert "unused_column" not in pitches
    assert list(pitches["pitch_number"][:2]) == [1, 2]
    pulled_days = pipe.raw.keys("savant", "statcast", 2026)
    assert pulled_days[0] == "2026-03-25" and pulled_days[-1] == "2026-03-29"

    players = pd.read_parquet(pipe.clean_root / "2026" / "players.parquet")
    assert players.set_index("mlbam_id").loc[1001, "bats"] == "L"

    boards = pipe.asof_dir(date(2026, 3, 29))
    assert (boards / "oaa.parquet").exists()
    assert not (boards / "park_factors.parquet").exists()  # failed board skipped
    fg = pd.read_parquet(boards / "fg_pitching.parquet")
    assert list(fg["mlbam_id"]) == [1101] and fg["sp_stuff"][0] == 112
    proj = pd.read_parquet(boards / "fg_projections_steamer_batting.parquet")
    assert list(proj["mlbam_id"]) == [2002]  # filled from the Chadwick register

    frozen = json.loads((out / "FROZEN.json").read_text())
    assert frozen["leaderboards_as_of"] == "2026-03-29"
    assert set(frozen["tables"]) == {
        "games",
        "batting",
        "pitching",
        "pitches",
        "transactions",
        "league_totals",
    }
    with pytest.raises(FrozenWeekError):
        pipe.pull_week(week)


def test_boxscores_are_fetched_once_and_normalize_is_reproducible(pipe):
    week = pipe.week(1)
    pipe.pull_week(week)
    pipe.pull_week(week)
    box_calls = [u for u, _ in pipe.http.calls if "/boxscore" in u]
    assert len(box_calls) == 3
    first = pipe.normalize_week(week)
    second = pipe.normalize_week(week)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
    manifest = (pipe.raw.root / "manifest.jsonl").read_text().splitlines()
    assert all(len(json.loads(line)["sha256"]) == 64 for line in manifest)


def test_lineup_lock_is_an_hour_before_the_weeks_first_pitch():
    games = pd.DataFrame(statsapi.parse_schedule(WEEK1))
    locks = lineup_locks(games, sim_weeks(load_calendar(2026)))
    assert locks == {1: datetime(2026, 3, 25, 22, 5, tzinfo=UTC)}  # Opening Night


def test_lineup_lock_ignores_openers_before_week_one():
    tokyo = sched_game(1, "2025-03-18", LAD, CHC, start="T10:10:00Z")
    opener = sched_game(2, "2025-03-27", NYY, SF, start="T17:05:00Z")
    games = pd.DataFrame(statsapi.parse_schedule(schedule(tokyo, opener)))
    locks = lineup_locks(games, sim_weeks(load_calendar(2025)))
    assert locks == {1: datetime(2025, 3, 27, 16, 5, tzinfo=UTC)}


def test_rates():
    r = rates(
        {"pa": 100, "ab": 90, "h": 25, "hr": 3, "so": 22, "sf": 1, "bb": 8, "r": 9, "games": 1}
    )
    assert r["k_pct"] == 0.22 and r["r_per_g"] == 4.5
    assert r["babip"] == pytest.approx(22 / 66)


def test_fangraphs_clean_drops_and_reports_unmatched():
    rows = [{"xMLBAMID": 1, "playerid": "a"}, {"xMLBAMID": None, "playerid": "b"}]
    out, unmatched = fangraphs.clean(rows, 2025, date(2025, 9, 28))
    assert list(out["mlbam_id"]) == [1]
    assert list(unmatched["playerid"]) == ["b"]


def test_park_factor_page_parser():
    html = '<script>var foo = 1; var data = [{"venue_id": "2395", "index_hr": "88"}];</script>'
    frame = savant.parse_park_factors_page(html)
    assert frame.loc[0, "index_hr"] == "88"
    with pytest.raises(ValueError):
        savant.parse_park_factors_page("<html></html>")


def test_sync_players_and_lineup_locks():
    engine = create_engine(os.environ.get("DATABASE_URL", "sqlite://"))
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as s:
            s.add(Season(year=2026))
            s.add(Player(mlbam_id=1001, name="A", position="RF"))
            s.add(Player(mlbam_id=9999, name="B", position="C"))
            s.add(
                SimWeek(
                    season_year=2026,
                    number=1,
                    start_date=date(2026, 3, 23),
                    end_date=date(2026, 3, 29),
                )
            )
            s.flush()
            players = pd.DataFrame(statsapi.parse_people(people([1001])))
            assert sync_players(s, players) == 1
            assert sync_players(s, players) == 0
            lock = datetime(2026, 3, 26, 17, 20, tzinfo=UTC)
            assert set_lineup_locks(s, 2026, {1: lock}) == 1
            a = s.query(Player).filter_by(mlbam_id=1001).one()
            assert (a.bats, a.throws) == ("L", "R")
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_framing_reads_the_new_leaderboard_csv():
    class BomCsv:
        def get(self, url, params=None):
            assert "/leaderboard/catcher-framing" in url and "minPitches=1" in url
            return '﻿"id","name","pitches","rv_tot"\n669221,"Raleigh, Cal",4000,12\n'.encode()

    raw, _ = savant.fetch_leaderboard(BomCsv(), "framing", 2025)
    out = savant.clean_leaderboard("framing", raw, 2025, date(2025, 9, 28))
    assert list(out["mlbam_id"]) == [669221] and out.loc[0, "rv_tot"] == 12


def test_showcase_game_on_a_config_off_day_is_accepted():
    slate = [
        sched_game(100 + i, d, NYY, SF) for d in ("2026-07-12", "2026-07-17") for i in range(9)
    ]
    raw = schedule(*slate, sched_game(1, "2026-07-16", NYY, SF))
    actual = calendar_check.derive_calendar(raw)
    config = actual.model_copy(update={"off_days": [date(2026, 7, d) for d in (13, 14, 15, 16)]})
    counts = calendar_check.games_per_date(raw)
    assert calendar_check.compare(config, actual, counts) == []
    config = actual.model_copy(update={"off_days": [date(2026, 7, d) for d in (13, 14, 15)]})
    assert calendar_check.compare(config, actual, counts) == []
    config = actual.model_copy(update={"off_days": [date(2026, 7, d) for d in (13, 14)]})
    assert calendar_check.compare(config, actual, counts)  # Jul 15 really is off
