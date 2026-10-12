from collections import Counter
from datetime import date

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from npl_sim.db.models import Base, Game, SimWeek
from npl_sim.db.seed import sync_alignment
from npl_sim.league import load_league_file
from npl_sim.schedule.calendar import load_calendar, sim_weeks
from npl_sim.schedule.matchups import build_series, split_games
from npl_sim.schedule.packer import pack
from npl_sim.schedule.store import save_schedule
from npl_sim.schedule.validate import validate

from .test_db import URL

ALIGNMENT = load_league_file().for_season(2026)


@pytest.fixture(scope="module")
def season_2026():
    weeks = sim_weeks(load_calendar(2026))
    return weeks, pack(build_series(ALIGNMENT, 2026), weeks, seed=1)


def test_split_games():
    assert split_games(2) == [2]
    assert split_games(4) == [2, 2]
    assert split_games(5) == [3, 2]
    assert split_games(8) == [3, 3, 2]
    assert split_games(9) == [3, 3, 3]


@pytest.mark.parametrize("season", [2026, 2027, 2028])
def test_every_team_plays_162_with_81_at_home(season):
    games, home = Counter(), Counter()
    for s in build_series(ALIGNMENT, season):
        games[s.home] += s.games
        games[s.away] += s.games
        home[s.home] += s.games
    assert len(games) == 24
    assert set(games.values()) == {162}
    assert set(home.values()) == {81}


def test_three_game_interleague_opponents_rotate():
    """Over six seasons each AL team meets every NL team once in a 3-game set."""
    for team in ALIGNMENT.leagues[0].divisions[0].teams:
        opponents = []
        for season in range(2026, 2032):
            opponents += [
                s.pair - {team}
                for s in build_series(ALIGNMENT, season)
                if s.kind == "interleague" and s.games == 3 and team in s.pair
            ]
        assert len(opponents) == 12
        assert len({next(iter(o)) for o in opponents}) == 12


def test_calendar_2026_weeks():
    weeks = sim_weeks(load_calendar(2026))
    assert len(weeks) == 27
    assert weeks[0].start == date(2026, 3, 23)
    assert weeks[0].game_days[0] == date(2026, 3, 26)
    assert weeks[-1].game_days[-1] == date(2026, 9, 27)
    assert all(w.start.weekday() == 0 for w in weeks)
    all_star = next(w for w in weeks if w.start == date(2026, 7, 13))
    assert all_star.game_days == (date(2026, 7, 17), date(2026, 7, 18), date(2026, 7, 19))


def test_packed_2026_schedule_is_valid(season_2026):
    weeks, schedule = season_2026
    assert validate(schedule, ALIGNMENT, weeks) == []


def test_weekly_load_and_no_back_to_back_opponents(season_2026):
    weeks, schedule = season_2026
    per_week = Counter()
    pair_weeks = {}
    for item in schedule:
        for team in (item.series.home, item.series.away):
            per_week[(team, item.week)] += item.series.games
        pair_weeks.setdefault(item.series.pair, []).append(item.week)
    full = {w.number for w in weeks if len(w.game_days) == 7}
    assert all(per_week[(t, w)] >= 5 for (t, w) in per_week if w in full)
    assert all(n <= 7 for n in per_week.values())
    for ws in pair_weeks.values():
        ws.sort()
        assert all(b - a >= 2 for a, b in zip(ws, ws[1:], strict=False))


def test_validator_catches_double_booking_and_wrong_counts(season_2026):
    weeks, schedule = season_2026
    extra = schedule[0]
    problems = validate(schedule + [extra], ALIGNMENT, weeks)
    assert any("plays 2 games on" in p for p in problems)
    assert any("not 162" in p for p in problems)


def test_save_schedule_writes_weeks_and_games(season_2026):
    weeks, schedule = season_2026
    engine = create_engine(URL)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as session:
            sync_alignment(session, load_league_file(), 2026)
            assert save_schedule(session, 2026, weeks, schedule) == 24 * 162 // 2
            assert session.scalar(select(func.count()).select_from(SimWeek)) == 27
            assert session.scalar(select(func.count()).select_from(Game)) == 1944
            with pytest.raises(RuntimeError, match="already has"):
                save_schedule(session, 2026, weeks, schedule)
            session.rollback()
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_mlb_style_series_off_days_and_road_trips(season_2026):
    """Dave (2026-10-12): 2- and 3-game series with off and travel days, so
    starters on four days' rest and rested relievers line up."""
    _, schedule = season_2026
    assert {item.series.games for item in schedule} == {2, 3}
    days: dict[str, list] = {}
    homes: dict[str, list[bool]] = {}
    for item in sorted(schedule, key=lambda x: x.dates[0]):
        for team in (item.series.home, item.series.away):
            days.setdefault(team, []).extend(item.dates)
            homes.setdefault(team, []).append(team == item.series.home)
    for team, played in days.items():
        streak = longest = 1
        for a, b in zip(played, played[1:], strict=False):
            streak = streak + 1 if (b - a).days == 1 else 1
            longest = max(longest, streak)
        assert longest <= 20  # MLB's limit
        off = (played[-1] - played[0]).days + 1 - len(played)
        assert 18 <= off <= 26  # All-Star break included
        run = 1
        for a, b in zip(homes[team], homes[team][1:], strict=False):
            run = run + 1 if a == b else 1
            assert run <= 4  # no homestand or road trip over 4 series
