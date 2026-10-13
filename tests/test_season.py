from datetime import date

from npl_sim.cli import main
from npl_sim.rosters import eligibility
from npl_sim.rosters.sources import players_from_dump
from npl_sim.schedule.calendar import Week
from npl_sim.schedule.matchups import Series
from npl_sim.schedule.packer import ScheduledSeries
from npl_sim.season import default_cards, sim_season, week_games

TEAMS = ("Midwest Ducksnorts", "Hill Valley McFlys", "Frozen Ropes", "Chicago Orphans")


def _rosters():
    out = {}
    for p in players_from_dump("data/rosters/npl-2026.json"):
        if p.team in TEAMS and p.is_active_mlb and p.mlb_id is not None:
            out.setdefault(p.team, []).append(p)
    return out


def _two_weeks():
    """Two MLB-style weeks: Tue-Thu and Fri-Sun series, then Mon-Wed,
    Thu-Fri and Sat-Sun."""
    weeks = [
        Week(1, date(2026, 4, 6), tuple(date(2026, 4, d) for d in range(7, 13))),
        Week(2, date(2026, 4, 13), tuple(date(2026, 4, d) for d in range(13, 20))),
    ]
    a, b, c, d = TEAMS

    def series(home, away, week, first, n):
        dates = tuple(date(2026, 4, first + i) for i in range(n))
        return ScheduledSeries(Series(home, away, n, "league"), week, dates)

    schedule = [
        series(a, b, 1, 7, 3), series(c, d, 1, 7, 3),
        series(a, c, 1, 10, 3), series(b, d, 1, 10, 3),
        series(d, a, 2, 13, 3), series(b, c, 2, 13, 3),
        series(c, a, 2, 16, 2), series(d, b, 2, 16, 2),
        series(b, a, 2, 18, 2), series(c, d, 2, 18, 2),
    ]  # fmt: skip
    return weeks, schedule


def test_week_games_are_days_of_the_week():
    weeks, schedule = _two_weeks()
    games = week_games(schedule, weeks[1])
    assert len(games) == 14 and games[0][0] == 0 and games[-1][0] == 6
    assert len({g[3] for g in games}) == sum(i.week == weeks[1].number for i in schedule)


def test_season_carries_starts_across_weeks():
    weeks, schedule = _two_weeks()
    rosters = _rosters()
    elig = eligibility.load(2026)
    cards = default_cards(rosters, elig)
    # The rotation is the five arms with the most MLB starts.
    ducks = cards["Midwest Ducksnorts"]
    gs = [elig.players[pid].pitching[2026]["gs"] for pid in ducks.rotation]
    assert min(gs) >= 20
    result = sim_season(cards, rosters, elig, schedule, weeks, seed=4)
    assert len(result.games) == 26
    standings = result.standings()
    assert sum(w for _, w, *_ in standings) == 26
    assert set(result.carry) == set(TEAMS)


def test_sim_season_command(tmp_path):
    out = tmp_path / "games.csv"
    assert main(["sim-season", "--csv", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert len(lines) == 1 + 1944
