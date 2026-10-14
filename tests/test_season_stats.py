from datetime import date, timedelta

from npl_sim.engine.box import box_score
from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.game import simulate_game
from npl_sim.engine.sample import average_team
from npl_sim.league import load_league_file
from npl_sim.season_stats import (
    SeasonStats,
    TeamRecord,
    division_standings,
    format_gb,
    league_standings,
)


def played(n: int = 6) -> tuple[SeasonStats, list]:
    """n games between two teams, alternating the home park."""
    stats, boxes = SeasonStats(), []
    for i in range(n):
        away, home = ("B", "A") if i % 2 else ("A", "B")
        box = box_score(simulate_game(average_team(away), average_team(home), DEFAULT_ENV, i))
        stats.add_game(date(2026, 4, 1) + timedelta(days=i), box)
        boxes.append(box)
    return stats, boxes


def test_records_follow_the_box_scores():
    stats, boxes = played()
    a, b = stats.teams["A"], stats.teams["B"]
    assert a.w + a.l == b.w + b.l == 6 and a.w == b.l and a.l == b.w
    assert a.rs == b.ra and a.ra == b.rs == sum(bx.runs["B"] for bx in boxes)
    assert a.home_w + a.home_l == 3 == a.away_w + a.away_l
    assert a.home_w + a.away_w == a.w
    assert stats.games == 6 and stats.last_date == date(2026, 4, 6)


def test_last_ten_and_streak():
    rec = TeamRecord(results=[True] * 4 + [False] * 9 + [True, True])
    assert rec.last10 == (2, 8) and rec.streak == "W2"
    assert TeamRecord(results=[True, False, False, False]).streak == "L3"
    assert TeamRecord().streak == "-" and TeamRecord().pct == 0.0


def test_player_totals_add_up_to_the_games():
    stats, boxes = played()
    for team in ("A", "B"):
        mine = [p for p in stats.batters.values() if p.team == team]
        assert sum(p.h for p in mine) == sum(bx.hits[team] for bx in boxes)
        assert sum(p.r for p in mine) == stats.teams[team].rs
        arms = [p for p in stats.pitchers.values() if p.team == team]
        assert sum(p.gs for p in arms) == 6  # one starter a game
        assert sum(p.outs for p in arms) >= 6 * 24  # a home win skips the 9th
        assert sum(p.r for p in arms) == stats.teams[team].ra
    # One win and one loss is credited per game.
    assert sum(p.w for p in stats.pitchers.values()) == 6
    assert sum(p.l for p in stats.pitchers.values()) == 6


def test_slash_line_and_pitching_rates():
    stats, _ = played()
    hitter = max(stats.batters.values(), key=lambda p: p.ab)
    assert 0 <= hitter.avg <= hitter.obp <= 1 and hitter.ops == hitter.obp + hitter.slg
    arm = max(stats.pitchers.values(), key=lambda p: p.outs)
    assert arm.era == 27 * arm.er / arm.outs and arm.whip >= 0 and arm.ip.count(".") == 1


def test_json_round_trip_keeps_everything():
    stats, _ = played()
    again = SeasonStats.from_json(stats.to_json())
    assert again.to_json() == stats.to_json()
    assert again.teams["A"].last10 == stats.teams["A"].last10
    again.add_game(
        date(2026, 4, 7),
        box_score(simulate_game(average_team("A"), average_team("B"), DEFAULT_ENV, 99)),
    )
    assert again.games == 7


def test_standings_games_back_and_order():
    alignment = load_league_file().for_season(2026)
    div = alignment.leagues[0].divisions[0]
    first, second, third, fourth = div.teams
    stats = SeasonStats()
    stats.teams[first] = TeamRecord(w=10, l=4, rs=60, ra=40)
    stats.teams[second] = TeamRecord(w=8, l=6, rs=50, ra=45)
    stats.teams[third] = TeamRecord(w=8, l=6, rs=55, ra=45)  # same record, better run diff
    rows = {(lg, d): r for lg, d, r in division_standings(stats, alignment)}
    group = rows[(alignment.leagues[0].name, div.name)]
    assert [r.team for r in group] == [first, third, second, fourth]
    assert [r.gb for r in group[:3]] == [0, 2, 2] and group[3].gb == 3  # an unplayed team is 3 back
    assert format_gb(0) == "-" and format_gb(2.5) == "2.5"
    leagues = league_standings(stats, alignment)
    assert len(leagues) == 2 and len(leagues[0][1]) == 12 and leagues[0][1][0].team == first
