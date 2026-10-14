"""The league block on the week page: league totals, the role split and the week's counts."""

from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.sample import average_week_team, busy_week
from npl_sim.engine.workload import MlbWeek, play_week
from npl_sim.playing_time import RealStats
from npl_sim.reporting.compare import league_compare, league_html
from npl_sim.reporting.week import render_week
from npl_sim.season_stats import BatterTotals, PitcherTotals

AWAY, HOME = average_week_team("Away"), average_week_team("Home")
GAMES = [(day, "Away", "Home") for day in range(6)]


def week():
    return play_week(
        {"Away": AWAY, "Home": HOME}, GAMES, MlbWeek(**busy_week(AWAY, HOME)), DEFAULT_ENV, seed=4
    )


def test_league_totals_add_up_the_teams():
    result = week()
    league = league_compare(result)
    assert league.facts["games"] == 6 and league.facts["team_games"] == 12
    runs = sum(g["events"][-1][s] for g in result.games for s in ("away", "home"))
    assert abs(league.facts["runs_per_team_game"] - runs / 12) < 1e-9
    rows = dict(league.hitting)
    assert rows["R"] == str(runs) and not league.mlb_hitting


def test_role_split_covers_every_out_once():
    result = week()
    league = league_compare(result)
    outs = sum(t.outs for t in league.roles.values())
    # Every out a pitcher recorded: nine innings a game for both teams, or more with extras.
    assert outs >= 6 * 2 * 27 - 6 * 3 and league.roles["Starters"].outs > 0


def test_mlb_rows_sum_the_real_stats_of_players_who_played():
    result = week()
    some_bat = next(
        p for g in result.games for e in g["events"] if e["type"] == "pa" for p in (e["batter"],)
    )
    some_arm = next(e["pitcher"] for g in result.games for e in g["events"] if e["type"] == "pa")
    real = RealStats(
        batters={some_bat: BatterTotals(pa=30, ab=25, h=10, hr=2, r=5, rbi=6, bb=4, k=5)},
        pitchers={some_arm: PitcherTotals(outs=27, h=5, r=2, er=2, bb=1, k=9, hr=1)},
    )
    league = league_compare(result, real)
    hit = dict(league.mlb_hitting)
    assert hit["HR"] == "2" and hit["AVG"] == ".400"
    assert dict(league.mlb_pitching)["ERA"] == "2.00"
    html = league_html(league)
    assert 'class="league"' in html and "Sim role" in html and "Relievers" in html
    assert 'class="mlb"' in html


def test_week_page_carries_the_league_block():
    page = render_week(week(), "Test week")
    assert 'class="league"' in page and "Bullpen games:" in page
