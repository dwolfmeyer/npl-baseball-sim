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
    expanded_records,
    format_gb,
    format_wcgb,
    league_standings,
    wild_card_race,
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


def test_one_run_extra_inning_and_opponent_records():
    stats, boxes = played(12)
    a = stats.teams["A"]
    assert a.vs["B"] == [a.w, a.l]
    one_run = [bx for bx in boxes if abs(bx.runs["A"] - bx.runs["B"]) == 1]
    assert sum(a.one_run) == len(one_run)
    assert a.one_run[0] == sum(bx.runs["A"] > bx.runs["B"] for bx in one_run)
    extras = [bx for bx in boxes if len(bx.line["A"]) > 9]
    assert sum(a.extra) == len(extras)


def test_records_survive_json_and_old_saves_still_load():
    stats, _ = played(4)
    back = SeasonStats.from_json(stats.to_json())
    assert back.teams["A"].vs == stats.teams["A"].vs
    assert back.teams["A"].one_run == stats.teams["A"].one_run
    old = stats.to_json()
    for row in old["teams"].values():  # a season saved before these fields
        for key in ("vs", "one_run", "extra"):
            del row[key]
    assert SeasonStats.from_json(old).teams["A"].one_run == [0, 0]


def test_expected_wins_follow_runs():
    assert TeamRecord().expected_w == 0.0
    even = TeamRecord(w=5, l=5, rs=40, ra=40)
    assert abs(even.expected_w - 5) < 1e-9
    strong = TeamRecord(w=5, l=5, rs=60, ra=30)
    assert strong.expected_w > 7


def test_wild_card_race_seeds_and_games_behind_the_cut():
    alignment = load_league_file().for_season(2026)
    stats = SeasonStats()
    al = [t for lg in alignment.leagues[:1] for d in lg.divisions for t in d.teams]
    # Teams win 12 - i games of 20, so the AL order is the listed order of `al`.
    for i, team in enumerate(al):
        stats.teams[team] = TeamRecord(w=20 - i, l=i, rs=100 - i, ra=50)
    (name, rows), _ = wild_card_race(stats, alignment)
    assert name == "AL" and len(rows) == 12
    winners = [r for r in rows if r.wcgb is None]
    assert len(winners) == 3 and [r.seed for r in winners] == [1, 2, 3]
    others = rows[3:]
    assert [r.seed for r in others[:2]] == [4, 5] and others[2].seed is None
    cut = others[1]
    assert cut.wcgb == 0 and others[0].wcgb < 0 and others[2].wcgb > 0
    assert format_wcgb(cut.wcgb) == "-" and format_wcgb(None) == "-"
    assert format_wcgb(-1.5) == "+1.5" and format_wcgb(2.0) == "2"
    assert (
        wild_card_race(stats, alignment.model_copy(update={"wild_cards": 0}))[0][1][3].seed is None
    )


def test_expanded_records_split_by_division_league_and_interleague():
    alignment = load_league_file().for_season(2026)
    ducks, bulldog = "Midwest Ducksnorts", "Los Angeles Bulldog"  # same division
    kodiaks = "Chicago Kodiaks"  # same division as the Ducksnorts
    orphans = "Chicago Orphans"  # AL Oil Can
    ropes = "Frozen Ropes"  # NL
    stats = SeasonStats()
    stats.teams[ducks] = TeamRecord(
        w=3,
        l=3,
        vs={bulldog: [1, 0], kodiaks: [0, 1], orphans: [1, 1], ropes: [1, 1]},
        one_run=[2, 1],
        extra=[1, 0],
    )
    stats.teams[bulldog] = TeamRecord(w=1, l=1)  # .500
    stats.teams[kodiaks] = TeamRecord(w=2, l=0)
    stats.teams[orphans] = TeamRecord(w=0, l=2)
    stats.teams[ropes] = TeamRecord(w=1, l=0)
    row = expanded_records(stats, alignment)[ducks]
    assert row["Div"] == "1-1" and row["League"] == "1-1" and row["Inter"] == "1-1"
    assert row["1-Run"] == "2-1" and row["Extra"] == "1-0"
    # Bulldog (.500), Kodiaks and Ropes are at or over .500; Orphans are under.
    assert row[">=.500"] == "2-2" and row["<.500"] == "1-1"
