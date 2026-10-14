import pandas as pd

from npl_sim.engine.box import box_score
from npl_sim.playing_time import RealStats, load_real_week_stats, real_week_stats
from npl_sim.reporting.recap import week_recaps
from npl_sim.reporting.week import render_week
from npl_sim.season_stats import BatterTotals, PitcherTotals

from .test_recap import sim


def test_real_week_stats_sum_a_players_games_and_rename_strikeouts():
    batting = pd.DataFrame(
        {
            "mlbam_id": [7, 7, 8],
            "pa": [4, 5, 3],
            "ab": [4, 4, 3],
            "h": [2, 1, 0],
            "hr": [1, 0, 0],
            "so": [1, 2, 3],
        }
    )
    pitching = pd.DataFrame(
        {"mlbam_id": [9, 9], "gs": [1, 0], "outs": [18, 3], "er": [2, 0], "so": [7, 1], "w": [1, 0]}
    )
    real = real_week_stats(batting, pitching)
    assert real.batters["7"].h == 3 and real.batters["7"].hr == 1 and real.batters["7"].k == 3
    assert real.batters["8"].k == 3 and real.batters["7"].pa == 9
    assert real.pitchers["9"].outs == 21 and real.pitchers["9"].k == 8 and real.pitchers["9"].w == 1
    assert real.pitchers["9"].gs == 1
    empty = real_week_stats(pd.DataFrame(), pd.DataFrame())
    assert empty.batters == {} and empty.pitchers == {}


def test_load_real_week_stats_reads_the_weeks_clean_tables(tmp_path):
    folder = tmp_path / "2026" / "week-03"
    folder.mkdir(parents=True)
    pd.DataFrame({"mlbam_id": [1], "pa": [4], "h": [2]}).to_parquet(folder / "batting.parquet")
    pd.DataFrame({"mlbam_id": [2], "gs": [1], "outs": [18]}).to_parquet(folder / "pitching.parquet")
    real = load_real_week_stats(tmp_path, 2026, 3)
    assert real.batters["1"].h == 2 and real.pitchers["2"].outs == 18


def real_for(result) -> RealStats:
    """A made-up MLB week: every player who appeared in the sim got 2 H, 1 HR and 3 R, and
    every pitcher 6 outs and 1 ER."""
    real = RealStats()
    for g in result.games:
        box = box_score(g["events"])
        for team in (box.away, box.home):
            for pid in box.batting[team]:
                real.batters[pid] = BatterTotals(ab=4, h=2, hr=1, r=3, pa=4)
            for pid in box.pitching[team]:
                real.pitchers[pid] = PitcherTotals(outs=6, er=1, h=3, k=2)
    return real


def test_recaps_carry_the_real_weeks_lines_for_the_players_who_played():
    teams, result = sim()
    real = real_for(result)
    for team, recap in week_recaps(result, real=real).items():
        boxes = [box_score(g["events"]) for g in result.games if team in (g["away"], g["home"])]
        hitters = {pid for b in boxes for pid in b.batting[team]}
        arms = {pid for b in boxes for pid in b.pitching[team]}
        mlb_hit, mlb_pit = dict(recap.mlb_hitting), dict(recap.mlb_pitching)
        assert mlb_hit["R"] == str(3 * len(hitters)) and mlb_hit["HR"] == str(len(hitters))
        assert mlb_hit["AVG"] == ".500"
        assert mlb_pit["ER"] == str(len(arms)) and mlb_pit["IP"] == f"{2 * len(arms)}.0"
        # The sim's own lines are still there, in the same columns.
        assert [k for k, _ in recap.hitting] == [k for k, _ in recap.mlb_hitting]
        assert [k for k, _ in recap.pitching] == [k for k, _ in recap.mlb_pitching]


def test_without_a_real_week_there_are_no_mlb_lines():
    teams, result = sim()
    for recap in week_recaps(result).values():
        assert recap.mlb_hitting == [] and recap.mlb_pitching == [] and recap.hitting
    page = render_week(result, "Test week")
    assert 'class="mlb"' not in page and "MLB rows" not in page


def test_week_page_shows_sim_and_mlb_rows_under_each_recap():
    teams, result = sim()
    page = render_week(result, "Test week", real=real_for(result), real_label="MLB 2026 week 27")
    assert page.count('<tr class="sim">') == 2 * (len(teams) + 1)  # and the league block
    assert page.count('<tr class="mlb">') == 2 * (len(teams) + 1)
    assert "MLB rows: the real MLB 2026 week 27 stats of the players who played" in page
    # A player the real tables don't know adds nothing rather than failing.
    assert (
        render_week(result, "Test week", real=RealStats(), real_label="x").count('class="mlb"') == 0
    )
