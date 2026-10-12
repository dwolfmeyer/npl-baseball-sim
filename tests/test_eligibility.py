import pandas as pd

from npl_sim.eligibility import (
    eligibility_table,
    eligible_positions,
    games_by_position,
    load_season_games,
    qualifying,
)


def test_twenty_games_last_season_or_ten_this_season():
    assert qualifying({"SS": 20}, {}) == {"SS"}
    assert qualifying({"SS": 19}, {}) == set()
    assert qualifying({}, {"2B": 10}) == {"2B"}
    assert qualifying({}, {"2B": 9}) == set()
    assert qualifying({"SS": 130, "3B": 21}, {"2B": 12, "1B": 3}) == {"SS", "3B", "2B"}


def test_outfield_games_add_up_to_the_most_played_spot():
    assert qualifying({"LF": 8, "CF": 2, "RF": 12}, {}) == {"RF"}
    assert qualifying({"LF": 25, "RF": 15}, {}) == {"LF"}  # already qualified at LF


def test_dh_always_and_listed_position_only_without_mlb_games():
    assert eligible_positions({"1B": 40}, {}, listed="OF") == {"1B", "DH"}
    assert eligible_positions({}, {}, listed="OF") == {"LF", "CF", "RF", "DH"}
    assert eligible_positions(None, None, listed="C") == {"C", "DH"}
    assert eligible_positions(None, None, listed=None) == {"DH"}


def test_games_by_position_counts_each_game_once_and_skips_ph_dh():
    batting = pd.DataFrame(
        [
            {"mlbam_id": 1, "game_pk": 10, "positions": "SS"},
            {"mlbam_id": 1, "game_pk": 11, "positions": "SS,2B"},
            {"mlbam_id": 1, "game_pk": 11, "positions": "SS,2B"},  # duplicate row
            {"mlbam_id": 2, "game_pk": 10, "positions": "PH,LF"},
            {"mlbam_id": 3, "game_pk": 10, "positions": "DH"},
            {"mlbam_id": 4, "game_pk": 10, "positions": None},
        ]
    )
    games = games_by_position(batting)
    assert games[1] == {"SS": 2, "2B": 1}
    assert games[2] == {"LF": 1}
    assert 3 not in games and 4 not in games


def test_load_season_games_reads_weekly_tables(tmp_path):
    for week, rows in ((1, [(1, 100, "C")] * 1), (2, [(1, 101, "C"), (1, 102, "1B")])):
        d = tmp_path / "2026" / f"week-{week:02d}"
        d.mkdir(parents=True)
        pd.DataFrame(rows, columns=["mlbam_id", "game_pk", "positions"]).to_parquet(
            d / "batting.parquet"
        )
    assert load_season_games(tmp_path, 2026) == {1: {"C": 2, "1B": 1}}
    assert load_season_games(tmp_path, 2026, through_week=1) == {1: {"C": 1}}
    assert load_season_games(tmp_path, 2025) == {}


def test_eligibility_table():
    table = eligibility_table([(1, "SS"), (2, "OF")], {1: {"SS": 100}}, {})
    assert table == {1: {"SS", "DH"}, 2: {"LF", "CF", "RF", "DH"}}
