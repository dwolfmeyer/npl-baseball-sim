from npl_sim.leaders import BATTING, PITCHING, leaders, league_leaders, team_games
from npl_sim.league import load_league_file
from npl_sim.season_stats import BatterTotals, PitcherTotals, SeasonStats, TeamRecord


def stats_with(batters=(), pitchers=(), games=100) -> SeasonStats:
    s = SeasonStats()
    s.teams["A"] = TeamRecord(w=games // 2, l=games - games // 2)
    s.batters = {b.name: b for b in batters}
    s.pitchers = {p.name: p for p in pitchers}
    return s


def bat(name, team="A", pa=400, ab=360, h=100, hr=10, **kw) -> BatterTotals:
    return BatterTotals(name=name, team=team, pa=pa, ab=ab, h=h, hr=hr, **kw)


def test_counting_stats_need_no_qualifier_rate_stats_do():
    part_timer = bat("Pinch", pa=12, ab=10, h=6, hr=2)  # .600 in 12 PA
    regular = bat("Regular", h=110)
    board = leaders(stats_with([part_timer, regular]))["batting"]
    assert [r.name for r in board["AVG"]] == ["Regular"]  # 310 PA needed at 100 games
    assert [r.name for r in board["HR"]] == ["Regular", "Pinch"] or board["HR"][0].value == "10"
    assert "Pinch" in {r.name for r in board["H"]}


def test_qualifiers_scale_with_games_played():
    early = stats_with([bat("A1", pa=31, ab=28, h=12)], games=10)
    assert (
        team_games(early) == 10 and leaders(early)["batting"]["AVG"][0].name == "A1"
    )  # 31 PA is enough
    assert leaders(stats_with([bat("A1", pa=31, ab=28, h=12)], games=11))["batting"]["AVG"] == []


def test_lowest_wins_for_era_and_whip_and_ties_share_a_rank():
    arms = [
        PitcherTotals(name="Ace", team="A", outs=330, er=30, h=80, bb=20),
        PitcherTotals(name="Mid", team="A", outs=330, er=45, h=90, bb=30),
        PitcherTotals(name="Twin", team="A", outs=330, er=45, h=90, bb=30),
        PitcherTotals(name="Short", team="A", outs=60, er=0, h=2, bb=0),  # under 1 IP a game
    ]
    board = leaders(stats_with(pitchers=arms))["pitching"]
    assert [r.name for r in board["ERA"]] == ["Ace", "Mid", "Twin"]
    assert [r.rank for r in board["ERA"]] == [1, 2, 2]
    assert board["ERA"][0].value == "2.45" and board["WHIP"][0].name == "Ace"


def test_top_n_and_empty_boards():
    many = [bat(f"P{i:02d}", hr=i + 1) for i in range(20)]
    board = leaders(stats_with(many), n=5)["batting"]["HR"]
    assert [r.value for r in board] == ["20", "19", "18", "17", "16"]
    empty = leaders(SeasonStats())
    assert all(rows == [] for rows in empty["batting"].values())
    assert set(empty["batting"]) == {c.label for c in BATTING}
    assert set(empty["pitching"]) == {c.label for c in PITCHING}


def test_each_league_ranks_its_own_players():
    alignment = load_league_file().for_season(2026)
    first, second = alignment.leagues[0], alignment.leagues[1]
    team1 = first.divisions[0].teams[0]
    team2 = second.divisions[0].teams[0]
    stats = SeasonStats()
    stats.teams[team1] = TeamRecord(w=1, l=0)
    stats.batters = {
        "x": bat("Slugger", team=team1, hr=40),
        "y": bat("Other", team=team2, hr=35),
    }
    boards = league_leaders(stats, alignment)
    assert boards[first.name]["batting"]["HR"][0].name == "Slugger"
    assert [r.name for r in boards[second.name]["batting"]["HR"]] == ["Other"]
