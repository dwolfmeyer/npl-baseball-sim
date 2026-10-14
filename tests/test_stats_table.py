from npl_sim.season_stats import BatterTotals, PitcherTotals, SeasonStats, TeamRecord
from npl_sim.stats_table import Query, build, qualifies, team_totals

LEAGUES = {"A": "AL", "B": "NL"}


def stats() -> SeasonStats:
    s = SeasonStats()
    s.teams = {"A": TeamRecord(w=5, l=5), "B": TeamRecord(w=4, l=6)}
    s.batters = {
        "1": BatterTotals("Ann", "A", g=10, pa=40, ab=36, h=12, hr=3),
        "2": BatterTotals("Bo", "A", g=10, pa=40, ab=36, h=9, hr=3),
        "3": BatterTotals("Cy", "B", g=10, pa=20, ab=18, h=9, hr=1),  # 31 PA needed: not qualified
    }
    s.pitchers = {
        "4": PitcherTotals("Di", "A", g=2, gs=2, outs=30, er=5, bb=2, h=6, k=9, w=1),
        "5": PitcherTotals("Ed", "B", g=5, outs=12, er=1, h=2, bb=0, k=4, sv=2),
    }
    return s


def names(table):
    return [r.name for r in table.rows]


def test_qualified_hitters_and_all_players():
    s = stats()
    assert not qualifies(s, "hitting", s.batters["3"]) and qualifies(s, "hitting", s.batters["1"])
    assert names(build(s, Query(sort="avg"), LEAGUES)) == ["Ann", "Bo"]
    assert names(build(s, Query(sort="avg", qualified=False), LEAGUES)) == ["Cy", "Ann", "Bo"]


def test_sort_directions_ties_and_lowest_first_columns():
    s = stats()
    # HR ties between Ann and Bo fall back to name.
    assert names(build(s, Query(sort="hr", qualified=False), LEAGUES)) == ["Ann", "Bo", "Cy"]
    assert names(build(s, Query(sort="hr", descending=False, qualified=False), LEAGUES)) == [
        "Cy",
        "Ann",
        "Bo",
    ]
    # ERA sorts best (lowest) first by default, and IP sorts by outs, not text.
    pitching = Query(kind="pitching", qualified=False)
    assert names(build(s, pitching, LEAGUES)) == ["Ed", "Di"]
    assert names(build(s, Query(kind="pitching", qualified=False, sort="ip"), LEAGUES)) == [
        "Di",
        "Ed",
    ]


def test_league_and_team_filters():
    s = stats()
    q = Query(qualified=False, league="NL")
    assert names(build(s, q, LEAGUES)) == ["Cy"]
    assert names(build(s, Query(qualified=False, team="A"), LEAGUES)) == ["Ann", "Bo"]


def test_team_lines_add_the_players_up():
    s = stats()
    bats = {t.team: t for t in team_totals(s, "hitting")}
    assert bats["A"].h == 21 and bats["A"].g == 10 and bats["A"].pa == 80
    arms = {t.team: t for t in team_totals(s, "pitching")}
    assert arms["B"].sv == 2 and arms["B"].outs == 12 and arms["B"].gs == 10
    table = build(s, Query(group="teams", sort="h"), LEAGUES)
    assert names(table) == ["A", "B"] and table.total == 2


def test_paging_clamps_and_formats_cells():
    s = stats()
    table = build(s, Query(qualified=False, page=9), LEAGUES, page_size=2)
    assert (table.page, table.pages, table.total) == (2, 2, 3)
    assert table.rows[0].rank == 3
    first = build(s, Query(sort="avg", qualified=False), LEAGUES).rows[0]
    labels = [c.label for c in build(s, Query(), LEAGUES).columns]
    assert first.cells[labels.index("AVG")] == ".500"
