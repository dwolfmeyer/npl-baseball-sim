import pytest

from npl_sim.cli import main
from npl_sim.engine.box import box_score
from npl_sim.reporting.week import render_week, standings
from npl_sim.rosters import eligibility
from npl_sim.simweek import round_robin, sim_week, typical_mlb_week
from npl_sim.web import lineup
from npl_sim.web.app import load_players


@pytest.fixture(autouse=True)
def test_teams(monkeypatch):
    """These tests are written for the Ducksnorts/McFlys/Ropes fixture, not the league."""
    monkeypatch.setenv("NPL_ROSTER_DUMP", "tests/fixtures/npl_2026_test_teams.json")


def rosters():
    out = {}
    for p in load_players():
        if p.is_active_mlb and p.mlb_id is not None:
            out.setdefault(p.team, []).append(p)
    return out


def test_round_robin_plays_series_not_single_games():
    games = round_robin(["A", "B", "C", "D"])
    series = {}
    for day, away, home, sid in games:
        series.setdefault(sid, []).append((day, away, home))
    assert len(series) == 4 and all(len(g) == 3 for g in series.values())
    for g in series.values():
        days = [d for d, _, _ in g]
        assert days == list(range(days[0], days[0] + 3))  # consecutive games
        assert len({(a, h) for _, a, h in g}) == 1  # same opponent, same park
    # Every team plays one series per block and sits Monday.
    for team in "ABCD":
        mine = [g for g in games if team in g[1:3]]
        assert len(mine) == 6 and {g[0] for g in mine} == {1, 2, 3, 4, 5, 6}
    # Odd count: one team sits each block.
    three = round_robin(["A", "B", "C"])
    assert len(three) == 6 and len({g[3] for g in three}) == 2


def test_sim_week_from_lineup_cards():
    ro = rosters()
    elig = eligibility.load(2026)
    teams = sorted(ro)
    cards = {t: lineup.default_card(ro[t], elig) for t in teams}
    result = sim_week(cards, ro, elig, round_robin(teams), None, seed=3)
    assert len(result.games) == 6
    for g in result.games:
        box = box_score(g["events"])  # renders like any other game
        assert box.runs[g["away"]] != box.runs[g["home"]]
    assert sum(w for _, w, *_ in standings(result)) == 6
    page = render_week(result, "Test week")
    assert "Standings" in page
    assert 'id="team"' in page and page.count('class="game" data-i=') == 6
    for t in teams:  # every team is a filter choice
        assert f'<option value="{t}">' in page
    assert set(result.carry) == set(teams)
    # A series is the same two teams for three games, a different starter each game.
    by_series = {}
    for (_, _, _, sid), g in zip(round_robin(teams), result.games, strict=True):
        by_series.setdefault(sid, []).append(g)
    for games in by_series.values():
        assert len({(g["away"], g["home"]) for g in games}) == 1
        for side in ("away", "home"):
            moves = [g["moves"][g[side]] for g in games]
            starters = [m.starter for m in moves if not m.bullpen_game]
            assert len(set(starters)) == len(starters)


def test_typical_week_follows_season_pace():
    ro = rosters()
    elig = eligibility.load(2026)
    week = typical_mlb_week(ro, elig, 2026, seed=1)
    everyday = [
        p
        for r in ro.values()
        for p in r
        if p.mlb_id in elig.players
        and sum(elig.players[p.mlb_id].games.get(2026, {}).values()) > 140
    ]
    assert everyday and all(week.pa[str(p.mlb_id)] >= 21 for p in everyday)


def test_sim_week_command(tmp_path, monkeypatch):
    monkeypatch.setenv("NPL_WEB_DATA", str(tmp_path / "web"))
    assert main(["sim-week", "--seed", "2", "--out", str(tmp_path / "week"), "--save"]) == 0
    assert (tmp_path / "week" / "index.html").exists()
    assert (tmp_path / "week" / "game-6.html").exists()
    assert len(list((tmp_path / "web" / "games").glob("*.json"))) == 6
