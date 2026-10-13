from collections import Counter

from npl_sim.cli import main
from npl_sim.engine.box import box_score
from npl_sim.reporting.week import render_week, standings
from npl_sim.rosters import eligibility
from npl_sim.simweek import round_robin, sim_week, typical_mlb_week
from npl_sim.web import lineup
from npl_sim.web.app import load_players


def rosters():
    out = {}
    for p in load_players():
        if p.is_active_mlb and p.mlb_id is not None:
            out.setdefault(p.team, []).append(p)
    return out


def test_round_robin_pairs_everyone_once_a_day():
    games = round_robin(["A", "B", "C"])
    assert len(games) == 7 and {d for d, _, _ in games} == set(range(7))
    pairs = Counter(frozenset((a, h)) for _, a, h in games)
    assert len(pairs) == 3  # every pair meets
    four = round_robin(["A", "B", "C", "D"])
    assert all(sum(d == day for d, _, _ in four) == 2 for day in range(7))


def test_sim_week_from_lineup_cards():
    ro = rosters()
    elig = eligibility.load(2026)
    teams = sorted(ro)
    cards = {t: lineup.default_card(ro[t], elig) for t in teams}
    result = sim_week(cards, ro, elig, round_robin(teams), None, seed=3)
    assert len(result.games) == 7
    for g in result.games:
        box = box_score(g["events"])  # renders like any other game
        assert box.runs[g["away"]] != box.runs[g["home"]]
    assert sum(w for _, w, *_ in standings(result)) == 7
    page = render_week(result, "Test week")
    assert "Standings" in page
    assert 'id="team"' in page and page.count('class="game" data-away=') == 7
    for t in teams:  # every team is a filter choice
        assert f'<option value="{t}">' in page
    assert set(result.carry) == set(teams)


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
    assert (tmp_path / "week" / "game-7.html").exists()
    assert len(list((tmp_path / "web" / "games").glob("*.json"))) == 7
