from collections import Counter

from npl_sim.engine.box import box_score
from npl_sim.reporting.recap import Recap, _game_notes, team_recap, week_recaps
from npl_sim.reporting.week import render_week
from npl_sim.rosters import eligibility
from npl_sim.simweek import round_robin, sim_week
from npl_sim.web import lineup

from .test_simweek import rosters


def sim():
    ro = rosters()
    elig = eligibility.load(2026)
    teams = sorted(ro)
    cards = {t: lineup.default_card(ro[t], elig) for t in teams}
    return teams, sim_week(cards, ro, elig, round_robin(teams), None, seed=3)


def test_every_team_gets_a_recap_that_matches_its_box_scores():
    teams, result = sim()
    recaps = week_recaps(result)
    assert set(recaps) == set(teams)
    wins = Counter(box_score(g["events"]).winner for g in result.games)
    for team, recap in recaps.items():
        assert isinstance(recap, Recap) and recap.team == team
        assert recap.record[0] == wins[team]
        assert recap.headline.startswith(f"{team} went {recap.record[0]}-{recap.record[1]}")
        assert any(line.startswith("Best bat:") for line in recap.lines)
        assert any(line.startswith("Best start:") for line in recap.lines)
    order = [r.record[0] for r in recaps.values()]
    assert order == sorted(order, reverse=True)  # best record first


def test_runs_in_the_headline_add_up():
    _, result = sim()
    team = result.games[0]["away"]
    mine = [g for g in result.games if team in (g["away"], g["home"])]
    boxes = [box_score(g["events"]) for g in mine]
    scored = sum(b.runs[team] for b in boxes)
    allowed = sum(sum(b.runs.values()) - b.runs[team] for b in boxes)
    assert f"scoring {scored} and allowing {allowed}" in team_recap(team, mine).headline


def test_game_notes_pick_walk_offs_extras_and_blowouts():
    _, result = sim()
    seen = []
    for g in result.games:
        box = box_score(g["events"])
        for team in (box.away, box.home):
            seen += _game_notes([g], [box], team)
    assert all(len(n) < 80 for n in seen)
    # one-run games, extra innings and blowouts are the kinds of game worth a line
    assert any("one-run" in n or "innings" in n or "Walked off" in n or " at " in n for n in seen)


def test_a_team_without_games_says_so():
    assert team_recap("Nobody", []).headline == "Nobody did not play this week."


def test_week_page_carries_the_recaps():
    teams, result = sim()
    page = render_week(result, "Test week")
    assert "<h2" in page and "Recaps" in page
    for team in teams:
        assert f"{team} went" in page
