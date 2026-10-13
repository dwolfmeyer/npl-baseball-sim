"""The sim audit log (npl_sim/audit.py) and the real stat lines behind it."""

import csv
import json

import pandas as pd
import pytest

from npl_sim.audit import game_records, write_audit
from npl_sim.cli import main
from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.game import simulate_game
from npl_sim.engine.sample import average_team, average_week_team, busy_week
from npl_sim.engine.workload import MlbWeek, play_week
from npl_sim.simweek import round_robin
from npl_sim.talent import blend as tb
from npl_sim.talent.env import default_league
from npl_sim.talent.lines import stat_lines
from npl_sim.talent.load import WeekTalent


def test_auditing_changes_nothing():
    away, home = average_team("Away"), average_team("Home")
    trail: list = []
    assert simulate_game(away, home, DEFAULT_ENV, 4, trail) == simulate_game(
        away, home, DEFAULT_ENV, 4
    )
    events = simulate_game(away, home, DEFAULT_ENV, 4)
    assert len(trail) == sum(e["type"] == "pa" for e in events)
    assert all(events[d["event"]]["type"] == "pa" for d in trail)


def week_talent(teams) -> WeekTalent:
    players, lines = {}, {}
    for t in teams.values():
        for p in t.players:
            idx = {h: {**dict.fromkeys(tb.STATS, 1.0), "K": 1.2} for h in "LR"}
            players[f"batter:{p}"] = tb.Talent(p, "batter", idx, layers={"baseline": {"K": 1.2}})
            lines[f"batter:{p}"] = {"season to date": {"vs_R": {"PA": 120, "K%": 0.25}}}
    return WeekTalent(league=default_league(), players=players, lines=lines)


def test_records_cover_every_plate_appearance():
    teams = {n: average_week_team(n) for n in ("Aces", "Duds")}
    talent = week_talent(teams)
    result = play_week(
        {n: talent.team(t) for n, t in teams.items()},
        round_robin(list(teams)),
        MlbWeek(**busy_week(*teams.values())),
        DEFAULT_ENV,
        seed=2,
        audit=True,
    )
    game = result.games[0]
    records = game_records(game, 1, seed=2, talent=talent)
    kinds = [r["record"] for r in records]
    pas = [r for r in records if r["record"] == "pa"]
    assert kinds[0] == "game" and len(pas) == sum(e["type"] == "pa" for e in game["events"])
    for r in pas:
        odds = r["probabilities"]
        assert sum(odds["by_result"].values()) == pytest.approx(1.0, abs=1e-4)
        assert sum(odds["draw"].values()) == pytest.approx(1.0, abs=1e-4)
        assert len(r["pitches"]["list"]) == r["pitch_count"] and r["pitches"]["simulated"]
        if r["outcome"]["result"] in ("GO", "FO", "LO", "PO", "1B", "2B", "3B"):
            assert r["outcome"]["batted_ball"] in ("GB", "LD", "FB", "PU")
    players = {(r["role"], r["id"]): r for r in records if r["record"] == "player"}
    batter = players[("batter", pas[0]["batter"]["id"])]
    assert batter["talent"]["index"]["R"]["K"] == 1.2
    assert batter["real"]["season to date"]["vs_R"]["PA"] == 120
    assert batter["card"]  # the rates the engine played him at
    assert json.dumps(records)  # plain JSON all the way down


def test_stat_lines_from_statcast_pas():
    rows = [("K", None, "R")] * 3 + [("HR", "FB", "R"), ("1B", "GB", "L"), ("OUT", "GB", "R")]
    rows += [("BB", None, "R")]
    pa = pd.DataFrame(
        {
            "batter": 7,
            "pitcher": 9,
            "outcome": [r[0] for r in rows],
            "bip": [r[1] for r in rows],
            "p_throws": [r[2] for r in rows],
            "stand": "L",
            "xba": [None, None, None, 0.9, 0.5, 0.1, None],
            "barrel": None,
            "hard": None,
        }
    )
    lines = stat_lines([("season to date", pa)])
    line = lines["batter:7"]["season to date"]["all"]
    assert line["PA"] == 7 and line["H"] == 2 and line["HR"] == 1
    assert line["AVG"] == pytest.approx(2 / 6, abs=0.001)
    assert line["OBP"] == pytest.approx(3 / 7, abs=0.001)
    assert line["SLG"] == pytest.approx(5 / 6, abs=0.001)
    assert line["GB%"] == pytest.approx(2 / 3, abs=0.001)
    assert lines["batter:7"]["season to date"]["vs_L"]["PA"] == 1
    assert lines["pitcher:9"]["season to date"]["vs_L"]["K%"] == pytest.approx(3 / 7, abs=0.001)


def test_write_audit(tmp_path):
    teams = {n: average_week_team(n) for n in ("Aces", "Duds")}
    result = play_week(
        teams,
        round_robin(list(teams)),
        MlbWeek(**busy_week(*teams.values())),
        DEFAULT_ENV,
        seed=1,
        audit=True,
    )
    out = write_audit(result.games, tmp_path / "audit", seed=1)
    assert len(list(out.glob("game-*.jsonl"))) == len(result.games)
    with open(out / "pas.csv") as fh:
        rows = list(csv.DictReader(fh))
    total = sum(e["type"] == "pa" for g in result.games for e in g["events"])
    assert len(rows) == total and rows[0]["p_K"] and rows[0]["result"]


def test_sim_week_audit_command(tmp_path, monkeypatch):
    monkeypatch.setenv("NPL_WEB_DATA", str(tmp_path / "web"))
    assert main(["sim-week", "--seed", "2", "--audit", str(tmp_path / "audit")]) == 0
    assert len(list((tmp_path / "audit").glob("game-*.jsonl"))) == 7
    assert (tmp_path / "audit" / "pas.csv").exists()
