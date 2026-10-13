"""Bunting, stealing and baserunning settings on the lineup card."""

import pytest
from fastapi.testclient import TestClient

from npl_sim.engine.manager import BUNT_STRATEGIES, NO_BUNTING, Baserunning, Bunting
from npl_sim.web.app import create_app
from npl_sim.web.lineup import LineupCard, RunningPlan, default_card, team_card
from npl_sim.web.store import Store

from .test_web import DUCKS, ELIG, PLAYERS, form_for, roster
from .test_web_manager import _lists, pinch_form


@pytest.fixture
def client(tmp_path):
    app = create_app(store=Store(tmp_path), players=PLAYERS, elig=ELIG, clean_root=tmp_path / "c")
    c = TestClient(app)
    c.app_store_root = tmp_path
    return c


def running_form(card: LineupCard) -> dict:
    data = pinch_form(card)
    run = card.running
    hitters = [pid for pid, _ in card.orders["rhp"]]
    data.update(
        run_form="1",
        bunting=run.bunting,
        steal=str(run.steal),
        extra_base=str(run.extra_base),
        steal_listed=[str(i) for i in hitters],
        steal_ok=[str(i) for i in hitters if i not in run.no_steal],
    )
    return data


def test_defaults_bunt_late_and_close_and_run_normally():
    r = roster(DUCKS)
    tc = team_card(DUCKS, "DUC", default_card(r, ELIG), r, 0, "R", ELIG)
    assert tc.bunt == Bunting() and tc.running == Baserunning()


def test_settings_round_trip_and_reach_the_engine(client):
    r = roster(DUCKS)
    card = default_card(r, ELIG)
    slow = card.orders["rhp"][3][0]
    card.running = RunningPlan(bunting="small_ball", steal=1.5, extra_base=0.5, no_steal=[slow])
    saved = client.post("/teams/midwest-ducksnorts/lineup", data=running_form(card))
    assert "Lineup card saved" in saved.text
    assert "Bunting, stealing and baserunning" in saved.text
    stored = Store(client.app_store_root).load_card(DUCKS)
    assert stored.running == card.running
    tc = team_card(DUCKS, "DUC", stored, r, 0, "R", ELIG)
    assert tc.bunt == BUNT_STRATEGIES["small_ball"][1]
    assert tc.running == Baserunning(steal=1.5, extra_base=0.5, no_steal=(str(slow),))
    # Unchecking nobody gives everyone the green light again.
    card.running = RunningPlan(bunting="never", steal=0.0)
    picked = LineupCard.from_form(_lists(running_form(card)))
    assert picked.running.no_steal == [] and picked.running.bunt_rule() == NO_BUNTING
    assert picked.running.rule().steal == 0.0


def test_older_forms_keep_the_defaults():
    card = default_card(roster(DUCKS), ELIG)
    assert LineupCard.from_form(_lists(form_for(card))).running == RunningPlan()
    assert LineupCard.from_json({}).running == RunningPlan()
