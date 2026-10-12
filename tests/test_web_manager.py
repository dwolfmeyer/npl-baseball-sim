"""Manager style and pinch-hitting settings on the lineup card."""

import pytest
from fastapi.testclient import TestClient

from npl_sim.engine.manager import DEFAULT_STYLE, STYLES
from npl_sim.web.app import create_app
from npl_sim.web.lineup import LineupCard, PinchPlan, default_card, team_card
from npl_sim.web.store import Store

from .test_web import DUCKS, ELIG, PLAYERS, form_for, roster


@pytest.fixture
def client(tmp_path):
    app = create_app(store=Store(tmp_path), players=PLAYERS, elig=ELIG, clean_root=tmp_path / "c")
    c = TestClient(app)
    c.app_store_root = tmp_path
    return c


def pinch_form(card: LineupCard, style_prev: str | None = None) -> dict:
    data = form_for(card)
    ph = card.pinch
    data.update(
        style=card.style,
        style_prev=style_prev or card.style,
        ph_form="1",
        ph_inning=str(ph.from_inning),
        ph_li=str(ph.min_leverage),
        ph_gain=str(ph.min_gain),
        ph_max=str(ph.max_per_game),
        ph_never=[str(i) for i in ph.never_for],
    )
    for key, on in (("ph_on", ph.enabled), ("ph_platoon", ph.platoon_only)):
        if on:
            data[key] = "1"
    if ph.out_of_position:
        data["ph_oop"] = "1"
    return data


def test_default_card_uses_the_default_style():
    card = default_card(roster(DUCKS), ELIG)
    assert card.style == DEFAULT_STYLE
    assert card.pinch.rule() == STYLES[DEFAULT_STYLE].pinch


def test_picking_a_style_fills_hooks_and_pinch_rules():
    card = default_card(roster(DUCKS), ELIG)
    card.pinch.never_for = [card.orders["rhp"][0][0]]
    card.style = "passive"
    picked = LineupCard.from_form(_lists(pinch_form(card, style_prev=DEFAULT_STYLE)))
    hook = STYLES["passive"].hook
    assert picked.style == "passive" and not picked.pinch.enabled
    assert all(p.hook() == hook for p in picked.plans)
    assert picked.pinch.never_for == card.pinch.never_for  # the owner's list stays

    card.style = "sabermetric"
    picked = LineupCard.from_form(_lists(pinch_form(card, style_prev="passive")))
    assert picked.plans[0].tto_hook == 2 and picked.plans[0].leverage == 1.8
    assert picked.pinch.rule().from_inning == 6 and not picked.pinch.platoon_only


def test_edits_after_picking_a_style_stick(client):
    card = default_card(roster(DUCKS), ELIG)
    card.style = "aggressive"
    card.apply_style("aggressive")
    card.pinch = PinchPlan(
        enabled=True, from_inning=8, min_leverage=1.5, platoon_only=False, min_gain=0.02,
        max_per_game=1, out_of_position=True, never_for=[card.orders["rhp"][2][0]],
    )  # fmt: skip
    card.plans[0].max_runs = 5
    saved = client.post("/teams/midwest-ducksnorts/lineup", data=pinch_form(card))
    assert "Lineup card saved" in saved.text
    stored = Store(client.app_store_root).load_card(DUCKS)
    assert stored.style == "aggressive" and stored.pinch == card.pinch
    assert stored.plans[0].max_runs == 5
    assert "Manager style" in saved.text and 'value="aggressive" selected' in saved.text


def test_style_change_rerenders_the_card(client):
    card = default_card(roster(DUCKS), ELIG)
    card.style = "old_school"
    page = client.post(
        "/teams/midwest-ducksnorts/lineup/check",
        data=pinch_form(card, style_prev=DEFAULT_STYLE),
        headers={"HX-Request": "true"},
    ).text
    assert 'value="old_school" selected' in page
    assert '<option value="8" selected>8th on</option>' in page


def test_bench_and_pinch_rules_reach_the_engine():
    r = roster(DUCKS)
    card = default_card(r, ELIG)
    never = card.orders["rhp"][0][0]
    card.pinch.never_for = [never]
    tc = team_card(DUCKS, "DUC", card, r, 0, "R", ELIG)
    starting = {str(pid) for pid, _ in card.orders["rhp"]}
    bench = {h.id for h in tc.bench}
    assert bench and not bench & starting
    hitters = {str(p.mlb_id) for p in r if p.position not in ("SP", "RP", "P")}
    assert bench <= hitters
    assert tc.pinch.never_for == (str(never),) and tc.pinch.enabled


def _lists(data: dict) -> dict:
    """A form dict the way from_form reads a non-Starlette form: lists of values."""
    return {k: v if isinstance(v, list) else [v] for k, v in data.items()}
