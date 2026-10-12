"""The web app: lineup card checks and the lineup-to-results flow."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from npl_sim.rosters.sources import players_from_dump
from npl_sim.web.app import create_app
from npl_sim.web.lineup import ACTIVE_LIMIT, LineupCard, default_card, team_card, validate
from npl_sim.web.store import Store

FIXTURE = Path(__file__).parent / "fixtures" / "npl_sample.json"
DUCKS, MCFLYS = "Midwest Ducksnorts", "Hill Valley McFlys"
PLAYERS = players_from_dump(FIXTURE)


def roster(team: str):
    return [p for p in PLAYERS if p.team == team and p.is_active_mlb and p.mlb_id]


@pytest.mark.parametrize("team", [DUCKS, MCFLYS])
def test_default_card_is_legal(team):
    card = default_card(roster(team))
    check = validate(card, roster(team))
    assert check.ok, check.errors
    assert len(card.active) == ACTIVE_LIMIT
    assert sum(role == "CL" for role in card.roles.values()) == 1


def test_validate_catches_bad_cards():
    r = roster(DUCKS)
    card = default_card(r)
    pitcher_id = card.rotation[0]
    rhp = list(card.orders["rhp"])
    rhp[0] = (pitcher_id, rhp[0][1])  # a pitcher batting leadoff
    rhp[1] = (rhp[2][0], rhp[1][1])  # someone batting twice
    rhp[3] = (rhp[3][0], rhp[4][1])  # two players at one position
    bad = LineupCard(
        active=[p.mlb_id for p in r],  # 32 active
        orders={**card.orders, "rhp": rhp},
        rotation=card.rotation[:4],
        roles={},
    )
    errors = " ".join(validate(bad, r).errors)
    assert "limit is 30" in errors
    assert "is a pitcher" in errors
    assert "bats twice" in errors
    assert "two players at" in errors
    assert "4 of 5 starters" in errors
    assert "one closer" in errors


def test_inactive_player_cannot_bat():
    r = roster(DUCKS)
    card = default_card(r)
    benched = card.orders["lhp"][0][0]
    card.active.remove(benched)
    assert any("not active" in e for e in validate(card, r).errors)


def test_team_card_uses_order_for_opposing_hand():
    r = roster(DUCKS)
    card = default_card(r)
    card.orders["lhp"] = list(reversed(card.orders["rhp"]))
    vs_l = team_card(DUCKS, "DUC", card, r, 0, "L")
    vs_r = team_card(DUCKS, "DUC", card, r, 0, "R")
    assert [h.id for h, _ in vs_l.lineup] == [h.id for h, _ in reversed(vs_r.lineup)]
    assert vs_l.starter.id == str(card.rotation[0])
    assert vs_l.lineup[0][0].name  # real names, not ids


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(store=Store(tmp_path), players=PLAYERS))


def form_for(card: LineupCard) -> dict:
    data = {"active": [str(a) for a in card.active]}
    for side, slots in card.orders.items():
        for i, (pid, pos) in enumerate(slots, 1):
            data[f"{side}_player_{i}"] = str(pid)
            data[f"{side}_pos_{i}"] = pos
    for i, pid in enumerate(card.rotation, 1):
        data[f"sp_{i}"] = str(pid)
    for pid, role in card.roles.items():
        data[f"role_{pid}"] = role
    return data


def test_lineup_card_round_trip(client):
    page = client.get("/teams/midwest-ducksnorts/lineup")
    assert page.status_code == 200 and "Batting order vs LHP" in page.text

    card = default_card(roster(DUCKS))
    card.orders["lhp"] = list(reversed(card.orders["lhp"]))
    saved = client.post("/teams/midwest-ducksnorts/lineup", data=form_for(card))
    assert saved.status_code == 200 and "Lineup card saved" in saved.text
    assert "card is legal" in saved.text

    bad = form_for(card)
    bad["sp_1"] = ""
    rejected = client.post("/teams/midwest-ducksnorts/lineup", data=bad)
    assert rejected.status_code == 422 and "4 of 5 starters" in rejected.text

    partial = client.post(
        "/teams/midwest-ducksnorts/lineup/check", data=bad, headers={"HX-Request": "true"}
    )
    assert partial.status_code == 200 and "<html" not in partial.text
    assert "4 of 5 starters" in partial.text

    assert client.get("/teams/no-such-team/lineup").status_code == 404


def test_sim_a_game(client):
    home = client.get("/")
    assert DUCKS in home.text and "No games yet" in home.text
    resp = client.post(
        "/games",
        data={"away": DUCKS, "home": MCFLYS, "away_slot": "2", "home_slot": "1", "seed": "7"},
    )
    assert resp.status_code == 200
    assert "Play-by-play" in resp.text and "Scoring plays" in resp.text
    assert "Seed 7" in resp.text
    # Real player names reach the box score.
    by_id = {p.mlb_id: p.name for p in roster(DUCKS)}
    assert all(by_id[pid] in resp.text for pid, _ in default_card(roster(DUCKS)).orders["rhp"])
    assert "at Hill Valley McFlys" in client.get("/").text
    assert client.post("/games", data={"away": DUCKS, "home": DUCKS}).status_code == 400
