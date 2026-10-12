"""The web app: lineup card checks and the lineup-to-results flow."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from npl_sim.rosters.eligibility import Eligibility, PlayerRecord
from npl_sim.rosters.sources import players_from_dump
from npl_sim.web.app import create_app
from npl_sim.web.lineup import (
    ACTIVE_LIMIT,
    POSITIONS,
    DefensiveSub,
    LineupCard,
    ReliefPlan,
    StarterPlan,
    autofill,
    default_card,
    fielding,
    team_card,
    validate,
)
from npl_sim.web.store import Store

FIXTURE = Path(__file__).parent / "fixtures" / "npl_2026_test_teams.json"
DUCKS, MCFLYS = "Midwest Ducksnorts", "Hill Valley McFlys"
PLAYERS = players_from_dump(FIXTURE)
HENDERSON, SANOJA, TROUT, GOODMAN, MATTHEWS = 683002, 691594, 545361, 696100, 694728
# Made-up games by position for a few players; the rest fall back to roster positions.
# The McFlys have no third baseman on their roster, so Matthews gets games there.
ELIG = Eligibility(
    season=2026,
    players={
        HENDERSON: PlayerRecord("L", "R", {2025: {"SS": 150, "3B": 4}}),
        SANOJA: PlayerRecord("R", "R", {2025: {"LF": 8}, 2026: {"2B": 12, "SS": 9}}),
        TROUT: PlayerRecord("R", "R", {2025: {"RF": 15}, 2026: {"RF": 3}}),
        GOODMAN: PlayerRecord("R", "R", {2025: {"C": 70, "RF": 21}}),
        MATTHEWS: PlayerRecord("R", "R", {2025: {"3B": 25, "2B": 30}}),
    },
)


def roster(team: str):
    return [p for p in PLAYERS if p.team == team and p.is_active_mlb and p.mlb_id]


@pytest.mark.parametrize("team", [DUCKS, MCFLYS])
def test_default_card_is_legal(team):
    card = default_card(roster(team), ELIG)
    check = validate(card, roster(team), ELIG)
    assert check.ok, check.errors
    assert len(card.active) == ACTIVE_LIMIT
    assert sum(role == "CL" for role in card.roles.values()) == 1


def test_eligibility_rule():
    by_id = {p.mlb_id: p for p in roster(DUCKS)}
    assert ELIG.positions(HENDERSON) == {"SS"}  # 4 games at 3B last season is not enough
    assert ELIG.positions(SANOJA) == {"2B"}  # 10 this season counts; 9 at SS does not
    # 15 games in RF last season and 3 this season don't qualify: listed OF stands in.
    assert ELIG.positions(TROUT, "OF") == {"LF", "CF", "RF"} and not ELIG.qualifies(TROUT)
    assert ELIG.positions(GOODMAN) == {"C", "RF"}
    # No MLB fielding record: the listed NPL position stands in.
    assert fielding(by_id[701538], ELIG) == ({"LF", "CF", "RF"}, False)

    card = default_card(roster(DUCKS), ELIG)
    rhp = [(pid, pos) for pid, pos in card.orders["rhp"] if pid != HENDERSON]
    card.orders["rhp"] = [*rhp, (HENDERSON, "3B")]
    assert HENDERSON in card.active
    warnings = " ".join(validate(card, roster(DUCKS), ELIG).warnings)
    assert "Gunnar Henderson is out of position at 3B vs RHP (eligible: SS)" in warnings


def test_validate_catches_bad_cards():
    r = roster(DUCKS)
    card = default_card(r, ELIG)
    pitcher_id = card.rotation[0]
    rhp = list(card.orders["rhp"])
    rhp[0] = (pitcher_id, rhp[0][1])  # a pitcher batting leadoff
    rhp[1] = (rhp[2][0], rhp[1][1])  # someone batting twice
    rhp[3] = (rhp[3][0], rhp[4][1])  # two players at one position
    bad = LineupCard(
        active=[p.mlb_id for p in r],  # 32 active
        orders={**card.orders, "rhp": rhp},
        rotation=card.rotation[:4],
        plans=[StarterPlan(opener=True, piggyback=card.rotation[1]), StarterPlan(pitch_limit=140)],
        roles={},
    )
    errors = " ".join(validate(bad, r, ELIG).errors)
    assert "limit is 30" in errors
    assert "is a pitcher" in errors
    assert "bats twice" in errors
    assert "two players at" in errors
    assert "4 of 5 starters" in errors
    assert "so he can't be starter 1's piggyback" in errors
    assert "pitch limit must be 60 to 120" in errors


def test_inactive_player_cannot_bat():
    r = roster(DUCKS)
    card = default_card(r, ELIG)
    benched = card.orders["lhp"][0][0]
    card.active.remove(benched)
    assert any("not active" in e for e in validate(card, r).errors)


def test_team_card_uses_order_for_opposing_hand():
    r = roster(DUCKS)
    card = default_card(r, ELIG)
    card.orders["lhp"] = list(reversed(card.orders["rhp"]))
    vs_l = team_card(DUCKS, "DUC", card, r, 0, "L", ELIG)
    vs_r = team_card(DUCKS, "DUC", card, r, 0, "R", ELIG)
    assert [h.id for h, _ in vs_l.lineup] == [h.id for h, _ in reversed(vs_r.lineup)]
    assert vs_l.starter.id == str(card.rotation[0])
    assert vs_l.lineup[0][0].name  # real names, not ids
    bats = {h.id: h.bats for h, _ in vs_l.lineup}
    assert bats[str(HENDERSON)] == "L"  # hands come from the eligibility file


def test_opener_and_piggyback_plan():
    r = roster(DUCKS)
    card = default_card(r, ELIG)
    relievers = [pid for pid in card.roles]
    piggyback = relievers[-1]
    card.plans[1] = StarterPlan(pitch_limit=80, opener=True, piggyback=piggyback)
    assert validate(card, r, ELIG).ok
    tc = team_card(DUCKS, "DUC", card, r, 1, "R", ELIG)
    # The listed pitcher opens; the piggyback follows with the slot's pitch limit.
    assert tc.starter.id == str(card.rotation[1]) and tc.starter.pitch_limit == 25
    assert tc.bullpen[0].id == str(piggyback) and tc.bullpen[0].pitch_limit == 80
    assert len({p.id for p in tc.bullpen}) == len(tc.bullpen)  # nobody listed twice
    # Without the opener box the piggyback doesn't apply.
    card.plans[1] = StarterPlan(pitch_limit=80, piggyback=piggyback)
    tc = team_card(DUCKS, "DUC", card, r, 1, "R", ELIG)
    assert tc.starter.id == str(card.rotation[1]) and tc.starter.pitch_limit == 80
    assert str(piggyback) in [p.id for p in tc.bullpen]  # he's just a reliever again


@pytest.fixture
def client(tmp_path):
    c = TestClient(create_app(store=Store(tmp_path), players=PLAYERS, elig=ELIG))
    c.app_store_root = tmp_path
    return c


def form_for(card: LineupCard) -> dict:
    data = {"active": [str(a) for a in card.active]}
    for side, slots in card.orders.items():
        for i, (pid, pos) in enumerate(slots, 1):
            data[f"{side}_player_{i}"] = str(pid)
            data[f"{side}_pos_{i}"] = pos
    for i, pid in enumerate(card.rotation, 1):
        data[f"sp_{i}"] = str(pid)
        plan = card.plan(i - 1)
        data[f"sp_limit_{i}"] = str(plan.pitch_limit)
        data[f"sp_tto_{i}"] = str(plan.tto_hook)
        data[f"sp_opener_{i}"] = "1" if plan.opener else ""
        data[f"sp_piggyback_{i}"] = str(plan.piggyback or "")
    for i, sub in enumerate(card.subs, 1):
        data[f"sub_player_{i}"] = str(sub.player)
        data[f"sub_pos_{i}"] = sub.position
        data[f"sub_inning_{i}"] = str(sub.from_inning)
        data[f"sub_replaces_{i}"] = str(sub.replaces or "")
        if sub.only_when_leading:
            data[f"sub_leading_{i}"] = "1"
    for key, on in card.defense.items():
        if on:
            data[f"def_{key}"] = "1"
    for pid, role in card.roles.items():
        data[f"role_{pid}"] = role
    for pid, rp in card.relief.items():
        data[f"rp_use_{pid}"] = rp.usage
        data[f"rp_vs_{pid}"] = rp.vs or ""
        data[f"rp_outs_{pid}"] = str(rp.max_outs or "")
        if rp.start_of_inning_only:
            data[f"rp_clean_{pid}"] = "1"
    return data


def test_lineup_card_round_trip(client):
    page = client.get("/teams/midwest-ducksnorts/lineup")
    assert page.status_code == 200 and "Batting order vs LHP" in page.text

    card = default_card(roster(DUCKS), ELIG)
    card.orders["lhp"] = list(reversed(card.orders["lhp"]))
    card.plans[0] = StarterPlan(pitch_limit=105, tto_hook=3)
    card.defense = {"infield_in": True}
    saved = client.post("/teams/midwest-ducksnorts/lineup", data=form_for(card))
    assert saved.status_code == 200 and "Lineup card saved" in saved.text
    assert "card is legal" in saved.text
    stored = Store(client.app_store_root).load_card(DUCKS)
    assert stored.plans[0] == StarterPlan(pitch_limit=105, tto_hook=3)
    assert stored.defense["infield_in"] and not stored.defense["dp_depth"]
    assert stored.orders["lhp"] == card.orders["lhp"]

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
    assert all(
        by_id[pid] in resp.text for pid, _ in default_card(roster(DUCKS), ELIG).orders["rhp"]
    )
    assert "at Hill Valley McFlys" in client.get("/").text
    assert client.post("/games", data={"away": DUCKS, "home": DUCKS}).status_code == 400


def test_parse_people_fielding():
    from npl_sim.rosters.eligibility import parse_people_fielding

    def split(season, pos, games):
        return {"season": str(season), "position": {"abbreviation": pos}, "stat": {"games": games}}

    raw = {
        "people": [
            {
                "id": HENDERSON,
                "batSide": {"code": "L"},
                "pitchHand": {"code": "R"},
                "stats": [
                    {
                        "splits": [
                            split(2025, "SS", 100),
                            split(2025, "SS", 50),  # traded: one split per team
                            split(2025, "3B", 4),
                            split(2025, "P", 1),
                            split(2024, "SS", 160),
                        ]
                    }
                ],
            }
        ]
    }
    rec = parse_people_fielding(raw, [2025])[HENDERSON]
    assert (rec.bats, rec.throws) == ("L", "R")
    assert rec.games == {2025: {"SS": 150, "3B": 4}}


def test_defense_boxes_become_engine_rules():
    r = roster(DUCKS)
    card = default_card(r, ELIG)
    assert team_card(DUCKS, "DUC", card, r, 0, "R", ELIG).alignment_rules == ()
    card.defense = {"guard_lines": True, "infield_in": True}
    rules = team_card(DUCKS, "DUC", card, r, 0, "R", ELIG).alignment_rules
    assert [rule.preset for rule in rules] == ["infield_in", "guard_lines"]


def test_autofill_uses_games_at_each_position():
    r = roster(DUCKS)
    elig = Eligibility(
        season=2026,
        players={
            **ELIG.players,
            # Two catchers: the one with more games behind the plate gets it.
            691788: PlayerRecord("L", "R", {2025: {"C": 90}}),
            GOODMAN: PlayerRecord("R", "R", {2025: {"C": 70, "RF": 21}}),
            # Starters and a closer by their pitching lines.
            543037: PlayerRecord(pitching={2025: {"g": 32, "gs": 32, "outs": 600}}),
            656730: PlayerRecord(pitching={2025: {"g": 60, "gs": 0, "outs": 180, "sv": 30}}),
        },
    )
    card = autofill(LineupCard(active=[p.mlb_id for p in r]), r, elig)
    assert len(card.active) == ACTIVE_LIMIT
    rhp = dict((pos, pid) for pid, pos in card.orders["rhp"])
    assert rhp["C"] == 691788 and rhp["SS"] == HENDERSON
    assert sorted(rhp) == sorted(["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"])
    assert card.rotation[0] == 543037 and len(card.rotation) == 5
    assert card.roles[656730] == "CL"
    assert validate(card, r, elig).ok, validate(card, r, elig).errors


def test_defensive_subs_round_trip_and_checks():
    r = roster(DUCKS)
    card = default_card(r, ELIG)
    starting = {pid for pid, _ in card.orders["rhp"]}
    bench = next(
        p.mlb_id for p in r if p.mlb_id in card.active and p.mlb_id not in starting and not p.throws
    )
    card.subs = [DefensiveSub(player=bench, position="LF", replaces=card.orders["rhp"][0][0])]
    assert LineupCard.from_json(card.to_json()).subs == card.subs
    assert validate(card, r, ELIG).ok
    card.subs.append(DefensiveSub(player=bench, position="DH"))
    errors = " ".join(validate(card, r, ELIG).errors)
    assert "listed as a defensive replacement twice" in errors
    assert "Pick a field position" in errors


def test_picked_players_drop_out_of_other_dropdowns(client):
    page = client.get("/teams/midwest-ducksnorts/lineup").text
    card = default_card(roster(DUCKS), ELIG)
    leadoff = card.orders["rhp"][0][0]
    # The leadoff man is an option only in his own vs-RHP slot.
    rhp_selects = [s for s in page.split("<select") if 'name="rhp_player_' in s]
    assert sum(f'value="{leadoff}"' in s for s in rhp_selects) == 1
    starters = [
        s
        for s in page.split("<select")
        if 'name="sp_' in s
        and "_limit" not in s
        and "_tto" not in s
        and "opener" not in s
        and "piggyback" not in s
    ]
    assert sum(f'value="{card.rotation[0]}"' in s for s in starters) == 1


def test_autofill_route(client):
    data = form_for(default_card(roster(DUCKS), ELIG))
    resp = client.post(
        "/teams/midwest-ducksnorts/lineup/autofill", data=data, headers={"HX-Request": "true"}
    )
    assert resp.status_code == 200 and "<html" not in resp.text and "card is legal" in resp.text


def test_any_player_any_position_and_no_closer_needed(client):
    # Every position dropdown offers every position; ineligible ones are marked.
    page = client.get("/teams/midwest-ducksnorts/lineup").text
    pos_selects = [s.split("</select>")[0] for s in page.split("<select") if 'name="rhp_pos_' in s]
    assert pos_selects and all(s.count("<option") == len(POSITIONS) for s in pos_selects)
    assert "out of pos" in page

    card = default_card(roster(DUCKS), ELIG)
    card.roles = {pid: ("MR" if role == "CL" else role) for pid, role in card.roles.items()}
    check = validate(card, roster(DUCKS), ELIG)
    assert check.ok, check.errors


def test_bullpen_grouped_by_role_for_dragging(client):
    page = client.get("/teams/midwest-ducksnorts/lineup").text
    card = default_card(roster(DUCKS), ELIG)
    closer = next(pid for pid, role in card.roles.items() if role == "CL")
    group = page.split('data-role-group="CL"')[1].split("</tbody>")[0]
    assert f'data-pid="{closer}"' in group
    assert 'data-role-group=""' in page and page.count('data-drag="sp"') == 5


def test_reliever_usage_round_trip(client):
    card = default_card(roster(DUCKS), ELIG)
    pen = [pid for pid in card.roles]
    card.relief = {
        pen[0]: ReliefPlan(usage="mopup", max_outs=9),
        pen[1]: ReliefPlan(vs="L", max_outs=1, start_of_inning_only=True),
    }
    back = LineupCard.from_form(
        {k: v if isinstance(v, list) else [v] for k, v in form_for(card).items()}
    )
    assert back.relief == card.relief
    assert LineupCard.from_json(card.to_json()).relief == card.relief

    resp = client.post(
        "/teams/midwest-ducksnorts/lineup/check",
        data=form_for(card),
        headers={"HX-Request": "true"},
    )
    html = resp.text
    assert f'name="rp_use_{pen[0]}"' in html and "Mop-up (blowouts)" in html
    use = html.split(f'name="rp_use_{pen[0]}"')[1].split("</select>")[0]
    assert '<option value="mopup" selected>' in use
