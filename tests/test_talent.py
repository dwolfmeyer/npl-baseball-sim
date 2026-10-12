from datetime import date

import pandas as pd
import pytest

from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.sample import hitter, pitcher
from npl_sim.talent import blend as tb
from npl_sim.talent.env import default_league, league
from npl_sim.talent.events import plate_appearances
from npl_sim.talent.load import build_week_talent, load_week_talent, stuff_by_pitcher
from tests.talent_synth import pitches

BATTERS = {i: {"hand": "LR"[i % 2]} for i in range(1, 120)}
PITCHERS = {1000 + i: {"hand": "LR"[i % 3 == 0]} for i in range(1, 80)}
BATTERS[1].update(k=1.6)
BATTERS[3].update(hr=2.0)
PITCHERS[1001].update(k=1.5)


@pytest.fixture(scope="module")
def history():
    prior = pitches(BATTERS, PITCHERS, 30000, seed=1, start=date(2025, 4, 1), days=180)
    to_date = pitches(BATTERS, PITCHERS, 20000, seed=2, start=date(2026, 4, 1), days=60)
    return prior, to_date


def test_plate_appearances_use_the_engine_outcomes():
    raw = pd.DataFrame(
        {
            "batter": [1, 1, 1, 1, 1],
            "pitcher": [9] * 5,
            "stand": ["L"] * 5,
            "p_throws": ["R"] * 5,
            "game_date": [date(2026, 5, 4)] * 5,
            "events": ["strikeout", "intent_walk", "double", "field_error", None],
            "bb_type": [None, None, "line_drive", None, None],
            "description": ["swinging_strike", "ball", "hit_into_play", "hit_into_play", "ball"],
        }
    )
    pa = plate_appearances(raw)
    assert list(pa["outcome"]) == ["K", "2B", "OUT"]  # no IBB; an error is an out
    assert list(pa["bip"].fillna("-")) == ["-", "LD", "GB"]


def test_league_env_from_real_pas(history):
    lg = league(plate_appearances(history[0]))
    env = lg.env()
    assert 0.2 < env.rates[("R", "R")].k < 0.26
    assert sum(env.rates[("L", "R")].bip.values()) == pytest.approx(1.0)
    assert env.hit_table["LD"]["1B"] > env.hit_table["GB"]["1B"]
    # Too few PAs: the placeholder environment.
    assert league(plate_appearances(history[0]).head(10)).pa == default_league().pa


def test_blend_recovers_known_talent(history):
    week = pitches(BATTERS, PITCHERS, 3000, seed=3)
    wt = build_week_talent(week, history[1], [(1, history[0])])
    k1 = wt.players["batter:1"].layers["baseline"]["K"]
    assert 1.4 < k1 < 1.8  # true 1.6, a little regressed
    assert wt.players["batter:3"].layers["baseline"]["HR"] > 1.4
    assert wt.players["pitcher:1001"].layers["baseline"]["K"] > 1.3
    avg = [wt.players[f"batter:{i}"].layers["baseline"]["K"] for i in range(5, 120)]
    assert 0.95 < sum(avg) / len(avg) < 1.05


def test_a_few_pa_barely_move_the_baseline(history):
    """4 PA of strikeouts and whiffs move K a little; a full week moves it more."""

    def week_of(n_pa, seed):
        hot = {7: {"hand": "R", "k": 3.0}}
        return pitches(hot, PITCHERS, n_pa, seed=seed)

    small = build_week_talent(week_of(4, 5), history[1], [(1, history[0])])
    big = build_week_talent(week_of(30, 5), history[1], [(1, history[0])])
    base = small.players["batter:7"].layers["baseline"]["K"]
    k_small = small.players["batter:7"].index["R"]["K"] / base
    k_big = big.players["batter:7"].index["R"]["K"] / base
    assert 1.0 <= k_small < 1.2  # even 4 Ks in 4 PA
    assert k_big > k_small


def test_results_count_for_less_than_process(history):
    """A week of Ks with an ordinary whiff rate moves K less than the
    same week with the whiffs to match."""
    others = pitches(BATTERS, PITCHERS, 2000, seed=6)

    def week(whiffs: bool):
        rows = []
        for i in range(30):
            k = i < 12
            for desc in ("swinging_strike" if (whiffs and k) else "foul", "ball"):
                rows.append({"batter": 500, "pitcher": 1002, "stand": "R", "p_throws": "R",
                             "game_date": date(2026, 5, 4), "zone": 5, "pitch_type": "FF",
                             "description": desc, "events": None})  # fmt: skip
            rows.append({"batter": 500, "pitcher": 1002, "stand": "R", "p_throws": "R",
                         "game_date": date(2026, 5, 4), "zone": 5, "pitch_type": "FF",
                         "description": "swinging_strike" if k else "hit_into_play",
                         "events": "strikeout" if k else "field_out",
                         "bb_type": None if k else "ground_ball"})  # fmt: skip
        return pd.concat([others, pd.DataFrame(rows)], ignore_index=True)

    out = {w: build_week_talent(week(w), history[1], [(1, history[0])]) for w in (False, True)}
    plain, backed = out[False].players["batter:500"], out[True].players["batter:500"]
    assert plain.layers["results"]["K"] > 1.0  # the Ks still count, a little
    assert backed.layers["process"]["K"] > plain.layers["process"]["K"]
    assert backed.index["R"]["K"] > plain.index["R"]["K"]


def test_platoon_uses_prior_seasons_unless_no_split_data():
    lg = default_league()

    def pas(season_rows):
        return pd.DataFrame(season_rows)

    def row(throws, outcome):
        return {"batter": 1, "pitcher": 2, "stand": "L", "p_throws": throws,
                "outcome": outcome, "bip": None if outcome in ("K", "BB", "HR") else "GB",
                "xba": None, "barrel": None, "hard": None}  # fmt: skip

    prior = pas([row("R", "OUT")] * 300)  # no PA vs LHP last year
    current = pas([row("L", "K")] * 40 + [row("R", "K")] * 40 + [row("R", "OUT")] * 40)
    seasons = [tb.Season(1, prior, lg), tb.Season(0, current, lg)]
    week = tb.Season(0, pas([]), lg)
    t = tb.blend(seasons, week)["batter:1"]
    # vs RHP: last year's split (no Ks) holds, this year's Ks vs RHP don't
    # shift the split; vs LHP: no prior data, so this year's (all Ks) counts.
    assert t.index["L"]["K"] > t.index["R"]["K"]


def test_stuff_plus_regressed_by_pitch_count():
    wild = tb.stuff_plus({"SL": 351.0, "FF": 110.0}, {"SL": 3, "FF": 600})
    assert 105 < wild < 112  # three sliders can't make him elite
    assert tb.stuff_plus({}, {"FF": 50}, overall=120.0) == pytest.approx(105.0)
    assert tb.stuff_plus({}, {}, overall=None) is None
    board = pd.DataFrame({"mlbam_id": [9], "sp_stuff": [104.0], "sp_s_FA": [130.0]})
    p = pd.DataFrame({"pitcher": [9] * 300, "pitch_type": ["FF"] * 300})
    assert stuff_by_pitcher(board, p)["9"] == pytest.approx(100 + 30 * 300 / 450)


def test_cards_from_talent():
    lg = default_league()
    avg = tb.hitter_card(hitter("1", bats="L"), None, lg)
    assert avg.vs_rhp.k == pytest.approx(DEFAULT_ENV.split("L", "R").k)
    slugger = tb.Talent("1", "batter", {h: {**dict.fromkeys(tb.STATS, 1.0), "HR": 2.0,
                                            "XBH": 1.2} for h in "LR"})  # fmt: skip
    card = tb.hitter_card(hitter("1", bats="S"), slugger, lg)
    assert card.vs_lhp.hr == pytest.approx(2 * DEFAULT_ENV.split("R", "L").hr)
    assert card.vs_rhp.power == pytest.approx(1.2)
    ace = tb.Talent("9", "pitcher", {h: {**dict.fromkeys(tb.STATS, 1.0), "K": 1.5} for h in "LR"})
    p = tb.pitcher_card(pitcher("9", "L"), ace, lg)
    assert p.vs_rhb.k == pytest.approx(1.5 * DEFAULT_ENV.split("R", "L").k)
    p.vs_rhb.true_outcomes()  # still a valid distribution


def test_load_week_talent_from_clean_tables(tmp_path, history):
    prior, to_date = history
    for season, frame in ((2025, prior), (2026, to_date)):
        d = tmp_path / str(season) / "week-01"
        d.mkdir(parents=True)
        frame.to_parquet(d / "pitches.parquet")
    week = tmp_path / "2026" / "week-02"
    week.mkdir()
    pitches(BATTERS, PITCHERS, 2000, seed=4, start=date(2026, 6, 1)).to_parquet(
        week / "pitches.parquet"
    )
    asof = tmp_path / "2026" / "asof-2026-06-07"
    asof.mkdir()
    pd.DataFrame({"mlbam_id": [1, 2], "sprint_speed": [30.0, 24.0]}).to_parquet(
        asof / "sprint_speed.parquet"
    )
    pd.DataFrame({"mlbam_id": [1001], "sp_stuff": [125.0]}).to_parquet(asof / "fg_pitching.parquet")
    wt = load_week_talent(tmp_path, 2026, 2)
    assert wt.speed["1"] == 2.0 and wt.speed["2"] == -2.0
    assert wt.players["pitcher:1001"].layers["process"]["K"] > 1.0  # Stuff+ 125
    h = wt.hitter(hitter("3", bats="R"))
    assert h.vs_rhp.hr > DEFAULT_ENV.split("R", "R").hr
    assert wt.hitter(hitter("AAA-x")).vs_rhp == hitter("AAA-x").vs_rhp
    with pytest.raises(FileNotFoundError):
        load_week_talent(tmp_path, 2026, 9)


def test_sim_week_plays_blended_talent():
    from npl_sim.engine.sample import average_week_team, busy_week
    from npl_sim.engine.workload import MlbWeek, play_week
    from npl_sim.simweek import round_robin
    from npl_sim.talent.load import WeekTalent

    teams = {n: average_week_team(n) for n in ("Aces", "Duds")}
    mlb = MlbWeek(**busy_week(*teams.values()))
    games = round_robin(list(teams))
    lg = default_league()

    def tal(role, pid, k, hr):
        idx = {h: {**dict.fromkeys(tb.STATS, 1.0), "K": k, "HR": hr} for h in "LR"}
        return tb.Talent(pid, role, idx)

    players = {}
    for name, (k, hr) in (("Aces", (0.7, 1.6)), ("Duds", (1.4, 0.5))):
        for p in teams[name].players:
            players[f"batter:{p}"] = tal("batter", p, k, hr)
            players[f"pitcher:{p}"] = tal("pitcher", p, 1 / k, 1 / hr)
    wt = WeekTalent(league=lg, players=players)
    runs = {"Aces": 0, "Duds": 0}
    for seed in range(3):
        result = play_week({n: wt.team(t) for n, t in teams.items()}, games, mlb,
                           lg.env(), seed)  # fmt: skip
        for g in result.games:
            final = g["events"][-1]
            runs[g["away"]] += final["away"]
            runs[g["home"]] += final["home"]
    assert runs["Aces"] > 1.5 * runs["Duds"]
