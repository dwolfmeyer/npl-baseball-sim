"""Slash lines by pitcher hand and fielding grades on the lineup card."""

import pandas as pd
from fastapi.testclient import TestClient

from npl_sim.web import stats
from npl_sim.web.app import create_app
from npl_sim.web.store import Store

from .test_web import DUCKS, ELIG, HENDERSON, PLAYERS


def pitches(rows):
    """rows: (batter, pitcher hand, event), each ending a PA; plus a non-final
    pitch and a stolen base, which must not count."""
    out = [{"batter": b, "p_throws": h, "events": e} for b, h, e in rows]
    out.append({"batter": rows[0][0], "p_throws": "R", "events": None})
    out.append({"batter": rows[0][0], "p_throws": "R", "events": "stolen_base_2b"})
    return pd.DataFrame(out)


def write_week(root, season, week, rows):
    folder = root / str(season) / f"week-{week:02d}"
    folder.mkdir(parents=True)
    pitches(rows).to_parquet(folder / "pitches.parquet", index=False)


def test_slash_line_counts():
    events = ["single", "home_run", "walk", "hit_by_pitch", "sac_fly", "strikeout", "field_out"]
    lines = stats.slash_lines(pitches([(1, "R", e) for e in events] + [(1, "L", "double")]))
    r = lines[1]["R"]
    assert (r.pa, r.ab, r.hits, r.bases) == (7, 4, 2, 5)
    assert r.text() == ".500/.571/1.250"
    assert lines[1]["L"].text() == "1.000/1.000/2.000"


def test_splits_read_every_week_and_fall_back_a_season(tmp_path):
    assert stats.load_splits(tmp_path, 2026).season is None
    write_week(tmp_path, 2025, 1, [(1, "R", "single")])
    assert stats.load_splits(tmp_path, 2026).season == 2025
    write_week(tmp_path, 2026, 1, [(1, "R", "single")])
    write_week(tmp_path, 2026, 2, [(1, "R", "strikeout"), (1, "L", "walk")])
    s = stats.load_splits(tmp_path, 2026)
    assert (s.season, s.weeks) == (2026, 2)
    assert s.line(1, "rhp").text() == ".500/.500/.500"
    assert s.line(1, "lhp").pa == 1 and s.line(2, "rhp") is None


def test_grades():
    assert stats.grade(0) == 50 and stats.grade(12) == 75 and stats.grade(-40) == 20
    g = stats.grades({"SS": 0.0})
    assert g["SS"] == 50 and g["2B"] >= 45 and g["C"] < g["2B"]


def test_card_shows_lines_and_grades(tmp_path):
    clean = tmp_path / "clean"
    app = create_app(store=Store(tmp_path), players=PLAYERS, elig=ELIG, clean_root=clean)
    client = TestClient(app)
    page = client.get("/teams/midwest-ducksnorts/lineup").text
    assert "Hitting and fielding" in page and "No MLB stats loaded yet" in page

    write_week(clean, 2026, 1, [(HENDERSON, "R", "home_run"), (HENDERSON, "L", "strikeout")])
    page = client.get("/teams/midwest-ducksnorts/lineup").text  # picks up the new week
    assert "2026 MLB stats, 1 week of clean data." in page
    assert "1.000/1.000/4.000" in page and ".000/.000/.000" in page
    assert DUCKS in page and "<b>50</b>" in page
