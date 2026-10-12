"""Where rostered players come from.

Live play reads the NPL app's Postgres database through a read-only login
(`NPL_DATABASE_URL`). A Django `dumpdata` export of the same tables is read
the same way, for tests and offline work.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from sqlalchemy import create_engine, text

from npl_sim.rosters.npl_app import ROSTER_FLAGS, NplPlayer, from_row

NPL_DATABASE_URL_ENV = "NPL_DATABASE_URL"

_PLAYER_QUERY = f"""
SELECT p.mlb_id, p.scoresheet_id, p.name, p.raw_name, p.position, p.mlb_org,
       {", ".join(f"p.{flag}" for flag in ROSTER_FLAGS)},
       t.full_name AS team_name
FROM npl_player p
JOIN npl_team t ON t.id = p.team_id
WHERE p.team_id IS NOT NULL
"""


def players_from_database(url: str | None = None) -> list[NplPlayer]:
    """Read every rostered player from the NPL app database (read-only)."""
    url = url or os.environ.get(NPL_DATABASE_URL_ENV)
    if not url:
        raise RuntimeError(f"set {NPL_DATABASE_URL_ENV} to the NPL app's read-only database URL")
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(_PLAYER_QUERY)).mappings().all()
    finally:
        engine.dispose()
    return [from_row(row, team=row["team_name"]) for row in rows]


def players_from_dump(path: Path | str) -> list[NplPlayer]:
    """Read rostered players from a Django dumpdata export of the NPL app."""
    with open(path) as fh:
        records = json.load(fh)
    teams = {r["pk"]: r["fields"]["full_name"] for r in records if r["model"] == "npl.team"}
    players = []
    for r in records:
        if r["model"] != "npl.player" or r["fields"].get("team") is None:
            continue
        team = teams.get(r["fields"]["team"])
        if team is None:
            continue
        players.append(from_row({**r["fields"], "mlb_id": r["pk"]}, team=team))
    return players
