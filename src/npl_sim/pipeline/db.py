"""Write pipeline facts the league database needs: player handedness and
each sim week's lineup lock."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from npl_sim.db.models import Player, SimWeek


def sync_players(session: Session, players: pd.DataFrame) -> int:
    """Fill bats/throws on rostered players from the Stats API. Returns rows changed."""
    by_id = {int(r.mlbam_id): r for r in players.itertuples() if pd.notna(r.mlbam_id)}
    changed = 0
    for player in session.scalars(select(Player).where(Player.mlbam_id.is_not(None))):
        info = by_id.get(player.mlbam_id)
        if info is None:
            continue
        bats, throws = info.bats or None, info.throws or None
        if (player.bats, player.throws) != (bats, throws):
            player.bats, player.throws = bats, throws
            changed += 1
    return changed


def set_lineup_locks(session: Session, season: int, locks: dict[int, datetime]) -> int:
    weeks = session.scalars(select(SimWeek).where(SimWeek.season_year == season))
    changed = 0
    for week in weeks:
        lock = locks.get(week.number)
        if lock is not None and week.lineup_lock_at != lock:
            week.lineup_lock_at = lock
            changed += 1
    return changed
