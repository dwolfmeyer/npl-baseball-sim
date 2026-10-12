"""Command line entry point: `npl-sim league check`, `npl-sim db sync-league`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from sqlalchemy.orm import Session

from npl_sim.db import get_engine
from npl_sim.db.seed import sync_alignment
from npl_sim.league import load_league_file, team_slots


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="npl-sim")
    sub = parser.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser("league-check", help="validate config/league.yaml for a season")
    check.add_argument("--season", type=int, default=2026)

    sync = sub.add_parser("sync-league", help="write a season's alignment to the database")
    sync.add_argument("--season", type=int, default=2026)

    args = parser.parse_args(argv)
    league_file = load_league_file()

    if args.cmd == "league-check":
        alignment = league_file.for_season(args.season)
        for slot in team_slots(alignment):
            print(f"{slot.league:3} {slot.division:18} {slot.team}")
        print(f"OK: {len(team_slots(alignment))} teams for {args.season}")
    elif args.cmd == "sync-league":
        with Session(get_engine()) as session, session.begin():
            sync_alignment(session, league_file, args.season)
        print(f"Synced {args.season} alignment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
