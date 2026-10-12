"""Command line entry point: `npl-sim league-check`, `sync-league`, `import-rosters`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date

from sqlalchemy.orm import Session

from npl_sim.db import get_engine
from npl_sim.db.seed import sync_alignment
from npl_sim.league import load_league_file, team_slots
from npl_sim.rosters.importer import get_or_create_week, import_rosters
from npl_sim.rosters.sources import players_from_database, players_from_dump


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="npl-sim")
    sub = parser.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser("league-check", help="validate config/league.yaml for a season")
    check.add_argument("--season", type=int, default=2026)

    sync = sub.add_parser("sync-league", help="write a season's alignment to the database")
    sync.add_argument("--season", type=int, default=2026)

    rosters = sub.add_parser(
        "import-rosters", help="import a week's rosters from the NPL app (NPL_DATABASE_URL)"
    )
    rosters.add_argument("--season", type=int, required=True)
    rosters.add_argument("--week", type=int, required=True, help="sim week number")
    rosters.add_argument("--week-start", type=date.fromisoformat, required=True, help="the Monday")
    rosters.add_argument("--from-dump", help="read a Django dumpdata export instead of the NPL DB")
    rosters.add_argument("--team", action="append", help="limit to these NPL teams (repeatable)")

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
    elif args.cmd == "import-rosters":
        players = players_from_dump(args.from_dump) if args.from_dump else players_from_database()
        with Session(get_engine()) as session, session.begin():
            week = get_or_create_week(session, args.season, args.week, args.week_start)
            report = import_rosters(session, players, week, teams=args.team)
        for team, tiers in sorted(report.tiers.items()):
            counts = ", ".join(f"{tier.value} {n}" for tier, n in sorted(tiers.items()))
            print(f"{team}: {counts}")
        for name in sorted(report.unknown_teams):
            print(f"Unknown NPL team (not in config/league.yaml): {name}")
        for p in report.unmatched:
            print(f"No MLBAM ID: {p.name} ({p.position}, {p.team}, {p.tier.value})")
        if report.unmatched_mlb_roster:
            print(f"FAILED: {len(report.unmatched_mlb_roster)} MLB Roster players lack an MLBAM ID")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
