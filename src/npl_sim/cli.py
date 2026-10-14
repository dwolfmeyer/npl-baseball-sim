"""Command line entry point: `npl-sim league-check`, `sync-league`, `import-rosters`,
`generate-schedule`, `demo-game`, `demo-week`, `sim-week`, `sim-season`, `calibrate`, `web`,
`eligibility`, `rosters-from-workbook`, and the MLB data pipeline (`pull-week`, `backfill`,
`check-calendar`, `sync-mlb`)."""

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from npl_sim.db import get_engine
from npl_sim.db.seed import sync_alignment
from npl_sim.league import load_league_file, team_slots
from npl_sim.pipeline.db import set_lineup_locks, sync_players
from npl_sim.pipeline.run import DEFAULT_CLEAN_ROOT, SOURCES, Pipeline, lineup_locks
from npl_sim.pipeline.snapshots import DEFAULT_RAW_ROOT, SnapshotStore
from npl_sim.rosters.importer import get_or_create_week, import_rosters
from npl_sim.rosters.sources import players_from_database, players_from_dump
from npl_sim.schedule.calendar import load_calendar, sim_weeks
from npl_sim.schedule.matchups import build_series
from npl_sim.schedule.packer import pack
from npl_sim.schedule.store import save_schedule
from npl_sim.schedule.validate import validate

# Interim league rosters, read from the roster workbook until the NPL app's
# database login is issued (`rosters-from-workbook`).
DEFAULT_ROSTERS = Path("data/rosters/npl-2026.json")


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

    sched = sub.add_parser("generate-schedule", help="build and validate a 162-game season")
    sched.add_argument("--season", type=int, required=True)
    sched.add_argument("--seed", type=int, default=0, help="solver seed")
    sched.add_argument("--csv", help="also write the games to this CSV file")
    sched.add_argument("--save", action="store_true", help="write weeks and games to the DB")

    demo = sub.add_parser("demo-game", help="sim one game between two average teams")
    demo.add_argument("--seed", type=int, default=1)
    demo.add_argument(
        "--plays",
        choices=("all", "scoring"),
        help="also print the play log: every play or scoring plays only",
    )
    demo.add_argument("--html", type=Path, help="write a Gameday-style game page here")

    demo_week = sub.add_parser(
        "demo-week", help="sim a week between two average teams under the playing-time budget"
    )
    demo_week.add_argument("--seed", type=int, default=1)

    cal = sub.add_parser(
        "calibrate",
        help="calibration harness: identities, run environment, archetypes, drift from baseline",
    )
    cal.add_argument("--games", type=int, default=1000)
    cal.add_argument("--seed", type=int, default=0)
    cal.add_argument(
        "--targets", default="placeholder", help="target set in config/calibration.yaml"
    )
    cal.add_argument(
        "--save-baseline",
        action="store_true",
        help="save these results as the engine baseline (after an intended engine change)",
    )

    simw = sub.add_parser(
        "sim-week",
        help="sim a week between the test teams' saved lineup cards under the playing-time budget",
    )
    simw.add_argument("--seed", type=int, default=1)
    simw.add_argument("--teams", nargs="*", help="team names (default: every team with a roster)")
    simw.add_argument("--season", type=int, default=2026)
    simw.add_argument(
        "--week",
        type=int,
        help="use this MLB week's real usage and the talent blend (needs clean data)",
    )
    simw.add_argument("--clean-root", type=Path, default=Path("data/clean"))
    simw.add_argument("--out", type=Path, help="write index.html and a page per game here")
    simw.add_argument("--save", action="store_true", help="also save the games to the web app")
    simw.add_argument(
        "--audit",
        type=Path,
        help="write an audit log here: game-NN.jsonl per game (every PA's inputs, odds, "
        "result, batted ball and pitches, plus players' real stats) and pas.csv",
    )

    seas = sub.add_parser(
        "sim-season",
        help="sim a whole season week by week between every team's default lineup card",
    )
    seas.add_argument("--season", type=int, default=2026)
    seas.add_argument("--seed", type=int, default=1)
    seas.add_argument("--schedule-seed", type=int, default=0, help="solver seed")
    seas.add_argument(
        "--rosters", type=Path, default=DEFAULT_ROSTERS, help="NPL app dump of every team"
    )
    seas.add_argument("--csv", type=Path, help="also write every game's score to this CSV file")

    wbk = sub.add_parser(
        "rosters-from-workbook",
        help="read the league's roster workbook (one tab per team) into an NPL app dump",
    )
    wbk.add_argument("workbook", type=Path)
    wbk.add_argument("--season", type=int, default=2026)
    wbk.add_argument("--out", type=Path, default=DEFAULT_ROSTERS)

    elig = sub.add_parser(
        "eligibility",
        help="pull position eligibility and hands for rostered players from the MLB Stats API",
    )
    elig.add_argument("--season", type=int, required=True, help="the NPL season")
    elig.add_argument("--from-dump", help="read rosters from a dumpdata export, not the NPL DB")

    web = sub.add_parser("web", help="run the web app (lineup cards and game results)")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8000)

    def pipeline_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--season", type=int, required=True)
        p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_ROOT)
        p.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_ROOT)
        p.add_argument("--source", action="append", choices=SOURCES, help="limit sources")

    pull = sub.add_parser("pull-week", help="pull, normalize and freeze one MLB week")
    pipeline_args(pull)
    pull.add_argument("--week", type=int, required=True, help="sim week number")
    pull.add_argument("--refresh", action="store_true", help="re-fetch snapshots we already have")
    pull.add_argument("--no-freeze", action="store_true", help="leave the week open")

    back = sub.add_parser("backfill", help="load a completed season week by week")
    pipeline_args(back)
    back.add_argument("--from-week", type=int, default=1)
    back.add_argument("--to-week", type=int)

    cal = sub.add_parser("check-calendar", help="compare config/calendar.yaml with MLB")
    pipeline_args(cal)

    mlb = sub.add_parser("sync-mlb", help="write handedness and lineup locks to the DB")
    pipeline_args(mlb)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
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
    elif args.cmd == "generate-schedule":
        alignment = league_file.for_season(args.season)
        weeks = sim_weeks(load_calendar(args.season))
        schedule = pack(build_series(alignment, args.season), weeks, seed=args.seed)
        problems = validate(schedule, alignment, weeks)
        for p in problems:
            print(f"INVALID: {p}")
        if problems:
            return 1
        print(f"Valid {args.season} schedule: {len(schedule)} series over {len(weeks)} weeks")
        if args.csv:
            with open(args.csv, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["week", "date", "away", "home", "series_games", "kind"])
                for item in schedule:
                    s = item.series
                    for d in item.dates:
                        writer.writerow([item.week, d, s.away, s.home, s.games, s.kind])
            print(f"Wrote {args.csv}")
        if args.save:
            with Session(get_engine()) as session, session.begin():
                sync_alignment(session, league_file, args.season)
                n = save_schedule(session, args.season, weeks, schedule)
            print(f"Saved {n} games")
    elif args.cmd == "demo-game":
        from npl_sim.engine.box import box_score, format_box
        from npl_sim.engine.defaults import DEFAULT_ENV
        from npl_sim.engine.game import simulate_game
        from npl_sim.engine.playlog import format_log, game_log
        from npl_sim.engine.sample import average_team
        from npl_sim.reporting.gameday import render_game

        events = simulate_game(average_team("Away"), average_team("Home"), DEFAULT_ENV, args.seed)
        box = box_score(events)
        print(format_box(box))
        if args.plays:
            print("\n" + format_log(game_log(events), args.plays))
        if args.html:
            args.html.write_text(render_game(box, game_log(events)))
            print(f"\nWrote {args.html}")
    elif args.cmd == "demo-week":
        from npl_sim.engine.defaults import DEFAULT_ENV
        from npl_sim.engine.sample import average_week_team, busy_week
        from npl_sim.engine.workload import MlbWeek, play_week

        teams = {n: average_week_team(n) for n in ("Away", "Home")}
        usage = busy_week(*teams.values())
        # Make the week interesting: a catcher who didn't play, a part-timer,
        # a starter who skipped his turn.
        usage["pa"].update({"Home-C": 0, "Home-1B": 8})
        usage["starts"]["Home-SP3"] = 0
        games = [(day, "Away", "Home") for day in range(7)]
        result = play_week(teams, games, MlbWeek(**usage), DEFAULT_ENV, args.seed)
        for g in result.games:
            final = g["events"][-1]
            print(f"Day {g['day']}: Away {final['away']}, Home {final['home']}")
            for team in ("Away", "Home"):
                m = g["moves"][team]
                notes = [f"starter {m.starter}" + (" (bullpen game)" if m.bullpen_game else "")]
                notes += [f"{r['into']} for {r['out']} at {r['position']}" for r in m.replaced]
                if m.unavailable:
                    notes.append("resting or used up: " + ", ".join(m.unavailable))
                print(f"  {team}: " + "; ".join(notes))
    elif args.cmd == "calibrate":
        return _calibrate(args)
    elif args.cmd == "sim-week":
        return _sim_week(args)
    elif args.cmd == "sim-season":
        return _sim_season(args, league_file)
    elif args.cmd == "rosters-from-workbook":
        from npl_sim.rosters.workbook import players_from_workbook, write_dump

        book = players_from_workbook(args.workbook, league_file.for_season(args.season))
        path = write_dump(book.players, args.out)
        active = [p for p in book.players if p.is_active_mlb]
        print(f"Wrote {path}: {len(book.players)} players, {len(active)} on MLB Rosters")
        for line in book.division_mismatches(league_file.for_season(args.season)):
            print(f"Division differs: {line}")
    elif args.cmd == "eligibility":
        from npl_sim.pipeline.http import Http
        from npl_sim.rosters import eligibility

        players = players_from_dump(args.from_dump) if args.from_dump else players_from_database()
        ids = [p.mlb_id for p in players if p.is_active_mlb and p.mlb_id is not None]
        path = eligibility.save(eligibility.fetch(Http(), ids, args.season))
        print(f"Wrote {path} ({len(ids)} players)")
    elif args.cmd == "web":
        import uvicorn

        from npl_sim.web.app import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)
    elif args.cmd in ("pull-week", "backfill", "check-calendar", "sync-mlb"):
        return _pipeline_command(args)
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    from npl_sim.calibration import harness as h

    games = h.average_games(args.games, args.seed)
    stats = h.summary(games)
    print(f"{args.games} games between league-average teams (seed {args.seed})\n")
    print("Run environment vs", args.targets)
    checks = h.check_targets(stats, h.load_targets(args.targets))
    for c in checks:
        print(f"  {c.stat:<8} {c.value:7.3f}   [{c.lo:.3f}, {c.hi:.3f}]  {'ok' if c.ok else 'OFF'}")
    bad = h.identities(games)
    print(f"\nAccounting identities: {'all add up' if not bad else f'{len(bad)} broken'}")
    for msg in bad[:10]:
        print("  " + msg)
    print("\nArchetypes (sim vs card, per PA)")
    for kind, r in h.archetype_report().items():
        sim, exp = r["sim"], r["expected"]
        cells = "  ".join(f"{k} {sim[k]:.3f}/{exp[k]:.3f}" for k in ("K", "BB", "HR"))
        print(f"  {kind:<20} {cells}  TTO {sim['tto']:.3f}")
    sd50, bin50 = h.k_spread(games, 50)
    sd200, bin200 = h.k_spread(games, 200)
    print(
        f"\nK% spread: 50 PA {sd50:.3f} (binomial {bin50:.3f}), 200 PA {sd200:.3f} ({bin200:.3f})"
    )
    fp = h.fingerprint(games)
    if args.save_baseline:
        print(f"\nSaved baseline to {h.save_baseline(fp)}")
        return 0
    if h.BASELINE.exists():
        d = h.drift(h.load_baseline(), fp)
        flags = {k: "ok" if d[k] <= h.DRIFT_LIMITS[k] else "SHIFTED" for k in d}
        print(
            f"\nDrift from baseline: event mix JSD {d['jsd']:.5f} ({flags['jsd']}), "
            f"runs/game Wasserstein {d['wasserstein']:.3f} ({flags['wasserstein']})"
        )
    return 0 if not bad and all(c.ok for c in checks) else 1


def _sim_week(args: argparse.Namespace) -> int:
    from npl_sim.engine.box import box_score
    from npl_sim.playing_time import load_mlb_week, load_real_week_stats
    from npl_sim.reporting.week import standings, write_week
    from npl_sim.rosters import eligibility as elig_data
    from npl_sim.simweek import round_robin, sim_week
    from npl_sim.web import lineup
    from npl_sim.web.app import load_players
    from npl_sim.web.store import Store

    elig = elig_data.load(args.season)
    rosters: dict[str, list] = {}
    for p in load_players():
        if p.is_active_mlb and p.mlb_id is not None:
            rosters.setdefault(p.team, []).append(p)
    teams = args.teams or sorted(rosters)
    missing = [t for t in teams if t not in rosters]
    if missing:
        print(f"No roster for: {', '.join(missing)}")
        return 1
    store = Store()
    cards = {}
    for t in teams:
        saved = store.load_card(t)
        cards[t] = saved or lineup.default_card(rosters[t], elig)
        print(f"{t}: {'saved lineup card' if saved else 'default lineup card (none saved)'}")
    mlb = talent = real = None
    usage = f"a typical week at each player's {args.season} pace (full time if no data)"
    ability = "league average (no --week)"
    if args.week is not None:
        from npl_sim.talent.load import load_week_talent

        mlb = load_mlb_week(args.clean_root, args.season, args.week)
        real = load_real_week_stats(args.clean_root, args.season, args.week)
        usage = f"real MLB usage from {args.season} week {args.week}"
        recs = {str(p.mlb_id): elig.players.get(p.mlb_id) for r in rosters.values() for p in r}
        hands = {pid: rec.bats for pid, rec in recs.items() if rec and rec.bats}
        talent = load_week_talent(args.clean_root, args.season, args.week, hands)
        ability = f"the talent blend for {args.season} week {args.week}"
    schedule = round_robin(teams)
    result = sim_week(
        cards, rosters, elig, schedule, mlb, args.seed, talent=talent, audit=bool(args.audit)
    )
    print(f"Playing time: {usage}\nAbility: {ability}\n")
    for g in result.games:
        f = g["events"][-1]
        print(f"Day {g['day'] + 1}: {g['away']} {f['away']}, {g['home']} {f['home']}")
    print("\nTeam                       W  L  RS  RA")
    for t, w, lo, rs, ra in standings(result):
        print(f"{t:<26} {w:>2} {lo:>2} {rs:>3} {ra:>3}")
    if args.out:
        title = f"NPL test week ({args.season}" + (f", week {args.week})" if args.week else ")")
        names = {str(p.mlb_id): p.name for r in rosters.values() for p in r}
        subtitle = f"Playing time: {usage}. Ability: {ability}. Seed {args.seed}."
        label = f"MLB {args.season} week {args.week}"
        index = write_week(result, args.out, title, subtitle, names, real, label)
        print(f"\nWrote {index}")
    if args.audit:
        from npl_sim.audit import write_audit

        print(f"Wrote the audit log to {write_audit(result.games, args.audit, args.seed, talent)}")
    if args.save:
        for g in result.games:
            box = box_score(g["events"])
            starters = {
                side: next(
                    p["name"]
                    for p in g["events"][0][side]["pitchers"]
                    if p["id"] == g["moves"][g[side]].starter
                )
                for side in ("away", "home")
            }
            store.save_game(
                {
                    "away": g["away"],
                    "home": g["home"],
                    "seed": args.seed,
                    "starters": starters,
                    "runs": {"away": box.runs[g["away"]], "home": box.runs[g["home"]]},
                    "events": g["events"],
                }
            )
        print(f"Saved {len(result.games)} games to the web app")
    return 0


def _sim_season(args: argparse.Namespace, league_file) -> int:
    from npl_sim.leaders import league_leaders
    from npl_sim.rosters import eligibility as elig_data
    from npl_sim.season import default_cards, sim_season
    from npl_sim.season_stats import division_standings, format_gb

    alignment = league_file.for_season(args.season)
    rosters: dict[str, list] = {}
    for p in players_from_dump(args.rosters):
        if p.is_active_mlb and p.mlb_id is not None:
            rosters.setdefault(p.team, []).append(p)
    teams = [s.team for s in team_slots(alignment)]
    missing = [t for t in teams if t not in rosters]
    if missing:
        print(f"No roster for: {', '.join(missing)}")
        return 1
    elig = elig_data.load(args.season)
    no_data = sum(p.mlb_id not in elig.players for r in rosters.values() for p in r)
    weeks = sim_weeks(load_calendar(args.season))
    schedule = pack(build_series(alignment, args.season), weeks, seed=args.schedule_seed)
    problems = validate(schedule, alignment, weeks)
    if problems:
        print("\n".join(f"INVALID: {p}" for p in problems))
        return 1
    print(f"{sum(len(r) for r in rosters.values())} MLB Roster players; {no_data} have no MLB")
    print("usage data yet and play a full-time week (`npl-sim eligibility` fills it in)\n")
    result = sim_season(
        default_cards(rosters, elig),
        rosters,
        elig,
        schedule,
        weeks,
        args.seed,
        on_week=lambda w, r: print(f"Week {w.number}: {len(r.games)} games", flush=True),
    )
    for league_name, div_name, rows in division_standings(result.stats, alignment):
        print(f"\n{league_name} {div_name:<22} W   L   PCT    GB   RS   RA  Home  Road  L10  Strk")
        for row in rows:
            r = row.record
            print(
                f"  {row.team:<26} {r.w:>3} {r.l:>3} {r.pct:>5.3f} {format_gb(row.gb):>5} "
                f"{r.rs:>4} {r.ra:>4} {r.home_w:>3}-{r.home_l:<3} {r.away_w:>3}-{r.away_l:<3} "
                f"{r.last10[0]}-{r.last10[1]:<3} {r.streak:>4}"
            )
    for league_name, boards in league_leaders(result.stats, alignment, n=3).items():
        print(f"\n{league_name} leaders")
        for kind in ("batting", "pitching"):
            for label, rows in boards[kind].items():
                top = ", ".join(f"{r.name} {r.value} ({r.team})" for r in rows)
                print(f"  {label:<4} {top}")
    t = result.totals
    pa = t["PA"]
    hits = sum(t[h] for h in ("1B", "2B", "3B", "HR"))
    ab = pa - t["BB"] - t["HBP"] - t["SF"]
    print(
        f"\nRuns/team-game {t['runs'] / t['games'] / 2:.2f}  K% {100 * t['K'] / pa:.1f}  "
        f"BB% {100 * t['BB'] / pa:.1f}  HR/PA {100 * t['HR'] / pa:.2f}%  AVG {hits / ab:.3f}  "
        f"SB {t['stolen_base']} CS {t['caught_stealing']} OOB {t['thrown_out']}  "
        f"pitches/PA {t['pitches'] / pa:.2f}"
    )
    bullpen = Counter(team for g in result.games for team, b in g.bullpen_game.items() if b)
    rest = result.starter_rest()
    print(
        f"Extra-inning games {sum(g.innings > 9 for g in result.games)} of {len(result.games)}; "
        f"bullpen games {sum(bullpen.values())} (most: "
        + ", ".join(f"{team} {n}" for team, n in bullpen.most_common(3))
        + f"); fewest days between a pitcher's starts {min(rest) if rest else '-'}"
    )
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["date", "away", "home", "away_runs", "home_runs", "innings"])
            for g in sorted(result.games, key=lambda g: (g.date, g.home)):
                w.writerow([g.date, g.away, g.home, g.runs[g.away], g.runs[g.home], g.innings])
        print(f"Wrote {args.csv}")
    return 0


def _pipeline_command(args: argparse.Namespace) -> int:
    pipe = Pipeline(
        season=args.season,
        calendar=load_calendar(args.season),
        raw=SnapshotStore(args.raw_dir),
        clean_root=args.clean_dir,
        sources=tuple(args.source or SOURCES),
    )
    if args.cmd == "pull-week":
        week = pipe.week(args.week)
        pipe.run_week(week, freeze=not args.no_freeze, refresh=args.refresh)
        print(f"{args.season} week {week.number} ({week.start} to {week.end}) done")
    elif args.cmd == "backfill":
        last = args.to_week or len(pipe.weeks)
        pipe.backfill(pipe.weeks[args.from_week - 1 : last])
        print(f"Backfilled {args.season} weeks {args.from_week}-{last}")
    elif args.cmd == "check-calendar":
        problems = pipe.check_calendar()
        for p in problems:
            print(f"MISMATCH: {p}")
        if problems:
            return 1
        print(f"config/calendar.yaml matches the MLB {args.season} schedule")
    elif args.cmd == "sync-mlb":
        season_dir = Path(args.clean_dir) / str(args.season)
        players = pd.read_parquet(season_dir / "players.parquet")
        frames = [pd.read_parquet(p) for p in sorted(season_dir.glob("week-*/games.parquet"))]
        games = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        with Session(get_engine()) as session, session.begin():
            n_players = sync_players(session, players)
            n_weeks = set_lineup_locks(session, args.season, lineup_locks(games, pipe.weeks))
        print(f"Updated handedness for {n_players} players and lineup locks for {n_weeks} weeks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
