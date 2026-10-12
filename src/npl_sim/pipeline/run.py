"""Weekly pull, normalize and freeze, and the season backfill.

Layout on disk:

    data/raw/<source>/<dataset>/<season>/<key>.<ext>    raw snapshots (+ manifest.jsonl)
    data/clean/<season>/week-NN/<table>.parquet          one sim week's MLB data
    data/clean/<season>/week-NN/FROZEN.json              written by the Monday freeze
    data/clean/<season>/asof-YYYY-MM-DD/<board>.parquet  season-to-date leaderboards
    data/clean/<season>/players.parquet                  handedness, position, birth date

A frozen week is never pulled or rewritten again unless forced, so a sim
week always runs on exactly the inputs it was frozen with.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from npl_sim.pipeline import calendar_check, fangraphs, ids, savant, statsapi
from npl_sim.pipeline.http import Http
from npl_sim.pipeline.snapshots import SnapshotStore
from npl_sim.schedule.calendar import SeasonCalendar, Week, sim_weeks, week_for_date

log = logging.getLogger(__name__)

DEFAULT_CLEAN_ROOT = Path("data/clean")
SOURCES = ("statsapi", "savant", "fangraphs")
WEEKLY_TABLES = ("games", "batting", "pitching", "pitches", "transactions", "league_totals")
PROJECTION_SYSTEMS = ("steamer",)


class FrozenWeekError(RuntimeError):
    pass


@dataclass
class Pipeline:
    season: int
    calendar: SeasonCalendar
    http: Http = field(default_factory=Http)
    raw: SnapshotStore = field(default_factory=SnapshotStore)
    clean_root: Path = DEFAULT_CLEAN_ROOT
    sources: tuple[str, ...] = SOURCES

    def __post_init__(self) -> None:
        self.clean_root = Path(self.clean_root)
        self.weeks = sim_weeks(self.calendar)

    # --- paths ---------------------------------------------------------------

    def week_dir(self, week: Week) -> Path:
        return self.clean_root / str(self.season) / f"week-{week.number:02d}"

    def asof_dir(self, as_of: date) -> Path:
        return self.clean_root / str(self.season) / f"asof-{as_of.isoformat()}"

    def is_frozen(self, week: Week) -> bool:
        return (self.week_dir(week) / "FROZEN.json").exists()

    def week(self, number: int) -> Week:
        return self.weeks[number - 1]

    def pull_range(self, week: Week) -> tuple[date, date]:
        """MLB dates whose stats count toward `week` (openers fold into week 1)."""
        start, end = week.start, week.end
        if week.number == 1 and self.calendar.openers:
            start = min(start, *self.calendar.openers)
        return max(start, self._first_game()), min(end, self.calendar.final_day)

    def _first_game(self) -> date:
        return min([self.calendar.opening_day, *self.calendar.openers])

    # --- pull ----------------------------------------------------------------

    def pull_week(self, week: Week, refresh: bool = False, force: bool = False) -> None:
        if self.is_frozen(week) and not force:
            raise FrozenWeekError(f"{self.season} week {week.number} is frozen")
        start, end = self.pull_range(week)
        key = f"week-{week.number:02d}"
        if "statsapi" in self.sources:
            self._pull_statsapi(key, start, end, refresh)
        if "savant" in self.sources:
            self._pull_statcast(start, end, refresh)

    def _pull_statsapi(self, key: str, start: date, end: date, refresh: bool) -> None:
        src = statsapi.SOURCE
        data, url = statsapi.fetch_schedule(self.http, start, end)
        self.raw.put_bytes(src, "schedule", self.season, key, "json", data, url)
        games = statsapi.parse_schedule(json.loads(data))
        player_ids: set[int] = set()
        for game in games:
            if not game["final"]:
                continue
            pk = str(game["game_pk"])
            if refresh or not self.raw.has(src, "boxscore", self.season, pk, "json"):
                body, burl = statsapi.fetch_boxscore(self.http, game["game_pk"])
                self.raw.put_bytes(src, "boxscore", self.season, pk, "json", body, burl)
            box = self.raw.read_json(src, "boxscore", self.season, pk)
            for side in ("away", "home"):
                team = box["teams"][side]
                player_ids.update(team.get("batters", []))
                player_ids.update(team.get("pitchers", []))
        for i, (body, purl) in enumerate(statsapi.fetch_people(self.http, player_ids)):
            self.raw.put_bytes(src, "people", self.season, f"{key}-{i:02d}", "json", body, purl)
        body, turl = statsapi.fetch_transactions(self.http, start, end)
        self.raw.put_bytes(src, "transactions", self.season, key, "json", body, turl)

    def _pull_statcast(self, start: date, end: date, refresh: bool) -> None:
        for day in _days(start, end):
            if day in self.calendar.off_days:
                continue
            key = day.isoformat()
            if not refresh and self.raw.has(savant.SOURCE, "statcast", self.season, key, "parquet"):
                continue
            frame = savant.fetch_statcast_day(day)
            self.raw.put_frame(
                savant.SOURCE, "statcast", self.season, key, frame, savant.statcast_url(day)
            )

    def pull_leaderboards(self, as_of: date, season_final: bool = False) -> None:
        """Season-to-date leaderboards and projections as of `as_of`.

        Savant leaderboards only exist as "today", so a backfill gets the
        season-final boards. FanGraphs takes a date window, so its boards are
        true as-of pulls; `season_final` asks for the whole season instead.
        """
        key = as_of.isoformat()
        if "savant" in self.sources:
            for name in savant.LEADERBOARDS:
                try:
                    frame, url = savant.fetch_leaderboard(self.http, name, self.season)
                except Exception as exc:  # one broken board must not sink the rest
                    log.warning("Savant %s leaderboard failed: %s", name, exc)
                    continue
                self.raw.put_frame(savant.SOURCE, name, self.season, key, frame, url)
        if "fangraphs" in self.sources:
            end = None if season_final else as_of
            for dataset in fangraphs.LEADERBOARDS:
                body, url = fangraphs.fetch_leaders(self.http, dataset, self.season, end)
                self.raw.put_bytes(fangraphs.SOURCE, dataset, self.season, key, "json", body, url)
            if not season_final:
                for system in PROJECTION_SYSTEMS:
                    for stats, label in fangraphs.PROJECTION_STATS.items():
                        body, url = fangraphs.fetch_projections(self.http, system, stats)
                        dataset = f"projections_{system}_{label}"
                        self.raw.put_bytes(
                            fangraphs.SOURCE, dataset, self.season, key, "json", body, url
                        )
            if not self.raw.has("chadwick", "register", self.season, key, "parquet"):
                register = ids.fetch_register()
                self.raw.put_frame(
                    "chadwick", "register", self.season, key, register, "pybaseball:chadwick"
                )

    # --- normalize -----------------------------------------------------------

    def normalize_week(self, week: Week, force: bool = False) -> dict[str, pd.DataFrame]:
        if self.is_frozen(week) and not force:
            raise FrozenWeekError(f"{self.season} week {week.number} is frozen")
        key = f"week-{week.number:02d}"
        tables: dict[str, pd.DataFrame] = {}
        src = statsapi.SOURCE
        if self.raw.has(src, "schedule", self.season, key, "json"):
            raw_schedule = self.raw.read_json(src, "schedule", self.season, key)
            schedule = statsapi.parse_schedule(raw_schedule)
            games = [g for g in schedule if self._belongs(g["official_date"], week)]
            batting, pitching = [], []
            for g in games:
                pk = str(g["game_pk"])
                if g["final"] and self.raw.has(src, "boxscore", self.season, pk, "json"):
                    box = self.raw.read_json(src, "boxscore", self.season, pk)
                    b, p = statsapi.parse_boxscore(box, g)
                    batting += b
                    pitching += p
            tables["games"] = _frame(games)
            tables["batting"] = _frame(batting)
            tables["pitching"] = _frame(pitching)
            tables["league_totals"] = league_totals(tables["games"], tables["batting"], week)
            if self.raw.has(src, "transactions", self.season, key, "json"):
                tx = statsapi.parse_transactions(
                    self.raw.read_json(src, "transactions", self.season, key)
                )
                tables["transactions"] = _frame(tx)
        start, end = self.pull_range(week)
        days = [
            d.isoformat()
            for d in _days(start, end)
            if self.raw.has(savant.SOURCE, "statcast", self.season, d.isoformat(), "parquet")
        ]
        if days:
            raw = pd.concat(
                [self.raw.read_frame(savant.SOURCE, "statcast", self.season, d) for d in days],
                ignore_index=True,
            )
            tables["pitches"] = savant.clean_pitches(raw)
        out = self.week_dir(week)
        out.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_parquet(out / f"{name}.parquet", index=False)
        self.normalize_players()
        return tables

    def _belongs(self, day: date, week: Week) -> bool:
        found = week_for_date(self.weeks, day)
        return found is not None and found.number == week.number

    def normalize_players(self) -> pd.DataFrame:
        rows = []
        for key in self.raw.keys(statsapi.SOURCE, "people", self.season):
            rows += statsapi.parse_people(
                self.raw.read_json(statsapi.SOURCE, "people", self.season, key)
            )
        players = _frame(rows)
        if not players.empty:
            players = players.drop_duplicates("mlbam_id", keep="last")
        path = self.clean_root / str(self.season) / "players.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        players.to_parquet(path, index=False)
        return players

    def normalize_leaderboards(self, as_of: date) -> dict[str, pd.DataFrame]:
        key = as_of.isoformat()
        tables: dict[str, pd.DataFrame] = {}
        for name in savant.LEADERBOARDS:
            if self.raw.has(savant.SOURCE, name, self.season, key, "parquet"):
                raw = self.raw.read_frame(savant.SOURCE, name, self.season, key)
                tables[name] = savant.clean_leaderboard(name, raw, self.season, as_of)
        fg_map = {}
        if self.raw.has("chadwick", "register", self.season, key, "parquet"):
            fg_map = ids.fangraphs_to_mlbam(
                self.raw.read_frame("chadwick", "register", self.season, key)
            )
        fg_sets = [f"fg_{d}" for d in fangraphs.LEADERBOARDS] + [
            f"fg_projections_{s}_{label}"
            for s in PROJECTION_SYSTEMS
            for label in fangraphs.PROJECTION_STATS.values()
        ]
        for name in fg_sets:
            dataset = name.removeprefix("fg_")
            if self.raw.has(fangraphs.SOURCE, dataset, self.season, key, "json"):
                body = self.raw.read_json(fangraphs.SOURCE, dataset, self.season, key)
                matched, unmatched = fangraphs.clean(
                    fangraphs.rows(body), self.season, as_of, fg_map
                )
                tables[name] = matched
                if not unmatched.empty:
                    log.warning("%s: %d rows with no MLBAM ID", name, len(unmatched))
                    tables[f"{name}_unmatched"] = unmatched
        out = self.asof_dir(as_of)
        out.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_parquet(out / f"{name}.parquet", index=False)
        return tables

    # --- freeze --------------------------------------------------------------

    def freeze_week(self, week: Week, as_of: date | None = None) -> dict:
        """Record exactly which clean tables the sim week runs on."""
        out = self.week_dir(week)
        tables = {p.stem: _sha256(p) for p in sorted(out.glob("*.parquet"))}
        missing = [t for t in ("games", "batting", "pitching") if t not in tables]
        if missing:
            raise RuntimeError(f"week {week.number} is missing clean tables: {missing}")
        record = {
            "season": self.season,
            "week": week.number,
            "start": week.start.isoformat(),
            "end": week.end.isoformat(),
            "tables": tables,
            "leaderboards_as_of": as_of.isoformat() if as_of else None,
            "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        (out / "FROZEN.json").write_text(json.dumps(record, indent=2) + "\n")
        return record

    # --- whole-week and whole-season runs ------------------------------------

    def run_week(self, week: Week, freeze: bool = True, refresh: bool = False) -> None:
        """The Monday job: close out the previous MLB week and freeze it."""
        self.pull_week(week, refresh=refresh)
        self.normalize_week(week)
        as_of = min(week.end, self.calendar.final_day)
        self.pull_leaderboards(as_of)
        self.normalize_leaderboards(as_of)
        if freeze:
            self.freeze_week(week, as_of)

    def backfill(self, weeks: Iterable[Week] | None = None, refresh: bool = False) -> None:
        """Load a completed season through the same pull/normalize/freeze path."""
        for week in weeks or self.weeks:
            if self.is_frozen(week):
                log.info("week %s already frozen, skipping", week.number)
                continue
            log.info("backfilling %s week %s", self.season, week.number)
            self.pull_week(week, refresh=refresh)
            self.normalize_week(week)
            self.freeze_week(week, self.calendar.final_day)
        self.pull_leaderboards(self.calendar.final_day, season_final=True)
        self.normalize_leaderboards(self.calendar.final_day)

    def check_calendar(self) -> list[str]:
        start = date(self.season, 3, 1)
        end = date(self.season, 10, 15)
        data, url = statsapi.fetch_schedule(self.http, start, end)
        self.raw.put_bytes(
            statsapi.SOURCE, "schedule", self.season, "full-season", "json", data, url
        )
        actual = calendar_check.derive_calendar(json.loads(data))
        return calendar_check.compare(self.calendar, actual)


def league_totals(games: pd.DataFrame, batting: pd.DataFrame, week: Week) -> pd.DataFrame:
    """League batting totals and run-environment rates for one week."""
    cols = ["pa", "ab", "h", "doubles", "triples", "hr", "bb", "ibb", "hbp", "so", "sf", "r"]
    sums = {c: int(batting[c].sum()) if c in batting else 0 for c in cols}
    played = int(games["final"].sum()) if "final" in games else 0
    row = {"week": week.number, "start": week.start, "games": played, **sums}
    row.update(rates(row))
    return pd.DataFrame([row])


def rates(t: dict) -> dict:
    """R/G per team game, K%, BB%, HR per PA and BABIP from summed counts."""
    pa, games = t["pa"], t["games"]
    bip = t["ab"] - t["so"] - t["hr"] + t["sf"]
    return {
        "r_per_g": t["r"] / (2 * games) if games else None,
        "k_pct": t["so"] / pa if pa else None,
        "bb_pct": t["bb"] / pa if pa else None,
        "hr_per_pa": t["hr"] / pa if pa else None,
        "babip": (t["h"] - t["hr"]) / bip if bip else None,
    }


def lineup_locks(games: pd.DataFrame, weeks: list[Week]) -> dict[int, datetime]:
    """Week number -> 1 hour before that week's first MLB first pitch."""
    locks: dict[int, datetime] = {}
    for _, g in games.iterrows():
        week = week_for_date(weeks, g["official_date"])
        # Openers before week 1's Monday don't set its lock.
        if week is None or g["official_date"] < week.start or not g.get("start_time_utc"):
            continue
        first = datetime.fromisoformat(str(g["start_time_utc"]).replace("Z", "+00:00"))
        lock = first - timedelta(hours=1)
        if week.number not in locks or lock < locks[week.number]:
            locks[week.number] = lock
    return locks


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if "mlbam_id" in frame:
        frame["mlbam_id"] = frame["mlbam_id"].astype("Int64")
    return frame
