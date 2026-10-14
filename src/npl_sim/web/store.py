"""Where the web app keeps lineup cards and simmed games for now: JSON files
under one directory (`NPL_WEB_DATA`, default `data/web`). The database tables
for cards and results come with the season runner (Milestone 9). The season to
date (`season.json`) is the running `SeasonStats` of every week simmed since it
was last reset."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from npl_sim.season_stats import SeasonStats
from npl_sim.web.lineup import LineupCard

WEB_DATA_ENV = "NPL_WEB_DATA"


def slug(team: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", team.lower()).strip("-")


class Store:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or os.environ.get(WEB_DATA_ENV) or "data/web")
        (self.root / "lineups").mkdir(parents=True, exist_ok=True)
        (self.root / "games").mkdir(parents=True, exist_ok=True)
        (self.root / "weeks").mkdir(parents=True, exist_ok=True)

    def load_card(self, team: str) -> LineupCard | None:
        path = self.root / "lineups" / f"{slug(team)}.json"
        if not path.exists():
            return None
        return LineupCard.from_json(json.loads(path.read_text())["card"])

    def card_saved_at(self, team: str) -> str | None:
        path = self.root / "lineups" / f"{slug(team)}.json"
        return json.loads(path.read_text())["saved_at"] if path.exists() else None

    def save_card(self, team: str, card: LineupCard) -> None:
        path = self.root / "lineups" / f"{slug(team)}.json"
        saved_at = datetime.now(UTC).isoformat(timespec="seconds")
        path.write_text(json.dumps({"team": team, "saved_at": saved_at, "card": card.to_json()}))

    def save_game(self, game: dict) -> str:
        existing = [int(p.stem) for p in (self.root / "games").glob("*.json") if p.stem.isdigit()]
        game_id = str(max(existing, default=0) + 1)
        game = {**game, "id": game_id, "played_at": datetime.now(UTC).isoformat(timespec="seconds")}
        (self.root / "games" / f"{game_id}.json").write_text(json.dumps(game))
        return game_id

    def load_game(self, game_id: str) -> dict | None:
        path = self.root / "games" / f"{game_id}.json"
        return json.loads(path.read_text()) if game_id.isdigit() and path.exists() else None

    def save_week(self, week: dict) -> str:
        existing = [int(p.stem) for p in (self.root / "weeks").glob("*.json") if p.stem.isdigit()]
        week_id = str(max(existing, default=0) + 1)
        week = {**week, "id": week_id, "played_at": datetime.now(UTC).isoformat(timespec="seconds")}
        (self.root / "weeks" / f"{week_id}.json").write_text(json.dumps(week))
        return week_id

    def load_week(self, week_id: str) -> dict | None:
        path = self.root / "weeks" / f"{week_id}.json"
        return json.loads(path.read_text()) if week_id.isdigit() and path.exists() else None

    def recent_weeks(self, limit: int = 10) -> list[dict]:
        paths = sorted(
            (p for p in (self.root / "weeks").glob("*.json") if p.stem.isdigit()),
            key=lambda p: int(p.stem),
            reverse=True,
        )
        return [json.loads(p.read_text()) for p in paths[:limit]]

    def load_season(self) -> tuple[SeasonStats, int]:
        """The season to date and how many weeks it holds."""
        path = self.root / "season.json"
        if not path.exists():
            return SeasonStats(), 0
        data = json.loads(path.read_text())
        return SeasonStats.from_json(data["stats"]), data["weeks"]

    def save_season(self, stats: SeasonStats, weeks: int) -> None:
        path = self.root / "season.json"
        path.write_text(json.dumps({"weeks": weeks, "stats": stats.to_json()}))

    def reset_season(self) -> None:
        (self.root / "season.json").unlink(missing_ok=True)

    def recent_games(self, limit: int = 20) -> list[dict]:
        paths = sorted(
            (p for p in (self.root / "games").glob("*.json") if p.stem.isdigit()),
            key=lambda p: int(p.stem),
            reverse=True,
        )
        return [json.loads(p.read_text()) for p in paths[:limit]]
