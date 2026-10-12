"""Games by position and batting hands for rostered players, saved for the
lineup card. The eligibility rule itself (20 MLB games at a position last
season or 10 this season) lives in `npl_sim.eligibility`.

Games come from the MLB Stats API's season fielding splits, which the lineup
card can use before the pipeline has a season of box scores. They are saved,
with each player's batting and throwing hand, in one JSON file per NPL season
(`data/eligibility/<season>.json`):

    {"season": 2026, "players": {"<mlbam id>": {"bats": "L", "throws": "R",
     "games": {"2025": {"SS": 152}, "2026": {"SS": 40, "3B": 3}}}}}
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from npl_sim.eligibility import FIELD_POSITIONS, eligible_positions, qualifying
from npl_sim.pipeline.http import Http
from npl_sim.pipeline.statsapi import BASE, PEOPLE_BATCH

DEFAULT_DIR = Path("data/eligibility")


@dataclass
class PlayerRecord:
    bats: str | None = None
    throws: str | None = None
    games: dict[int, dict[str, int]] = field(default_factory=dict)  # season -> pos -> games


@dataclass
class Eligibility:
    season: int
    players: dict[int, PlayerRecord] = field(default_factory=dict)

    def _games(self, mlbam_id: int) -> tuple[dict | None, dict | None]:
        rec = self.players.get(mlbam_id)
        if rec is None:
            return None, None
        return rec.games.get(self.season - 1), rec.games.get(self.season)

    def positions(self, mlbam_id: int, listed: str | None = None) -> set[str]:
        """Field positions this player may play (DH aside), falling back to his
        listed NPL position when he has no qualifying games."""
        return eligible_positions(*self._games(mlbam_id), listed=listed) - {"DH"}

    def qualifies(self, mlbam_id: int) -> bool:
        """Whether his eligibility comes from MLB games rather than the fallback."""
        return bool(qualifying(*self._games(mlbam_id)))

    def games_text(self, mlbam_id: int) -> str:
        """e.g. "SS 152, 3B 4 ('25) · SS 40 ('26)" for tooltips."""
        rec = self.players.get(mlbam_id)
        if rec is None:
            return "no MLB fielding record"
        parts = []
        for season in (self.season - 1, self.season):
            games = rec.games.get(season, {})
            if games:
                listed = ", ".join(
                    f"{p} {g}" for p, g in sorted(games.items(), key=lambda x: -x[1])
                )
                parts.append(f"{listed} ('{season % 100:02d})")
        return " · ".join(parts) or "no MLB fielding games"

    def to_json(self) -> dict:
        return {
            "season": self.season,
            "players": {
                str(pid): {
                    "bats": rec.bats,
                    "throws": rec.throws,
                    "games": {str(s): g for s, g in sorted(rec.games.items())},
                }
                for pid, rec in sorted(self.players.items())
            },
        }

    @classmethod
    def from_json(cls, data: dict) -> Eligibility:
        players = {
            int(pid): PlayerRecord(
                bats=rec.get("bats"),
                throws=rec.get("throws"),
                games={int(s): dict(g) for s, g in rec.get("games", {}).items()},
            )
            for pid, rec in data.get("players", {}).items()
        }
        return cls(season=int(data["season"]), players=players)


def path_for(season: int, root: Path | str = DEFAULT_DIR) -> Path:
    return Path(root) / f"{season}.json"


def load(season: int, root: Path | str = DEFAULT_DIR) -> Eligibility:
    path = path_for(season, root)
    if not path.exists():
        return Eligibility(season=season)
    return Eligibility.from_json(json.loads(path.read_text()))


def save(elig: Eligibility, root: Path | str = DEFAULT_DIR) -> Path:
    path = path_for(elig.season, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(elig.to_json(), indent=1) + "\n")
    return path


def parse_people_fielding(raw: dict, seasons: Iterable[int]) -> dict[int, PlayerRecord]:
    """Players' hands and games by position from a `/people` response
    hydrated with season fielding stats."""
    wanted = set(seasons)
    out = {}
    for person in raw.get("people", []):
        rec = PlayerRecord(
            bats=person.get("batSide", {}).get("code"),
            throws=person.get("pitchHand", {}).get("code"),
        )
        for block in person.get("stats", []):
            for split in block.get("splits", []):
                season = int(split.get("season") or 0)
                pos = split.get("position", {}).get("abbreviation")
                if season not in wanted or pos not in FIELD_POSITIONS:
                    continue
                games = int(split.get("stat", {}).get("games") or 0)
                by_pos = rec.games.setdefault(season, {})
                # Players traded mid-season have one split per team.
                by_pos[pos] = by_pos.get(pos, 0) + games
        out[int(person["id"])] = rec
    return out


def fetch(http: Http, ids: Iterable[int], season: int) -> Eligibility:
    """Pull last season's and this season's fielding games for these players."""
    ids = sorted(set(ids))
    seasons = (season - 1, season)
    elig = Eligibility(season=season)
    for i in range(0, len(ids), PEOPLE_BATCH):
        merged: dict[int, PlayerRecord] = {}
        for s in seasons:
            params = {
                "personIds": ",".join(map(str, ids[i : i + PEOPLE_BATCH])),
                "hydrate": f"stats(group=[fielding],type=[season],season={s})",
            }
            raw = json.loads(http.get(f"{BASE}/people", params))
            for pid, rec in parse_people_fielding(raw, [s]).items():
                into = merged.setdefault(pid, PlayerRecord(bats=rec.bats, throws=rec.throws))
                into.games.update(rec.games)
        elig.players.update(merged)
    return elig
