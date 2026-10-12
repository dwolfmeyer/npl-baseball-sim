"""Who's who in one game, read from the event log's `game_start` event."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from npl_sim.engine.detail import POSITION_TITLE

ORDINAL = {1: "1st", 2: "2nd", 3: "3rd"}
BASE_NAME = {"1B": "1st", "2B": "2nd", "3B": "3rd", "score": "home"}
NEXT_BASE = {"B": "1st", "1B": "2nd", "2B": "3rd", "3B": "home"}


def ordinal(n: int) -> str:
    if n in ORDINAL:
        return ORDINAL[n]
    return f"{n}th"


@dataclass
class Team:
    name: str
    abbr: str
    lineup: list[dict]
    pitchers: list[dict]
    starter: str
    bench: list[dict] = field(default_factory=list)
    positions: dict[str, str] = field(default_factory=dict)  # position -> player id, now

    def __post_init__(self) -> None:
        self.positions = {p["position"]: p["id"] for p in self.lineup}

    def substitute(self, e: dict) -> None:
        """Apply a `defensive_sub` event to who plays where."""
        if e.get("moved"):
            self.positions[e["moved"]["position"]] = e["moved"]["player"]
        self.positions[e["position"]] = e["into"]


class Roster:
    """Names and positions for both teams, keyed by side ('away'/'home')."""

    def __init__(self, events: list[dict]):
        start = next((e for e in events if e["type"] == "game_start"), None)
        if start is None:
            raise ValueError("event log has no game_start event")
        self.teams = {side: Team(**start[side]) for side in ("away", "home")}
        self.full: dict[str, str] = {}
        self.short: dict[str, str] = {}
        for team in self.teams.values():
            people = team.lineup + team.pitchers + team.bench
            for p in people:
                self.full[p["id"]] = p["name"]
            lasts = Counter(_last(p["name"]) for p in people)
            for p in people:
                last = _last(p["name"])
                if lasts[last] > 1 and " " in p["name"]:
                    self.short[p["id"]] = f"{p['name'][0]}. {last}"
                else:
                    self.short[p["id"]] = last

    def side_of(self, team_name: str) -> str:
        return "away" if self.teams["away"].name == team_name else "home"

    def fielder(self, side: str, position: str, pitcher: str) -> str:
        """Full name of whoever plays `position` for `side`; P is the current pitcher."""
        if position == "P":
            return self.full[pitcher]
        pid = self.teams[side].positions.get(position)
        return self.full.get(pid, POSITION_TITLE[position])

    def fielder_id(self, side: str, position: str, pitcher: str) -> str | None:
        if position == "P":
            return pitcher
        return self.teams[side].positions.get(position)


def _last(name: str) -> str:
    parts = name.split()
    if len(parts) > 1 and parts[-1].rstrip(".") in ("Jr", "Sr", "II", "III", "IV"):
        return " ".join(parts[-2:])
    return parts[-1] if parts else name
