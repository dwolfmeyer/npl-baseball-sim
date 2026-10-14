"""League alignment config: leagues, divisions and teams, read from data.

The schedule generator and standings read structure from here so a
realignment is a config edit, never a code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "league.yaml"

LEAGUES = 2
DIVISIONS_PER_LEAGUE = 3
TEAMS_PER_DIVISION = 4


class DivisionConfig(BaseModel):
    name: str
    teams: list[str]
    # Two rival pairs who play 18; the other division pairs play 17.
    # Defaults to the first two and last two teams as listed.
    rivals: list[tuple[str, str]] | None = None

    @model_validator(mode="after")
    def _check_rivals(self) -> DivisionConfig:
        if self.rivals is None:
            return self
        paired = [t for pair in self.rivals for t in pair]
        if len(self.rivals) != 2 or sorted(paired) != sorted(self.teams):
            raise ValueError(f"{self.name}: rivals must pair off all four division teams")
        return self

    def rival_pairs(self) -> list[tuple[str, str]]:
        if self.rivals is not None:
            return [tuple(p) for p in self.rivals]
        t = self.teams
        return [(t[0], t[1]), (t[2], t[3])]


class LeagueConfig(BaseModel):
    name: str
    dh: bool = True
    divisions: list[DivisionConfig]


class Alignment(BaseModel):
    from_season: int
    wild_cards: int = Field(default=2, ge=0)  # wild card spots per league
    leagues: list[LeagueConfig]

    @model_validator(mode="after")
    def _check_shape(self) -> Alignment:
        if len(self.leagues) != LEAGUES:
            raise ValueError(f"expected {LEAGUES} leagues, got {len(self.leagues)}")
        seen: set[str] = set()
        for league in self.leagues:
            if len(league.divisions) != DIVISIONS_PER_LEAGUE:
                raise ValueError(
                    f"{league.name}: expected {DIVISIONS_PER_LEAGUE} divisions, "
                    f"got {len(league.divisions)}"
                )
            for div in league.divisions:
                if len(div.teams) != TEAMS_PER_DIVISION:
                    raise ValueError(
                        f"{league.name} {div.name}: expected {TEAMS_PER_DIVISION} teams, "
                        f"got {len(div.teams)}"
                    )
                for team in div.teams:
                    if team in seen:
                        raise ValueError(f"team listed twice: {team}")
                    seen.add(team)
        return self


class LeagueFile(BaseModel):
    alignments: list[Alignment] = Field(min_length=1)

    def for_season(self, season: int) -> Alignment:
        """The alignment in force for `season` (latest `from_season` <= season)."""
        eligible = [a for a in self.alignments if a.from_season <= season]
        if not eligible:
            raise LookupError(f"no alignment covers season {season}")
        return max(eligible, key=lambda a: a.from_season)


@dataclass(frozen=True)
class TeamSlot:
    team: str
    league: str
    division: str


def load_league_file(path: Path | str = DEFAULT_CONFIG) -> LeagueFile:
    with open(path) as fh:
        return LeagueFile.model_validate(yaml.safe_load(fh))


def team_slots(alignment: Alignment) -> list[TeamSlot]:
    return [
        TeamSlot(team=team, league=league.name, division=div.name)
        for league in alignment.leagues
        for div in league.divisions
        for team in div.teams
    ]
