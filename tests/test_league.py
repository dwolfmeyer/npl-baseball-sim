import copy

import pytest
import yaml
from pydantic import ValidationError

from npl_sim.league import DEFAULT_CONFIG, LeagueFile, load_league_file, team_slots


@pytest.fixture
def raw():
    with open(DEFAULT_CONFIG) as fh:
        return yaml.safe_load(fh)


def test_shipped_config_has_24_teams_in_six_divisions():
    alignment = load_league_file().for_season(2026)
    slots = team_slots(alignment)
    assert len(slots) == 24
    assert len({(s.league, s.division) for s in slots}) == 6
    assert all(league.dh for league in alignment.leagues)


def test_test_teams_are_where_the_spec_puts_them():
    slots = {s.team: s for s in team_slots(load_league_file().for_season(2026))}
    assert (slots["Midwest Ducksnorts"].league, slots["Midwest Ducksnorts"].division) == (
        "AL",
        "Splendid Splinter",
    )
    assert (slots["Hill Valley McFlys"].league, slots["Hill Valley McFlys"].division) == (
        "AL",
        "Oil Can",
    )


def test_duplicate_team_is_rejected(raw):
    bad = copy.deepcopy(raw)
    bad["alignments"][0]["leagues"][1]["divisions"][0]["teams"][0] = "Midwest Ducksnorts"
    with pytest.raises(ValidationError, match="listed twice"):
        LeagueFile.model_validate(bad)


def test_wrong_division_size_is_rejected(raw):
    bad = copy.deepcopy(raw)
    bad["alignments"][0]["leagues"][0]["divisions"][0]["teams"].pop()
    with pytest.raises(ValidationError, match="expected 4 teams"):
        LeagueFile.model_validate(bad)


def test_realignment_picks_latest_alignment_for_season(raw):
    realigned = copy.deepcopy(raw["alignments"][0])
    realigned["from_season"] = 2028
    al_divs = realigned["leagues"][0]["divisions"]
    # Swap two teams between AL divisions.
    al_divs[0]["teams"][0], al_divs[1]["teams"][0] = al_divs[1]["teams"][0], al_divs[0]["teams"][0]
    lf = LeagueFile.model_validate({"alignments": [raw["alignments"][0], realigned]})

    def div_of(season, team):
        return {s.team: s.division for s in team_slots(lf.for_season(season))}[team]

    assert div_of(2027, "Milwaukee DockHounds") == "Oil Can"
    assert div_of(2028, "Milwaukee DockHounds") == "Splendid Splinter"
    with pytest.raises(LookupError):
        lf.for_season(2025)
