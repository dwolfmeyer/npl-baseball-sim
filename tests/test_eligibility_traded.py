from npl_sim.rosters.eligibility import parse_people_fielding


def test_traded_player_uses_combined_split_not_sum():
    def split(games, team=None, num_teams=None):
        s = {"season": "2025", "position": {"abbreviation": "CF"}, "stat": {"games": games}}
        if team:
            s["team"] = {"name": team}
        if num_teams:
            s["numTeams"] = num_teams
        return s

    raw = {
        "people": [
            {
                "id": 656775,
                "batSide": {"code": "L"},
                "pitchHand": {"code": "L"},
                "stats": [
                    {"splits": [split(130, num_teams=2), split(88, "BAL"), split(42, "NYM")]}
                ],
            }
        ]
    }
    assert parse_people_fielding(raw, [2025])[656775].games == {2025: {"CF": 130}}


def test_per_team_splits_sum_without_combined_row():
    raw = {
        "people": [
            {
                "id": 1,
                "stats": [
                    {
                        "splits": [
                            {
                                "season": "2025",
                                "position": {"abbreviation": "SS"},
                                "stat": {"games": 10},
                                "team": {"name": "A"},
                            },
                            {
                                "season": "2025",
                                "position": {"abbreviation": "SS"},
                                "stat": {"games": 5},
                                "team": {"name": "B"},
                            },
                        ]
                    }
                ],
            }
        ]
    }
    assert parse_people_fielding(raw, [2025])[1].games == {2025: {"SS": 15}}
