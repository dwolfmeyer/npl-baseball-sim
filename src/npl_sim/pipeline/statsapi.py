"""MLB Stats API (statsapi.mlb.com): schedule, box scores, players and
transactions. Fetchers return raw bytes; parsers turn the stored JSON into
clean rows keyed by MLBAM ID and gamePk."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from npl_sim.pipeline.http import Http

BASE = "https://statsapi.mlb.com/api/v1"
SOURCE = "statsapi"
PEOPLE_BATCH = 100

# States in which a scheduled game produced no final box score.
_NOT_PLAYED = ("Postponed", "Cancelled", "Suspended")


def schedule_url(start: date, end: date) -> tuple[str, dict]:
    return f"{BASE}/schedule", {
        "sportId": 1,
        "gameType": "R",
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
    }


def fetch_schedule(http: Http, start: date, end: date) -> tuple[bytes, str]:
    url, params = schedule_url(start, end)
    return http.get(url, params), _full(url, params)


def fetch_boxscore(http: Http, game_pk: int) -> tuple[bytes, str]:
    url = f"{BASE}/game/{game_pk}/boxscore"
    return http.get(url), url


def fetch_people(http: Http, ids: Iterable[int]) -> list[tuple[bytes, str]]:
    ids = sorted(set(ids))
    out = []
    for i in range(0, len(ids), PEOPLE_BATCH):
        params = {"personIds": ",".join(map(str, ids[i : i + PEOPLE_BATCH]))}
        url = f"{BASE}/people"
        out.append((http.get(url, params), _full(url, params)))
    return out


def fetch_transactions(http: Http, start: date, end: date) -> tuple[bytes, str]:
    url = f"{BASE}/transactions"
    params = {"sportId": 1, "startDate": start.isoformat(), "endDate": end.isoformat()}
    return http.get(url, params), _full(url, params)


def _full(url: str, params: dict) -> str:
    return url + "?" + "&".join(f"{k}={v}" for k, v in params.items())


# --- parsers -----------------------------------------------------------------


def parse_schedule(raw: dict) -> list[dict]:
    """One row per regular-season game, deduplicated by gamePk.

    A postponed or suspended game appears on more than one date; the row
    kept is the one where it was played (final), else its latest listing.
    """
    games: dict[int, dict] = {}
    for day in raw.get("dates", []):
        for g in day.get("games", []):
            if g.get("gameType", "R") != "R":
                continue
            status = g.get("status", {})
            detailed = status.get("detailedState", "")
            final = status.get("abstractGameState") == "Final" and not detailed.startswith(
                _NOT_PLAYED
            )
            away, home = g["teams"]["away"], g["teams"]["home"]
            row = {
                "game_pk": g["gamePk"],
                "official_date": date.fromisoformat(g.get("officialDate", day["date"])),
                "start_time_utc": g.get("gameDate"),
                "status": detailed,
                "final": final,
                "away_team_id": away["team"]["id"],
                "away_team": away["team"].get("name"),
                "home_team_id": home["team"]["id"],
                "home_team": home["team"].get("name"),
                "away_score": away.get("score"),
                "home_score": home.get("score"),
                "venue_id": g.get("venue", {}).get("id"),
                "venue": g.get("venue", {}).get("name"),
                "double_header": g.get("doubleHeader", "N"),
                "game_number": g.get("gameNumber", 1),
            }
            prev = games.get(row["game_pk"])
            if prev is None or row["final"] or not prev["final"]:
                games[row["game_pk"]] = row
    return sorted(games.values(), key=lambda r: (r["official_date"], r["game_pk"]))


BATTING_FIELDS = {
    "pa": "plateAppearances",
    "ab": "atBats",
    "r": "runs",
    "h": "hits",
    "doubles": "doubles",
    "triples": "triples",
    "hr": "homeRuns",
    "rbi": "rbi",
    "bb": "baseOnBalls",
    "ibb": "intentionalWalks",
    "hbp": "hitByPitch",
    "so": "strikeOuts",
    "sf": "sacFlies",
    "sh": "sacBunts",
    "sb": "stolenBases",
    "cs": "caughtStealing",
    "gidp": "groundIntoDoublePlay",
}

PITCHING_FIELDS = {
    "gs": "gamesStarted",
    "outs": "outs",
    "bf": "battersFaced",
    "pitches": "numberOfPitches",
    "strikes": "strikes",
    "h": "hits",
    "r": "runs",
    "er": "earnedRuns",
    "bb": "baseOnBalls",
    "ibb": "intentionalWalks",
    "so": "strikeOuts",
    "hr": "homeRuns",
    "hbp": "hitByPitch",
    "wp": "wildPitches",
    "bk": "balks",
    "w": "wins",
    "l": "losses",
    "sv": "saves",
    "hld": "holds",
    "bs": "blownSaves",
    "inherited": "inheritedRunners",
    "inherited_scored": "inheritedRunnersScored",
}


def _outs(ip: str | None) -> int:
    """'6.2' innings pitched -> 20 outs."""
    if not ip:
        return 0
    whole, _, part = str(ip).partition(".")
    return int(whole) * 3 + int(part or 0)


def parse_boxscore(raw: dict, game: dict) -> tuple[list[dict], list[dict]]:
    """Batting and pitching lines for every player who appeared."""
    batting, pitching = [], []
    for side, opp in (("away", "home"), ("home", "away")):
        team = raw["teams"][side]
        team_id = team["team"]["id"]
        opp_id = raw["teams"][opp]["team"]["id"]
        common = {
            "game_pk": game["game_pk"],
            "official_date": game["official_date"],
            "team_id": team_id,
            "opponent_id": opp_id,
            "home": side == "home",
            "venue_id": game.get("venue_id"),
        }
        players = team.get("players", {})
        for pid in team.get("batters", []):
            p = players.get(f"ID{pid}", {})
            stats = p.get("stats", {}).get("batting", {})
            order = str(p.get("battingOrder", ""))
            positions = [pos.get("abbreviation") for pos in p.get("allPositions", [])]
            row = {
                **common,
                "mlbam_id": pid,
                "batting_order": int(order[0]) if order[:1].isdigit() else None,
                "started": order.endswith("00"),
                "positions": ",".join(x for x in positions if x),
            }
            row.update({k: int(stats.get(v, 0) or 0) for k, v in BATTING_FIELDS.items()})
            batting.append(row)
        for i, pid in enumerate(team.get("pitchers", [])):
            stats = players.get(f"ID{pid}", {}).get("stats", {}).get("pitching", {})
            row = {**common, "mlbam_id": pid, "appearance": i + 1}
            row.update({k: int(stats.get(v, 0) or 0) for k, v in PITCHING_FIELDS.items()})
            if "outs" not in stats:
                row["outs"] = _outs(stats.get("inningsPitched"))
            pitching.append(row)
    return batting, pitching


def parse_people(raw: dict) -> list[dict]:
    rows = []
    for p in raw.get("people", []):
        rows.append(
            {
                "mlbam_id": p["id"],
                "name": p.get("fullName"),
                "birth_date": p.get("birthDate"),
                "bats": p.get("batSide", {}).get("code"),
                "throws": p.get("pitchHand", {}).get("code"),
                "primary_position": p.get("primaryPosition", {}).get("abbreviation"),
                "mlb_debut": p.get("mlbDebutDate"),
                "active": p.get("active"),
            }
        )
    return rows


def parse_transactions(raw: dict) -> list[dict]:
    rows = []
    for t in raw.get("transactions", []):
        person = t.get("person") or {}
        if "id" not in person:
            continue
        rows.append(
            {
                "transaction_id": t.get("id"),
                "mlbam_id": person["id"],
                "name": person.get("fullName"),
                "date": t.get("date"),
                "effective_date": t.get("effectiveDate"),
                "type_code": t.get("typeCode"),
                "type": t.get("typeDesc"),
                "from_team_id": (t.get("fromTeam") or {}).get("id"),
                "to_team_id": (t.get("toTeam") or {}).get("id"),
                "description": t.get("description"),
            }
        )
    return rows
