"""NPL app roster rules, ported from jeremyjbowers/npl `npl/rosters.py`.

The NPL app loads the league's roster workbook into its own database, keyed
by MLBAM ID, with one boolean flag per roster section. A player is on the
active MLB Roster when he is on the 40-man and on none of the other lists;
that is the only sim-eligible tier.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from npl_sim.db.models import RosterTier

# A 40-man player on any of these lists is not on the active MLB roster.
MLB_BLOCKERS = (
    "roster_tripleA",
    "roster_tripleA_option",
    "roster_7dayIL",
    "roster_56dayIL",
    "roster_eosIL",
    "roster_restricted",
    "roster_outrighted",
    "roster_foreign",
    "roster_retired",
    "roster_nonroster",
    "roster_doubleA",
    "roster_singleA",
)

ROSTER_FLAGS = ("roster_40man", *MLB_BLOCKERS)

# Section order follows the team page (npl/rosters.py SECTION_SPECS).
_FLAG_TIERS = (
    ("roster_7dayIL", RosterTier.IL_7),
    ("roster_56dayIL", RosterTier.IL_56),
    ("roster_eosIL", RosterTier.IL_EOS),
    ("roster_restricted", RosterTier.RESTRICTED),
    ("roster_foreign", RosterTier.FOREIGN),
    ("roster_retired", RosterTier.RETIRED),
    ("roster_outrighted", RosterTier.OUTRIGHTED),
    ("roster_nonroster", RosterTier.NON_ROSTER),
    ("roster_tripleA", RosterTier.AAA),
    ("roster_tripleA_option", RosterTier.AAA),
    ("roster_doubleA", RosterTier.AA),
    ("roster_singleA", RosterTier.A),
)


@dataclass(frozen=True)
class NplPlayer:
    """One rostered player as the NPL app stores him."""

    mlb_id: int | None
    name: str
    position: str
    mlb_org: str | None
    team: str
    flags: frozenset[str] = field(default_factory=frozenset)
    scoresheet_id: str | None = None

    @property
    def is_active_mlb(self) -> bool:
        return "roster_40man" in self.flags and not any(f in self.flags for f in MLB_BLOCKERS)

    @property
    def tier(self) -> RosterTier | None:
        if self.is_active_mlb:
            return RosterTier.MLB_ROSTER
        for flag, tier in _FLAG_TIERS:
            if flag in self.flags:
                return tier
        return None

    @property
    def throws(self) -> str | None:
        return {"LHP": "L", "RHP": "R"}.get(self.position.upper())


def parse_mlb_id(value: object) -> int | None:
    """The NPL app stores MLBAM IDs as text; anything non-numeric is unmatched."""
    text = "" if value is None else str(value).strip()
    return int(text) if text.isdigit() else None


def from_row(row: Mapping[str, object], team: str) -> NplPlayer:
    return NplPlayer(
        mlb_id=parse_mlb_id(row.get("mlb_id")),
        name=str(row.get("name") or row.get("raw_name") or "").strip(),
        position=str(row.get("position") or "").strip(),
        mlb_org=(str(row["mlb_org"]).strip() or None) if row.get("mlb_org") else None,
        team=team,
        flags=frozenset(f for f in ROSTER_FLAGS if row.get(f)),
        scoresheet_id=str(row["scoresheet_id"]) if row.get("scoresheet_id") else None,
    )
