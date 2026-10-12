"""Write a week's NPL rosters into the sim database."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from npl_sim.db.models import Player, RosterEntry, RosterTier, Season, SimWeek, Team
from npl_sim.rosters.npl_app import NplPlayer


@dataclass
class ImportReport:
    tiers: dict[str, Counter[RosterTier]] = field(default_factory=lambda: defaultdict(Counter))
    # Players the NPL app has without an MLBAM ID; they can't be simmed.
    unmatched: list[NplPlayer] = field(default_factory=list)
    # NPL team names with no team in the sim's league config.
    unknown_teams: set[str] = field(default_factory=set)
    # Players with no roster flag at all.
    untiered: list[NplPlayer] = field(default_factory=list)

    @property
    def unmatched_mlb_roster(self) -> list[NplPlayer]:
        return [p for p in self.unmatched if p.tier is RosterTier.MLB_ROSTER]


def get_or_create_week(session: Session, season: int, number: int, start: date) -> SimWeek:
    if start.weekday() != 0:
        raise ValueError(f"sim weeks start on a Monday; {start} is not")
    if session.get(Season, season) is None:
        session.add(Season(year=season))
    week = session.scalar(
        select(SimWeek).where(SimWeek.season_year == season, SimWeek.number == number)
    )
    if week is None:
        week = SimWeek(
            season_year=season, number=number, start_date=start, end_date=start + timedelta(days=6)
        )
        session.add(week)
        session.flush()
    elif week.start_date != start:
        raise ValueError(f"week {number} of {season} already starts on {week.start_date}")
    return week


def import_rosters(
    session: Session,
    players: Iterable[NplPlayer],
    week: SimWeek,
    teams: Iterable[str] | None = None,
) -> ImportReport:
    """Replace `week`'s roster entries with `players`.

    `teams` limits the import to those NPL team names (e.g. the two test teams).
    """
    wanted = set(teams) if teams is not None else None
    report = ImportReport()
    team_ids = {t.name: t.id for t in session.scalars(select(Team))}
    by_mlbam = {
        p.mlbam_id: p for p in session.scalars(select(Player).where(Player.mlbam_id.is_not(None)))
    }

    session.execute(delete(RosterEntry).where(RosterEntry.week_id == week.id))

    for npl in players:
        if wanted is not None and npl.team not in wanted:
            continue
        if npl.team not in team_ids:
            report.unknown_teams.add(npl.team)
            continue
        tier = npl.tier
        if tier is None:
            report.untiered.append(npl)
            continue
        if npl.mlb_id is None:
            report.unmatched.append(npl)
            continue

        player = by_mlbam.get(npl.mlb_id)
        if player is None:
            player = Player(mlbam_id=npl.mlb_id, name=npl.name, position=npl.position)
            session.add(player)
            by_mlbam[npl.mlb_id] = player
        player.name = npl.name
        player.position = npl.position
        player.mlb_team = npl.mlb_org
        player.throws = npl.throws or player.throws
        session.flush()

        session.add(
            RosterEntry(week_id=week.id, team_id=team_ids[npl.team], player_id=player.id, tier=tier)
        )
        report.tiers[npl.team][tier] += 1

    session.flush()
    return report
