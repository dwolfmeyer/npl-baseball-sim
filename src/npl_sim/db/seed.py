"""Sync a season's league alignment from config into the database."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from npl_sim.db.models import Division, League, Season, Team, TeamSeason
from npl_sim.league import LeagueFile


def sync_alignment(session: Session, league_file: LeagueFile, season: int) -> None:
    """Create or update leagues, divisions, teams and `season`'s team-division links."""
    alignment = league_file.for_season(season)
    if session.get(Season, season) is None:
        session.add(Season(year=season))

    for league_cfg in alignment.leagues:
        league = session.scalar(select(League).where(League.name == league_cfg.name))
        if league is None:
            league = League(name=league_cfg.name)
            session.add(league)
        league.dh = league_cfg.dh

        for div_cfg in league_cfg.divisions:
            division = session.scalar(
                select(Division).where(Division.league == league, Division.name == div_cfg.name)
            )
            if division is None:
                division = Division(league=league, name=div_cfg.name)
                session.add(division)

            for team_name in div_cfg.teams:
                team = session.scalar(select(Team).where(Team.name == team_name))
                if team is None:
                    team = Team(name=team_name)
                    session.add(team)
                session.flush()
                link = session.scalar(
                    select(TeamSeason).where(
                        TeamSeason.season_year == season, TeamSeason.team_id == team.id
                    )
                )
                if link is None:
                    session.add(TeamSeason(season_year=season, team=team, division=division))
                else:
                    link.division = division
    session.flush()
