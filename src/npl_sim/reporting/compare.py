"""League-wide look at a sim week (Dave, 2026-10-14): the whole league's hitting and
pitching next to the real MLB week for the same players, the sim's pitching split into
starters, relievers and (AAA) arms, and a few counts that show whether the week looks like
baseball (bullpen games, blowouts, steals). Everything comes from the week's box scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from html import escape

from npl_sim.engine.box import box_score
from npl_sim.engine.workload import WeekResult
from npl_sim.playing_time import RealStats
from npl_sim.reporting.recap import _ip, _pairs, _sum
from npl_sim.reporting.week import _stat_table
from npl_sim.season_stats import BAT_COUNTS, PITCH_COUNTS, BatterTotals, PitcherTotals, SeasonStats

BLOWOUT = 14  # runs for one team that real baseball sees about once in 200 to 300 team-games
ROLES = ("Starters", "Relievers", "(AAA) arms")


@dataclass
class League:
    hitting: list[tuple[str, str]] = field(default_factory=list)
    pitching: list[tuple[str, str]] = field(default_factory=list)
    mlb_hitting: list[tuple[str, str]] = field(default_factory=list)
    mlb_pitching: list[tuple[str, str]] = field(default_factory=list)
    roles: dict[str, PitcherTotals] = field(default_factory=dict)
    facts: dict[str, int | float] = field(default_factory=dict)


def league_compare(result: WeekResult, real: RealStats | None = None) -> League:
    boxes = [box_score(g["events"]) for g in result.games]
    week = SeasonStats()
    for box in boxes:
        week.add_game(date.min, box)
    sim_bat = _sum(BatterTotals(), week.batters.values(), BAT_COUNTS)
    sim_pit = _sum(PitcherTotals(), week.pitchers.values(), PITCH_COUNTS)
    league = League(
        hitting=_pairs("hitting", sim_bat) if week.batters else [],
        pitching=_pairs("pitching", sim_pit) if week.pitchers else [],
    )
    if real is not None:
        bats = [real.batters[p] for p in week.batters if p in real.batters]
        pits = [real.pitchers[p] for p in week.pitchers if p in real.pitchers]
        if bats:
            league.mlb_hitting = _pairs("hitting", _sum(BatterTotals(), bats, BAT_COUNTS))
        if pits:
            league.mlb_pitching = _pairs("pitching", _sum(PitcherTotals(), pits, PITCH_COUNTS))

    league.roles = {role: PitcherTotals(name=role) for role in ROLES}
    for box in boxes:
        for team in (box.away, box.home):
            for i, (pid, line) in enumerate(box.pitching[team].items()):
                role = ROLES[2] if pid.startswith("AAA-") else ROLES[0] if i == 0 else ROLES[1]
                total = league.roles[role]
                for n in PITCH_COUNTS:
                    setattr(total, n, getattr(total, n) + getattr(line, n))

    team_games = 2 * len(boxes)
    scores = [runs for box in boxes for runs in box.runs.values()]
    league.facts = {
        "games": len(boxes),
        "runs_per_team_game": sum(scores) / team_games if team_games else 0.0,
        "bullpen_games": sum(m.bullpen_game for g in result.games for m in g["moves"].values()),
        "team_games": team_games,
        "blowouts": sum(s >= BLOWOUT for s in scores),
        "sb": sim_bat.sb,
        "cs": sim_bat.cs,
    }
    return league


def league_html(league: League) -> str:
    """The league block for the top of the recaps column."""
    if not league.hitting:
        return ""
    f = league.facts
    total_outs = sum(t.outs for t in league.roles.values()) or 1
    rows = "".join(
        f"<tr><td>{escape(role)}</td><td>{_ip(t.outs)}</td>"
        f"<td>{100 * t.outs / total_outs:.0f}%</td>"
        f"<td>{t.era:.2f}</td><td>{t.whip:.2f}</td><td>{t.k}</td><td>{t.bb}</td><td>{t.hr}</td></tr>"
        for role, t in league.roles.items()
        if t.outs
    )
    split = (
        '<table class="roles"><tr><td>Sim role</td><th>IP</th><th>Share</th><th>ERA</th>'
        f"<th>WHIP</th><th>SO</th><th>BB</th><th>HR</th></tr>{rows}</table>"
    )
    bullpen = f["bullpen_games"]
    team_games = f["team_games"] or 1
    facts = (
        f"{f['games']} games, {f['runs_per_team_game']:.2f} runs per team-game. "
        f"Bullpen games: {bullpen} of {f['team_games']} team-games "
        f"({100 * bullpen / team_games:.0f}%). "
        f"Team scores of {BLOWOUT}+ runs: {f['blowouts']}. "
        f"Steals: {f['sb']} (caught {f['cs']})."
    )
    return (
        '<div class="league"><p><strong>League</strong></p>'
        + _stat_table("Hitting", league.hitting, league.mlb_hitting)
        + _stat_table("Pitching", league.pitching, league.mlb_pitching)
        + split
        + f'<p class="sub">{escape(facts)}</p></div>'
    )
