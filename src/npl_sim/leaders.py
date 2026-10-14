"""League leaders (Milestone 9): the top players in each batting and pitching
category, from the season's totals (`season_stats.SeasonStats`).

Counting stats (HR, RBI, W, K) need no minimum. Rate stats follow MLB's
qualifiers, scaled to the games played so far: a hitter needs 3.1 plate
appearances per team game and a pitcher 1 inning per team game, so the
board fills in as the season goes. Games played is the most any team has
played, as in MLB.
"""

from __future__ import annotations

from dataclasses import dataclass

from npl_sim.league import Alignment
from npl_sim.season_stats import BatterTotals, PitcherTotals, SeasonStats

PA_PER_GAME = 3.1
OUTS_PER_GAME = 3  # one inning


@dataclass(frozen=True)
class Category:
    label: str  # "HR", "AVG"
    attr: str  # the totals property or field it ranks by
    rate: bool = False  # needs the qualifier
    lowest: bool = False  # lowest is best (ERA, WHIP)
    fmt: str = "{:d}"


BATTING = (
    Category("AVG", "avg", rate=True, fmt="{:.3f}"),
    Category("HR", "hr"),
    Category("RBI", "rbi"),
    Category("R", "r"),
    Category("H", "h"),
    Category("2B", "doubles"),
    Category("SB", "sb"),
    Category("BB", "bb"),
    Category("OBP", "obp", rate=True, fmt="{:.3f}"),
    Category("SLG", "slg", rate=True, fmt="{:.3f}"),
    Category("OPS", "ops", rate=True, fmt="{:.3f}"),
)
PITCHING = (
    Category("ERA", "era", rate=True, lowest=True, fmt="{:.2f}"),
    Category("W", "w"),
    Category("SV", "sv"),
    Category("K", "k"),
    Category("WHIP", "whip", rate=True, lowest=True, fmt="{:.2f}"),
    Category("K/9", "k9", rate=True, fmt="{:.1f}"),
    Category("HLD", "hld"),
)


@dataclass(frozen=True)
class Leader:
    rank: int  # ties share a rank
    name: str
    team: str
    value: str


def team_games(stats: SeasonStats) -> int:
    return max((r.games for r in stats.teams.values()), default=0)


def _board(players, cat: Category, qualifies, n: int) -> list[Leader]:
    pool = [p for p in players if qualifies(p)] if cat.rate else list(players)
    sign = 1 if cat.lowest else -1
    pool = [p for p in pool if cat.rate or getattr(p, cat.attr) > 0]
    pool.sort(key=lambda p: (sign * getattr(p, cat.attr), p.name))
    rows, last, rank = [], None, 0
    for i, p in enumerate(pool):
        value = getattr(p, cat.attr)
        if value != last:
            rank, last = i + 1, value
        if rank > n:
            break
        rows.append(Leader(rank, p.name, p.team, cat.fmt.format(value)))
    return rows


def leaders(
    stats: SeasonStats, teams: set[str] | None = None, n: int = 5
) -> dict[str, dict[str, list[Leader]]]:
    """{"batting": {"AVG": [...], ...}, "pitching": {...}} for players on
    `teams` (default: everyone), the top `n` of each with ties shown."""
    games = team_games(stats)
    batters: list[BatterTotals] = [
        p for p in stats.batters.values() if teams is None or p.team in teams
    ]
    pitchers: list[PitcherTotals] = [
        p for p in stats.pitchers.values() if teams is None or p.team in teams
    ]

    def bats(p: BatterTotals) -> bool:
        return p.pa >= PA_PER_GAME * games

    def arms(p: PitcherTotals) -> bool:
        return p.outs >= OUTS_PER_GAME * games

    return {
        "batting": {c.label: _board(batters, c, bats, n) for c in BATTING},
        "pitching": {c.label: _board(pitchers, c, arms, n) for c in PITCHING},
    }


def league_leaders(
    stats: SeasonStats, alignment: Alignment, n: int = 5
) -> dict[str, dict[str, dict[str, list[Leader]]]]:
    """Each league's leaders, as MLB's leaderboards are kept, keyed by league name."""
    return {
        league.name: leaders(stats, {t for div in league.divisions for t in div.teams}, n)
        for league in alignment.leagues
    }
