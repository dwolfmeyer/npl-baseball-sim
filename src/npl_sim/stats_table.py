"""The stats page's tables (Milestone 9): every player's or team's batting or
pitching line from the season totals, filtered, sorted and cut into pages the
way mlb.com/stats does it.

Rate stats follow MLB's qualifiers, per team: a hitter needs 3.1 plate
appearances and a pitcher 1 inning for each game his team has played.
Team lines add up the team's players.
"""

from __future__ import annotations

from dataclasses import dataclass

from npl_sim.leaders import OUTS_PER_GAME, PA_PER_GAME
from npl_sim.season_stats import (
    BAT_COUNTS,
    PITCH_COUNTS,
    BatterTotals,
    PitcherTotals,
    SeasonStats,
)

PAGE_SIZE = 50


@dataclass(frozen=True)
class Column:
    key: str  # the totals attribute it reads
    label: str
    fmt: str = "{}"
    lowest: bool = False  # lowest is best, so it sorts ascending first
    sort_key: str | None = None  # attribute to sort by when the shown text isn't numeric
    bare: bool = False  # drop the leading zero of a rate under 1, as ".333"


HITTING = (
    Column("g", "G"),
    Column("pa", "PA"),
    Column("ab", "AB"),
    Column("r", "R"),
    Column("h", "H"),
    Column("doubles", "2B"),
    Column("triples", "3B"),
    Column("hr", "HR"),
    Column("rbi", "RBI"),
    Column("bb", "BB"),
    Column("k", "SO"),
    Column("sb", "SB"),
    Column("cs", "CS"),
    Column("avg", "AVG", "{:.3f}", bare=True),
    Column("obp", "OBP", "{:.3f}", bare=True),
    Column("slg", "SLG", "{:.3f}", bare=True),
    Column("ops", "OPS", "{:.3f}", bare=True),
)
PITCHING = (
    Column("w", "W"),
    Column("l", "L"),
    Column("era", "ERA", "{:.2f}", lowest=True),
    Column("g", "G"),
    Column("gs", "GS"),
    Column("sv", "SV"),
    Column("hld", "HLD"),
    Column("ip", "IP", sort_key="outs"),
    Column("h", "H"),
    Column("r", "R"),
    Column("er", "ER"),
    Column("hr", "HR"),
    Column("bb", "BB"),
    Column("k", "SO"),
    Column("whip", "WHIP", "{:.2f}", lowest=True),
)
COLUMNS = {"hitting": HITTING, "pitching": PITCHING}
DEFAULT_SORT = {"hitting": "hr", "pitching": "era"}


@dataclass(frozen=True)
class Query:
    kind: str = "hitting"  # hitting | pitching
    group: str = "players"  # players | teams
    league: str = ""  # "" for both
    team: str = ""
    qualified: bool = True  # players only
    sort: str = ""
    descending: bool | None = None  # None: the column's best end first
    page: int = 1


@dataclass
class Row:
    rank: int
    name: str
    team: str
    cells: list[str]


@dataclass
class Table:
    columns: tuple[Column, ...]
    rows: list[Row]
    sort: str
    descending: bool
    page: int
    pages: int
    total: int


def cell(col: Column, value) -> str:
    """A value as the stats table shows it (".333", "3.45", "12.1")."""
    text = col.fmt.format(value)
    return text[1:] if col.bare and text.startswith("0.") else text


def team_totals(stats: SeasonStats, kind: str) -> list[BatterTotals | PitcherTotals]:
    """One line per team, adding up its players."""
    out: dict[str, BatterTotals | PitcherTotals] = {}
    if kind == "hitting":
        for p in stats.batters.values():
            t = out.setdefault(p.team, BatterTotals(name=p.team, team=p.team))
            for n in BAT_COUNTS:
                setattr(t, n, getattr(t, n) + getattr(p, n))
    else:
        for p in stats.pitchers.values():
            t = out.setdefault(p.team, PitcherTotals(name=p.team, team=p.team))
            for n in (*PITCH_COUNTS, "w", "l", "sv", "hld"):
                setattr(t, n, getattr(t, n) + getattr(p, n))
    for team, t in out.items():
        t.g = stats.teams[team].games if team in stats.teams else 0
        if kind == "pitching":
            t.gs = t.g
    return list(out.values())


def qualifies(stats: SeasonStats, kind: str, p: BatterTotals | PitcherTotals) -> bool:
    games = stats.teams[p.team].games if p.team in stats.teams else 0
    if kind == "hitting":
        return p.pa >= PA_PER_GAME * games
    return p.outs >= OUTS_PER_GAME * games


def build(
    stats: SeasonStats, query: Query, leagues: dict[str, str], page_size: int = PAGE_SIZE
) -> Table:
    """`leagues` maps team -> league name for the league filter."""
    kind = query.kind if query.kind in COLUMNS else "hitting"
    columns = COLUMNS[kind]
    by_key = {c.key: c for c in columns}
    sort = query.sort if query.sort in by_key else DEFAULT_SORT[kind]
    col = by_key[sort]
    descending = (not col.lowest) if query.descending is None else query.descending

    if query.group == "teams":
        pool = team_totals(stats, kind)
    else:
        pool = list(stats.batters.values() if kind == "hitting" else stats.pitchers.values())
        if query.qualified:
            pool = [p for p in pool if qualifies(stats, kind, p)]
    if query.league:
        pool = [p for p in pool if leagues.get(p.team) == query.league]
    if query.team:
        pool = [p for p in pool if p.team == query.team]

    attr = col.sort_key or col.key
    pool.sort(key=lambda p: p.name)  # ties fall back to name
    pool.sort(key=lambda p: getattr(p, attr), reverse=descending)
    pages = max(1, -(-len(pool) // page_size))
    page = min(max(query.page, 1), pages)
    start = (page - 1) * page_size
    rows = [
        Row(
            start + i + 1,
            p.name,
            p.team,
            [cell(c, getattr(p, c.key)) for c in columns],
        )
        for i, p in enumerate(pool[start : start + page_size])
    ]
    return Table(columns, rows, sort, descending, page, pages, len(pool))
