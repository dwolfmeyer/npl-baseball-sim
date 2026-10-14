"""Season to date (Milestone 9): each team's record and every player's batting
and pitching totals, built one box score at a time, and the standings read
from them.

`SeasonStats.add_game(date, box)` folds in a finished game. The totals are
plain numbers (`to_json` / `from_json`), so a season's state can be stored
between sim weeks and picked up for the next one. `division_standings` turns
the records into the plan's standings page: games back, run differential,
home and road records, last 10 and the current streak.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import date

from npl_sim.engine.box import BoxScore
from npl_sim.league import Alignment

RECENT = 10  # games in "last 10"


@dataclass
class TeamRecord:
    w: int = 0
    l: int = 0  # noqa: E741
    rs: int = 0
    ra: int = 0
    home_w: int = 0
    home_l: int = 0
    away_w: int = 0
    away_l: int = 0
    results: list[bool] = field(default_factory=list)  # every game in order, True = win
    one_run: list[int] = field(default_factory=lambda: [0, 0])  # [W, L] in one-run games
    extra: list[int] = field(default_factory=lambda: [0, 0])  # [W, L] in extra innings
    vs: dict[str, list[int]] = field(default_factory=dict)  # [W, L] against each opponent

    @property
    def games(self) -> int:
        return self.w + self.l

    @property
    def pct(self) -> float:
        return self.w / self.games if self.games else 0.0

    @property
    def diff(self) -> int:
        return self.rs - self.ra

    @property
    def expected_w(self) -> float:
        """Pythagenpat wins from runs scored and allowed over the games played."""
        if not self.rs and not self.ra:
            return 0.0
        exp = ((self.rs + self.ra) / self.games) ** 0.287 if self.games else 1.83
        return self.games * self.rs**exp / (self.rs**exp + self.ra**exp)

    @property
    def last10(self) -> tuple[int, int]:
        recent = self.results[-RECENT:]
        return sum(recent), len(recent) - sum(recent)

    @property
    def streak(self) -> str:
        if not self.results:
            return "-"
        last, n = self.results[-1], 0
        for won in reversed(self.results):
            if won != last:
                break
            n += 1
        return f"{'W' if last else 'L'}{n}"


@dataclass
class BatterTotals:
    name: str = ""
    team: str = ""
    g: int = 0
    pa: int = 0
    ab: int = 0
    r: int = 0
    h: int = 0
    doubles: int = 0
    triples: int = 0
    hr: int = 0
    rbi: int = 0
    bb: int = 0
    hbp: int = 0
    k: int = 0
    sb: int = 0
    cs: int = 0
    sf: int = 0
    sh: int = 0

    @property
    def avg(self) -> float:
        return self.h / self.ab if self.ab else 0.0

    @property
    def obp(self) -> float:
        den = self.ab + self.bb + self.hbp + self.sf
        return (self.h + self.bb + self.hbp) / den if den else 0.0

    @property
    def slg(self) -> float:
        bases = self.h + self.doubles + 2 * self.triples + 3 * self.hr
        return bases / self.ab if self.ab else 0.0

    @property
    def ops(self) -> float:
        return self.obp + self.slg


@dataclass
class PitcherTotals:
    name: str = ""
    team: str = ""
    g: int = 0
    gs: int = 0
    w: int = 0
    l: int = 0  # noqa: E741
    sv: int = 0
    hld: int = 0
    outs: int = 0
    bf: int = 0
    h: int = 0
    r: int = 0
    er: int = 0
    bb: int = 0
    hbp: int = 0
    k: int = 0
    hr: int = 0

    @property
    def ip(self) -> str:
        return f"{self.outs // 3}.{self.outs % 3}"

    @property
    def era(self) -> float:
        return 27 * self.er / self.outs if self.outs else 0.0

    @property
    def whip(self) -> float:
        return 3 * (self.h + self.bb) / self.outs if self.outs else 0.0

    @property
    def k9(self) -> float:
        return 27 * self.k / self.outs if self.outs else 0.0


def _add(total, line, names: tuple[str, ...]) -> None:
    for n in names:
        setattr(total, n, getattr(total, n) + getattr(line, n))


BAT_COUNTS = (
    "pa ab r h doubles triples hr rbi bb hbp k sb cs sf sh".split()
)  # fmt: skip
PITCH_COUNTS = ("outs", "bf", "h", "r", "er", "bb", "hbp", "k", "hr")


@dataclass
class SeasonStats:
    teams: dict[str, TeamRecord] = field(default_factory=dict)
    batters: dict[str, BatterTotals] = field(default_factory=dict)  # by player id
    pitchers: dict[str, PitcherTotals] = field(default_factory=dict)
    games: int = 0
    last_date: date | None = None

    def add_game(self, played: date, box: BoxScore) -> None:
        """Fold one finished game in (games must arrive in date order for the
        last 10 and the streak to read right)."""
        self.games += 1
        self.last_date = played
        for team, other, at_home in ((box.away, box.home, False), (box.home, box.away, True)):
            won = box.runs[team] > box.runs[other]
            rec = self.teams.setdefault(team, TeamRecord())
            rec.w += won
            rec.l += not won
            rec.rs += box.runs[team]
            rec.ra += box.runs[other]
            if at_home:
                rec.home_w += won
                rec.home_l += not won
            else:
                rec.away_w += won
                rec.away_l += not won
            rec.results.append(won)
            rec.vs.setdefault(other, [0, 0])[not won] += 1
            if abs(box.runs[team] - box.runs[other]) == 1:
                rec.one_run[not won] += 1
            if len(box.line[team]) > 9:
                rec.extra[not won] += 1
            for pid, line in box.batting[team].items():
                bat = self.batters.setdefault(pid, BatterTotals(name=line.name, team=team))
                bat.team = team
                bat.g += 1
                _add(bat, line, tuple(BAT_COUNTS))
            for i, (pid, line) in enumerate(box.pitching[team].items()):
                pit = self.pitchers.setdefault(pid, PitcherTotals(name=line.name, team=team))
                pit.team = team
                pit.g += 1
                pit.gs += i == 0  # the first arm in the box is the starter
                _add(pit, line, PITCH_COUNTS)
                for d, n in (("W", "w"), ("L", "l"), ("S", "sv"), ("H", "hld")):
                    setattr(pit, n, getattr(pit, n) + line.decisions.count(d))

    def to_json(self) -> dict:
        return {
            "games": self.games,
            "last_date": self.last_date.isoformat() if self.last_date else None,
            "teams": {t: asdict(r) for t, r in self.teams.items()},
            "batters": {p: asdict(b) for p, b in self.batters.items()},
            "pitchers": {p: asdict(b) for p, b in self.pitchers.items()},
        }

    @classmethod
    def from_json(cls, data: dict) -> SeasonStats:
        def build(kind, row):
            return kind(**{f.name: row[f.name] for f in fields(kind) if f.name in row})

        stats = cls(games=data["games"])
        if data["last_date"]:
            stats.last_date = date.fromisoformat(data["last_date"])
        for t, row in data["teams"].items():
            stats.teams[t] = build(TeamRecord, row)
        stats.batters = {p: build(BatterTotals, row) for p, row in data["batters"].items()}
        stats.pitchers = {p: build(PitcherTotals, row) for p, row in data["pitchers"].items()}
        return stats


@dataclass
class StandingsRow:
    team: str
    record: TeamRecord
    gb: float  # games behind the leader of the group; 0 for the leader


def _behind(a: TeamRecord, b: TeamRecord) -> float:
    """Games `a` is behind `b` (negative when it is ahead)."""
    return ((b.w - a.w) + (a.l - b.l)) / 2


def _group(stats: SeasonStats, teams: list[str]) -> list[StandingsRow]:
    """`teams` ranked by winning percentage, then run differential, then name,
    with games back from the first."""
    recs = {t: stats.teams.get(t, TeamRecord()) for t in teams}
    order = sorted(teams, key=lambda t: (-recs[t].pct, -recs[t].diff, t))
    lead = recs[order[0]]
    return [StandingsRow(t, recs[t], max(0.0, _behind(recs[t], lead))) for t in order]


def division_standings(
    stats: SeasonStats, alignment: Alignment
) -> list[tuple[str, str, list[StandingsRow]]]:
    """[(league, division, rows best first)] in the alignment's order."""
    return [
        (league.name, div.name, _group(stats, list(div.teams)))
        for league in alignment.leagues
        for div in league.divisions
    ]


def league_standings(
    stats: SeasonStats, alignment: Alignment
) -> list[tuple[str, list[StandingsRow]]]:
    """[(league, rows best first)] across its divisions."""
    return [
        (league.name, _group(stats, [t for div in league.divisions for t in div.teams]))
        for league in alignment.leagues
    ]


def format_gb(gb: float) -> str:
    return "-" if gb == 0 else f"{gb:g}"


@dataclass
class RaceRow:
    team: str
    record: TeamRecord
    gb: float  # back of the division leader
    wcgb: float | None  # back of the last wild card spot (negative: ahead of it); None for leaders
    seed: int | None  # playoff seed: division winners by record, then wild cards


def _record_order(row: StandingsRow) -> tuple:
    return (-row.record.pct, -row.record.diff, row.team)


def wild_card_race(stats: SeasonStats, alignment: Alignment) -> list[tuple[str, list[RaceRow]]]:
    """[(league, rows)] with the division winners first (best record first), then
    every other team by record. `alignment.wild_cards` spots go to the best of the
    others; `wcgb` measures each from the last of those spots."""
    out = []
    for league in alignment.leagues:
        winners, others = [], []
        for div in league.divisions:
            rows = _group(stats, list(div.teams))
            winners.append(rows[0])
            others += rows[1:]
        winners.sort(key=_record_order)
        others.sort(key=_record_order)
        spots = alignment.wild_cards
        cut = others[spots - 1].record if 0 < spots <= len(others) else None
        rows = [RaceRow(r.team, r.record, 0, None, i + 1) for i, r in enumerate(winners)]
        for i, r in enumerate(others):
            seed = len(winners) + i + 1 if i < spots else None
            wcgb = _behind(r.record, cut) if cut is not None else None
            rows.append(RaceRow(r.team, r.record, r.gb, wcgb, seed))
        out.append((league.name, rows))
    return out


def format_wcgb(wcgb: float | None) -> str:
    """ "-" for a division leader or the last wild card team, "+1.5" for a team ahead of
    the cut, "2" for one behind it."""
    if wcgb is None or wcgb == 0:
        return "-"
    return f"+{-wcgb:g}" if wcgb < 0 else f"{wcgb:g}"


EXPANDED = ("1-Run", "Extra", "Div", "League", "Inter", ">=.500", "<.500")


def expanded_records(stats: SeasonStats, alignment: Alignment) -> dict[str, dict[str, str]]:
    """The expanded standings columns per team, as "W-L" text: one-run games,
    extra innings, and the record against its own division, the rest of its league,
    the other league, and teams now at or over and under .500."""
    division, league = {}, {}
    for lg in alignment.leagues:
        for div in lg.divisions:
            for t in div.teams:
                division[t], league[t] = div.name, lg.name
    out = {}
    for team, rec in stats.teams.items():
        sums = {k: [0, 0] for k in EXPANDED[2:]}
        for opp, (won, lost) in rec.vs.items():
            if opp not in league or team not in league:
                continue
            if league[opp] != league[team]:
                keys = ["Inter"]
            elif division[opp] == division[team]:
                keys = ["Div"]
            else:
                keys = ["League"]
            keys.append(">=.500" if stats.teams[opp].pct >= 0.5 else "<.500")
            for k in keys:
                sums[k][0] += won
                sums[k][1] += lost
        out[team] = {
            "1-Run": f"{rec.one_run[0]}-{rec.one_run[1]}",
            "Extra": f"{rec.extra[0]}-{rec.extra[1]}",
            **{k: f"{v[0]}-{v[1]}" for k, v in sums.items()},
        }
    return out
