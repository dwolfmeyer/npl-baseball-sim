"""Box scores and play-by-play, derived only from a game's event log."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

HITS = ("1B", "2B", "3B", "HR")
NOT_AB = ("BB", "HBP", "SF")
DESCRIBE = {
    "K": "strikes out",
    "BB": "walks",
    "HBP": "is hit by a pitch",
    "HR": "homers",
    "1B": "singles",
    "2B": "doubles",
    "3B": "triples",
    "GO": "grounds out",
    "FO": "flies out",
    "LO": "lines out",
    "PO": "pops out",
    "SF": "hits a sacrifice fly",
    "DP": "grounds into a double play",
    "FC": "reaches on a fielder's choice",
    "E": "reaches on an error",
}


@dataclass
class BattingLine:
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


@dataclass
class PitchingLine:
    outs: int = 0
    bf: int = 0
    h: int = 0
    r: int = 0
    er: int = 0
    bb: int = 0
    hbp: int = 0
    k: int = 0
    hr: int = 0
    pitches: int = 0

    @property
    def ip(self) -> str:
        return f"{self.outs // 3}.{self.outs % 3}"


@dataclass
class BoxScore:
    away: str
    home: str
    runs: dict[str, int]
    line: dict[str, list[int]]
    batting: dict[str, dict[str, BattingLine]] = field(default_factory=dict)
    pitching: dict[str, dict[str, PitchingLine]] = field(default_factory=dict)
    hits: dict[str, int] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)
    play_by_play: list[str] = field(default_factory=list)


def box_score(events: list[dict], away: str, home: str) -> BoxScore:
    final = events[-1]
    assert final["type"] == "final"
    batting: dict[str, dict[str, BattingLine]] = {away: {}, home: {}}
    pitching: dict[str, dict[str, PitchingLine]] = {away: {}, home: {}}
    hits = {away: 0, home: 0}
    errors = {away: 0, home: 0}
    pbp: list[str] = []
    other = {away: home, home: away}
    runs_by_runner: dict[str, int] = defaultdict(int)

    def bat(team, pid):
        return batting[team].setdefault(pid, BattingLine())

    def pit(team, pid):
        return pitching[team].setdefault(pid, PitchingLine())

    for e in events:
        offense = away if e["half"] == "top" else home
        defense = other[offense]
        tag = f"{'Top' if e['half'] == 'top' else 'Bot'} {e['inning']}"
        if e["type"] == "pa":
            b, p = bat(offense, e["batter"]), pit(defense, e["pitcher"])
            res = e["result"]
            b.pa += 1
            p.bf += 1
            p.pitches += e["pitches"]
            p.outs += e["outs"]
            if res not in NOT_AB:
                b.ab += 1
            if res in HITS:
                b.h += 1
                p.h += 1
                hits[offense] += 1
            b.doubles += res == "2B"
            b.triples += res == "3B"
            b.hr += res == "HR"
            p.hr += res == "HR"
            b.bb += res == "BB"
            p.bb += res == "BB"
            b.hbp += res == "HBP"
            p.hbp += res == "HBP"
            b.k += res == "K"
            p.k += res == "K"
            b.sf += res == "SF"
            b.rbi += e["rbi"]
            errors[defense] += res == "E"
            for s in e["scored"]:
                bat(offense, s["runner"]).r += 1
                runs_by_runner[s["runner"]] += 1
                charged = pit(defense, s["pitcher"])
                charged.r += 1
                charged.er += s["earned"]
            scored = f" ({len(e['scored'])} scores)" if e["scored"] else ""
            pbp.append(f"{tag}: {e['batter']} {DESCRIBE[res]}{scored}.")
        elif e["type"] in ("stolen_base", "caught_stealing"):
            stole = e["type"] == "stolen_base"
            line = bat(offense, e["runner"])
            line.sb += stole
            line.cs += not stole
            if not stole:
                pit(defense, e["pitcher"]).outs += 1
            pbp.append(f"{tag}: {e['runner']} {'steals' if stole else 'is caught stealing'}.")
        elif e["type"] == "pitching_change":
            pit(e["team"], e["into"])
            pbp.append(f"{tag}: pitching change, {e['into']} replaces {e['out']}.")
        elif e["type"] == "auto_runner":
            pbp.append(f"{tag}: {e['runner']} starts the inning on second.")

    return BoxScore(
        away=away,
        home=home,
        runs={away: final["away"], home: final["home"]},
        line={away: final["line"]["away"], home: final["line"]["home"]},
        batting=batting,
        pitching=pitching,
        hits=hits,
        errors=errors,
        play_by_play=pbp,
    )


def format_box(box: BoxScore) -> str:
    """Plain-text box score."""
    innings = max(len(box.line[box.away]), len(box.line[box.home]))
    head = "".ljust(24) + " ".join(f"{i + 1:>2}" for i in range(innings)) + "    R  H  E"
    rows = [head]
    for team in (box.away, box.home):
        cells = [f"{r:>2}" for r in box.line[team]] + [" X"] * (innings - len(box.line[team]))
        rows.append(
            f"{team[:24]:<24}"
            + " ".join(cells)
            + f"   {box.runs[team]:>2} {box.hits[team]:>2} {box.errors[team]:>2}"
        )
    for team in (box.away, box.home):
        rows += ["", f"{team} batting   AB  R  H 2B 3B HR RBI BB  K SB"]
        for pid, b in box.batting[team].items():
            rows.append(
                f"  {pid[:16]:<16} {b.ab:>3} {b.r:>2} {b.h:>2} {b.doubles:>2} "
                f"{b.triples:>2} {b.hr:>2} {b.rbi:>3} {b.bb:>2} {b.k:>2} {b.sb:>2}"
            )
        rows += [f"{team} pitching    IP  H  R ER BB  K HR  P"]
        for pid, p in box.pitching[team].items():
            rows.append(
                f"  {pid[:16]:<16} {p.ip:>4} {p.h:>2} {p.r:>2} {p.er:>2} {p.bb:>2} "
                f"{p.k:>2} {p.hr:>2} {p.pitches:>3}"
            )
    return "\n".join(rows)
