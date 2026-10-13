"""MLB-style box scores, derived only from a game's event log.

The layout follows MLB Gameday's box: line score with R/H/E/LOB, batting and
pitching tables in lineup and appearance order, pitcher decisions, and the
batting, baserunning, fielding, pitching and game-info notes underneath.
Single-game sims have no season context, so the notes omit season totals.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from npl_sim.engine.detail import POSITION_NUMBER
from npl_sim.engine.names import Roster, ordinal

HITS = ("1B", "2B", "3B", "HR")
NOT_AB = ("BB", "HBP", "SF", "SH")
GROUND_OUTS = ("GO", "DP", "FC", "SH")
AIR_OUTS = ("FO", "LO", "PO", "SF")
TOTAL_BASES = {"1B": 1, "2B": 2, "3B": 3, "HR": 4}


@dataclass
class BattingLine:
    name: str = ""
    position: str = ""  # every spot he played, in order, e.g. "LF-RF"
    substitute: bool = False  # entered during the game (listed under the starter he followed)
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
    lob: int = 0


@dataclass
class PitchingLine:
    name: str = ""
    decisions: list[str] = field(default_factory=list)  # W, L, S, H, BS
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
    strikes: int = 0
    ground_outs: int = 0
    air_outs: int = 0
    inherited: int = 0
    inherited_scored: int = 0

    @property
    def ip(self) -> str:
        return f"{self.outs // 3}.{self.outs % 3}"

    @property
    def label(self) -> str:
        return f"{self.name} ({', '.join(self.decisions)})" if self.decisions else self.name


@dataclass
class BoxScore:
    away: str
    home: str
    abbr: dict[str, str]
    runs: dict[str, int]
    line: dict[str, list[int]]
    batting: dict[str, dict[str, BattingLine]] = field(default_factory=dict)
    pitching: dict[str, dict[str, PitchingLine]] = field(default_factory=dict)
    hits: dict[str, int] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)
    lob: dict[str, int] = field(default_factory=dict)
    # Per team: [(heading, [(label, text), ...]), ...] for Batting/Baserunning/Fielding.
    team_notes: dict[str, list[tuple[str, list[tuple[str, str]]]]] = field(default_factory=dict)
    # Game-wide pitching and info notes: [(label, text), ...].
    game_notes: list[tuple[str, str]] = field(default_factory=list)

    @property
    def winner(self) -> str:
        return self.away if self.runs[self.away] > self.runs[self.home] else self.home


@dataclass
class _Stint:
    pid: str
    entry_lead: int
    save_situation: bool
    on_base: int = 0
    outs: int = 0
    blown: bool = False
    exit_lead: int = 0


def _tally(items: list[tuple[str, str]]) -> str:
    """'Freeman (Peralta), Betts 2 (Peralta; Megill)' from (name, detail) pairs."""
    grouped: dict[str, list[str]] = {}
    for name, extra in items:
        grouped.setdefault(name, []).append(extra)
    parts = []
    for name, extras in grouped.items():
        count = f" {len(extras)}" if len(extras) > 1 else ""
        shown = list(dict.fromkeys(x for x in extras if x))
        sep = "; " if any("," in x for x in shown) else ", "
        parts.append(f"{name}{count}" + (f" ({sep.join(shown)})" if shown else ""))
    return ", ".join(parts) + "."


def _counts(items: dict[str, int]) -> str:
    return ", ".join(f"{n} {c}" if c > 1 else n for n, c in items.items() if c) + "."


def box_score(events: list[dict], away: str | None = None, home: str | None = None) -> BoxScore:
    roster = Roster(events)
    final = events[-1]
    assert final["type"] == "final"
    sides = ("away", "home")
    other = {"away": "home", "home": "away"}
    team_name = {s: roster.teams[s].name for s in sides}
    if away is not None and home is not None:
        assert (away, home) == (team_name["away"], team_name["home"])

    batting: dict[str, dict[str, BattingLine]] = {s: {} for s in sides}
    pitching: dict[str, dict[str, PitchingLine]] = {s: {} for s in sides}
    slots: dict[str, list[list[str]]] = {s: [] for s in sides}  # batting-order slots
    for s in sides:
        for p in roster.teams[s].lineup:
            batting[s][p["id"]] = BattingLine(roster.short[p["id"]], p["position"])
            slots[s].append([p["id"]])
    hits = dict.fromkeys(sides, 0)
    errors = dict.fromkeys(sides, 0)
    lob = dict.fromkeys(sides, 0)
    risp = {s: [0, 0] for s in sides}  # hits, at bats

    notes = {
        s: {k: [] for k in ("2B", "3B", "HR", "SF", "SAC", "GIDP", "SB", "CS", "E", "DP", "PB")}
        for s in sides
    }
    tb = {s: defaultdict(int) for s in sides}
    rbi = {s: defaultdict(int) for s in sides}
    two_out_rbi = {s: defaultdict(int) for s in sides}
    risp_left = {s: defaultdict(int) for s in sides}
    hbp_notes: list[tuple[str, str]] = []
    wp_notes: list[tuple[str, str]] = []
    balk_notes: list[tuple[str, str]] = []

    current = {s: roster.teams[s].starter for s in sides}
    score = dict.fromkeys(sides, 0)
    stints: dict[str, list[_Stint]] = {s: [_Stint(current[s], 0, False)] for s in sides}
    go_ahead: tuple[str, str, str] | None = None  # (side, winning pitcher, losing pitcher)

    def pit(side: str, pid: str) -> PitchingLine:
        if pid not in pitching[side]:
            pitching[side][pid] = PitchingLine(roster.short[pid])
        return pitching[side][pid]

    for s in sides:
        pit(s, current[s])

    def lead(side: str) -> int:
        return score[side] - score[other[side]]

    def add_runs(offense: str, scored: list[dict]) -> None:
        nonlocal go_ahead
        defense = other[offense]
        for run in scored:
            before = lead(offense)
            score[offense] += 1
            batting[offense][run["runner"]].r += 1
            charged = pit(defense, run["pitcher"])
            charged.r += 1
            charged.er += run["earned"]
            if run["pitcher"] != current[defense]:
                pit(defense, current[defense]).inherited_scored += 1
            if before == 0:
                go_ahead = (offense, current[offense], run["pitcher"])
            stint = stints[defense][-1]
            if stint.save_situation and lead(defense) <= 0:
                stint.blown = True

    for e in events:
        if e["type"] in ("game_start", "final"):
            continue
        offense = "away" if e["half"] == "top" else "home"
        defense = other[offense]
        if e["type"] != "pa" and e.get("pitches"):
            # A PA cut short by the third out on the bases or a walk-off wild
            # pitch: its pitches still count, though the batter gets no PA.
            p = pit(defense, e["pitcher"])
            p.pitches += e["pitches"]
            p.strikes += e["strikes"]
        if e["type"] == "pa":
            res = e["result"]
            b = batting[offense][e["batter"]]
            p = pit(defense, e["pitcher"])
            batter = roster.short[e["batter"]]
            pitcher = roster.short[e["pitcher"]]
            b.pa += 1
            p.bf += 1
            p.pitches += e["pitches"]
            p.strikes += e["strikes"]
            p.outs += e["outs"]
            stints[defense][-1].outs += e["outs"]
            is_ab = res not in NOT_AB
            b.ab += is_ab
            on_scoring = any(e["bases_before"][1:])
            if is_ab and on_scoring:
                risp[offense][1] += 1
                risp[offense][0] += res in HITS
            if res in HITS:
                b.h += 1
                p.h += 1
                hits[offense] += 1
                tb[offense][batter] += TOTAL_BASES[res]
            if res in ("2B", "3B"):
                notes[offense][res].append((batter, pitcher))
            if res == "HR":
                on = sum(r is not None for r in e["bases_before"])
                notes[offense]["HR"].append(
                    (
                        batter,
                        f"{ordinal(e['inning'])} inning off {pitcher}, "
                        f"{on} on, {e['outs_before']} out",
                    )
                )
            b.doubles += res == "2B"
            b.triples += res == "3B"
            b.hr += res == "HR"
            p.hr += res == "HR"
            b.bb += res == "BB"
            p.bb += res == "BB"
            b.hbp += res == "HBP"
            p.hbp += res == "HBP"
            if res == "HBP":
                hbp_notes.append((batter, f"by {pitcher}"))
            b.k += res == "K"
            p.k += res == "K"
            b.sf += res == "SF"
            if res == "SF":
                notes[offense]["SF"].append((batter, ""))
            b.sh += res == "SH"
            if res == "SH":
                notes[offense]["SAC"].append((batter, ""))
            if res == "DP":
                notes[offense]["GIDP"].append((batter, ""))
                notes[defense]["DP"].append(("", _dp_chain(roster, defense, e)))
            if res == "E":
                errors[defense] += 1
                fid = roster.fielder_id(defense, e["fielder"], e["pitcher"])
                if fid:
                    notes[defense]["E"].append((roster.short[fid], "fielding"))
            if res in GROUND_OUTS and e["bip"] == "GB":
                p.ground_outs += 1
            elif res in GROUND_OUTS or res in AIR_OUTS:
                p.air_outs += 1
            b.rbi += e["rbi"]
            if e["rbi"]:
                rbi[offense][batter] += e["rbi"]
                if e["outs_before"] == 2:
                    two_out_rbi[offense][batter] += e["rbi"]
            batter_out = e["runners"][0]["end"] == "out"
            if batter_out:
                b.lob += sum(r is not None for r in e["bases_after"])
                if e["outs_before"] + e["outs"] == 3 and on_scoring and e["outs_before"] == 2:
                    risp_left[offense][batter] += 1
            add_runs(offense, e["scored"])
        elif e["type"] in ("stolen_base", "caught_stealing"):
            stole = e["type"] == "stolen_base"
            line = batting[offense][e["runner"]]
            catcher = roster.fielder_id(defense, "C", e["pitcher"])
            battery = roster.short[e["pitcher"]] + (f"/{roster.short[catcher]}" if catcher else "")
            base = e["base"].replace("2B", "2nd base").replace("3B", "3rd base")
            if stole:
                line.sb += 1
                notes[offense]["SB"].append((line.name, f"{base} off {battery}"))
            else:
                line.cs += 1
                pit(defense, e["pitcher"]).outs += 1
                stints[defense][-1].outs += 1
                notes[offense]["CS"].append((line.name, f"{base} by {battery}"))
        elif e["type"] in ("wild_pitch", "passed_ball", "balk"):
            pitcher = roster.short[e["pitcher"]]
            if e["type"] == "wild_pitch":
                wp_notes.append((pitcher, ""))
            elif e["type"] == "balk":
                balk_notes.append((pitcher, ""))
            else:
                catcher = roster.fielder_id(defense, "C", e["pitcher"])
                notes[defense]["PB"].append((roster.short[catcher] if catcher else "C", ""))
            add_runs(offense, e["scored"])
        elif e["type"] in ("defensive_sub", "pinch_hitter"):
            side = roster.side_of(e["team"])
            roster.teams[side].substitute(e)
            pos = e["position"]
            if e["type"] == "pinch_hitter":
                pos = "PH" if pos == "DH" else f"PH-{pos}"
            batting[side][e["into"]] = BattingLine(roster.short[e["into"]], pos, substitute=True)
            slots[side][e["slot"]].append(e["into"])
            if e.get("moved"):
                batting[side][e["moved"]["player"]].position += "-" + e["moved"]["position"]
        elif e["type"] == "pitching_change":
            side = roster.side_of(e["team"])
            stints[side][-1].exit_lead = lead(side)
            current[side] = e["into"]
            new = pit(side, e["into"])
            new.inherited += e["on_base"]
            entry = lead(side)
            save_sit = entry > 0 and (entry <= 3 or entry <= e["on_base"] + 2)
            stints[side].append(_Stint(e["into"], entry, save_sit, e["on_base"]))
        elif e["type"] == "half_end":
            lob[offense] += e["left_on_base"]

    for s in sides:
        stints[s][-1].exit_lead = lead(s)
        batting[s] = {pid: batting[s][pid] for slot in slots[s] for pid in slot}
    assert score["away"] == final["away"] and score["home"] == final["home"]
    _decide(pitching, stints, go_ahead, roster, score)

    team_notes: dict[str, list] = {}
    for s in sides:
        n = notes[s]
        batting_notes = [
            ("2B", _tally(n["2B"])),
            ("3B", _tally(n["3B"])),
            ("HR", _tally(n["HR"])),
            ("TB", _counts(tb[s])),
            ("RBI", _counts(rbi[s])),
            ("2-out RBI", _counts(two_out_rbi[s])),
            ("Runners left in scoring position, 2 out", _counts(risp_left[s])),
            ("SF", _tally(n["SF"])),
            ("SAC", _tally(n["SAC"])),
            ("GIDP", _tally(n["GIDP"])),
            ("Team RISP", f"{risp[s][0]}-for-{risp[s][1]}."),
            ("Team LOB", f"{lob[s]}."),
        ]
        dp = n["DP"]
        start = events[0]
        misplaced = [
            f"{roster.short[p['id']]} ({p['position']})"
            for p in start[s]["lineup"]
            if p.get("out_of_position")
        ]
        fielding = [
            ("Out of position", ", ".join(misplaced) + "."),
            ("E", _tally(n["E"])),
            ("PB", _tally(n["PB"])),
            ("DP", f"{len(dp)} ({', '.join(x for _, x in dp)})." if dp else "."),
        ]
        team_notes[s] = [
            (heading, [(k, v) for k, v in rows if v != "." and v != "0-for-0."])
            for heading, rows in (
                ("Batting", batting_notes),
                ("Baserunning", [("SB", _tally(n["SB"])), ("CS", _tally(n["CS"]))]),
                ("Fielding", fielding),
            )
        ]
        team_notes[s] = [(h, rows) for h, rows in team_notes[s] if rows]

    every = [p for s in sides for p in pitching[s].values()]
    game_notes = [
        ("WP", _tally(wp_notes)),
        ("Balk", _tally(balk_notes)),
        ("HBP", _tally(hbp_notes)),
        ("Pitches-strikes", ", ".join(f"{p.name} {p.pitches}-{p.strikes}" for p in every) + "."),
        (
            "Groundouts-flyouts",
            ", ".join(f"{p.name} {p.ground_outs}-{p.air_outs}" for p in every) + ".",
        ),
        ("Batters faced", ", ".join(f"{p.name} {p.bf}" for p in every) + "."),
        (
            "Inherited runners-scored",
            ", ".join(f"{p.name} {p.inherited}-{p.inherited_scored}" for p in every if p.inherited)
            + ".",
        ),
    ]
    game_notes = [(k, v) for k, v in game_notes if v != "."]

    return BoxScore(
        away=team_name["away"],
        home=team_name["home"],
        abbr={team_name[s]: roster.teams[s].abbr for s in sides},
        runs={team_name[s]: final[s] for s in sides},
        line={team_name[s]: final["line"][s] for s in sides},
        batting={team_name[s]: batting[s] for s in sides},
        pitching={team_name[s]: pitching[s] for s in sides},
        hits={team_name[s]: hits[s] for s in sides},
        errors={team_name[s]: errors[s] for s in sides},
        lob={team_name[s]: lob[s] for s in sides},
        team_notes={team_name[s]: team_notes[s] for s in sides},
        game_notes=game_notes,
    )


def _dp_chain(roster: Roster, defense: str, e: dict) -> str:
    """Fielders on a ground-ball double play, e.g. 'Adames-Turang-Hoskins'."""
    chain = dp_positions(e["fielder"])
    ids = [roster.fielder_id(defense, pos, e["pitcher"]) for pos in chain]
    return "-".join(roster.short[i] if i else pos for i, pos in zip(ids, chain, strict=True))


def dp_positions(fielder: str) -> list[str]:
    """Scorebook path of a 6-4-3 style double play started by `fielder`."""
    if fielder == "2B":
        return ["2B", "SS", "1B"]
    if fielder in ("1B", "P", "C"):
        return [fielder, "SS", "1B"] if fielder != "1B" else ["1B", "SS", "P"]
    return [fielder, "2B", "1B"]


def dp_numbers(fielder: str) -> str:
    return "-".join(str(POSITION_NUMBER[p]) for p in dp_positions(fielder))


def _decide(pitching, stints, go_ahead, roster: Roster, score) -> None:
    """Win, loss, save, hold and blown-save credits under the MLB scoring rules."""
    win_side = "away" if score["away"] > score["home"] else "home"
    lose_side = "home" if win_side == "away" else "away"
    assert go_ahead is not None and go_ahead[0] == win_side
    _, winner, loser = go_ahead

    starter = roster.teams[win_side].starter
    if winner == starter and pitching[win_side][starter].outs < 15:
        relievers = stints[win_side][1:]
        if relievers:
            winner = max(relievers, key=lambda s: s.outs).pid  # earliest wins ties
    pitching[win_side][winner].decisions.append("W")
    pitching[lose_side][loser].decisions.append("L")

    finisher = stints[win_side][-1]
    if finisher.pid != winner and len(stints[win_side]) > 1 and not finisher.blown:
        lead_in = finisher.entry_lead
        if lead_in > 0 and (
            (lead_in <= 3 and finisher.outs >= 3)
            or lead_in <= finisher.on_base + 2  # tying run on base, at bat or on deck
            or finisher.outs >= 9
        ):
            pitching[win_side][finisher.pid].decisions.append("S")

    for side in ("away", "home"):
        for i, stint in enumerate(stints[side]):
            if i == 0:
                continue
            line = pitching[side][stint.pid]
            if stint.blown:
                line.decisions.append("BS")
            elif (
                stint.save_situation
                and stint.outs >= 1
                and stint.exit_lead > 0
                and i < len(stints[side]) - 1
                and not {"W", "S"} & set(line.decisions)
            ):
                line.decisions.append("H")


def format_box(box: BoxScore) -> str:
    """Plain-text box score in Gameday's layout."""
    innings = max(len(box.line[box.away]), len(box.line[box.home]))
    head = "".ljust(6) + " ".join(f"{i + 1:>2}" for i in range(innings)) + "    R  H  E LOB"
    rows = [head]
    for team in (box.away, box.home):
        cells = [f"{r:>2}" for r in box.line[team]] + [" X"] * (innings - len(box.line[team]))
        rows.append(
            f"{box.abbr[team]:<6}"
            + " ".join(cells)
            + f"   {box.runs[team]:>2} {box.hits[team]:>2} {box.errors[team]:>2} {box.lob[team]:>3}"
        )
    for team in (box.away, box.home):
        rows += ["", f"{team} Batters".ljust(26) + "AB  R  H RBI BB SO LOB"]
        totals = BattingLine()
        for b in box.batting[team].values():
            rows.append(
                f"  {(('  ' if b.substitute else '') + b.name + ' ' + b.position)[:22]:<22} "
                f"{b.ab:>3} {b.r:>2} {b.h:>2} "
                f"{b.rbi:>3} {b.bb:>2} {b.k:>2} {b.lob:>3}"
            )
            for k in ("ab", "r", "h", "rbi", "bb", "k", "lob"):
                setattr(totals, k, getattr(totals, k) + getattr(b, k))
        rows.append(
            f"  {'Totals':<22} {totals.ab:>3} {totals.r:>2} {totals.h:>2} "
            f"{totals.rbi:>3} {totals.bb:>2} {totals.k:>2} {totals.lob:>3}"
        )
        for heading, notes in box.team_notes[team]:
            rows += ["", f"  {heading.upper()}"]
            rows += [f"  {label}: {text}" for label, text in notes]
    for team in (box.away, box.home):
        rows += ["", f"{team} Pitchers".ljust(26) + "  IP  H  R ER BB SO HR"]
        for p in box.pitching[team].values():
            rows.append(
                f"  {p.label[:22]:<22} {p.ip:>5} {p.h:>2} {p.r:>2} {p.er:>2} {p.bb:>2} "
                f"{p.k:>2} {p.hr:>2}"
            )
    rows += [""] + [f"{label}: {text}" for label, text in box.game_notes]
    return "\n".join(rows)
