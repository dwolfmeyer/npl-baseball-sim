"""The game log: Gameday-style play-by-play, filterable to all plays or scoring plays."""

from __future__ import annotations

from dataclasses import dataclass, field

from npl_sim.engine.box import dp_positions
from npl_sim.engine.detail import POSITION_TITLE
from npl_sim.engine.leverage import index as leverage_index
from npl_sim.engine.names import BASE_NAME, NEXT_BASE, Roster, ordinal
from npl_sim.engine.pitchseq import Pitch, sequence
from npl_sim.engine.winprob import win_expectancy

BALL_TYPE = {"GB": "ground ball", "LD": "line drive", "FB": "fly ball", "PU": "pop up"}
FIELD_NAME = {
    "C": "catcher",
    "1B": "first base",
    "2B": "second base",
    "3B": "third base",
    "SS": "shortstop",
    "LF": "left field",
    "CF": "center field",
    "RF": "right field",
}
HIT_VERB = {"1B": "singles", "2B": "doubles", "3B": "triples"}
OUT_VERB = {"FO": "flies out", "LO": "lines out", "PO": "pops out"}


RESULT_TITLE = {
    "K": "Strikeout", "BB": "Walk", "HBP": "Hit By Pitch", "HR": "Home Run", "1B": "Single",
    "2B": "Double", "3B": "Triple", "GO": "Groundout", "FO": "Flyout", "LO": "Lineout",
    "PO": "Pop Out", "SF": "Sac Fly", "DP": "Grounded Into DP", "FC": "Forceout",
    "E": "Field Error", "SH": "Sac Bunt",
}  # fmt: skip


@dataclass
class AtBat:
    """What the Gameday at-bat pop-up shows beyond the play itself."""

    title: str  # Strikeout, Single ...
    batter: str
    pitcher: str
    bats: str  # the side he hit from
    throws: str
    position: str
    outs: int  # before the PA
    bases: tuple[bool, bool, bool]  # occupied before the PA
    score: dict[str, int]  # by team abbreviation, before the PA
    pitches: list[Pitch]


@dataclass
class Play:
    description: str
    kind: str  # pa, stolen_base, caught_stealing, pitching_change, defensive_sub,
    # pinch_hitter, auto_runner
    batter: str | None = None
    pitcher: str | None = None
    count: str | None = None  # "1-2" on the final pitch of a plate appearance
    outs: int = 0  # outs after the play
    runs: int = 0
    score: dict[str, int] = field(default_factory=dict)  # by team abbreviation, after the play
    home_we: float | None = None  # home team's win expectancy after the play
    wpa: float | None = None  # win probability the batting team gained on the play
    li: float | None = None  # leverage index before a plate appearance
    at_bat: AtBat | None = None  # plate appearances only

    @property
    def scoring(self) -> bool:
        return self.runs > 0

    @property
    def score_text(self) -> str:
        return ", ".join(f"{abbr} {runs}" for abbr, runs in self.score.items())


@dataclass
class HalfInning:
    inning: int
    half: str  # top or bottom
    batting: str  # team name
    plays: list[Play] = field(default_factory=list)
    runs: int = 0
    hits: int = 0
    errors: int = 0
    left_on_base: int = 0

    @property
    def label(self) -> str:
        return f"{'Top' if self.half == 'top' else 'Bottom'} {ordinal(self.inning)}"

    @property
    def summary(self) -> str:
        return (
            f"Runs: {self.runs}, Hits: {self.hits}, Errors: {self.errors}, "
            f"Left on base: {self.left_on_base}"
        )


@dataclass
class GameLog:
    away: str
    home: str
    halves: list[HalfInning]
    home_we_start: float = 0.5

    def win_curve(self) -> list[tuple[HalfInning, Play]]:
        """Plays that move win expectancy (plate appearances and steals), in order."""
        return [(h, p) for h in self.halves for p in h.plays if p.wpa is not None]

    def plays(self, filter: str = "all") -> list[tuple[HalfInning, list[Play]]]:
        """Half innings with their plays; filter is 'all' or 'scoring'."""
        if filter not in ("all", "scoring"):
            raise ValueError(f"unknown play filter {filter!r}")
        out = []
        for half in self.halves:
            plays = [p for p in half.plays if filter == "all" or p.scoring]
            if plays:
                out.append((half, plays))
        return out


def game_log(events: list[dict]) -> GameLog:
    roster = Roster(events)
    abbr = {s: roster.teams[s].abbr for s in ("away", "home")}
    score = {"away": 0, "home": 0}
    current = {s: roster.teams[s].starter for s in ("away", "home")}
    halves: list[HalfInning] = []
    full = roster.full
    # Extra innings start with a runner on second unless the league turned it off.
    extras = [e for e in events if e.get("inning", 0) > 9 and e["type"] == "pa"]
    auto = not extras or any(e["type"] == "auto_runner" for e in events)

    def we(e: dict, bases: list, outs: int) -> float:
        lead = score["home"] - score["away"]
        return win_expectancy(e["inning"], e["half"], lead, bases, outs, auto)

    home_we = we_start = win_expectancy(1, "top", 0, [None] * 3, 0, auto)
    bats = {
        p["id"]: p["bats"] for t in roster.teams.values() for p in t.lineup + t.bench if "bats" in p
    }
    arms = {p["id"]: p for t in roster.teams.values() for p in t.pitchers}

    for i, e in enumerate(events):
        if e["type"] in ("game_start", "final"):
            continue
        offense = "away" if e["half"] == "top" else "home"
        defense = "home" if offense == "away" else "away"
        if not halves or (halves[-1].inning, halves[-1].half) != (e["inning"], e["half"]):
            halves.append(HalfInning(e["inning"], e["half"], roster.teams[offense].name))
        half = halves[-1]

        def fielder(pos: str, defense: str = defense) -> str:
            return f"{POSITION_TITLE[pos]} {roster.fielder(defense, pos, current[defense])}"

        play: Play | None = None
        if e["type"] == "pa":
            lead = score["home"] - score["away"]
            li = leverage_index(e["inning"], e["half"], lead, e["bases_before"], e["outs_before"])
            runs = len(e["scored"])
            before = {abbr["away"]: score["away"], abbr["home"]: score["home"]}
            score[offense] += runs
            half.runs += runs
            half.hits += e["result"] in ("1B", "2B", "3B", "HR")
            half.errors += e["result"] == "E"
            play = Play(
                description=_describe_pa(e, full, fielder),
                kind="pa",
                batter=e["batter"],
                pitcher=e["pitcher"],
                count=f"{e['count'][0]}-{e['count'][1]}",
                outs=e["outs_before"] + e["outs"],
                runs=runs,
                li=li,
                at_bat=_at_bat(e, i, before, bats, arms, full),
            )
        elif e["type"] == "stolen_base":
            base = BASE_NAME[e.get("base", "2B")]
            play = Play(f"{full[e['runner']]} steals {base} base.", e["type"], outs=e["outs_after"])
        elif e["type"] == "caught_stealing":
            base = e.get("base", "2B")
            cover = fielder("3B" if base == "3B" else "SS")
            desc = (
                f"{full[e['runner']]} caught stealing {BASE_NAME[base]} base, "
                f"{fielder('C')} to {cover}."
            )
            play = Play(desc, e["type"], outs=e["outs_after"])
        elif e["type"] == "pitching_change":
            side = roster.side_of(e["team"])
            current[side] = e["into"]
            play = Play(
                f"Pitching Change: {full[e['into']]} replaces {full[e['out']]}.",
                e["type"],
                pitcher=e["into"],
                outs=e["outs"],
            )
        elif e["type"] == "defensive_sub":
            side = roster.side_of(e["team"])
            team = roster.teams[side]
            out_pos = next((p for p, pid in team.positions.items() if pid == e["out"]), None)
            replaced = f"{POSITION_TITLE[out_pos]} " if out_pos in POSITION_TITLE else ""
            desc = (
                f"Defensive Substitution: {full[e['into']]} replaces {replaced}{full[e['out']]}, "
                f"batting {ordinal(e['slot'] + 1)}, playing {FIELD_NAME[e['position']]}."
            )
            if e.get("moved"):
                m = e["moved"]
                was = next((p for p, pid in team.positions.items() if pid == m["player"]), None)
                if was in FIELD_NAME and m["position"] in FIELD_NAME:
                    desc += (
                        f" Defensive switch from {FIELD_NAME[was]} to "
                        f"{FIELD_NAME[m['position']]} for {full[m['player']]}."
                    )
            team.substitute(e)
            play = Play(desc, e["type"], outs=0)
        elif e["type"] == "pinch_hitter":
            side = roster.side_of(e["team"])
            roster.teams[side].substitute(e)
            play = Play(
                f"Offensive Substitution: Pinch-hitter {full[e['into']]} replaces "
                f"{full[e['out']]}.",
                e["type"],
                outs=e.get("outs", 0),
            )
        elif e["type"] in ("wild_pitch", "passed_ball", "balk"):
            runs = len(e["scored"])
            score[offense] += runs
            half.runs += runs
            pitcher = full[e["pitcher"]]
            lead = {
                "wild_pitch": f"Wild pitch by pitcher {pitcher}.",
                "passed_ball": f"Passed ball by {fielder('C')}.",
                "balk": f"{pitcher} balks.",
            }[e["type"]]
            play = Play(
                " ".join([lead, *_runner_notes(e, full)]),
                e["type"],
                pitcher=e["pitcher"],
                outs=e["outs_after"],
                runs=runs,
            )
        elif e["type"] == "auto_runner":
            play = Play(f"Runner placed on 2nd: {full[e['runner']]}.", e["type"])
        elif e["type"] == "half_end":
            half.left_on_base = e["left_on_base"]
        if play is not None and e["type"] in (
            "pa",
            "stolen_base",
            "caught_stealing",
            "wild_pitch",
            "passed_ball",
            "balk",
        ):
            after = we(e, e["bases_after"], play.outs)
            play.home_we = after
            play.wpa = (after - home_we) * (1 if offense == "home" else -1)
            home_we = after
        if play is not None:
            play.score = {abbr["away"]: score["away"], abbr["home"]: score["home"]}
            half.plays.append(play)

    return GameLog(roster.teams["away"].name, roster.teams["home"].name, halves, we_start)


def _at_bat(e: dict, i: int, score: dict, bats: dict, arms: dict, full: dict) -> AtBat:
    arm = arms.get(e["pitcher"], {})
    throws = arm.get("throws", "R")
    side = bats.get(e["batter"], "R")
    if side == "S":
        side = "L" if throws == "R" else "R"
    pitches = sequence(
        f"{i}:{e['batter']}:{e['pitcher']}:{e['inning']}{e['half']}",
        e["result"],
        e["pitches"],
        tuple(e["count"]),
        arm.get("arsenal", ()),
        side,
        e.get("k_type"),
        len(e["scored"]),
    )
    return AtBat(
        title=RESULT_TITLE.get(e["result"], e["result"]),
        batter=full[e["batter"]],
        pitcher=full[e["pitcher"]],
        bats=side,
        throws=throws,
        position=e.get("position", ""),
        outs=e["outs_before"],
        bases=tuple(b is not None for b in e["bases_before"]),
        score=score,
        pitches=pitches,
    )


def _describe_pa(e: dict, full: dict[str, str], fielder) -> str:
    res, bip, pos = e["result"], e["bip"], e["fielder"]
    batter = full[e["batter"]]
    ball = BALL_TYPE.get(bip or "", "")
    if e.get("bunt"):
        lead = _describe_bunt(e, batter, fielder)
    elif res == "K":
        lead = (
            f"{batter} strikes out swinging."
            if e.get("k_type") == "swinging"
            else f"{batter} called out on strikes."
        )
    elif res == "BB":
        lead = f"{batter} walks."
    elif res == "HBP":
        lead = f"{batter} hit by pitch."
    elif res == "HR":
        ball = BALL_TYPE[e.get("batted_ball", "FB")]
        lead = f"{batter} homers on a {ball} to {e['direction']}."
    elif res == "2B" and e.get("ground_rule"):
        lead = f"{batter} hits a ground-rule double on a {ball} to {fielder(pos)}."
    elif res in HIT_VERB:
        lead = f"{batter} {HIT_VERB[res]} on a {ball} to {fielder(pos)}."
    elif res == "GO":
        if pos == "1B":
            lead = f"{batter} grounds out to {fielder('1B')}."
        else:
            lead = f"{batter} grounds out, {fielder(pos)} to {fielder('1B')}."
    elif res in OUT_VERB:
        lead = f"{batter} {OUT_VERB[res]} to {fielder(pos)}."
    elif res == "SF":
        lead = f"{batter} out on a sacrifice fly to {fielder(pos)}."
    elif res == "DP":
        chain = " to ".join(fielder(p) for p in dp_positions(pos))
        lead = f"{batter} grounds into a double play, {chain}."
    elif res == "FC":
        cover = "SS" if pos == "2B" else "2B"
        lead = f"{batter} grounds into a force out, {fielder(pos)} to {fielder(cover)}."
    elif res == "E":
        lead = f"{batter} reaches on a fielding error by {fielder(pos)}."
    else:  # pragma: no cover - every engine result is handled above
        lead = f"{batter} {res}."
    return " ".join([lead, *_runner_notes(e, full)])


def _describe_bunt(e: dict, batter: str, fielder) -> str:
    res, pos = e["result"], e["fielder"]
    if res == "K":
        return f"{batter} strikes out on a foul bunt."
    if res == "SH":
        if pos == "1B":
            return f"{batter} out on a sacrifice bunt to {fielder('1B')}."
        return f"{batter} out on a sacrifice bunt, {fielder(pos)} to {fielder('1B')}."
    if res == "1B":
        return f"{batter} singles on a bunt ground ball to {fielder(pos)}."
    if res == "PO":
        return f"{batter} pops out on a bunt to {fielder(pos)}."
    if res == "DP":
        return (
            f"{batter} bunts into a double play, {fielder(pos)} to {fielder('SS')} "
            f"to {fielder('1B')}."
        )
    cover = "3B" if e["bases_before"][1] else "SS"
    return f"{batter} bunts into a force out, {fielder(pos)} to {fielder(cover)}."


def _runner_notes(e: dict, full: dict[str, str]) -> list[str]:
    """'Mookie Betts scores. Shohei Ohtani to 3rd.' for runners already on base."""
    order = {"3B": 0, "2B": 1, "1B": 2}
    moves = sorted((m for m in e["runners"] if m["start"] != "B"), key=lambda m: order[m["start"]])
    notes = []
    for m in moves:
        name = full[m["runner"]]
        if m["end"] == "score":
            notes.append(f"{name} scores.")
        elif m["end"] == "out" and m.get("out_at"):
            at = "home" if m["out_at"] == "home" else BASE_NAME[m["out_at"]]
            notes.append(f"{name} thrown out at {at}.")
        elif m["end"] == "out":
            notes.append(f"{name} out at {NEXT_BASE[m['start']]}.")
        elif m["end"] != m["start"]:
            notes.append(f"{name} to {BASE_NAME[m['end']]}.")
    return notes


def format_log(log: GameLog, filter: str = "all") -> str:
    """Plain-text game log, all plays or scoring plays only."""
    rows: list[str] = []
    for half, plays in log.plays(filter):
        rows += ["", f"{half.label} ({half.batting})"]
        for p in plays:
            count = f" [{p.count}]" if p.count else ""
            score = f"  ({p.score_text})" if p.scoring else ""
            rows.append(f"  {p.description}{count}{score}")
        if filter == "all":
            rows.append(f"  {half.summary}")
    return "\n".join(rows[1:]) if rows else "No scoring plays."
