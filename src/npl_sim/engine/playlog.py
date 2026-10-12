"""The game log: Gameday-style play-by-play, filterable to all plays or scoring plays."""

from __future__ import annotations

from dataclasses import dataclass, field

from npl_sim.engine.box import dp_positions
from npl_sim.engine.detail import POSITION_TITLE
from npl_sim.engine.names import BASE_NAME, NEXT_BASE, Roster, ordinal

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


@dataclass
class Play:
    description: str
    kind: str  # pa, stolen_base, caught_stealing, pitching_change, defensive_sub, auto_runner
    batter: str | None = None
    pitcher: str | None = None
    count: str | None = None  # "1-2" on the final pitch of a plate appearance
    outs: int = 0  # outs after the play
    runs: int = 0
    score: dict[str, int] = field(default_factory=dict)  # by team abbreviation, after the play

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

    for e in events:
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
            runs = len(e["scored"])
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
            )
        elif e["type"] == "stolen_base":
            play = Play(f"{full[e['runner']]} steals 2nd base.", e["type"], outs=e["outs_after"])
        elif e["type"] == "caught_stealing":
            desc = (
                f"{full[e['runner']]} caught stealing 2nd base, {fielder('C')} to {fielder('SS')}."
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
        elif e["type"] == "auto_runner":
            play = Play(f"Runner placed on 2nd: {full[e['runner']]}.", e["type"])
        elif e["type"] == "half_end":
            half.left_on_base = e["left_on_base"]
        if play is not None:
            play.score = {abbr["away"]: score["away"], abbr["home"]: score["home"]}
            half.plays.append(play)

    return GameLog(roster.teams["away"].name, roster.teams["home"].name, halves)


def _describe_pa(e: dict, full: dict[str, str], fielder) -> str:
    res, bip, pos = e["result"], e["bip"], e["fielder"]
    batter = full[e["batter"]]
    ball = BALL_TYPE.get(bip or "", "")
    if res == "K":
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


def _runner_notes(e: dict, full: dict[str, str]) -> list[str]:
    """'Mookie Betts scores. Shohei Ohtani to 3rd.' for runners already on base."""
    order = {"3B": 0, "2B": 1, "1B": 2}
    moves = sorted((m for m in e["runners"] if m["start"] != "B"), key=lambda m: order[m["start"]])
    notes = []
    for m in moves:
        name = full[m["runner"]]
        if m["end"] == "score":
            notes.append(f"{name} scores.")
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
