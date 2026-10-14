"""Weekly team recaps (Milestone 9): a short template-written summary of a sim
week for each team from its box scores: the record, the best hitter and
starter, the games worth mentioning, and how the bullpen and the lineup
were covered. Everything comes from the week's events, so a recap never
disagrees with a box score. An LLM-written recap from the same facts can
replace the templates later.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from npl_sim.engine.box import BoxScore, box_score
from npl_sim.engine.workload import WeekResult
from npl_sim.reporting.week import _names
from npl_sim.season_stats import SeasonStats
from npl_sim.stats_table import COLUMNS, cell, team_totals

KEY_GAMES = 3
# The team lines under each recap, from the stats page's columns.
HITTING = ("avg", "obp", "slg", "ops", "r", "hr", "rbi", "sb", "bb", "k")
PITCHING = ("era", "whip", "ip", "h", "er", "hr", "bb", "k")


@dataclass
class Recap:
    team: str
    record: tuple[int, int]
    headline: str
    lines: list[str] = field(default_factory=list)
    hitting: list[tuple[str, str]] = field(default_factory=list)  # (label, value)
    pitching: list[tuple[str, str]] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join([self.headline, *self.lines])


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _ip(outs: int) -> str:
    return f"{outs // 3}.{outs % 3}"


def _hitter_line(totals: dict) -> str:
    parts = [f"{totals['h']}-for-{totals['ab']}"]
    for key, label in (("hr", "HR"), ("rbi", "RBI"), ("sb", "SB")):
        if totals[key]:
            parts.append(f"{totals[key]} {label}")
    return ", ".join(parts)


def _best_hitter(boxes: list[BoxScore], team: str) -> str | None:
    week: dict[str, dict] = {}
    for box in boxes:
        for pid, line in box.batting[team].items():
            t = week.setdefault(
                pid, defaultdict(int, name=line.name)
            )  # fmt: skip
            for key in ("ab", "h", "doubles", "triples", "hr", "r", "rbi", "bb", "sb"):
                t[key] += getattr(line, key)
    if not week:
        return None

    def value(t) -> int:  # total bases, runs, RBI, walks and steals
        return (
            t["h"]
            + t["doubles"]
            + 2 * t["triples"]
            + 3 * t["hr"]
            + t["r"]
            + t["rbi"]
            + t["bb"]
            + t["sb"]
        )

    best = max(week.values(), key=lambda t: (value(t), t["h"]))
    if best["ab"] == 0:
        return None
    return f"Best bat: {best['name']}, {_hitter_line(best)}."


def _best_starter(boxes: list[BoxScore], team: str) -> str | None:
    week: dict[str, dict] = {}
    for box in boxes:
        arms = box.pitching[team]
        pid, line = next(iter(arms.items()))  # the first arm in the box is the starter
        t = week.setdefault(pid, defaultdict(int, name=line.name))
        t["gs"] += 1
        for key in ("outs", "er", "k"):
            t[key] += getattr(line, key)
        t["w"] += "W" in line.decisions
        t["l"] += "L" in line.decisions
    if not week:
        return None
    best = max(week.values(), key=lambda t: (t["outs"] - 6 * t["er"] + t["k"], t["outs"]))
    decisions = f", {best['w']}-{best['l']}" if best["w"] or best["l"] else ""
    starts = "" if best["gs"] == 1 else f"{_plural(best['gs'], 'start')}, "
    return (
        f"Best start: {best['name']}, {starts}{_ip(best['outs'])} IP, {best['er']} ER, "
        f"{best['k']} K{decisions}."
    )


def _game_notes(games: list[dict], boxes: list[BoxScore], team: str) -> list[str]:
    notes: list[tuple[int, str]] = []
    for g, box in zip(games, boxes, strict=True):
        other = box.home if team == box.away else box.away
        mine, theirs = box.runs[team], box.runs[other]
        won = mine > theirs
        final = g["events"][-1]
        innings = final["inning"]
        where = "on the road" if team == box.away else "at home"
        score = f"{max(mine, theirs)}-{min(mine, theirs)}"
        verb = "beat" if won else "lost to"
        walk_off = won and team == box.home and final["half"] == "bottom" and innings >= 9
        if walk_off and innings > 9:
            text, weight = f"Walked off {other} {score} in {innings} innings.", 5
        elif walk_off and mine - theirs == 1:
            text, weight = f"Walked off {other} {score}.", 4
        elif innings > 9:
            text, weight = f"{verb.capitalize()} {other} {score} in {innings} innings.", 5
        elif abs(mine - theirs) >= 6:
            text, weight = f"{verb.capitalize()} {other} {score} {where}.", 3
        elif abs(mine - theirs) == 1:
            text, weight = f"{verb.capitalize()} {other} {score}, a one-run game.", 2
        else:
            continue
        notes.append((weight, text))
    notes.sort(key=lambda n: -n[0])
    return [text for _, text in notes[:KEY_GAMES]]


def _coverage(games: list[dict], team: str, names: dict[str, str]) -> list[str]:
    moves = [g["moves"][team] for g in games]
    lines = []
    bullpen = sum(m.bullpen_game for m in moves)
    standin_arm = sum(m.standin_pitcher for m in moves)
    if bullpen:
        lines.append(f"Bullpen games: {bullpen}.")
    if standin_arm:
        lines.append(
            f"The whole pen was used up in {_plural(standin_arm, 'game')}: a (AAA) arm relieved."
        )
    replaced: dict[str, set[str]] = defaultdict(set)
    standins = 0
    for m in moves:
        for r in m.replaced:
            replaced[r["out"]].add(r["into"])
            standins += bool(r["standin"])
    if replaced:
        who = ", ".join(sorted(names.get(pid, pid) for pid in replaced))
        extra = f"; a (AAA) stand-in played in {_plural(standins, 'spot')}" if standins else ""
        lines.append(f"Out of playing time or covered: {who}{extra}.")
    return lines


def _bullpen_use(boxes: list[BoxScore], team: str) -> str | None:
    outs: dict[str, list] = {}
    saves = 0
    for box in boxes:
        for i, (pid, line) in enumerate(box.pitching[team].items()):
            saves += "S" in line.decisions
            if i:  # every arm after the starter
                row = outs.setdefault(pid, [line.name, 0, 0])
                row[1] += line.outs
                row[2] += 1
    if not outs:
        return None
    top = max(outs.values(), key=lambda r: (r[1], r[2]))
    work = f"{top[0]} ({_ip(top[1])} IP in {_plural(top[2], 'game')})"
    text = f"Bullpen: {len(outs)} relievers used, most work from {work}"
    return text + (f"; {_plural(saves, 'save')}." if saves else ".")


def _team_lines(team: str, boxes: list[BoxScore]) -> tuple[list, list]:
    """The team's hitting and pitching totals for the week as (label, text) pairs."""
    week = SeasonStats()
    for box in boxes:
        week.add_game(date.min, box)
    out = []
    for kind, keys in (("hitting", HITTING), ("pitching", PITCHING)):
        total = next((t for t in team_totals(week, kind) if t.team == team), None)
        by_key = {c.key: c for c in COLUMNS[kind]}
        out.append(
            [(by_key[k].label, cell(by_key[k], getattr(total, k))) for k in keys] if total else []
        )
    return out[0], out[1]


def team_recap(team: str, games: list[dict], known: dict[str, str] | None = None) -> Recap:
    """`games` are the week's `WeekResult.games` that `team` played in; `known` maps ids to
    names for players who sat out and so aren't in any box score."""
    boxes = [box_score(g["events"]) for g in games]
    wins = sum(box.winner == team for box in boxes)
    losses = len(boxes) - wins
    runs = sum(box.runs[team] for box in boxes)
    allowed = sum(box.runs[next(t for t in box.runs if t != team)] for box in boxes)
    names: dict[str, str] = dict(known or {})
    for g in games:
        names.update(_names(g["events"]))
    headline = (
        f"{team} went {wins}-{losses} this week, scoring {runs} and allowing {allowed}."
        if boxes
        else f"{team} did not play this week."
    )
    lines = [
        line for line in (_best_hitter(boxes, team), _best_starter(boxes, team)) if line is not None
    ]
    lines += _game_notes(games, boxes, team)
    pen = _bullpen_use(boxes, team)
    if pen:
        lines.append(pen)
    lines += _coverage(games, team, names)
    hitting, pitching = _team_lines(team, boxes)
    return Recap(team, (wins, losses), headline, lines, hitting, pitching)


def week_recaps(result: WeekResult, names: dict[str, str] | None = None) -> dict[str, Recap]:
    """A recap for every team that played, best record first."""
    by_team: dict[str, list[dict]] = defaultdict(list)
    for g in result.games:
        by_team[g["away"]].append(g)
        by_team[g["home"]].append(g)
    recaps = {team: team_recap(team, games, names) for team, games in by_team.items()}
    return dict(sorted(recaps.items(), key=lambda kv: (-kv[1].record[0], kv[0])))
