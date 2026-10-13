"""Weekly playing time: each player's sim PA and innings follow his real week.

The spec keeps Scoresheet's weekly budget ("PA and IP capped at roughly the
week's MLB totals, with fudge factors and carried-over starts") and adds rest
days and multi-day reliever fatigue. The rules here:

Hitters
- Budget = MLB plate appearances that week x PA_SLACK, rounded.
- A hitter with no MLB PA that week doesn't play. Before each game, a hitter
  starts only with at least MIN_PA_TO_START left; otherwise the best bench
  player who fits his position takes the slot, and when none fits a (AAA)
  stand-in at Scoresheet's penalty levels does (replacement.py).

Starters
- Sim starts = MLB starts that week + starts carried from last week (at most
  MAX_CARRIED_STARTS). The rotation turns in the owner's order, skipping a
  starter with no starts left or under START_REST days since his last start.
  When nobody can start, the rested reliever with the most innings left opens
  a bullpen game. Unused starts carry into next week.

Relievers
- Budget = MLB relief outs that week x RELIEF_SLACK, rounded up. In a game a
  reliever comes out when he reaches it (or stays, if nobody else is left).
- A reliever with no MLB relief outs that week doesn't pitch.
- Rest: no third day in a row; a day off after REST_PITCHES_1 pitches and
  two after REST_PITCHES_2.

Days are 0 (Monday) to 6 (Sunday); rest carries across weeks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from npl_sim.engine.game import Hitter, Pitcher, TeamCard, simulate_game
from npl_sim.engine.replacement import standin_hitter, standin_pitcher

PA_SLACK = 1.25
MIN_PA_TO_START = 3
RELIEF_SLACK = 1.5
MAX_CARRIED_STARTS = 1
START_REST = 4  # days off between starts
REST_PITCHES_1 = 25
REST_PITCHES_2 = 40
BULLPEN_GAME_PITCHES = 50


@dataclass(frozen=True)
class MlbWeek:
    """What each player did in MLB this week, by engine player id."""

    pa: dict[str, int] = field(default_factory=dict)
    starts: dict[str, int] = field(default_factory=dict)
    relief_outs: dict[str, int] = field(default_factory=dict)


@dataclass
class Carry:
    """What one team brings into next week."""

    starts: dict[str, int] = field(default_factory=dict)
    rotation_next: int = 0  # index of the next starter up
    # Day of each pitcher's last start / appearances and pitches, relative to
    # next Monday (negative = last week).
    last_start: dict[str, int] = field(default_factory=dict)
    pitched: dict[str, dict[int, int]] = field(default_factory=dict)


@dataclass
class WeekState:
    """One team's budgets and rest through the week."""

    pa_left: dict[str, int]
    starts_left: dict[str, int]
    outs_left: dict[str, int]
    rotation_next: int = 0
    last_start: dict[str, int] = field(default_factory=dict)
    pitched: dict[str, dict[int, int]] = field(default_factory=dict)  # id -> {day: pitches}

    @classmethod
    def open(
        cls, mlb: MlbWeek, carry: Carry | None = None, players: set[str] | None = None
    ) -> WeekState:
        """Budgets for `players` (default: everyone in `mlb`)."""
        carry = carry or Carry()
        mine = (lambda p: True) if players is None else players.__contains__
        starts = dict(mlb.starts)
        for pid, n in carry.starts.items():
            starts[pid] = starts.get(pid, 0) + min(n, MAX_CARRIED_STARTS)
        return cls(
            pa_left={p: round(n * PA_SLACK) for p, n in mlb.pa.items() if n > 0 and mine(p)},
            starts_left={p: n for p, n in starts.items() if n > 0 and mine(p)},
            outs_left={
                p: math.ceil(n * RELIEF_SLACK) for p, n in mlb.relief_outs.items() if n and mine(p)
            },
            rotation_next=carry.rotation_next,
            last_start=dict(carry.last_start),
            pitched={p: dict(d) for p, d in carry.pitched.items()},
        )

    def close(self, days: int = 7) -> Carry:
        """Carry into next week: unused starts, the rotation turn, rest."""
        return Carry(
            starts={p: min(n, MAX_CARRIED_STARTS) for p, n in self.starts_left.items() if n},
            rotation_next=self.rotation_next,
            last_start={p: d - days for p, d in self.last_start.items() if d - days > -7},
            pitched={
                p: {d - days: n for d, n in by_day.items() if d - days >= -2}
                for p, by_day in self.pitched.items()
            },
        )

    # -- availability --------------------------------------------------------
    def can_start(self, pid: str, day: int) -> bool:
        if self.starts_left.get(pid, 0) <= 0:
            return False
        last = self.last_start.get(pid)
        return last is None or day - last > START_REST

    def rested(self, pid: str, day: int) -> bool:
        """Can this pitcher go today on rest alone?"""
        by_day = self.pitched.get(pid, {})
        if (day - 1) in by_day and (day - 2) in by_day:
            return False
        if by_day.get(day - 1, 0) >= REST_PITCHES_1:
            return False
        return by_day.get(day - 2, 0) < REST_PITCHES_2

    def can_relieve(self, pid: str, day: int) -> bool:
        return self.outs_left.get(pid, 0) > 0 and self.rested(pid, day)

    def can_hit(self, pid: str) -> bool:
        return self.pa_left.get(pid, 0) >= MIN_PA_TO_START

    # -- after a game --------------------------------------------------------
    def record(self, events: list[dict], team: str, day: int, moves: Moves) -> None:
        """Charge this team's PA, outs, pitches and start from one game."""
        starter = None if moves.bullpen_game else moves.starter
        if starter is not None:
            self.starts_left[starter] = self.starts_left.get(starter, 0) - 1
            self.last_start[starter] = day
        pitches: dict[str, int] = {}
        for e in events:
            if e["type"] == "pa" and e["team"] == team:
                self.pa_left[e["batter"]] = self.pa_left.get(e["batter"], 0) - 1
            elif e["type"] == "pa":
                pitches[e["pitcher"]] = pitches.get(e["pitcher"], 0) + e["pitches"]
                if e["pitcher"] != starter:
                    self._out(e["pitcher"], e["outs"])
            elif e["type"] == "caught_stealing" and e["pitcher"] != starter:
                self._out(e["pitcher"], 1)  # only our pitchers have relief budgets here
        for pid, n in pitches.items():
            self.pitched.setdefault(pid, {})[day] = n

    def _out(self, pid: str, n: int) -> None:
        if pid in self.outs_left:
            self.outs_left[pid] -= n


@dataclass(frozen=True)
class WeekTeam:
    """One team's settings for the week, before playing time is applied.

    `card` carries the bullpen, relief designations, hooks and the rest of
    the lineup card; `lineups` is the batting order vs LHP ("L") and RHP
    ("R"), `rotation` the starters in order, `bench` the active hitters not
    in the lineup, and `fields` the positions each hitter can play.
    """

    card: TeamCard
    lineups: dict[str, list[tuple[Hitter, str]]]
    rotation: list[Pitcher]
    bench: list[Hitter] = field(default_factory=list)
    fields: dict[str, set[str]] = field(default_factory=dict)
    # Per-starter cards (by starter id) from his rotation slot's plan.
    slot_cards: dict[str, TeamCard] = field(default_factory=dict)

    @property
    def players(self) -> set[str]:
        hitters = [h for lineup in self.lineups.values() for h, _ in lineup]
        hitters += [h for h, _ in self.card.lineup] + self.bench
        hitters += [s.player for s in self.card.defensive_subs]
        arms = self.rotation + self.card.bullpen
        arms += [p for c in self.slot_cards.values() for p in c.bullpen]
        return {p.id for p in hitters + arms}


@dataclass
class Moves:
    """What playing time changed for one team in one game."""

    starter: str
    bullpen_game: bool = False
    replaced: list[dict] = field(default_factory=list)  # {out, into, position, standin}
    unavailable: list[str] = field(default_factory=list)  # relievers resting or used up
    standin_pitcher: bool = False  # the whole pen was out: a (AAA) arm relieves


def next_starter(team: WeekTeam, state: WeekState, day: int) -> Pitcher | None:
    n = len(team.rotation)
    for k in range(n):
        i = (state.rotation_next + k) % n
        if state.can_start(team.rotation[i].id, day):
            state.rotation_next = (i + 1) % n
            return team.rotation[i]
    return None


def game_card(
    team: WeekTeam, state: WeekState, day: int, starter: Pitcher | None, opposing_throws: str
) -> tuple[TeamCard, Moves]:
    """The engine's card for one game after this week's playing time, with
    `starter` from next_starter (None = a bullpen game)."""
    # The starter's own rotation slot can carry its pitch limit, hook, an
    # opener's piggyback and the pen built around it.
    card = team.slot_cards.get(starter.id, team.card) if starter else team.card
    if starter is not None:
        starter = card.starter if card.starter.id == starter.id else starter

    def available(p: Pitcher) -> bool:
        if p.id in card.follow:  # a piggyback: a starter's budget or a reliever's
            return state.rested(p.id, day) and (
                state.outs_left.get(p.id, 0) > 0 or state.starts_left.get(p.id, 0) > 0
            )
        return state.can_relieve(p.id, day)

    relievers = [p for p in card.bullpen if available(p)]
    moves = Moves(starter="")
    moves.unavailable = [p.id for p in card.bullpen if p not in relievers]
    if starter is None:
        # Bullpen game: the rested arm with the most outs left opens.
        opener = max(relievers, key=lambda p: state.outs_left.get(p.id, 0), default=None)
        if opener is None:
            opener = card.bullpen[0] if card.bullpen else team.rotation[0]
        starter = replace(opener, pitch_limit=BULLPEN_GAME_PITCHES)
        relievers = [p for p in relievers if p.id != opener.id]
        moves.bullpen_game = True
    moves.starter = starter.id
    if not relievers:
        relievers = [standin_pitcher(f"AAA-{card.short}-P")]
        moves.standin_pitcher = True

    lineup = list(team.lineups.get(opposing_throws) or card.lineup)
    in_use = {h.id for h, _ in lineup}
    for slot, (hitter, pos) in enumerate(lineup):
        if state.can_hit(hitter.id):
            continue
        sub = _bench_fit(team, state, pos, in_use)
        standin = sub is None
        if standin:
            sub = standin_hitter(f"AAA-{card.short}-{pos}-{slot + 1}", pos)
        lineup[slot] = (sub, pos)
        in_use.add(sub.id)
        moves.replaced.append(
            {"out": hitter.id, "into": sub.id, "position": pos, "standin": standin}
        )

    outs_limit = {
        p.id: state.outs_left.get(p.id, 0)
        for p in relievers
        if p.id not in card.follow and not p.id.startswith("AAA-")
    }
    game = replace(
        card,
        lineup=lineup,
        starter=starter,
        bullpen=relievers,
        outs_limit=outs_limit,
        defensive_subs=tuple(
            s
            for s in card.defensive_subs
            if s.player.id not in in_use and state.can_hit(s.player.id)
        ),
        # Pinch hitters: bench bats with PA left who aren't starting today.
        bench=tuple(h for h in team.bench if h.id not in in_use and state.can_hit(h.id)),
    )
    return game, moves


def _bench_fit(team: WeekTeam, state: WeekState, pos: str, in_use: set[str]) -> Hitter | None:
    """The bench player with the most PA left who can play `pos`."""
    fits = [
        h
        for h in team.bench
        if h.id not in in_use
        and state.can_hit(h.id)
        and (pos == "DH" or pos in team.fields.get(h.id, set()))
    ]
    return max(fits, key=lambda h: state.pa_left.get(h.id, 0), default=None)


@dataclass
class WeekResult:
    games: list[dict]  # {day, away, home, events, moves: {team: Moves}, audit?}
    states: dict[str, WeekState]
    carry: dict[str, Carry]


def play_week(
    teams: dict[str, WeekTeam],
    games: list[tuple[int, str, str]],
    mlb: MlbWeek,
    env,
    seed: int,
    carry: dict[str, Carry] | None = None,
    audit: bool = False,
) -> WeekResult:
    """Play one sim week. `teams` is keyed by team name, `games` lists
    (day, away, home), and `mlb` is the league's real week by player id.
    `audit` keeps each game's per-PA inputs (npl_sim/audit.py) as "audit"."""
    carry = carry or {}
    states = {name: WeekState.open(mlb, carry.get(name), t.players) for name, t in teams.items()}
    played = []
    for n, (day, away, home) in enumerate(sorted(games, key=lambda g: g[0])):
        starters = {k: next_starter(teams[k], states[k], day) for k in (away, home)}

        # A bullpen game's opener isn't known until the card is built.
        throws = {k: p.throws if p is not None else "R" for k, p in starters.items()}
        cards, moves = {}, {}
        for k, other in ((away, home), (home, away)):
            cards[k], moves[k] = game_card(teams[k], states[k], day, starters[k], throws[other])
        trail: list | None = [] if audit else None
        events = simulate_game(cards[away], cards[home], env, seed * 1000 + n, trail)
        for k in (away, home):
            states[k].record(events, k, day, moves[k])
        game = {"day": day, "away": away, "home": home, "events": events, "moves": moves}
        if audit:
            game["audit"] = trail
        played.append(game)
    return WeekResult(played, states, {k: s.close() for k, s in states.items()})
