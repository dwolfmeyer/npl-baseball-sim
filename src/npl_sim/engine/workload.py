"""Weekly playing time: each player's sim PA and innings follow his real week.

The spec keeps Scoresheet's weekly budget ("PA and IP capped at roughly the
week's MLB totals, with fudge factors and carried-over starts"). There are no
rest-day rules (Dave, 2026-10-13: they left too many games with replacement
pitchers): the budgets are the only limit. The rules here:

Hitters
- Budget = MLB plate appearances that week x PA_SLACK, rounded.
- A hitter with no MLB PA that week doesn't play. Before each game, a hitter
  starts while he has any PA left. His spot is covered the way
  a manager uses his bench (Dave, 2026-10-13), hardest positions first:
  1. a bench player with PA left who is eligible there;
  2. the DH, if he is eligible there (the DH spot then opens);
  3. the same two through a move a manager would make (FLEX): only eligible
     catchers catch; a first baseman to a corner outfield spot; a third
     baseman to first or a corner; a shortstop or second baseman anywhere on
     the infield; an outfielder anywhere in the outfield;
  4. another fielder moving over (eligible or FLEX) while a bench player or
     the DH covers his spot;
  5. only then a (AAA) stand-in at Scoresheet's penalty levels
     (replacement.py).
  An open DH spot goes to the bench player with the most PA left, else a
  stand-in. The sub bats in the spot of the player he replaced; a player who
  moves keeps his own. Defensive-spectrum costs still apply (fielding.py).

Starters
- Sim starts = MLB starts that week + starts carried from last week (at most
  MAX_CARRIED_STARTS). The rotation turns in the owner's order, skipping a
  starter with no starts left. When nobody can start, the reliever with the
  most innings left opens a bullpen game. Unused starts carry into next week.
- A starter pitches at most once in a series (Dave, 2026-10-13): once he has
  pitched in any role, he doesn't start again or follow an opener in the
  rest of that series, though he can in the next one. Games carry a series id
  for this (play_week's optional fourth element); games without one are
  unrestricted.

Relievers
- Budget = MLB relief outs that week x RELIEF_SLACK, rounded up. In a game a
  reliever comes out when he reaches it (or stays, if nobody else is left).
- A reliever with no MLB relief outs that week doesn't pitch.

Days are 0 (Monday) to 6 (Sunday).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from npl_sim.engine.game import Hitter, Pitcher, TeamCard, simulate_game
from npl_sim.engine.replacement import standin_hitter, standin_pitcher

PA_SLACK = 1.25
RELIEF_SLACK = 1.5
MAX_CARRIED_STARTS = 1
BULLPEN_GAME_PITCHES = 50
RELIEVER_START_OUTS = 9  # a reliever starting in the sim goes 3 innings at most
# Positions a player can be moved to beyond his eligible ones (see above).
FLEX = {
    "1B": {"LF", "RF"},
    "3B": {"1B", "LF", "RF"},
    "2B": {"1B", "3B", "SS"},
    "SS": {"1B", "2B", "3B"},
    "LF": {"CF", "RF"},
    "CF": {"LF", "RF"},
    "RF": {"LF", "CF"},
}
# Hardest spots are covered first, so scarce gloves go where they're needed.
FILL_ORDER = ("C", "SS", "CF", "2B", "3B", "RF", "LF", "1B", "DH")


@dataclass(frozen=True)
class MlbWeek:
    """What each player did in MLB this week, by engine player id."""

    pa: dict[str, int] = field(default_factory=dict)
    starts: dict[str, int] = field(default_factory=dict)
    relief_outs: dict[str, int] = field(default_factory=dict)
    # Outs a pitcher threw as an MLB starter. They count toward his budget only
    # on a team that doesn't start him: a manager who keeps an MLB starter in
    # the pen gets those innings from him in relief.
    start_outs: dict[str, int] = field(default_factory=dict)


@dataclass
class Carry:
    """What one team brings into next week."""

    starts: dict[str, int] = field(default_factory=dict)
    rotation_next: int = 0  # index of the next starter up


@dataclass
class WeekState:
    """One team's budgets through the week."""

    pa_left: dict[str, int]
    starts_left: dict[str, int]
    outs_left: dict[str, int]
    rotation_next: int = 0
    # Pitchers who have pitched in each series so far, by series id.
    series_pitchers: dict[object, set[str]] = field(default_factory=dict)

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
        )

    def close(self) -> Carry:
        """Carry into next week: unused starts and the rotation turn."""
        return Carry(
            starts={p: min(n, MAX_CARRIED_STARTS) for p, n in self.starts_left.items() if n},
            rotation_next=self.rotation_next,
        )

    # -- availability --------------------------------------------------------
    def can_start(self, pid: str, series: object = None) -> bool:
        return self.starts_left.get(pid, 0) > 0 and not self.pitched_in(pid, series)

    def pitched_in(self, pid: str, series: object) -> bool:
        """Has this pitcher already pitched in this series?"""
        return series is not None and pid in self.series_pitchers.get(series, ())

    def can_relieve(self, pid: str) -> bool:
        return self.outs_left.get(pid, 0) > 0

    def can_hit(self, pid: str) -> bool:
        return self.pa_left.get(pid, 0) > 0

    # -- after a game --------------------------------------------------------
    def record(self, events: list[dict], team: str, moves: Moves, series: object = None) -> None:
        """Charge this team's PA, outs and start from one game, and note who
        pitched in its series."""
        starter = None if moves.bullpen_game else moves.starter
        if moves.bullpen_game and self.starts_left.get(moves.starter, 0) > 0:
            self.starts_left[moves.starter] -= 1  # a reliever who started in MLB opened
        if starter is not None:
            self.starts_left[starter] = self.starts_left.get(starter, 0) - 1
        for e in events:
            if e["type"] == "pa" and e["team"] == team:
                self.pa_left[e["batter"]] = self.pa_left.get(e["batter"], 0) - 1
            elif e["type"] == "pa":
                if e["pitcher"] != starter:
                    self._out(e["pitcher"], e["outs"])
            elif e["type"] == "caught_stealing" and e["pitcher"] != starter:
                self._out(e["pitcher"], 1)  # only our pitchers have relief budgets here
        if series is not None:
            used = self.series_pitchers.setdefault(series, set())
            used.add(moves.starter)
            used.update(e["pitcher"] for e in events if e["type"] == "pa" and e["team"] != team)

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
    shifted: list[dict] = field(default_factory=list)  # {player, from, to}: moved to cover
    unavailable: list[str] = field(default_factory=list)  # relievers used up for the week
    standin_pitcher: bool = False  # the whole pen was out: a (AAA) arm relieves


def next_starter(team: WeekTeam, state: WeekState, series: object = None) -> Pitcher | None:
    n = len(team.rotation)
    for k in range(n):
        i = (state.rotation_next + k) % n
        if state.can_start(team.rotation[i].id, series):
            state.rotation_next = (i + 1) % n
            return team.rotation[i]
    return None


def game_card(
    team: WeekTeam,
    state: WeekState,
    starter: Pitcher | None,
    opposing_throws: str,
    series: object = None,
) -> tuple[TeamCard, Moves]:
    """The engine's card for one game after this week's playing time, with
    `starter` from next_starter (None = a bullpen game)."""
    # The starter's own rotation slot can carry its pitch limit, hook, an
    # opener's piggyback and the pen built around it.
    card = team.slot_cards.get(starter.id, team.card) if starter else team.card
    if starter is not None:
        starter = card.starter if card.starter.id == starter.id else starter

    rotation = {p.id for p in team.rotation}

    def available(p: Pitcher) -> bool:
        if p.id in card.follow:  # a piggyback: a starter's budget or a reliever's
            if p.id in rotation and state.pitched_in(p.id, series):
                return False  # a starter pitches once a series
            return state.outs_left.get(p.id, 0) > 0 or state.starts_left.get(p.id, 0) > 0
        return state.can_relieve(p.id)

    relievers = [p for p in card.bullpen if available(p)]
    moves = Moves(starter="")
    moves.unavailable = [p.id for p in card.bullpen if p not in relievers]
    forced_outs: dict[str, int] = {}
    if starter is None:
        # Bullpen game: only a reliever who started a game in MLB this week opens
        # (three innings at most); anyone else would be a closer starting, so a
        # (AAA) arm opens instead. A listed opener is a rotation slot, not here.
        openers = [p for p in relievers if state.starts_left.get(p.id, 0) > 0]
        opener = max(openers, key=lambda p: state.starts_left.get(p.id, 0), default=None)
        if opener is None:
            opener = standin_pitcher(f"AAA-{card.short}-S")
        else:
            relievers = [p for p in relievers if p.id != opener.id]
            forced_outs[opener.id] = RELIEVER_START_OUTS
        starter = replace(opener, pitch_limit=BULLPEN_GAME_PITCHES)
        moves.bullpen_game = True
    moves.starter = starter.id
    if not relievers:
        relievers = [standin_pitcher(f"AAA-{card.short}-P")]
        moves.standin_pitcher = True

    lineup = list(team.lineups.get(opposing_throws) or card.lineup)
    lineup, in_use = _cover(team, state, lineup, card.short, moves)

    outs_limit = {
        p.id: state.outs_left.get(p.id, 0)
        for p in relievers
        if p.id not in card.follow and not p.id.startswith("AAA-")
    }
    outs_limit.update(forced_outs)
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


def can_cover(fields: set[str], pos: str, flex: bool = True) -> bool:
    """Whether a player eligible at `fields` can play `pos` (with FLEX moves)."""
    if pos == "DH" or pos in fields:
        return True
    return flex and any(pos in FLEX.get(f, ()) for f in fields)


def _cover(team: WeekTeam, state: WeekState, lineup: list, short: str, moves: Moves):
    """The lineup with every hitter out of PA covered (see the module docstring)."""
    in_use = {h.id for h, _ in lineup}
    out = {i: h for i, (h, _) in enumerate(lineup) if not state.can_hit(h.id)}
    todo = sorted(out, key=lambda i: FILL_ORDER.index(lineup[i][1]))
    filled: dict[int, Hitter] = {}  # slot -> who bats there now

    def fields(h: Hitter) -> set[str]:
        return team.fields.get(h.id, set())

    def bench(pos: str, flex: bool) -> Hitter | None:
        return _bench_fit(team, state, pos, in_use, flex)

    def dh_for(pos: str, flex: bool) -> int | None:
        return next(
            (
                j
                for j, (h, p) in enumerate(lineup)
                if p == "DH" and j not in out and can_cover(fields(h), pos, flex)
            ),
            None,
        )

    def sub_in(slot: int, h: Hitter, pos: str, standin: bool = False) -> None:
        lineup[slot] = (h, pos)
        in_use.add(h.id)
        filled[slot] = h
        moves.replaced.append(
            {"out": out[slot].id, "into": h.id, "position": pos, "standin": standin}
        )

    def shift(j: int, to: str) -> None:
        h, frm = lineup[j]
        lineup[j] = (h, to)
        moves.shifted.append({"player": h.id, "from": frm, "to": to})

    while todo:
        slot = todo.pop(0)
        pos = lineup[slot][1]
        if pos == "DH":
            sub = bench("DH", False)
            sub_in(slot, sub or standin_hitter(f"AAA-{short}-DH-{slot + 1}", "DH"), "DH", not sub)
            continue
        done = False
        for flex in (False, True):
            sub = bench(pos, flex)
            if sub is not None:
                sub_in(slot, sub, pos)
                done = True
                break
            j = dh_for(pos, flex)
            if j is not None:  # the DH takes the field; the open spot becomes the DH
                shift(j, pos)
                lineup[slot] = (out[slot], "DH")
                todo.insert(0, slot)
                done = True
                break
        if done:
            continue
        # Another fielder slides over and the bench (or the DH) covers his spot.
        for j, (h, q) in enumerate(lineup):
            if j == slot or j in out or q == "DH" or not can_cover(fields(h), pos):
                continue
            sub = bench(q, True)
            if sub is not None:
                shift(j, pos)
                sub_in(slot, sub, q)
                done = True
                break
            d = dh_for(q, True)
            if d is not None:
                shift(j, pos)
                shift(d, q)
                lineup[slot] = (out[slot], "DH")
                todo.insert(0, slot)
                done = True
                break
        if not done:
            sub_in(slot, standin_hitter(f"AAA-{short}-{pos}-{slot + 1}", pos), pos, True)
    return lineup, in_use


def _bench_fit(
    team: WeekTeam, state: WeekState, pos: str, in_use: set[str], flex: bool = False
) -> Hitter | None:
    """The bench player with the most PA left who can play `pos`."""
    fits = [
        h
        for h in team.bench
        if h.id not in in_use
        and state.can_hit(h.id)
        and can_cover(team.fields.get(h.id, set()), pos, flex)
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
    (day, away, home) or (day, away, home, series id), and `mlb` is the
    league's real week by player id.
    `audit` keeps each game's per-PA inputs (npl_sim/audit.py) as "audit"."""
    carry = carry or {}
    states = {name: WeekState.open(mlb, carry.get(name), t.players) for name, t in teams.items()}
    for name, t in teams.items():
        in_rotation = {p.id for p in t.rotation}
        for pid in t.players & mlb.start_outs.keys() - in_rotation:
            outs = math.ceil(mlb.start_outs[pid] * RELIEF_SLACK)
            states[name].outs_left[pid] = states[name].outs_left.get(pid, 0) + outs
    played = []
    for n, (day, away, home, *rest) in enumerate(sorted(games, key=lambda g: g[0])):
        series = rest[0] if rest else None
        starters = {k: next_starter(teams[k], states[k], series) for k in (away, home)}

        # A bullpen game's opener isn't known until the card is built.
        throws = {k: p.throws if p is not None else "R" for k, p in starters.items()}
        cards, moves = {}, {}
        for k, other in ((away, home), (home, away)):
            cards[k], moves[k] = game_card(teams[k], states[k], starters[k], throws[other], series)
        trail: list | None = [] if audit else None
        events = simulate_game(cards[away], cards[home], env, seed * 1000 + n, trail)
        for k in (away, home):
            states[k].record(events, k, moves[k], series)
        game = {"day": day, "away": away, "home": home, "events": events, "moves": moves}
        if audit:
            game["audit"] = trail
        played.append(game)
    return WeekResult(played, states, {k: s.close() for k, s in states.items()})
