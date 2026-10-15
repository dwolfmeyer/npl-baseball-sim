"""Simulate one game, plate appearance by plate appearance.

Milestone 5 scope: full PA model, baserunning, steals, double plays, sac
flies, errors, extra innings with the automatic runner, walk-offs, and a
starter hooks by pitch count, times through the order, runs allowed and
leverage (hooks.py), and relievers by the owner's bullpen designations.

MLB playing rules it follows: the extra-inning runner on second; the
three-batter minimum (a pitcher faces three batters or finishes the half
inning before he can be pulled); a walk-off ends the game when the winning
run scores, the batter credited with only as many bases as that runner
advanced (all runs count on a home run); ground-rule doubles (every runner
gets exactly two bases); sacrifice flies and sacrifice bunts (not at bats);
wild pitches, passed balls (runs unearned) and balks.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field

from npl_sim.engine import bullpen as pen
from npl_sim.engine import detail, fielding
from npl_sim.engine.alignment import GROUND_BALL_PLAY, AlignmentRule, choose
from npl_sim.engine.hooks import StarterHook
from npl_sim.engine.manager import Baserunning, Bunting, PinchHitting, run_value
from npl_sim.engine.model import (
    LEAGUE_FATIGUE,
    Defense,
    LeagueEnv,
    ParkFactors,
    Rates,
    pa_probabilities,
)

# Mean pitches per PA by outcome.
PITCHES = {"K": 4.8, "BB": 5.7, "HBP": 3.6, "HR": 3.5, "BIP": 3.4, "BUNT": 2.6}
# Share of doubles in the air that bounce out of play (ground-rule doubles).
GROUND_RULE_DOUBLE = 0.07
# Chance of each before a PA with runners on, tuned to recent MLB (per team
# game about 0.33 wild pitches, 0.05 passed balls, 0.03 balks); a wild
# pitcher's chance scales with his walk rate.
WILD_PITCH, PASSED_BALL, BALK = 0.019, 0.003, 0.0025
# How a sacrifice-bunt attempt turns out: sacrifice, bunt single, lead runner
# forced, popped up, double play, struck out on a foul bunt.
BUNT_ODDS = {
    "BUNT:SH": 0.64,
    "BUNT:1B": 0.07,
    "BUNT:FC": 0.13,
    "BUNT:PO": 0.06,
    "BUNT:DP": 0.02,
    "K": 0.08,
}
HIT_BASES = {"1B": 1, "2B": 2, "3B": 3}
# A throwing error on a single or double lets the batter take one more base (the official
# scorer's error, but the batter doesn't reach on it). With the reach-on-error rate, about
# 0.50 errors per team-game, the 2025 MLB level. Flagged for Milestone 8.
THROWING_ERROR = 0.02
# Retrosheet 2023 to 2025: the batter reaches on an error on this share of balls in play that
# aren't hits, by type (grounders are most of them). On 2.1% of ground-ball outs with a runner
# on (5.5% of those that aren't double plays) a throw gets away and the lead runner takes a
# base. Defense.error_rate scales the first relative to the league's DEFAULT_ERROR_RATE.
# Not modeled: errors on steals and pickoffs (0.065 per team-game) and on fly balls (0.03).
ERROR_ON_BIP = {"GB": 0.0238, "LD": 0.0037, "FB": 0.0023, "PU": 0.0026}
ERROR_ON_GB_OUT = 0.055
DEFAULT_ERROR_RATE = Defense().error_rate
# Pitcher limits that no card setting can lift.
MAX_PITCHES = 120
NO_HITTER_PITCHES = 130  # a starter still working a no-hitter
MOP_UP_RUNS = 7  # a reliever is pulled after allowing this many
# Base stealing (_maybe_steal). A runner's attempt rate per time on first
# becomes a chance before each PA he's on first with second open; tuned so a
# league of average runners tries about as often as MLB in 2024-25 (0.96 a
# team game, about one in ten of them for third).
STEAL_PER_PA = 1.12
STEAL_THIRD = 0.6  # relative chance of trying for third from second
# Success, on the log-odds scale: per unit of catcher arm (about 5 points of
# success rate), the pitcher's hand holding the runner at first, the
# easier steal of third, and per step of owner aggressiveness above normal.
STEAL_CATCHER = 0.3
STEAL_LEFTY, STEAL_RIGHTY = -0.2, 0.07  # about even over MLB's mix of hands
STEAL_THIRD_EASIER = 0.3
STEAL_AGGRESSION = 0.3
# A runner without a real record: the league's, scaled by sprint speed so a
# league of runners averages out to the league rate.
SPEED_ATTEMPT, SPEED_SUCCESS = 0.9, 0.25
# Taking the extra base (_send): how much a runner's sprint speed (z-score)
# and the outfield arms move the odds he's safe, and how much riskier the
# extra sends of an aggressive manager are, on the log-odds scale.
SEND_SPEED, SEND_ARM, SEND_AGGRESSION = 0.5, 0.4, 0.6


@dataclass(frozen=True)
class Hitter:
    id: str
    name: str
    bats: str  # L, R or S
    vs_lhp: Rates
    vs_rhp: Rates
    speed: float = 0.0  # roughly sprint-speed z-score, -2..+2
    # Fielding value (runs per 150 games vs the average there) at each position
    # he's experienced at; empty = average anywhere. See fielding.py.
    positions: dict[str, float] = field(default_factory=dict, hash=False, compare=False)
    # Throwing arm, about -2..+2: pop time for a catcher, arm strength for an
    # outfielder (talent/fielding.py). Holds runners in fielding.team_defense.
    arm: float = 0.0
    # Base stealing from his real MLB record (talent/running.py): attempts per
    # time on first and the share he makes. None = the league's, adjusted for
    # his sprint speed.
    steal_attempt: float | None = None
    steal_success: float | None = None

    def side_vs(self, throws: str) -> str:
        if self.bats == "S":
            return "L" if throws == "R" else "R"
        return self.bats

    def rates_vs(self, throws: str) -> Rates:
        return self.vs_lhp if throws == "L" else self.vs_rhp


@dataclass(frozen=True)
class Pitcher:
    id: str
    name: str
    throws: str
    vs_lhb: Rates
    vs_rhb: Rates
    pitch_limit: int = 95
    # (Statcast pitch type, share, average mph) from the talent data; only the
    # Gameday at-bat pop-up reads it (engine/pitchseq.py).
    arsenal: tuple[tuple[str, float, float], ...] = ()

    def rates_vs(self, side: str) -> Rates:
        return self.vs_lhb if side == "L" else self.vs_rhb


FIELD_POSITIONS = ("C", "1B", "2B", "3B", "SS", "LF", "CF", "RF")


@dataclass(frozen=True)
class DefensiveSub:
    """A late-game defensive replacement from the bench.

    At the top of each half inning the team takes the field from `from_inning`
    on (and only while leading, unless `only_when_leading` is off), the sub
    comes in at `position`, taking the batting-order slot of `replaces` (a
    starter's id) or, when that is None, of whoever is playing `position`.
    If `replaces` plays somewhere else, the player at `position` moves over to
    the replaced player's spot. Each sub is used at most once.
    """

    player: Hitter
    position: str
    from_inning: int = 9
    only_when_leading: bool = True
    replaces: str | None = None

    def __post_init__(self) -> None:
        if self.position not in FIELD_POSITIONS:
            raise ValueError(f"defensive sub needs a field position, not {self.position!r}")


@dataclass
class TeamCard:
    """What the engine needs from one team for one game."""

    name: str
    lineup: list[tuple[Hitter, str]]  # batting order: (hitter, position)
    starter: Pitcher
    bullpen: list[Pitcher] = field(default_factory=list)
    defense: Defense = field(default_factory=Defense)
    park: ParkFactors | None = None  # used when this team is home
    # Owner's opt-in alignment rules, checked in order; none = straight up all game.
    alignment_rules: tuple[AlignmentRule, ...] = ()
    abbr: str = ""  # short name for scores in the play log; defaults to the first 3 letters
    defensive_subs: tuple[DefensiveSub, ...] = ()
    # Bullpen designations by pitcher id. Empty = relievers enter in listed order.
    relief: dict[str, pen.ReliefUsage] = field(default_factory=dict)
    # Bullpen arms who follow in fixed order before anyone is picked by
    # leverage: the starter after an opener, a piggyback.
    follow: tuple[str, ...] = ()
    # When the starter (and the bulk arm after an opener) comes out, besides
    # his pitch limit: times through the order, runs allowed, leverage.
    hook: StarterHook = field(default_factory=StarterHook)
    # Outs each pitcher may still get this week (workload.py); none = no cap.
    outs_limit: dict[str, int] = field(default_factory=dict)
    # Bench bats available to pinch-hit, and the owner's rules for it (manager.py).
    bench: tuple[Hitter, ...] = ()
    pinch: PinchHitting = field(default_factory=PinchHitting)
    bunt: Bunting = field(default_factory=Bunting)
    # Stealing and taking the extra base: how aggressive, and who may steal.
    running: Baserunning = field(default_factory=Baserunning)
    # The starter is an opener (Dave, 2026-10-13): he never goes past one
    # inning, though he can still come out before finishing it.
    opener: bool = False

    @property
    def short(self) -> str:
        return self.abbr or self.name[:3].upper()


@dataclass
class Runner:
    hitter: Hitter
    pitcher_id: str  # pitcher charged if he scores
    earned: bool = True


@dataclass
class _Side:
    card: TeamCard
    slot: int = 0
    pitcher: Pitcher | None = None
    used: set[str] = field(default_factory=set)  # pitchers who have appeared
    pitcher_outs: int = 0  # outs recorded by the current pitcher
    pitches: int = 0
    batters_faced: int = 0
    hits_allowed: int = (
        0  # by the whole staff this game; a starter with none is working a no-hitter
    )
    runs: int = 0
    pitchers_used: list[str] = field(default_factory=list)
    lineup: list[tuple[Hitter, str]] = field(default_factory=list)  # this game's, after subs
    subs_used: set[int] = field(default_factory=set)
    removed: set[str] = field(default_factory=set)  # players who left the game
    pinch_hitters: int = 0
    charged: dict[str, int] = field(default_factory=dict)  # runs charged per pitcher
    defense: Defense = field(default_factory=Defense)  # the card's, for who plays where now

    def set_defense(self) -> None:
        self.defense = fielding.team_defense(self.card.defense, self.lineup)


SPARE_ARMS = 4  # (AAA) pitchers a team can burn through after its whole staff is used


def _spares(card: TeamCard) -> list[Pitcher]:
    from npl_sim.engine.replacement import standin_pitcher  # replacement imports this module

    return [standin_pitcher(f"AAA-{card.short}-R{i}") for i in range(1, SPARE_ARMS + 1)]


class GameSim:
    def __init__(
        self, away: TeamCard, home: TeamCard, env: LeagueEnv, seed: int, audit: list | None = None
    ):
        self.env = env
        # When a list, each PA appends what went into it (npl_sim/audit.py);
        # recording draws no random numbers, so results are the same either way.
        self.audit = audit
        self.rng = random.Random(seed)
        # Play detail (fielders, counts, strikes) draws from its own stream and never changes
        # what happens in the game; see detail.py. Anything that feeds back (the pitches a
        # steal or wild pitch comes on) draws from self.rng.
        self.detail_rng = random.Random(seed * 7919 + 17)
        self.park = home.park or ParkFactors()
        self.sides = {"away": _Side(away), "home": _Side(home)}
        for side in self.sides.values():
            side.pitcher = side.card.starter
            side.lineup = list(side.card.lineup)
            side.set_defense()
            side.pitchers_used.append(side.card.starter.id)
        self.events: list[dict] = []
        self.inning = 1
        self.half = "top"
        self.outs = 0
        # Outs the defense would have made but for errors this half-inning (rule 9.16).
        self._phantom_outs = 0
        self.bases: list[Runner | None] = [None, None, None]
        self.line: dict[str, list[int]] = {"away": [], "home": []}
        self.alignment = "normal"  # the fielding team's alignment for the current PA
        self._homer = False  # every run counts on a walk-off home run
        self._ground_rule = False  # the last double bounced out of play
        self._throwing = False  # a throwing error let the batter take an extra base
        self._outs_on_bases: list[dict] = []  # runners thrown out on the last PA
        self._pa_pitches = 0  # pitches of the coming PA thrown on steals and wild pitches

    # -- helpers -------------------------------------------------------------
    def _during_pa(self, pitched: bool = True) -> dict:
        """Where a steal, wild pitch, passed ball or balk falls in the coming
        PA: the batter at the plate and the pitch it came on (a balk comes
        before the next pitch, so it adds none)."""
        if pitched:
            self._pa_pitches += 1 + (self.rng.random() < 0.4)
        batter = self._batting().lineup[self._batting().slot][0]
        return {"batter": batter.id, "pitch": self._pa_pitches}

    def _cut_short(self) -> None:
        """The PA ended before it finished (third out on the bases, or the
        winning run scoring): its pitches count for the pitcher, carried on
        the event that ended it."""
        if not self._pa_pitches:
            return
        n = self._pa_pitches
        self._fielding().pitches += n
        strikes = sum(self.detail_rng.random() < 0.55 for _ in range(n - 1))
        self.events[-1] |= {"pitches": n, "strikes": strikes}

    def _log(self, kind: str, **data) -> None:
        self.events.append({"type": kind, "inning": self.inning, "half": self.half, **data})

    def _batting(self) -> _Side:
        return self.sides["away" if self.half == "top" else "home"]

    def _fielding(self) -> _Side:
        return self.sides["home" if self.half == "top" else "away"]

    def _score(self, runner: Runner, scored: list) -> None:
        if self._walkoff() and not self._homer:
            return  # the game ended when the winning run scored
        self._batting().runs += 1
        charged = self._fielding().charged
        charged[runner.pitcher_id] = charged.get(runner.pitcher_id, 0) + 1
        # Rule 9.16: rebuild the inning with each error an out. A run scoring once that
        # inning would have had three outs is unearned, whoever scored it. (Simplified: an
        # error runner later retired on a force still counts as the one phantom out.)
        if self.outs + self._phantom_outs >= 3:
            runner.earned = False
        scored.append(
            {"runner": runner.hitter.id, "pitcher": runner.pitcher_id, "earned": runner.earned}
        )

    def _maybe_change_pitcher(self, mid_inning: bool) -> None:
        side = self._fielding()
        reason = self._should_pull(side, mid_inning)
        if reason is None:
            return
        new, detail_ = self._next_pitcher(side, mid_inning)
        if new is None:
            return
        old = side.pitcher
        on_base = sum(r is not None for r in self.bases) if mid_inning else 0
        if self.audit is not None:
            self.audit.append(
                {
                    "event": len(self.events),  # the pitching_change logged below
                    "pitches": side.pitches,
                    "pitch_limit": old.pitch_limit,
                    "batters_faced": side.batters_faced,
                    "outs_recorded": side.pitcher_outs,
                    "runs_charged": side.charged.get(old.id, 0),
                    "available": [
                        p.id for p in side.card.bullpen if p.id not in side.used and p.id != old.id
                    ],
                }
            )
        side.pitcher = new
        side.used.add(new.id)
        side.pitches = 0
        side.batters_faced = 0
        side.pitcher_outs = 0
        side.pitchers_used.append(side.pitcher.id)
        self._log(
            "pitching_change",
            team=side.card.name,
            out=old.id,
            into=side.pitcher.id,
            outs=self.outs if mid_inning else 0,
            on_base=on_base,
            reason=reason,
            **{"leverage": self._leverage(side, mid_inning), **detail_},
        )

    def _leverage(self, side: _Side, mid_inning: bool) -> float:
        """Leverage of the coming PA for the fielding `side`."""
        lead = side.runs - self._batting().runs
        bases, outs = (self.bases, self.outs) if mid_inning else ([None] * 3, 0)
        return pen.leverage(self.inning, self.half, lead, bases, outs)

    def _should_pull(self, side: _Side, mid_inning: bool) -> str | None:
        """Why the current pitcher comes out before the next PA, or None."""
        p, card = side.pitcher, side.card
        # Hard stops come before the three-batter minimum: nobody throws more
        # than 120 pitches (130 for a starter with a no-hitter) or sits through
        # a mop-up blowout of 7 runs (Dave, 2026-10-14).
        no_hitter = p.id == card.starter.id and side.hits_allowed == 0
        if side.pitches >= (NO_HITTER_PITCHES if no_hitter else MAX_PITCHES):
            return "pitch_count"
        if p.id != card.starter.id and side.charged.get(p.id, 0) >= MOP_UP_RUNS:
            return "mop_up"
        if mid_inning and side.batters_faced < 3:
            return None  # three-batter minimum: he faces three or finishes the inning
        if side.pitches >= p.pitch_limit + (15 if mid_inning else 0):
            return "pitch_count"
        if p.id in card.outs_limit and side.pitcher_outs >= card.outs_limit[p.id]:
            return "workload"
        if card.opener and p.id == card.starter.id and not mid_inning and side.pitcher_outs >= 3:
            return "opener"  # one inning, even a clean one
        if p.id == card.starter.id or p.id in card.follow:
            return card.hook.reason(
                side.batters_faced,
                side.charged.get(p.id, 0),
                self.inning,
                self._leverage(side, mid_inning) if card.hook.leverage is not None else None,
            )
        plan = card.relief.get(p.id)
        if mid_inning or plan is None:
            return None
        # Between innings: would he run past his stint if he went out again?
        return "stint" if side.pitcher_outs + 3 > plan.outs + 1 else None

    def _next_pitcher(self, side: _Side, mid_inning: bool):
        card = side.card
        available = [p for p in card.bullpen if p.id not in side.used and p.id != side.pitcher.id]
        for pid in card.follow:
            nxt = next((p for p in available if p.id == pid), None)
            if nxt is not None:
                return nxt, {}
        available = [p for p in available if p.id not in card.follow]
        if not available:
            # Out of arms: a (AAA) replacement takes over rather than leaving a
            # tired pitcher in.
            spare = next((p for p in _spares(card) if p.id not in side.used), None)
            return spare, {}
        if not card.relief:
            return available[0], {}  # no designations: listed order
        offense = self._batting()
        batter = offense.lineup[offense.slot][0] if offense.lineup else None
        lead = side.runs - offense.runs
        li = self._leverage(side, mid_inning)
        starter_out = side.pitcher.id == card.starter.id or side.pitcher.id in card.follow
        tier = pen.situation(self.inning, lead, li, starter_out)
        pick = pen.choose(available, card.relief, tier, mid_inning, batter)
        return pick, {"leverage": li, "situation": tier}

    # -- the game ------------------------------------------------------------
    def _setup(self, card: TeamCard) -> dict:
        """The owner's settings behind each manager decision, for the audit log."""
        return {
            "team": card.name,
            "pitch_limits": {p.id: p.pitch_limit for p in (card.starter, *card.bullpen)},
            "hook": asdict(card.hook),
            "follow": list(card.follow),
            "relief": {pid: {**asdict(u), "outs": u.outs} for pid, u in card.relief.items()},
            "outs_limit": dict(card.outs_limit),
            "pinch": asdict(card.pinch),
            "defensive_subs": [
                {
                    "player": s.player.id,
                    "position": s.position,
                    "from_inning": s.from_inning,
                    "only_when_leading": s.only_when_leading,
                    "replaces": s.replaces,
                }
                for s in card.defensive_subs
            ],
            "bunt": asdict(card.bunt),
            "running": asdict(card.running),
            "alignment_rules": [asdict(r) for r in card.alignment_rules],
        }

    def _start(self) -> None:
        def team(card: TeamCard) -> dict:
            pitchers = [card.starter, *card.bullpen, *_spares(card)]
            return {
                "name": card.name,
                "abbr": card.short,
                "lineup": [
                    {
                        "id": h.id,
                        "name": h.name,
                        "position": pos,
                        "bats": h.bats,
                        **(
                            {"out_of_position": True}
                            if fielding.out_of_position(h.positions, pos)
                            else {}
                        ),
                    }
                    for h, pos in card.lineup
                ],
                "pitchers": [
                    {
                        "id": p.id,
                        "name": p.name,
                        "throws": p.throws,
                        **({"arsenal": [list(a) for a in p.arsenal]} if p.arsenal else {}),
                    }
                    for p in pitchers
                ],
                "starter": card.starter.id,
                "bench": [
                    {"id": h.id, "name": h.name, "bats": h.bats}
                    for h in {
                        h.id: h for h in [s.player for s in card.defensive_subs] + list(card.bench)
                    }.values()
                ],
            }

        if self.audit is not None:
            self.audit.append(
                {"event": 0, "setup": {k: self._setup(v.card) for k, v in self.sides.items()}}
            )
        self._log(
            "game_start",
            away=team(self.sides["away"].card),
            home=team(self.sides["home"].card),
        )

    def _bases(self) -> list[str | None]:
        return [r.hitter.id if r else None for r in self.bases]

    def play(self) -> list[dict]:
        self._start()
        while True:
            self._half_inning()
            if self._game_over():
                break
            if self.half == "top":
                self.half = "bottom"
            else:
                self.half, self.inning = "top", self.inning + 1
        self._log(
            "final", away=self.sides["away"].runs, home=self.sides["home"].runs, line=self.line
        )
        return self.events

    def _game_over(self) -> bool:
        away, home = self.sides["away"].runs, self.sides["home"].runs
        if self.inning < 9:
            return False
        if self.half == "top":
            return home > away  # home leads after the top of the 9th or later
        return home != away

    def _half_inning(self) -> None:
        self.outs = 0
        self._phantom_outs = 0
        self.bases = [None, None, None]
        side = self._batting()
        start_runs = side.runs
        self._defensive_subs()
        self._maybe_change_pitcher(mid_inning=False)
        if self.inning > 9 and self.env.auto_runner:
            # The batter before this inning's leadoff man starts on second.
            prev = side.lineup[(side.slot - 1) % len(side.lineup)][0]
            self.bases[1] = Runner(prev, self._fielding().pitcher.id, earned=False)
            self._log("auto_runner", runner=prev.id)
        while self.outs < 3:
            if self._walkoff():
                break
            # Steals, wild pitches, passed balls and balks happen during the
            # coming PA, on its pitches (_pa_pitches counts the ones thrown so
            # far). A caught stealing for the third out or a walk-off wild
            # pitch ends it early; its pitches still count.
            self._pa_pitches = 0
            self._maybe_pinch_hit()
            self._maybe_steal()
            if self.outs >= 3:
                self._cut_short()
                break
            self._maybe_wild_pitch()
            if self._walkoff():
                self._cut_short()
                break
            self._plate_appearance()
            if self.outs < 3 and not self._walkoff():
                self._maybe_change_pitcher(mid_inning=True)
        self._log("half_end", left_on_base=sum(r is not None for r in self.bases))
        self.line["away" if self.half == "top" else "home"].append(side.runs - start_runs)

    def _defensive_subs(self) -> None:
        side = self._fielding()
        lead = side.runs - self._batting().runs
        for i, sub in enumerate(side.card.defensive_subs):
            if i in side.subs_used or self.inning < sub.from_inning:
                continue
            if sub.only_when_leading and lead <= 0:
                continue
            ids = [h.id for h, _ in side.lineup]
            if sub.player.id in ids or sub.player.id in side.removed:
                continue
            at_pos = next((j for j, (_, p) in enumerate(side.lineup) if p == sub.position), None)
            target = ids.index(sub.replaces) if sub.replaces in ids else None
            if sub.replaces is not None and target is None:
                continue  # that starter is already out of the game
            target = at_pos if target is None else target
            if target is None:
                continue
            out, out_pos = side.lineup[target]
            moved = None
            if at_pos is not None and at_pos != target:
                mover = side.lineup[at_pos][0]
                side.lineup[at_pos] = (mover, out_pos)
                moved = {"player": mover.id, "position": out_pos}
            side.lineup[target] = (sub.player, sub.position)
            side.subs_used.add(i)
            side.removed.add(out.id)
            side.set_defense()
            self._log(
                "defensive_sub",
                team=side.card.name,
                into=sub.player.id,
                out=out.id,
                position=sub.position,
                slot=target,
                moved=moved,
            )

    def _maybe_pinch_hit(self) -> None:
        """Send up a bench bat for the batter due up when the owner's rules
        allow it and one is expected to do clearly better here."""
        offense, defense = self._batting(), self._fielding()
        rule = offense.card.pinch
        if not rule.enabled or not offense.card.bench or self.inning < rule.from_inning:
            return
        if offense.pinch_hitters >= rule.max_per_game:
            return
        hitter, pos = offense.lineup[offense.slot]
        if hitter.id in rule.never_for:
            return
        li = self._leverage(defense, mid_inning=True)
        if li < rule.min_leverage:
            return
        in_game = {h.id for h, _ in offense.lineup}
        bench = [
            h for h in offense.card.bench if h.id not in in_game and h.id not in offense.removed
        ]
        pitcher = defense.pitcher

        def edge(h: Hitter) -> bool:
            return h.side_vs(pitcher.throws) != pitcher.throws

        def catches(h: Hitter) -> bool:
            return "C" in h.positions

        def ok(h: Hitter) -> bool:
            if rule.platoon_only and (edge(hitter) or not edge(h)):
                return False
            if not rule.out_of_position and fielding.out_of_position(h.positions, pos):
                return False
            # Never leave the team without a catcher.
            return pos != "C" or catches(h) or any(catches(o) for o in bench if o is not h)

        candidates = [h for h in bench if ok(h)]
        if not candidates:
            return
        now = self._expected_runs(hitter, pitcher, defense)
        best, value = max(
            ((h, self._expected_runs(h, pitcher, defense)) for h in candidates),
            key=lambda hv: hv[1],
        )
        if value - now < rule.min_gain:
            return
        slot = offense.slot
        offense.lineup[slot] = (best, pos)
        offense.removed.add(hitter.id)
        offense.pinch_hitters += 1
        offense.set_defense()
        self._log(
            "pinch_hitter",
            team=offense.card.name,
            into=best.id,
            out=hitter.id,
            slot=slot,
            position=pos,
            leverage=round(li, 2),
            gain=round(value - now, 4),
            outs=self.outs,
        )

    def _fatigue(self, defense: _Side) -> float:
        if not self.env.fatigue:
            return LEAGUE_FATIGUE
        p = defense.pitcher
        fatigue = max(0.0, (defense.pitches - 0.8 * p.pitch_limit) / (0.5 * p.pitch_limit))
        fatigue += 0.12 * max(0, defense.batters_faced // 9 - 1)
        return min(fatigue, 1.0)

    def _expected_runs(self, hitter: Hitter, pitcher: Pitcher, defense: _Side) -> float:
        side = hitter.side_vs(pitcher.throws)
        probs = pa_probabilities(
            hitter.rates_vs(pitcher.throws),
            pitcher.rates_vs(side),
            self.env,
            side,
            pitcher.throws,
            self.park,
            defense.defense,
            self._fatigue(defense),
        )
        return run_value(probs)

    def _walkoff(self) -> bool:
        return (
            self.half == "bottom"
            and self.inning >= 9
            and self.sides["home"].runs > self.sides["away"].runs
        )

    def _maybe_steal(self) -> None:
        """Before a pitch: a runner on first with second open may steal second,
        or one on second with first and third open may steal third (never
        with two out). How often comes from his real attempt rate times the
        owner's aggressiveness; whether he makes it from his real success rate,
        the catcher's arm (pop time and caught-stealing record), and the
        pitcher's hand holding him at first."""
        offense, defense = self._batting(), self._fielding()
        run, b = offense.card.running, self.bases
        if b[0] is not None and b[1] is None:
            runner, base = b[0], "2B"
        elif b[1] is not None and b[0] is None and b[2] is None and self.outs < 2:
            runner, base = b[1], "3B"
        else:
            return
        if run.steal <= 0 or runner.hitter.id in run.no_steal or not self._steal_spot(offense):
            return
        attempt, success = steal_odds(runner.hitter, self.env)
        chance = STEAL_PER_PA * attempt * run.steal * (STEAL_THIRD if base == "3B" else 1.0)
        if self.rng.random() >= chance:
            return
        logit = _logit(success) - STEAL_CATCHER * defense.defense.catcher_arm
        logit -= STEAL_AGGRESSION * (run.steal - 1)  # aggressive managers run in worse spots
        if base == "3B":
            logit += STEAL_THIRD_EASIER
        else:  # a lefty faces first and holds the runner; righties a little easier
            logit += STEAL_LEFTY if defense.pitcher.throws == "L" else STEAL_RIGHTY
        made = _sigmoid(logit)
        if self.rng.random() < made:
            if base == "2B":
                self.bases[0], self.bases[1] = None, runner
            else:
                self.bases[1], self.bases[2] = None, runner
            kind = "stolen_base"
        else:
            self.bases[0 if base == "2B" else 1] = None
            self.outs += 1
            defense.pitcher_outs += 1
            kind = "caught_stealing"
        self._log(
            kind,
            runner=runner.hitter.id,
            pitcher=defense.pitcher.id,
            **self._during_pa(),
            base=base,
            outs_after=self.outs,
            bases_after=self._bases(),
            odds={"attempt": round(chance, 4), "success": round(made, 4)},
        )

    def _steal_spot(self, offense: _Side) -> bool:
        """Baseball's custom: nobody runs with a big lead, or trailing by three
        or more late, when an out costs more than the base gains."""
        lead = offense.runs - self._fielding().runs
        return lead < 5 and not (self.inning >= 7 and lead <= -3)

    def _maybe_wild_pitch(self) -> None:
        """A wild pitch, passed ball or balk: every runner moves up a base."""
        if not any(self.bases):
            return
        defense = self._fielding()
        p = defense.pitcher
        walks = p.rates_vs("R").bb / self.env.split("R", p.throws).bb
        wild = WILD_PITCH * min(max(walks, 0.5), 2.0)
        r = self.rng.random()
        if r < wild:
            kind = "wild_pitch"
        elif r < wild + PASSED_BALL:
            kind = "passed_ball"
        elif r < wild + PASSED_BALL + BALK:
            kind = "balk"
        else:
            return
        b, scored = self.bases, []
        before = self._bases()
        moves = [
            {"runner": r.hitter.id, "start": base, "end": end}
            for r, base, end in zip(b, ("1B", "2B", "3B"), ("2B", "3B", "score"), strict=True)
            if r is not None
        ]
        if b[2] is not None:
            if kind == "passed_ball":
                b[2].earned = False  # a run that scores on a passed ball is unearned
            self._score(b[2], scored)
        self.bases = [None, b[0], b[1]]
        self._log(
            kind,
            pitcher=p.id,
            **self._during_pa(pitched=kind != "balk"),
            scored=scored,
            runners=moves,
            bases_before=before,
            bases_after=self._bases(),
            outs_after=self.outs,
        )

    def _bunt_odds(self, hitter: Hitter, pitcher: Pitcher, side: str, probs: dict, fatigue: float):
        """Outcome odds of a sacrifice bunt when the owner's rules call for one
        here, else None (swing away)."""
        offense, defense = self._batting(), self._fielding()
        rule, b = offense.card.bunt, self.bases
        if not rule.enabled or self.outs or b[2] or not (b[0] or b[1]):
            return None
        if self.inning < rule.from_inning:
            return None
        if not -rule.max_deficit <= offense.runs - defense.runs <= rule.max_lead:
            return None
        average = pa_probabilities(
            self.env.split(side, pitcher.throws),
            pitcher.rates_vs(side),
            self.env,
            side,
            pitcher.throws,
            self.park,
            defense.defense,
            fatigue,
            self.alignment,
        )
        if run_value(average) - run_value(probs) < rule.min_shortfall:
            return None
        if self.rng.random() >= rule.rate:
            return None
        odds = dict(BUNT_ODDS)
        hit = min(max(0.07 + 0.02 * hitter.speed, 0.02), 0.15)
        odds["BUNT:SH"] += odds["BUNT:1B"] - hit
        odds["BUNT:1B"] = hit
        if b[1]:  # no double play to turn with the lead runner going to third
            odds["BUNT:FC"] += odds.pop("BUNT:DP")
        return odds

    def _plate_appearance(self) -> None:
        offense, defense = self._batting(), self._fielding()
        hitter, pos = offense.lineup[offense.slot]
        offense.slot = (offense.slot + 1) % len(offense.lineup)
        pitcher = defense.pitcher
        side = hitter.side_vs(pitcher.throws)

        self.alignment = choose(
            defense.card.alignment_rules,
            self.inning,
            self.outs,
            self.bases,
            defense.runs - offense.runs,
        )
        fatigue = self._fatigue(defense)
        probs = pa_probabilities(
            hitter.rates_vs(pitcher.throws),
            pitcher.rates_vs(side),
            self.env,
            side,
            pitcher.throws,
            self.park,
            defense.defense,
            fatigue,
            self.alignment,
        )
        bunt = self._bunt_odds(hitter, pitcher, side, probs, fatigue)
        if bunt is not None:
            probs = bunt
        if self.audit is not None:
            self.audit.append(
                {
                    "event": len(self.events),  # index of the PA event logged below
                    "batter_rates": asdict(hitter.rates_vs(pitcher.throws)),
                    "pitcher_rates": asdict(pitcher.rates_vs(side)),
                    "league_rates": asdict(self.env.split(side, pitcher.throws)),
                    "park": {k: getattr(self.park, k)[side] for k in ("single", "xbh", "hr")},
                    "defense": asdict(defense.defense),
                    "fatigue": fatigue,
                    "pitches_before": defense.pitches,
                    "batters_faced": defense.batters_faced,
                    "probabilities": dict(probs),
                }
            )
        result = self.rng.choices(list(probs), weights=list(probs.values()))[0]
        kind = result.split(":")[0] if ":" in result else result
        mean = PITCHES["BUNT" if bunt else "BIP" if ":" in result else kind]
        pitches = max(1, round(mean + self.rng.uniform(-1.5, 1.5)))
        pitches = max(
            pitches, self._pa_pitches + 1
        )  # the PA's steals and wild pitches came on its pitches
        defense.pitches += pitches
        defense.batters_faced += 1

        outs_before = self.outs
        before = list(self.bases)
        scored: list[dict] = []
        self._homer = self._ground_rule = self._throwing = False
        self._outs_on_bases = []
        outcome = self._resolve(result, hitter, pitcher, scored)
        if outcome in HIT_BASES and scored and self._walkoff():
            outcome = self._walkoff_hit(outcome, hitter, before, scored)
        rbi = len(scored) if outcome not in ("E", "DP") else 0
        defense.pitcher_outs += self.outs - outs_before
        if outcome in HIT_BASES or outcome == "HR":
            defense.hits_allowed += 1
        bip = kind if ":" in result else None
        if bunt is not None and bip:
            bip = "PU" if outcome == "PO" else "GB"
        drng = self.detail_rng
        balls, strikes, strikes_thrown = detail.count(drng, "BIP" if bip else kind, pitches)
        extra = {"alignment": self.alignment} if self.alignment != "normal" else {}
        if bunt is not None:
            extra["bunt"] = True
        if self._ground_rule and outcome == "2B":
            extra["ground_rule"] = True
        if self._throwing:
            extra["throwing_error"] = True
        if self._outs_on_bases:
            extra["thrown_out"] = self._outs_on_bases
        if outcome == "K":
            extra["k_type"] = "bunt" if bunt is not None else detail.strikeout_kind(drng)
        elif outcome == "HR":
            extra["direction"] = detail.hr_direction(drng, side)
            extra["batted_ball"] = detail.hr_ball(drng)
        self._log(
            "pa",
            batter=hitter.id,
            pitcher=pitcher.id,
            team=offense.card.name,
            position=pos,
            result=outcome,
            bip=bip,
            fielder=(
                detail.bunt_fielder(drng, outcome)
                if bunt is not None
                else detail.fielder(drng, outcome, bip, side)
            ),
            pitches=pitches,
            count=[balls, strikes],
            strikes=strikes_thrown,
            outs=self.outs - outs_before,
            outs_before=outs_before,
            scored=scored,
            rbi=rbi,
            runners=self._movements(hitter, before, scored),
            bases_before=[r.hitter.id if r else None for r in before],
            bases_after=self._bases(),
            **extra,
        )

    def _movements(self, hitter: Hitter, before: list, scored: list) -> list[dict]:
        """Where the batter and each runner started and ended on the play."""
        after = {r.hitter.id: i for i, r in enumerate(self.bases) if r}
        home = {s["runner"] for s in scored}
        bases = ("1B", "2B", "3B")
        moves = []
        for start, r in [("B", hitter), *((bases[i], r.hitter) for i, r in enumerate(before) if r)]:
            if r.id in home:
                end = "score"
            elif r.id in after:
                end = bases[after[r.id]]
            elif self._walkoff():
                continue  # the game ended before this runner got home; he isn't out
            else:
                end = "out"
            move = {"runner": r.id, "start": start, "end": end}
            at = next((o["base"] for o in self._outs_on_bases if o["runner"] == r.id), None)
            if end == "out" and at:
                move["out_at"] = at
            moves.append(move)
        return moves

    def _walkoff_hit(self, outcome: str, hitter: Hitter, before: list, scored: list) -> str:
        """A game-ending hit counts for only as many bases as the winning run
        advanced (rule 9.06(f)): a double that scores the winner from third is
        a single."""
        winner = scored[-1]["runner"]
        start = next(i for i, r in enumerate(before) if r is not None and r.hitter.id == winner)
        bases = min(HIT_BASES[outcome], 3 - start)
        if bases == HIT_BASES[outcome]:
            return outcome
        at = next(i for i, r in enumerate(self.bases) if r is not None and r.hitter.id == hitter.id)
        runner, self.bases[at] = self.bases[at], None
        self.bases[bases - 1] = runner
        return {1: "1B", 2: "2B", 3: "3B"}[bases]

    # -- baserunning ---------------------------------------------------------
    def _force(self, batter: Runner, scored: list) -> None:
        """Walk/HBP: batter to first, forced runners move up one."""
        b = self.bases
        if b[0] is not None:
            if b[1] is not None:
                if b[2] is not None:
                    self._score(b[2], scored)
                b[2] = b[1]
            b[1] = b[0]
        b[0] = batter

    def _resolve(self, result: str, hitter: Hitter, pitcher: Pitcher, scored: list) -> str:
        rng, b = self.rng, self.bases
        arm = self._fielding().defense.of_arm
        batter = Runner(hitter, pitcher.id)
        if result == "K":
            self.outs += 1
            return "K"
        if result in ("BB", "HBP"):
            self._force(batter, scored)
            return result
        if result == "HR":
            self._homer = True
            for r in b:
                if r is not None:
                    self._score(r, scored)
            self._score(batter, scored)
            self.bases = [None, None, None]
            return "HR"

        bip, res = result.split(":")
        if bip == "BUNT":
            return self._bunt(res, batter)
        if res == "1B":
            if b[2]:
                self._score(b[2], scored)
            third = None
            if b[1]:
                two_out = 0.25 if self.outs == 2 else 0  # running on contact
                go = self._send(b[1], 0.62 + 0.08 * b[1].hitter.speed - 0.06 * arm + two_out, 0.95)
                if go:
                    self._score(b[1], scored)
                elif go is False:
                    self._thrown_out(b[1], "home")
                else:
                    third = b[1]
            second = None
            if b[0]:
                go = None
                if third is None and self.outs < 3:
                    go = self._send(b[0], 0.27 + 0.07 * b[0].hitter.speed - 0.05 * arm, 0.96)
                if go:
                    third = b[0]
                elif go is False:
                    self._thrown_out(b[0], "3B")
                else:
                    second = b[0]
            self.bases = [batter, second, third]
            self._throwing_error(batter, 0, "1B")
            return "1B"
        if res == "2B":
            # A ground-rule double gives every runner exactly two bases.
            self._ground_rule = bip in ("LD", "FB") and rng.random() < GROUND_RULE_DOUBLE
            for r in (b[2], b[1]):
                if r:
                    self._score(r, scored)
            third = None
            if b[0] and self._ground_rule:
                third = b[0]
            elif b[0]:
                two_out = 0.2 if self.outs == 2 else 0
                go = self._send(b[0], 0.42 + 0.10 * b[0].hitter.speed - 0.06 * arm + two_out, 0.92)
                if go:
                    self._score(b[0], scored)
                elif go is False:
                    self._thrown_out(b[0], "home")
                else:
                    third = b[0]
            self.bases = [None, batter, third]
            self._throwing_error(batter, 1, "2B")
            return "2B"
        if res == "3B":
            for r in b:
                if r:
                    self._score(r, scored)
            self.bases = [None, None, batter]
            return "3B"

        # A ball in play turned into an out, unless the fielder errs.
        error = ERROR_ON_BIP[bip] * self._fielding().defense.error_rate / DEFAULT_ERROR_RATE
        if rng.random() < error:
            batter.earned = False
            self._phantom_outs += 1
            self._force(batter, scored)
            return "E"
        if bip == "GB":
            runners = any(b)
            result = self._ground_out(batter, scored)
            if (
                result in ("GO", "FC")
                and runners
                and self.outs < 3
                and rng.random() < ERROR_ON_GB_OUT
            ):
                self._runner_error(scored)
            return result
        self.outs += 1
        if bip == "FB" and self.outs < 3:
            go = (
                self._send(b[2], 0.55 + 0.08 * b[2].hitter.speed - 0.06 * arm, 0.96)
                if b[2]
                else None
            )
            if go:
                self._score(b[2], scored)
                b[2] = None
                if b[1] and rng.random() < 0.25:
                    b[2], b[1] = b[1], None
                return "SF"
            if go is False:  # tagged and thrown out at the plate: a double play
                self._thrown_out(b[2], "home")
                b[2] = None
            elif b[1] and not b[2]:
                go = self._send(b[1], 0.22, 0.97)
                if go:
                    b[2], b[1] = b[1], None
                elif go is False:
                    self._thrown_out(b[1], "3B")
                    b[1] = None
        return {"FB": "FO", "LD": "LO", "PU": "PO"}[bip]

    def _throwing_error(self, batter: Runner, at: int, result: str) -> None:
        """A throw on the hit gets away and the batter takes one more base (not on a
        walk-off). His run is unearned: without the error he stops where the hit put him."""
        if self._walkoff() or self.bases[at + 1] is not None or self.outs >= 3:
            return
        if self.rng.random() < THROWING_ERROR:
            self.bases[at], self.bases[at + 1] = None, batter
            batter.earned = False
            self._throwing = True

    def _runner_error(self, scored: list) -> None:
        """A throw on a ground-ball out gets away: the lead runner takes one more base. His
        run, if it scores, is unearned."""
        b = self.bases
        for at in (2, 1, 0):
            runner = b[at]
            if runner is None:
                continue
            if at == 2:
                runner.earned = False
                self._score(runner, scored)
                b[2] = None
            elif b[at + 1] is None:
                b[at], b[at + 1] = None, runner
                runner.earned = False
            else:
                continue
            self._throwing = True
            return

    def _send(self, runner: Runner, take: float, safe: float) -> bool | None:
        """Whether a runner goes for the extra base on a hit or a fly, and how
        it ends: None he holds, True he makes it, False he's thrown out.

        `take` is how often a runner makes it with a normal manager (the old
        MLB-calibrated rate, already moved by his speed and the outfield arms)
        and `safe` how often a sent runner of average speed is safe. The owner's
        extra-base aggressiveness sends him more or less often, and the roll for
        safe is weighted by his sprint speed against the arms; an aggressive
        manager's extra sends are closer plays."""
        if self._walkoff() or self.outs >= 3:
            return None
        g = self._batting().card.running.extra_base
        if g <= 0:
            return None
        go = 1 - (1 - min(max(take / safe, 0.0), 0.98)) ** g
        if self.rng.random() >= go:
            return None
        arm = self._fielding().defense.of_arm
        odds = _logit(safe) + SEND_SPEED * runner.hitter.speed - SEND_ARM * arm
        return self.rng.random() < _sigmoid(odds - SEND_AGGRESSION * (g - 1))

    def _thrown_out(self, runner: Runner, base: str) -> None:
        self.outs += 1
        self._outs_on_bases.append({"runner": runner.hitter.id, "base": base})

    def _bunt(self, res: str, batter: Runner) -> str:
        """A sacrifice-bunt attempt with nobody out, a runner on first or
        second and third base open (see _bunt_odds)."""
        b = self.bases
        if res == "SH":  # batter out at first, the runners move up
            self.outs += 1
            b[2], b[1], b[0] = b[1], b[0], None
        elif res == "1B":
            b[2], b[1], b[0] = b[1], b[0], batter
        elif res == "FC":  # the lead runner is forced out, the batter safe at first
            self.outs += 1
            if b[1]:
                b[1] = b[0]
            b[0] = batter
        elif res == "DP":  # only with a runner on first alone
            self.outs += 2
            b[0] = None
        else:  # popped up: runners hold
            self.outs += 1
        return res

    def _ground_out(self, batter: Runner, scored: list) -> str:
        rng, b = self.rng, self.bases
        play = GROUND_BALL_PLAY.get(self.alignment, {})
        if b[0] and self.outs < 2:
            dp = (0.42 - 0.06 * batter.hitter.speed) * play.get("double_play", 1.0)
            if rng.random() < dp:
                self.outs += 2
                b[0] = None
                if self.outs < 3:
                    if b[2]:
                        self._score(b[2], scored)
                    b[2], b[1] = b[1], None
                return "DP"
            if rng.random() < 0.35:  # fielder's choice: lead runner out at second
                self.outs += 1
                if self.outs < 3:
                    if b[2] and rng.random() < min(0.5 * play.get("score_from_third", 1.0), 0.95):
                        self._score(b[2], scored)
                        b[2] = None
                    if b[1] and b[2] is None:
                        b[2], b[1] = b[1], None
                b[0] = batter
                return "FC"
        self.outs += 1
        if self.outs < 3:
            # The out is made at first, so runners forced on contact (a
            # runner on first, and any runners in a row ahead of him) are
            # already moving and take the next base; the others may try.
            on_second = b[1] is not None and b[0] is not None
            on_third = b[2] is not None and on_second
            home = (
                (0.48 + 0.06 * b[2].hitter.speed) * play.get("score_from_third", 1.0) if b[2] else 0
            )
            if b[2] and (on_third or rng.random() < min(home, 0.95)):
                self._score(b[2], scored)
                b[2] = None
            if b[1] and b[2] is None and (on_second or rng.random() < 0.55):
                b[2], b[1] = b[1], None
            if b[0] and b[1] is None:
                b[1], b[0] = b[0], None
        return "GO"


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def steal_odds(h: Hitter, env: LeagueEnv) -> tuple[float, float]:
    """(attempts per time on first, success rate) for a runner: his real
    record when the talent data has one, else the league's for his speed."""
    attempt = h.steal_attempt
    if attempt is None:
        attempt = env.steal_attempt * math.exp(SPEED_ATTEMPT * h.speed - SPEED_ATTEMPT**2 / 2)
    success = h.steal_success
    if success is None:
        success = _sigmoid(_logit(env.steal_success) + SPEED_SUCCESS * h.speed)
    return attempt, success


def simulate_game(
    away: TeamCard, home: TeamCard, env: LeagueEnv, seed: int, audit: list | None = None
) -> list[dict]:
    """Play one game and return its event log. Pass a list as `audit` to also
    collect each PA's inputs and outcome odds (see npl_sim/audit.py)."""
    return GameSim(away, home, env, seed, audit).play()
