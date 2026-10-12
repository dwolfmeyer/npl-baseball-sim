"""Simulate one game, plate appearance by plate appearance.

Milestone 5 scope: full PA model, baserunning, steals, double plays, sac
flies, errors, extra innings with the automatic runner, walk-offs, and a
starter hooks by pitch count, times through the order, runs allowed and
leverage (hooks.py), and relievers by the owner's bullpen designations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from npl_sim.engine import bullpen as pen
from npl_sim.engine import detail, fielding
from npl_sim.engine.alignment import GROUND_BALL_PLAY, AlignmentRule, choose
from npl_sim.engine.hooks import StarterHook
from npl_sim.engine.model import Defense, LeagueEnv, ParkFactors, Rates, pa_probabilities

# Mean pitches per PA by outcome.
PITCHES = {"K": 4.8, "BB": 5.7, "HBP": 3.6, "HR": 3.5, "BIP": 3.4}


@dataclass(frozen=True)
class Hitter:
    id: str
    name: str
    bats: str  # L, R or S
    vs_lhp: Rates
    vs_rhp: Rates
    speed: float = 0.0  # roughly sprint-speed z-score, -2..+2
    # Fielding value (runs per 600 BIP vs the average there) at each position
    # he's experienced at; empty = average anywhere. See fielding.py.
    positions: dict[str, float] = field(default_factory=dict, hash=False, compare=False)
    # Throwing arm, about -2..+2: pop time for a catcher, arm strength for an
    # outfielder (talent/fielding.py). Holds runners in fielding.team_defense.
    arm: float = 0.0

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
    steal_aggressiveness: float = 1.0  # 0 = never run, 1 = normal, 2 = aggressive
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
    runs: int = 0
    pitchers_used: list[str] = field(default_factory=list)
    lineup: list[tuple[Hitter, str]] = field(default_factory=list)  # this game's, after subs
    subs_used: set[int] = field(default_factory=set)
    removed: set[str] = field(default_factory=set)  # players who left the game
    charged: dict[str, int] = field(default_factory=dict)  # runs charged per pitcher
    defense: Defense = field(default_factory=Defense)  # the card's, for who plays where now

    def set_defense(self) -> None:
        self.defense = fielding.team_defense(self.card.defense, self.lineup)


class GameSim:
    def __init__(self, away: TeamCard, home: TeamCard, env: LeagueEnv, seed: int):
        self.env = env
        self.rng = random.Random(seed)
        # Play detail (fielders, counts) draws from its own stream; see detail.py.
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
        self.bases: list[Runner | None] = [None, None, None]
        self.line: dict[str, list[int]] = {"away": [], "home": []}
        self.alignment = "normal"  # the fielding team's alignment for the current PA

    # -- helpers -------------------------------------------------------------
    def _log(self, kind: str, **data) -> None:
        self.events.append({"type": kind, "inning": self.inning, "half": self.half, **data})

    def _batting(self) -> _Side:
        return self.sides["away" if self.half == "top" else "home"]

    def _fielding(self) -> _Side:
        return self.sides["home" if self.half == "top" else "away"]

    def _score(self, runner: Runner, scored: list) -> None:
        self._batting().runs += 1
        charged = self._fielding().charged
        charged[runner.pitcher_id] = charged.get(runner.pitcher_id, 0) + 1
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
        if side.pitches >= p.pitch_limit + (15 if mid_inning else 0):
            return "pitch_count"
        if p.id in card.outs_limit and side.pitcher_outs >= card.outs_limit[p.id]:
            return "workload"
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
            return None, {}
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
    def _start(self) -> None:
        def team(card: TeamCard) -> dict:
            pitchers = [card.starter, *card.bullpen]
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
                "pitchers": [{"id": p.id, "name": p.name, "throws": p.throws} for p in pitchers],
                "starter": card.starter.id,
                "bench": [
                    {"id": s.player.id, "name": s.player.name, "bats": s.player.bats}
                    for s in card.defensive_subs
                ],
            }

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
            self._maybe_steal()
            if self.outs >= 3:
                break
            self._plate_appearance()
            if self.outs < 3:
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

    def _walkoff(self) -> bool:
        return (
            self.half == "bottom"
            and self.inning >= 9
            and self.sides["home"].runs > self.sides["away"].runs
        )

    def _maybe_steal(self) -> None:
        runner = self.bases[0]
        if runner is None or self.bases[1] is not None:
            return
        offense, defense = self._batting(), self._fielding()
        attempt = 0.045 * offense.card.steal_aggressiveness * (1 + 0.8 * runner.hitter.speed)
        if attempt <= 0 or self.rng.random() >= attempt:
            return
        success = 0.78 + 0.06 * runner.hitter.speed - 0.08 * defense.defense.catcher_arm
        if self.rng.random() < min(max(success, 0.4), 0.95):
            self.bases[0], self.bases[1] = None, runner
            kind = "stolen_base"
        else:
            self.bases[0] = None
            self.outs += 1
            defense.pitcher_outs += 1
            kind = "caught_stealing"
        self._log(
            kind,
            runner=runner.hitter.id,
            pitcher=defense.pitcher.id,
            base="2B",
            outs_after=self.outs,
            bases_after=self._bases(),
        )

    def _plate_appearance(self) -> None:
        offense, defense = self._batting(), self._fielding()
        hitter, pos = offense.lineup[offense.slot]
        offense.slot = (offense.slot + 1) % len(offense.lineup)
        pitcher = defense.pitcher
        side = hitter.side_vs(pitcher.throws)

        times_through = defense.batters_faced // 9
        fatigue = max(
            0.0, (defense.pitches - 0.8 * pitcher.pitch_limit) / (0.5 * pitcher.pitch_limit)
        )
        fatigue += 0.12 * max(0, times_through - 1)
        self.alignment = choose(
            defense.card.alignment_rules,
            self.inning,
            self.outs,
            self.bases,
            defense.runs - offense.runs,
        )
        probs = pa_probabilities(
            hitter.rates_vs(pitcher.throws),
            pitcher.rates_vs(side),
            self.env,
            side,
            pitcher.throws,
            self.park,
            defense.defense,
            min(fatigue, 1.0),
            self.alignment,
        )
        result = self.rng.choices(list(probs), weights=list(probs.values()))[0]
        kind = result.split(":")[0] if ":" in result else result
        mean = PITCHES["BIP" if ":" in result else kind]
        pitches = max(1, round(mean + self.rng.uniform(-1.5, 1.5)))
        defense.pitches += pitches
        defense.batters_faced += 1

        outs_before = self.outs
        before = list(self.bases)
        scored: list[dict] = []
        outcome = self._resolve(result, hitter, pitcher, scored)
        rbi = len(scored) if outcome not in ("E", "DP") else 0
        defense.pitcher_outs += self.outs - outs_before
        bip = kind if ":" in result else None
        drng = self.detail_rng
        balls, strikes, strikes_thrown = detail.count(drng, "BIP" if bip else kind, pitches)
        extra = {"alignment": self.alignment} if self.alignment != "normal" else {}
        if outcome == "K":
            extra["k_type"] = detail.strikeout_kind(drng)
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
            fielder=detail.fielder(drng, outcome, bip, side),
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
            else:
                end = "out"
            moves.append({"runner": r.id, "start": start, "end": end})
        return moves

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
            for r in b:
                if r is not None:
                    self._score(r, scored)
            self._score(batter, scored)
            self.bases = [None, None, None]
            return "HR"

        bip, res = result.split(":")
        if res == "1B":
            if b[2]:
                self._score(b[2], scored)
            third = None
            if b[1]:
                p = 0.62 + 0.08 * b[1].hitter.speed - 0.06 * arm + (0.25 if self.outs == 2 else 0)
                if rng.random() < p:
                    self._score(b[1], scored)
                else:
                    third = b[1]
            second = None
            if b[0]:
                p = 0.27 + 0.07 * b[0].hitter.speed - 0.05 * arm
                if third is None and rng.random() < p:
                    third = b[0]
                else:
                    second = b[0]
            self.bases = [batter, second, third]
            return "1B"
        if res == "2B":
            for r in (b[2], b[1]):
                if r:
                    self._score(r, scored)
            third = None
            if b[0]:
                p = 0.42 + 0.10 * b[0].hitter.speed - 0.06 * arm + (0.2 if self.outs == 2 else 0)
                if rng.random() < p:
                    self._score(b[0], scored)
                else:
                    third = b[0]
            self.bases = [None, batter, third]
            return "2B"
        if res == "3B":
            for r in b:
                if r:
                    self._score(r, scored)
            self.bases = [None, None, batter]
            return "3B"

        # A ball in play turned into an out, unless the fielder errs.
        if rng.random() < self._fielding().defense.error_rate:
            batter.earned = False
            self._force(batter, scored)
            return "E"
        if bip == "GB":
            return self._ground_out(batter, scored)
        self.outs += 1
        if bip == "FB" and self.outs < 3:
            if b[2] and rng.random() < 0.55 + 0.08 * b[2].hitter.speed - 0.06 * arm:
                self._score(b[2], scored)
                b[2] = None
                if b[1] and rng.random() < 0.25:
                    b[2], b[1] = b[1], None
                return "SF"
            if b[1] and not b[2] and rng.random() < 0.22:
                b[2], b[1] = b[1], None
        return {"FB": "FO", "LD": "LO", "PU": "PO"}[bip]

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
            home = (
                (0.48 + 0.06 * b[2].hitter.speed) * play.get("score_from_third", 1.0) if b[2] else 0
            )
            if b[2] and rng.random() < min(home, 0.95):
                self._score(b[2], scored)
                b[2] = None
            if b[1] and b[2] is None and rng.random() < 0.55:
                b[2], b[1] = b[1], None
            if b[0] and b[1] is None:
                b[1], b[0] = b[0], None
        return "GO"


def simulate_game(away: TeamCard, home: TeamCard, env: LeagueEnv, seed: int) -> list[dict]:
    """Play one game and return its event log."""
    return GameSim(away, home, env, seed).play()
