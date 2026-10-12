"""Simulate one game, plate appearance by plate appearance.

Milestone 5 scope: full PA model, baserunning, steals, double plays, sac
flies, errors, extra innings with the automatic runner, walk-offs, and a
simple pitch-count hook with relievers in listed order. The owner's full
lineup card (leverage bullpen, hooks, pinch hitters, defensive subs) is
Milestone 6.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

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


@dataclass
class TeamCard:
    """What the engine needs from one team for one game."""

    name: str
    lineup: list[tuple[Hitter, str]]  # batting order: (hitter, position)
    starter: Pitcher
    bullpen: list[Pitcher] = field(default_factory=list)
    defense: Defense = field(default_factory=Defense)
    park: ParkFactors | None = None  # used when this team is home
    alignment: str = "normal"
    steal_aggressiveness: float = 1.0  # 0 = never run, 1 = normal, 2 = aggressive


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
    pen_index: int = 0
    pitches: int = 0
    batters_faced: int = 0
    runs: int = 0
    pitchers_used: list[str] = field(default_factory=list)


class GameSim:
    def __init__(self, away: TeamCard, home: TeamCard, env: LeagueEnv, seed: int):
        self.env = env
        self.rng = random.Random(seed)
        self.park = home.park or ParkFactors()
        self.sides = {"away": _Side(away), "home": _Side(home)}
        for side in self.sides.values():
            side.pitcher = side.card.starter
            side.pitchers_used.append(side.card.starter.id)
        self.events: list[dict] = []
        self.inning = 1
        self.half = "top"
        self.outs = 0
        self.bases: list[Runner | None] = [None, None, None]
        self.line: dict[str, list[int]] = {"away": [], "home": []}

    # -- helpers -------------------------------------------------------------
    def _log(self, kind: str, **data) -> None:
        self.events.append({"type": kind, "inning": self.inning, "half": self.half, **data})

    def _batting(self) -> _Side:
        return self.sides["away" if self.half == "top" else "home"]

    def _fielding(self) -> _Side:
        return self.sides["home" if self.half == "top" else "away"]

    def _score(self, runner: Runner, scored: list) -> None:
        self._batting().runs += 1
        scored.append(
            {"runner": runner.hitter.id, "pitcher": runner.pitcher_id, "earned": runner.earned}
        )

    def _maybe_change_pitcher(self, mid_inning: bool) -> None:
        side = self._fielding()
        limit = side.pitcher.pitch_limit
        over = side.pitches >= limit + (15 if mid_inning else 0)
        if not over or side.pen_index >= len(side.card.bullpen):
            return
        old = side.pitcher
        side.pitcher = side.card.bullpen[side.pen_index]
        side.pen_index += 1
        side.pitches = 0
        side.batters_faced = 0
        side.pitchers_used.append(side.pitcher.id)
        self._log("pitching_change", team=side.card.name, out=old.id, into=side.pitcher.id)

    # -- the game ------------------------------------------------------------
    def play(self) -> list[dict]:
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
        self._maybe_change_pitcher(mid_inning=False)
        if self.inning > 9 and self.env.auto_runner:
            # The batter before this inning's leadoff man starts on second.
            prev = side.card.lineup[(side.slot - 1) % len(side.card.lineup)][0]
            self.bases[1] = Runner(prev, self._fielding().pitcher.id, earned=False)
            self._log("auto_runner", runner=prev.id)
        while self.outs < 3:
            if self._walkoff():
                break
            self._maybe_steal()
            if self.outs >= 3:
                break
            self._plate_appearance()
            self._maybe_change_pitcher(mid_inning=True)
        self.line["away" if self.half == "top" else "home"].append(side.runs - start_runs)

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
        success = 0.78 + 0.06 * runner.hitter.speed - 0.08 * defense.card.defense.catcher_arm
        if self.rng.random() < min(max(success, 0.4), 0.95):
            self.bases[0], self.bases[1] = None, runner
            self._log("stolen_base", runner=runner.hitter.id, pitcher=defense.pitcher.id)
        else:
            self.bases[0] = None
            self.outs += 1
            self._log("caught_stealing", runner=runner.hitter.id, pitcher=defense.pitcher.id)

    def _plate_appearance(self) -> None:
        offense, defense = self._batting(), self._fielding()
        hitter, pos = offense.card.lineup[offense.slot]
        offense.slot = (offense.slot + 1) % len(offense.card.lineup)
        pitcher = defense.pitcher
        side = hitter.side_vs(pitcher.throws)

        times_through = defense.batters_faced // 9
        fatigue = max(
            0.0, (defense.pitches - 0.8 * pitcher.pitch_limit) / (0.5 * pitcher.pitch_limit)
        )
        fatigue += 0.12 * max(0, times_through - 1)
        probs = pa_probabilities(
            hitter.rates_vs(pitcher.throws),
            pitcher.rates_vs(side),
            self.env,
            side,
            pitcher.throws,
            self.park,
            defense.card.defense,
            min(fatigue, 1.0),
            defense.card.alignment,
        )
        result = self.rng.choices(list(probs), weights=list(probs.values()))[0]
        kind = result.split(":")[0] if ":" in result else result
        mean = PITCHES["BIP" if ":" in result else kind]
        pitches = max(1, round(mean + self.rng.uniform(-1.5, 1.5)))
        defense.pitches += pitches
        defense.batters_faced += 1

        outs_before = self.outs
        scored: list[dict] = []
        outcome = self._resolve(result, hitter, pitcher, scored)
        rbi = len(scored) if outcome not in ("E", "DP") else 0
        self._log(
            "pa",
            batter=hitter.id,
            pitcher=pitcher.id,
            team=offense.card.name,
            position=pos,
            result=outcome,
            bip=kind if ":" in result else None,
            pitches=pitches,
            outs=self.outs - outs_before,
            scored=scored,
            rbi=rbi,
        )

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
        arm = self._fielding().card.defense.of_arm
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
        if rng.random() < self._fielding().card.defense.error_rate:
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
        if b[0] and self.outs < 2:
            dp = 0.42 - 0.06 * batter.hitter.speed
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
                    if b[2] and rng.random() < 0.5:
                        self._score(b[2], scored)
                        b[2] = None
                    if b[1] and b[2] is None:
                        b[2], b[1] = b[1], None
                b[0] = batter
                return "FC"
        self.outs += 1
        if self.outs < 3:
            if b[2] and rng.random() < 0.48 + 0.06 * b[2].hitter.speed:
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
