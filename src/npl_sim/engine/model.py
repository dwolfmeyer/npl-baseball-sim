"""Plate-appearance outcome model.

A PA resolves in two stages:
1. True outcome: K, BB, HBP, HR or ball in play (BIP).
2. For a BIP: batted-ball type (GB, LD, FB, PU), then hit or out given the
   fielders behind the pitcher.

Batter and pitcher rates are combined with the multinomial odds-ratio
(log5) method against the league rate for the same handedness matchup.
Every context adjustment is applied exactly once, in `pa_probabilities`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TRUE_OUTCOMES = ("K", "BB", "HBP", "HR", "BIP")
BIP_TYPES = ("GB", "LD", "FB", "PU")
HIT_TYPES = ("1B", "2B", "3B")


# Pitcher workload f (game.GameSim._fatigue): batters faced / 18 plus half the share of the
# late-pitch tail past 80% of the pitch limit. It rises from the first batter, as the
# times-through-the-order studies find (Brill 2023: a steady decline per batter, not a cliff
# at the third look). Season rates already include tired arms at the role's usual workload,
# so a PA moves only by how far f is from `typical` (starters ~0.71, relievers ~0.25, see
# game.STARTER_WORKLOAD). LEAGUE_FATIGUE is the all-PA mean, the default when no role is given.
# Effect per unit of f on each outcome. The slopes are fits to 2023-25 starter splits
# (K -1.1%, HR +0.7% per batter faced) and are flagged for Milestone 8.
WORKLOAD_EFFECTS = {"K": -0.20, "HR": 0.15, "BB": 0.05}
WORKLOAD_HIT_EFFECT = 0.02  # on the share of balls in play that fall in
LEAGUE_FATIGUE = 0.56


@dataclass(frozen=True)
class Rates:
    """Per-PA rates for one side of a handedness split."""

    k: float
    bb: float
    hbp: float
    hr: float
    # Share of balls in play by type; sums to 1.
    bip: dict[str, float] = field(default_factory=dict)
    # Multiplier on hit probability per BIP (BABIP skill / contact quality).
    hit: float = 1.0
    # Multiplier on the extra-base share of hits.
    power: float = 1.0

    def true_outcomes(self) -> dict[str, float]:
        bip = 1.0 - self.k - self.bb - self.hbp - self.hr
        if bip <= 0:
            raise ValueError(f"rates leave no balls in play: {self}")
        return {"K": self.k, "BB": self.bb, "HBP": self.hbp, "HR": self.hr, "BIP": bip}


@dataclass(frozen=True)
class LeagueEnv:
    """League-wide rates the sim is anchored to (current MLB season).

    `rates` is keyed by (batter hand, pitcher hand) so platoon effects come
    from the league split as well as each player's own split.
    Hit tables give P(1B/2B/3B) for each BIP type against average defense in
    a neutral park.
    """

    rates: dict[tuple[str, str], Rates]
    hit_table: dict[str, dict[str, float]]
    # Extra-inning automatic runner and other rules are on by default.
    auto_runner: bool = True
    # False plays every pitcher at the league-average fatigue (no effect), to check
    # that a league of average teams reproduces these rates.
    fatigue: bool = True
    # Base stealing in today's game: attempts per time a runner reaches first
    # (singles, walks, hit by pitch) and the share that succeed. These are
    # 2024-25 MLB levels; talent/running.py measures them from the season's
    # box scores when they're at hand.
    steal_attempt: float = 0.105
    steal_success: float = 0.775

    def split(self, bats: str, throws: str) -> Rates:
        return self.rates[(bats, throws)]


def _mix(rates: Rates, outcomes: tuple[str, ...]) -> dict[str, float]:
    return rates.true_outcomes() if outcomes == TRUE_OUTCOMES else dict(rates.bip)


def log5(batter: dict[str, float], pitcher: dict[str, float], league: dict[str, float]):
    """Multinomial odds-ratio combine of two distributions against the league's."""
    scores = {o: batter[o] * pitcher[o] / league[o] for o in league if league[o] > 0}
    total = sum(scores.values())
    return {o: s / total for o, s in scores.items()}


@dataclass(frozen=True)
class ParkFactors:
    """Multipliers by outcome and batter hand ('L'/'R'), 1.0 = neutral.

    Factors are relative to league average and applied once per PA.
    Owner-set custom parks use the same shape (see `custom_park`).
    """

    single: dict[str, float] = field(default_factory=lambda: {"L": 1.0, "R": 1.0})
    xbh: dict[str, float] = field(default_factory=lambda: {"L": 1.0, "R": 1.0})
    hr: dict[str, float] = field(default_factory=lambda: {"L": 1.0, "R": 1.0})


NEUTRAL_PARK = ParkFactors()

CUSTOM_PARK_FACTOR_CAP = 0.10
CUSTOM_PARK_RUN_CAP = 0.05
# Approximate run value of each factor relative to the others, for the
# overall run-impact cap (HR moves scoring most).
_RUN_WEIGHTS = {"single": 0.45, "xbh": 0.20, "hr": 0.35}


def custom_park(single: dict[str, float], xbh: dict[str, float], hr: dict[str, float]):
    """An owner-set park, checked against the caps Dave asked for."""
    park = ParkFactors(single=single, xbh=xbh, hr=hr)
    for name in ("single", "xbh", "hr"):
        for hand, f in getattr(park, name).items():
            if abs(f - 1.0) > CUSTOM_PARK_FACTOR_CAP + 1e-9:
                raise ValueError(f"{name} factor for {hand}HB {f} exceeds +/-10%")
    for hand in ("L", "R"):
        impact = sum(w * (getattr(park, n)[hand] - 1.0) for n, w in _RUN_WEIGHTS.items())
        if abs(impact) > CUSTOM_PARK_RUN_CAP + 1e-9:
            raise ValueError(f"run impact for {hand}HB {impact:+.3f} exceeds +/-5%")
    return park


@dataclass(frozen=True)
class Defense:
    """The fielders behind a pitcher, reduced to what the PA model needs.

    Multipliers on hit probability by BIP type (below 1 = better defense),
    catcher framing on K/BB, and the error rate per BIP.
    """

    hit_mult: dict[str, float] = field(
        default_factory=lambda: {"GB": 1.0, "LD": 1.0, "FB": 1.0, "PU": 1.0}
    )
    framing: float = 0.0  # +0.02 = 2% more strikeouts, 2% fewer walks
    error_rate: float = 0.02
    of_arm: float = 0.0  # >0 holds runners from taking extra bases
    catcher_arm: float = 0.0  # >0 throws out more base stealers


# Owner alignment presets: per BIP type and hit type multipliers on the league hit table.
# infield_in and double_play move every grounder hit alike. The guard_lines and no_doubles
# values are estimates, not measurements: the public studies give only the net cost (the THT
# "Guarding the lines" study: about 0.5 win-expectancy points) and straight-up grounder rates
# (75% outs, 22% singles, 2% doubles). They aim at about -5% and -10% extra-base hits.
# Flagged for Milestone 8.
_ALL = ("1B", "2B", "3B")
ALIGNMENTS: dict[str, dict[str, dict[str, float]]] = {
    "normal": {},
    "infield_in": {"GB": dict.fromkeys(_ALL, 1.12)},  # more grounders sneak through
    "double_play": {"GB": dict.fromkeys(_ALL, 1.03)},
    "guard_lines": {
        "GB": {"1B": 1.025, "2B": 0.80, "3B": 0.80},
        "LD": {"1B": 1.01, "2B": 0.96, "3B": 1.0},
    },
    "no_doubles": {  # outfield deep: fewer extra-base hits, more singles
        "FB": {"1B": 1.20, "2B": 0.88, "3B": 0.80},
        "LD": {"1B": 1.04, "2B": 0.90, "3B": 0.85},
    },
}


def pa_probabilities(
    batter: Rates,
    pitcher: Rates,
    env: LeagueEnv,
    bats: str,
    throws: str,
    park: ParkFactors,
    defense: Defense,
    fatigue: float = LEAGUE_FATIGUE,
    alignment: str = "normal",
    typical: float = LEAGUE_FATIGUE,
) -> dict[str, float]:
    """Final outcome probabilities for one PA, every adjustment applied once.

    `batter` and `pitcher` are already the split for this matchup (batter vs
    the pitcher's hand, pitcher vs the batter's side). Returns probabilities
    over K, BB, HBP, HR and BIP results keyed "<type>:<result>", e.g. "GB:1B".
    """
    league = env.split(bats, throws)
    # Platoon-aware combine against the league rate for this hand matchup.
    true = log5(batter.true_outcomes(), pitcher.true_outcomes(), league.true_outcomes())
    # Workload: a pitcher deep into his outing misses fewer bats and gives up more HR and
    # hits, a fresh one the opposite, relative to his role's typical workload (see above).
    for outcome, effect in WORKLOAD_EFFECTS.items():
        true[outcome] *= (1 + effect * fatigue) / (1 + effect * typical)
    # Catcher framing.
    true["K"] *= 1 + defense.framing
    true["BB"] *= 1 - defense.framing
    # Park: HR here, hits below. This function is the only place park enters.
    true["HR"] *= park.hr[bats]
    total = sum(true.values())
    true = {o: p / total for o, p in true.items()}

    bip_mix = log5(batter.bip, pitcher.bip, league.bip)
    hit_skill = batter.hit * pitcher.hit / league.hit
    hit_skill *= (1 + WORKLOAD_HIT_EFFECT * fatigue) / (1 + WORKLOAD_HIT_EFFECT * typical)
    power = batter.power * pitcher.power / league.power
    align = ALIGNMENTS[alignment]

    out = {o: true[o] for o in ("K", "BB", "HBP", "HR")}
    for bip_type, share in bip_mix.items():
        table = env.hit_table[bip_type]
        # Defense and alignment move hit probability on this BIP type.
        mult = defense.hit_mult.get(bip_type, 1.0)
        a1, a2, a3 = (align.get(bip_type, {}).get(h, 1.0) for h in _ALL)
        # Hit skill, defense and alignment set how many balls fall in. Power then splits
        # those hits between singles and extra bases (talent/blend.py builds it as the
        # XBH index over the hit index, so a power hitter has the same BABIP as before
        # and more of his hits go for extra bases). The parks' factors are levels on
        # singles and on doubles and triples, as FanGraphs and Statcast count them.
        t1, t2, t3 = table["1B"] * a1, table["2B"] * a2, table["3B"] * a3
        base = t1 + t2 + t3
        hits = base * hit_skill * mult
        xbh_table = t2 + t3
        xbh = min(1.0, xbh_table / base * power) if base else 0.0
        p1 = hits * (1 - xbh) * park.single[bats]
        p2 = hits * xbh * (t2 / xbh_table if xbh_table else 0.0) * park.xbh[bats]
        p3 = hits * xbh * (t3 / xbh_table if xbh_table else 0.0) * park.xbh[bats]
        p_hit = p1 + p2 + p3
        if p_hit > 0.95:
            p1, p2, p3 = (p * 0.95 / p_hit for p in (p1, p2, p3))
            p_hit = 0.95
        bip_p = true["BIP"] * share
        out[f"{bip_type}:1B"] = bip_p * p1
        out[f"{bip_type}:2B"] = bip_p * p2
        out[f"{bip_type}:3B"] = bip_p * p3
        out[f"{bip_type}:OUT"] = bip_p * (1 - p_hit)
    return out
