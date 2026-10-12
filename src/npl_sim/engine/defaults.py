"""Placeholder league environment and average cards for tests and demos.

Rates approximate recent MLB seasons. Milestone 4 replaces these with the
current season's league rates; Milestone 8 calibrates the engine against
a replayed season.
"""

from __future__ import annotations

from npl_sim.engine.model import LeagueEnv, Rates

LEAGUE_BIP = {"GB": 0.44, "LD": 0.21, "FB": 0.25, "PU": 0.10}

LEAGUE_HIT_TABLE = {
    "GB": {"1B": 0.235, "2B": 0.020, "3B": 0.002},
    "LD": {"1B": 0.490, "2B": 0.160, "3B": 0.012},
    "FB": {"1B": 0.045, "2B": 0.075, "3B": 0.012},
    "PU": {"1B": 0.015, "2B": 0.000, "3B": 0.000},
}


def _rates(k, bb, hbp, hr):
    return Rates(k=k, bb=bb, hbp=hbp, hr=hr, bip=dict(LEAGUE_BIP))


# (batter hand, pitcher hand). Opposite-hand matchups favor the hitter.
LEAGUE_RATES = {
    ("R", "R"): _rates(0.232, 0.077, 0.012, 0.029),
    ("R", "L"): _rates(0.212, 0.088, 0.010, 0.032),
    ("L", "R"): _rates(0.214, 0.090, 0.010, 0.032),
    ("L", "L"): _rates(0.248, 0.078, 0.012, 0.025),
}

DEFAULT_ENV = LeagueEnv(rates=LEAGUE_RATES, hit_table=LEAGUE_HIT_TABLE)


def average_split(bats: str, throws: str) -> Rates:
    return DEFAULT_ENV.split(bats, throws)
