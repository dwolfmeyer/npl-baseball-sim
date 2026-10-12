"""Average teams built from league rates, for tests and demos."""

from __future__ import annotations

from dataclasses import replace

from npl_sim.engine.defaults import average_split
from npl_sim.engine.game import Hitter, Pitcher, TeamCard
from npl_sim.engine.model import Rates

POSITIONS = ("C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH")


def hitter(pid: str, bats: str = "R", speed: float = 0.0, scale: dict | None = None) -> Hitter:
    def card(throws: str) -> Rates:
        side = "L" if bats == "S" and throws == "R" else ("R" if bats == "S" else bats)
        return replace(average_split(side, throws), **(scale or {}))

    return Hitter(pid, pid, bats, vs_lhp=card("L"), vs_rhp=card("R"), speed=speed)


def pitcher(pid: str, throws: str = "R", limit: int = 95, scale: dict | None = None) -> Pitcher:
    def card(side: str) -> Rates:
        return replace(average_split(side, throws), **(scale or {}))

    return Pitcher(pid, pid, throws, vs_lhb=card("L"), vs_rhb=card("R"), pitch_limit=limit)


def average_team(name: str, hands: str = "RLRLRRLRS") -> TeamCard:
    lineup = [
        (hitter(f"{name}-{pos}", bats=h), pos) for pos, h in zip(POSITIONS, hands, strict=True)
    ]
    bullpen = [pitcher(f"{name}-RP{i}", "R" if i % 3 else "L", limit=25) for i in range(1, 8)]
    return TeamCard(name=name, lineup=lineup, starter=pitcher(f"{name}-SP"), bullpen=bullpen)
