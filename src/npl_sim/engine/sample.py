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


def average_week_team(name: str):
    """An average team for a sim week: five starters, the pen, four bench bats."""
    from npl_sim.engine.workload import WeekTeam

    card = average_team(name)
    rotation = [pitcher(f"{name}-SP{i}", "R" if i % 2 else "L") for i in range(1, 6)]
    bench = [hitter(f"{name}-BN{i}", bats="L" if i % 2 else "R") for i in range(1, 5)]
    fields = {h.id: {pos} for h, pos in card.lineup if pos != "DH"}
    fields.update(
        {
            f"{name}-BN1": {"C"},
            f"{name}-BN2": {"1B", "2B", "3B", "SS"},
            f"{name}-BN3": {"LF", "CF", "RF"},
            f"{name}-BN4": {"1B", "LF", "RF"},
        }
    )
    return WeekTeam(card=card, lineups={}, rotation=rotation, bench=bench, fields=fields)


def busy_week(*teams) -> dict:
    """MLB usage for a typical full week: regulars ~26 PA, bench ~8, each
    starter one start (the first two two), relievers about 3 innings."""
    pa, starts, relief = {}, {}, {}
    for t in teams:
        for h, _ in t.card.lineup:
            pa[h.id] = 26
        for h in t.bench:
            pa[h.id] = 8
        for i, p in enumerate(t.rotation):
            starts[p.id] = 2 if i < 2 else 1
        for p in t.card.bullpen:
            relief[p.id] = 9
    return {"pa": pa, "starts": starts, "relief_outs": relief}
