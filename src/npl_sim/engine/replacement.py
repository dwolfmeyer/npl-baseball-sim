"""(AAA) stand-ins at Scoresheet's penalty levels.

When a hitter can't play and no bench player fits his position, a stand-in
takes the slot (spec, "Replacements"): OF/1B/DH hit about .185 with a .230
slugging percentage, catchers about .135/.180, other infielders about
.170/.200. A stand-in pitcher, used only when a team's whole pen is resting
or out of innings, allows about twice the league's runs. The rates were
tuned so the engine produces those lines against league-average players
(tests/test_workload.py checks them).
"""

from __future__ import annotations

from dataclasses import replace

from npl_sim.engine.defaults import average_split
from npl_sim.engine.game import Hitter, Pitcher

# Position group -> changes to league-average rates.
STANDIN = {
    "OF": {"k": 0.27, "bb": 0.05, "hr": 0.004, "hit": 0.96, "power": 0.6},
    "C": {"k": 0.33, "bb": 0.04, "hr": 0.003, "hit": 0.78, "power": 0.8},
    "IF": {"k": 0.28, "bb": 0.045, "hr": 0.002, "hit": 0.93, "power": 0.45},
}
GROUP = {"C": "C", "2B": "IF", "3B": "IF", "SS": "IF"}  # everything else: OF


def standin_hitter(pid: str, position: str) -> Hitter:
    scale = STANDIN[GROUP.get(position, "OF")]

    def card(throws: str):
        return replace(average_split("R", throws), **scale)

    glove = {} if position == "DH" else {position: 0.0}  # a stand-in plays his own spot
    return Hitter(
        pid, f"(AAA) {position}", "R", vs_lhp=card("L"), vs_rhp=card("R"), positions=glove
    )


# A (AAA) arm: about 2x the league's run rate.
STANDIN_PITCHER = {"k": 0.14, "bb": 0.13, "hr": 0.05, "hit": 1.15, "power": 1.2}


def standin_pitcher(pid: str) -> Pitcher:
    def card(side: str):
        return replace(average_split(side, "R"), **STANDIN_PITCHER)

    return Pitcher(pid, "(AAA) P", "R", vs_lhb=card("L"), vs_rhb=card("R"), pitch_limit=90)
