"""Owner defensive-alignment rules: opt-in checkboxes, straight up otherwise.

Per the league discussion (2026-10-12), alignment is not a standing setting.
Each owner may tick situational rules, such as "infield in with a runner on
third" or "double-play depth and concede the run", optionally limited to late
innings or a score range. With nothing ticked the team plays straight up all
game. Rules are checked in the owner's order before every pitch of a PA; the
first that fits the situation sets the alignment for that PA.
"""

from __future__ import annotations

from dataclasses import dataclass

# What a preset needs on the bases (and outs) to apply at all.
NEEDS = {
    "infield_in": "runner on third, fewer than two outs",
    "double_play": "runner on first, fewer than two outs",
    "guard_lines": "any",
    "no_doubles": "any",
}

# Effects on ground balls in play beyond hit probability (model.ALIGNMENTS):
# how often a runner on third scores on a grounder out, and how often a
# double play is turned.
GROUND_BALL_PLAY = {
    "infield_in": {"score_from_third": 0.35, "double_play": 0.7},
    "double_play": {"score_from_third": 1.3, "double_play": 1.25},
}


@dataclass(frozen=True)
class AlignmentRule:
    preset: str  # infield_in, double_play, guard_lines or no_doubles
    from_inning: int = 1
    # Score range, from the fielding team's side: lead of at least / at most.
    min_lead: int | None = None
    max_lead: int | None = None

    def __post_init__(self) -> None:
        if self.preset not in NEEDS:
            raise ValueError(f"unknown alignment {self.preset!r}")


def choose(
    rules: tuple[AlignmentRule, ...] | list[AlignmentRule],
    inning: int,
    outs: int,
    bases: list,
    lead: int,
) -> str:
    """The alignment for this PA: the first rule that fits, else 'normal'."""
    for rule in rules:
        if inning < rule.from_inning:
            continue
        if rule.min_lead is not None and lead < rule.min_lead:
            continue
        if rule.max_lead is not None and lead > rule.max_lead:
            continue
        if rule.preset == "infield_in" and not (bases[2] is not None and outs < 2):
            continue
        if rule.preset == "double_play" and not (bases[0] is not None and outs < 2):
            continue
        return rule.preset
    return "normal"
