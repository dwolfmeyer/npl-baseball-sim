"""The manager's weekly lineup card: who is active, batting orders against
right- and left-handed starters, the rotation and bullpen roles.

This is the web app's first cut. It checks the card against the roster and
turns it into the engine's `TeamCard`. Until the weekly cards from the data
pipeline land, every player sims with league-average rates; the engine's own
lineup and bullpen logic (Milestone 6) replaces the simple reliever order here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from npl_sim.engine.game import TeamCard
from npl_sim.engine.sample import hitter, pitcher
from npl_sim.rosters.npl_app import NplPlayer

ACTIVE_LIMIT = 30
ROTATION_SIZE = 5
POSITIONS = ("C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH")
SIDES = {"rhp": "vs RHP", "lhp": "vs LHP"}
ROLES = {"CL": "Closer", "SU": "Setup", "MR": "Middle", "LR": "Long"}
# Relievers enter in this order when the starter tires (closer last).
ROLE_ORDER = ("LR", "MR", "SU", "CL")

# Positions each NPL roster position can play without a warning.
_ELIGIBLE = {
    "C": {"C"},
    "1B": {"1B"},
    "2B": {"2B"},
    "3B": {"3B", "1B"},
    "SS": {"SS", "2B", "3B"},
    "IF": {"1B", "2B", "3B", "SS"},
    "OF": {"LF", "CF", "RF"},
    "LF": {"LF"},
    "CF": {"CF", "LF", "RF"},
    "RF": {"RF", "LF"},
    "UT": {"1B", "2B", "3B", "SS", "LF", "CF", "RF"},
}


def is_pitcher(p: NplPlayer) -> bool:
    return p.throws is not None or p.position.upper() == "P"


def can_play(p: NplPlayer, position: str) -> bool:
    return position == "DH" or position in _ELIGIBLE.get(p.position.upper(), set())


@dataclass
class LineupCard:
    active: list[int] = field(default_factory=list)
    # Batting order by starter hand ("rhp"/"lhp"): nine (player id, position) slots.
    orders: dict[str, list[tuple[int | None, str]]] = field(default_factory=dict)
    rotation: list[int | None] = field(default_factory=list)
    roles: dict[int, str] = field(default_factory=dict)  # reliever id -> role

    def to_json(self) -> dict:
        return {
            "active": self.active,
            "orders": {k: [list(s) for s in v] for k, v in self.orders.items()},
            "rotation": self.rotation,
            "roles": {str(k): v for k, v in self.roles.items()},
        }

    @classmethod
    def from_json(cls, data: dict) -> LineupCard:
        return cls(
            active=list(data.get("active", [])),
            orders={k: [(s[0], s[1]) for s in v] for k, v in data.get("orders", {}).items()},
            rotation=list(data.get("rotation", [])),
            roles={int(k): v for k, v in data.get("roles", {}).items()},
        )

    @classmethod
    def from_form(cls, form) -> LineupCard:
        """Read the lineup card form (a Starlette FormData or a dict of lists)."""

        def get(key: str) -> str:
            value = form.get(key) if hasattr(form, "getlist") else (form.get(key) or [""])[0]
            return (value or "").strip()

        def pid(key: str) -> int | None:
            value = get(key)
            return int(value) if value.isdigit() else None

        active = form.getlist("active") if hasattr(form, "getlist") else form.get("active", [])
        orders = {
            side: [(pid(f"{side}_player_{i}"), get(f"{side}_pos_{i}")) for i in range(1, 10)]
            for side in SIDES
        }
        rotation = [pid(f"sp_{i}") for i in range(1, ROTATION_SIZE + 1)]
        roles = {}
        for key in form.keys():
            if key.startswith("role_") and key[5:].isdigit() and get(key) in ROLES:
                roles[int(key[5:])] = get(key)
        return cls(
            active=[int(a) for a in active if str(a).isdigit()],
            orders=orders,
            rotation=rotation,
            roles=roles,
        )


@dataclass
class Check:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(card: LineupCard, roster: list[NplPlayer]) -> Check:
    check = Check()
    by_id = {p.mlb_id: p for p in roster}
    active = set(card.active)

    def name(pid: int) -> str:
        return by_id[pid].name if pid in by_id else f"player {pid}"

    unknown = active - set(by_id)
    if unknown:
        check.errors.append(f"{len(unknown)} active player(s) are not on this roster.")
    if len(active) > ACTIVE_LIMIT:
        check.errors.append(f"{len(active)} players active; the limit is {ACTIVE_LIMIT}.")

    for side, label in SIDES.items():
        slots = card.orders.get(side, [])
        players = [pid for pid, _ in slots if pid is not None]
        if len(players) < len(POSITIONS):
            check.errors.append(f"Batting order {label} has {len(players)} of 9 hitters.")
        for pid in {p for p in players if players.count(p) > 1}:
            check.errors.append(f"{name(pid)} bats twice in the order {label}.")
        positions = [pos for pid, pos in slots if pid is not None]
        missing = [pos for pos in POSITIONS if pos not in positions]
        doubled = sorted({pos for pos in positions if positions.count(pos) > 1})
        if doubled:
            check.errors.append(f"Order {label} has two players at {', '.join(doubled)}.")
        if missing and len(players) == len(POSITIONS):
            check.errors.append(f"Order {label} has nobody at {', '.join(missing)}.")
        for pid, pos in slots:
            if pid is None:
                continue
            if pid not in active:
                check.errors.append(f"{name(pid)} is in the order {label} but not active.")
            elif pid in by_id and is_pitcher(by_id[pid]):
                check.errors.append(f"{name(pid)} is a pitcher in the order {label}.")
            elif pid in by_id and pos and not can_play(by_id[pid], pos):
                check.warnings.append(
                    f"{name(pid)} ({by_id[pid].position}) plays {pos} out of position {label}."
                )

    starters = [pid for pid in card.rotation if pid is not None]
    if len(starters) < ROTATION_SIZE:
        check.errors.append(f"The rotation has {len(starters)} of {ROTATION_SIZE} starters.")
    for pid in {p for p in starters if starters.count(p) > 1}:
        check.errors.append(f"{name(pid)} is in the rotation twice.")
    for pid in starters:
        if pid not in active:
            check.errors.append(f"{name(pid)} is in the rotation but not active.")
        elif pid in by_id and not is_pitcher(by_id[pid]):
            check.errors.append(f"{name(pid)} is in the rotation but is not a pitcher.")

    relievers = [pid for pid in card.roles if pid in active and pid not in starters]
    closers = [pid for pid in relievers if card.roles[pid] == "CL"]
    if len(closers) != 1:
        check.errors.append(f"Name one closer ({len(closers)} set).")
    if len(relievers) < 5:
        check.warnings.append(f"Only {len(relievers)} relievers have a bullpen role.")
    return check


def default_card(roster: list[NplPlayer]) -> LineupCard:
    """A legal starting card: fill the positions, the rotation and the pen in
    roster order, then cut to the active limit."""
    hitters = [p for p in roster if not is_pitcher(p)]
    arms = [p for p in roster if is_pitcher(p)]
    order: list[tuple[int, str]] = []
    used: set[int] = set()
    # Scarce positions first, exact roster position before anyone eligible.
    for pos in ("C", "SS", "2B", "CF", "3B", "1B", "LF", "RF", "DH"):
        pick = next((h for h in hitters if h.mlb_id not in used and h.position == pos), None)
        pick = pick or next((h for h in hitters if h.mlb_id not in used and can_play(h, pos)), None)
        pick = pick or next((h for h in hitters if h.mlb_id not in used), None)
        if pick is not None:
            used.add(pick.mlb_id)
            order.append((pick.mlb_id, pos))
    rotation = [p.mlb_id for p in arms[:ROTATION_SIZE]]
    pen = arms[ROTATION_SIZE:]
    room = max(0, ACTIVE_LIMIT - len(order) - len(rotation))
    bench = [h for h in hitters if h.mlb_id not in used]
    # At least eight relievers when the roster has them; hitters fill the rest.
    pen = pen[: min(room, max(8, room - len(bench)))]
    bench = bench[: room - len(pen)]
    roles = {}
    for i, p in enumerate(pen):
        roles[p.mlb_id] = "CL" if i == 0 else "SU" if i < 3 else "LR" if i == len(pen) - 1 else "MR"
    active = [pid for pid, _ in order] + [h.mlb_id for h in bench] + rotation + list(roles)
    return LineupCard(
        active=active,
        orders={side: list(order) for side in SIDES},
        rotation=rotation,
        roles=roles,
    )


def team_card(
    team: str,
    abbr: str,
    card: LineupCard,
    roster: list[NplPlayer],
    starter_slot: int,
    opposing_throws: str,
) -> TeamCard:
    """The engine's card for one game: the order for the opposing starter's
    hand, the starter from this rotation slot (0-based) and the pen by role."""
    by_id = {p.mlb_id: p for p in roster}
    side = "lhp" if opposing_throws == "L" else "rhp"
    lineup = []
    for pid, pos in card.orders[side]:
        p = by_id[pid]
        # Bats are not in the NPL roster data yet; right-handed until the pipeline adds them.
        lineup.append((replace(hitter(str(pid)), name=p.name), pos))

    def arm(pid: int, limit: int):
        p = by_id[pid]
        return replace(pitcher(str(pid), p.throws or "R", limit=limit), name=p.name)

    starters = [pid for pid in card.rotation if pid is not None]
    starter = arm(starters[starter_slot % len(starters)], 95)
    pen_ids = sorted(
        (pid for pid, role in card.roles.items() if pid in card.active and pid not in starters),
        key=lambda pid: ROLE_ORDER.index(card.roles[pid]),
    )
    bullpen = [arm(pid, 25) for pid in pen_ids]
    return TeamCard(name=team, lineup=lineup, starter=starter, bullpen=bullpen, abbr=abbr)
