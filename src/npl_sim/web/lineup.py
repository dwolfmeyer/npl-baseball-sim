"""The manager's weekly lineup card: who is active, batting orders against
right- and left-handed starters, the rotation with each starter's plan
(pitch limit, times-through-the-order hook, opener, piggyback), bullpen roles
and situational defense.

The card is checked against the roster and position eligibility (see
`npl_sim.rosters.eligibility`) and turned into the engine's `TeamCard`. Until
the weekly cards from the data pipeline land, every player sims with
league-average rates. The engine uses pitch limits, openers and piggybacks
now; the times-through-the-order hook and situational defense are saved on
the card for the engine's lineup and bullpen logic (Milestone 6).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from npl_sim.engine.game import TeamCard
from npl_sim.engine.sample import hitter, pitcher
from npl_sim.rosters.eligibility import Eligibility
from npl_sim.rosters.npl_app import NplPlayer

ACTIVE_LIMIT = 30
ROTATION_SIZE = 5
POSITIONS = ("C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH")
SIDES = {"rhp": "vs RHP", "lhp": "vs LHP"}
ROLES = {"CL": "Closer", "SU": "Setup", "MR": "Middle", "LR": "Long"}
# Relievers enter in this order when the starter tires (closer last).
ROLE_ORDER = ("LR", "MR", "SU", "CL")
PITCH_LIMITS = range(60, 125, 5)
DEFAULT_PITCH_LIMIT = 95
TTO_HOOKS = {0: "No hook", 2: "Before 3rd time through", 3: "Before 4th time through"}
OPENER_PITCHES = 25
PIGGYBACK_PITCHES = 60
DEFENSE_RULES = {
    "infield_in": "Infield in with a runner on 3rd and fewer than 2 outs, "
    "7th inning or later, tied or within a run",
    "dp_depth": "Double-play depth with a runner on 1st and fewer than 2 outs",
    "guard_lines": "Guard the lines in the 8th or later with a lead of 1 to 3 runs",
}

# Roster positions used when a player has no MLB fielding record at all.
_ROSTER_POSITIONS = {
    "C": {"C"},
    "1B": {"1B"},
    "2B": {"2B"},
    "3B": {"3B"},
    "SS": {"SS"},
    "IF": {"1B", "2B", "3B", "SS"},
    "OF": {"LF", "CF", "RF"},
    "LF": {"LF"},
    "CF": {"CF"},
    "RF": {"RF"},
    "UT": {"1B", "2B", "3B", "SS", "LF", "CF", "RF"},
}


_INFIELD = {"1B", "2B", "3B", "SS"}


def is_pitcher(p: NplPlayer) -> bool:
    return p.throws is not None or p.position.upper() == "P"


def fielding(p: NplPlayer, elig: Eligibility | None) -> tuple[set[str], bool]:
    """Positions the player may field and whether that came from MLB games
    (True) or, lacking any MLB fielding record, his NPL roster position."""
    if elig is not None and p.mlb_id is not None:
        rec = elig.players.get(p.mlb_id)
        if rec is not None and any(rec.games.get(s) for s in (elig.season - 1, elig.season)):
            return elig.positions(p.mlb_id) or set(), True
    return set(_ROSTER_POSITIONS.get(p.position.upper(), set())), False


def can_play(p: NplPlayer, position: str, elig: Eligibility | None = None) -> bool:
    return position == "DH" or position in fielding(p, elig)[0]


@dataclass
class StarterPlan:
    pitch_limit: int = DEFAULT_PITCH_LIMIT
    tto_hook: int = 0  # 0 = none; 2 = pull before the 3rd time through; 3 = before the 4th
    opener: int | None = None  # a reliever who pitches the 1st, then the starter
    piggyback: int | None = None  # a reliever who follows the starter for bulk innings


@dataclass
class LineupCard:
    active: list[int] = field(default_factory=list)
    # Batting order by starter hand ("rhp"/"lhp"): nine (player id, position) slots.
    orders: dict[str, list[tuple[int | None, str]]] = field(default_factory=dict)
    rotation: list[int | None] = field(default_factory=list)
    plans: list[StarterPlan] = field(default_factory=list)  # one per rotation slot
    roles: dict[int, str] = field(default_factory=dict)  # reliever id -> role
    defense: dict[str, bool] = field(default_factory=dict)  # DEFENSE_RULES key -> on

    def plan(self, slot: int) -> StarterPlan:
        return self.plans[slot] if slot < len(self.plans) else StarterPlan()

    def to_json(self) -> dict:
        return {
            "active": self.active,
            "orders": {k: [list(s) for s in v] for k, v in self.orders.items()},
            "rotation": self.rotation,
            "plans": [asdict(p) for p in self.plans],
            "roles": {str(k): v for k, v in self.roles.items()},
            "defense": self.defense,
        }

    @classmethod
    def from_json(cls, data: dict) -> LineupCard:
        return cls(
            active=list(data.get("active", [])),
            orders={k: [(s[0], s[1]) for s in v] for k, v in data.get("orders", {}).items()},
            rotation=list(data.get("rotation", [])),
            plans=[StarterPlan(**p) for p in data.get("plans", [])],
            roles={int(k): v for k, v in data.get("roles", {}).items()},
            defense={k: bool(v) for k, v in data.get("defense", {}).items() if k in DEFENSE_RULES},
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
        plans = [
            StarterPlan(
                pitch_limit=pid(f"sp_limit_{i}") or DEFAULT_PITCH_LIMIT,
                tto_hook=pid(f"sp_tto_{i}") or 0,
                opener=pid(f"sp_opener_{i}"),
                piggyback=pid(f"sp_piggyback_{i}"),
            )
            for i in range(1, ROTATION_SIZE + 1)
        ]
        roles = {}
        for key in form.keys():
            if key.startswith("role_") and key[5:].isdigit() and get(key) in ROLES:
                roles[int(key[5:])] = get(key)
        return cls(
            active=[int(a) for a in active if str(a).isdigit()],
            orders=orders,
            rotation=rotation,
            plans=plans,
            roles=roles,
            defense={key: bool(get(f"def_{key}")) for key in DEFENSE_RULES},
        )


@dataclass
class Check:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(card: LineupCard, roster: list[NplPlayer], elig: Eligibility | None = None) -> Check:
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
            elif pid in by_id and pos and not can_play(by_id[pid], pos, elig):
                where = ", ".join(sorted(fielding(by_id[pid], elig)[0])) or "DH only"
                # Allowed, as in Scoresheet: he plays out of position with a defensive penalty.
                check.warnings.append(
                    f"{name(pid)} is out of position at {pos} {label} (eligible: {where})."
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

    for slot in range(len(card.rotation)):
        plan = card.plan(slot)
        if plan.pitch_limit not in PITCH_LIMITS:
            check.errors.append(f"Starter {slot + 1}'s pitch limit must be 60 to 120.")
        if plan.opener is not None and plan.opener == plan.piggyback:
            check.errors.append(f"Starter {slot + 1} has the same opener and piggyback.")
        for what, pid in (("opener", plan.opener), ("piggyback", plan.piggyback)):
            if pid is None:
                continue
            if pid not in active or pid not in by_id or not is_pitcher(by_id[pid]):
                check.errors.append(f"Starter {slot + 1}'s {what} must be an active pitcher.")
            elif pid in starters:
                check.errors.append(
                    f"{name(pid)} is in the rotation, so he can't be starter {slot + 1}'s {what}."
                )

    relievers = [pid for pid in card.roles if pid in active and pid not in starters]
    closers = [pid for pid in relievers if card.roles[pid] == "CL"]
    if len(closers) != 1:
        check.errors.append(f"Name one closer ({len(closers)} set).")
    if len(relievers) < 5:
        check.warnings.append(f"Only {len(relievers)} relievers have a bullpen role.")
    return check


def default_card(roster: list[NplPlayer], elig: Eligibility | None = None) -> LineupCard:
    """A legal starting card: fill the positions with eligible hitters, the
    rotation and the pen in roster order, then cut to the active limit."""
    hitters = [p for p in roster if not is_pitcher(p)]
    arms = [p for p in roster if is_pitcher(p)]
    picks: dict[str, int] = {}
    used: set[int] = set()
    fields = ("C", "SS", "2B", "CF", "3B", "1B", "LF", "RF")
    # Scarce positions first; among eligible hitters, those with fewest options first.
    for pos in fields:
        options = [h for h in hitters if h.mlb_id not in used and can_play(h, pos, elig)]
        if options:
            pick = min(options, key=lambda h: len(fielding(h, elig)[0]))
            picks[pos] = pick.mlb_id
            used.add(pick.mlb_id)
    # Nobody eligible: put someone there anyway (an infielder for an infield spot,
    # an outfielder for the outfield) and let the check flag it. Then the DH.
    for pos in (*fields, "DH"):
        if pos in picks:
            continue
        group = _INFIELD if pos in _INFIELD else {"LF", "CF", "RF"}
        rest = [h for h in hitters if h.mlb_id not in used]
        rest.sort(key=lambda h: pos != "DH" and not (fielding(h, elig)[0] & group))
        if rest:
            picks[pos] = rest[0].mlb_id
            used.add(rest[0].mlb_id)
    order = [(pid, pos) for pos, pid in picks.items()]
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
        plans=[StarterPlan() for _ in rotation],
        roles=roles,
    )


def team_card(
    team: str,
    abbr: str,
    card: LineupCard,
    roster: list[NplPlayer],
    starter_slot: int,
    opposing_throws: str,
    elig: Eligibility | None = None,
) -> TeamCard:
    """The engine's card for one game: the order for the opposing starter's
    hand, the starter from this rotation slot (0-based) with his plan, and the
    pen by role."""
    by_id = {p.mlb_id: p for p in roster}
    side = "lhp" if opposing_throws == "L" else "rhp"
    lineup = []
    for pid, pos in card.orders[side]:
        rec = elig.players.get(pid) if elig else None
        # Right-handed when the hand isn't known yet.
        bats = rec.bats if rec and rec.bats in ("L", "R", "S") else "R"
        lineup.append((replace(hitter(str(pid), bats=bats), name=by_id[pid].name), pos))

    def arm(pid: int, limit: int):
        p = by_id[pid]
        return replace(pitcher(str(pid), p.throws or "R", limit=limit), name=p.name)

    slots = [(i, pid) for i, pid in enumerate(card.rotation) if pid is not None]
    slot, starter_id = slots[starter_slot % len(slots)]
    plan = card.plan(slot)
    starters = {pid for _, pid in slots}
    starter = arm(starter_id, plan.pitch_limit)
    # An opener takes the first inning or so; the starter follows, then the piggyback.
    first = arm(plan.opener, OPENER_PITCHES) if plan.opener else starter
    bulk = [starter] if plan.opener else []
    if plan.piggyback:
        bulk.append(arm(plan.piggyback, PIGGYBACK_PITCHES))
    planned = {plan.opener, plan.piggyback}
    pen_ids = sorted(
        (
            pid
            for pid in card.roles
            if pid in card.active and pid not in starters and pid not in planned
        ),
        key=lambda pid: ROLE_ORDER.index(card.roles[pid]),
    )
    bullpen = bulk + [arm(pid, 25) for pid in pen_ids]
    return TeamCard(name=team, lineup=lineup, starter=first, bullpen=bullpen, abbr=abbr)
