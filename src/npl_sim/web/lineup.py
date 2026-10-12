"""The manager's weekly lineup card: who is active, batting orders against
right- and left-handed starters, the rotation with each starter's plan
(pitch limit, times-through-the-order hook, opener, piggyback), bullpen roles
and situational defense.

The card is checked against the roster and position eligibility (see
`npl_sim.rosters.eligibility`) and turned into the engine's `TeamCard`. Until
the weekly cards from the data pipeline land, every player sims with
league-average rates. The engine uses pitch limits, openers, piggybacks and
situational defense now; the times-through-the-order hook is saved on the
card for the engine's lineup and bullpen logic (Milestone 6).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from npl_sim.eligibility import FIELD_POSITIONS, LISTED_FALLBACK
from npl_sim.engine.alignment import AlignmentRule
from npl_sim.engine.game import DefensiveSub as EngineSub
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
# Pull the starter once he has faced the order this many times (0 = no limit).
TTO_HOOKS = {
    0: "No limit",
    1: "After 1st time through",
    2: "After 2nd time through",
    3: "After 3rd time through",
}
OPENER_PITCHES = 25
DEFENSE_RULES = {
    "infield_in": "Infield in with a runner on 3rd and fewer than 2 outs, "
    "7th inning or later, tied or within a run",
    "dp_depth": "Double-play depth with a runner on 1st and fewer than 2 outs",
    "guard_lines": "Guard the lines in the 8th or later with a lead of 1 or 2 runs",
}
# The engine's rule for each box, checked in this order; the first that fits wins.
DEFENSE_ENGINE_RULES = {
    "infield_in": AlignmentRule("infield_in", from_inning=7, min_lead=-1, max_lead=1),
    "dp_depth": AlignmentRule("double_play"),
    "guard_lines": AlignmentRule("guard_lines", from_inning=8, min_lead=1, max_lead=2),
}

_INFIELD = {"1B", "2B", "3B", "SS"}


def is_pitcher(p: NplPlayer) -> bool:
    return p.throws is not None or p.position.upper() == "P"


def fielding(p: NplPlayer, elig: Eligibility | None) -> tuple[set[str], bool]:
    """Positions the player may field and whether that came from MLB games
    (True) or from his listed NPL position (False)."""
    if elig is not None and p.mlb_id is not None:
        return elig.positions(p.mlb_id, p.position), elig.qualifies(p.mlb_id)
    return set(LISTED_FALLBACK.get(p.position.upper(), set())), False


def can_play(p: NplPlayer, position: str, elig: Eligibility | None = None) -> bool:
    return position == "DH" or position in fielding(p, elig)[0]


@dataclass
class StarterPlan:
    pitch_limit: int = DEFAULT_PITCH_LIMIT
    tto_hook: int = 0  # pull after this many times through the order; 0 = no limit
    # An opener game: the listed pitcher goes an inning or so and the piggyback
    # pitches the bulk innings after him (the piggyback only applies then).
    opener: bool = False
    piggyback: int | None = None

    def __post_init__(self) -> None:
        # Cards saved before the opener was a checkbox stored a pitcher id.
        self.opener = bool(self.opener)


@dataclass
class DefensiveSub:
    """A late-game defensive replacement: `player` comes off the bench to play
    `position` from `from_inning` on, replacing `replaces` (a specific starter)
    or, when that is None, whoever is playing the position."""

    player: int
    position: str
    from_inning: int = 9
    only_when_leading: bool = True
    replaces: int | None = None


DEFENSIVE_SUB_ROWS = 4
# How a reliever is used; "" follows his role (closer and setup high leverage,
# middle medium, long man long relief).
USAGE = {
    "": "By role",
    "high": "High leverage",
    "medium": "Medium leverage",
    "low": "Low leverage",
    "mopup": "Mop-up (blowouts)",
    "long": "Long relief",
    "any": "Any situation",
}
ROLE_USAGE = {"CL": "high", "SU": "high", "MR": "medium", "LR": "long"}
MATCHUPS = {"": "Any batter", "L": "Lefty specialist", "R": "Righty specialist"}
# Stint length in outs; None = the engine default (3, or 9 for long and mop-up).
STINTS = {None: "Default", 1: "1 out", 2: "2 outs", 3: "1 inning", 6: "2 innings", 9: "3 innings"}


@dataclass
class ReliefPlan:
    usage: str = ""
    start_of_inning_only: bool = False
    vs: str | None = None  # "L"/"R": brought in for same-side batters
    max_outs: int | None = None

    def is_default(self) -> bool:
        return self == ReliefPlan()


SUB_INNINGS = range(6, 13)


@dataclass
class LineupCard:
    active: list[int] = field(default_factory=list)
    # Batting order by starter hand ("rhp"/"lhp"): nine (player id, position) slots.
    orders: dict[str, list[tuple[int | None, str]]] = field(default_factory=dict)
    rotation: list[int | None] = field(default_factory=list)
    plans: list[StarterPlan] = field(default_factory=list)  # one per rotation slot
    roles: dict[int, str] = field(default_factory=dict)  # reliever id -> role
    defense: dict[str, bool] = field(default_factory=dict)  # DEFENSE_RULES key -> on
    subs: list[DefensiveSub] = field(default_factory=list)
    relief: dict[int, ReliefPlan] = field(default_factory=dict)  # reliever id -> usage

    def relief_plan(self, pid: int) -> ReliefPlan:
        return self.relief.get(pid) or ReliefPlan()

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
            "subs": [asdict(sub) for sub in self.subs],
            "relief": {str(k): asdict(v) for k, v in self.relief.items()},
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
            subs=[DefensiveSub(**sub) for sub in data.get("subs", [])],
            relief={int(k): ReliefPlan(**v) for k, v in data.get("relief", {}).items()},
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
                opener=bool(get(f"sp_opener_{i}")),
                piggyback=pid(f"sp_piggyback_{i}") if get(f"sp_opener_{i}") else None,
            )
            for i in range(1, ROTATION_SIZE + 1)
        ]
        roles, relief = {}, {}
        for key in form.keys():
            if key.startswith("role_") and key[5:].isdigit() and get(key) in ROLES:
                roles[int(key[5:])] = get(key)
            if key.startswith("rp_use_") and key[7:].isdigit():
                rid = key[7:]
                outs = pid(f"rp_outs_{rid}")
                plan = ReliefPlan(
                    usage=get(key) if get(key) in USAGE else "",
                    start_of_inning_only=bool(get(f"rp_clean_{rid}")),
                    vs=get(f"rp_vs_{rid}") or None if get(f"rp_vs_{rid}") in MATCHUPS else None,
                    max_outs=outs if outs in STINTS else None,
                )
                if not plan.is_default():
                    relief[int(rid)] = plan
        return cls(
            active=[int(a) for a in active if str(a).isdigit()],
            orders=orders,
            rotation=rotation,
            plans=plans,
            roles=roles,
            defense={key: bool(get(f"def_{key}")) for key in DEFENSE_RULES},
            subs=[
                DefensiveSub(
                    player=player,
                    position=get(f"sub_pos_{i}") or "",
                    from_inning=pid(f"sub_inning_{i}") or 9,
                    only_when_leading=bool(get(f"sub_leading_{i}")),
                    replaces=pid(f"sub_replaces_{i}"),
                )
                for i in range(1, DEFENSIVE_SUB_ROWS + 1)
                if (player := pid(f"sub_player_{i}")) is not None
            ],
            relief=relief,
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
        pid = plan.piggyback
        if plan.opener and pid is None:
            check.warnings.append(
                f"Starter {slot + 1} opens with no piggyback; the bullpen takes over after him."
            )
        if pid is None:
            continue
        if pid not in active or pid not in by_id or not is_pitcher(by_id[pid]):
            check.errors.append(f"Starter {slot + 1}'s piggyback must be an active pitcher.")
        elif pid in starters:
            check.errors.append(
                f"{name(pid)} is in the rotation, so he can't be starter {slot + 1}'s piggyback."
            )

    starting = {pid for slots in card.orders.values() for pid, _ in slots if pid is not None}
    used_subs = [sub.player for sub in card.subs]
    for sub in card.subs:
        who = name(sub.player)
        if sub.player not in active or sub.player not in by_id or is_pitcher(by_id[sub.player]):
            check.errors.append(f"Defensive replacement {who} must be an active hitter.")
            continue
        if used_subs.count(sub.player) > 1:
            check.errors.append(f"{who} is listed as a defensive replacement twice.")
        if sub.position not in POSITIONS or sub.position == "DH":
            check.errors.append(f"Pick a field position for defensive replacement {who}.")
        elif not can_play(by_id[sub.player], sub.position, elig):
            check.warnings.append(
                f"Defensive replacement {who} is out of position at {sub.position}."
            )
        if sub.replaces is not None and sub.replaces not in starting:
            check.errors.append(
                f"{who} replaces {name(sub.replaces)}, who isn't in a batting order."
            )
        if sub.replaces == sub.player:
            check.errors.append(f"{who} can't replace himself.")
        if all(sub.player in [pid for pid, _ in slots] for slots in card.orders.values()):
            check.warnings.append(
                f"{who} starts in both batting orders, so he's never on the bench."
            )

    relievers = [pid for pid in card.roles if pid in active and pid not in starters]
    closers = [pid for pid in relievers if card.roles[pid] == "CL"]
    if len(closers) > 1:
        # Not required, but two closers just pitch the 9th in role order.
        check.warnings.append(f"{len(closers)} relievers are set as closer.")
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


def autofill(card: LineupCard, roster: list[NplPlayer], elig: Eligibility) -> LineupCard:
    """Fill the batting orders, rotation and bullpen roles from the active
    players, using position eligibility and MLB games over the last two seasons:

    - each field position goes to the eligible hitter with the most games
      there; the DH is the remaining hitter with the most games, and the order
      runs from most games played to fewest;
    - the five pitchers with the most starts form the rotation; the closer has
      the most saves, two setup men the most holds, the long man the most
      innings per game, and the rest pitch middle relief.

    Active players are kept; when there are more than 30 the ones with the
    fewest games sit. Plans, defense and defensive subs carry over.
    """
    seasons = (elig.season - 1, elig.season)
    by_id = {p.mlb_id: p for p in roster}
    rec = {pid: elig.players.get(pid) for pid in by_id}

    def played(pid: int) -> int:
        r = rec.get(pid)
        if r is None:
            return 0
        return r.total("g", seasons) if is_pitcher(by_id[pid]) else r.games_played(seasons)

    active = [pid for pid in card.active if pid in by_id] or list(by_id)
    active = sorted(active, key=played, reverse=True)[:ACTIVE_LIMIT]
    hitters = [pid for pid in active if not is_pitcher(by_id[pid])]
    arms = [pid for pid in active if is_pitcher(by_id[pid])]

    def at(pid: int, pos: str) -> int:
        r = rec.get(pid)
        return sum(r.games.get(s, {}).get(pos, 0) for s in seasons) if r else 0

    # Most games at a position wins it, one pair at a time.
    picks: dict[str, int] = {}
    pairs = sorted(
        (
            (at(pid, pos), pid, pos)
            for pid in hitters
            for pos in POSITIONS[:-1]
            if can_play(by_id[pid], pos, elig)
        ),
        reverse=True,
    )
    for _, pid, pos in pairs:
        if pos not in picks and pid not in picks.values():
            picks[pos] = pid
    for pos in POSITIONS:
        if pos not in picks:
            rest = sorted((h for h in hitters if h not in picks.values()), key=played)
            if rest:
                picks[pos] = rest[-1]
    order = sorted(((pid, pos) for pos, pid in picks.items()), key=lambda s: -played(s[0]))

    def pitching(pid: int, stat: str) -> int:
        r = rec.get(pid)
        return r.total(stat, seasons) if r else 0

    # Most starts; among pitchers short on starts, keep the late-inning arms in the pen.
    rotation = sorted(
        arms,
        key=lambda pid: (
            pitching(pid, "gs"),
            -pitching(pid, "sv") - pitching(pid, "hld"),
            pitching(pid, "outs"),
        ),
        reverse=True,
    )[:ROTATION_SIZE]
    pen = [pid for pid in arms if pid not in rotation]
    roles: dict[int, str] = {}
    if pen:
        closer = max(pen, key=lambda pid: (pitching(pid, "sv"), pitching(pid, "hld")))
        roles[closer] = "CL"
        setup = sorted(
            (pid for pid in pen if pid not in roles),
            key=lambda pid: (pitching(pid, "hld"), pitching(pid, "sv")),
            reverse=True,
        )[:2]
        roles.update(dict.fromkeys(setup, "SU"))
        rest = [pid for pid in pen if pid not in roles]
        if rest:
            long_man = max(rest, key=lambda pid: pitching(pid, "outs") / max(1, pitching(pid, "g")))
            roles[long_man] = "LR"
        roles.update({pid: "MR" for pid in pen if pid not in roles})
    return replace(
        card,
        active=active,
        orders={side: list(order) for side in SIDES},
        rotation=rotation,
        plans=[card.plan(i) for i in range(len(rotation))],
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

    def bat(pid: int):
        rec = elig.players.get(pid) if elig else None
        # Right-handed when the hand isn't known yet.
        bats = rec.bats if rec and rec.bats in ("L", "R", "S") else "R"
        return replace(hitter(str(pid), bats=bats), name=by_id[pid].name)

    lineup = [(bat(pid), pos) for pid, pos in card.orders[side]]
    starting = {pid for pid, _ in card.orders[side]}
    # Subs who start today, or who are only for a starter sitting today, stay out.
    defensive_subs = tuple(
        EngineSub(
            player=bat(sub.player),
            position=sub.position,
            from_inning=sub.from_inning,
            only_when_leading=sub.only_when_leading,
            replaces=str(sub.replaces) if sub.replaces else None,
        )
        for sub in card.subs
        if sub.player in by_id
        and sub.player in card.active
        and sub.player not in starting
        and sub.position in FIELD_POSITIONS
        and (sub.replaces is None or sub.replaces in starting)
    )
    alignment_rules = tuple(
        rule for key, rule in DEFENSE_ENGINE_RULES.items() if card.defense.get(key)
    )

    def arm(pid: int, limit: int):
        p = by_id[pid]
        return replace(pitcher(str(pid), p.throws or "R", limit=limit), name=p.name)

    slots = [(i, pid) for i, pid in enumerate(card.rotation) if pid is not None]
    slot, starter_id = slots[starter_slot % len(slots)]
    plan = card.plan(slot)
    starters = {pid for _, pid in slots}
    # In an opener game the listed pitcher goes an inning or so and the piggyback
    # takes the bulk innings, up to the slot's pitch limit.
    if plan.opener:
        first = arm(starter_id, OPENER_PITCHES)
        bulk = [arm(plan.piggyback, plan.pitch_limit)] if plan.piggyback else []
    else:
        first, bulk = arm(starter_id, plan.pitch_limit), []
    planned = {plan.piggyback} if plan.opener else set()
    pen_ids = sorted(
        (
            pid
            for pid in card.roles
            if pid in card.active and pid not in starters and pid not in planned
        ),
        key=lambda pid: ROLE_ORDER.index(card.roles[pid]),
    )
    bullpen = bulk + [arm(pid, 25) for pid in pen_ids]
    return TeamCard(
        name=team,
        lineup=lineup,
        starter=first,
        bullpen=bullpen,
        abbr=abbr,
        alignment_rules=alignment_rules,
        defensive_subs=defensive_subs,
    )
