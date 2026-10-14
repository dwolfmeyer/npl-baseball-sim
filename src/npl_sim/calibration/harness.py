"""Calibration test harness (Milestone 8).

Dave (2026-10-12), from the baiseball review: test suites for accounting
identities, synthetic player types, run-environment targets, and spotting
when a code change shifts results. Each check here returns plain data so the
same code serves the pytest gates (tests/test_calibration.py) and the
`npl-sim calibrate` report.

- Identities: every box score adds up (PA, hits, runs, outs, line score).
- Archetypes: known-talent hitters (three true outcomes, slap, power, speed)
  play back the rates their cards say.
- Run environment: league R/G, HR, K%, BB% and slash line against targets
  in config/calibration.yaml.
- Spread: identical hitters' K% varies across samples the way binomial noise
  says it should, and shrinks as samples grow.
- Regression: a fingerprint of the engine's results (event mix and the
  runs-per-game distribution) compared with a saved baseline by
  Jensen-Shannon divergence and Wasserstein distance.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from npl_sim.engine.box import box_score
from npl_sim.engine.defaults import DEFAULT_ENV
from npl_sim.engine.game import Hitter, TeamCard, simulate_game
from npl_sim.engine.model import NEUTRAL_PARK, Defense, LeagueEnv, pa_probabilities
from npl_sim.engine.sample import average_team, hitter

CONFIG = Path(__file__).resolve().parents[3] / "config" / "calibration.yaml"
BASELINE = Path(__file__).resolve().parents[3] / "tests" / "baselines" / "engine.json"


def play(away: TeamCard, home: TeamCard, games: int, seed: int = 0, env=DEFAULT_ENV):
    """Event logs for `games` games between the same two cards."""
    return [simulate_game(away, home, env, seed + i) for i in range(games)]


def average_games(games: int = 1000, seed: int = 0, env=DEFAULT_ENV) -> list[list[dict]]:
    return play(average_team("Away"), average_team("Home"), games, seed, env)


def input_rates(games: list[list[dict]], env: LeagueEnv = DEFAULT_ENV) -> dict[str, float]:
    """K, BB, HBP and HR per PA that `env` gives the batter-pitcher handedness mix these
    average-team games actually played: what the sim should reproduce."""
    hands = {}
    for team in (average_team("Away"), average_team("Home")):
        for hitter_, _ in team.lineup:
            hands[hitter_.id] = hitter_
        for pitcher_ in (team.starter, *team.bullpen):
            hands[pitcher_.id] = pitcher_
    mix = Counter()
    for events in games:
        for e in events:
            if e["type"] == "pa":
                bat, pit = hands[e["batter"]], hands[e["pitcher"]]
                mix[(bat.side_vs(pit.throws), pit.throws)] += 1
    n = sum(mix.values())
    out = dict.fromkeys(("K", "BB", "HBP", "HR"), 0.0)
    for key, count in mix.items():
        rates = env.split(*key)
        for stat, value in (("K", rates.k), ("BB", rates.bb), ("HBP", rates.hbp), ("HR", rates.hr)):
            out[stat] += value * count / n
    return out


# --- run environment -------------------------------------------------------------


def summary(games: list[list[dict]]) -> dict[str, float]:
    """League batting per team-game and rate stats, from the box scores."""
    t = Counter()
    for events in games:
        box = box_score(events)
        for team in (box.away, box.home):
            for b in box.batting[team].values():
                for k in ("pa", "ab", "h", "doubles", "triples", "hr", "bb", "hbp", "k", "sf"):
                    t[k] += getattr(b, k)
                t["sb"] += b.sb
                t["cs"] += b.cs
            t["r"] += box.runs[team]
            t["team_games"] += 1
    n, pa, ab = t["team_games"], t["pa"], t["ab"]
    singles = t["h"] - t["doubles"] - t["triples"] - t["hr"]
    bases = singles + 2 * t["doubles"] + 3 * t["triples"] + 4 * t["hr"]
    bip = ab - t["k"] - t["hr"] + t["sf"]
    return {
        "team_games": n,
        "r_per_g": t["r"] / n,
        "hr_per_g": t["hr"] / n,
        "k_per_g": t["k"] / n,
        "bb_per_g": t["bb"] / n,
        "sb_per_g": t["sb"] / n,
        "k_pct": t["k"] / pa,
        "bb_pct": t["bb"] / pa,
        "hr_pct": t["hr"] / pa,
        "avg": t["h"] / ab,
        "obp": (t["h"] + t["bb"] + t["hbp"]) / (ab + t["bb"] + t["hbp"] + t["sf"]),
        "slg": bases / ab,
        "babip": (t["h"] - t["hr"]) / bip if bip else 0.0,
    }


@dataclass
class Check:
    stat: str
    value: float
    lo: float
    hi: float

    @property
    def ok(self) -> bool:
        return self.lo <= self.value <= self.hi


def load_targets(name: str, path: Path = CONFIG) -> dict[str, dict]:
    """A target set from config/calibration.yaml: {stat: {target, tol} or {lo, hi}}."""
    sets = yaml.safe_load(path.read_text())["targets"]
    if name not in sets:
        raise KeyError(f"no calibration targets named {name!r}; have {', '.join(sets)}")
    return sets[name]


def check_targets(stats: dict[str, float], targets: dict[str, dict]) -> list[Check]:
    """Each targeted stat against its band: `lo`/`hi`, or `target` +/- `tol`
    (a fraction of the target, e.g. 0.03 = within 3%)."""
    out = []
    for stat, t in targets.items():
        if "target" in t:
            lo, hi = t["target"] * (1 - t["tol"]), t["target"] * (1 + t["tol"])
        else:
            lo, hi = t["lo"], t["hi"]
        out.append(Check(stat, stats[stat], lo, hi))
    return out


# --- accounting identities --------------------------------------------------------


def identities(games: list[list[dict]]) -> list[str]:
    """Every way a box score fails to add up, as messages (empty = clean)."""
    bad = []
    for i, events in enumerate(games):
        box = box_score(events)
        final = events[-1]
        for team, opp in ((box.away, box.home), (box.home, box.away)):
            where = f"game {i} {team}"
            bat = box.batting[team].values()
            pit = box.pitching[opp].values()
            for b in bat:
                if b.pa != b.ab + b.bb + b.hbp + b.sf:
                    bad.append(f"{where}: {b.name} PA {b.pa} != AB+BB+HBP+SF")
                if b.doubles + b.triples + b.hr > b.h:
                    bad.append(f"{where}: {b.name} extra-base hits exceed hits")
            runs = box.runs[team]
            if sum(b.r for b in bat) != runs:
                bad.append(f"{where}: batters' runs != team runs")
            if sum(p.r for p in pit) != runs:
                bad.append(f"{where}: runs charged to {opp} pitchers != {team} runs")
            if sum(box.line[team]) != runs:
                bad.append(f"{where}: line score != runs")
            if sum(b.h for b in bat) != box.hits[team] or sum(p.h for p in pit) != box.hits[team]:
                bad.append(f"{where}: hits don't match across batting, pitching and line")
            if sum(b.pa for b in bat) != sum(p.bf for p in pit):
                bad.append(f"{where}: PA != batters faced")
            if sum(b.k for b in bat) != sum(p.k for p in pit):
                bad.append(f"{where}: strikeouts don't match")
            if sum(p.er for p in pit) > runs:
                bad.append(f"{where}: more earned runs than runs")
        if final["away"] == final["home"]:
            bad.append(f"game {i}: ended tied")
        if sum(p.outs for p in box.pitching[box.home].values()) != 3 * len(box.line[box.away]):
            bad.append(f"game {i}: home pitchers' outs != 3 x innings the away team batted")
    return bad


# --- archetypes --------------------------------------------------------------------

# Per-PA rates on the hitter's card (multipliers on hit/power), vs both hands.
ARCHETYPES = {
    "three_true_outcomes": {"k": 0.33, "bb": 0.13, "hr": 0.065},
    "slap": {"k": 0.12, "bb": 0.06, "hr": 0.008, "hit": 1.12, "power": 0.7},
    "power": {"k": 0.26, "bb": 0.10, "hr": 0.060, "power": 1.3},
    "average": {},
}


def archetype_team(kind: str, speed: float = 0.0) -> TeamCard:
    """A team whose nine hitters all have archetype `kind`'s card."""
    base = average_team(f"{kind}")
    lineup = [
        (
            replace(hitter(h.id, bats=h.bats, speed=speed, scale=ARCHETYPES[kind]), name=h.id),
            pos,
        )
        for h, pos in base.lineup
    ]
    return replace(base, lineup=lineup)


def expected_rates(h: Hitter, pitchers, env: LeagueEnv = DEFAULT_ENV) -> dict[str, float]:
    """K, BB and HR per PA the model gives `h` against a staff (equal weight)."""
    tot = Counter()
    for p in pitchers:
        side = h.side_vs(p.throws)
        probs = pa_probabilities(
            h.rates_vs(p.throws), p.rates_vs(side), env, side, p.throws, NEUTRAL_PARK, Defense()
        )
        for k in ("K", "BB", "HR"):
            tot[k] += probs[k] / len(pitchers)
    return dict(tot)


def observed_rates(games: list[list[dict]], team: str) -> dict[str, float]:
    c = Counter(e["result"] for g in games for e in g if e["type"] == "pa" and e["team"] == team)
    pa = sum(c.values())
    out = {k: c[k] / pa for k in ("K", "BB", "HR")}
    out["pa"] = pa
    out["tto"] = out["K"] + out["BB"] + out["HR"]
    return out


def archetype_report(games: int = 150, seed: int = 0) -> dict[str, dict]:
    """For each archetype: the sim's K/BB/HR per PA next to what its cards give."""
    opp = average_team("Opp")
    staff = [opp.starter, *opp.bullpen]
    report = {}
    for kind in ARCHETYPES:
        team = archetype_team(kind)
        logs = play(team, opp, games, seed)
        sim = observed_rates(logs, team.name)
        exp = Counter()
        for h, _ in team.lineup:
            for k, v in expected_rates(h, staff).items():
                exp[k] += v / len(team.lineup)
        report[kind] = {"sim": sim, "expected": dict(exp)}
    return report


def steals_per_game(speed: float, games: int = 150, seed: int = 0) -> float:
    team = archetype_team("average", speed=speed)
    logs = play(team, average_team("Opp"), games, seed)
    # The team is the away side, so it runs in the top halves.
    tries = sum(
        1
        for g in logs
        for e in g
        if e["type"] in ("stolen_base", "caught_stealing") and e["half"] == "top"
    )
    return tries / games


# --- spread and sample size ---------------------------------------------------------


def k_spread(games: list[list[dict]], block: int) -> tuple[float, float]:
    """(observed, binomial) standard deviation of K% across blocks of `block`
    consecutive PAs for each hitter. Every hitter in `games` should have the
    same true K% for the comparison to mean anything."""
    seq: dict[str, list[int]] = {}
    for g in games:
        for e in g:
            if e["type"] == "pa":
                seq.setdefault(e["batter"], []).append(e["result"] == "K")
    rates = []
    for ks in seq.values():
        for i in range(0, len(ks) - block + 1, block):
            rates.append(sum(ks[i : i + block]) / block)
    mean = sum(rates) / len(rates)
    sd = math.sqrt(sum((r - mean) ** 2 for r in rates) / (len(rates) - 1))
    return sd, math.sqrt(mean * (1 - mean) / block)


# --- regression detection ------------------------------------------------------------

MAX_RUNS = 20


def fingerprint(games: list[list[dict]]) -> dict:
    """What the engine produced: the PA result mix and the distribution of a
    team's runs in a game (0..MAX_RUNS, the last bin holding the rest)."""
    mix = Counter(e["result"] for g in games for e in g if e["type"] == "pa")
    total = sum(mix.values())
    runs = Counter()
    for g in games:
        for side in ("away", "home"):
            runs[min(MAX_RUNS, g[-1][side])] += 1
    n = sum(runs.values())
    return {
        "games": len(games),
        "event_mix": {k: v / total for k, v in sorted(mix.items())},
        "runs": [runs[i] / n for i in range(MAX_RUNS + 1)],
        "summary": summary(games),
    }


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence (base 2, 0..1) between two distributions."""
    keys = set(p) | set(q)

    def kl(a, b):
        return sum(a[k] * math.log2(a[k] / b[k]) for k in keys if a.get(k, 0) > 0)

    m = {k: (p.get(k, 0.0) + q.get(k, 0.0)) / 2 for k in keys}
    pp = {k: p.get(k, 0.0) for k in keys}
    qq = {k: q.get(k, 0.0) for k in keys}
    return (kl(pp, m) + kl(qq, m)) / 2


def wasserstein(p: list[float], q: list[float]) -> float:
    """Earth mover's distance between two distributions on 0, 1, 2, ... runs."""
    total, cp, cq = 0.0, 0.0, 0.0
    for a, b in zip(p, q, strict=True):
        cp += a
        cq += b
        total += abs(cp - cq)
    return total


# Same seeds and same engine give exactly 0. A change that only reorders
# random draws looks like a fresh set of 1,000 games (JSD about 0.00005,
# W1 about 0.11), inside these limits; a 15% change in home runs is caught.
# Wasserstein noise between two 1,000-game seed sets runs 0.05 to 0.20; a 25% HR bump gives 0.45.
DRIFT_LIMITS = {"jsd": 0.0001, "wasserstein": 0.25}


def drift(base: dict, now: dict) -> dict[str, float]:
    return {
        "jsd": js_divergence(base["event_mix"], now["event_mix"]),
        "wasserstein": wasserstein(base["runs"], now["runs"]),
    }


def save_baseline(fp: dict, path: Path = BASELINE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fp, indent=1, sort_keys=True) + "\n")
    return path


def load_baseline(path: Path = BASELINE) -> dict:
    return json.loads(path.read_text())
