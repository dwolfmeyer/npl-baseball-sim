"""The sim audit log: every plate appearance of a sim week with what the
engine knew going in, the odds it drew from and what happened, plus the
real MLB stats behind each player's card, so the engine can be checked for
realism.

`npl-sim sim-week --audit DIR` writes, per game, `game-NN.jsonl` with three
kinds of records (the "record" key):

- game: teams, day, seed, final score.
- setup: one per team, the owner's settings behind the manager decisions:
  pitch limits, the starter's hook (runs, times through, leverage), bullpen
  roles and stint lengths, weekly outs caps, pinch-hit rules, defensive
  subs, steal aggressiveness and alignment rules.
- decision: one per non-PA event, in game order with the PAs: pitching
  changes (why, the outgoing pitcher's pitches, batters faced and runs,
  leverage, the bullpen situation tier, the reliever's role and who else was
  available), pinch hitters (leverage, expected run gain), defensive subs,
  steals and caught stealing, the extra-inning runner, and wild pitches,
  passed balls and balks. Sacrifice-bunt PAs carry `"bunt": true` in
  their outcome and the bunt's odds in place of the swing-away odds.
- player: each batter and pitcher who took part, with the card rates the
  engine used (vs LHP/RHP or vs LHB/RHB), the talent blend's indexes and
  layers (baseline, process, results), and real stat lines by season and
  by opponent hand (sim week, season to date, last season, two seasons ago).
- pa: one per plate appearance: game state, the matchup's input rates
  (batter, pitcher, league), park, defense and fatigue, the final outcome
  probabilities, the drawn result and batted ball (type, fielder, HR
  direction), runners, pitches, count, win expectancy, WPA and leverage.

and `pas.csv` across the week, one flat row per PA, for spreadsheets.

The engine draws one result per PA; batted-ball type, fielder, count and pitch
total are what it decides. The pitch-by-pitch list is the Gameday pop-up's
sequence (engine/pitchseq.py), simulated after the game to fit those, and is
flagged `"simulated": true`.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from npl_sim.engine.playlog import game_log
from npl_sim.talent.lines import HITS

AUDIT_REAL_SEASON = "season to date"  # the real line pas.csv shows beside each PA
DECISIONS = (
    "pitching_change",
    "pinch_hitter",
    "defensive_sub",
    "stolen_base",
    "caught_stealing",
    "auto_runner",
    "wild_pitch",
    "passed_ball",
    "balk",
)


def _decision(e: dict, number: int, score: dict, extra: dict, names: dict, setup: dict) -> dict:
    """A manager or baserunning decision: the event, player names, and for a
    pitching change the outgoing pitcher's workload and the reliever's role."""
    rec = {
        "record": "decision",
        "game": number,
        "kind": e["type"],
        "inning": e["inning"],
        "half": e["half"],
        "score_before": dict(score),
    }
    rec |= {k: v for k, v in e.items() if k not in ("type", "inning", "half")}
    for key in ("out", "into", "runner", "pitcher"):
        if key in e:
            rec[f"{key}_name"] = names.get(e[key], (e[key], None))[0]
    if e["type"] == "pitching_change":
        rec |= {k: v for k, v in extra.items() if k != "event"}
        side = "home" if e["half"] == "top" else "away"
        relief = setup.get(side, {}).get("relief", {})
        if e["into"] in relief:
            rec["role"] = relief[e["into"]]["role"]
            rec["stint_outs"] = relief[e["into"]]["outs"]
        if e["into"] in setup.get(side, {}).get("follow", []):
            rec["role"] = "follow"
    return rec


def summarize(probs: dict[str, float]) -> dict[str, dict[str, float]]:
    """Outcome odds grouped by result (K, BB, HBP, HR, 1B, 2B, 3B, OUT) and by
    batted-ball type (GB, LD, FB, PU; home runs aside)."""
    result: dict[str, float] = defaultdict(float)
    ball: dict[str, float] = defaultdict(float)
    for key, p in probs.items():
        if ":" in key:
            kind, res = key.split(":")
            result[res] += p
            ball[kind] += p
        else:
            result[key] += p
    return {
        "by_result": {k: round(v, 5) for k, v in result.items()},
        "by_batted_ball": {k: round(v, 5) for k, v in ball.items()},
    }


def game_records(game: dict, number: int, seed: int | None = None, talent=None) -> list[dict]:
    """The audit records for one played game ({away, home, day, events, audit})."""
    events, trail = game["events"], game.get("audit")
    if trail is None:
        raise ValueError("game was played without audit=True")
    start = events[0]
    names = {
        p["id"]: (p["name"], side)
        for side in ("away", "home")
        for p in start[side]["lineup"] + start[side]["pitchers"] + start[side]["bench"]
    }
    team = {side: start[side]["name"] for side in ("away", "home")}
    final = events[-1]
    records: list[dict] = [
        {
            "record": "game",
            "game": number,
            "day": game.get("day"),
            "away": team["away"],
            "home": team["home"],
            "seed": seed,
            "runs": {"away": final["away"], "home": final["home"]},
            "line": final.get("line"),
        }
    ]
    by_event = {d["event"]: d for d in trail}
    setup = by_event.get(0, {}).get("setup", {})
    for side in ("away", "home"):
        if side in setup:
            records.append({"record": "setup", "game": number, "side": side, **setup[side]})
    plays = iter([p for _, ps in game_log(events).plays() for p in ps if p.kind == "pa"])
    cards: dict[tuple[str, str], dict] = {}
    score = {"away": 0, "home": 0}
    n = 0
    for at, e in enumerate(events):
        offense = "away" if e.get("half") == "top" else "home"
        if e["type"] in DECISIONS:
            records.append(_decision(e, number, score, by_event.get(at, {}), names, setup))
            score[offense] += len(e.get("scored", []))
            continue
        if e["type"] != "pa":
            continue
        n += 1
        play, detail = next(plays), by_event[at]
        before = dict(score)
        score[offense] += len(e["scored"])
        ab = play.at_bat
        batter_card = cards.setdefault(("batter", e["batter"]), {})
        batter_card[f"vs_{ab.throws}HP"] = detail["batter_rates"]
        pitcher_card = cards.setdefault(("pitcher", e["pitcher"]), {})
        pitcher_card[f"vs_{ab.bats}HB"] = detail["pitcher_rates"]
        records.append(
            {
                "record": "pa",
                "game": number,
                "pa": n,
                "inning": e["inning"],
                "half": e["half"],
                "outs_before": e["outs_before"],
                "bases_before": e["bases_before"],
                "score_before": before,
                "batting": team[offense],
                "batter": {"id": e["batter"], "name": ab.batter, "bats": ab.bats},
                "position": e.get("position"),
                "pitcher": {
                    "id": e["pitcher"],
                    "name": ab.pitcher,
                    "throws": ab.throws,
                    "pitches_before": detail["pitches_before"],
                    "batters_faced": detail["batters_faced"],
                    "fatigue": round(detail["fatigue"], 4),
                },
                "inputs": {
                    "batter_rates": detail["batter_rates"],
                    "pitcher_rates": detail["pitcher_rates"],
                    "league_rates": detail["league_rates"],
                    "park": detail["park"],
                    "defense": detail["defense"],
                    "alignment": e.get("alignment", "normal"),
                },
                "probabilities": {
                    **summarize(detail["probabilities"]),
                    "draw": {k: round(v, 5) for k, v in detail["probabilities"].items()},
                },
                "outcome": {
                    "result": e["result"],
                    "title": ab.title,
                    "description": play.description,
                    "batted_ball": e.get("batted_ball") or e.get("bip"),
                    "fielder": e.get("fielder"),
                    "direction": e.get("direction"),
                    "k_type": e.get("k_type"),
                    "bunt": bool(e.get("bunt")),
                    "ground_rule": bool(e.get("ground_rule")),
                    "outs": e["outs"],
                    "runs": len(e["scored"]),
                    "rbi": e["rbi"],
                    "runners": e["runners"],
                    "bases_after": e["bases_after"],
                },
                "count": e["count"],
                "pitch_count": e["pitches"],
                "pitches": {
                    "simulated": True,
                    "list": [asdict(q) | {"name": q.name} for q in ab.pitches],
                },
                "win_expectancy_home": round(play.home_we, 4),
                "wpa": round(play.wpa, 4),
                "leverage": play.li,
            }
        )
    for (role, pid), card in cards.items():
        name, side = names.get(pid, (pid, None))
        rec = {
            "record": "player",
            "game": number,
            "role": role,
            "id": pid,
            "name": name,
            "team": team.get(side),
            "card": card,
        }
        if talent is not None:
            t = talent.players.get(f"{role}:{pid}")
            if t is not None:
                rec["talent"] = {"index": t.index, "layers": t.layers}
            rec["real"] = talent.lines.get(f"{role}:{pid}", {})
        records.append(rec)
    return records


CSV_FIELDS = [
    "game", "pa", "inning", "half", "outs", "bases", "batting", "batter", "batter_id", "bats",
    "pitcher", "pitcher_id", "throws", "pitcher_pitches", "fatigue",
    "p_K", "p_BB", "p_HBP", "p_HR", "p_1B", "p_2B", "p_3B", "p_OUT",
    "p_GB", "p_LD", "p_FB", "p_PU",
    "result", "batted_ball", "fielder", "pitches", "count", "runs", "wpa", "leverage",
    "bat_k", "bat_bb", "bat_hr", "pit_k", "pit_bb", "pit_hr", "lg_k", "lg_bb", "lg_hr",
    "bat_real_PA", "bat_real_K%", "bat_real_BB%", "bat_real_HR%", "bat_real_AVG",
    "bat_real_OBP", "bat_real_SLG",
    "pit_real_PA", "pit_real_K%", "pit_real_BB%", "pit_real_HR%", "pit_real_AVG",
    "pit_real_OBP", "pit_real_SLG",
]  # fmt: skip


def csv_row(pa: dict, talent=None) -> dict:
    by_result = pa["probabilities"]["by_result"]
    by_ball = pa["probabilities"]["by_batted_ball"]
    i = pa["inputs"]
    row = {
        "game": pa["game"],
        "pa": pa["pa"],
        "inning": pa["inning"],
        "half": pa["half"],
        "outs": pa["outs_before"],
        "bases": "".join(str(b + 1) if r else "_" for b, r in enumerate(pa["bases_before"])),
        "batting": pa["batting"],
        "batter": pa["batter"]["name"],
        "batter_id": pa["batter"]["id"],
        "bats": pa["batter"]["bats"],
        "pitcher": pa["pitcher"]["name"],
        "pitcher_id": pa["pitcher"]["id"],
        "throws": pa["pitcher"]["throws"],
        "pitcher_pitches": pa["pitcher"]["pitches_before"],
        "fatigue": pa["pitcher"]["fatigue"],
        **{f"p_{k}": round(by_result.get(k, 0.0), 4) for k in ("K", "BB", "HBP", *HITS, "OUT")},
        **{f"p_{k}": round(by_ball.get(k, 0.0), 4) for k in ("GB", "LD", "FB", "PU")},
        "result": pa["outcome"]["result"],
        "batted_ball": pa["outcome"]["batted_ball"],
        "fielder": pa["outcome"]["fielder"],
        "pitches": pa["pitch_count"],
        "count": "-".join(map(str, pa["count"])),
        "runs": pa["outcome"]["runs"],
        "wpa": pa["wpa"],
        "leverage": pa["leverage"],
    }
    for prefix, rates in (
        ("bat", "batter_rates"),
        ("pit", "pitcher_rates"),
        ("lg", "league_rates"),
    ):
        for k in ("k", "bb", "hr"):
            row[f"{prefix}_{k}"] = round(i[rates][k], 4)
    if talent is not None:
        for prefix, role, opp in (
            ("bat", "batter", f"vs_{pa['pitcher']['throws']}"),
            ("pit", "pitcher", f"vs_{pa['batter']['bats']}"),
        ):
            line = talent.lines.get(f"{role}:{pa[role]['id']}", {})
            line = line.get(AUDIT_REAL_SEASON, {}).get(opp, {})
            for k in ("PA", "K%", "BB%", "HR%", "AVG", "OBP", "SLG"):
                row[f"{prefix}_real_{k}"] = line.get(k)
    return row


def write_audit(games: list[dict], out: Path, seed: int | None = None, talent=None) -> Path:
    """Write game-NN.jsonl per game and pas.csv for the week under `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, game in enumerate(games, 1):
        records = game_records(game, n, seed, talent)
        with open(out / f"game-{n:02d}.jsonl", "w") as fh:
            for r in records:
                fh.write(json.dumps(r, default=float) + "\n")
        rows += [csv_row(r, talent) for r in records if r["record"] == "pa"]
    with open(out / "pas.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return out
