"""Real MLB stat lines behind the talent blend, for the sim audit log: each
player's plate appearances by season and by the opponent's hand, as counts
and the usual rates. Built from the same Statcast PA tables the blend reads."""

from __future__ import annotations

import pandas as pd

HITS = ("1B", "2B", "3B", "HR")
COUNTED = ("K", "BB", "HBP", *HITS)
BIP = ("GB", "LD", "FB", "PU")


def stat_lines(seasons: list[tuple[str, pd.DataFrame]]) -> dict[str, dict]:
    """{"batter:<id>" or "pitcher:<id>": {label: {"all"|"vs_L"|"vs_R": line}}}
    from [(label, plate_appearances table)], e.g. ("season to date", ...)."""
    out: dict[str, dict] = {}
    for label, pa in seasons:
        if pa is None or pa.empty:
            continue
        for role, me, opp in (("batter", "batter", "p_throws"), ("pitcher", "pitcher", "stand")):
            for key, frame in (
                ("all", pa),
                ("vs_L", pa[pa[opp] == "L"]),
                ("vs_R", pa[pa[opp] == "R"]),
            ):
                for pid, line in _lines(frame, me).items():
                    out.setdefault(f"{role}:{pid}", {}).setdefault(label, {})[key] = line
    return out


def _lines(pa: pd.DataFrame, me: str) -> dict[str, dict]:
    if pa.empty:
        return {}
    counts = pd.crosstab(pa[me], pa["outcome"])
    bip = pd.crosstab(pa[me], pa["bip"]) if pa["bip"].notna().any() else None
    contact = pa.groupby(me)[["xba", "barrel", "hard"]].mean()
    out = {}
    for pid, row in counts.iterrows():
        n = {k: int(row.get(k, 0)) for k in (*COUNTED, "OUT")}
        pas = sum(n.values())
        hits = sum(n[h] for h in HITS)
        ab = pas - n["BB"] - n["HBP"]  # Statcast sacrifices count as outs here
        tb = n["1B"] + 2 * n["2B"] + 3 * n["3B"] + 4 * n["HR"]
        line = {"PA": pas, **{k: n[k] for k in COUNTED}, "H": hits}
        line |= {
            "AVG": _r(hits, ab),
            "OBP": _r(hits + n["BB"] + n["HBP"], pas),
            "SLG": _r(tb, ab),
            "K%": _r(n["K"], pas),
            "BB%": _r(n["BB"], pas),
            "HR%": _r(n["HR"], pas),
        }
        if bip is not None and pid in bip.index:
            b = {k: int(bip.loc[pid].get(k, 0)) for k in BIP}
            total = sum(b.values())
            line |= {f"{k}%": _r(b[k], total) for k in BIP}
        c = contact.loc[pid]
        for col, name in (("xba", "xBA"), ("barrel", "Barrel%"), ("hard", "HardHit%")):
            if pd.notna(c[col]):
                line[name] = round(float(c[col]), 3)
        out[str(pid)] = line
    return out


def _r(num: float, den: float) -> float | None:
    return round(num / den, 3) if den else None
