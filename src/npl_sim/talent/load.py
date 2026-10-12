"""The talent blend for one sim week, read from the pipeline's clean tables
(data/clean/<season>/week-NN/pitches.parquet and asof-DATE/ leaderboards),
and applied to a week's lineup cards."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

import pandas as pd

from npl_sim.engine.game import Hitter, Pitcher, TeamCard
from npl_sim.engine.workload import WeekTeam
from npl_sim.talent import blend as tb
from npl_sim.talent.env import League, default_league, league
from npl_sim.talent.events import pitch_flags, plate_appearances

READ_COLUMNS = [
    "game_date", "batter", "pitcher", "stand", "p_throws", "events", "bb_type", "description",
    "zone", "launch_speed", "launch_speed_angle", "estimated_ba_using_speedangle",
    "pitch_type", "release_speed",
]  # fmt: skip
PRIOR_SEASONS = 2
# FanGraphs pitch codes that differ from Statcast's.
FG_PITCH = {"FA": "FF", "CS": "CU"}
SPRINT_AVERAGE, SPRINT_SD = 27.0, 1.5  # ft/s


@dataclass
class WeekTalent:
    """Everything the sim needs from the blend for one week."""

    league: League
    players: dict[str, tb.Talent] = field(default_factory=dict)
    speed: dict[str, float] = field(default_factory=dict)

    def hitter(self, h: Hitter) -> Hitter:
        if h.id.startswith("AAA-"):
            return h
        card = tb.hitter_card(h, self.players.get(f"batter:{h.id}"), self.league)
        return replace(card, speed=self.speed.get(h.id, h.speed))

    def pitcher(self, p: Pitcher) -> Pitcher:
        if p.id.startswith("AAA-"):
            return p
        return tb.pitcher_card(p, self.players.get(f"pitcher:{p.id}"), self.league)

    def card(self, c: TeamCard) -> TeamCard:
        return replace(
            c,
            lineup=[(self.hitter(h), pos) for h, pos in c.lineup],
            starter=self.pitcher(c.starter),
            bullpen=[self.pitcher(p) for p in c.bullpen],
            defensive_subs=tuple(
                replace(d, player=self.hitter(d.player)) for d in c.defensive_subs
            ),
        )

    def team(self, t: WeekTeam) -> WeekTeam:
        return replace(
            t,
            card=self.card(t.card),
            lineups={vs: [(self.hitter(h), pos) for h, pos in o] for vs, o in t.lineups.items()},
            rotation=[self.pitcher(p) for p in t.rotation],
            bench=[self.hitter(h) for h in t.bench],
            slot_cards={k: self.card(c) for k, c in t.slot_cards.items()},
        )


def read_pitches(folder: Path) -> pd.DataFrame:
    path = folder / "pitches.parquet"
    if not path.exists():
        return pd.DataFrame(columns=READ_COLUMNS)
    frame = pd.read_parquet(path)
    return frame[[c for c in READ_COLUMNS if c in frame.columns]]


def _weeks(season_dir: Path) -> list[Path]:
    return sorted(p for p in season_dir.glob("week-*") if p.is_dir())


def load_week_talent(clean_root: Path, season: int, week: int, hands=None) -> WeekTalent:
    """The blend for `week` of `season`: the week's pitches against this
    season before it and the two seasons before that. `hands` maps id ->
    batting hand for platoon regression."""
    root = Path(clean_root)
    this = root / str(season)
    week_dir = this / f"week-{week:02d}"
    if not week_dir.exists():
        raise FileNotFoundError(f"no clean data for {season} week {week} under {root}")
    week_pitches = read_pitches(week_dir)
    before = [read_pitches(p) for p in _weeks(this) if _number(p) < week]
    to_date = pd.concat(before, ignore_index=True) if before else week_pitches.iloc[0:0]
    prior = []
    for back in range(1, PRIOR_SEASONS + 1):
        frames = [read_pitches(p) for p in _weeks(root / str(season - back))]
        if frames:
            prior.append((back, pd.concat(frames, ignore_index=True)))
    stuff_board, sprint_board = _boards(this, _last_day(week_pitches))
    return build_week_talent(week_pitches, to_date, prior, stuff_board, sprint_board, hands)


def build_week_talent(
    week_pitches: pd.DataFrame,
    to_date: pd.DataFrame,
    prior: list[tuple[int, pd.DataFrame]],
    stuff_board: pd.DataFrame | None = None,
    sprint_board: pd.DataFrame | None = None,
    hands: dict[str, str] | None = None,
) -> WeekTalent:
    """The blend from pitch tables: the sim week's, this season's before it,
    and [(seasons back, pitches)] for prior seasons."""
    week_pa = plate_appearances(week_pitches)
    to_date_pa = plate_appearances(to_date)
    prior_pa = [(back, plate_appearances(p), p) for back, p in prior]
    last = next((pa for back, pa, _ in prior_pa if back == 1), None)
    current = league(pd.concat([to_date_pa, week_pa], ignore_index=True), prior=last)
    seasons = [tb.Season(0, to_date_pa, current, pitch_flags(to_date))]
    for back, pa, pitches in prior_pa:
        seasons.append(tb.Season(back, pa, league(pa), pitch_flags(pitches)))
    week = tb.Season(0, week_pa, current, pitch_flags(week_pitches))
    season_pitches = pd.concat([to_date, week_pitches], ignore_index=True)
    stuff = stuff_by_pitcher(stuff_board, season_pitches) if stuff_board is not None else {}
    players = tb.blend(seasons, week, stuff=stuff, players=hands)
    speed = sprint_speed(sprint_board) if sprint_board is not None else {}
    return WeekTalent(league=current, players=players, speed=speed)


def stuff_by_pitcher(board: pd.DataFrame, pitches: pd.DataFrame) -> dict[str, float]:
    """Each pitcher's Stuff+ (FanGraphs, season to date), per pitch type
    regressed by how many of that pitch he's thrown this season."""
    per_pitch = {
        m.group(1): c for c in board.columns if (m := re.fullmatch(r"sp_s_([A-Za-z]{2})", c))
    }
    counts = (
        pitches.groupby(["pitcher", "pitch_type"]).size()
        if not pitches.empty and "pitch_type" in pitches
        else pd.Series(dtype=int)
    )
    out = {}
    for _, row in board.iterrows():
        pid = row.get("mlbam_id")
        if pd.isna(pid):
            continue
        pid = int(pid)
        mine = counts.loc[pid].to_dict() if pid in counts.index.get_level_values(0) else {}
        values = {FG_PITCH.get(code, code): _num(row[col]) for code, col in per_pitch.items()}
        s = tb.stuff_plus(values, mine, _num(row.get("sp_stuff")))
        if s is not None:
            out[str(pid)] = s
    return out


def sprint_speed(board: pd.DataFrame) -> dict[str, float]:
    """Sprint speed (ft/s) as the engine's runner speed, about -2..+2."""
    col = next((c for c in ("sprint_speed", "hp_to_1b_sprint_speed") if c in board), None)
    if col is None:
        return {}
    out = {}
    for pid, v in zip(board["mlbam_id"], board[col], strict=True):
        v = _num(v)
        if pd.notna(pid) and v is not None:
            out[str(int(pid))] = max(-2.0, min(2.0, (v - SPRINT_AVERAGE) / SPRINT_SD))
    return out


def _boards(season_dir: Path, through: date | None):
    """The latest as-of leaderboards on or before `through` (else the latest)."""
    dirs = sorted(season_dir.glob("asof-*"))
    if through is not None:
        on_time = [d for d in dirs if d.name.removeprefix("asof-") <= through.isoformat()]
        dirs = on_time or dirs
    if not dirs:
        return None, None
    d = dirs[-1]

    def read(name):
        p = d / f"{name}.parquet"
        return pd.read_parquet(p) if p.exists() else None

    return read("fg_pitching"), read("sprint_speed")


def _last_day(pitches: pd.DataFrame) -> date | None:
    if pitches.empty or "game_date" not in pitches:
        return None
    return pd.to_datetime(pitches["game_date"]).max().date()


def _number(week_dir: Path) -> int:
    return int(week_dir.name.removeprefix("week-"))


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


__all__ = ["WeekTalent", "build_week_talent", "default_league", "load_week_talent"]
