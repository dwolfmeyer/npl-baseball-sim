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
from npl_sim.talent import fielding as tf
from npl_sim.talent.env import League, default_league, league
from npl_sim.talent.events import pitch_flags, plate_appearances

READ_COLUMNS = [
    "game_date", "batter", "pitcher", "stand", "p_throws", "events", "bb_type", "description",
    "zone", "launch_speed", "launch_speed_angle", "estimated_ba_using_speedangle",
    "pitch_type", "release_speed", "home_team", *(f"fielder_{n}" for n in range(2, 10)),
]  # fmt: skip
PRIOR_SEASONS = 2
# FanGraphs' per-pitch Stuff+ columns (sp_s_CH, _CU, _FC, _FF, _FO, _FS, _KC,
# _SI, _SL) use Statcast codes but fold sweepers and slurves into SL and CU;
# fold the pitch data the same way so counts line up.
STATCAST_TO_FG = {"ST": "SL", "SV": "CU", "FA": "FF", "CS": "CU"}
SPRINT_AVERAGE, SPRINT_SD = 27.0, 1.5  # ft/s
ARSENAL_MIN = 150  # pitches this season before his mix replaces last season's


@dataclass
class WeekTalent:
    """Everything the sim needs from the blend for one week."""

    league: League
    players: dict[str, tb.Talent] = field(default_factory=dict)
    speed: dict[str, float] = field(default_factory=dict)
    # Fielding (talent/fielding.py): primary-position range rating, catcher
    # framing (runs per 600 BIP) and arm by player id.
    range: dict[str, tuple[str, float]] = field(default_factory=dict)
    framing: dict[str, float] = field(default_factory=dict)
    arm: dict[str, float] = field(default_factory=dict)
    # Pitch mix by pitcher id: ((Statcast type, share, mph), ...).
    arsenal: dict[str, tuple] = field(default_factory=dict)

    def hitter(self, h: Hitter) -> Hitter:
        if h.id.startswith("AAA-"):
            return h
        card = tb.hitter_card(h, self.players.get(f"batter:{h.id}"), self.league)
        rating, framing = self.range.get(h.id), self.framing.get(h.id)
        gloves = h.positions
        if gloves or rating is not None or framing is not None:
            gloves = tf.positions(h.positions, rating, framing)
        return replace(
            card,
            speed=self.speed.get(h.id, h.speed),
            positions=gloves,
            arm=self.arm.get(h.id, h.arm),
        )

    def pitcher(self, p: Pitcher) -> Pitcher:
        if p.id.startswith("AAA-"):
            return p
        card = tb.pitcher_card(p, self.players.get(f"pitcher:{p.id}"), self.league)
        return replace(card, arsenal=self.arsenal.get(p.id, card.arsenal))

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
    boards = _boards(this, _last_day(week_pitches))
    last_boards = _boards(root / str(season - 1), None)
    talent = build_week_talent(
        week_pitches,
        to_date,
        prior,
        boards.get("fg_pitching"),
        boards.get("sprint_speed"),
        hands,
        parks=tf.park_factors(boards.get("park_factors")),
    )
    season_pitches = pd.concat([to_date, week_pitches], ignore_index=True)
    last = next((p for back, p in prior if back == 1), None)
    add_fielding(talent, boards, season_pitches, last_boards, last)
    return talent


def add_fielding(talent, boards: dict, pitches, last_boards: dict, last_pitches) -> None:
    """Range, framing and arms from this season's and last season's boards."""
    on_field = tf.bip_on_field(pitches)
    last_field = tf.bip_on_field(last_pitches) if last_pitches is not None else on_field.iloc[0:0]
    talent.range = tf.range_ratings(
        [(0, boards.get("oaa"), on_field), (1, last_boards.get("oaa"), last_field)]
    )
    talent.framing = tf.framing_ratings(
        [(0, boards.get("framing"), on_field), (1, last_boards.get("framing"), last_field)]
    )
    talent.arm = tf.arms(boards.get("arm_strength"), boards.get("poptime"))


def build_week_talent(
    week_pitches: pd.DataFrame,
    to_date: pd.DataFrame,
    prior: list[tuple[int, pd.DataFrame]],
    stuff_board: pd.DataFrame | None = None,
    sprint_board: pd.DataFrame | None = None,
    hands: dict[str, str] | None = None,
    parks: dict | None = None,
) -> WeekTalent:
    """The blend from pitch tables: the sim week's, this season's before it,
    and [(seasons back, pitches)] for prior seasons. `parks` (fielding.park_factors)
    neutralizes every season's record for the parks it was played in."""
    week_pa = plate_appearances(week_pitches)
    to_date_pa = plate_appearances(to_date)
    prior_pa = [(back, plate_appearances(p), p) for back, p in prior]
    last = next((pa for back, pa, _ in prior_pa if back == 1), None)
    current = league(pd.concat([to_date_pa, week_pa], ignore_index=True), prior=last)
    seasons = [tb.Season(0, to_date_pa, current, pitch_flags(to_date), parks)]
    for back, pa, pitches in prior_pa:
        seasons.append(tb.Season(back, pa, league(pa), pitch_flags(pitches), parks))
    week = tb.Season(0, week_pa, current, pitch_flags(week_pitches), parks)
    season_pitches = pd.concat([to_date, week_pitches], ignore_index=True)
    stuff = stuff_by_pitcher(stuff_board, season_pitches) if stuff_board is not None else {}
    players = tb.blend(seasons, week, stuff=stuff, players=hands)
    speed = sprint_speed(sprint_board) if sprint_board is not None else {}
    last_pitches = next((p for back, p in prior if back == 1), None)
    return WeekTalent(
        league=current,
        players=players,
        speed=speed,
        arsenal=arsenals(season_pitches, last_pitches),
    )


def arsenals(
    season: pd.DataFrame, last: pd.DataFrame | None = None, min_pitches: int = ARSENAL_MIN
) -> dict[str, tuple]:
    """Each pitcher's pitch mix this season (last season's when he has thrown
    fewer than `min_pitches` this year): share and average speed by pitch type,
    dropping pitches under 2% of his total."""
    out: dict[str, tuple] = {}
    for frame in (last, season):  # this season overwrites last
        if frame is None or frame.empty or "pitch_type" not in frame:
            continue
        typed = frame[frame["pitch_type"].notna() & frame["release_speed"].notna()]
        for pid, mine in typed.groupby("pitcher"):
            if frame is season and len(mine) < min_pitches and str(pid) in out:
                continue
            stats = mine.groupby("pitch_type")["release_speed"].agg(["size", "mean"])
            share = stats["size"] / stats["size"].sum()
            keep = stats[share >= 0.02].sort_values("size", ascending=False)
            total = keep["size"].sum()
            out[str(pid)] = tuple(
                (str(code), round(float(r["size"] / total), 3), round(float(r["mean"]), 1))
                for code, r in keep.iterrows()
            )
    return out


def stuff_by_pitcher(board: pd.DataFrame, pitches: pd.DataFrame) -> dict[str, float]:
    """Each pitcher's Stuff+ (FanGraphs, season to date), per pitch type
    regressed by how many of that pitch he's thrown this season."""
    per_pitch = {
        m.group(1): c for c in board.columns if (m := re.fullmatch(r"sp_s_([A-Za-z]{2})", c))
    }
    if not pitches.empty and "pitch_type" in pitches:
        kinds = pitches["pitch_type"].replace(STATCAST_TO_FG)
        counts = pitches.assign(pitch_type=kinds).groupby(["pitcher", "pitch_type"]).size()
    else:
        counts = pd.Series(dtype=int)
    out = {}
    for _, row in board.iterrows():
        pid = row.get("mlbam_id")
        if pd.isna(pid):
            continue
        pid = int(pid)
        mine = counts.loc[pid].to_dict() if pid in counts.index.get_level_values(0) else {}
        values = {code: _num(row[col]) for code, col in per_pitch.items()}
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


BOARDS = ("fg_pitching", "sprint_speed", "oaa", "framing", "poptime", "arm_strength",
          "park_factors")  # fmt: skip


def _boards(season_dir: Path, through: date | None) -> dict[str, pd.DataFrame]:
    """The latest as-of leaderboards on or before `through` (else the latest)."""
    dirs = sorted(season_dir.glob("asof-*"))
    if through is not None:
        on_time = [d for d in dirs if d.name.removeprefix("asof-") <= through.isoformat()]
        dirs = on_time or dirs
    if not dirs:
        return {}
    d = dirs[-1]
    return {n: pd.read_parquet(d / f"{n}.parquet") for n in BOARDS if (d / f"{n}.parquet").exists()}


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
