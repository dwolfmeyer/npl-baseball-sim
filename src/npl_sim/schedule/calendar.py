"""MLB season calendar split into Monday-Sunday sim weeks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import yaml
from pydantic import BaseModel

DEFAULT_CALENDAR = Path(__file__).resolve().parents[3] / "config" / "calendar.yaml"


class SeasonCalendar(BaseModel):
    opening_day: date
    final_day: date
    off_days: list[date] = []
    # Regular-season games before opening_day (Opening Night, international
    # series). No NPL games those days; their MLB stats count toward week 1.
    openers: list[date] = []


@dataclass(frozen=True)
class Week:
    number: int  # 1-based
    start: date  # Monday
    game_days: tuple[date, ...]

    @property
    def end(self) -> date:
        return self.start + timedelta(days=6)


def load_calendar(season: int, path: Path | str = DEFAULT_CALENDAR) -> SeasonCalendar:
    with open(path) as fh:
        raw = yaml.safe_load(fh)["seasons"]
    if season not in raw:
        raise LookupError(f"no calendar for season {season} in {path}")
    return SeasonCalendar.model_validate(raw[season])


def sim_weeks(cal: SeasonCalendar) -> list[Week]:
    off = set(cal.off_days)
    monday = cal.opening_day - timedelta(days=cal.opening_day.weekday())
    weeks = []
    while monday <= cal.final_day:
        days = tuple(
            d
            for d in (monday + timedelta(days=i) for i in range(7))
            if cal.opening_day <= d <= cal.final_day and d not in off
        )
        weeks.append(Week(number=len(weeks) + 1, start=monday, game_days=days))
        monday += timedelta(days=7)
    return weeks


def week_for_date(weeks: list[Week], day: date) -> Week | None:
    """The sim week whose MLB stats include `day`.

    Opener games before week 1 fold into week 1; days after the last week
    belong to no week.
    """
    if not weeks:
        return None
    if day < weeks[0].start:
        return weeks[0]
    for week in weeks:
        if week.start <= day <= week.end:
            return week
    return None
