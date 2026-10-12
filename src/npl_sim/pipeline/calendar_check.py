"""Check config/calendar.yaml against the MLB Stats API schedule."""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta

from npl_sim.schedule.calendar import SeasonCalendar

# The first date with at least this many games is Opening Day; any earlier
# regular-season dates are one-off openers.
FULL_SLATE = 8


def games_per_date(raw_schedule: dict) -> Counter[date]:
    counts: Counter[date] = Counter()
    for day in raw_schedule.get("dates", []):
        for g in day.get("games", []):
            if g.get("gameType", "R") == "R":
                counts[date.fromisoformat(day["date"])] += 1
    return counts


def derive_calendar(raw_schedule: dict) -> SeasonCalendar:
    counts = games_per_date(raw_schedule)
    if not counts:
        raise ValueError("schedule has no regular-season games")
    days = sorted(counts)
    opening = next(d for d in days if counts[d] >= FULL_SLATE)
    final = days[-1]
    off = []
    d = opening
    while d <= final:
        if counts[d] == 0:
            off.append(d)
        d += timedelta(days=1)
    return SeasonCalendar(
        opening_day=opening,
        final_day=final,
        off_days=off,
        openers=[d for d in days if d < opening],
    )


def compare(config: SeasonCalendar, actual: SeasonCalendar) -> list[str]:
    problems = []
    for field in ("opening_day", "final_day"):
        if getattr(config, field) != getattr(actual, field):
            problems.append(
                f"{field}: config {getattr(config, field)}, MLB {getattr(actual, field)}"
            )
    for field in ("off_days", "openers"):
        want, got = sorted(getattr(config, field)), sorted(getattr(actual, field))
        if want != got:
            problems.append(
                f"{field}: config {[str(d) for d in want]}, MLB {[str(d) for d in got]}"
            )
    return problems
