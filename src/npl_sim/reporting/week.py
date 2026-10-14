"""A sim week as HTML: standings, each day's games with who started, who sat
and who filled in, and a Gameday page per game (`write_week`)."""

from __future__ import annotations

from html import escape
from pathlib import Path

from npl_sim.engine.box import box_score
from npl_sim.engine.playlog import game_log
from npl_sim.engine.workload import WeekResult
from npl_sim.reporting.gameday import CSS, render_game

DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

WEEK_CSS = """
.games{display:grid;gap:12px}.game{border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.game .score{display:flex;justify-content:space-between;gap:12px;font-weight:600}
.game .score a{color:var(--accent);font-weight:400;white-space:nowrap}
.game ul{margin:6px 0 0;padding-left:18px;color:var(--muted)}.game li{margin:2px 0}
.win{font-weight:700}
.recap{border-top:1px solid var(--line);padding:6px 0}.recap p{margin:3px 0}
.filter{margin:0 0 12px;display:flex;align-items:center;gap:8px}
"""


# Limits the game list to one team's games (page works without it: all show).
FILTER_JS = """
const sel=document.getElementById('team'),games=[...document.querySelectorAll('.game')];
function apply(){const t=sel.value;let n=0;
for(const g of games){const show=!t||g.dataset.away===t||g.dataset.home===t;
g.hidden=!show;n+=show}
document.getElementById('count').textContent=t?n+' of '+games.length+' games':'';
const u=new URL(location);t?u.searchParams.set('team',t):u.searchParams.delete('team');
history.replaceState(null,'',u)}
sel.onchange=apply;
sel.value=new URLSearchParams(location.search).get('team')||'';
if(sel.value!==(new URLSearchParams(location.search).get('team')||''))sel.value='';
apply();
"""


def _names(events: list[dict]) -> dict[str, str]:
    start = events[0]
    names = {}
    for side in ("away", "home"):
        t = start[side]
        for p in t["lineup"] + t["pitchers"] + t["bench"]:
            names[p["id"]] = p["name"]
    return names


def standings(result: WeekResult) -> list[tuple[str, int, int, int, int]]:
    """(team, W, L, runs scored, runs allowed), best record first."""
    rows: dict[str, list[int]] = {}
    for g in result.games:
        final = g["events"][-1]
        for team, side, other in ((g["away"], "away", "home"), (g["home"], "home", "away")):
            r = rows.setdefault(team, [0, 0, 0, 0])
            won = final[side] > final[other]
            r[0 if won else 1] += 1
            r[2] += final[side]
            r[3] += final[other]
    return sorted(
        ((t, *r) for t, r in rows.items()), key=lambda r: (-r[1] / max(1, r[1] + r[2]), r[4] - r[3])
    )


def _notes(g: dict, team: str, names: dict[str, str]) -> list[str]:
    m = g["moves"][team]

    def name(pid: str) -> str:
        return names.get(pid, pid)

    starter = name(m.starter) + (" (bullpen game)" if m.bullpen_game else "")
    notes = [f"{team}: {starter} started"]
    for r in m.replaced:
        who = "a (AAA) stand-in" if r["standin"] else name(r["into"])
        notes.append(f"{name(r['out'])} out of PA or didn't play: {who} at {r['position']}")
    for m_ in getattr(m, "shifted", []):
        notes.append(f"{name(m_['player'])} moved from {m_['from']} to {m_['to']} to cover")
    if m.unavailable:
        notes.append("Resting or out of innings: " + ", ".join(name(p) for p in m.unavailable))
    if m.standin_pitcher:
        notes.append("Whole pen unavailable: a (AAA) arm relieved")
    return notes


def render_week(
    result: WeekResult, title: str, subtitle: str = "", link=None, names: dict | None = None
) -> str:
    """The week's page. `link(i)` gives the href of game i's page, or None;
    `names` maps player ids to names for players who didn't appear in a game."""
    table = "".join(
        f"<tr><td>{escape(t)}</td><td>{w}</td><td>{lo}</td><td>{rs}</td><td>{ra}</td></tr>"
        for t, w, lo, rs, ra in standings(result)
    )
    from npl_sim.reporting.recap import week_recaps  # recap reads this module's helpers

    recaps = "".join(
        f'<div class="recap"><p><strong>{escape(r.headline)}</strong></p>'
        + "".join(f"<p>{escape(line)}</p>" for line in r.lines)
        + "</div>"
        for r in week_recaps(result).values()
    )
    teams = sorted({t for g in result.games for t in (g["away"], g["home"])})
    options = "".join(f'<option value="{escape(t)}">{escape(t)}</option>' for t in teams)
    games = []
    for i, g in enumerate(result.games):
        final = g["events"][-1]
        who = {**(names or {}), **_names(g["events"])}
        a, h = final["away"], final["home"]
        away = f'<span class="{"win" if a > h else ""}">{escape(g["away"])} {a}</span>'
        home = f'<span class="{"win" if h > a else ""}">{escape(g["home"])} {h}</span>'
        href = link(i) if link else None
        box = f'<a href="{escape(href)}">Box score</a>' if href else ""
        notes = _notes(g, g["away"], who) + _notes(g, g["home"], who)
        items = "".join(f"<li>{escape(n)}</li>" for n in notes)
        games.append(
            f'<div class="game" data-away="{escape(g["away"])}" data-home="{escape(g["home"])}">'
            f'<div class="score"><span>{DAY_NAMES[g["day"] % 7]}: '
            f"{away} at {home}</span>{box}</div><ul>{items}</ul></div>"
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>{CSS}{WEEK_CSS}</style></head><body><main>
<h1>{escape(title)}</h1><p class="sub">{escape(subtitle)}</p>
<h2>Standings</h2><div class="wrap"><table><tr><th>Team</th><th>W</th><th>L</th><th>RS</th>
<th>RA</th></tr>{table}</table></div>
<h2 style="margin-top:20px">Recaps</h2><div class="recaps">{recaps}</div>
<h2 style="margin-top:20px">Games</h2>
<label class="filter">Team <select id="team"><option value="">All teams</option>{options}</select>
<span id="count" class="sub"></span></label>
<div class="games">{"".join(games)}</div>
</main><script>{FILTER_JS}</script></body></html>"""


def write_week(
    result: WeekResult, out: Path, title: str, subtitle: str = "", names: dict | None = None
) -> Path:
    """index.html plus game-N.html for each game; returns the index path."""
    out.mkdir(parents=True, exist_ok=True)
    for i, g in enumerate(result.games):
        box = box_score(g["events"])
        day = DAY_NAMES[g["day"] % 7]
        page = render_game(box, game_log(g["events"]), subtitle=f"{title}, {day}")
        (out / f"game-{i + 1}.html").write_text(page)
    index = out / "index.html"
    index.write_text(
        render_week(result, title, subtitle, link=lambda i: f"game-{i + 1}.html", names=names)
    )
    return index
