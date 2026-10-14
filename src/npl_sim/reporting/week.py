"""A sim week as HTML in three columns: the standings, a recap per team with its hitting
and pitching, and each day's games with who started, who sat and who filled in, each
opening its full box score (a Gameday page per game, `write_week`)."""

from __future__ import annotations

from html import escape
from pathlib import Path

from npl_sim.engine.box import box_score
from npl_sim.engine.playlog import game_log
from npl_sim.engine.workload import WeekResult
from npl_sim.reporting.gameday import CSS, render_game

DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

WEEK_CSS = """
main.week{max-width:1560px}
.game{border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin:0 0 8px}
.game.go{cursor:pointer}.game.go:hover{background:var(--panel)}
.game .score{display:flex;gap:8px;font-weight:600;align-items:baseline}
.game .day{color:var(--muted);font-weight:400;width:32px;flex:none}
.game .teams-line{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:2px 10px;flex:1}
.game .teams-line span{overflow-wrap:anywhere}.game .teams-line b{text-align:right}
.game .box{display:block;margin:4px 0 0 40px;font-size:12px}
.game details{margin-top:6px}.game summary{color:var(--muted);font-size:12px;cursor:pointer}
.game ul{margin:4px 0 0;padding-left:18px;color:var(--muted);font-size:12px}.game li{margin:2px 0}
.win{font-weight:700}.lose{font-weight:400;color:var(--muted)}
.recap{border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin:0 0 8px}
.recap p{margin:3px 0}
.col .recap table{margin:8px 0 0;table-layout:fixed}
.col .recap th,.col .recap td{text-align:center;padding:2px 0;font-size:10px}
.col .recap tr.mlb td{color:var(--muted)}
.col .recap th:first-child,.col .recap td:first-child{width:42px;text-align:left;color:var(--muted);
font-weight:600}
.bar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:0 0 12px}
.bar label{display:flex;align-items:center;gap:8px}
.cols{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;align-items:start}
.col{min-width:0;max-height:calc(100vh - 150px);overflow-y:auto;overflow-x:hidden;
padding-right:4px}
.col>h2{position:sticky;top:0;z-index:1;background:var(--bg);margin:0 0 8px;padding:4px 0;
border-bottom:2px solid var(--line);font-size:13px;letter-spacing:.06em;text-transform:uppercase}
.col table{table-layout:auto}.col th,.col td{white-space:normal;padding:3px 4px;font-size:12px}
.col td.num,.col th.num{white-space:nowrap}
.tabbar{display:none;gap:4px;margin:0 0 12px;border-bottom:1px solid var(--line)}
.tabbar button{background:none;border:0;color:var(--muted);font:inherit;padding:8px 14px;
cursor:pointer;border-bottom:2px solid transparent}
.tabbar button[aria-selected=true]{color:var(--fg);border-color:var(--accent);font-weight:600}
@media (max-width:1099px){
.tabbar{display:flex}.cols{display:block}.col{max-height:none;overflow:visible}
.col>h2{position:static}
.cols[data-show=results]>.col:not(#c-results),.cols[data-show=recaps]>.col:not(#c-recaps),
.cols[data-show=games]>.col:not(#c-games){display:none}}
"""


# Limits games and recaps to one team (the page works without JS: everything shows), and
# sends a click on a game to its full box score page.
FILTER_JS = """
const sel=document.getElementById('team'),cols=document.querySelector('.cols'),
games=[...document.querySelectorAll('.game')],recaps=[...document.querySelectorAll('.recap')];
function show(c){cols.dataset.show=c;
document.querySelectorAll('.tabbar button').forEach(
b=>b.setAttribute('aria-selected',b.dataset.c===c));}
document.querySelectorAll('.tabbar button').forEach(b=>b.onclick=()=>show(b.dataset.c));
for(const g of games){const a=g.querySelector('a.box');
if(a)g.onclick=e=>{if(!e.target.closest('summary,a'))location.href=a.href};}
function apply(){const t=sel.value;let n=0;
for(const g of games){const on=!t||g.dataset.away===t||g.dataset.home===t;g.hidden=!on;n+=on}
for(const r of recaps)r.hidden=!!t&&r.dataset.team!==t;
document.getElementById('count').textContent=t?n+' of '+games.length+' games':'';
const u=new URL(location);t?u.searchParams.set('team',t):u.searchParams.delete('team');
history.replaceState(null,'',u)}
sel.onchange=apply;
sel.value=new URLSearchParams(location.search).get('team')||'';
if(sel.value!==(new URLSearchParams(location.search).get('team')||''))sel.value='';
show('results');apply();
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


def _stat_table(label: str, sim: list[tuple[str, str]], mlb: list[tuple[str, str]]) -> str:
    """A hitting or pitching line for the week: the sim's, and the real MLB week's below it."""
    if not sim:
        return ""
    head = "".join(f"<th>{escape(k)}</th>" for k, _ in sim)

    def row(name: str, pairs: list[tuple[str, str]]) -> str:
        cells = "".join(f"<td>{escape(v)}</td>" for _, v in pairs)
        return f'<tr class="{name.lower()}"><td>{name}</td>{cells}</tr>'

    rows = row("Sim", sim) + (row("MLB", mlb) if mlb else "")
    return f"<table><tr><td>{label}</td>{head}</tr>{rows}</table>"


def render_week(
    result: WeekResult,
    title: str,
    subtitle: str = "",
    link=None,
    names: dict | None = None,
    real=None,
    real_label: str = "",
) -> str:
    """The week's page in three columns: results (the standings), recaps with each team's
    hitting and pitching, and the games, each opening its full box score page. `link(i)` gives
    the href of game i's page, or None; `names` maps player ids to names for players who
    didn't appear in a game. `real` (playing_time.RealStats) and `real_label` ("MLB 2026 week 27")
    add the real week's lines under each team's sim lines."""
    table = "".join(
        f'<tr><td>{escape(t)}</td><td class="num">{w}</td><td class="num">{lo}</td>'
        f'<td class="num">{rs}</td><td class="num">{ra}</td></tr>'
        for t, w, lo, rs, ra in standings(result)
    )
    from npl_sim.reporting.recap import week_recaps  # recap reads this module's helpers

    recaps = "".join(
        f'<div class="recap" data-team="{escape(team)}">'
        f"<p><strong>{escape(r.headline)}</strong></p>"
        + "".join(f"<p>{escape(line)}</p>" for line in r.lines)
        + _stat_table("Hitting", r.hitting, r.mlb_hitting)
        + _stat_table("Pitching", r.pitching, r.mlb_pitching)
        + "</div>"
        for team, r in week_recaps(result, names, real).items()
    )
    compare = (
        f'<p class="sub">MLB rows: the real {escape(real_label)} stats of the players who played '
        "for the team in the sim week.</p>"
        if real is not None and real_label
        else ""
    )
    teams = sorted({t for g in result.games for t in (g["away"], g["home"])})
    options = "".join(f'<option value="{escape(t)}">{escape(t)}</option>' for t in teams)
    games = []
    for i, g in enumerate(result.games):
        final = g["events"][-1]
        who = {**(names or {}), **_names(g["events"])}
        a, h = final["away"], final["home"]
        day = DAY_NAMES[g["day"] % 7]
        notes = _notes(g, g["away"], who) + _notes(g, g["home"], who)
        items = "".join(f"<li>{escape(n)}</li>" for n in notes)
        href = link(i) if link else None
        box = f'<a class="box" href="{escape(href)}">Box score</a>' if href else ""
        games.append(
            f'<div class="game{" go" if href else ""}" data-i="{i}" '
            f'data-away="{escape(g["away"])}" '
            f'data-home="{escape(g["home"])}"><div class="score"><span class="day">{day}</span>'
            f'<div class="teams-line"><span class="{"win" if a > h else "lose"}">'
            f'{escape(g["away"])}</span><b class="{"win" if a > h else "lose"}">{a}</b>'
            f'<span class="{"win" if h > a else "lose"}">{escape(g["home"])}</span>'
            f'<b class="{"win" if h > a else "lose"}">{h}</b></div></div>{box}'
            f"<details><summary>Lineup notes ({len(notes)})</summary><ul>{items}</ul></details>"
            "</div>"
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>{CSS}{WEEK_CSS}</style></head>
<body><main class="week">
<h1>{escape(title)}</h1><p class="sub">{escape(subtitle)}</p>
<div class="bar"><label>Team <select id="team"><option value="">All teams</option>{options}
</select></label><span id="count" class="sub"></span></div>
<div class="tabbar" role="tablist"><button role="tab" data-c="results">Results</button>
<button role="tab" data-c="recaps">Recaps</button>
<button role="tab" data-c="games">Games</button></div>
<div class="cols" data-show="results">
<section class="col" id="c-results"><h2>Results</h2>
<h3>Standings</h3><table><tr><th>Team</th><th class="num">W</th><th class="num">L</th>
<th class="num">RS</th><th class="num">RA</th></tr>{table}</table></section>
<section class="col" id="c-recaps"><h2>Recaps</h2>{compare}{recaps}</section>
<section class="col" id="c-games"><h2>Games</h2>{"".join(games)}</section>
</div>
</main><script>{FILTER_JS}</script></body></html>"""


def write_week(
    result: WeekResult,
    out: Path,
    title: str,
    subtitle: str = "",
    names: dict | None = None,
    real=None,
    real_label: str = "",
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
        render_week(
            result,
            title,
            subtitle,
            link=lambda i: f"game-{i + 1}.html",
            names=names,
            real=real,
            real_label=real_label,
        )
    )
    return index
