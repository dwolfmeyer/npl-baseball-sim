"""A sim week as HTML in three columns (results, recaps, box scores): standings, each
day's games with who started, who sat and who filled in, a recap per team, and the box
score of whichever game is picked, plus a Gameday page per game (`write_week`)."""

from __future__ import annotations

from html import escape
from pathlib import Path

from npl_sim.engine.box import box_score
from npl_sim.engine.playlog import game_log
from npl_sim.engine.workload import WeekResult
from npl_sim.reporting.gameday import CSS, box_fragment, render_game

DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

WEEK_CSS = """
main.week{max-width:1560px}
.game{border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin:0 0 8px;cursor:pointer}
.game:hover{background:var(--panel)}.game.sel{outline:2px solid var(--accent);outline-offset:-1px}
.game .score{display:flex;gap:8px;font-weight:600;align-items:baseline}
.game .day{color:var(--muted);font-weight:400;width:32px;flex:none}
.game .teams-line{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:2px 10px;flex:1}
.game .teams-line span{overflow-wrap:anywhere}.game .teams-line b{text-align:right}
.game details{margin-top:6px}.game summary{color:var(--muted);font-size:12px;cursor:pointer}
.game ul{margin:4px 0 0;padding-left:18px;color:var(--muted);font-size:12px}.game li{margin:2px 0}
.win{font-weight:700}.lose{font-weight:400;color:var(--muted)}
.recap{border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin:0 0 8px}
.recap p{margin:3px 0}
.bar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:0 0 12px}
.bar label{display:flex;align-items:center;gap:8px}
.cols{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr) minmax(0,1.25fr);gap:16px;
align-items:start}
.col{min-width:0;max-height:calc(100vh - 150px);overflow-y:auto;overflow-x:hidden;
padding-right:4px}
.col>h2{position:sticky;top:0;z-index:1;background:var(--bg);margin:0 0 8px;padding:4px 0;
border-bottom:2px solid var(--line);font-size:13px;letter-spacing:.06em;text-transform:uppercase}
.col table{table-layout:auto}.col th,.col td{white-space:normal;padding:3px 4px;font-size:12px}
.col td.num,.col th.num{white-space:nowrap}
.col .wrap{overflow:visible;margin-bottom:10px}.col .teams{grid-template-columns:1fr;gap:0}
.col section{margin:0 0 14px}.col .notes{font-size:12px}
.col .line th,.col .line td{padding:3px 2px;text-align:center}
.col .line{table-layout:fixed}.col .line th:first-child,.col .line td:first-child{width:38px;
text-align:left}
.boxhead{display:flex;justify-content:space-between;gap:8px;align-items:baseline;margin:0 0 8px}
.boxhead a{white-space:nowrap}.empty{color:var(--muted);padding:12px 0}
.tabbar{display:none;gap:4px;margin:0 0 12px;border-bottom:1px solid var(--line)}
.tabbar button{background:none;border:0;color:var(--muted);font:inherit;padding:8px 14px;
cursor:pointer;border-bottom:2px solid transparent}
.tabbar button[aria-selected=true]{color:var(--fg);border-color:var(--accent);font-weight:600}
@media (max-width:1099px){
.tabbar{display:flex}.cols{display:block}.col{max-height:none;overflow:visible}
.col>h2{position:static}
.cols[data-show=results]>.col:not(#c-results),.cols[data-show=recaps]>.col:not(#c-recaps),
.cols[data-show=box]>.col:not(#c-box){display:none}}
"""


# Limits games and recaps to one team (the page works without JS: everything shows, and the
# box scores open in order), and shows the box score of the clicked game.
FILTER_JS = """
const sel=document.getElementById('team'),cols=document.querySelector('.cols'),
games=[...document.querySelectorAll('.game')],recaps=[...document.querySelectorAll('.recap')],
boxes=[...document.querySelectorAll('.boxview')],empty=document.getElementById('nobox'),
narrow=matchMedia('(max-width:1099px)');
function pick(i,click){for(const g of games)g.classList.toggle('sel',g.dataset.i===i);
for(const b of boxes)b.hidden=b.dataset.i!==i;empty.hidden=i!==null;
if(i!==null)history.replaceState(null,'','#g'+i);
if(click&&narrow.matches)show('box');}
function show(c){cols.dataset.show=c;
document.querySelectorAll('.tabbar button').forEach(
b=>b.setAttribute('aria-selected',b.dataset.c===c));}
document.querySelectorAll('.tabbar button').forEach(b=>b.onclick=()=>show(b.dataset.c));
for(const g of games)g.onclick=e=>{if(!e.target.closest('summary,a'))pick(g.dataset.i,true)};
function apply(pref){const t=sel.value;let n=0;
for(const g of games){const on=!t||g.dataset.away===t||g.dataset.home===t;g.hidden=!on;n+=on}
for(const r of recaps)r.hidden=!!t&&r.dataset.team!==t;
document.getElementById('count').textContent=t?n+' of '+games.length+' games':'';
const u=new URL(location);t?u.searchParams.set('team',t):u.searchParams.delete('team');
history.replaceState(null,'',u);
const cur=games.find(g=>g.classList.contains('sel')&&!g.hidden)||
games.find(g=>g.dataset.i===pref&&!g.hidden)||games.find(g=>!g.hidden);
pick(cur?cur.dataset.i:null,false)}
sel.onchange=()=>apply();
sel.value=new URLSearchParams(location.search).get('team')||'';
if(sel.value!==(new URLSearchParams(location.search).get('team')||''))sel.value='';
show('results');apply(location.hash.slice(2));
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
    """The week's page in three columns: results (standings and the games), recaps, and the
    box score of the game picked in the results. `link(i)` gives the href of game i's full
    page, or None; `names` maps player ids to names for players who didn't appear in a game."""
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
        + "</div>"
        for team, r in week_recaps(result, names).items()
    )
    teams = sorted({t for g in result.games for t in (g["away"], g["home"])})
    options = "".join(f'<option value="{escape(t)}">{escape(t)}</option>' for t in teams)
    games, boxes = [], []
    for i, g in enumerate(result.games):
        final = g["events"][-1]
        who = {**(names or {}), **_names(g["events"])}
        a, h = final["away"], final["home"]
        day = DAY_NAMES[g["day"] % 7]
        notes = _notes(g, g["away"], who) + _notes(g, g["home"], who)
        items = "".join(f"<li>{escape(n)}</li>" for n in notes)
        games.append(
            f'<div class="game" data-i="{i}" data-away="{escape(g["away"])}" '
            f'data-home="{escape(g["home"])}"><div class="score"><span class="day">{day}</span>'
            f'<div class="teams-line"><span class="{"win" if a > h else "lose"}">'
            f'{escape(g["away"])}</span><b class="{"win" if a > h else "lose"}">{a}</b>'
            f'<span class="{"win" if h > a else "lose"}">{escape(g["home"])}</span>'
            f'<b class="{"win" if h > a else "lose"}">{h}</b></div></div>'
            f"<details><summary>Lineup notes ({len(notes)})</summary><ul>{items}</ul></details>"
            "</div>"
        )
        href = link(i) if link else None
        page = f'<a href="{escape(href)}">Full game page</a>' if href else ""
        boxes.append(
            f'<div class="boxview" data-i="{i}" hidden><div class="boxhead"><strong>'
            f"{escape(g['away'])} at {escape(g['home'])}</strong>{page}</div>"
            f"{box_fragment(box_score(g['events']))}</div>"
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>{CSS}{WEEK_CSS}</style>
<noscript><style>.boxview[hidden]{{display:block!important}}</style></noscript></head>
<body><main class="week">
<h1>{escape(title)}</h1><p class="sub">{escape(subtitle)}</p>
<div class="bar"><label>Team <select id="team"><option value="">All teams</option>{options}
</select></label><span id="count" class="sub"></span></div>
<div class="tabbar" role="tablist"><button role="tab" data-c="results">Results</button>
<button role="tab" data-c="recaps">Recaps</button>
<button role="tab" data-c="box">Box scores</button></div>
<div class="cols" data-show="results">
<section class="col" id="c-results"><h2>Results</h2>
<h3>Standings</h3><table><tr><th>Team</th><th class="num">W</th><th class="num">L</th>
<th class="num">RS</th><th class="num">RA</th></tr>{table}</table>
<h3 style="margin-top:16px">Games</h3>{"".join(games)}</section>
<section class="col" id="c-recaps"><h2>Recaps</h2>{recaps}</section>
<section class="col" id="c-box"><h2>Box scores</h2>{"".join(boxes)}
<p id="nobox" class="empty" hidden>No games to show.</p></section>
</div>
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
