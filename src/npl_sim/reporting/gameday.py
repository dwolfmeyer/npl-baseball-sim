"""A standalone Gameday-style game page: line score, a win probability chart of
the game's flow, box score, and a play log filtered to all plays or scoring
plays. The web app (Milestone 10) serves the same markup."""

from __future__ import annotations

from html import escape

from npl_sim.engine.box import BoxScore
from npl_sim.engine.playlog import GameLog

CSS = """
:root{--bg:#fff;--fg:#14171a;--muted:#5f6b76;--line:#e3e7eb;--panel:#f6f8fa;--accent:#0b5cad;
--score:#fff4d6}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#111418;--fg:#e8ecef;
--muted:#9aa6b2;--line:#2a3138;--panel:#181d22;--accent:#6fb1ff;--score:#2e2816}}
:root[data-theme=dark]{--bg:#111418;--fg:#e8ecef;--muted:#9aa6b2;--line:#2a3138;--panel:#181d22;
--accent:#6fb1ff;--score:#2e2816}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 16px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:4px 6px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600;font-size:12px}
td.sub{padding-left:20px}
.wrap{overflow-x:auto}.line td.big{font-weight:700}.line td:first-child{width:28%;font-weight:600}
.tabs{display:flex;gap:4px;margin:20px 0 12px;border-bottom:1px solid var(--line)}
.tabs button,.filter button{background:none;border:0;color:var(--muted);font:inherit;
padding:8px 12px;cursor:pointer;border-bottom:2px solid transparent}
.tabs button[aria-selected=true]{color:var(--fg);border-color:var(--accent);font-weight:600}
.filter{display:inline-flex;border:1px solid var(--line);border-radius:6px;margin-bottom:12px}
.filter button{border:0;padding:6px 12px}
.filter button[aria-pressed=true]{background:var(--panel);color:var(--fg);font-weight:600}
.teams{display:grid;grid-template-columns:1fr;gap:24px}
@media (min-width:880px){.teams{grid-template-columns:1fr 1fr}}
h2{font-size:15px;margin:0 0 8px}h3{font-size:12px;letter-spacing:.06em;color:var(--muted);
margin:14px 0 4px}.notes p{margin:2px 0}.notes b{font-weight:600}
.half{margin:0 0 16px}.half h3{font-size:13px;color:var(--fg);letter-spacing:0;
background:var(--panel);padding:6px 8px;margin:0}
.play{display:flex;gap:12px;justify-content:space-between;padding:6px 8px;
border-bottom:1px solid var(--line)}.play.scoring{background:var(--score)}
.play .meta{color:var(--muted);white-space:nowrap;text-align:right}.play .sc{font-weight:600;
color:var(--fg)}.event{color:var(--muted);font-style:italic}.summary{color:var(--muted);
font-size:12px;padding:6px 8px}[hidden]{display:none!important}
.wp{margin:20px 0 0}.wp svg{width:100%;min-width:520px;height:auto;display:block}
.wp text{fill:var(--muted);
font-size:11px}.wp .grid{stroke:var(--line)}.wp .mid{stroke:var(--muted);stroke-dasharray:3 3}
.wp .curve{fill:none;stroke:var(--accent);stroke-width:2}.wp .hit{fill:transparent}
.wp .hit:hover{fill:var(--accent)}.wp .key{fill:var(--accent)}.wp .team{fill:var(--fg);
font-weight:600}.wp ol{margin:6px 0 0;padding-left:20px;color:var(--muted)}
.wp li span{color:var(--fg)}
"""

JS = """
const tabs=document.querySelectorAll('.tabs button');
tabs.forEach(b=>b.addEventListener('click',()=>{tabs.forEach(t=>{
t.setAttribute('aria-selected',t===b);document.getElementById(t.dataset.panel).hidden=t!==b;});}));
const filters=document.querySelectorAll('.filter button');
filters.forEach(b=>b.addEventListener('click',()=>{const only=b.dataset.filter==='scoring';
filters.forEach(f=>f.setAttribute('aria-pressed',f===b));
document.querySelectorAll('.play').forEach(p=>p.hidden=only&&!p.classList.contains('scoring'));
document.querySelectorAll('.summary').forEach(s=>s.hidden=only);
document.querySelectorAll('.half').forEach(h=>h.hidden=only&&!h.querySelector('.play.scoring'));
}));
"""


def _line_score(box: BoxScore) -> str:
    innings = max(len(box.line[box.away]), len(box.line[box.home]))
    head = "".join(f"<th>{i + 1}</th>" for i in range(innings))
    rows = []
    for team in (box.away, box.home):
        cells = [str(r) for r in box.line[team]] + ["X"] * (innings - len(box.line[team]))
        rows.append(
            f"<tr><td>{escape(box.abbr[team])}</td>"
            + "".join(f"<td>{c}</td>" for c in cells)
            + f'<td class="big">{box.runs[team]}</td><td class="big">{box.hits[team]}</td>'
            f'<td class="big">{box.errors[team]}</td><td>{box.lob[team]}</td></tr>'
        )
    return (
        '<div class="wrap"><table class="line"><tr><th></th>'
        + head
        + "<th>R</th><th>H</th><th>E</th><th>LOB</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )


def _batting(box: BoxScore, team: str) -> str:
    cols = ("ab", "r", "h", "rbi", "bb", "k", "lob")
    rows, totals = [], dict.fromkeys(cols, 0)
    for b in box.batting[team].values():
        cells = "".join(f"<td>{getattr(b, c)}</td>" for c in cols)
        rows.append(
            f"<tr><td{' class=sub' if b.substitute else ''}>{escape(b.name)} "
            f"<small>{escape(b.position)}</small></td>{cells}</tr>"
        )
        for c in cols:
            totals[c] += getattr(b, c)
    rows.append("<tr><td><b>Totals</b></td>" + "".join(f"<td>{totals[c]}</td>" for c in cols))
    head = "".join(f"<th>{h}</th>" for h in ("AB", "R", "H", "RBI", "BB", "SO", "LOB"))
    notes = "".join(
        f"<h3>{escape(heading.upper())}</h3>"
        + "".join(f"<p><b>{escape(k)}:</b> {escape(v)}</p>" for k, v in items)
        for heading, items in box.team_notes[team]
    )
    return (
        f'<section><h2>{escape(team)}</h2><div class="wrap"><table><tr><th>Batters</th>{head}'
        f'</tr>{"".join(rows)}</table></div><div class="notes">{notes}</div></section>'
    )


def _pitching(box: BoxScore, team: str) -> str:
    head = "".join(f"<th>{h}</th>" for h in ("IP", "H", "R", "ER", "BB", "SO", "HR", "P"))
    rows = "".join(
        f"<tr><td>{escape(p.label)}</td><td>{p.ip}</td><td>{p.h}</td><td>{p.r}</td>"
        f"<td>{p.er}</td><td>{p.bb}</td><td>{p.k}</td><td>{p.hr}</td><td>{p.pitches}</td></tr>"
        for p in box.pitching[team].values()
    )
    return (
        f'<section><h2>{escape(team)}</h2><div class="wrap"><table><tr><th>Pitchers</th>{head}'
        f"</tr>{rows}</table></div></section>"
    )


def _pct(x: float) -> str:
    return f"{x:+.0%}".replace("+0%", "0%").replace("-0%", "0%")


def _win_chart(box: BoxScore, log: GameLog) -> str:
    """Home win expectancy after every play, Tango-style: the top of the chart
    is a sure home win, the bottom a sure road win."""
    curve = log.win_curve()
    if not curve:
        return ""
    w, h, left, top, plot_h = 720, 190, 44, 8, 160
    step = (w - left - 8) / len(curve)

    def y(we: float) -> float:
        return top + (1 - we) * plot_h

    points = [(left, y(log.home_we_start))]
    points += [(left + (i + 1) * step, y(p.home_we)) for i, (_, p) in enumerate(curve)]
    path = " ".join(f"{x:.1f},{yy:.1f}" for x, yy in points)
    grid, labels, seen = [], [], set()
    for i, (half, _) in enumerate(curve):
        if half.inning not in seen:
            seen.add(half.inning)
            x = left + i * step
            grid.append(
                f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + plot_h}"/>'
            )
            labels.append(f'<text x="{x + 3:.1f}" y="{h - 6}">{half.inning}</text>')
    biggest = sorted(range(len(curve)), key=lambda i: -abs(curve[i][1].wpa))[:3]
    dots = []
    for i, (half, p) in enumerate(curve):
        x, yy = points[i + 1]
        tip = (
            f"{half.label}: {p.description} {box.abbr[box.home]} {p.home_we:.0%}, "
            f"WPA {_pct(p.wpa)} for {half.batting}"
        )
        cls = "key" if i in biggest else "hit"
        dots.append(
            f'<circle class="{cls}" cx="{x:.1f}" cy="{yy:.1f}" r="{4 if cls == "key" else 5}">'
            f"<title>{escape(tip)}</title></circle>"
        )
    mid = y(0.5)
    svg = (
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Win probability by play">'
        f'<line class="grid" x1="{left}" x2="{w - 8}" y1="{top}" y2="{top}"/>'
        f'<line class="grid" x1="{left}" x2="{w - 8}" y1="{top + plot_h}" y2="{top + plot_h}"/>'
        f'<line class="mid" x1="{left}" x2="{w - 8}" y1="{mid}" y2="{mid}"/>{"".join(grid)}'
        f'<text class="team" x="0" y="{top + 10}">{escape(box.abbr[box.home])}</text>'
        f'<text class="team" x="0" y="{top + plot_h}">{escape(box.abbr[box.away])}</text>'
        f'<text x="0" y="{mid + 4}">50%</text>{"".join(labels)}'
        f'<polyline class="curve" points="{path}"/>{"".join(dots)}</svg>'
    )
    items = "".join(
        f"<li><span>{escape(curve[i][0].label)}: {escape(curve[i][1].description)}</span> "
        f"{_pct(curve[i][1].wpa)} for {escape(curve[i][0].batting)}</li>"
        for i in sorted(biggest)
    )
    return (
        f'<section class="wp"><h2>Win probability</h2><div class="wrap">{svg}</div>'
        f"<h3>BIGGEST PLAYS</h3><ol>{items}</ol></section>"
    )


def _plays(log: GameLog) -> str:
    out = []
    for half, plays in log.plays("all"):
        items = []
        for p in plays:
            cls = "play scoring" if p.scoring else "play"
            desc = escape(p.description)
            if p.kind != "pa":
                desc = f'<span class="event">{desc}</span>'
            meta = []
            if p.scoring:
                meta.append(f'<span class="sc">{escape(p.score_text)}</span>')
            if p.count:
                meta.append(f"{p.count}, {p.outs} out")
            if p.wpa is not None:
                swing = f"WPA {_pct(p.wpa)}"
                meta.append(f"{swing} · LI {p.li:.1f}" if p.li is not None else swing)
            items.append(
                f'<div class="{cls}"><div>{desc}</div>'
                f'<div class="meta">{"<br>".join(meta)}</div></div>'
            )
        out.append(
            f'<div class="half"><h3>{escape(half.label)} · {escape(half.batting)}</h3>'
            + "".join(items)
            + f'<div class="summary">{escape(half.summary)}</div></div>'
        )
    return "".join(out)


def game_body(box: BoxScore, log: GameLog) -> str:
    """The line score, win probability chart, Box and Play-by-play tabs, for
    embedding in a page that includes `CSS` and `JS` (the web app's game page does)."""
    game_notes = "".join(f"<p><b>{escape(k)}:</b> {escape(v)}</p>" for k, v in box.game_notes)
    return f"""{_line_score(box)}
{_win_chart(box, log)}
<div class="tabs" role="tablist">
<button role="tab" aria-selected="true" data-panel="box">Box</button>
<button role="tab" aria-selected="false" data-panel="plays">Play-by-play</button></div>
<div id="box"><div class="teams">{_batting(box, box.away)}{_batting(box, box.home)}</div>
<div class="teams" style="margin-top:24px">{_pitching(box, box.away)}{_pitching(box, box.home)}
</div><div class="notes" style="margin-top:16px">{game_notes}</div></div>
<div id="plays" hidden><div class="filter" role="group" aria-label="Plays">
<button aria-pressed="true" data-filter="all">All plays</button>
<button aria-pressed="false" data-filter="scoring">Scoring plays</button></div>
{_plays(log)}</div>"""


def render_game(box: BoxScore, log: GameLog, title: str | None = None, subtitle: str = "") -> str:
    """Full HTML page for one game."""
    title = title or f"{box.away} {box.runs[box.away]}, {box.home} {box.runs[box.home]}"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>{CSS}</style></head><body><main>
<h1>{escape(title)}</h1><p class="sub">{escape(subtitle)}</p>
{game_body(box, log)}
</main><script>{JS}</script></body></html>"""
