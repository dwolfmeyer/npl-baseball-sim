# NPL Baseball Simulator

A modern replacement for Scoresheet Baseball for the NPL. Managers set weekly
lineups and bullpen plans; the engine sims every game of a 162-game MLB-style
season from each player's real MLB performance that week, set in today's run
environment and adjusted for park, handedness and defense.

The v1 build plan (architecture, data pipeline, engine design and milestones)
lives in the project's plan doc. Work lands milestone by milestone as pull
requests.

## Development

Requires Python 3.12+ and Postgres 16.

```sh
uv venv && uv pip install -e ".[dev]"
export DATABASE_URL=postgresql+psycopg://localhost/npl_sim
alembic upgrade head          # create the schema
npl-sim league-check          # validate config/league.yaml
npl-sim sync-league --season 2026
pytest                        # uses SQLite unless DATABASE_URL is set
```

### Rosters

Rosters come from the NPL league app ([jeremyjbowers/npl](https://github.com/jeremyjbowers/npl)),
which already stores every rostered player by MLBAM ID with his team and
roster tier. The importer reads its database through a read-only login and
uses the app's own rule for the active MLB Roster (40-man, on no other list).

```sh
export NPL_DATABASE_URL=postgresql+psycopg://<read-only user>@<host>/<db>   # keep in secrets
npl-sim import-rosters --season 2026 --week 1 --week-start 2026-03-23
# offline, from a Django dumpdata export of the NPL app:
npl-sim import-rosters --season 2026 --week 1 --week-start 2026-03-23 \
    --from-dump tests/fixtures/npl_sample.json
```

The command fails if any MLB Roster player has no MLBAM ID.

Until the NPL app's login is issued, every team's roster comes from the
league's roster workbook (one tab per team), read the way the NPL app reads
it: section headers set the tier, column A's 1 marks the 40-man. The
result is saved as an NPL app dump in `data/rosters/npl-2026.json`, which
`sim-season` reads (and the web app and `sim-week` can, with
`NPL_ROSTER_DUMP=data/rosters/npl-2026.json`). The command also lists
teams whose tab header names a different division than `config/league.yaml`.

```sh
npl-sim rosters-from-workbook National_Pastime_League_Rosters.xlsx
```

### Schedule

```sh
npl-sim generate-schedule --season 2026 --csv schedule-2026.csv --save
```

Builds MLB's balanced 162-game mix from the league config (rival 18,
division 17, own league 8, interleague 4 as home-and-home 2-game sets and 3
vs two rotating opponents), gives every team 81 home games, all in 2- and
3-game series, packs the series into the Monday-Sunday weeks in
`config/calendar.yaml` with OR-Tools CP-SAT, and validates the result
before saving. Division rivals default to the first two and last two teams
listed; set `rivals` on a division to change them.

Weeks look like MLB's: a 6-game week is a midweek and a weekend series with
an off day (Monday or Thursday), a 7-game week is three series with none,
and 7-game weeks never come back to back. A 2-game series in a 3-day slot
leaves a travel day. In 2026 each team gets 20 off days plus the All-Star
break, at most 16 straight game days (MLB's limit is 20), 5 to 7 games in
every full week, and no homestand or road trip over 4 series, so a
five-man rotation on four days' rest covers every week.

### Season sim

```sh
npl-sim sim-season --season 2026 --csv season-games.csv
```

Plays the whole schedule week by week through `sim_week` between every
team's default lineup card (rotation: the five arms with the most MLB
starts; closer: most saves plus holds), carrying unused starts, the
rotation turn and rest into the next week, then prints standings by
division (games back, run differential, home and road records, last 10, streak) and league
sanity stats: runs per game, K%, BB%, HR rate, steals,
extra innings, bullpen games and the shortest rest between starts. Playing
time is each player's typical week at his season pace from the eligibility
data, or a full-time week (about 26 PA, 6.2 starts per five-man rotation,
3 relief innings) where it has none yet. Ability is league average until
the weekly talent blend is wired in.

Along the way it keeps season to date (`npl_sim/season_stats.py`): each team's record and every
player's batting and pitching totals, folded in from each game's box score and saved as
plain JSON (`SeasonStats.to_json`). Standings and, next, leaders and recaps read from it.

### Game engine

`src/npl_sim/engine` is a pure library: frozen player cards, lineups, park
and league environment in; an event log out. Box scores and play-by-play are
derived from that log only, in MLB Gameday's format: line score with LOB,
batting and pitching tables, W/L/S/H/BS decisions, the batting, baserunning,
fielding and pitching notes, and a play log by half inning that filters to
all plays or scoring plays. A win probability chart under the line score
shows the game's flow and marks its three biggest plays, and each play in the
log carries its WPA and leverage index (`engine/winprob.py`: win expectancy
by inning, half, score, runners and outs, chained from how many runs the
engine scores from each base-out state, as in Tango's tables). Clicking a
plate appearance opens an mlb.com-style at-bat pop-up: score, result, the
batting team's win probability and WPA, pitcher and batter, bases, outs and
count, and each pitch with its call, speed, type and spot in the strike zone.
The engine doesn't play pitch by pitch, so `engine/pitchseq.py` fills in a
sequence that matches the logged pitch count and final count, on its own
random stream (results never change), using the pitcher's real pitch mix and
speeds from Statcast when the talent data has them. Each PA combines batter and pitcher with
multinomial log5 against the league rate for that handedness matchup, then
applies fatigue, catcher framing, park (the only place park factors enter)
and defense/alignment once each. Teams play straight up unless the owner
ticks alignment rules (`engine/alignment.py`): infield in with a runner on
third, double-play depth conceding the run, guard the lines or no doubles,
each optionally limited by inning and score.

Bullpen (`engine/bullpen.py`): owners tag relievers high, medium or low
leverage, mop-up, long or any, start-of-inning only, a same-side matchup
specialist, and a stint length. No closer is required. When a pitcher comes
out the engine reads the spot's leverage from Tango's Leverage Index table
(`engine/leverage.py`: inning, half, score, outs, runners) and picks from the
matching tier, preferring the platoon edge on the next batter,
then the owner's order. Openers' starters and piggybacks (`TeamCard.follow`)
enter first, in order. Untagged pens enter in listed order. A checked opener
(`TeamCard.opener`) goes one inning at most, even a clean one, though he can
still come out sooner (pitch limit, hook, runs, the three-batter minimum).

Starter hooks (`engine/hooks.py`, `TeamCard.hook`): besides his pitch limit,
a starter (or the bulk arm after an opener) comes out after a set number of
times through the order, once he has been charged with a set number of runs,
or from a set inning on when the spot's leverage index reaches a threshold.
Defaults when the owner sets nothing: out after 6 runs, and from the 7th in
spots of LI 2.0 or more. Each pitching change logs its reason.

MLB playing rules: extra innings start with a runner on second; the
three-batter minimum (a pitcher faces three batters or finishes the half
inning before he can come out); a walk-off ends the game when the winning run
scores, crediting the batter with only as many bases as that runner advanced
(every run counts on a home run); ground-rule doubles (about 6% of doubles,
every runner gets exactly two bases); sacrifice flies; sacrifice bunts
(`engine/manager.py` `Bunting`, `TeamCard.bunt`: by default a hitter at least
0.03 runs per PA worse than average bunts half the time with nobody out, a
runner on first or second and third open, from the 7th inning, tied or down
one; outcomes include bunt singles, force outs, pop-ups and foul-bunt
strikeouts); wild pitches, passed balls (runs unearned) and balks at recent
MLB rates. Steals, wild pitches and passed balls happen on a pitch of the PA
in progress: that PA throws at least one more pitch, the play-by-play reads
"With X batting, ..." and the at-bat pop-up marks the pitch. When a caught
stealing makes the third out or a wild pitch scores the winning run, the PA
ends early and its pitches still count toward the pitcher's total, his hook
and the box score.

Stealing and baserunning (`engine/game.py` `_maybe_steal`, `_send`; owner
settings in `engine/manager.py` `Baserunning` and `BUNT_STRATEGIES`, on the
lineup card under "Bunting, stealing and baserunning"). Runners steal second,
and third with fewer than two out, as often as they do in MLB: each runner's
attempts per time on first and success rate come from his statsapi box-score
SB and CS over this season and the two before (`talent/running.py`),
regressed toward the league's rate for his sprint speed. The league's own
rates come from this season's box scores (default 0.105 attempts per time on
first, 77.5% success, about 0.96 tries a team game). Success also moves with
the catcher's arm (pop time to second and FanGraphs rSB caught-stealing runs,
this season and the three before), the pitcher's hand, and the owner's
aggressiveness (Never, Conservative, Normal, Aggressive, Very aggressive; a
more aggressive manager runs more and in worse spots, so more runners are
caught). Nobody runs with a lead of five or more, or trailing by three or more
from the 7th, and owners can give any hitter a red light. Taking the extra base
on singles, doubles and fly balls is two rolls: whether the runner is sent
(the MLB rate for the play, moved by his speed, the outfield arms and the
owner's extra-base setting) and whether he's safe or thrown out (weighted by
his sprint speed against the outfield arms). Runners thrown out show in the
play-by-play, the audit log (`thrown_out`) and the season totals (OOB).

Weekly playing time (`engine/workload.py`, `play_week`): each player's sim
week follows his real one. Hitters get 1.25x their MLB PA and start while
they have any left. A hitter who can't is covered the way a manager uses
his bench, hardest positions first: a bench player eligible there, else the
DH if he's eligible there (a bench bat then DHs), else the same through a
manager's move (only catchers catch; 1B to a corner outfield spot; 3B to 1B or
a corner; SS/2B anywhere on the infield; OF anywhere in the outfield), else
another fielder sliding over while the bench or DH covers his spot. Only then
does a (AAA) stand-in play, at Scoresheet's penalty levels
(`engine/replacement.py`). Defensive-spectrum costs still apply. Starters get their MLB
starts plus up to one carried from last week, in rotation order; with nobody
able to start, the reliever with the most innings left opens a bullpen game.
Relievers get 1.5x their MLB relief outs. There are no rest-day rules (no
days between starts, no limit on pitching days in a row), so the budgets are
the only limit, except that a starter pitches at most once in a series (the weekly
games carry a series id; he can start again in the next series, and with no
starter free the pen opens a bullpen game). Unused starts and the rotation turn carry into next week. `npl_sim/playing_time.py`
reads a real week from the pipeline's clean tables.

`npl-sim sim-week` (`npl_sim/simweek.py`) plays a week between the test teams'
saved lineup cards (default cards when none are saved): two three-game series per team (Tue-Thu and
Fri-Sun, Monday off, a different opponent each series), each
team's orders vs LHP and RHP, the rotation with each slot's plan, the pen and
bench. The web app's "Sim a week" plays the latest week of clean data (week 27 of 2026
once the season is in) with that week's real PA, starts and relief outs, falling back to a
typical week when the batting and pitching tables aren't there. Without `--week` each player gets a typical week at his season pace
from the eligibility data (full time when he has none). `--out` writes a week
page with standings and who sat or filled in, plus a Gameday page per game;
`--save` adds the games to the web app. Without `--week` everyone is league
average; with it, each player plays at his talent blend for that week.
With `--audit DIR` it also writes an audit log for checking the engine's realism: `game-NN.jsonl` per game with every
plate appearance (game state, the batter's, pitcher's and league's input
rates, park, defense and fatigue, the outcome odds the engine drew from, the
result, batted-ball type, fielder, runners, count, the simulated pitches,
WPA and leverage) and each player's card, talent-blend layers and real MLB
stat lines by season and opponent hand. Between the PAs, in game order, are
the manager and baserunning decisions (pitching changes with the reason, the
outgoing pitcher's pitches and runs, leverage, the bullpen tier and the
reliever's role; pinch hitters; defensive subs; steals), and each team's
settings behind them (pitch limits, hook, bullpen roles and stints, pinch-hit
rules). Plus `pas.csv`, one row per PA.

Calibration harness (`npl_sim/calibration/`, Milestone 8): `npl-sim
calibrate` and `tests/test_calibration.py` check that box scores add up,
that the league run environment lands inside the targets in
`config/calibration.yaml` (a placeholder band now; the 2025 and 2026 MLB
seasons within 3% for the replay), that known-talent archetypes (three true
outcomes, slap, power, speed) play back their cards, that K% spreads the way
binomial noise says and shrinks with sample size, and that the engine's
results haven't drifted from `tests/baselines/engine.json` (Jensen-Shannon
divergence on the PA result mix, Wasserstein distance on runs per game).
After an intended engine change, run `npl-sim calibrate --save-baseline`
and commit the new baseline with it.

Pinch hitting and manager styles (`engine/manager.py`): before each PA the
batting team may send up a bench bat when the owner's `PinchHitting` rules
allow it (from an inning, at or above a leverage index, for the platoon edge
only or any edge, a minimum gain in expected runs per PA, a per-game limit,
never out of position unless allowed, never leaving the team without a
catcher). The comparison uses the same PA model that plays the game, so the
talent blend and splits decide it. Five manager styles (sabermetric,
aggressive, balanced, old-school, passive) bundle a starter hook with
pinch-hitting rules; owners can change any setting. In test weeks the
default (balanced) pinch-hits about 0.4 times per team-game.

Weekly talent blend (`npl_sim/talent/`, Milestone 7): each player's ability
for a sim week, built from Statcast pitches in three layers, each an index
against the league for the same handedness matchup. Baseline weights this
season before the week and the two seasons before it 5/4/3, every PA measured
against its own season's league, regressed by stabilization points; platoon
splits come from prior seasons (this season only for players with no split
data). Process compares the week's whiff, chase, zone, barrel, xBA-on-contact,
hard-hit, fastball velocity and Stuff+ (per pitch, regressed by pitch count)
with the player's own baseline. Results add what the week's outcomes show
beyond that, regressed three times as hard. The league environment comes from
the same season's PAs, and every PA's expectations are scaled by Savant's park
factors for the park it was played in, so hitters' records are park-neutral.
Fielding (`talent/fielding.py`): at each position, the average of Statcast OAA
(in runs), DRS and UZR (FanGraphs) over this season to date and the three
before, weighted by innings, per 150 games and regressed by 700 innings. A
player without them is league average at each eligible position. Catcher
framing runs per 150 games over the same seasons; arms from pop time
(catchers) and outfield arm strength. On the 20-80 scale each 2.5 runs is a
grade (5 points). `npl-sim sim-week
--week N` plays at the blend; the weights are first guesses for Milestone 8
calibration.

Out-of-position defense (`engine/fielding.py`): each hitter carries his
fielding value at the positions he's experienced at (`Hitter.positions`).
Anywhere else, Tango's defensive spectrum (C +8, CF +5, SS +4, 2B 0, 3B -1,
LF/RF -4, 1B -8 runs per 150 games), averaged with Zimmerman's measured
position-pair differences (2015) where he has them, sets the cost: moving down
keeps half the gap, moving up loses all of it, plus an experience penalty and
more for crossing infield/outfield or catching. A catcher moving out gets no
credit for the catcher's spot. An outfielder can play any outfield spot, and
a shortstop second or third, with no experience penalty and no out-of-position
flag; his rating carries along the spectrum, so an elite center fielder in
right is better than elite there. The fielders' values move hit rates on the
ball types they handle, so a +R fielder saves R runs over 150 games; the box
score lists who played out of position.

Position eligibility (`npl_sim/eligibility.py`): 20 MLB games at a position
last season or 10 this season, counted from box scores; players with no
qualifying games yet fall back to their listed NPL position. Anyone can DH.

```sh
npl-sim demo-game --seed 3 --plays scoring   # two league-average teams
npl-sim demo-game --seed 3 --html game.html  # Gameday-style page: Box / Play-by-play
npl-sim demo-week --seed 2                   # a week under the playing-time budget
npl-sim sim-week --out week/                 # the test teams' lineup cards, one week
npl-sim sim-week --week 12 --out week/       # ...with 2026 week 12's real MLB usage
npl-sim sim-week --week 12 --audit audit/    # ...and an audit log of every PA (npl_sim/audit.py)
```

League structure lives in `config/league.yaml`. When divisions realign, add a
new alignment block with its `from_season`; past seasons keep their divisions.

### Web app

```sh
npl-sim web                   # http://127.0.0.1:8000
```

Each team's page is its weekly lineup card: up to 30 active players, batting
orders against right- and left-handed starters (with DH), a five-man rotation
and bullpen roles, checked against the roster as you edit. "Sim a game" plays
two teams' saved cards and shows the Gameday box score and play-by-play.
Rosters come from `NPL_DATABASE_URL` when set, otherwise from a dump
(`NPL_ROSTER_DUMP`, default the current Ducksnorts and McFlys MLB rosters).
Drag rows to reorder a batting order, or drag a bench player onto a row. Each
starter has a pitch limit, a times-through-the-order hook, and an optional
opener and piggyback; situational defense is a set of checkboxes.

Hitters can field a position with 20 MLB games there last season or 10 this
season, and anyone can DH. Anyone else can still play a position, out of
position (the card warns). Games by position and batting hands come from the
MLB Stats API:

```sh
npl-sim eligibility --season 2026 --from-dump tests/fixtures/npl_2026_test_teams.json
```

which writes `data/eligibility/2026.json`, with season pitching lines (G, GS,
outs, saves, holds) that the card's Autofill button uses for the rotation and
bullpen roles. A player with no MLB fielding
record falls back to his NPL roster position. Cards and
games are stored as JSON under `NPL_WEB_DATA` (default `data/web`). Until the
weekly MLB cards are wired in, every player sims at league-average rates.

### MLB data

```sh
npl-sim check-calendar --season 2026      # config/calendar.yaml vs the MLB schedule
npl-sim pull-week --season 2026 --week 1  # Monday job: pull, normalize, freeze
npl-sim backfill --season 2025            # a completed season, week by week
npl-sim sync-mlb --season 2026            # handedness + lineup locks into the DB
```

Each pull lands raw under `data/raw/<source>/<dataset>/<season>/` (logged with
URL and SHA-256 in `data/raw/manifest.jsonl`), then becomes clean Parquet
tables keyed by MLBAM ID under `data/clean/<season>/`:

| Where | Tables |
| --- | --- |
| `week-NN/` | `games`, `batting`, `pitching` (per game, MLB Stats API box scores), `pitches` (Statcast), `transactions`, `league_totals` |
| `asof-YYYY-MM-DD/` | Savant `oaa`, `framing`, `poptime`, `sprint_speed`, `arm_strength`, `park_factors`; FanGraphs `fg_batting`, `fg_pitching`, `fg_fielding` (Stuff+, DRS), `fg_projections_*` |
| `players.parquet` | bats, throws, position, birth date |

`FROZEN.json` in a week folder records the hash of every table the sim week
runs on; a frozen week is never pulled or rewritten again. Regular-season
games before Opening Day (Opening Night, international series) count toward
week 1. Savant leaderboards are only published season-to-date, so a backfill
gets season-final boards; FanGraphs boards are true as-of pulls.
