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

### Schedule

```sh
npl-sim generate-schedule --season 2026 --csv schedule-2026.csv --save
```

Builds MLB's balanced 162-game mix from the league config (rival 18,
division 17, own league 8, interleague 4 as home-and-home 2-game sets and 3
vs two rotating opponents), gives every team 81 home games, packs the series
into the Monday-Sunday weeks in `config/calendar.yaml` with OR-Tools CP-SAT,
and validates the result before saving. Division rivals default to the first
two and last two teams listed; set `rivals` on a division to change them.

### Game engine

`src/npl_sim/engine` is a pure library: frozen player cards, lineups, park
and league environment in; an event log out. Box scores and play-by-play are
derived from that log only, in MLB Gameday's format: line score with LOB,
batting and pitching tables, W/L/S/H/BS decisions, the batting, baserunning,
fielding and pitching notes, and a play log by half inning that filters to
all plays or scoring plays. Each PA combines batter and pitcher with
multinomial log5 against the league rate for that handedness matchup, then
applies fatigue, catcher framing, park (the only place park factors enter)
and defense/alignment once each. Teams play straight up unless the owner
ticks alignment rules (`engine/alignment.py`): infield in with a runner on
third, double-play depth conceding the run, guard the lines or no doubles,
each optionally limited by inning and score.

Position eligibility (`npl_sim/eligibility.py`): 20 MLB games at a position
last season or 10 this season, counted from box scores; players with no
qualifying games yet fall back to their listed NPL position. Anyone can DH.

```sh
npl-sim demo-game --seed 3 --plays scoring   # two league-average teams
npl-sim demo-game --seed 3 --html game.html  # Gameday-style page: Box / Play-by-play
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
