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
derived from that log only. Each PA combines batter and pitcher with
multinomial log5 against the league rate for that handedness matchup, then
applies fatigue, catcher framing, park (the only place park factors enter)
and defense/alignment once each.

```sh
npl-sim demo-game --seed 3 --pbp   # two league-average teams
```

League structure lives in `config/league.yaml`. When divisions realign, add a
new alignment block with its `from_season`; past seasons keep their divisions.
