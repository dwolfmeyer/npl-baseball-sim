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

League structure lives in `config/league.yaml`. When divisions realign, add a
new alignment block with its `from_season`; past seasons keep their divisions.
