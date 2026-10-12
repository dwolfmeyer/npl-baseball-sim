"""MLB data pipeline (milestone 4).

Every source lands raw in a dated snapshot under `data/raw`, then a
normalize step writes clean Parquet tables keyed by MLBAM ID under
`data/clean`. The sim only ever reads the clean tables, never a live API.

Sources: MLB Stats API (schedule, box scores, players, transactions),
Baseball Savant (pitch-level Statcast and fielding/running leaderboards, via
pybaseball where it covers them) and FanGraphs (leaderboards and
projections).
"""
