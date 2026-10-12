"""core league schema

Revision ID: 0001
Revises:
Create Date: 2026-10-12 16:41:34.226489
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "league",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=32), nullable=False),
        sa.Column("dh", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "player",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("mlbam_id", sa.Integer(), nullable=True),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("position", sa.String(length=16), nullable=False),
        sa.Column("bats", sa.String(length=1), nullable=True),
        sa.Column("throws", sa.String(length=1), nullable=True),
        sa.Column("mlb_team", sa.String(length=8), nullable=True),
        sa.Column("is_standin", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("mlbam_id"),
    )
    op.create_table(
        "season",
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=True),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.PrimaryKeyConstraint("year"),
    )
    op.create_table(
        "team",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "division",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("league_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["league_id"],
            ["league.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("league_id", "name"),
    )
    op.create_table(
        "sim_week",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("season_year", sa.Integer(), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column("lineup_lock_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["season_year"],
            ["season.year"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("season_year", "number"),
    )
    op.create_table(
        "game",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("week_id", sa.Integer(), nullable=False),
        sa.Column("game_date", sa.Date(), nullable=False),
        sa.Column("home_team_id", sa.Integer(), nullable=False),
        sa.Column("away_team_id", sa.Integer(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.ForeignKeyConstraint(
            ["away_team_id"],
            ["team.id"],
        ),
        sa.ForeignKeyConstraint(
            ["home_team_id"],
            ["team.id"],
        ),
        sa.ForeignKeyConstraint(
            ["week_id"],
            ["sim_week.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "roster_entry",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("week_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column(
            "tier",
            sa.Enum(
                "mlb_roster",
                "il_7",
                "il_56",
                "il_eos",
                "outrighted",
                "non_roster",
                "aaa",
                "aa",
                "a",
                name="roster_tier",
            ),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["player_id"],
            ["player.id"],
        ),
        sa.ForeignKeyConstraint(
            ["team_id"],
            ["team.id"],
        ),
        sa.ForeignKeyConstraint(
            ["week_id"],
            ["sim_week.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("week_id", "player_id"),
    )
    op.create_table(
        "team_season",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("season_year", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("division_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["division_id"],
            ["division.id"],
        ),
        sa.ForeignKeyConstraint(
            ["season_year"],
            ["season.year"],
        ),
        sa.ForeignKeyConstraint(
            ["team_id"],
            ["team.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("season_year", "team_id"),
    )


def downgrade() -> None:
    op.drop_table("team_season")
    op.drop_table("roster_entry")
    op.drop_table("game")
    op.drop_table("sim_week")
    op.drop_table("division")
    op.drop_table("team")
    op.drop_table("season")
    op.drop_table("player")
    op.drop_table("league")
    sa.Enum(name="roster_tier").drop(op.get_bind(), checkfirst=True)
