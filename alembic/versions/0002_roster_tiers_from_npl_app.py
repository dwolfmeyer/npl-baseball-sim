"""roster tiers from the NPL app: restricted, foreign, retired

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-12
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NEW_VALUES = ("restricted", "foreign", "retired")


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        for value in NEW_VALUES:
            op.execute(f"ALTER TYPE roster_tier ADD VALUE IF NOT EXISTS '{value}'")


def downgrade() -> None:
    # Postgres can't drop enum values; delete rows using them so 0001 code still reads cleanly.
    if op.get_bind().dialect.name == "postgresql":
        values = ", ".join(f"'{v}'" for v in NEW_VALUES)
        op.execute(f"DELETE FROM roster_entry WHERE tier::text IN ({values})")
