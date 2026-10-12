"""Database layer: SQLAlchemy models and session helpers."""

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

DEFAULT_URL = "postgresql+psycopg://localhost/npl_sim"


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_URL)


def get_engine(url: str | None = None) -> Engine:
    return create_engine(url or database_url())
