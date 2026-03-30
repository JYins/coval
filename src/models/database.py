"""Database connection helpers for Coval."""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


def normalize_database_url(url: str) -> str:
    value = url.strip()
    if value.startswith("postgres://"):
        return value.replace("postgres://", "postgresql+psycopg2://", 1)
    if value.startswith("postgresql://"):
        return value.replace("postgresql://", "postgresql+psycopg2://", 1)
    return value


DATABASE_URL = normalize_database_url(
    os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/coval",
    )
)


engine_kwargs = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

# for now keep this simple
engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

