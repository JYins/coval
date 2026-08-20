"""Apply versioned PostgreSQL schema migrations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
MIGRATION_DIR = ROOT / "scripts" / "migrations"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Coval database migrations")
    parser.add_argument(
        "--list",
        action="store_true",
        help="list migration names without connecting to the database",
    )
    return parser.parse_args()


def migration_files() -> list[Path]:
    return sorted(MIGRATION_DIR.glob("*.sql"))


def apply_migrations() -> list[str]:
    from src.models import Base
    from src.models.database import DATABASE_URL, engine

    # create_all bootstraps a fresh database; SQL files upgrade existing tables
    Base.metadata.create_all(bind=engine)
    if not DATABASE_URL.startswith("postgresql"):
        print("non-PostgreSQL database: create_all complete, SQL migrations skipped")
        return []

    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "name VARCHAR(255) PRIMARY KEY, "
                "applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
                ")"
            )
        )

    applied = []
    for path in migration_files():
        with engine.begin() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM schema_migrations WHERE name = :name"),
                {"name": path.name},
            ).scalar()
            if exists:
                continue
            conn.exec_driver_sql(path.read_text(encoding="utf-8"))
            conn.execute(
                text("INSERT INTO schema_migrations (name) VALUES (:name)"),
                {"name": path.name},
            )
        applied.append(path.name)
    return applied


def main() -> None:
    args = parse_args()
    if args.list:
        for path in migration_files():
            print(path.name)
        return
    applied = apply_migrations()
    if applied:
        for name in applied:
            print(f"applied migration: {name}")
    else:
        print("database migrations are current")


if __name__ == "__main__":
    main()
