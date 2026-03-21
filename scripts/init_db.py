"""Create the local PostgreSQL database tables."""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from src.models import Base
from src.models.database import DATABASE_URL, engine

# import models so metadata sees all tables
from src.models.chunk import Chunk  # noqa: F401
from src.models.conversation import Conversation  # noqa: F401
from src.models.interaction import Interaction  # noqa: F401
from src.models.person import Person  # noqa: F401
from src.models.personality_profile import PersonalityProfile  # noqa: F401
from src.models.user import User  # noqa: F401


def ensure_database(url_text: str) -> None:
    url = make_url(url_text)
    if not url.drivername.startswith("postgresql"):
        return

    db_name = url.database
    if not db_name:
        raise ValueError("database name is missing in DATABASE_URL")

    safe_db_name = db_name.replace('"', "")
    admin_url = url.set(database="postgres")
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")

    with admin_engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"),
            {"name": safe_db_name},
        ).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{safe_db_name}"'))
            print(f"created database: {safe_db_name}")


def main() -> None:
    ensure_database(DATABASE_URL)
    Base.metadata.create_all(bind=engine)

    table_names = sorted(Base.metadata.tables.keys())
    print("created tables:")
    for name in table_names:
        print(f"- {name}")


if __name__ == "__main__":
    main()

