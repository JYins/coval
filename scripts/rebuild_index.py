"""Rebuild saved chunks for existing conversations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild chunk index for existing persons")
    parser.add_argument("--user-email", default="")
    parser.add_argument("--person-name", default="")
    return parser.parse_args()


def main() -> None:
    from src.models.database import SessionLocal
    from src.models.person import Person
    from src.models.user import User
    from src.rag.indexing import rebuild_chunks_for_person
    from src.rag.retriever import load_default_config

    args = parse_args()
    config = load_default_config()

    db = SessionLocal()
    try:
        query = db.query(Person)
        if args.user_email:
            query = query.join(User).filter(User.email == args.user_email)
        if args.person_name:
            query = query.filter(Person.name == args.person_name)

        persons = query.order_by(Person.name.asc()).all()
        if not persons:
            print("no matching persons found")
            return

        total_persons = 0
        total_conversations = 0
        total_chunks = 0

        for person in persons:
            summary = rebuild_chunks_for_person(db, person, config=config)
            total_persons += 1
            total_conversations += int(summary["conversation_count"])
            total_chunks += int(summary["chunk_count"])
            print(
                f"rebuilt {summary['person_name']}: "
                f"{summary['conversation_count']} conversations, "
                f"{summary['chunk_count']} chunks"
            )

        print("rebuild done")
        print(f"persons: {total_persons}")
        print(f"conversations: {total_conversations}")
        print(f"chunks: {total_chunks}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
