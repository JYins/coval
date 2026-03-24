"""Seed a few demo rows for local development."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def build_seed_rows() -> list[dict[str, object]]:
    now = datetime.now(timezone.utc)
    return [
        {
            "name": "Alice",
            "relationship_type": "friend",
            "notes": "Met at a jazz meetup downtown.",
            "first_met": now - timedelta(days=120),
            "last_contact": now - timedelta(days=3),
            "conversations": [
                {
                    "source_type": "manual",
                    "language": "en",
                    "raw_content": (
                        "Alice likes jazz bars and quiet coffee chats. "
                        "She does not like crowded clubs."
                    ),
                },
                {
                    "source_type": "manual",
                    "language": "en",
                    "raw_content": (
                        "She prefers Friday evening hangouts and live piano."
                    ),
                },
            ],
        },
        {
            "name": "Ben",
            "relationship_type": "client",
            "notes": "Startup founder client from Toronto.",
            "first_met": now - timedelta(days=90),
            "last_contact": now - timedelta(days=5),
            "conversations": [
                {
                    "source_type": "manual",
                    "language": "en",
                    "raw_content": (
                        "Ben prefers short morning calls and bullet point follow-ups."
                    ),
                },
                {
                    "source_type": "manual",
                    "language": "en",
                    "raw_content": (
                        "He dislikes long small talk before business and wants clear next steps."
                    ),
                },
            ],
        },
        {
            "name": "Emily",
            "relationship_type": "family",
            "notes": "Family catch-up sample row.",
            "first_met": now - timedelta(days=365),
            "last_contact": now - timedelta(days=1),
            "conversations": [
                {
                    "source_type": "manual",
                    "language": "mixed",
                    "raw_content": (
                        "Emily likes tea, park walks, and quiet restaurants. "
                        "She speaks Chinese with her parents."
                    ),
                },
                {
                    "source_type": "manual",
                    "language": "mixed",
                    "raw_content": (
                        "She values checking in on health first before talking about work."
                    ),
                },
            ],
        },
    ]


def main() -> None:
    from src.analysis.personality import refresh_personality_profile
    from src.api.deps import hash_password
    from src.models.conversation import Conversation
    from src.models.database import SessionLocal
    from src.models.person import Person
    from src.models.user import User

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == "demo@coval.local").first()
        if user is None:
            user = User(
                email="demo@coval.local",
                password_hash=hash_password("demo-password"),
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        rows = build_seed_rows()
        created_persons = 0
        created_conversations = 0

        for row in rows:
            person = (
                db.query(Person)
                .filter(Person.user_id == user.id, Person.name == row["name"])
                .first()
            )
            if person is None:
                person = Person(
                    user_id=user.id,
                    name=str(row["name"]),
                    relationship_type=str(row["relationship_type"]),
                    notes=str(row["notes"]),
                    first_met=row["first_met"],
                    last_contact=row["last_contact"],
                )
                db.add(person)
                db.commit()
                db.refresh(person)
                created_persons += 1

            existing_count = (
                db.query(Conversation)
                .filter(Conversation.person_id == person.id)
                .count()
            )
            if existing_count == 0:
                for item in row["conversations"]:
                    conversation = Conversation(
                        person_id=person.id,
                        source_type=str(item["source_type"]),
                        language=str(item["language"]),
                        raw_content=str(item["raw_content"]),
                    )
                    db.add(conversation)
                    created_conversations += 1
                db.commit()

            refresh_personality_profile(db, person)

        print("seed done")
        print(f"user_email: {user.email}")
        print(f"persons_created: {created_persons}")
        print(f"conversations_created: {created_conversations}")
    finally:
        db.close()


if __name__ == "__main__":
    main()

