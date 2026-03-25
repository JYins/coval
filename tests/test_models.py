"""Tests for basic model construction."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from src.models.chunk import Chunk
from src.models.conversation import Conversation
from src.models.interaction import Interaction
from src.models.person import Person
from src.models.personality_profile import PersonalityProfile
from src.models.user import User


def test_user_and_person_models_hold_expected_fields():
    user = User(email="demo@coval.local", password_hash="hashed-value")
    person = Person(
        user_id=uuid4(),
        name="Alice",
        relationship_type="friend",
        notes="met at a jazz meetup",
    )

    assert user.email == "demo@coval.local"
    assert person.name == "Alice"
    assert person.relationship_type == "friend"
    assert person.notes == "met at a jazz meetup"


def test_conversation_related_models_construct_cleanly():
    person_id = uuid4()
    conversation_id = uuid4()
    now = datetime.now(timezone.utc)

    conversation = Conversation(
        person_id=person_id,
        source_type="manual",
        raw_content="Alice likes quiet coffee chats.",
        conversation_date=now,
        language="en",
    )
    chunk = Chunk(
        conversation_id=conversation_id,
        chunk_text="Alice: Alice likes quiet coffee chats.",
        person_name_prefix="Alice",
        chunk_index=0,
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
    )
    profile = PersonalityProfile(
        person_id=person_id,
        big_five_scores={"openness": 0.7},
        communication_style={"tone": "warm"},
        preferences=["quiet places"],
        topics_of_interest=["coffee"],
    )
    interaction = Interaction(
        person_id=person_id,
        interaction_type="briefing",
        ai_advice_given="Keep the tone relaxed.",
        user_rating=4,
    )

    assert conversation.person_id == person_id
    assert chunk.person_name_prefix == "Alice"
    assert profile.preferences == ["quiet places"]
    assert interaction.interaction_type == "briefing"

