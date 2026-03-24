"""Personality profile helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from src.analysis.communication import analyze_communication_style
from src.llm.client import build_llm_client
from src.models.conversation import Conversation
from src.models.person import Person
from src.models.personality_profile import PersonalityProfile
from src.rag.retriever import load_default_config


ROOT_DIR = Path(__file__).resolve().parents[2]
PROMPT_PATH = ROOT_DIR / "configs" / "prompts" / "personality_analysis.txt"


def load_personality_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def load_person_conversations(db: Session, person_id) -> list[Conversation]:
    return (
        db.query(Conversation)
        .filter(Conversation.person_id == person_id)
        .order_by(Conversation.conversation_date.asc(), Conversation.id.asc())
        .all()
    )


def build_personality_prompt(person: Person, conversations: list[Conversation]) -> str:
    prompt = load_personality_prompt()
    conversation_text = "\n".join(item.raw_content for item in conversations if item.raw_content)
    return (
        f"{prompt}\n\n"
        f"Person name: {person.name}\n"
        f"Relationship type: {person.relationship_type}\n"
        f"Notes: {person.notes or 'None'}\n"
        "Conversation history:\n"
        f"{conversation_text}"
    )


def parse_profile_payload(text: str) -> dict[str, Any]:
    data = json.loads(text)
    return {
        "big_five_scores": dict(data.get("big_five_scores", {})),
        "preferences": list(data.get("preferences", [])),
        "topics_of_interest": list(data.get("topics_of_interest", [])),
    }


def refresh_personality_profile(
    db: Session,
    person: Person,
    config: dict[str, Any] | None = None,
) -> PersonalityProfile:
    config_data = dict(config or load_default_config())
    conversations = load_person_conversations(db, person.id)
    if not conversations:
        raise ValueError("person has no conversations yet")

    client = build_llm_client(config_data)
    personality_text = client.generate(
        system_prompt="You analyze conversation history and return JSON only.",
        user_prompt=build_personality_prompt(person, conversations),
    )
    personality_data = parse_profile_payload(personality_text)
    communication_style = analyze_communication_style(
        person=person,
        conversations=conversations,
        config=config_data,
        client=client,
    )

    profile = (
        db.query(PersonalityProfile)
        .filter(PersonalityProfile.person_id == person.id)
        .first()
    )
    if profile is None:
        profile = PersonalityProfile(person_id=person.id)
        db.add(profile)

    profile.big_five_scores = personality_data["big_five_scores"]
    profile.preferences = personality_data["preferences"]
    profile.topics_of_interest = personality_data["topics_of_interest"]
    profile.communication_style = communication_style
    db.commit()
    db.refresh(profile)
    return profile

