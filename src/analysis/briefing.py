"""Briefing generator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from src.llm.client import build_llm_client
from src.models.person import Person
from src.models.personality_profile import PersonalityProfile
from src.rag.retriever import load_default_config, run_retrieval


ROOT_DIR = Path(__file__).resolve().parents[2]
PROMPT_PATH = ROOT_DIR / "configs" / "prompts" / "briefing.txt"


def load_briefing_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def load_personality_profile(db: Session, person_id) -> PersonalityProfile | None:
    return (
        db.query(PersonalityProfile)
        .filter(PersonalityProfile.person_id == person_id)
        .first()
    )


def format_profile_summary(profile: PersonalityProfile | None) -> str:
    if profile is None:
        return "No personality profile yet."

    return "\n".join(
        [
            f"big_five_scores: {profile.big_five_scores}",
            f"communication_style: {profile.communication_style}",
            f"preferences: {profile.preferences}",
            f"topics_of_interest: {profile.topics_of_interest}",
        ]
    )


def generate_person_briefing(
    db: Session,
    person: Person,
    top_k: int | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_data = dict(config or load_default_config())
    if top_k is not None:
        config_data["top_k"] = top_k

    question = f"What should I remember before I meet {person.name} next time?"
    retrieval_data = run_retrieval(db, person, question, config=config_data)
    profile = load_personality_profile(db, person.id)

    prompt = "\n\n".join(
        [
            load_briefing_prompt(),
            f"Person name: {person.name}",
            f"Relationship type: {person.relationship_type}",
            f"Notes: {person.notes or 'None'}",
            "Personality profile:",
            format_profile_summary(profile),
            "Retrieved context:",
            str(retrieval_data["user_prompt"]),
        ]
    )
    client = build_llm_client(config_data)
    briefing = client.generate(
        system_prompt="You write concise pre-meeting briefings.",
        user_prompt=prompt,
    )
    return {
        "briefing": briefing,
        "retrieved_chunks": retrieval_data["retrieved_chunks"],
    }

