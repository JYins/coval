"""Communication style helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.llm.client import LLMClient, build_llm_client
from src.models.conversation import Conversation
from src.models.person import Person
from src.rag.retriever import load_default_config


ROOT_DIR = Path(__file__).resolve().parents[2]
PROMPT_PATH = ROOT_DIR / "configs" / "prompts" / "communication_advice.txt"


def load_communication_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_communication_prompt(person: Person, conversations: list[Conversation]) -> str:
    prompt = load_communication_prompt()
    conversation_text = "\n".join(item.raw_content for item in conversations if item.raw_content)
    return (
        f"{prompt}\n\n"
        f"Person name: {person.name}\n"
        f"Relationship type: {person.relationship_type}\n"
        "Conversation history:\n"
        f"{conversation_text}"
    )


def analyze_communication_style(
    person: Person,
    conversations: list[Conversation],
    config: dict[str, Any] | None = None,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    config_data = dict(config or load_default_config())
    llm_client = client or build_llm_client(config_data)
    text = llm_client.generate(
        system_prompt="You analyze communication style and return JSON only.",
        user_prompt=build_communication_prompt(person, conversations),
    )
    data = json.loads(text)
    return dict(data.get("communication_style", {}))

