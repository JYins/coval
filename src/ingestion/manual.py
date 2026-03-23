"""Manual text ingestion helpers."""

from __future__ import annotations

from datetime import datetime


def build_manual_conversation(
    raw_content: str,
    language: str,
    conversation_date: datetime | None = None,
) -> dict[str, object]:
    text = raw_content.strip()
    if not text:
        raise ValueError("raw_content is required for manual input")

    return {
        "source_type": "manual",
        "raw_content": text,
        "language": language.strip().lower() or "en",
        "conversation_date": conversation_date,
    }

