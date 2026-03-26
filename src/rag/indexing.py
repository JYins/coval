"""Chunk persistence helpers."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from src.models.chunk import Chunk
from src.models.conversation import Conversation
from src.rag.chunking import chunk_conversation
from src.rag.embedding import DEFAULT_EMBEDDING_MODEL
from src.rag.retriever import load_default_config


def build_chunking_settings(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    chunking_config = dict(config.get("chunking", {}))
    strategy = str(chunking_config.get("strategy", "sentence"))

    chunk_kwargs: dict[str, Any] = {}
    if strategy == "sentence":
        chunk_kwargs["max_sentences"] = int(chunking_config.get("max_sentences", 2))
    elif strategy == "fixed":
        chunk_kwargs["chunk_size"] = int(chunking_config.get("chunk_size", 120))
        chunk_kwargs["overlap"] = int(chunking_config.get("overlap", 20))

    return strategy, chunk_kwargs


def build_chunk_rows(
    conversation: Conversation,
    person_name: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    strategy, chunk_kwargs = build_chunking_settings(config)
    rows = chunk_conversation(
        raw_content=conversation.raw_content,
        person_name=person_name,
        strategy=strategy,
        **chunk_kwargs,
    )

    payload = []
    for row in rows:
        item = dict(row)
        item["conversation_id"] = conversation.id
        item["embedding_model"] = str(
            config.get("model_name", DEFAULT_EMBEDDING_MODEL)
        )
        payload.append(item)
    return payload


def save_chunks_for_conversation(
    db: Session,
    conversation: Conversation,
    person_name: str,
    config: dict[str, Any] | None = None,
) -> list[Chunk]:
    config_data = dict(config or load_default_config())
    rows = build_chunk_rows(conversation, person_name, config_data)

    db.query(Chunk).filter(Chunk.conversation_id == conversation.id).delete()

    chunks = []
    for row in rows:
        chunk = Chunk(
            conversation_id=conversation.id,
            chunk_text=str(row["chunk_text"]),
            person_name_prefix=str(row["person_name_prefix"]),
            chunk_index=int(row["chunk_index"]),
            embedding_model=str(row["embedding_model"]),
        )
        db.add(chunk)
        chunks.append(chunk)

    db.commit()
    for chunk in chunks:
        db.refresh(chunk)
    return chunks
