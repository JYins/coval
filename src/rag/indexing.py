"""Chunk persistence helpers."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from src.models.chunk import Chunk
from src.models.conversation import Conversation
from src.models.person import Person
from src.rag.chunking import chunk_conversation
from src.rag.embedding import DEFAULT_EMBEDDING_MODEL, Embedder
from src.rag.retriever import load_default_config
from src.rag.vector_store import QdrantVectorStore


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
    sync_chunks_to_vector_store(chunks, config_data)
    return chunks


def sync_chunks_to_vector_store(
    chunks: list[Chunk],
    config: dict[str, Any],
) -> None:
    if not chunks:
        return
    if str(config.get("vector_backend", "memory")).lower() != "qdrant":
        return

    model_name = str(config.get("model_name", DEFAULT_EMBEDDING_MODEL))
    texts = [chunk.chunk_text for chunk in chunks]
    embedder = Embedder(model_name=model_name)
    vectors = embedder.embed_texts(texts)

    rows = []
    for chunk in chunks:
        rows.append(
            {
                "chunk_id": str(chunk.id),
                "conversation_id": str(chunk.conversation_id),
                "chunk_text": chunk.chunk_text,
                "person_name_prefix": chunk.person_name_prefix,
                "chunk_index": chunk.chunk_index,
                "embedding_model": chunk.embedding_model,
            }
        )

    vector_config = dict(config.get("vector_store", {}))
    store = QdrantVectorStore(
        collection_name=str(vector_config.get("collection_name", "conversation_chunks")),
        url=vector_config.get("url"),
        api_key=vector_config.get("api_key"),
        vector_size=len(vectors[0]),
    )
    store.ensure_collection()
    store.upsert_chunks(rows, vectors)


def list_person_conversations(
    db: Session,
    person_id,
) -> list[Conversation]:
    return (
        db.query(Conversation)
        .filter(Conversation.person_id == person_id)
        .order_by(Conversation.conversation_date.asc(), Conversation.id.asc())
        .all()
    )


def rebuild_chunks_for_person(
    db: Session,
    person: Person,
    config: dict[str, Any] | None = None,
) -> dict[str, object]:
    config_data = dict(config or load_default_config())
    conversations = list_person_conversations(db, person.id)

    chunk_count = 0
    for conversation in conversations:
        chunks = save_chunks_for_conversation(
            db=db,
            conversation=conversation,
            person_name=person.name,
            config=config_data,
        )
        chunk_count += len(chunks)

    return {
        "person_id": str(person.id),
        "person_name": person.name,
        "conversation_count": len(conversations),
        "chunk_count": chunk_count,
    }
