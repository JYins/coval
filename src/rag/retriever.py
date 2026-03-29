"""Retriever helpers for person-level RAG."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from sqlalchemy.orm import Session

from src.models.chunk import Chunk
from src.models.conversation import Conversation
from src.models.person import Person
from src.rag.chunking import chunk_conversation
from src.rag.embedding import DEFAULT_EMBEDDING_MODEL, Embedder

try:
    from src.rag.vector_store import QdrantVectorStore
except ModuleNotFoundError:
    QdrantVectorStore = None


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "default.yaml"


def load_default_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return dict(data)


def build_person_summary(person: Person) -> str:
    lines = [
        f"name: {person.name}",
        f"relationship_type: {person.relationship_type}",
    ]
    if person.notes:
        lines.append(f"notes: {person.notes}")
    if person.last_contact:
        lines.append(f"last_contact: {person.last_contact.isoformat()}")
    return "\n".join(lines)


def load_saved_chunks(db: Session, person_id: UUID) -> list[dict[str, Any]]:
    rows = (
        db.query(Chunk, Conversation)
        .join(Conversation, Chunk.conversation_id == Conversation.id)
        .filter(Conversation.person_id == person_id)
        .order_by(Conversation.conversation_date.asc(), Chunk.chunk_index.asc())
        .all()
    )

    payload = []
    for chunk, conversation in rows:
        payload.append(
            {
                "chunk_id": str(chunk.id),
                "conversation_id": str(conversation.id),
                "person_id": str(person_id),
                "chunk_text": chunk.chunk_text,
                "person_name_prefix": chunk.person_name_prefix,
                "chunk_index": chunk.chunk_index,
                "embedding_model": chunk.embedding_model,
                "conversation_date": conversation.conversation_date.isoformat(),
            }
        )
    return payload


def build_runtime_chunks(
    db: Session,
    person: Person,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    saved_chunks = load_saved_chunks(db, person.id)
    if saved_chunks:
        return saved_chunks

    chunking_config = dict(config.get("chunking", {}))
    strategy = str(chunking_config.get("strategy", "sentence"))
    max_sentences = int(chunking_config.get("max_sentences", 2))
    chunk_size = int(chunking_config.get("chunk_size", 120))
    overlap = int(chunking_config.get("overlap", 20))

    conversations = (
        db.query(Conversation)
        .filter(Conversation.person_id == person.id)
        .order_by(Conversation.conversation_date.asc(), Conversation.id.asc())
        .all()
    )
    if not conversations:
        raise ValueError("person has no conversations yet")

    rows = []
    for conversation in conversations:
        chunks = chunk_conversation(
            raw_content=conversation.raw_content,
            person_name=person.name,
            strategy=strategy,
            max_sentences=max_sentences,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        for chunk in chunks:
            row = dict(chunk)
            row["chunk_id"] = f"{conversation.id}_{chunk['chunk_index']}"
            row["conversation_id"] = str(conversation.id)
            row["person_id"] = str(person.id)
            row["embedding_model"] = str(
                config.get("model_name", DEFAULT_EMBEDDING_MODEL)
            )
            row["conversation_date"] = conversation.conversation_date.isoformat()
            rows.append(row)
    return rows


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def search_memory(
    chunks: list[dict[str, Any]],
    query: str,
    embedder: Embedder,
    top_k: int,
) -> list[dict[str, Any]]:
    if not chunks:
        return []

    texts = [row["chunk_text"] for row in chunks]
    chunk_vectors = embedder.embed_texts(texts)
    query_vector = embedder.embed_query(query)

    scored_rows = []
    for chunk, vector in zip(chunks, chunk_vectors):
        row = dict(chunk)
        row["score"] = cosine_similarity(vector, query_vector)
        scored_rows.append(row)

    scored_rows.sort(key=lambda item: item["score"], reverse=True)
    picked = []
    for rank, row in enumerate(scored_rows[:top_k], start=1):
        picked_row = dict(row)
        picked_row["rank"] = rank
        picked.append(picked_row)
    return picked


def search_qdrant(
    chunks: list[dict[str, Any]],
    query: str,
    embedder: Embedder,
    top_k: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    store_class = QdrantVectorStore
    if store_class is None:
        from src.rag.vector_store import QdrantVectorStore as store_class

    if not chunks:
        return []

    texts = [row["chunk_text"] for row in chunks]
    chunk_vectors = embedder.embed_texts(texts)
    query_vector = embedder.embed_query(query)

    vector_config = dict(config.get("vector_store", {}))
    store = store_class(
        collection_name=str(
            vector_config.get("collection_name", "conversation_chunks")
        ),
        url=vector_config.get("url"),
        api_key=vector_config.get("api_key"),
        vector_size=len(query_vector),
    )
    store.recreate_collection()
    store.upsert_chunks(chunks, chunk_vectors)

    matches = store.search(query_vector, top_k=top_k)
    picked = []
    for rank, row in enumerate(matches, start=1):
        picked_row = dict(row)
        picked_row["rank"] = rank
        picked.append(picked_row)
    return picked


def retrieve_chunks(
    db: Session,
    person: Person,
    query: str,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    config_data = dict(config or load_default_config())
    top_k = int(config_data.get("top_k", 3))
    model_name = str(config_data.get("model_name", DEFAULT_EMBEDDING_MODEL))
    embedder = Embedder(model_name=model_name)

    chunks = build_runtime_chunks(db, person, config_data)
    vector_backend = str(config_data.get("vector_backend", "memory")).lower()
    if vector_backend == "qdrant":
        return search_qdrant(chunks, query, embedder, top_k, config_data)
    if vector_backend == "memory":
        return search_memory(chunks, query, embedder, top_k)
    raise ValueError(f"unknown vector backend: {vector_backend}")


def format_context_blocks(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No retrieved context."

    lines = []
    for item in results:
        score = float(item.get("score", 0.0))
        lines.append(
            f"[{item['rank']}] score={score:.3f} chunk={item['chunk_index']} "
            f"text={item['chunk_text']}"
        )
    return "\n".join(lines)


def build_prompt(
    person: Person,
    question: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    system_prompt = (
        "You are a relationship analyst. Use the retrieved context first, "
        "stay practical, and do not invent personal history that is not in the notes."
    )
    user_prompt = "\n\n".join(
        [
            "Person summary:",
            build_person_summary(person),
            "Retrieved conversation chunks:",
            format_context_blocks(results),
            f"Question: {question}",
        ]
    )
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "retrieved_chunks": results,
    }


def run_retrieval(
    db: Session,
    person: Person,
    question: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results = retrieve_chunks(db, person, question, config=config)
    return build_prompt(person, question, results)

