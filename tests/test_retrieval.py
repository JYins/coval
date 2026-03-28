"""Tests for retrieval helpers."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import src.rag.indexing as rag_indexing
from src.rag.eval_metrics import mean_metrics, score_query
from src.rag.retriever import search_memory


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        rows = []
        for text in texts:
            lower_text = text.lower()
            if "jazz" in lower_text:
                rows.append([1.0, 0.0])
            elif "coffee" in lower_text:
                rows.append([0.8, 0.2])
            else:
                rows.append([0.0, 1.0])
        return rows

    def embed_query(self, text: str) -> list[float]:
        if "jazz" in text.lower():
            return [1.0, 0.0]
        return [0.0, 1.0]


def test_search_memory_ranks_matching_chunk_first():
    chunks = [
        {"chunk_id": "chunk-1", "chunk_text": "Alice likes jazz bars.", "chunk_index": 0},
        {"chunk_id": "chunk-2", "chunk_text": "Alice prefers morning meetings.", "chunk_index": 1},
    ]

    results = search_memory(
        chunks=chunks,
        query="What jazz place should I pick?",
        embedder=FakeEmbedder(),
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["rank"] == 1
    assert results[0]["score"] > results[1]["score"]


def test_score_query_and_mean_metrics():
    results = [
        {"chunk_id": "chunk-1"},
        {"chunk_id": "chunk-2"},
        {"chunk_id": "chunk-3"},
    ]

    query_scores = score_query(results, {"chunk-2", "chunk-3"}, ks=(1, 2, 3))

    assert query_scores["mrr"] == 0.5
    assert query_scores["recall@1"] == 0.0
    assert query_scores["recall@2"] == 0.5
    assert query_scores["recall@3"] == 1.0

    summary = mean_metrics(
        [
            query_scores,
            {"mrr": 1.0, "recall@1": 1.0, "recall@2": 1.0, "recall@3": 1.0},
        ]
    )

    assert summary["mrr"] == 0.75
    assert summary["recall@1"] == 0.5


def test_sync_chunks_to_vector_store_skips_memory_backend(monkeypatch):
    called = {"upsert": False}

    class FakeStore:
        def __init__(self, *args, **kwargs):
            called["constructed"] = True

        def ensure_collection(self):
            called["ensure"] = True

        def upsert_chunks(self, chunks, vectors):
            called["upsert"] = True

    monkeypatch.setattr(rag_indexing, "QdrantVectorStore", FakeStore)

    chunks = [
        SimpleNamespace(
            id=uuid4(),
            conversation_id=uuid4(),
            chunk_text="Alice likes jazz bars.",
            person_name_prefix="Alice",
            chunk_index=0,
            embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
        )
    ]

    rag_indexing.sync_chunks_to_vector_store(chunks, {"vector_backend": "memory"})

    assert called["upsert"] is False


def test_sync_chunks_to_vector_store_pushes_qdrant_rows(monkeypatch):
    seen = {}

    class FakeEmbedder:
        def __init__(self, model_name):
            seen["model_name"] = model_name

        def embed_texts(self, texts):
            seen["texts"] = list(texts)
            return [[0.1, 0.9] for _ in texts]

    class FakeStore:
        def __init__(self, collection_name, url=None, api_key=None, vector_size=0):
            seen["collection_name"] = collection_name
            seen["vector_size"] = vector_size

        def ensure_collection(self):
            seen["ensured"] = True

        def upsert_chunks(self, chunks, vectors):
            seen["chunks"] = list(chunks)
            seen["vectors"] = list(vectors)

    monkeypatch.setattr(rag_indexing, "Embedder", FakeEmbedder)
    monkeypatch.setattr(rag_indexing, "QdrantVectorStore", FakeStore)

    chunks = [
        SimpleNamespace(
            id=uuid4(),
            conversation_id=uuid4(),
            chunk_text="Alice likes jazz bars.",
            person_name_prefix="Alice",
            chunk_index=0,
            embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
        ),
        SimpleNamespace(
            id=uuid4(),
            conversation_id=uuid4(),
            chunk_text="She prefers quiet coffee chats.",
            person_name_prefix="Alice",
            chunk_index=1,
            embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
        ),
    ]

    rag_indexing.sync_chunks_to_vector_store(
        chunks,
        {
            "vector_backend": "qdrant",
            "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
            "vector_store": {"collection_name": "conversation_chunks"},
        },
    )

    assert seen["model_name"] == "paraphrase-multilingual-MiniLM-L12-v2"
    assert seen["collection_name"] == "conversation_chunks"
    assert seen["vector_size"] == 2
    assert seen["ensured"] is True
    assert len(seen["chunks"]) == 2
    assert len(seen["vectors"]) == 2


def test_rebuild_chunks_for_person_collects_counts(monkeypatch):
    person = SimpleNamespace(id=uuid4(), name="Alice")
    conversations = [
        SimpleNamespace(id=uuid4(), raw_content="one"),
        SimpleNamespace(id=uuid4(), raw_content="two"),
    ]
    seen = {"calls": []}

    monkeypatch.setattr(rag_indexing, "list_person_conversations", lambda db, person_id: conversations)

    def fake_save_chunks_for_conversation(db, conversation, person_name, config=None):
        seen["calls"].append((conversation.id, person_name))
        return [SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())]

    monkeypatch.setattr(rag_indexing, "save_chunks_for_conversation", fake_save_chunks_for_conversation)

    summary = rag_indexing.rebuild_chunks_for_person(
        db=object(),
        person=person,
        config={"vector_backend": "memory"},
    )

    assert summary["person_name"] == "Alice"
    assert summary["conversation_count"] == 2
    assert summary["chunk_count"] == 4
    assert len(seen["calls"]) == 2

