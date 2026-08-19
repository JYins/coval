"""Tests for retrieval helpers."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import src.rag.indexing as rag_indexing
import src.rag.retriever as rag_retriever
from src.rag.eval_metrics import mean_metrics, score_query
from src.rag.retriever import search_memory
from src.rag.vector_store import QdrantVectorStore


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

    rag_indexing.sync_chunks_to_vector_store(
        chunks,
        {"vector_backend": "memory"},
        user_id=uuid4(),
        person_id=uuid4(),
    )

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

        def delete_conversation_chunks(
            self,
            *,
            user_id,
            person_id,
            conversation_id,
        ):
            seen.setdefault("deleted", []).append(
                (user_id, person_id, conversation_id)
            )

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
    user_id = uuid4()
    person_id = uuid4()

    rag_indexing.sync_chunks_to_vector_store(
        chunks,
        {
            "vector_backend": "qdrant",
            "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
            "vector_store": {"collection_name": "conversation_chunks"},
        },
        user_id=user_id,
        person_id=person_id,
    )

    assert seen["model_name"] == "paraphrase-multilingual-MiniLM-L12-v2"
    assert seen["collection_name"] == "conversation_chunks"
    assert seen["vector_size"] == 2
    assert seen["ensured"] is True
    assert len(seen["chunks"]) == 2
    assert len(seen["vectors"]) == 2
    assert {row["user_id"] for row in seen["chunks"]} == {str(user_id)}
    assert {row["person_id"] for row in seen["chunks"]} == {str(person_id)}
    assert {row[:2] for row in seen["deleted"]} == {
        (str(user_id), str(person_id))
    }
    assert {row[2] for row in seen["deleted"]} == {
        str(chunk.conversation_id) for chunk in chunks
    }


def test_qdrant_store_search_adds_user_and_person_filter():
    seen = {}

    class FakeClient:
        def query_points(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(points=[])

    store = object.__new__(QdrantVectorStore)
    store.collection_name = "conversation_chunks"
    store.client = FakeClient()

    results = store.search(
        [0.1, 0.9],
        user_id="user-1",
        person_id="person-1",
        top_k=3,
    )

    assert results == []
    assert seen["collection_name"] == "conversation_chunks"
    assert seen["limit"] == 3
    conditions = seen["query_filter"].must
    assert [(item.key, item.match.value) for item in conditions] == [
        ("user_id", "user-1"),
        ("person_id", "person-1"),
    ]


def test_qdrant_local_store_isolates_search_and_delete_by_tenant():
    store = QdrantVectorStore(
        collection_name="tenant_isolation_test",
        url=":memory:",
        vector_size=2,
    )
    store.ensure_collection()
    first_id = str(uuid4())
    second_id = str(uuid4())
    store.upsert_chunks(
        [
            {
                "chunk_id": first_id,
                "conversation_id": "shared-conversation",
                "user_id": "user-1",
                "person_id": "person-1",
                "chunk_text": "private row one",
            },
            {
                "chunk_id": second_id,
                "conversation_id": "shared-conversation",
                "user_id": "user-2",
                "person_id": "person-1",
                "chunk_text": "private row two",
            },
        ],
        [[1.0, 0.0], [1.0, 0.0]],
    )

    first_results = store.search(
        [1.0, 0.0],
        user_id="user-1",
        person_id="person-1",
        top_k=5,
    )
    second_results = store.search(
        [1.0, 0.0],
        user_id="user-2",
        person_id="person-1",
        top_k=5,
    )
    assert [row["chunk_id"] for row in first_results] == [first_id]
    assert [row["chunk_id"] for row in second_results] == [second_id]

    store.delete_conversation_chunks(
        user_id="user-1",
        person_id="person-1",
        conversation_id="shared-conversation",
    )

    assert store.search(
        [1.0, 0.0],
        user_id="user-1",
        person_id="person-1",
        top_k=5,
    ) == []
    remaining = store.search(
        [1.0, 0.0],
        user_id="user-2",
        person_id="person-1",
        top_k=5,
    )
    assert [row["chunk_id"] for row in remaining] == [second_id]


def test_search_qdrant_only_queries_existing_index(monkeypatch):
    seen = {}

    class QueryEmbedder:
        def embed_query(self, text):
            seen["query"] = text
            return [0.2, 0.8]

        def embed_texts(self, texts):
            raise AssertionError("query path should not re-embed saved chunks")

    class FakeStore:
        def __init__(self, collection_name, url=None, api_key=None, vector_size=0):
            seen["collection_name"] = collection_name
            seen["vector_size"] = vector_size

        def recreate_collection(self):
            raise AssertionError("query path should not recreate collection")

        def upsert_chunks(self, chunks, vectors):
            raise AssertionError("query path should not upsert chunks")

        def search(self, query_vector, *, user_id, person_id, top_k):
            seen["search"] = (query_vector, user_id, person_id, top_k)
            return [
                {
                    "chunk_id": "chunk-1",
                    "chunk_text": "Alice likes jazz bars.",
                    "chunk_index": 0,
                    "score": 0.91,
                }
            ]

    monkeypatch.setattr(rag_retriever, "QdrantVectorStore", FakeStore)

    results = rag_retriever.search_qdrant(
        "Where should we go?",
        QueryEmbedder(),
        3,
        {"vector_store": {"collection_name": "conversation_chunks"}},
        user_id=uuid4(),
        person_id=uuid4(),
    )

    assert results[0]["rank"] == 1
    assert seen["query"] == "Where should we go?"
    assert seen["vector_size"] == 2
    assert seen["search"][3] == 3


def test_rebuild_chunks_for_person_collects_counts(monkeypatch):
    person = SimpleNamespace(id=uuid4(), user_id=uuid4(), name="Alice")
    conversations = [
        SimpleNamespace(id=uuid4(), raw_content="one"),
        SimpleNamespace(id=uuid4(), raw_content="two"),
    ]
    seen = {"calls": []}

    monkeypatch.setattr(rag_indexing, "list_person_conversations", lambda db, person_id: conversations)

    def fake_save_chunks_for_conversation(
        db,
        conversation,
        person_name,
        config=None,
        *,
        user_id,
        person_id,
    ):
        seen["calls"].append(
            (conversation.id, person_name, user_id, person_id)
        )
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
    assert {call[2] for call in seen["calls"]} == {person.user_id}
    assert {call[3] for call in seen["calls"]} == {person.id}

