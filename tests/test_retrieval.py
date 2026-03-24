"""Tests for retrieval helpers."""

from __future__ import annotations

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

