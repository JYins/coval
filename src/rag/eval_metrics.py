"""Retrieval metrics for relationship eval."""

from __future__ import annotations

from statistics import mean


# reused from rag-eval-pipeline/src/eval_metrics.py, adapted for chunk ids
def top_chunk_ids(results: list[dict[str, object]], top_k: int) -> list[str]:
    if top_k <= 0:
        raise ValueError("top_k should be > 0")
    return [str(item["chunk_id"]) for item in results[:top_k]]


def recall_at_k(
    results: list[dict[str, object]],
    gold_chunk_ids: set[str],
    top_k: int,
) -> float:
    if not gold_chunk_ids:
        return 0.0
    picked = set(top_chunk_ids(results, top_k))
    return len(picked & gold_chunk_ids) / len(gold_chunk_ids)


def reciprocal_rank(
    results: list[dict[str, object]],
    gold_chunk_ids: set[str],
) -> float:
    for rank, item in enumerate(results, start=1):
        if str(item["chunk_id"]) in gold_chunk_ids:
            return 1.0 / rank
    return 0.0


def score_query(
    results: list[dict[str, object]],
    gold_chunk_ids: set[str],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    metrics = {"mrr": reciprocal_rank(results, gold_chunk_ids)}
    for top_k in ks:
        metrics[f"recall@{top_k}"] = recall_at_k(results, gold_chunk_ids, top_k)
    return metrics


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("rows should not be empty")
    keys = rows[0].keys()
    return {key: mean(row[key] for row in rows) for key in keys}

