"""Run retrieval eval on a small relationship test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval eval for Coval")
    parser.add_argument("--config", default="configs/eval.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    config_path = ROOT_DIR / path
    with config_path.open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def load_eval_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    data_path = ROOT_DIR / str(config["data_path"])
    with data_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    return list(rows)


def slugify_name(name: str) -> str:
    value = name.strip().lower().replace(" ", "_")
    return "".join(char for char in value if char.isalnum() or char == "_")


def build_eval_chunks(row: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    from src.rag.chunking import chunk_conversation

    chunking_config = dict(config.get("chunking", {}))
    strategy = str(chunking_config.get("strategy", "sentence"))
    max_sentences = int(chunking_config.get("max_sentences", 2))
    chunk_size = int(chunking_config.get("chunk_size", 120))
    overlap = int(chunking_config.get("overlap", 20))

    person_slug = slugify_name(str(row["person_name"]))
    chunks = []
    for conversation_index, raw_content in enumerate(row["conversations"]):
        chunk_kwargs: dict[str, Any] = {}
        if strategy == "sentence":
            chunk_kwargs["max_sentences"] = max_sentences
        elif strategy == "fixed":
            chunk_kwargs["chunk_size"] = chunk_size
            chunk_kwargs["overlap"] = overlap

        piece_rows = chunk_conversation(
            raw_content=raw_content,
            person_name=str(row["person_name"]),
            strategy=strategy,
            **chunk_kwargs,
        )
        for piece in piece_rows:
            chunk_row = dict(piece)
            chunk_row["chunk_id"] = (
                f"{person_slug}_{conversation_index}_{piece['chunk_index']}"
            )
            chunk_row["person_id"] = person_slug
            chunks.append(chunk_row)
    return chunks


def run_one_query(
    row: dict[str, Any],
    config: dict[str, Any],
    embedder,
) -> dict[str, Any]:
    from src.rag.eval_metrics import score_query
    from src.rag.retriever import search_memory

    chunks = build_eval_chunks(row, config)
    top_k = int(config.get("top_k", 5))
    results = search_memory(chunks, str(row["question"]), embedder, top_k=top_k)
    metrics = score_query(
        results,
        gold_chunk_ids={str(item) for item in row["gold_chunk_ids"]},
        ks=tuple(int(value) for value in config.get("ks", [1, 3, 5])),
    )
    return {
        "person_name": row["person_name"],
        "question": row["question"],
        "gold_chunk_ids": row["gold_chunk_ids"],
        "retrieved_chunk_ids": [item["chunk_id"] for item in results],
        "retrieved_chunks": [
            {
                "chunk_id": item["chunk_id"],
                "score": round(float(item["score"]), 4),
                "chunk_text": item["chunk_text"],
            }
            for item in results
        ],
        **metrics,
    }


def write_results(
    summary: dict[str, float],
    query_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    metrics_path = ROOT_DIR / str(config["metrics_output"])
    per_query_path = ROOT_DIR / str(config["per_query_output"])

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    per_query_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(
        [{"metric": key, "value": round(value, 4)} for key, value in summary.items()]
    )
    metrics_df.to_csv(metrics_path, index=False)

    with per_query_path.open("w", encoding="utf-8") as handle:
        json.dump(query_rows, handle, ensure_ascii=False, indent=2)


def main() -> None:
    from src.rag.embedding import DEFAULT_EMBEDDING_MODEL, Embedder
    from src.rag.eval_metrics import mean_metrics

    args = parse_args()
    config = load_config(args.config)
    rows = load_eval_rows(config)

    model_name = str(config.get("model_name", DEFAULT_EMBEDDING_MODEL))
    embedder = Embedder(model_name=model_name)

    query_rows = [run_one_query(row, config, embedder) for row in rows]
    score_rows = [
        {
            "mrr": float(row["mrr"]),
            "recall@1": float(row["recall@1"]),
            "recall@3": float(row["recall@3"]),
            "recall@5": float(row["recall@5"]),
        }
        for row in query_rows
    ]
    summary = mean_metrics(score_rows)
    write_results(summary, query_rows, config)

    print("eval done")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()

