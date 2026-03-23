"""Qdrant vector store wrapper."""

from __future__ import annotations

import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


DEFAULT_COLLECTION_NAME = "conversation_chunks"


class QdrantVectorStore:
    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        url: str | None = None,
        api_key: str | None = None,
        vector_size: int = 384,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(
            url=url or os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=api_key or os.getenv("QDRANT_API_KEY"),
        )

    def recreate_collection(self) -> None:
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

    def upsert_chunks(self, chunks: list[dict[str, Any]], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors should have the same length")

        points = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            chunk_id = chunk.get("chunk_id") or f"{self.collection_name}_{index}"
            payload = dict(chunk)
            points.append(
                PointStruct(
                    id=str(chunk_id),
                    vector=vector,
                    payload=payload,
                )
            )

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )

        matches = []
        for item in results.points:
            row = dict(item.payload or {})
            row["score"] = item.score
            row["chunk_id"] = item.id
            matches.append(row)
        return matches

