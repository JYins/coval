"""Qdrant vector store wrapper."""

from __future__ import annotations

import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


DEFAULT_COLLECTION_NAME = "conversation_chunks"


def build_tenant_filter(
    user_id: str,
    person_id: str,
    conversation_id: str | None = None,
) -> Filter:
    conditions = [
        FieldCondition(
            key="user_id",
            match=MatchValue(value=str(user_id)),
        ),
        FieldCondition(
            key="person_id",
            match=MatchValue(value=str(person_id)),
        ),
    ]
    if conversation_id is not None:
        conditions.append(
            FieldCondition(
                key="conversation_id",
                match=MatchValue(value=str(conversation_id)),
            )
        )
    return Filter(must=conditions)


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
        effective_url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        if effective_url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(
                url=effective_url,
                api_key=api_key or os.getenv("QDRANT_API_KEY"),
            )

    def ensure_collection(self) -> None:
        if self.client.collection_exists(collection_name=self.collection_name):
            return
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
        except UnexpectedResponse as exc:
            # another worker may create the shared collection at the same time
            if exc.status_code != 409:
                raise

    def upsert_chunks(self, chunks: list[dict[str, Any]], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors should have the same length")

        points = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            missing = {"user_id", "person_id"} - chunk.keys()
            if missing:
                fields = ", ".join(sorted(missing))
                raise ValueError(f"chunk payload missing tenant fields: {fields}")
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

    def delete_conversation_chunks(
        self,
        *,
        user_id: str,
        person_id: str,
        conversation_id: str,
    ) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=build_tenant_filter(
                user_id=user_id,
                person_id=person_id,
                conversation_id=conversation_id,
            ),
            wait=True,
        )

    def search(
        self,
        query_vector: list[float],
        *,
        user_id: str,
        person_id: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        tenant_filter = build_tenant_filter(
            user_id=user_id,
            person_id=person_id,
        )
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=tenant_filter,
            limit=top_k,
        )

        matches = []
        for item in results.points:
            row = dict(item.payload or {})
            row["score"] = item.score
            row["chunk_id"] = item.id
            matches.append(row)
        return matches

