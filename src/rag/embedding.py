"""Embedding wrapper for sentence-transformers."""

from __future__ import annotations

import hashlib
import math
import os

DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MOCK_DIMENSION = 64


class Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self.provider = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers").strip()
        self.mock_dimension = int(
            os.getenv("MOCK_EMBEDDING_DIMENSION", str(DEFAULT_MOCK_DIMENSION))
        )

        if self.provider == "mock":
            self.model = None
            return

        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "mock":
            return [self._mock_embed(text) for text in texts]
        vectors = self.model.encode(texts, convert_to_numpy=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        if not vectors:
            raise ValueError("query text should not be empty")
        return vectors[0]

    def _mock_embed(self, text: str) -> list[float]:
        if not text.strip():
            return [0.0] * self.mock_dimension

        values = []
        for index in range(self.mock_dimension):
            digest = hashlib.sha256(f"{index}:{text}".encode("utf-8")).digest()
            number = int.from_bytes(digest[:4], "big")
            values.append((number / 2**32) * 2 - 1)

        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            return values
        return [value / norm for value in values]

