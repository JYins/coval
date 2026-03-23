"""Chunking helpers for conversation retrieval."""

from __future__ import annotations

import re
from typing import Any

from src.rag.cleaning import clean_text, has_cjk


# reused from rag-eval-pipeline/src/chunking.py, adapted for conversation data
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def split_sentences(text: str) -> list[str]:
    value = clean_text(text)
    if not value:
        return []

    parts = [clean_text(part) for part in SENTENCE_SPLIT_RE.split(value)]
    parts = [part for part in parts if part]
    if parts:
        return parts
    return [value]


def get_fixed_units(text: str) -> tuple[list[str], str]:
    value = clean_text(text)
    if not value:
        return [], " "

    words = value.split()
    if len(words) > 1 or not has_cjk(value):
        return words, " "
    return list(value), ""


def make_chunk(
    person_name: str,
    text: str,
    chunk_index: int,
    strategy: str,
    sentence_ids: list[int] | None = None,
    paragraph_index: int | None = None,
    include_person_name: bool = True,
) -> dict[str, Any]:
    chunk_text = clean_text(text)
    if not chunk_text:
        raise ValueError("chunk text should not be empty")

    person_name_prefix = clean_text(person_name)
    if include_person_name and person_name_prefix:
        # person name prefix = sermon title trick
        chunk_text = clean_text(f"{person_name_prefix}: {chunk_text}")

    return {
        "chunk_index": chunk_index,
        "person_name_prefix": person_name_prefix,
        "chunk_text": chunk_text,
        "strategy": strategy,
        "sentence_ids": sentence_ids or [],
        "paragraph_index": paragraph_index,
    }


def chunk_fixed_size(
    raw_content: str,
    person_name: str,
    chunk_size: int = 120,
    overlap: int = 20,
    include_person_name: bool = True,
) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size should be > 0")
    if overlap < 0:
        raise ValueError("overlap should be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap should be smaller than chunk_size")

    units, joiner = get_fixed_units(raw_content)
    if not units:
        return []

    chunks = []
    step = chunk_size - overlap
    for chunk_index, start in enumerate(range(0, len(units), step)):
        piece = units[start : start + chunk_size]
        if not piece:
            continue
        chunks.append(
            make_chunk(
                person_name=person_name,
                text=joiner.join(piece),
                chunk_index=chunk_index,
                strategy="fixed",
                include_person_name=include_person_name,
            )
        )
    return chunks


def chunk_by_sentence(
    raw_content: str,
    person_name: str,
    max_sentences: int = 3,
    include_person_name: bool = True,
) -> list[dict[str, Any]]:
    if max_sentences <= 0:
        raise ValueError("max_sentences should be > 0")

    sentences = split_sentences(raw_content)
    if not sentences:
        return []

    chunks = []
    for chunk_index, start in enumerate(range(0, len(sentences), max_sentences)):
        piece = sentences[start : start + max_sentences]
        sentence_ids = list(range(start, start + len(piece)))
        chunks.append(
            make_chunk(
                person_name=person_name,
                text=" ".join(piece),
                chunk_index=chunk_index,
                strategy="sentence",
                sentence_ids=sentence_ids,
                include_person_name=include_person_name,
            )
        )
    return chunks


def chunk_by_paragraph(
    raw_content: str,
    person_name: str,
    include_person_name: bool = True,
) -> list[dict[str, Any]]:
    text = str(raw_content)
    if not text.strip():
        return []

    parts = [clean_text(part) for part in text.split("\n\n")]
    parts = [part for part in parts if part]
    if not parts:
        return []

    chunks = []
    for chunk_index, part in enumerate(parts):
        chunks.append(
            make_chunk(
                person_name=person_name,
                text=part,
                chunk_index=chunk_index,
                strategy="paragraph",
                paragraph_index=chunk_index,
                include_person_name=include_person_name,
            )
        )
    return chunks


def chunk_conversation(
    raw_content: str,
    person_name: str,
    strategy: str = "sentence",
    **kwargs: Any,
) -> list[dict[str, Any]]:
    if strategy == "fixed":
        return chunk_fixed_size(raw_content, person_name, **kwargs)
    if strategy == "sentence":
        return chunk_by_sentence(raw_content, person_name, **kwargs)
    if strategy == "paragraph":
        return chunk_by_paragraph(raw_content, person_name, **kwargs)
    raise ValueError(f"unknown chunk strategy: {strategy}")

