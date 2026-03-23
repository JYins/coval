"""Tests for chunking helpers."""

from __future__ import annotations

from src.rag.chunking import chunk_conversation
from src.rag.cleaning import clean_text


def test_clean_text_normalizes_spaces():
    text = "  Alice\u00a0likes   jazz  "
    assert clean_text(text) == "Alice likes jazz"


def test_sentence_chunking_adds_person_name_prefix():
    raw_content = "We talked about music. She likes jazz. She hates crowded bars."

    chunks = chunk_conversation(
        raw_content=raw_content,
        person_name="Alice",
        strategy="sentence",
        max_sentences=2,
    )

    assert len(chunks) == 2
    assert chunks[0]["person_name_prefix"] == "Alice"
    assert chunks[0]["chunk_text"].startswith("Alice:")
    assert chunks[0]["sentence_ids"] == [0, 1]


def test_paragraph_chunking_keeps_order():
    raw_content = "First topic here.\n\nSecond topic here."

    chunks = chunk_conversation(
        raw_content=raw_content,
        person_name="Bob",
        strategy="paragraph",
    )

    assert len(chunks) == 2
    assert chunks[0]["paragraph_index"] == 0
    assert chunks[1]["paragraph_index"] == 1
    assert chunks[1]["chunk_text"].startswith("Bob:")

