"""Basic text cleaning helpers for conversation data."""

from __future__ import annotations

import re
import unicodedata


# reused from rag-eval-pipeline/src/cleaning.py, adapted for conversation data
SPACE_RE = re.compile(r"\s+")
CJK_RE = re.compile(r"[\u3400-\u9fff]")


def clean_text(text: str | None) -> str:
    if text is None:
        return ""

    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("\u00a0", " ")
    value = value.replace("\u200b", " ")
    value = SPACE_RE.sub(" ", value)
    return value.strip()


def clean_lines(lines: list[str] | None) -> list[str]:
    if not lines:
        return []

    cleaned = []
    for line in lines:
        text = clean_text(line)
        if text:
            cleaned.append(text)
    return cleaned


def has_cjk(text: str | None) -> bool:
    if text is None:
        return False
    return bool(CJK_RE.search(str(text)))

