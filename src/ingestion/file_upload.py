"""File upload ingestion helpers."""

from __future__ import annotations

import csv
from datetime import datetime
from io import StringIO
from pathlib import Path


def decode_bytes(data: bytes) -> str:
    return data.decode("utf-8-sig").strip()


def parse_txt_upload(data: bytes) -> str:
    text = decode_bytes(data)
    if not text:
        raise ValueError("uploaded txt file is empty")
    return text


def parse_csv_upload(data: bytes) -> str:
    text = decode_bytes(data)
    if not text:
        raise ValueError("uploaded csv file is empty")

    rows = []
    reader = csv.reader(StringIO(text))
    for row in reader:
        cells = [cell.strip() for cell in row if cell.strip()]
        if cells:
            rows.append(" | ".join(cells))

    if not rows:
        raise ValueError("uploaded csv file has no usable rows")
    return "\n".join(rows)


def build_file_upload_conversation(
    filename: str,
    data: bytes,
    language: str,
    conversation_date: datetime | None = None,
) -> dict[str, object]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        raw_content = parse_txt_upload(data)
    elif suffix == ".csv":
        raw_content = parse_csv_upload(data)
    else:
        raise ValueError("only .txt and .csv uploads are supported for now")

    return {
        "source_type": "file_upload",
        "raw_content": raw_content,
        "language": language.strip().lower() or "en",
        "conversation_date": conversation_date,
    }

