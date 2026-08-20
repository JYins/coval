"""Compatibility entrypoint for the old conversation upload route."""

from __future__ import annotations


def transcribe_voice(*args, **kwargs):
    raise NotImplementedError("use POST /api/voice/jobs for reviewed voice ingestion")

