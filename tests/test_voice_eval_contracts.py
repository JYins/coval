"""Tests for frozen Voice reference and prediction contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.voice.eval_contracts import (
    VoicePredictionManifest,
    VoiceReferenceManifest,
)


HASH = "a" * 64


def reference_payload():
    return {
        "schema_version": "1",
        "dataset_name": "fictional-meeting-fixture",
        "dataset_artifact_id": "fictional-crm-meeting-audio",
        "dataset_split": "test",
        "corpus_kind": "synthetic",
        "license_name": "test-only fixture",
        "license_url": "https://example.invalid/license",
        "dataset_source_sha256": HASH,
        "samples": [
            {
                "sample_id": "meeting-001",
                "duration_ms": 1000,
                "vad_intervals": [{"start_ms": 0, "end_ms": 900}],
                "speaker_turns": [
                    {
                        "start_ms": 0,
                        "end_ms": 900,
                        "speaker_label": "speaker_0",
                        "text": "周二开会",
                    }
                ],
                "transcript": "周二开会",
                "candidates": [],
            }
        ],
    }


def prediction_payload():
    return {
        "schema_version": "1",
        "baseline_id": "fixture-runtime",
        "runtime_artifact_id": "fixture-runtime-code",
        "runtime_name": "fixture",
        "runtime_version": "1",
        "runtime_revision": "1",
        "runtime_sha256": HASH,
        "model_artifact_id": "fixture-model-weights",
        "model_name": "fixture",
        "model_revision": "1",
        "model_sha256": HASH,
        "config_sha256": HASH,
        "hardware": {
            "cpu": "test-cpu",
            "os": "test-os",
            "python": "3.12",
            "device": "cpu",
            "threads": "1",
        },
        "model_size_mb": 1.0,
        "artifact_pins": [
            {
                "artifact_id": "fixture-runtime-code",
                "revision": "1",
                "sha256": HASH,
            },
            {
                "artifact_id": "fixture-model-weights",
                "revision": "1",
                "sha256": HASH,
            },
        ],
        "samples": [
            {
                "sample_id": "meeting-001",
                "duration_ms": 1000,
                "vad_intervals": [{"start_ms": 0, "end_ms": 900}],
                "speaker_turns": [
                    {
                        "start_ms": 0,
                        "end_ms": 900,
                        "speaker_label": "speaker_0",
                        "text": "周二开会",
                    }
                ],
                "transcript": "周二开会",
                "candidates": [],
                "candidate_decisions": {},
                "processing_seconds": 0.5,
                "first_partial_latency_ms": None,
                "completion_latency_ms": 500,
                "peak_ram_mb": 32,
                "peak_vram_mb": None,
            }
        ],
    }


def test_voice_eval_contracts_accept_pinned_fixture():
    reference = VoiceReferenceManifest.model_validate(reference_payload())
    prediction = VoicePredictionManifest.model_validate(prediction_payload())

    assert reference.samples[0].sample_id == prediction.samples[0].sample_id
    assert prediction.samples[0].first_partial_latency_ms is None


def test_voice_eval_contracts_reject_placeholder_hash_and_bad_interval():
    prediction = prediction_payload()
    prediction["model_sha256"] = "TODO"
    with pytest.raises(ValidationError):
        VoicePredictionManifest.model_validate(prediction)

    reference = reference_payload()
    reference["samples"][0]["vad_intervals"][0] = {
        "start_ms": 900,
        "end_ms": 100,
    }
    with pytest.raises(ValidationError):
        VoiceReferenceManifest.model_validate(reference)
