"""Integration tests for frozen Voice baseline aggregation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from src.voice.eval_contracts import (
    VoicePredictionManifest,
    VoiceReferenceManifest,
)
from src.voice.evaluator import build_speaker_map, evaluate_manifests
from scripts.run_voice_eval import sha256_file
from scripts.validate_voice_assets import EXPECTED_BASELINE_ARTIFACTS


HASH = "b" * 64


def reference_manifest():
    return VoiceReferenceManifest.model_validate(
        {
            "schema_version": "1",
            "dataset_name": "fictional-meeting",
            "dataset_artifact_id": "fictional-crm-meeting-audio",
            "dataset_split": "test",
            "corpus_kind": "synthetic",
            "license_name": "fixture",
            "license_url": "https://example.invalid/license",
            "dataset_source_sha256": HASH,
            "samples": [
                {
                    "sample_id": "one",
                    "duration_ms": 1000,
                    "vad_intervals": [{"start_ms": 0, "end_ms": 1000}],
                    "speaker_turns": [
                        {
                            "start_ms": 0,
                            "end_ms": 500,
                            "speaker_label": "alice",
                            "text": "周二开会",
                        },
                        {
                            "start_ms": 500,
                            "end_ms": 1000,
                            "speaker_label": "bob",
                            "text": "好的",
                        },
                    ],
                    "transcript": "周二开会好的",
                    "candidates": [
                        {
                            "candidate_id": "candidate-1",
                            "candidate_type": "commitment",
                            "content": "周二开会",
                            "speaker_label": "alice",
                            "source_start_ms": 0,
                            "source_end_ms": 500,
                            "slots": {
                                "proper_name": "Alice",
                                "date": "周二",
                                "action_owner": "alice",
                            },
                        }
                    ],
                }
            ],
        }
    )


def prediction_manifest():
    return VoicePredictionManifest.model_validate(
        {
            "schema_version": "1",
            "baseline_id": "fixture",
            "runtime_artifact_id": "fixture-runtime-code",
            "runtime_name": "fixture-runtime",
            "runtime_version": "1",
            "runtime_revision": "1",
            "runtime_sha256": HASH,
            "model_artifact_id": "fixture-model-weights",
            "model_name": "fixture-model",
            "model_revision": "1",
            "model_sha256": HASH,
            "config_sha256": HASH,
            "hardware": {
                "cpu": "fixture-cpu",
                "os": "fixture-os",
                "python": "3.12",
                "device": "cpu",
                "threads": "1",
            },
            "model_size_mb": 10,
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
                    "sample_id": "one",
                    "duration_ms": 1000,
                    "vad_intervals": [{"start_ms": 0, "end_ms": 1000}],
                    "speaker_turns": [
                        {
                            "start_ms": 0,
                            "end_ms": 500,
                            "speaker_label": "speaker_1",
                            "text": "周二开会",
                        },
                        {
                            "start_ms": 500,
                            "end_ms": 1000,
                            "speaker_label": "speaker_0",
                            "text": "好的",
                        },
                    ],
                    "transcript": "周二开会好的",
                    "candidates": [
                        {
                            "candidate_id": "candidate-1",
                            "candidate_type": "commitment",
                            "content": "周二开会",
                            "speaker_label": "speaker_1",
                            "source_start_ms": 0,
                            "source_end_ms": 500,
                            "slots": {
                                "proper_name": "Alice",
                                "date": "周二",
                                "action_owner": "speaker_1",
                            },
                        }
                    ],
                    "candidate_decisions": {"candidate-1": "approve"},
                    "processing_seconds": 0.5,
                    "first_partial_latency_ms": 100,
                    "completion_latency_ms": 500,
                    "peak_ram_mb": 128,
                    "peak_vram_mb": None,
                }
            ],
        }
    )


def test_evaluator_aligns_anonymous_speakers_and_aggregates_metrics():
    result = evaluate_manifests(
        reference_manifest(),
        prediction_manifest(),
        frame_ms=10,
        collar_ms=0,
    )

    assert result["per_sample"][0]["speaker_map"] == {
        "speaker_0": "bob",
        "speaker_1": "alice",
    }
    assert result["summary"]["cer"] == 0.0
    assert result["summary"]["der"] == 0.0
    assert result["summary"]["jer"] == 0.0
    assert result["summary"]["speaker_attributed_cer"] == 0.0
    assert result["summary"]["structured_f1"] == 1.0
    assert result["summary"]["proper_name_recall"] == 1.0
    assert result["summary"]["number_date_unit_exact_match"] == 1.0
    assert result["summary"]["speaker_attribution_accuracy"] == 1.0
    assert result["summary"]["action_owner_accuracy"] == 1.0
    assert result["summary"]["false_approved_fact_rate"] == 0.0
    assert result["summary"]["rtf"] == 0.5
    assert result["summary"]["peak_ram_mb"] == 128


def test_evaluator_fails_when_sample_sets_differ():
    prediction = prediction_manifest()
    prediction.samples[0].sample_id = "other"

    with pytest.raises(ValueError, match="sample IDs differ"):
        evaluate_manifests(reference_manifest(), prediction)


def test_speaker_map_is_one_to_one():
    reference = reference_manifest().samples[0].speaker_turns
    prediction = prediction_manifest().samples[0].speaker_turns

    mapping = build_speaker_map(reference, prediction)

    assert len(mapping.values()) == len(set(mapping.values()))


def test_voice_eval_cli_writes_a_reproducible_result():
    with TemporaryDirectory(prefix=".voice-eval-", dir=".") as temp_dir:
        temp_path = Path(temp_dir).resolve()
        reference_path = temp_path / "reference.json"
        prediction_path = temp_path / "prediction.json"
        result_path = temp_path / "result.json"
        config_path = temp_path / "eval.yaml"
        inventory_path = temp_path / "licenses.yaml"
        asset_manifest_path = temp_path / "artifacts.local.yaml"
        eval_reference = reference_manifest().model_copy(
            update={
                "dataset_name": "public-fixture",
                "dataset_artifact_id": "public-fixture",
                "corpus_kind": "public",
            }
        )
        reference_path.write_text(
            eval_reference.model_dump_json(indent=2),
            encoding="utf-8",
        )
        baseline_id = "funasr-sensevoice-fsmn-campplus"
        prediction_data = prediction_manifest().model_dump()
        prediction_data["baseline_id"] = baseline_id
        prediction_data["runtime_artifact_id"] = "funasr-code"
        prediction_data["model_artifact_id"] = "sensevoice-weights"
        prediction_data["artifact_pins"] = [
            {"artifact_id": artifact_id, "revision": "1", "sha256": HASH}
            for artifact_id in sorted(EXPECTED_BASELINE_ARTIFACTS[baseline_id])
        ]
        prediction = VoicePredictionManifest.model_validate(prediction_data)
        prediction_path.write_text(
            prediction.model_dump_json(indent=2),
            encoding="utf-8",
        )
        selected_ids = set().union(*EXPECTED_BASELINE_ARTIFACTS.values())
        inventory_path.write_text(
            yaml.safe_dump(
                {
                    "artifacts": [
                        {
                            "artifact_id": artifact_id,
                            "kind": (
                                "model_weights"
                                if "weights" in artifact_id
                                else "code"
                            ),
                            "license_name": "fixture license",
                            "license_url": "https://example.invalid/license",
                        }
                        for artifact_id in sorted(selected_ids)
                    ]
                    + [
                        {
                            "artifact_id": "fictional-crm-meeting-audio",
                            "kind": "dataset",
                            "license_name": "fixture license",
                            "license_url": "https://example.invalid/synthetic",
                        },
                        {
                            "artifact_id": "public-fixture",
                            "kind": "dataset",
                            "license_name": "fixture license",
                            "license_url": "https://example.invalid/public",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        asset_manifest_path.write_text(
            yaml.safe_dump(
                {
                    "run_name": "cli-fixture",
                    "metric_config": {
                        "der_frame_ms": 10,
                        "der_collar_ms": 0,
                        "score_overlap": True,
                    },
                    "baselines": [
                        {
                            "baseline_id": name,
                            "config_sha256": HASH,
                            "artifacts": [
                                {
                                    "artifact_id": artifact_id,
                                    "revision": "1",
                                    "sha256": HASH,
                                }
                                for artifact_id in sorted(artifact_ids)
                            ],
                        }
                        for name, artifact_ids in EXPECTED_BASELINE_ARTIFACTS.items()
                    ],
                    "evaluation_data": [
                        {
                            "artifact_id": "public-fixture",
                            "corpus_kind": "public",
                            "split_or_subset": "test",
                            "local_path": "D:/fixture/public",
                            "source_revision": "1",
                            "source_sha256": HASH,
                            "reference_manifest_sha256": sha256_file(reference_path),
                            "expected_sample_count": 1,
                        },
                        {
                            "artifact_id": "fictional-crm-meeting-audio",
                            "corpus_kind": "synthetic",
                            "split_or_subset": "test",
                            "local_path": "D:/fixture/synthetic",
                            "source_revision": "1",
                            "source_sha256": HASH,
                            "reference_manifest_sha256": HASH,
                            "expected_sample_count": 6,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        config_path.write_text(
            yaml.safe_dump(
                {
                    "reference_manifest": str(reference_path),
                    "prediction_manifest": str(prediction_path),
                    "output": str(result_path),
                    "license_inventory": str(inventory_path),
                    "asset_manifest": str(asset_manifest_path),
                    "metrics": {
                        "der_frame_ms": 10,
                        "der_collar_ms": 0,
                        "score_overlap": True,
                    },
                }
            ),
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_voice_eval.py",
                "--config",
                str(config_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))

        assert "samples: 1" in completed.stdout
        assert result["summary"]["cer"] == 0.0
        assert result["baseline"]["hardware"]["device"] == "cpu"
