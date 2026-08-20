"""Tests for evidence-bound Voice G1 recomputation and coverage."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from scripts.check_voice_g1 import EXPECTED_BASELINES, check_voice_g1
from scripts.run_voice_eval import sha256_file
from scripts.validate_voice_assets import (
    EXPECTED_BASELINE_ARTIFACTS,
    validate_run_manifest,
)
from src.voice.eval_contracts import (
    VoicePredictionManifest,
    VoiceReferenceManifest,
)
from src.voice.evaluator import evaluate_manifests


HASH = "c" * 64


def reference_payload(corpus_kind, artifact_id, transcript):
    sample_count = 6 if corpus_kind == "synthetic" else 1
    return {
        "schema_version": "1",
        "dataset_name": f"{corpus_kind}-fixture",
        "dataset_artifact_id": artifact_id,
        "dataset_split": "test",
        "corpus_kind": corpus_kind,
        "license_name": "fixture license",
        "license_url": "https://example.invalid/license",
        "dataset_source_sha256": HASH,
        "samples": [
            {
                "sample_id": f"{corpus_kind}-{index}",
                "duration_ms": 1000,
                "vad_intervals": [{"start_ms": 0, "end_ms": 1000}],
                "speaker_turns": [
                    {
                        "start_ms": 0,
                        "end_ms": 1000,
                        "speaker_label": "alice",
                        "text": transcript,
                    }
                ],
                "transcript": transcript,
                "candidates": [
                    {
                        "candidate_id": "ref-1",
                        "candidate_type": "commitment",
                        "content": transcript,
                        "speaker_label": "alice",
                        "source_start_ms": 0,
                        "source_end_ms": 1000,
                        "slots": {
                            "proper_name": "Alice",
                            "date": "周二",
                            "action_owner": "alice",
                        },
                    }
                ],
            }
            for index in range(sample_count)
        ],
    }


def prediction_payload(baseline_id, corpus_kind, transcript):
    artifacts = sorted(EXPECTED_BASELINE_ARTIFACTS[baseline_id])
    runtime_id = (
        "funasr-code"
        if baseline_id == "funasr-sensevoice-fsmn-campplus"
        else "sherpa-onnx-runtime"
    )
    model_id = (
        "sensevoice-weights"
        if baseline_id == "funasr-sensevoice-fsmn-campplus"
        else "sherpa-sensevoice-onnx-weights"
    )
    sample_count = 6 if corpus_kind == "synthetic" else 1
    return {
        "schema_version": "1",
        "baseline_id": baseline_id,
        "runtime_artifact_id": runtime_id,
        "runtime_name": runtime_id,
        "runtime_version": "1",
        "runtime_revision": "1",
        "runtime_sha256": HASH,
        "model_artifact_id": model_id,
        "model_name": model_id,
        "model_revision": "1",
        "model_sha256": HASH,
        "config_sha256": HASH,
        "hardware": {
            "os": "fixture-os",
            "cpu": "fixture-cpu",
            "python": "3.12",
            "device": "cpu",
            "threads": "1",
        },
        "model_size_mb": 10,
        "artifact_pins": [
            {"artifact_id": artifact_id, "revision": "1", "sha256": HASH}
            for artifact_id in artifacts
        ],
        "samples": [
            {
                "sample_id": f"{corpus_kind}-{index}",
                "duration_ms": 1000,
                "vad_intervals": [{"start_ms": 0, "end_ms": 1000}],
                "speaker_turns": [
                    {
                        "start_ms": 0,
                        "end_ms": 1000,
                        "speaker_label": "speaker_0",
                        "text": transcript,
                    }
                ],
                "transcript": transcript,
                "candidates": [
                    {
                        "candidate_id": "pred-1",
                        "candidate_type": "commitment",
                        "content": transcript,
                        "speaker_label": "speaker_0",
                        "source_start_ms": 0,
                        "source_end_ms": 1000,
                        "slots": {
                            "proper_name": "Alice",
                            "date": "周二",
                            "action_owner": "speaker_0",
                        },
                    }
                ],
                "candidate_decisions": {"pred-1": "approve"},
                "processing_seconds": 0.5,
                "first_partial_latency_ms": 100,
                "completion_latency_ms": 500,
                "peak_ram_mb": 128,
                "peak_vram_mb": None,
            }
            for index in range(sample_count)
        ],
    }


def build_evidence(root: Path):
    selected_ids = set().union(*EXPECTED_BASELINE_ARTIFACTS.values())
    inventory = {
        "artifacts": [
            {
                "artifact_id": artifact_id,
                "kind": "model_weights" if "weights" in artifact_id else "code",
                "license_name": "fixture license",
                "license_url": "https://example.invalid/license",
            }
            for artifact_id in sorted(selected_ids)
        ]
        + [
            {
                "artifact_id": f"{kind}-data",
                "kind": "dataset",
                "license_name": "fixture license",
                "license_url": "https://example.invalid/license",
            }
            for kind in ("public", "synthetic")
        ]
    }
    references = {}
    for kind, text in (("public", "周二开会"), ("synthetic", "周三开会")):
        path = root / f"reference-{kind}.json"
        row = VoiceReferenceManifest.model_validate(
            reference_payload(kind, f"{kind}-data", text)
        )
        path.write_text(row.model_dump_json(indent=2), encoding="utf-8")
        references[kind] = (path, row)

    run_manifest = {
        "run_name": "evidence-fixture",
        "metric_config": {
            "der_frame_ms": 10,
            "der_collar_ms": 0,
            "score_overlap": True,
        },
        "baselines": [
            {
                "baseline_id": baseline_id,
                "config_sha256": HASH,
                "artifacts": [
                    {"artifact_id": artifact_id, "revision": "1", "sha256": HASH}
                    for artifact_id in sorted(artifact_ids)
                ],
            }
            for baseline_id, artifact_ids in EXPECTED_BASELINE_ARTIFACTS.items()
        ],
        "evaluation_data": [
            {
                "artifact_id": f"{kind}-data",
                "corpus_kind": kind,
                "split_or_subset": "test",
                "local_path": f"D:/fixture/{kind}",
                "source_revision": "1",
                "source_sha256": HASH,
                "reference_manifest_sha256": sha256_file(references[kind][0]),
                "expected_sample_count": 6 if kind == "synthetic" else 1,
            }
            for kind in ("public", "synthetic")
        ],
    }
    inventory_path = root / "licenses.yaml"
    run_manifest_path = root / "artifacts.yaml"
    inventory_path.write_text(yaml.safe_dump(inventory), encoding="utf-8")
    run_manifest_path.write_text(yaml.safe_dump(run_manifest), encoding="utf-8")
    asset_summary = validate_run_manifest(inventory, run_manifest)

    results = []
    for baseline_id in sorted(EXPECTED_BASELINES):
        for kind, text in (("public", "周二开会"), ("synthetic", "周三开会")):
            prediction_path = root / f"prediction-{baseline_id}-{kind}.json"
            prediction = VoicePredictionManifest.model_validate(
                prediction_payload(baseline_id, kind, text)
            )
            prediction_path.write_text(
                prediction.model_dump_json(indent=2),
                encoding="utf-8",
            )
            result = evaluate_manifests(
                references[kind][1],
                prediction,
                frame_ms=10,
                collar_ms=0,
            )
            result["provenance"] = {
                "asset_gate": asset_summary,
                "license_inventory_sha256": sha256_file(inventory_path),
                "asset_manifest_sha256": sha256_file(run_manifest_path),
                "reference_manifest_path": references[kind][0].name,
                "reference_manifest_sha256": sha256_file(references[kind][0]),
                "prediction_manifest_path": prediction_path.name,
                "prediction_manifest_sha256": sha256_file(prediction_path),
            }
            results.append(result)
    return inventory, run_manifest, inventory_path, run_manifest_path, results


def test_g1_gate_recomputes_four_bound_results():
    with TemporaryDirectory(prefix=".voice-g1-", dir=".") as temp_dir:
        root = Path(temp_dir).resolve()
        inventory, run_manifest, inventory_path, run_path, results = build_evidence(root)

        summary = check_voice_g1(
            results,
            root=root,
            inventory=inventory,
            run_manifest=run_manifest,
            inventory_sha256=sha256_file(inventory_path),
            run_manifest_sha256=sha256_file(run_path),
        )

        assert summary["status"] == "G1_PASSED"
        assert summary["result_count"] == 4


def test_g1_gate_rejects_handwritten_or_incomplete_results():
    with TemporaryDirectory(prefix=".voice-g1-", dir=".") as temp_dir:
        root = Path(temp_dir).resolve()
        inventory, run_manifest, inventory_path, run_path, results = build_evidence(root)
        forged = [
            {
                "schema_version": "1",
                "baseline": {"baseline_id": baseline},
                "dataset": {"corpus_kind": corpus},
                "summary": {},
            }
            for baseline in sorted(EXPECTED_BASELINES)
            for corpus in ("public", "synthetic")
        ]

        with pytest.raises(ValueError, match="license inventory"):
            check_voice_g1(
                forged,
                root=root,
                inventory=inventory,
                run_manifest=run_manifest,
                inventory_sha256=sha256_file(inventory_path),
                run_manifest_sha256=sha256_file(run_path),
            )
        with pytest.raises(ValueError, match="coverage is incomplete"):
            check_voice_g1(
                results[:-1],
                root=root,
                inventory=inventory,
                run_manifest=run_manifest,
                inventory_sha256=sha256_file(inventory_path),
                run_manifest_sha256=sha256_file(run_path),
            )
