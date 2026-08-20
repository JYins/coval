"""Tests for the fail-loud Voice asset/license gate."""

from __future__ import annotations

import pytest

from scripts.validate_voice_assets import (
    EXPECTED_BASELINE_ARTIFACTS,
    validate_run_manifest,
)


HASH = "a" * 64


def inventory():
    selected_ids = set().union(*EXPECTED_BASELINE_ARTIFACTS.values())
    rows = [
        {
            "artifact_id": artifact_id,
            "kind": "model_weights" if "weights" in artifact_id else "code",
            "license_name": "test license",
            "license_url": "https://example.invalid/license",
        }
        for artifact_id in sorted(selected_ids)
    ]
    rows.extend(
        [
            {
                "artifact_id": "public-data",
                "kind": "dataset",
                "source_url": "https://example.invalid/public-source",
                "license_name": "test license",
                "license_url": "https://example.invalid/public",
                "redistribution_policy": "test only",
            },
            {
                "artifact_id": "synthetic-data",
                "kind": "dataset",
                "source_url": "https://example.invalid/synthetic-source",
                "license_name": "test license",
                "license_url": "https://example.invalid/synthetic",
                "redistribution_policy": "test only",
            },
        ]
    )
    return {"artifacts": rows}


def run_manifest():
    return {
        "run_name": "fixture-run",
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
                "artifact_id": "public-data",
                "corpus_kind": "public",
                "split_or_subset": "test",
                "local_path": "D:/datasets/public",
                "source_revision": "1",
                "source_sha256": HASH,
                "reference_manifest_sha256": HASH,
                "expected_sample_count": 1,
            },
            {
                "artifact_id": "synthetic-data",
                "corpus_kind": "synthetic",
                "split_or_subset": "test",
                "local_path": "D:/datasets/synthetic",
                "source_revision": "1",
                "source_sha256": HASH,
                "reference_manifest_sha256": HASH,
                "expected_sample_count": 6,
            },
        ],
    }


def test_voice_asset_gate_accepts_two_complete_pinned_baselines():
    summary = validate_run_manifest(inventory(), run_manifest())

    assert summary == {"baselines": 2, "artifacts": 13, "datasets": 2}


def test_voice_asset_gate_rejects_todo_and_incomplete_pipeline():
    manifest = run_manifest()
    manifest["baselines"][0]["artifacts"][0]["revision"] = "REQUIRED/TODO"
    with pytest.raises(ValueError, match="must be pinned"):
        validate_run_manifest(inventory(), manifest)

    manifest = run_manifest()
    manifest["baselines"][1]["artifacts"].pop()
    with pytest.raises(ValueError, match="audited pipeline"):
        validate_run_manifest(inventory(), manifest)


def test_voice_asset_gate_rejects_unlicensed_synthetic_data():
    rows = inventory()
    synthetic = next(
        row for row in rows["artifacts"] if row["artifact_id"] == "synthetic-data"
    )
    synthetic["source_url"] = "REQUIRED/TODO"

    with pytest.raises(ValueError, match="synthetic-data.source_url"):
        validate_run_manifest(rows, run_manifest())
