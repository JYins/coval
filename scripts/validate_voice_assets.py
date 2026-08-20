"""Fail-loud license and checksum gate for opt-in Voice experiments."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDER = "REQUIRED/TODO"
EXPECTED_BASELINE_ARTIFACTS = {
    "funasr-sensevoice-fsmn-campplus": {
        "funasr-code",
        "sensevoice-weights",
        "fsmn-vad-weights",
        "three-d-speaker-code",
        "campplus-weights",
    },
    "sherpa-sensevoice-separate-diarization": {
        "sherpa-onnx-runtime",
        "sherpa-sensevoice-onnx-weights",
        "pyannote-segmentation-3-weights",
        "eres2net-weights",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate pinned Voice assets")
    parser.add_argument(
        "--inventory",
        default="configs/voice/licenses.yaml",
    )
    parser.add_argument("--run-manifest", required=True)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def require_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text or text == PLACEHOLDER:
        raise ValueError(f"{field} must be pinned before a Voice experiment")
    return text


def require_sha256(value: Any, field: str) -> str:
    text = require_text(value, field).lower()
    if not SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return text


def validate_run_manifest(
    inventory: dict[str, Any],
    run_manifest: dict[str, Any],
) -> dict[str, int]:
    inventory_rows = list(inventory.get("artifacts", []))
    known = {str(row["artifact_id"]): row for row in inventory_rows}
    if not known:
        raise ValueError("voice license inventory is empty")

    metric_config = dict(run_manifest.get("metric_config", {}))
    if int(metric_config.get("der_frame_ms", 0)) <= 0:
        raise ValueError("Voice DER frame size must be positive")
    if int(metric_config.get("der_collar_ms", -1)) < 0:
        raise ValueError("Voice DER collar cannot be negative")
    if not isinstance(metric_config.get("score_overlap"), bool):
        raise ValueError("Voice overlap scoring choice must be pinned")

    require_text(run_manifest.get("run_name"), "run_name")
    baseline_count = 0
    artifact_count = 0
    for baseline in run_manifest.get("baselines", []):
        baseline_id = require_text(baseline.get("baseline_id"), "baseline_id")
        require_sha256(
            baseline.get("config_sha256"),
            f"{baseline_id}.config_sha256",
        )
        baseline_count += 1
        selected_artifacts = list(baseline.get("artifacts", []))
        if not selected_artifacts:
            raise ValueError(f"{baseline_id} has no selected artifacts")
        for selected in selected_artifacts:
            artifact_id = require_text(
                selected.get("artifact_id"),
                f"{baseline_id}.artifact_id",
            )
            if artifact_id not in known:
                raise ValueError(f"unknown Voice artifact: {artifact_id}")
            require_text(selected.get("revision"), f"{artifact_id}.revision")
            require_sha256(selected.get("sha256"), f"{artifact_id}.sha256")
            license_row = known[artifact_id]
            require_text(license_row.get("license_name"), f"{artifact_id}.license_name")
            require_text(license_row.get("license_url"), f"{artifact_id}.license_url")
            artifact_count += 1

        selected_ids = {
            str(selected["artifact_id"]) for selected in selected_artifacts
        }
        expected_ids = EXPECTED_BASELINE_ARTIFACTS.get(baseline_id)
        if expected_ids is None or selected_ids != expected_ids:
            raise ValueError(
                f"{baseline_id} artifact set does not match the audited pipeline"
            )

    if baseline_count != 2 or {
        str(row.get("baseline_id")) for row in run_manifest.get("baselines", [])
    } != EXPECTED_BASELINE_ARTIFACTS.keys():
        raise ValueError("Voice G1 requires exactly two baseline pipelines")

    datasets = list(run_manifest.get("evaluation_data", []))
    if not datasets:
        raise ValueError("Voice G1 requires pinned evaluation data")
    corpus_kinds = set()
    for dataset in datasets:
        artifact_id = require_text(dataset.get("artifact_id"), "dataset.artifact_id")
        row = known.get(artifact_id)
        if row is None or row.get("kind") != "dataset":
            raise ValueError(f"unknown Voice dataset artifact: {artifact_id}")
        require_text(dataset.get("split_or_subset"), f"{artifact_id}.split_or_subset")
        require_text(dataset.get("local_path"), f"{artifact_id}.local_path")
        require_text(dataset.get("source_revision"), f"{artifact_id}.source_revision")
        require_sha256(dataset.get("source_sha256"), f"{artifact_id}.source_sha256")
        require_sha256(
            dataset.get("reference_manifest_sha256"),
            f"{artifact_id}.reference_manifest_sha256",
        )
        corpus_kind = require_text(
            dataset.get("corpus_kind"),
            f"{artifact_id}.corpus_kind",
        )
        if corpus_kind not in {"public", "synthetic"}:
            raise ValueError(f"invalid Voice corpus kind: {corpus_kind}")
        corpus_kinds.add(corpus_kind)
        expected_count = int(dataset.get("expected_sample_count", 0))
        if expected_count <= 0:
            raise ValueError(f"{artifact_id}.expected_sample_count must be positive")
        if corpus_kind == "synthetic" and expected_count < 6:
            raise ValueError("Voice synthetic evaluation requires at least 6 meetings")

    if corpus_kinds != {"public", "synthetic"}:
        raise ValueError("Voice G1 requires public and synthetic evaluation data")

    return {
        "baselines": baseline_count,
        "artifacts": artifact_count,
        "datasets": len(datasets),
    }


def main() -> None:
    args = parse_args()
    inventory = load_yaml(ROOT / args.inventory)
    run_manifest = load_yaml(ROOT / args.run_manifest)
    summary = validate_run_manifest(inventory, run_manifest)
    print("Voice asset gate passed")
    for name, count in summary.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
