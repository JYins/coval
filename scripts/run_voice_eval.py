"""Score one pinned local Voice baseline without loading speech models in CI."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Coval Voice baseline evaluation")
    parser.add_argument("--config", default="configs/voice/eval.yaml")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved)


def main() -> None:
    from src.voice.eval_contracts import (
        load_prediction_manifest,
        load_reference_manifest,
    )
    from src.voice.evaluator import evaluate_manifests
    from scripts.validate_voice_assets import load_yaml, validate_run_manifest

    args = parse_args()
    config = load_config(ROOT / args.config)
    metric_config = dict(config.get("metrics", {}))
    reference_path = (ROOT / str(config["reference_manifest"])).resolve()
    prediction_path = (ROOT / str(config["prediction_manifest"])).resolve()
    inventory_path = (ROOT / str(config["license_inventory"])).resolve()
    run_manifest_path = (ROOT / str(config["asset_manifest"])).resolve()
    inventory = load_yaml(inventory_path)
    run_manifest = load_yaml(run_manifest_path)
    asset_summary = validate_run_manifest(inventory, run_manifest)
    reference = load_reference_manifest(reference_path)
    prediction = load_prediction_manifest(prediction_path)
    if metric_config != dict(run_manifest["metric_config"]):
        raise ValueError("evaluation metrics do not match the pinned run manifest")

    baseline = next(
        (
            row
            for row in run_manifest["baselines"]
            if row["baseline_id"] == prediction.baseline_id
        ),
        None,
    )
    if baseline is None:
        raise ValueError("prediction baseline is missing from the asset manifest")
    selected_pins = {
        (row["artifact_id"], row["revision"], row["sha256"])
        for row in baseline["artifacts"]
    }
    prediction_pins = {
        (row.artifact_id, row.revision, row.sha256)
        for row in prediction.artifact_pins
    }
    if selected_pins != prediction_pins:
        raise ValueError("prediction artifacts do not match the asset manifest")
    if baseline["config_sha256"] != prediction.config_sha256:
        raise ValueError("prediction config hash does not match the asset manifest")

    dataset = next(
        (
            row
            for row in run_manifest["evaluation_data"]
            if row["artifact_id"] == reference.dataset_artifact_id
        ),
        None,
    )
    if dataset is None:
        raise ValueError("reference dataset is missing from the asset manifest")
    reference_sha256 = sha256_file(reference_path)
    if dataset["corpus_kind"] != reference.corpus_kind:
        raise ValueError("reference corpus kind does not match the asset manifest")
    if dataset["split_or_subset"] != reference.dataset_split:
        raise ValueError("reference split does not match the asset manifest")
    if dataset["source_sha256"] != reference.dataset_source_sha256:
        raise ValueError("reference source hash does not match the asset manifest")
    if dataset["reference_manifest_sha256"] != reference_sha256:
        raise ValueError("reference file hash does not match the asset manifest")
    if int(dataset["expected_sample_count"]) != len(reference.samples):
        raise ValueError("reference sample count does not match the asset manifest")

    result = evaluate_manifests(
        reference,
        prediction,
        frame_ms=int(metric_config.get("der_frame_ms", 10)),
        collar_ms=int(metric_config.get("der_collar_ms", 250)),
        score_overlap=bool(metric_config.get("score_overlap", True)),
    )
    result["provenance"] = {
        "asset_gate": asset_summary,
        "license_inventory_path": evidence_path(inventory_path),
        "license_inventory_sha256": sha256_file(inventory_path),
        "asset_manifest_path": evidence_path(run_manifest_path),
        "asset_manifest_sha256": sha256_file(run_manifest_path),
        "reference_manifest_path": evidence_path(reference_path),
        "reference_manifest_sha256": reference_sha256,
        "prediction_manifest_path": evidence_path(prediction_path),
        "prediction_manifest_sha256": sha256_file(prediction_path),
    }

    output = ROOT / str(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Voice evaluation written: {output}")
    print(f"samples: {result['summary']['sample_count']}")
    print(f"CER: {result['summary']['cer']:.4f}")
    print(f"DER: {result['summary']['der']:.4f}")


if __name__ == "__main__":
    main()
