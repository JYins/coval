"""Recompute and verify the complete Voice G1 evidence set."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_voice_eval import sha256_file  # noqa: E402
from scripts.validate_voice_assets import (  # noqa: E402
    load_yaml,
    validate_run_manifest,
)
from src.voice.eval_contracts import (  # noqa: E402
    load_prediction_manifest,
    load_reference_manifest,
)
from src.voice.evaluator import evaluate_manifests  # noqa: E402


EXPECTED_BASELINES = {
    "funasr-sensevoice-fsmn-campplus",
    "sherpa-sensevoice-separate-diarization",
}
EXPECTED_CORPORA = {"public", "synthetic"}
REQUIRED_METRICS = {
    "cer",
    "vad_f1",
    "der",
    "jer",
    "speaker_attributed_cer",
    "structured_f1",
    "proper_name_recall",
    "number_date_unit_exact_match",
    "speaker_attribution_accuracy",
    "action_owner_accuracy",
    "false_approved_fact_rate",
    "review_rate",
    "rtf",
    "peak_ram_mb",
}
REQUIRED_COUNTS = {
    "audio_duration_ms",
    "cer_reference_chars",
    "vad_reference_speech_ms",
    "der_scored_ms",
    "jer_speaker_count",
    "structured_reference_count",
    "approved_fact_count",
    "proper_name_reference_count",
    "number_date_unit_sample_count",
    "speaker_attribution_reference_count",
    "action_owner_reference_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Coval Voice G1 evidence")
    parser.add_argument("--inventory", default="configs/voice/licenses.yaml")
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("results", nargs="+")
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def resolve_evidence_path(root: Path, value: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Voice evidence manifests must live inside the repository") from exc
    if not resolved.is_file():
        raise ValueError(f"Voice evidence file is missing: {value}")
    return resolved


def pin_set(rows: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    return {
        (str(row["artifact_id"]), str(row["revision"]), str(row["sha256"]))
        for row in rows
    }


def check_voice_g1(
    results: list[dict[str, Any]],
    *,
    root: Path,
    inventory: dict[str, Any],
    run_manifest: dict[str, Any],
    inventory_sha256: str,
    run_manifest_sha256: str,
) -> dict[str, Any]:
    asset_summary = validate_run_manifest(inventory, run_manifest)
    baseline_rows = {
        str(row["baseline_id"]): row for row in run_manifest["baselines"]
    }
    dataset_rows = {
        str(row["artifact_id"]): row for row in run_manifest["evaluation_data"]
    }
    coverage: dict[str, set[str]] = {name: set() for name in EXPECTED_BASELINES}
    reference_by_corpus: dict[str, tuple[str, str, str]] = {}
    prediction_hashes = set()
    seen_pairs = set()

    for result in results:
        if result.get("schema_version") != "1":
            raise ValueError("Voice result schema_version must be 1")
        provenance = dict(result.get("provenance", {}))
        if provenance.get("license_inventory_sha256") != inventory_sha256:
            raise ValueError("Voice result is not bound to this license inventory")
        if provenance.get("asset_manifest_sha256") != run_manifest_sha256:
            raise ValueError("Voice result is not bound to this asset manifest")
        if provenance.get("asset_gate") != asset_summary:
            raise ValueError("Voice result asset-gate summary does not match")

        reference_path = resolve_evidence_path(
            root,
            str(provenance.get("reference_manifest_path", "")),
        )
        prediction_path = resolve_evidence_path(
            root,
            str(provenance.get("prediction_manifest_path", "")),
        )
        reference_sha256 = sha256_file(reference_path)
        prediction_sha256 = sha256_file(prediction_path)
        if provenance.get("reference_manifest_sha256") != reference_sha256:
            raise ValueError("Voice reference file hash changed after evaluation")
        if provenance.get("prediction_manifest_sha256") != prediction_sha256:
            raise ValueError("Voice prediction file hash changed after evaluation")
        if prediction_sha256 in prediction_hashes:
            raise ValueError("Voice prediction evidence was reused for multiple results")
        prediction_hashes.add(prediction_sha256)

        reference = load_reference_manifest(reference_path)
        prediction = load_prediction_manifest(prediction_path)
        metric_config = dict(result.get("metric_config", {}))
        if metric_config != dict(run_manifest["metric_config"]):
            raise ValueError("Voice result metric config is not pinned")
        recomputed = evaluate_manifests(
            reference,
            prediction,
            frame_ms=int(metric_config.get("der_frame_ms", 10)),
            collar_ms=int(metric_config.get("der_collar_ms", 250)),
            score_overlap=bool(metric_config.get("score_overlap", True)),
        )
        measured = {key: value for key, value in result.items() if key != "provenance"}
        if measured != recomputed:
            raise ValueError("Voice result metrics do not match recomputed evidence")

        baseline_id = prediction.baseline_id
        corpus_kind = reference.corpus_kind
        pair = (baseline_id, corpus_kind)
        if baseline_id not in EXPECTED_BASELINES:
            raise ValueError(f"unexpected Voice baseline: {baseline_id}")
        if corpus_kind not in EXPECTED_CORPORA:
            raise ValueError(f"unexpected Voice corpus kind: {corpus_kind}")
        if pair in seen_pairs:
            raise ValueError(f"duplicate Voice result pair: {pair}")
        seen_pairs.add(pair)

        selected_baseline = baseline_rows[baseline_id]
        if pin_set(selected_baseline["artifacts"]) != {
            (row.artifact_id, row.revision, row.sha256)
            for row in prediction.artifact_pins
        }:
            raise ValueError("Voice result artifact pins do not match the run manifest")
        if selected_baseline["config_sha256"] != prediction.config_sha256:
            raise ValueError("Voice result config hash does not match the run manifest")
        selected_dataset = dataset_rows.get(reference.dataset_artifact_id)
        if selected_dataset is None:
            raise ValueError("Voice result dataset is missing from the run manifest")
        if (
            selected_dataset["corpus_kind"] != corpus_kind
            or selected_dataset["split_or_subset"] != reference.dataset_split
            or selected_dataset["source_sha256"]
            != reference.dataset_source_sha256
            or selected_dataset["reference_manifest_sha256"] != reference_sha256
            or int(selected_dataset["expected_sample_count"])
            != len(reference.samples)
        ):
            raise ValueError("Voice result dataset provenance does not match")

        reference_identity = (
            reference.dataset_artifact_id,
            reference.dataset_split,
            reference_sha256,
        )
        prior_reference = reference_by_corpus.setdefault(
            corpus_kind,
            reference_identity,
        )
        if prior_reference != reference_identity:
            raise ValueError("baselines did not use the same frozen corpus evidence")

        summary = dict(result.get("summary", {}))
        missing = REQUIRED_METRICS - summary.keys()
        if missing:
            raise ValueError(f"Voice result is missing metrics: {sorted(missing)}")
        for metric in REQUIRED_METRICS:
            value = summary[metric]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Voice metric must be finite: {metric}")
        for count_metric in REQUIRED_COUNTS:
            if int(summary.get(count_metric, 0)) <= 0:
                raise ValueError(f"Voice result has no labeled examples: {count_metric}")
        coverage[baseline_id].add(corpus_kind)

    incomplete = {
        baseline: sorted(EXPECTED_CORPORA - corpora)
        for baseline, corpora in coverage.items()
        if corpora != EXPECTED_CORPORA
    }
    if incomplete:
        raise ValueError(f"Voice G1 result coverage is incomplete: {incomplete}")
    if len(set(reference_by_corpus.values())) != len(EXPECTED_CORPORA):
        raise ValueError("public and synthetic Voice results reused one reference set")
    return {
        "status": "G1_PASSED",
        "baseline_count": len(coverage),
        "result_count": len(results),
        "corpus_kinds": sorted(EXPECTED_CORPORA),
    }


def main() -> None:
    args = parse_args()
    inventory_path = (ROOT / args.inventory).resolve()
    run_manifest_path = (ROOT / args.run_manifest).resolve()
    inventory = load_yaml(inventory_path)
    run_manifest = load_yaml(run_manifest_path)
    results = [load_result((ROOT / path).resolve()) for path in args.results]
    summary = check_voice_g1(
        results,
        root=ROOT,
        inventory=inventory,
        run_manifest=run_manifest,
        inventory_sha256=sha256_file(inventory_path),
        run_manifest_sha256=sha256_file(run_manifest_path),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
