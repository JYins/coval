"""Aggregate frozen Voice references and normalized baseline predictions."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Any

from src.voice.eval_contracts import (
    EvalCandidate,
    EvalInterval,
    VoicePredictionManifest,
    VoicePredictionSample,
    VoiceReferenceManifest,
    VoiceReferenceSample,
)
from src.voice.eval_metrics import (
    character_error_rate,
    diarization_error_rate,
    false_approved_fact_rate,
    jaccard_error_rate,
    latency_percentiles,
    normalize_text,
    real_time_factor,
    review_abstention_rates,
    speaker_attributed_cer,
    vad_interval_scores,
)


def overlap_ms(left: EvalInterval, right: EvalInterval) -> int:
    return max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))


def build_speaker_map(
    reference: Iterable[EvalInterval],
    hypothesis: Iterable[EvalInterval],
) -> dict[str, str]:
    """Find a one-to-one anonymous-speaker mapping with maximum time overlap."""
    ref_turns = [row for row in reference if row.speaker_label]
    hyp_turns = [row for row in hypothesis if row.speaker_label]
    ref_labels = sorted({str(row.speaker_label) for row in ref_turns})
    hyp_labels = sorted({str(row.speaker_label) for row in hyp_turns})
    weights = {
        (hyp_label, ref_label): sum(
            overlap_ms(hyp, ref)
            for hyp in hyp_turns
            if hyp.speaker_label == hyp_label
            for ref in ref_turns
            if ref.speaker_label == ref_label
        )
        for hyp_label in hyp_labels
        for ref_label in ref_labels
    }

    @lru_cache(maxsize=None)
    def assign(hyp_index: int, used_mask: int) -> tuple[int, tuple[int, ...]]:
        if hyp_index == len(hyp_labels):
            return 0, ()
        best_score, best_choices = assign(hyp_index + 1, used_mask)
        best_choices = (-1, *best_choices)
        for ref_index, ref_label in enumerate(ref_labels):
            if used_mask & (1 << ref_index):
                continue
            tail_score, tail_choices = assign(
                hyp_index + 1,
                used_mask | (1 << ref_index),
            )
            score = weights[(hyp_labels[hyp_index], ref_label)] + tail_score
            choices = (ref_index, *tail_choices)
            if score > best_score:
                best_score = score
                best_choices = choices
        return best_score, best_choices

    _, choices = assign(0, 0)
    return {
        hyp_labels[index]: ref_labels[choice]
        for index, choice in enumerate(choices)
        if choice >= 0
    }


def interval_pairs(rows: Iterable[EvalInterval]) -> list[tuple[float, float]]:
    return [(row.start_ms, row.end_ms) for row in rows]


def speaker_interval_rows(
    rows: Iterable[EvalInterval],
) -> list[tuple[float, float, str]]:
    return [
        (row.start_ms, row.end_ms, str(row.speaker_label))
        for row in rows
        if row.speaker_label is not None
    ]


def candidate_rows(
    rows: Iterable[EvalCandidate],
    speaker_map: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    mapping = dict(speaker_map or {})
    return [
        {
            "candidate_type": row.candidate_type,
            "speaker_id": mapping.get(row.speaker_label, row.speaker_label),
            "text": row.content,
        }
        for row in rows
    ]


def speaker_text_from_turns(
    rows: Iterable[EvalInterval],
    speaker_map: dict[str, str] | None = None,
) -> dict[str, str]:
    mapping = dict(speaker_map or {})
    collected: dict[str, list[str]] = {}
    for row in sorted(rows, key=lambda item: (item.start_ms, item.end_ms)):
        speaker = str(row.speaker_label)
        target = mapping.get(speaker, speaker)
        collected.setdefault(target, []).append(str(row.text))
    return {speaker: " ".join(texts) for speaker, texts in collected.items()}


def candidate_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row["candidate_type"].strip().lower(),
        row["speaker_id"].strip().lower(),
        normalize_text(row["text"]),
    )


def span_iou(left: EvalCandidate, right: EvalCandidate) -> float:
    intersection = max(
        0,
        min(left.source_end_ms, right.source_end_ms)
        - max(left.source_start_ms, right.source_start_ms),
    )
    union = (
        max(left.source_end_ms, right.source_end_ms)
        - min(left.source_start_ms, right.source_start_ms)
    )
    return intersection / union if union else 0.0


def traceable_candidate_scores(
    reference: list[EvalCandidate],
    hypothesis: list[EvalCandidate],
    *,
    speaker_map: dict[str, str],
    minimum_span_iou: float = 0.5,
) -> dict[str, float | int]:
    """Match exact normalized facts only when their source spans also overlap."""
    matched_hypothesis: set[int] = set()
    true_positive = 0
    for ref in reference:
        for index, hyp in enumerate(hypothesis):
            if index in matched_hypothesis:
                continue
            same_fact = (
                ref.candidate_type == hyp.candidate_type
                and normalize_text(ref.content) == normalize_text(hyp.content)
                and ref.speaker_label
                == speaker_map.get(hyp.speaker_label, hyp.speaker_label)
            )
            if same_fact and span_iou(ref, hyp) >= minimum_span_iou:
                matched_hypothesis.add(index)
                true_positive += 1
                break
    false_positive = len(hypothesis) - true_positive
    false_negative = len(reference) - true_positive
    precision = ratio(true_positive, true_positive + false_positive)
    recall = ratio(true_positive, true_positive + false_negative)
    return {
        "reference_count": len(reference),
        "hypothesis_count": len(hypothesis),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": ratio(2 * precision * recall, precision + recall),
        "minimum_span_iou": minimum_span_iou,
    }


def slot_pairs(
    rows: Iterable[EvalCandidate],
    keys: set[str],
    speaker_map: dict[str, str] | None = None,
) -> set[tuple[str, str]]:
    mapping = dict(speaker_map or {})
    pairs = set()
    for row in rows:
        for key, value in row.slots.items():
            if key not in keys:
                continue
            mapped_value = mapping.get(value, value)
            pairs.add((key, normalize_text(mapped_value)))
    return pairs


def set_scores(
    reference: set[tuple[str, str]],
    hypothesis: set[tuple[str, str]],
) -> dict[str, float | int]:
    true_positive = len(reference & hypothesis)
    precision = ratio(true_positive, len(hypothesis), empty=1.0 if not reference else 0.0)
    recall = ratio(true_positive, len(reference), empty=1.0 if not hypothesis else 0.0)
    f1 = ratio(2 * precision * recall, precision + recall)
    return {
        "reference_count": len(reference),
        "hypothesis_count": len(hypothesis),
        "true_positive": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": float(reference == hypothesis),
    }


def attribution_accuracy(
    reference: Iterable[EvalCandidate],
    hypothesis: Iterable[EvalCandidate],
    *,
    speaker_map: dict[str, str],
    owner_slot: str | None = None,
) -> dict[str, float | int]:
    """Score speaker or action-owner labels for exact type/content matches."""
    predicted = {}
    for row in hypothesis:
        key = (row.candidate_type, normalize_text(row.content))
        value = row.speaker_label if owner_slot is None else row.slots.get(owner_slot)
        if value is not None:
            predicted[key] = speaker_map.get(value, value)
    total = correct = 0
    for row in reference:
        key = (row.candidate_type, normalize_text(row.content))
        value = row.speaker_label if owner_slot is None else row.slots.get(owner_slot)
        if value is None:
            continue
        total += 1
        correct += predicted.get(key) == value
    return {
        "reference_count": total,
        "correct": correct,
        "accuracy": ratio(correct, total, empty=1.0),
    }


def evaluate_sample(
    reference: VoiceReferenceSample,
    prediction: VoicePredictionSample,
    *,
    frame_ms: int,
    collar_ms: int,
    score_overlap: bool,
) -> dict[str, Any]:
    if prediction.sample_id != reference.sample_id:
        raise ValueError("Voice reference and prediction sample IDs do not match")
    if prediction.duration_ms != reference.duration_ms:
        raise ValueError("Voice reference and prediction durations do not match")
    speaker_map = build_speaker_map(
        reference.speaker_turns,
        prediction.speaker_turns,
    )
    reference_candidates = candidate_rows(reference.candidates)
    approved_rows = [
        row
        for row in prediction.candidates
        if prediction.candidate_decisions.get(row.candidate_id) == "approve"
    ]
    approved_candidates = candidate_rows(approved_rows, speaker_map)
    supported = {candidate_key(row) for row in reference_candidates}
    approved_validity = [candidate_key(row) in supported for row in approved_candidates]

    return {
        "sample_id": reference.sample_id,
        "duration_ms": reference.duration_ms,
        "speaker_map": speaker_map,
        "cer": character_error_rate(reference.transcript, prediction.transcript),
        "vad": vad_interval_scores(
            interval_pairs(reference.vad_intervals),
            interval_pairs(prediction.vad_intervals),
        ),
        "der": diarization_error_rate(
            speaker_interval_rows(reference.speaker_turns),
            speaker_interval_rows(prediction.speaker_turns),
            frame_ms=frame_ms,
            collar_ms=collar_ms,
            overlap_handling=(
                "score_all" if score_overlap else "ignore_reference_overlap"
            ),
            speaker_map=speaker_map,
        ),
        "jer": jaccard_error_rate(
            speaker_interval_rows(reference.speaker_turns),
            speaker_interval_rows(prediction.speaker_turns),
            speaker_map=speaker_map,
        ),
        "speaker_attributed_cer": speaker_attributed_cer(
            speaker_text_from_turns(reference.speaker_turns),
            speaker_text_from_turns(prediction.speaker_turns, speaker_map),
        ),
        "structured": traceable_candidate_scores(
            reference.candidates,
            prediction.candidates,
            speaker_map=speaker_map,
        ),
        "proper_names": set_scores(
            slot_pairs(reference.candidates, {"name", "person_name", "proper_name"}),
            slot_pairs(
                prediction.candidates,
                {"name", "person_name", "proper_name"},
                speaker_map,
            ),
        ),
        "numbers_dates_units": set_scores(
            slot_pairs(reference.candidates, {"number", "date", "unit"}),
            slot_pairs(
                prediction.candidates,
                {"number", "date", "unit"},
                speaker_map,
            ),
        ),
        "speaker_attribution": attribution_accuracy(
            reference.candidates,
            prediction.candidates,
            speaker_map=speaker_map,
        ),
        "action_owner": attribution_accuracy(
            reference.candidates,
            prediction.candidates,
            speaker_map=speaker_map,
            owner_slot="action_owner",
        ),
        "safety": false_approved_fact_rate(approved_validity),
        "review": review_abstention_rates(
            len(prediction.candidates),
            len(prediction.candidate_decisions),
            sum(
                decision == "abstain"
                for decision in prediction.candidate_decisions.values()
            ),
        ),
        "runtime": {
            "processing_seconds": prediction.processing_seconds,
            "rtf": real_time_factor(
                prediction.processing_seconds * 1000,
                reference.duration_ms,
            ),
            "first_partial_latency_ms": prediction.first_partial_latency_ms,
            "completion_latency_ms": prediction.completion_latency_ms,
            "peak_ram_mb": prediction.peak_ram_mb,
            "peak_vram_mb": prediction.peak_vram_mb,
        },
    }


def ratio(numerator: float, denominator: float, *, empty: float = 0.0) -> float:
    return numerator / denominator if denominator else empty


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cer_errors = sum(row["cer"]["errors"] for row in rows)
    cer_chars = sum(row["cer"]["reference_chars"] for row in rows)
    vad_overlap = sum(row["vad"]["overlap_ms"] for row in rows)
    vad_ref = sum(row["vad"]["reference_speech_ms"] for row in rows)
    vad_hyp = sum(row["vad"]["hypothesis_speech_ms"] for row in rows)
    vad_precision = ratio(vad_overlap, vad_hyp, empty=1.0 if vad_ref == 0 else 0.0)
    vad_recall = ratio(vad_overlap, vad_ref, empty=1.0 if vad_hyp == 0 else 0.0)
    vad_f1 = ratio(2 * vad_precision * vad_recall, vad_precision + vad_recall)
    der_miss = sum(row["der"]["miss_ms"] for row in rows)
    der_false_alarm = sum(row["der"]["false_alarm_ms"] for row in rows)
    der_confusion = sum(row["der"]["confusion_ms"] for row in rows)
    der_scored = sum(row["der"]["scored_ms"] for row in rows)
    jer_values = [
        value
        for row in rows
        for value in row["jer"]["per_speaker"].values()
    ]
    sa_errors = sum(row["speaker_attributed_cer"]["errors"] for row in rows)
    sa_chars = sum(row["speaker_attributed_cer"]["reference_chars"] for row in rows)
    structured_tp = sum(row["structured"]["true_positive"] for row in rows)
    structured_fp = sum(row["structured"]["false_positive"] for row in rows)
    structured_fn = sum(row["structured"]["false_negative"] for row in rows)
    structured_precision = ratio(structured_tp, structured_tp + structured_fp)
    structured_recall = ratio(structured_tp, structured_tp + structured_fn)
    structured_f1 = ratio(
        2 * structured_precision * structured_recall,
        structured_precision + structured_recall,
    )
    approved = sum(row["safety"]["approved_facts"] for row in rows)
    false_approved = sum(row["safety"]["false_approved_facts"] for row in rows)
    total_candidates = sum(row["review"]["total_candidates"] for row in rows)
    reviewed = sum(row["review"]["reviewed_candidates"] for row in rows)
    abstained = sum(row["review"]["abstained_candidates"] for row in rows)
    proper_name_ref = sum(row["proper_names"]["reference_count"] for row in rows)
    proper_name_tp = sum(row["proper_names"]["true_positive"] for row in rows)
    number_exact = [
        row["numbers_dates_units"]["exact_match"]
        for row in rows
        if row["numbers_dates_units"]["reference_count"] > 0
    ]
    number_ref = sum(row["numbers_dates_units"]["reference_count"] for row in rows)
    number_hyp = sum(row["numbers_dates_units"]["hypothesis_count"] for row in rows)
    number_tp = sum(row["numbers_dates_units"]["true_positive"] for row in rows)
    speaker_ref = sum(row["speaker_attribution"]["reference_count"] for row in rows)
    speaker_correct = sum(row["speaker_attribution"]["correct"] for row in rows)
    owner_ref = sum(row["action_owner"]["reference_count"] for row in rows)
    owner_correct = sum(row["action_owner"]["correct"] for row in rows)
    total_runtime_ms = sum(row["runtime"]["processing_seconds"] * 1000 for row in rows)
    total_audio_ms = sum(row["duration_ms"] for row in rows)
    completion = [row["runtime"]["completion_latency_ms"] for row in rows]
    first_partial = [
        row["runtime"]["first_partial_latency_ms"]
        for row in rows
        if row["runtime"]["first_partial_latency_ms"] is not None
    ]

    return {
        "sample_count": len(rows),
        "audio_duration_ms": total_audio_ms,
        "cer_reference_chars": cer_chars,
        "cer": ratio(cer_errors, cer_chars, empty=1.0 if cer_errors else 0.0),
        "vad_reference_speech_ms": vad_ref,
        "vad_precision": vad_precision,
        "vad_recall": vad_recall,
        "vad_f1": vad_f1,
        "der": ratio(der_miss + der_false_alarm + der_confusion, der_scored),
        "der_scored_ms": der_scored,
        "jer": sum(jer_values) / len(jer_values) if jer_values else None,
        "jer_speaker_count": len(jer_values),
        "speaker_attributed_cer": ratio(
            sa_errors,
            sa_chars,
            empty=1.0 if sa_errors else 0.0,
        ),
        "structured_precision": structured_precision,
        "structured_recall": structured_recall,
        "structured_f1": structured_f1,
        "structured_reference_count": structured_tp + structured_fn,
        "proper_name_recall": (
            proper_name_tp / proper_name_ref if proper_name_ref else None
        ),
        "proper_name_reference_count": proper_name_ref,
        "number_date_unit_exact_match": (
            sum(number_exact) / len(number_exact) if number_exact else None
        ),
        "number_date_unit_sample_count": len(number_exact),
        "number_date_unit_precision": ratio(number_tp, number_hyp),
        "number_date_unit_recall": ratio(number_tp, number_ref),
        "number_date_unit_f1": ratio(
            2 * ratio(number_tp, number_hyp) * ratio(number_tp, number_ref),
            ratio(number_tp, number_hyp) + ratio(number_tp, number_ref),
        ),
        "speaker_attribution_accuracy": (
            speaker_correct / speaker_ref if speaker_ref else None
        ),
        "speaker_attribution_reference_count": speaker_ref,
        "action_owner_accuracy": owner_correct / owner_ref if owner_ref else None,
        "action_owner_reference_count": owner_ref,
        "false_approved_fact_rate": ratio(false_approved, approved),
        "approved_fact_count": approved,
        "false_approved_fact_count": false_approved,
        "review_rate": ratio(reviewed, total_candidates),
        "abstention_rate": ratio(abstained, total_candidates),
        "rtf": real_time_factor(total_runtime_ms, total_audio_ms),
        "first_partial_latency_ms": latency_percentiles(first_partial, (50, 95)),
        "completion_latency_ms": latency_percentiles(completion, (50, 95)),
        "peak_ram_mb": max(row["runtime"]["peak_ram_mb"] for row in rows),
        "peak_vram_mb": max(
            (
                row["runtime"]["peak_vram_mb"]
                for row in rows
                if row["runtime"]["peak_vram_mb"] is not None
            ),
            default=None,
        ),
    }


def evaluate_manifests(
    reference: VoiceReferenceManifest,
    prediction: VoicePredictionManifest,
    *,
    frame_ms: int = 10,
    collar_ms: int = 250,
    score_overlap: bool = True,
) -> dict[str, Any]:
    references = {row.sample_id: row for row in reference.samples}
    predictions = {row.sample_id: row for row in prediction.samples}
    if len(references) != len(reference.samples):
        raise ValueError("reference manifest contains duplicate sample IDs")
    if len(predictions) != len(prediction.samples):
        raise ValueError("prediction manifest contains duplicate sample IDs")
    if references.keys() != predictions.keys():
        missing = sorted(references.keys() - predictions.keys())
        extra = sorted(predictions.keys() - references.keys())
        raise ValueError(f"Voice sample IDs differ: missing={missing}, extra={extra}")

    per_sample = [
        evaluate_sample(
            references[sample_id],
            predictions[sample_id],
            frame_ms=frame_ms,
            collar_ms=collar_ms,
            score_overlap=score_overlap,
        )
        for sample_id in sorted(references)
    ]
    return {
        "schema_version": "1",
        "dataset": {
            "name": reference.dataset_name,
            "split": reference.dataset_split,
            "corpus_kind": reference.corpus_kind,
            "license_name": reference.license_name,
            "license_url": reference.license_url,
            "source_sha256": reference.dataset_source_sha256,
        },
        "baseline": {
            "baseline_id": prediction.baseline_id,
            "runtime_artifact_id": prediction.runtime_artifact_id,
            "runtime_name": prediction.runtime_name,
            "runtime_version": prediction.runtime_version,
            "runtime_revision": prediction.runtime_revision,
            "runtime_sha256": prediction.runtime_sha256,
            "model_artifact_id": prediction.model_artifact_id,
            "model_name": prediction.model_name,
            "model_revision": prediction.model_revision,
            "model_sha256": prediction.model_sha256,
            "config_sha256": prediction.config_sha256,
            "hardware": prediction.hardware,
            "model_size_mb": prediction.model_size_mb,
            "artifact_pins": [row.model_dump() for row in prediction.artifact_pins],
        },
        "metric_config": {
            "der_frame_ms": frame_ms,
            "der_collar_ms": collar_ms,
            "score_overlap": score_overlap,
        },
        "summary": aggregate_results(per_sample),
        "per_sample": per_sample,
    }
