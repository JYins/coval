"""Unit tests for the provider-independent Voice G1 metrics."""

from __future__ import annotations

import math

import pytest

from src.voice.eval_metrics import (
    character_error_rate,
    diarization_error_rate,
    false_approved_fact_rate,
    jaccard_error_rate,
    latency_percentile,
    latency_percentiles,
    normalize_text,
    real_time_factor,
    review_abstention_rates,
    speaker_attributed_cer,
    structured_candidate_scores,
    vad_interval_scores,
)


def test_normalizes_chinese_mixed_text_and_scores_character_cer():
    assert normalize_text("你 好，World！１２３") == "你好world123"

    score = character_error_rate("你好，World！１２３", "你好 world 123")

    assert score["reference_chars"] == 10
    assert score["errors"] == 0
    assert score["cer"] == 0.0


def test_cer_has_explicit_empty_reference_rule():
    assert character_error_rate("", "")["cer"] == 0.0
    assert character_error_rate("", "多出来")["cer"] == 1.0


def test_vad_scores_overlap_and_empty_cases():
    score = vad_interval_scores([(0, 100), (200, 300)], [(50, 150), (250, 350)])

    assert score["overlap_ms"] == 100
    assert score["precision"] == 0.5
    assert score["recall"] == 0.5
    assert score["f1"] == 0.5
    assert vad_interval_scores([], [])["f1"] == 1.0
    assert vad_interval_scores([(0, 1)], [])["f1"] == 0.0


def test_der_reports_speaker_confusion_and_optional_label_mapping():
    reference = [(0, 100, "alice")]
    hypothesis = [(0, 100, "speaker_0")]

    confused = diarization_error_rate(reference, hypothesis, frame_ms=10)
    mapped = diarization_error_rate(
        reference, hypothesis, frame_ms=10, speaker_map={"speaker_0": "alice"}
    )

    assert confused == {
        "miss_ms": 0.0,
        "false_alarm_ms": 0.0,
        "confusion_ms": 100.0,
        "scored_ms": 100.0,
        "der": 1.0,
    }
    assert mapped["der"] == 0.0


def test_der_overlap_handling_and_collar_are_reproducible():
    reference = [(0, 100, "alice"), (0, 100, "bob")]
    hypothesis = [(0, 100, "alice")]

    score_all = diarization_error_rate(reference, hypothesis, frame_ms=10)
    ignore_overlap = diarization_error_rate(
        reference, hypothesis, frame_ms=10, overlap_handling="ignore_reference_overlap"
    )
    collared = diarization_error_rate(
        [(0, 100, "alice")], [(0, 100, "alice")], frame_ms=10, collar_ms=10
    )

    assert score_all["miss_ms"] == 100.0
    assert score_all["scored_ms"] == 200.0
    assert score_all["der"] == 0.5
    assert ignore_overlap["scored_ms"] == 0.0
    assert ignore_overlap["der"] == 0.0
    assert collared["scored_ms"] == 80.0


def test_speaker_attributed_cer_aggregates_per_speaker():
    score = speaker_attributed_cer(
        {"alice": ["你好", "世界"], "bob": "ok"},
        {"alice": "你好 世界", "bob": "ox"},
    )

    assert score["per_speaker"]["alice"]["cer"] == 0.0
    assert score["per_speaker"]["bob"]["substitutions"] == 1
    assert score["reference_chars"] == 6
    assert score["cer"] == pytest.approx(1 / 6)


def test_jaccard_error_rate_maps_speakers_and_penalizes_missing_time():
    score = jaccard_error_rate(
        [(0, 100, "alice")],
        [(0, 50, "speaker_0")],
        speaker_map={"speaker_0": "alice"},
    )

    assert score["per_speaker"]["alice"] == 0.5
    assert score["jer"] == 0.5


def test_structured_candidate_scores_use_exact_normalized_tuples():
    reference = [
        {"candidate_type": "commitment", "speaker_id": "Lin", "text": "周五 跟进。"},
        {"candidate_type": "preference", "speaker_id": "Mia", "text": "喜欢咖啡"},
    ]
    hypothesis = [
        {"candidate_type": "COMMITMENT", "speaker_id": "lin", "text": "周五跟进"},
        {"candidate_type": "preference", "speaker_id": "mia", "text": "喜欢茶"},
    ]

    score = structured_candidate_scores(reference, hypothesis)

    assert score["true_positive"] == 1
    assert score["false_positive"] == 1
    assert score["false_negative"] == 1
    assert score["precision"] == score["recall"] == score["f1"] == 0.5


def test_safety_review_and_runtime_helpers():
    assert false_approved_fact_rate([True, False, True]) == {
        "approved_facts": 3,
        "false_approved_facts": 1,
        "false_approved_fact_rate": pytest.approx(1 / 3),
    }
    assert false_approved_fact_rate([])["false_approved_fact_rate"] == 0.0
    assert review_abstention_rates(10, 8, 3)["review_rate"] == 0.8
    assert review_abstention_rates(10, 8, 3)["abstention_rate"] == 0.3
    assert review_abstention_rates(0, 0, 0)["review_rate"] == 0.0
    assert real_time_factor(250, 1000) == 0.25
    assert real_time_factor(1, 0) == math.inf
    assert latency_percentile([1, 2, 3, 4], 50) == 2.5
    assert latency_percentiles([1, 2, 3, 4]) == {
        "p50": 2.5,
        "p95": pytest.approx(3.85),
        "p99": pytest.approx(3.97),
    }
    assert latency_percentile([], 95) is None


def test_invalid_metric_inputs_fail_loudly():
    with pytest.raises(ValueError):
        diarization_error_rate([], [], frame_ms=0)
    with pytest.raises(ValueError):
        diarization_error_rate([], [], overlap_handling="unknown")
    with pytest.raises(ValueError):
        review_abstention_rates(1, 0, 1)
    with pytest.raises(TypeError):
        false_approved_fact_rate([1])
