"""Small, dependency-free metrics for the Voice G1 experiment harness.

The helpers here deliberately score normalized provider output instead of
calling a speech model.  This keeps one evaluation contract for all runtime
providers and makes every zero-denominator rule visible in code.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from itertools import chain
from typing import Any


Interval = tuple[float, float]
SpeakerInterval = tuple[float, float, str]


def normalize_text(text: str) -> str:
    """Normalize Chinese or mixed-language text before character scoring.

    It applies Unicode NFKC, lowercases Latin letters, and removes whitespace,
    punctuation, and symbols. Chinese characters and both Arabic and Chinese
    number characters are kept; this function does not transliterate numbers.
    """
    value = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        char
        for char in value
        if unicodedata.category(char)[0] in {"L", "M", "N"}
    )


def character_error_rate(reference: str, hypothesis: str) -> dict[str, float | int]:
    """Return character-level CER after :func:`normalize_text`.

    For an empty reference, CER is zero only when the hypothesis is also empty;
    otherwise it is one. This avoids an undefined denominator while retaining a
    bounded error rate for a spurious transcript.
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    substitutions, deletions, insertions = _edit_counts(ref, hyp)
    errors = substitutions + deletions + insertions
    if not ref:
        cer = 0.0 if not hyp else 1.0
    else:
        cer = errors / len(ref)
    return {
        "reference_chars": len(ref),
        "hypothesis_chars": len(hyp),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,
        "cer": cer,
    }


def vad_interval_scores(
    reference: Iterable[Interval], hypothesis: Iterable[Interval]
) -> dict[str, float]:
    """Score VAD intervals by milliseconds of speech overlap.

    Intervals are treated as half-open ``[start_ms, end_ms)`` and overlapping
    intervals from one side are unioned first. If both sides contain no speech,
    precision, recall, and F1 are all 1.0. If only one side is empty, all three
    values are 0.0.
    """
    ref = _merge_intervals(reference)
    hyp = _merge_intervals(hypothesis)
    ref_ms = _interval_duration(ref)
    hyp_ms = _interval_duration(hyp)
    overlap_ms = _overlap_duration(ref, hyp)

    if ref_ms == 0 and hyp_ms == 0:
        precision = recall = f1 = 1.0
    elif ref_ms == 0 or hyp_ms == 0:
        precision = recall = f1 = 0.0
    else:
        precision = overlap_ms / hyp_ms
        recall = overlap_ms / ref_ms
        f1 = _f1(precision, recall)

    return {
        "reference_speech_ms": ref_ms,
        "hypothesis_speech_ms": hyp_ms,
        "overlap_ms": overlap_ms,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def diarization_error_rate(
    reference: Iterable[SpeakerInterval],
    hypothesis: Iterable[SpeakerInterval],
    *,
    frame_ms: float = 10.0,
    collar_ms: float = 0.0,
    overlap_handling: str = "score_all",
    speaker_map: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Calculate a simple, reproducible frame-based diarization error rate.

    ``scored_ms`` is reference *speaker-time*: two simultaneous reference
    speakers contribute two frame durations when ``overlap_handling`` is
    ``"score_all"``. ``"ignore_reference_overlap"`` skips frames with two or
    more reference speakers. A collar excludes frames whose centre falls within
    ``collar_ms`` of any reference segment boundary.

    This is intentionally not a full pyannote-style DER implementation. Speaker
    labels are compared directly after the optional ``speaker_map`` (hypothesis
    label -> reference label); callers with anonymous diarization labels should
    provide that mapping explicitly. In a scored frame, unmatched paired speaker
    activity is counted as confusion, surplus reference activity as miss, and
    surplus hypothesis activity as false alarm.
    """
    if frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    if collar_ms < 0:
        raise ValueError("collar_ms cannot be negative")
    if overlap_handling not in {"score_all", "ignore_reference_overlap"}:
        raise ValueError("overlap_handling must be score_all or ignore_reference_overlap")

    ref = _speaker_intervals(reference)
    hyp = _speaker_intervals(hypothesis)
    all_segments = list(chain(ref, hyp))
    if not all_segments:
        return _der_result(0.0, 0.0, 0.0, 0.0)

    start_ms = min(item[0] for item in all_segments)
    end_ms = max(item[1] for item in all_segments)
    boundaries = [boundary for start, end, _ in ref for boundary in (start, end)]
    mapping = dict(speaker_map or {})
    miss_ms = false_alarm_ms = confusion_ms = scored_ms = 0.0

    frame_start = start_ms
    while frame_start < end_ms:
        frame_end = min(frame_start + frame_ms, end_ms)
        width = frame_end - frame_start
        centre = frame_start + width / 2
        frame_start = frame_end
        if any(abs(centre - boundary) <= collar_ms for boundary in boundaries):
            continue

        ref_speakers = _active_speakers(ref, centre)
        hyp_speakers = {
            mapping.get(speaker, speaker) for speaker in _active_speakers(hyp, centre)
        }
        if overlap_handling == "ignore_reference_overlap" and len(ref_speakers) > 1:
            continue
        if not ref_speakers and not hyp_speakers:
            continue

        matched = len(ref_speakers & hyp_speakers)
        paired_unmatched = min(len(ref_speakers), len(hyp_speakers)) - matched
        miss_ms += max(len(ref_speakers) - len(hyp_speakers), 0) * width
        false_alarm_ms += max(len(hyp_speakers) - len(ref_speakers), 0) * width
        confusion_ms += paired_unmatched * width
        scored_ms += len(ref_speakers) * width

    return _der_result(miss_ms, false_alarm_ms, confusion_ms, scored_ms)


def speaker_attributed_cer(
    reference: Mapping[str, str | Iterable[str]],
    hypothesis: Mapping[str, str | Iterable[str]],
) -> dict[str, Any]:
    """Score CER after concatenating each speaker's own turns in turn order.

    This does not perform speaker matching: map anonymous diarization labels to
    the reference speaker identifiers before calling it. Extra hypothesis-only
    speakers contribute insertions to the aggregate error count.
    """
    per_speaker: dict[str, dict[str, float | int]] = {}
    for speaker in sorted(set(reference) | set(hypothesis)):
        per_speaker[speaker] = character_error_rate(
            _join_speaker_text(reference.get(speaker, "")),
            _join_speaker_text(hypothesis.get(speaker, "")),
        )

    reference_chars = sum(item["reference_chars"] for item in per_speaker.values())
    hypothesis_chars = sum(item["hypothesis_chars"] for item in per_speaker.values())
    substitutions = sum(item["substitutions"] for item in per_speaker.values())
    deletions = sum(item["deletions"] for item in per_speaker.values())
    insertions = sum(item["insertions"] for item in per_speaker.values())
    errors = substitutions + deletions + insertions
    cer = 0.0 if reference_chars == 0 and errors == 0 else (1.0 if reference_chars == 0 else errors / reference_chars)
    return {
        "per_speaker": per_speaker,
        "reference_chars": reference_chars,
        "hypothesis_chars": hypothesis_chars,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,
        "cer": cer,
    }


def jaccard_error_rate(
    reference: Iterable[SpeakerInterval],
    hypothesis: Iterable[SpeakerInterval],
    *,
    speaker_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return mean per-reference-speaker Jaccard error over active time.

    Hypothesis labels are mapped before scoring. Reference speakers with no
    activity are not included. Extra hypothesis-only speakers receive JER 1.0.
    This interval-union definition has no collar and is reported separately
    from the frame/collar DER configuration.
    """
    ref = _speaker_intervals(reference)
    mapping = dict(speaker_map or {})
    hyp = [
        (start, end, mapping.get(speaker, speaker))
        for start, end, speaker in _speaker_intervals(hypothesis)
    ]
    speakers = sorted({row[2] for row in ref} | {row[2] for row in hyp})
    per_speaker: dict[str, float] = {}
    for speaker in speakers:
        ref_intervals = _merge_intervals(
            (start, end) for start, end, label in ref if label == speaker
        )
        hyp_intervals = _merge_intervals(
            (start, end) for start, end, label in hyp if label == speaker
        )
        intersection = _overlap_duration(ref_intervals, hyp_intervals)
        union = (
            _interval_duration(ref_intervals)
            + _interval_duration(hyp_intervals)
            - intersection
        )
        per_speaker[speaker] = 0.0 if union == 0 else 1 - intersection / union
    return {
        "per_speaker": per_speaker,
        "jer": sum(per_speaker.values()) / len(per_speaker) if per_speaker else 0.0,
    }


def structured_candidate_scores(
    reference: Iterable[Mapping[str, Any]], hypothesis: Iterable[Mapping[str, Any]]
) -> dict[str, float | int]:
    """Score de-duplicated exact normalized candidate tuples.

    Each candidate must contain ``candidate_type``, ``speaker_id``, and ``text``.
    Comparison uses ``(lowercased type, lowercased speaker, normalize_text(text))``.
    Both empty sets receive precision, recall, and F1 of 1.0; exactly one empty
    set receives zeros.
    """
    ref = {_candidate_tuple(item) for item in reference}
    hyp = {_candidate_tuple(item) for item in hypothesis}
    true_positive = len(ref & hyp)
    false_positive = len(hyp - ref)
    false_negative = len(ref - hyp)
    if not ref and not hyp:
        precision = recall = f1 = 1.0
    elif not ref or not hyp:
        precision = recall = f1 = 0.0
    else:
        precision = true_positive / len(hyp)
        recall = true_positive / len(ref)
        f1 = _f1(precision, recall)
    return {
        "reference_count": len(ref),
        "hypothesis_count": len(hyp),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def false_approved_fact_rate(factual_validity: Iterable[bool]) -> dict[str, float | int]:
    """Return post-audit false-approved fact rate for approved candidates only.

    ``factual_validity`` has one boolean per approved fact after a human audit:
    ``True`` means supported by the source and ``False`` means false. With no
    approved facts the rate is 0.0, because no unsafe approval occurred.
    """
    values = list(factual_validity)
    if any(not isinstance(value, bool) for value in values):
        raise TypeError("factual_validity must contain booleans")
    false_approved = sum(not value for value in values)
    return {
        "approved_facts": len(values),
        "false_approved_facts": false_approved,
        "false_approved_fact_rate": false_approved / len(values) if values else 0.0,
    }


def review_abstention_rates(
    total_candidates: int, reviewed_candidates: int, abstained_candidates: int
) -> dict[str, float | int]:
    """Return review and abstention rates for a labeled candidate batch.

    Abstentions are a subset of reviewed candidates. For an empty batch both
    rates are 0.0 rather than undefined.
    """
    if min(total_candidates, reviewed_candidates, abstained_candidates) < 0:
        raise ValueError("candidate counts cannot be negative")
    if reviewed_candidates > total_candidates:
        raise ValueError("reviewed_candidates cannot exceed total_candidates")
    if abstained_candidates > reviewed_candidates:
        raise ValueError("abstained_candidates cannot exceed reviewed_candidates")
    if total_candidates == 0:
        return {
            "total_candidates": 0,
            "reviewed_candidates": 0,
            "abstained_candidates": 0,
            "review_rate": 0.0,
            "abstention_rate": 0.0,
        }
    return {
        "total_candidates": total_candidates,
        "reviewed_candidates": reviewed_candidates,
        "abstained_candidates": abstained_candidates,
        "review_rate": reviewed_candidates / total_candidates,
        "abstention_rate": abstained_candidates / total_candidates,
    }


def real_time_factor(runtime_ms: float, audio_ms: float) -> float:
    """Return runtime/audio duration; zero audio is 0.0 only for zero runtime."""
    if runtime_ms < 0 or audio_ms < 0:
        raise ValueError("durations cannot be negative")
    if audio_ms == 0:
        return 0.0 if runtime_ms == 0 else math.inf
    return runtime_ms / audio_ms


def latency_percentile(latencies_ms: Sequence[float], percentile: float) -> float | None:
    """Calculate a linear-interpolated percentile over millisecond samples.

    The rank is ``(n - 1) * percentile / 100``. Empty input returns ``None``;
    percentile must be between 0 and 100 inclusive and latencies non-negative.
    """
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    if any(value < 0 for value in latencies_ms):
        raise ValueError("latencies cannot be negative")
    if not latencies_ms:
        return None
    values = sorted(float(value) for value in latencies_ms)
    position = (len(values) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def latency_percentiles(
    latencies_ms: Sequence[float], percentiles: Iterable[float] = (50, 95, 99)
) -> dict[str, float | None]:
    """Convenience wrapper returning named percentile values."""
    return {f"p{percentile:g}": latency_percentile(latencies_ms, percentile) for percentile in percentiles}


def _edit_counts(reference: str, hypothesis: str) -> tuple[int, int, int]:
    # Each cell keeps (distance, substitutions, deletions, insertions).
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    matrix = [[(0, 0, 0, 0) for _ in range(columns)] for _ in range(rows)]
    for index in range(1, rows):
        matrix[index][0] = (index, 0, index, 0)
    for index in range(1, columns):
        matrix[0][index] = (index, 0, 0, index)
    for row, ref_char in enumerate(reference, start=1):
        for column, hyp_char in enumerate(hypothesis, start=1):
            if ref_char == hyp_char:
                matrix[row][column] = matrix[row - 1][column - 1]
                continue
            distance, sub, delete, insert = matrix[row - 1][column - 1]
            substitute = (distance + 1, sub + 1, delete, insert)
            distance, sub, delete, insert = matrix[row - 1][column]
            remove = (distance + 1, sub, delete + 1, insert)
            distance, sub, delete, insert = matrix[row][column - 1]
            add = (distance + 1, sub, delete, insert + 1)
            matrix[row][column] = min(substitute, remove, add)
    _, substitutions, deletions, insertions = matrix[-1][-1]
    return substitutions, deletions, insertions


def _merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    values = sorted(_validate_interval(start, end) for start, end in intervals)
    merged: list[Interval] = []
    for start, end in values:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _validate_interval(start: float, end: float) -> Interval:
    if end < start:
        raise ValueError("interval end must be greater than or equal to start")
    return float(start), float(end)


def _interval_duration(intervals: Iterable[Interval]) -> float:
    return sum(end - start for start, end in intervals)


def _overlap_duration(left: Sequence[Interval], right: Sequence[Interval]) -> float:
    left_index = right_index = 0
    overlap = 0.0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        overlap += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return overlap


def _speaker_intervals(intervals: Iterable[SpeakerInterval]) -> list[SpeakerInterval]:
    values: list[SpeakerInterval] = []
    for start, end, speaker in intervals:
        checked_start, checked_end = _validate_interval(start, end)
        if checked_start != checked_end:
            values.append((checked_start, checked_end, str(speaker)))
    return values


def _active_speakers(intervals: Iterable[SpeakerInterval], time_ms: float) -> set[str]:
    return {speaker for start, end, speaker in intervals if start <= time_ms < end}


def _der_result(miss_ms: float, false_alarm_ms: float, confusion_ms: float, scored_ms: float) -> dict[str, float]:
    errors = miss_ms + false_alarm_ms + confusion_ms
    return {
        "miss_ms": miss_ms,
        "false_alarm_ms": false_alarm_ms,
        "confusion_ms": confusion_ms,
        "scored_ms": scored_ms,
        "der": errors / scored_ms if scored_ms else 0.0,
    }


def _join_speaker_text(value: str | Iterable[str]) -> str:
    return value if isinstance(value, str) else " ".join(value)


def _candidate_tuple(candidate: Mapping[str, Any]) -> tuple[str, str, str]:
    try:
        candidate_type = str(candidate["candidate_type"]).strip().lower()
        speaker_id = str(candidate["speaker_id"]).strip().lower()
        text = normalize_text(str(candidate["text"]))
    except KeyError as error:
        raise ValueError(f"candidate is missing {error.args[0]}") from error
    return candidate_type, speaker_id, text


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
