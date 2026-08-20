"""Typed JSON contracts shared by local Voice baseline adapters and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


SHA256_PATTERN = r"^[0-9a-f]{64}$"


class EvalInterval(BaseModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    speaker_label: str | None = None
    overlap: bool = False
    text: str | None = None

    @model_validator(mode="after")
    def check_time_range(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("interval end_ms should be after start_ms")
        return self


class EvalCandidate(BaseModel):
    candidate_id: str = Field(min_length=1, max_length=255)
    candidate_type: Literal[
        "stated_fact",
        "stated_preference",
        "commitment",
        "follow_up_action",
    ]
    content: str = Field(min_length=1, max_length=4000)
    speaker_label: str
    source_start_ms: int = Field(ge=0)
    source_end_ms: int = Field(gt=0)
    slots: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_source_range(self):
        if self.source_end_ms <= self.source_start_ms:
            raise ValueError("candidate source_end_ms should be after source_start_ms")
        return self


class VoiceReferenceSample(BaseModel):
    sample_id: str = Field(min_length=1, max_length=255)
    duration_ms: int = Field(gt=0)
    vad_intervals: list[EvalInterval]
    speaker_turns: list[EvalInterval]
    transcript: str
    candidates: list[EvalCandidate]

    @model_validator(mode="after")
    def check_evidence_links(self):
        validate_sample_evidence(
            self.duration_ms,
            self.speaker_turns,
            self.candidates,
        )
        return self


class VoicePredictionSample(BaseModel):
    sample_id: str = Field(min_length=1, max_length=255)
    duration_ms: int = Field(gt=0)
    vad_intervals: list[EvalInterval]
    speaker_turns: list[EvalInterval]
    transcript: str
    candidates: list[EvalCandidate]
    candidate_decisions: dict[
        str,
        Literal["approve", "reject", "abstain"],
    ] = Field(default_factory=dict)
    processing_seconds: float = Field(gt=0)
    first_partial_latency_ms: float | None = Field(default=None, ge=0)
    completion_latency_ms: float = Field(ge=0)
    peak_ram_mb: float = Field(gt=0)
    peak_vram_mb: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def check_review_counts(self):
        validate_sample_evidence(self.duration_ms, self.speaker_turns, self.candidates)
        candidate_ids = {row.candidate_id for row in self.candidates}
        unknown = self.candidate_decisions.keys() - candidate_ids
        if unknown:
            raise ValueError(f"decisions reference unknown candidates: {sorted(unknown)}")
        return self


class VoiceReferenceManifest(BaseModel):
    schema_version: Literal["1"]
    dataset_name: str = Field(min_length=1)
    dataset_artifact_id: str = Field(min_length=1)
    dataset_split: str = Field(min_length=1)
    corpus_kind: Literal["public", "synthetic"]
    license_name: str = Field(min_length=1)
    license_url: str = Field(min_length=1)
    dataset_source_sha256: str = Field(pattern=SHA256_PATTERN)
    samples: list[VoiceReferenceSample] = Field(min_length=1)


class VoiceArtifactPin(BaseModel):
    artifact_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)


class VoicePredictionManifest(BaseModel):
    schema_version: Literal["1"]
    baseline_id: str = Field(min_length=1)
    runtime_artifact_id: str = Field(min_length=1)
    runtime_name: str = Field(min_length=1)
    runtime_version: str = Field(min_length=1)
    runtime_revision: str = Field(min_length=1)
    runtime_sha256: str = Field(pattern=SHA256_PATTERN)
    model_artifact_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    model_sha256: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    hardware: dict[str, str]
    model_size_mb: float = Field(gt=0)
    artifact_pins: list[VoiceArtifactPin] = Field(min_length=1)
    samples: list[VoicePredictionSample] = Field(min_length=1)

    @model_validator(mode="after")
    def check_reproducibility_metadata(self):
        required = {"os", "cpu", "python", "device", "threads"}
        missing = required - self.hardware.keys()
        if missing:
            raise ValueError(f"hardware metadata is missing: {sorted(missing)}")
        for field in ("runtime_version", "runtime_revision", "model_revision"):
            if getattr(self, field) == "REQUIRED/TODO":
                raise ValueError(f"{field} must be pinned")
        pins = {row.artifact_id: row for row in self.artifact_pins}
        if len(pins) != len(self.artifact_pins):
            raise ValueError("prediction artifact IDs must be unique")
        runtime = pins.get(self.runtime_artifact_id)
        model = pins.get(self.model_artifact_id)
        if runtime is None or model is None:
            raise ValueError("runtime and model artifacts must be pinned")
        if (
            runtime.revision != self.runtime_revision
            or runtime.sha256 != self.runtime_sha256
        ):
            raise ValueError("runtime metadata does not match its artifact pin")
        if model.revision != self.model_revision or model.sha256 != self.model_sha256:
            raise ValueError("model metadata does not match its artifact pin")
        return self


def load_reference_manifest(path: Path) -> VoiceReferenceManifest:
    with path.open("r", encoding="utf-8") as handle:
        return VoiceReferenceManifest.model_validate(json.load(handle))


def load_prediction_manifest(path: Path) -> VoicePredictionManifest:
    with path.open("r", encoding="utf-8") as handle:
        return VoicePredictionManifest.model_validate(json.load(handle))


def validate_sample_evidence(
    duration_ms: int | None,
    turns: list[EvalInterval],
    candidates: list[EvalCandidate],
) -> None:
    labels = {row.speaker_label for row in turns if row.speaker_label is not None}
    if any(row.speaker_label is None or row.text is None for row in turns):
        raise ValueError("every speaker turn needs a speaker label and transcript text")
    if duration_ms is not None and any(row.end_ms > duration_ms for row in turns):
        raise ValueError("speaker turn exceeds sample duration")
    candidate_ids = [row.candidate_id for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be unique within a sample")
    for candidate in candidates:
        if candidate.speaker_label not in labels:
            raise ValueError("candidate speaker is missing from speaker turns")
        if duration_ms is not None and candidate.source_end_ms > duration_ms:
            raise ValueError("candidate source span exceeds sample duration")
        linked = any(
            turn.speaker_label == candidate.speaker_label
            and min(turn.end_ms, candidate.source_end_ms)
            > max(turn.start_ms, candidate.source_start_ms)
            for turn in turns
        )
        if not linked:
            raise ValueError("candidate source span does not overlap its speaker turn")
