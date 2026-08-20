"""Typed candidate extraction after Voice transcription."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


CandidateType = Literal[
    "stated_fact",
    "stated_preference",
    "commitment",
    "follow_up_action",
]


class CandidateTurnInput(BaseModel):
    turn_index: int = Field(ge=0)
    speaker_label: str = Field(pattern=r"^speaker_[0-9]+$")
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: str = Field(min_length=1)
    alternatives: list[str] = Field(min_length=1)
    diarization_confidence: float | None = Field(default=None, ge=0, le=1)
    overlap: bool = False

    @model_validator(mode="after")
    def check_time_range(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("turn end_ms should be after start_ms")
        return self


class VoiceCandidateResult(BaseModel):
    turn_index: int = Field(ge=0)
    candidate_type: CandidateType
    content: str = Field(min_length=1, max_length=4000)
    source_start_ms: int = Field(ge=0)
    source_end_ms: int = Field(gt=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    uncertainty: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_source_range(self):
        if self.source_end_ms <= self.source_start_ms:
            raise ValueError("candidate source_end_ms should be after source_start_ms")
        return self


class CandidateExtractionResult(BaseModel):
    extractor_name: str = Field(min_length=1)
    extractor_version: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    candidates: list[VoiceCandidateResult]


class FakeCandidateExtractor:
    name = "fake"
    version = "1"

    def extract(
        self,
        turns: list[CandidateTurnInput],
        *,
        language: str,
        fixture_name: str,
    ) -> CandidateExtractionResult:
        if language not in {"zh", "mixed"}:
            raise ValueError("fake candidate fixture supports zh or mixed")
        if fixture_name != "mandarin_two_speaker_v1":
            raise ValueError("unknown fake candidate fixture")
        by_index = {row.turn_index: row for row in turns}
        unknown = set(by_index) - {0, 1}
        if unknown:
            raise ValueError(f"fake candidate fixture has unknown turns: {sorted(unknown)}")
        candidates = []
        first = by_index.get(0)
        if first is not None:
            first_content = (
                "偏好周二上午开会，并把会议控制在二十分钟。"
                if first.text == "我下周二上午有时间，开会最好控制在二十分钟。"
                else first.text
            )
            candidates.append(
                VoiceCandidateResult(
                    turn_index=0,
                    candidate_type="stated_preference",
                    content=first_content,
                    source_start_ms=first.start_ms,
                    source_end_ms=first.end_ms,
                    confidence=0.71 if first_content != first.text else None,
                    uncertainty={
                        "asr_alternatives": len(first.alternatives),
                        "speaker_ambiguous": True,
                        "requires_review": True,
                    },
                )
            )
        second = by_index.get(1)
        if second is not None:
            second_content = (
                "会提前发送一份简短议程。"
                if second.text == "好的，我会提前发一份简短议程。"
                else second.text
            )
            candidates.append(
                VoiceCandidateResult(
                    turn_index=1,
                    candidate_type="commitment",
                    content=second_content,
                    source_start_ms=second.start_ms,
                    source_end_ms=second.end_ms,
                    confidence=0.88 if second_content != second.text else None,
                    uncertainty={
                        "asr_alternatives": len(second.alternatives),
                        "speaker_ambiguous": False,
                        "requires_review": True,
                    },
                )
            )
        return CandidateExtractionResult(
            extractor_name=self.name,
            extractor_version=self.version,
            model_name="synthetic-candidate-fixture",
            model_revision=fixture_name,
            candidates=candidates,
        )


class ManualCandidateExtractor:
    name = "manual"
    version = "1"

    def extract(
        self,
        turns: list[CandidateTurnInput],
        *,
        language: str,
        fixture_name: str,
    ) -> CandidateExtractionResult:
        del turns, language, fixture_name
        return CandidateExtractionResult(
            extractor_name=self.name,
            extractor_version=self.version,
            model_name="human-review",
            model_revision="1",
            candidates=[],
        )


def build_candidate_extractor(extractor: str):
    normalized = extractor.strip().lower()
    if normalized == "fake":
        return FakeCandidateExtractor()
    if normalized == "manual":
        return ManualCandidateExtractor()
    raise ValueError(f"voice candidate extractor is not available: {extractor}")
