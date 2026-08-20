"""Voice provider contract and deterministic G0 fake provider."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


CandidateType = Literal[
    "stated_fact",
    "stated_preference",
    "commitment",
    "follow_up_action",
]


class VoiceAlternativeResult(BaseModel):
    text: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)


class VoiceSegmentResult(BaseModel):
    segment_index: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    channel: int = Field(default=0, ge=0)
    vad_confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def check_time_range(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("segment end_ms should be after start_ms")
        return self


class VoiceTurnResult(BaseModel):
    turn_index: int = Field(ge=0)
    segment_index: int = Field(ge=0)
    speaker_label: str = Field(pattern=r"^speaker_[0-9]+$")
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    diarization_confidence: float | None = Field(default=None, ge=0, le=1)
    overlap: bool = False
    alternatives: list[VoiceAlternativeResult] = Field(min_length=1)

    @model_validator(mode="after")
    def check_time_range(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("turn end_ms should be after start_ms")
        return self


class VoiceCandidateResult(BaseModel):
    candidate_index: int = Field(ge=0)
    turn_index: int = Field(ge=0)
    candidate_type: CandidateType
    content: str = Field(min_length=1, max_length=4000)
    confidence: float | None = Field(default=None, ge=0, le=1)
    uncertainty: dict[str, object] = Field(default_factory=dict)


class VoiceTranscriptionResult(BaseModel):
    provider_version: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    segments: list[VoiceSegmentResult] = Field(min_length=1)
    turns: list[VoiceTurnResult] = Field(min_length=1)
    candidates: list[VoiceCandidateResult] = Field(min_length=1)


class FakeVoiceProvider:
    name = "fake"
    version = "1"

    def transcribe(
        self,
        audio: bytes,
        *,
        language: str,
        fixture_name: str,
    ) -> VoiceTranscriptionResult:
        if not audio:
            raise ValueError("audio should not be empty")
        if language not in {"zh", "mixed"}:
            raise ValueError("fake voice fixture supports zh or mixed")
        if fixture_name != "mandarin_two_speaker_v1":
            raise ValueError("unknown fake voice fixture")

        return VoiceTranscriptionResult(
            provider_version=self.version,
            model_name="synthetic-fixture",
            model_revision="mandarin_two_speaker_v1",
            segments=[
                VoiceSegmentResult(
                    segment_index=0,
                    start_ms=0,
                    end_ms=4200,
                    channel=0,
                    vad_confidence=0.95,
                ),
                VoiceSegmentResult(
                    segment_index=1,
                    start_ms=4200,
                    end_ms=7600,
                    channel=0,
                    vad_confidence=0.97,
                ),
            ],
            turns=[
                VoiceTurnResult(
                    turn_index=0,
                    segment_index=0,
                    speaker_label="speaker_0",
                    start_ms=0,
                    end_ms=4200,
                    diarization_confidence=0.72,
                    alternatives=[
                        VoiceAlternativeResult(
                            text="我下周二上午有时间，开会最好控制在二十分钟。",
                            confidence=0.86,
                        ),
                        VoiceAlternativeResult(
                            text="我下周四上午有时间，开会最好控制在二十分钟。",
                            confidence=0.31,
                        ),
                    ],
                ),
                VoiceTurnResult(
                    turn_index=1,
                    segment_index=1,
                    speaker_label="speaker_1",
                    start_ms=4200,
                    end_ms=7600,
                    diarization_confidence=0.91,
                    alternatives=[
                        VoiceAlternativeResult(
                            text="好的，我会提前发一份简短议程。",
                            confidence=0.93,
                        )
                    ],
                ),
            ],
            candidates=[
                VoiceCandidateResult(
                    candidate_index=0,
                    turn_index=0,
                    candidate_type="stated_preference",
                    content="偏好周二上午开会，并把会议控制在二十分钟。",
                    confidence=0.71,
                    uncertainty={
                        "asr_alternatives": 2,
                        "speaker_ambiguous": True,
                        "requires_review": True,
                    },
                ),
                VoiceCandidateResult(
                    candidate_index=1,
                    turn_index=1,
                    candidate_type="commitment",
                    content="会提前发送一份简短议程。",
                    confidence=0.88,
                    uncertainty={
                        "asr_alternatives": 1,
                        "speaker_ambiguous": False,
                        "requires_review": True,
                    },
                ),
            ],
        )


def build_voice_provider(provider: str):
    normalized = provider.strip().lower()
    if normalized == "fake":
        return FakeVoiceProvider()
    raise ValueError(f"voice provider is not available: {provider}")
