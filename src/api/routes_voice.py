"""Reviewed Voice G0 API routes."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from src.api.deps import get_current_user
from src.models.database import get_db
from src.models.user import User
from src.models.voice import (
    ApprovedMemoryEvent,
    AudioSegment,
    ExtractedCandidate,
    ReviewDecision,
    SpeakerTurn,
    TranscriptAlternative,
    TranscriptRevision,
    VoiceIngestionJob,
)
from src.voice.service import (
    MAX_AUDIO_BYTES,
    VoiceConflict,
    VoiceNotFound,
    add_transcript_revision,
    cancel_voice_job,
    create_manual_candidate,
    create_voice_job,
    decide_candidate,
    get_voice_job,
    mark_audio_deleted,
)


router = APIRouter(prefix="/api/voice/jobs", tags=["voice"])


class AlternativeResponse(BaseModel):
    id: UUID
    rank: int
    text: str
    confidence: float | None
    provider: str


class RevisionResponse(BaseModel):
    id: UUID
    revision_no: int
    text: str
    raw_text: str
    normalized_text: str
    provenance: dict
    source: str
    reason: str | None
    created_at: datetime


class SpeakerTurnResponse(BaseModel):
    id: UUID
    turn_index: int
    segment_id: UUID
    speaker_label: str
    start_ms: int
    end_ms: int
    diarization_confidence: float | None
    overlap: bool
    needs_review: bool
    version: int
    alternatives: list[AlternativeResponse]
    revisions: list[RevisionResponse]


class AudioSegmentResponse(BaseModel):
    id: UUID
    segment_index: int
    start_ms: int
    end_ms: int
    channel: int
    vad_confidence: float | None


class ReviewDecisionResponse(BaseModel):
    id: UUID
    decision: str
    edited_content: str | None
    created_at: datetime


class ApprovedMemoryEventResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    content: str
    source_turn_ids: list[str]
    source_audio_sha256: str
    source_spans: list[dict]
    provider: str
    model_name: str | None
    model_revision: str | None
    provider_artifacts: dict
    extractor_provenance: dict
    index_status: str
    indexed_at: datetime | None
    error_code: str | None


class CandidateResponse(BaseModel):
    id: UUID
    candidate_index: int
    revision_id: UUID
    candidate_type: str
    content: str
    confidence: float | None
    uncertainty: dict
    extractor_provenance: dict
    source_turn_id: UUID
    speaker_label: str
    source_start_ms: int
    source_end_ms: int
    status: str
    version: int
    decision: ReviewDecisionResponse | None
    approved_memory_event: ApprovedMemoryEventResponse | None


class VoiceJobResponse(BaseModel):
    id: UUID
    person_id: UUID
    status: str
    provider: str
    candidate_extractor: str
    provider_version: str | None
    model_name: str | None
    model_revision: str | None
    provider_artifacts: dict
    fixture_name: str | None
    language: str
    recorded_at: datetime | None
    audio_sha256: str
    audio_size_bytes: int
    audio_mime_type: str
    retention_policy: str
    audio_deleted_at: datetime | None
    error_code: str | None
    trace_id: str
    retry_count: int
    version: int
    segments: list[AudioSegmentResponse]
    turns: list[SpeakerTurnResponse]
    candidates: list[CandidateResponse]
    created_at: datetime
    updated_at: datetime


class TranscriptRevisionCreate(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    reason: str | None = Field(default=None, max_length=255)
    expected_version: int = Field(ge=1)


class CandidateDecisionCreate(BaseModel):
    decision: Literal["approve", "reject"]
    edited_content: str | None = Field(default=None, max_length=4000)
    expected_version: int = Field(ge=1)


class ManualCandidateCreate(BaseModel):
    candidate_type: Literal[
        "stated_fact",
        "stated_preference",
        "commitment",
        "follow_up_action",
    ]
    content: str = Field(min_length=1, max_length=4000)
    source_start_ms: int | None = Field(default=None, ge=0)
    source_end_ms: int | None = Field(default=None, gt=0)
    expected_turn_version: int = Field(ge=1)


class VoiceJobCancelCreate(BaseModel):
    expected_version: int = Field(ge=1)


def build_alternative_response(row: TranscriptAlternative) -> AlternativeResponse:
    return AlternativeResponse(
        id=row.id,
        rank=row.rank,
        text=row.text,
        confidence=row.confidence,
        provider=row.provider,
    )


def build_revision_response(row: TranscriptRevision) -> RevisionResponse:
    return RevisionResponse(
        id=row.id,
        revision_no=row.revision_no,
        text=row.text,
        raw_text=row.raw_text,
        normalized_text=row.normalized_text,
        provenance=dict(row.provenance),
        source=row.source,
        reason=row.reason,
        created_at=row.created_at,
    )


def build_decision_response(
    row: ReviewDecision | None,
) -> ReviewDecisionResponse | None:
    if row is None:
        return None
    return ReviewDecisionResponse(
        id=row.id,
        decision=row.decision,
        edited_content=row.edited_content,
        created_at=row.created_at,
    )


def build_event_response(
    row: ApprovedMemoryEvent | None,
) -> ApprovedMemoryEventResponse | None:
    if row is None:
        return None
    return ApprovedMemoryEventResponse(
        id=row.id,
        conversation_id=row.conversation_id,
        content=row.content,
        source_turn_ids=list(row.source_turn_ids),
        source_audio_sha256=row.source_audio_sha256,
        source_spans=list(row.source_spans),
        provider=row.provider,
        model_name=row.model_name,
        model_revision=row.model_revision,
        provider_artifacts=dict(row.provider_artifacts),
        extractor_provenance=dict(row.extractor_provenance),
        index_status=row.index_status,
        indexed_at=row.indexed_at,
        error_code=row.error_code,
    )


def build_voice_job_response(db: Session, job: VoiceIngestionJob) -> VoiceJobResponse:
    db.refresh(job)
    segments = (
        db.query(AudioSegment)
        .filter(AudioSegment.job_id == job.id)
        .order_by(AudioSegment.segment_index.asc())
        .all()
    )
    turns = (
        db.query(SpeakerTurn)
        .filter(SpeakerTurn.job_id == job.id)
        .order_by(SpeakerTurn.turn_index.asc())
        .all()
    )
    candidates = (
        db.query(ExtractedCandidate)
        .filter(ExtractedCandidate.job_id == job.id)
        .order_by(ExtractedCandidate.candidate_index.asc())
        .all()
    )

    return VoiceJobResponse(
        id=job.id,
        person_id=job.person_id,
        status=job.status,
        provider=job.provider,
        candidate_extractor=job.candidate_extractor,
        provider_version=job.provider_version,
        model_name=job.model_name,
        model_revision=job.model_revision,
        provider_artifacts=dict(job.provider_artifacts),
        fixture_name=job.fixture_name,
        language=job.language,
        recorded_at=job.recorded_at,
        audio_sha256=job.audio_sha256,
        audio_size_bytes=job.audio_size_bytes,
        audio_mime_type=job.audio_mime_type,
        retention_policy=job.retention_policy,
        audio_deleted_at=job.audio_deleted_at,
        error_code=job.error_code,
        trace_id=job.trace_id,
        retry_count=job.retry_count,
        version=job.version,
        segments=[
            AudioSegmentResponse(
                id=row.id,
                segment_index=row.segment_index,
                start_ms=row.start_ms,
                end_ms=row.end_ms,
                channel=row.channel,
                vad_confidence=row.vad_confidence,
            )
            for row in segments
        ],
        turns=[
            SpeakerTurnResponse(
                id=row.id,
                turn_index=row.turn_index,
                segment_id=row.segment_id,
                speaker_label=row.speaker_label,
                start_ms=row.start_ms,
                end_ms=row.end_ms,
                diarization_confidence=row.diarization_confidence,
                overlap=row.overlap,
                needs_review=row.needs_review,
                version=row.version,
                alternatives=[
                    build_alternative_response(item) for item in row.alternatives
                ],
                revisions=[build_revision_response(item) for item in row.revisions],
            )
            for row in turns
        ],
        candidates=[
            CandidateResponse(
                id=row.id,
                candidate_index=row.candidate_index,
                revision_id=row.revision_id,
                candidate_type=row.candidate_type,
                content=row.content,
                confidence=row.confidence,
                uncertainty=dict(row.uncertainty),
                extractor_provenance=dict(row.extractor_provenance),
                source_turn_id=row.source_turn_id,
                speaker_label=row.speaker_label,
                source_start_ms=row.source_start_ms,
                source_end_ms=row.source_end_ms,
                status=row.status,
                version=row.version,
                decision=build_decision_response(row.decision),
                approved_memory_event=build_event_response(row.approved_event),
            )
            for row in candidates
        ],
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def raise_voice_http_error(db: Session, exc: Exception) -> None:
    db.rollback()
    if isinstance(exc, VoiceNotFound):
        raise HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, VoiceConflict):
        raise HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, StaleDataError):
        raise HTTPException(
            status_code=409,
            detail="voice resource changed; refresh and retry",
        )
    if isinstance(exc, IntegrityError):
        raise HTTPException(status_code=409, detail="voice write conflict")
    raise HTTPException(status_code=400, detail=str(exc))


@router.post("", response_model=VoiceJobResponse, status_code=status.HTTP_201_CREATED)
async def upload_voice_job(
    person_id: UUID = Form(...),
    language: str = Form("zh"),
    recorded_at: datetime | None = Form(None),
    provider: str = Form("fake"),
    candidate_extractor: str = Form("auto"),
    fixture_name: str = Form("mandarin_two_speaker_v1"),
    audio: UploadFile = File(...),
    idempotency_key: str = Header(
        alias="Idempotency-Key",
        min_length=1,
        max_length=255,
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> VoiceJobResponse:
    job = None
    created_by_request = False
    try:
        audio_bytes = await audio.read(MAX_AUDIO_BYTES + 1)
        job, created_by_request = create_voice_job(
            db,
            user_id=current_user.id,
            person_id=person_id,
            audio=audio_bytes,
            mime_type=audio.content_type or "application/octet-stream",
            language=language,
            recorded_at=recorded_at,
            provider_name=provider,
            fixture_name=fixture_name,
            idempotency_key=idempotency_key,
            extractor_name=candidate_extractor,
        )
    except (ValueError, IntegrityError, StaleDataError) as exc:
        raise_voice_http_error(db, exc)
    finally:
        await audio.close()
        if job is not None and created_by_request:
            mark_audio_deleted(db, job)
    return build_voice_job_response(db, job)


@router.get("/{job_id}", response_model=VoiceJobResponse)
def get_voice_job_route(
    job_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> VoiceJobResponse:
    job = get_voice_job(db, current_user.id, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="voice job not found")
    return build_voice_job_response(db, job)


@router.post("/{job_id}/cancel", response_model=VoiceJobResponse)
def cancel_voice_job_route(
    job_id: UUID,
    payload: VoiceJobCancelCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> VoiceJobResponse:
    try:
        job = cancel_voice_job(
            db,
            user_id=current_user.id,
            job_id=job_id,
            expected_version=payload.expected_version,
        )
    except (ValueError, IntegrityError, StaleDataError) as exc:
        raise_voice_http_error(db, exc)
    return build_voice_job_response(db, job)


@router.post(
    "/{job_id}/turns/{turn_id}/revisions",
    response_model=VoiceJobResponse,
)
def revise_transcript(
    job_id: UUID,
    turn_id: UUID,
    payload: TranscriptRevisionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> VoiceJobResponse:
    try:
        add_transcript_revision(
            db,
            user_id=current_user.id,
            job_id=job_id,
            turn_id=turn_id,
            text=payload.text,
            reason=payload.reason,
            expected_version=payload.expected_version,
        )
        job = get_voice_job(db, current_user.id, job_id)
    except (ValueError, IntegrityError, StaleDataError) as exc:
        raise_voice_http_error(db, exc)
    return build_voice_job_response(db, job)


@router.post(
    "/{job_id}/turns/{turn_id}/candidates",
    response_model=VoiceJobResponse,
    status_code=status.HTTP_201_CREATED,
)
def add_manual_candidate(
    job_id: UUID,
    turn_id: UUID,
    payload: ManualCandidateCreate,
    idempotency_key: str = Header(
        alias="Idempotency-Key",
        min_length=1,
        max_length=255,
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> VoiceJobResponse:
    try:
        create_manual_candidate(
            db,
            user_id=current_user.id,
            job_id=job_id,
            turn_id=turn_id,
            candidate_type=payload.candidate_type,
            content=payload.content,
            source_start_ms=payload.source_start_ms,
            source_end_ms=payload.source_end_ms,
            expected_turn_version=payload.expected_turn_version,
            idempotency_key=idempotency_key,
        )
        job = get_voice_job(db, current_user.id, job_id)
    except (ValueError, IntegrityError, StaleDataError) as exc:
        raise_voice_http_error(db, exc)
    return build_voice_job_response(db, job)


@router.post(
    "/{job_id}/candidates/{candidate_id}/decision",
    response_model=VoiceJobResponse,
)
def review_candidate(
    job_id: UUID,
    candidate_id: UUID,
    payload: CandidateDecisionCreate,
    idempotency_key: str = Header(
        alias="Idempotency-Key",
        min_length=1,
        max_length=255,
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> VoiceJobResponse:
    try:
        decide_candidate(
            db,
            user_id=current_user.id,
            job_id=job_id,
            candidate_id=candidate_id,
            decision=payload.decision,
            edited_content=payload.edited_content,
            expected_version=payload.expected_version,
            idempotency_key=idempotency_key,
        )
        job = get_voice_job(db, current_user.id, job_id)
    except (ValueError, IntegrityError, StaleDataError) as exc:
        raise_voice_http_error(db, exc)
    return build_voice_job_response(db, job)
