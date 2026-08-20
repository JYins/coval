"""Persistence and review services for the Voice G0 pipeline."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from src.models.conversation import Conversation
from src.models.person import Person
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
from src.rag.indexing import save_chunks_for_conversation
from src.rag.retriever import load_default_config
from src.voice.extractors import (
    CandidateExtractionResult,
    CandidateTurnInput,
    build_candidate_extractor,
)
from src.voice.providers import VoiceTranscriptionResult, build_voice_provider


MAX_AUDIO_BYTES = 25 * 1024 * 1024
INDEX_LEASE_MINUTES = 10
ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp4",
    "application/octet-stream",
}


class VoiceNotFound(ValueError):
    pass


class VoiceConflict(ValueError):
    pass


def build_fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_user_person(db: Session, user_id: UUID, person_id: UUID) -> Person | None:
    return (
        db.query(Person)
        .filter(Person.id == person_id, Person.user_id == user_id)
        .first()
    )


def get_voice_job(
    db: Session,
    user_id: UUID,
    job_id: UUID,
) -> VoiceIngestionJob | None:
    return (
        db.query(VoiceIngestionJob)
        .filter(VoiceIngestionJob.id == job_id, VoiceIngestionJob.user_id == user_id)
        .first()
    )


def get_voice_job_for_update(
    db: Session,
    user_id: UUID,
    job_id: UUID,
) -> VoiceIngestionJob | None:
    return (
        db.query(VoiceIngestionJob)
        .filter(VoiceIngestionJob.id == job_id, VoiceIngestionJob.user_id == user_id)
        .populate_existing()
        .with_for_update()
        .first()
    )


def validate_voice_request(
    *,
    audio: bytes,
    mime_type: str,
    language: str,
    idempotency_key: str,
) -> None:
    if not audio:
        raise ValueError("audio should not be empty")
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError("audio should be at most 25 MB")
    if mime_type not in ALLOWED_AUDIO_TYPES:
        raise ValueError("unsupported audio content type")
    if language not in {"zh", "mixed"}:
        raise ValueError("voice language should be zh or mixed")
    if not idempotency_key:
        raise ValueError("Idempotency-Key is required")
    if len(idempotency_key) > 255:
        raise ValueError("Idempotency-Key should be at most 255 characters")


def create_voice_job(
    db: Session,
    *,
    user_id: UUID,
    person_id: UUID,
    audio: bytes,
    mime_type: str,
    language: str,
    recorded_at: datetime | None,
    provider_name: str,
    fixture_name: str,
    idempotency_key: str,
    extractor_name: str = "auto",
) -> tuple[VoiceIngestionJob, bool]:
    key = idempotency_key.strip()
    normalized_provider = provider_name.strip().lower()
    if not normalized_provider:
        raise ValueError("voice provider is required")
    resolved_extractor = extractor_name.strip().lower()
    if resolved_extractor == "auto":
        resolved_extractor = "fake" if normalized_provider == "fake" else "manual"
    extractor = build_candidate_extractor(resolved_extractor)
    normalized_mime = mime_type.strip().lower() or "application/octet-stream"
    normalized_language = language.strip().lower()
    validate_voice_request(
        audio=audio,
        mime_type=normalized_mime,
        language=normalized_language,
        idempotency_key=key,
    )
    person = get_user_person(db, user_id, person_id)
    if person is None:
        raise VoiceNotFound("person not found")
    provider = build_voice_provider(normalized_provider)

    audio_sha256 = hashlib.sha256(audio).hexdigest()
    request_payload = {
        "person_id": str(person_id),
        "audio_sha256": audio_sha256,
        "mime_type": normalized_mime,
        "language": normalized_language,
        "recorded_at": recorded_at.isoformat() if recorded_at else None,
        "provider": normalized_provider,
        "provider_identity": provider.request_identity,
        "candidate_extractor": resolved_extractor,
        "fixture_name": fixture_name,
    }
    fingerprint = build_fingerprint(request_payload)
    existing = (
        db.query(VoiceIngestionJob)
        .filter(
            VoiceIngestionJob.user_id == user_id,
            VoiceIngestionJob.idempotency_key == key,
        )
        .first()
    )
    if existing is not None:
        check_job_fingerprint(existing, fingerprint)
        return existing, False

    job = VoiceIngestionJob(
        user_id=user_id,
        person_id=person_id,
        status="RECEIVED",
        provider=normalized_provider,
        candidate_extractor=resolved_extractor,
        fixture_name=fixture_name,
        language=normalized_language,
        recorded_at=recorded_at,
        audio_sha256=audio_sha256,
        audio_size_bytes=len(audio),
        audio_mime_type=normalized_mime,
        retention_policy="delete_after_processing",
        idempotency_key=key,
        request_fingerprint=fingerprint,
        trace_id=str(uuid.uuid4()),
    )
    db.add(job)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = (
            db.query(VoiceIngestionJob)
            .filter(
                VoiceIngestionJob.user_id == user_id,
                VoiceIngestionJob.idempotency_key == key,
            )
            .one()
        )
        check_job_fingerprint(existing, fingerprint)
        return existing, False
    db.refresh(job)

    job.status = "TRANSCRIBING"
    db.add(job)
    db.commit()
    db.refresh(job)
    try:
        result = provider.transcribe(
            audio,
            language=normalized_language,
            fixture_name=fixture_name,
            mime_type=normalized_mime,
        )
        job.provider_version = result.provider_version
        job.model_name = result.model_name
        job.model_revision = result.model_revision
        job.provider_artifacts = dict(result.artifact_provenance)
        revisions = persist_transcription(db, job, result)
        extraction = extractor.extract(
            build_candidate_inputs(result),
            language=normalized_language,
            fixture_name=fixture_name,
        )
        persist_candidates(db, job, revisions, extraction)
        job.status = "REVIEW_READY"
    except StaleDataError:
        db.rollback()
        current = get_voice_job(db, user_id, job.id)
        if current is None or current.status != "CANCELED":
            raise
        return current, True
    except Exception as exc:
        db.rollback()
        job = get_voice_job(db, user_id, job.id)
        job.status = "FAILED"
        job.error_code = type(exc).__name__[:100]
    # the API transport closes its upload before marking audio retention complete
    db.add(job)
    try:
        db.commit()
    except StaleDataError:
        db.rollback()
        current = get_voice_job(db, user_id, job.id)
        if current is None or current.status != "CANCELED":
            raise
        return current, True
    db.refresh(job)
    return job, True


def check_job_fingerprint(job: VoiceIngestionJob, fingerprint: str) -> None:
    if job.request_fingerprint != fingerprint:
        raise VoiceConflict("Idempotency-Key was already used with another voice upload")


def mark_audio_deleted(db: Session, job: VoiceIngestionJob) -> VoiceIngestionJob:
    if job.audio_deleted_at is not None:
        return job
    job.audio_deleted_at = datetime.now(timezone.utc)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def persist_transcription(
    db: Session,
    job: VoiceIngestionJob,
    result: VoiceTranscriptionResult,
) -> dict[int, TranscriptRevision]:
    segments: dict[int, AudioSegment] = {}
    for item in result.segments:
        segment = AudioSegment(
            job_id=job.id,
            segment_index=item.segment_index,
            start_ms=item.start_ms,
            end_ms=item.end_ms,
            channel=item.channel,
            vad_confidence=item.vad_confidence,
        )
        db.add(segment)
        db.flush()
        segments[item.segment_index] = segment

    revisions: dict[int, TranscriptRevision] = {}
    for item in result.turns:
        segment = segments.get(item.segment_index)
        if segment is None:
            raise ValueError("voice turn references an unknown segment")
        turn = SpeakerTurn(
            job_id=job.id,
            segment_id=segment.id,
            turn_index=item.turn_index,
            speaker_label=item.speaker_label,
            start_ms=item.start_ms,
            end_ms=item.end_ms,
            diarization_confidence=item.diarization_confidence,
            overlap=item.overlap,
            needs_review=True,
        )
        db.add(turn)
        db.flush()

        alternatives = []
        for rank, alternative in enumerate(item.alternatives, start=1):
            row = TranscriptAlternative(
                turn_id=turn.id,
                rank=rank,
                text=alternative.text,
                confidence=alternative.confidence,
                provider=job.provider,
            )
            db.add(row)
            alternatives.append(row)
        db.flush()

        revision = TranscriptRevision(
            turn_id=turn.id,
            revision_no=1,
            text=alternatives[0].text,
            raw_text=alternatives[0].text,
            normalized_text=alternatives[0].text,
            provenance={
                "provider": job.provider,
                "provider_version": job.provider_version,
                "model_name": job.model_name,
                "model_revision": job.model_revision,
                "alternative_rank": 1,
            },
            source="PROVIDER",
            base_alternative_id=alternatives[0].id,
        )
        db.add(revision)
        db.flush()
        revisions[item.turn_index] = revision

    db.flush()
    return revisions


def build_candidate_inputs(
    result: VoiceTranscriptionResult,
) -> list[CandidateTurnInput]:
    return [
        CandidateTurnInput(
            turn_index=item.turn_index,
            speaker_label=item.speaker_label,
            start_ms=item.start_ms,
            end_ms=item.end_ms,
            text=item.alternatives[0].text,
            alternatives=[row.text for row in item.alternatives],
            diarization_confidence=item.diarization_confidence,
            overlap=item.overlap,
        )
        for item in result.turns
    ]


def persist_candidates(
    db: Session,
    job: VoiceIngestionJob,
    revisions: dict[int, TranscriptRevision],
    result: CandidateExtractionResult,
    *,
    next_index: int = 0,
) -> None:
    provenance = {
        "extractor_name": result.extractor_name,
        "extractor_version": result.extractor_version,
        "model_name": result.model_name,
        "model_revision": result.model_revision,
    }
    for offset, item in enumerate(result.candidates):
        revision = revisions.get(item.turn_index)
        if revision is None:
            raise ValueError("voice candidate references an unknown turn")
        turn = revision.turn
        if (
            item.source_start_ms < turn.start_ms
            or item.source_end_ms > turn.end_ms
        ):
            raise ValueError("voice candidate source span exceeds its turn")
        db.add(
            ExtractedCandidate(
                job_id=job.id,
                revision_id=revision.id,
                candidate_index=next_index + offset,
                candidate_type=item.candidate_type,
                content=item.content,
                confidence=item.confidence,
                uncertainty=dict(item.uncertainty),
                extractor_provenance=dict(provenance),
                source_turn_id=revision.turn_id,
                speaker_label=turn.speaker_label,
                source_start_ms=item.source_start_ms,
                source_end_ms=item.source_end_ms,
                status="PENDING",
            )
        )
    db.flush()


def add_transcript_revision(
    db: Session,
    *,
    user_id: UUID,
    job_id: UUID,
    turn_id: UUID,
    text: str,
    reason: str | None,
    expected_version: int,
) -> TranscriptRevision:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("revision text is required")
    if len(cleaned) > 10000:
        raise ValueError("revision text should be at most 10000 characters")
    if reason and len(reason) > 255:
        raise ValueError("revision reason should be at most 255 characters")

    job = get_voice_job_for_update(db, user_id, job_id)
    if job is None:
        raise VoiceNotFound("voice job not found")
    turn = (
        db.query(SpeakerTurn)
        .filter(SpeakerTurn.id == turn_id, SpeakerTurn.job_id == job.id)
        .with_for_update()
        .first()
    )
    if turn is None:
        raise VoiceNotFound("speaker turn not found")
    if job.status in {"COMPLETED", "COMPLETED_WITH_ERRORS", "FAILED", "CANCELED"}:
        raise VoiceConflict(f"{job.status.lower()} voice job cannot be revised")
    if turn.version != expected_version:
        raise VoiceConflict(
            f"speaker turn version changed: expected {expected_version}, got {turn.version}"
        )

    reviewed_candidate = (
        db.query(ExtractedCandidate)
        .join(TranscriptRevision, ExtractedCandidate.revision_id == TranscriptRevision.id)
        .filter(
            ExtractedCandidate.job_id == job.id,
            ExtractedCandidate.status.in_({"APPROVED", "REJECTED"}),
            TranscriptRevision.turn_id == turn.id,
        )
        .first()
    )
    if reviewed_candidate is not None:
        raise VoiceConflict("turn already has a reviewed candidate")

    latest_revision = (
        db.query(TranscriptRevision)
        .filter(TranscriptRevision.turn_id == turn.id)
        .order_by(TranscriptRevision.revision_no.desc())
        .first()
    )
    revision = TranscriptRevision(
        turn_id=turn.id,
        revision_no=latest_revision.revision_no + 1,
        text=cleaned,
        raw_text=latest_revision.raw_text,
        normalized_text=cleaned,
        provenance={
            "source": "human_revision",
            "previous_revision_id": str(latest_revision.id),
        },
        source="USER",
        editor_user_id=user_id,
        reason=reason.strip() if reason else None,
    )
    db.add(revision)
    db.flush()

    pending = (
        db.query(ExtractedCandidate)
        .join(TranscriptRevision, ExtractedCandidate.revision_id == TranscriptRevision.id)
        .filter(
            ExtractedCandidate.job_id == job.id,
            ExtractedCandidate.status == "PENDING",
            TranscriptRevision.turn_id == turn.id,
        )
        .order_by(ExtractedCandidate.candidate_index.asc())
        .all()
    )
    next_index = (
        db.query(ExtractedCandidate)
        .filter(ExtractedCandidate.job_id == job.id)
        .count()
    )
    for old in pending:
        old.status = "STALE"
        db.add(old)
    extractor = build_candidate_extractor(job.candidate_extractor)
    extraction = extractor.extract(
        [
            CandidateTurnInput(
                turn_index=turn.turn_index,
                speaker_label=turn.speaker_label,
                start_ms=turn.start_ms,
                end_ms=turn.end_ms,
                text=cleaned,
                alternatives=[cleaned],
                diarization_confidence=turn.diarization_confidence,
                overlap=turn.overlap,
            )
        ],
        language=job.language,
        fixture_name=job.fixture_name or "",
    )
    persist_candidates(
        db,
        job,
        {turn.turn_index: revision},
        extraction,
        next_index=next_index,
    )

    turn.needs_review = True
    turn.revision_count += 1
    db.add(turn)
    db.commit()
    db.refresh(revision)
    return revision


def create_manual_candidate(
    db: Session,
    *,
    user_id: UUID,
    job_id: UUID,
    turn_id: UUID,
    candidate_type: str,
    content: str,
    source_start_ms: int | None,
    source_end_ms: int | None,
    expected_turn_version: int,
    idempotency_key: str,
) -> ExtractedCandidate:
    allowed_types = {
        "stated_fact",
        "stated_preference",
        "commitment",
        "follow_up_action",
    }
    normalized_type = candidate_type.strip().lower()
    cleaned = content.strip()
    key = idempotency_key.strip()
    if normalized_type not in allowed_types:
        raise ValueError("unsupported voice candidate type")
    if not cleaned or len(cleaned) > 4000:
        raise ValueError("candidate content should contain 1 to 4000 characters")
    if not key or len(key) > 255:
        raise ValueError("Idempotency-Key should contain 1 to 255 characters")

    job = get_voice_job_for_update(db, user_id, job_id)
    if job is None:
        raise VoiceNotFound("voice job not found")
    existing = (
        db.query(ExtractedCandidate)
        .filter(
            ExtractedCandidate.job_id == job.id,
            ExtractedCandidate.idempotency_key == key,
        )
        .first()
    )
    turn = (
        db.query(SpeakerTurn)
        .filter(SpeakerTurn.id == turn_id, SpeakerTurn.job_id == job.id)
        .with_for_update()
        .first()
    )
    if turn is None:
        raise VoiceNotFound("speaker turn not found")
    start_ms = turn.start_ms if source_start_ms is None else source_start_ms
    end_ms = turn.end_ms if source_end_ms is None else source_end_ms
    if start_ms < turn.start_ms or end_ms > turn.end_ms or end_ms <= start_ms:
        raise ValueError("candidate source span should stay inside its speaker turn")
    payload = {
        "turn_id": str(turn_id),
        "candidate_type": normalized_type,
        "content": cleaned,
        "source_start_ms": start_ms,
        "source_end_ms": end_ms,
        "expected_turn_version": expected_turn_version,
    }
    fingerprint = build_fingerprint(payload)
    if existing is not None:
        if existing.request_fingerprint != fingerprint:
            raise VoiceConflict("Idempotency-Key was used for another candidate")
        return existing
    if job.candidate_extractor != "manual":
        raise VoiceConflict("manual candidates require the manual extractor")
    if job.status != "REVIEW_READY":
        raise VoiceConflict(f"{job.status.lower()} voice job cannot add candidates")
    if turn.version != expected_turn_version:
        raise VoiceConflict(
            f"speaker turn version changed: expected {expected_turn_version}, "
            f"got {turn.version}"
        )
    revision = (
        db.query(TranscriptRevision)
        .filter(TranscriptRevision.turn_id == turn.id)
        .order_by(TranscriptRevision.revision_no.desc())
        .first()
    )
    next_index = (
        db.query(ExtractedCandidate)
        .filter(ExtractedCandidate.job_id == job.id)
        .count()
    )
    row = ExtractedCandidate(
        job_id=job.id,
        revision_id=revision.id,
        candidate_index=next_index,
        candidate_type=normalized_type,
        content=cleaned,
        confidence=None,
        uncertainty={"source": "human_candidate", "requires_review": True},
        extractor_provenance={
            "extractor_name": "manual",
            "extractor_version": "1",
            "model_name": "human-review",
            "model_revision": "1",
            "user_id": str(user_id),
        },
        idempotency_key=key,
        request_fingerprint=fingerprint,
        source_turn_id=turn.id,
        speaker_label=turn.speaker_label,
        source_start_ms=start_ms,
        source_end_ms=end_ms,
        status="PENDING",
    )
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = (
            db.query(ExtractedCandidate)
            .filter(
                ExtractedCandidate.job_id == job.id,
                ExtractedCandidate.idempotency_key == key,
            )
            .one_or_none()
        )
        if existing is None or existing.request_fingerprint != fingerprint:
            raise
        return existing
    db.refresh(row)
    return row


def decide_candidate(
    db: Session,
    *,
    user_id: UUID,
    job_id: UUID,
    candidate_id: UUID,
    decision: str,
    edited_content: str | None,
    expected_version: int,
    idempotency_key: str,
    config: dict[str, Any] | None = None,
) -> ReviewDecision:
    normalized = decision.strip().lower()
    key = idempotency_key.strip()
    edited = edited_content.strip() if edited_content is not None else None
    if normalized not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject")
    if not key or len(key) > 255:
        raise ValueError("Idempotency-Key should contain 1 to 255 characters")
    if edited_content is not None and not edited:
        raise ValueError("edited content should not be blank")
    if edited and len(edited) > 4000:
        raise ValueError("edited content should be at most 4000 characters")
    if normalized == "reject" and edited:
        raise ValueError("edited content is only accepted with approve")

    job = get_voice_job(db, user_id, job_id)
    if job is None:
        raise VoiceNotFound("voice job not found")
    if job.status in {"FAILED", "CANCELED"}:
        raise VoiceConflict(f"{job.status.lower()} voice job cannot be reviewed")
    candidate = (
        db.query(ExtractedCandidate)
        .filter(
            ExtractedCandidate.id == candidate_id,
            ExtractedCandidate.job_id == job.id,
        )
        .first()
    )
    if candidate is None:
        raise VoiceNotFound("candidate not found")

    payload = {
        "decision": normalized,
        "edited_content": edited,
        "expected_version": expected_version,
    }
    fingerprint = build_fingerprint(payload)
    existing = (
        db.query(ReviewDecision)
        .filter(ReviewDecision.candidate_id == candidate.id)
        .first()
    )
    if existing is not None:
        if (
            existing.idempotency_key != key
            or existing.request_fingerprint != fingerprint
        ):
            raise VoiceConflict("candidate already has another review decision")
        retry_memory_index(db, job, candidate, config=config)
        update_voice_job_status(db, job)
        return existing

    if candidate.status != "PENDING":
        raise VoiceConflict(f"candidate is {candidate.status.lower()}")
    if candidate.version != expected_version:
        raise VoiceConflict(
            f"candidate version changed: expected {expected_version}, got {candidate.version}"
        )

    row = ReviewDecision(
        candidate_id=candidate.id,
        user_id=user_id,
        decision=normalized,
        edited_content=edited,
        idempotency_key=key,
        request_fingerprint=fingerprint,
    )
    db.add(row)
    candidate.status = "APPROVED" if normalized == "approve" else "REJECTED"
    db.add(candidate)

    if normalized == "approve":
        approved_content = edited or candidate.content
        conversation_data = {
            "person_id": job.person_id,
            "source_type": "voice",
            "raw_content": approved_content,
            "language": job.language,
        }
        if job.recorded_at is not None:
            conversation_data["conversation_date"] = job.recorded_at
        conversation = Conversation(**conversation_data)
        db.add(conversation)
        db.flush()
        db.add(
            ApprovedMemoryEvent(
                user_id=user_id,
                person_id=job.person_id,
                job_id=job.id,
                candidate_id=candidate.id,
                review_decision_id=row.id,
                conversation_id=conversation.id,
                content=approved_content,
                source_turn_ids=[str(candidate.revision.turn_id)],
                source_audio_sha256=job.audio_sha256,
                source_spans=[
                    {
                        "turn_id": str(candidate.source_turn_id),
                        "speaker_label": candidate.speaker_label,
                        "start_ms": candidate.source_start_ms,
                        "end_ms": candidate.source_end_ms,
                    }
                ],
                provider=job.provider,
                model_name=job.model_name,
                model_revision=job.model_revision,
                provider_artifacts=dict(job.provider_artifacts),
                extractor_provenance=dict(candidate.extractor_provenance),
                index_status="PENDING",
            )
        )

    try:
        db.commit()
    except (IntegrityError, StaleDataError):
        db.rollback()
        existing = (
            db.query(ReviewDecision)
            .filter(ReviewDecision.candidate_id == candidate.id)
            .first()
        )
        if existing is None:
            raise
        if (
            existing.idempotency_key != key
            or existing.request_fingerprint != fingerprint
        ):
            raise VoiceConflict("candidate already has another review decision")
        job = get_voice_job(db, user_id, job_id)
        candidate = (
            db.query(ExtractedCandidate)
            .filter(ExtractedCandidate.id == candidate_id)
            .one()
        )
        retry_memory_index(db, job, candidate, config=config)
        update_voice_job_status(db, job)
        return existing
    db.refresh(row)

    if normalized == "approve":
        retry_memory_index(db, job, candidate, config=config)
    update_voice_job_status(db, job)
    return row


def retry_memory_index(
    db: Session,
    job: VoiceIngestionJob,
    candidate: ExtractedCandidate,
    *,
    config: dict[str, Any] | None,
) -> ApprovedMemoryEvent | None:
    event = (
        db.query(ApprovedMemoryEvent)
        .filter(ApprovedMemoryEvent.candidate_id == candidate.id)
        .first()
    )
    if event is None or event.index_status == "INDEXED":
        return event
    now = datetime.now(timezone.utc)
    if (
        event.index_status == "INDEXING"
        and event.index_lease_expires_at is not None
        and normalize_utc(event.index_lease_expires_at) > now
    ):
        return event

    if event.index_status == "INDEX_FAILED":
        job.retry_count += 1
        db.add(job)

    event.index_status = "INDEXING"
    runner_id = str(uuid.uuid4())
    event.index_runner_id = runner_id
    event.index_lease_expires_at = now + timedelta(minutes=INDEX_LEASE_MINUTES)
    db.add(event)
    db.commit()
    db.refresh(event)
    owned_event = (
        db.query(ApprovedMemoryEvent)
        .filter(
            ApprovedMemoryEvent.id == event.id,
            ApprovedMemoryEvent.index_status == "INDEXING",
            ApprovedMemoryEvent.index_runner_id == runner_id,
        )
        .with_for_update()
        .one_or_none()
    )
    if owned_event is None:
        db.rollback()
        raise VoiceConflict("voice memory index lease was lost")
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == event.conversation_id)
        .one()
    )
    person = get_user_person(db, job.user_id, job.person_id)
    if person is None:
        raise VoiceNotFound("person not found")

    try:
        chunks = save_chunks_for_conversation(
            db,
            conversation,
            person.name,
            config=dict(config or load_default_config()),
            user_id=job.user_id,
            person_id=job.person_id,
            commit=False,
        )
        if not chunks:
            raise RuntimeError("approved memory produced no chunks")
    except Exception:
        db.rollback()
        return finish_memory_index(
            db,
            event_id=event.id,
            runner_id=runner_id,
            index_status="INDEX_FAILED",
            error_code="VOICE_MEMORY_INDEX_ERROR",
        )

    return finish_memory_index(
        db,
        event_id=event.id,
        runner_id=runner_id,
        index_status="INDEXED",
        error_code=None,
    )


def finish_memory_index(
    db: Session,
    *,
    event_id: UUID,
    runner_id: str,
    index_status: str,
    error_code: str | None,
) -> ApprovedMemoryEvent:
    values = {
        "index_status": index_status,
        "index_runner_id": None,
        "index_lease_expires_at": None,
        "indexed_at": (
            datetime.now(timezone.utc) if index_status == "INDEXED" else None
        ),
        "error_code": error_code,
        "version": ApprovedMemoryEvent.version + 1,
    }
    updated = (
        db.query(ApprovedMemoryEvent)
        .filter(
            ApprovedMemoryEvent.id == event_id,
            ApprovedMemoryEvent.index_status == "INDEXING",
            ApprovedMemoryEvent.index_runner_id == runner_id,
        )
        .update(values, synchronize_session=False)
    )
    if updated != 1:
        db.rollback()
        raise VoiceConflict("voice memory index lease was lost")
    db.commit()
    return (
        db.query(ApprovedMemoryEvent)
        .filter(ApprovedMemoryEvent.id == event_id)
        .one()
    )


def normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def cancel_voice_job(
    db: Session,
    *,
    user_id: UUID,
    job_id: UUID,
    expected_version: int,
) -> VoiceIngestionJob:
    job = get_voice_job_for_update(db, user_id, job_id)
    if job is None:
        raise VoiceNotFound("voice job not found")
    if job.status == "CANCELED":
        return job
    if job.status in {"COMPLETED", "COMPLETED_WITH_ERRORS"}:
        raise VoiceConflict("completed voice job cannot be canceled")
    if job.version != expected_version:
        raise VoiceConflict(
            f"voice job version changed: expected {expected_version}, got {job.version}"
        )
    if db.query(ReviewDecision).join(ExtractedCandidate).filter(
        ExtractedCandidate.job_id == job.id
    ).first():
        raise VoiceConflict("voice job with review decisions cannot be canceled")

    for candidate in db.query(ExtractedCandidate).filter(
        ExtractedCandidate.job_id == job.id,
        ExtractedCandidate.status == "PENDING",
    ):
        candidate.status = "STALE"
        db.add(candidate)
    job.status = "CANCELED"
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def update_voice_job_status(
    db: Session,
    job: VoiceIngestionJob,
) -> VoiceIngestionJob:
    locked_job = get_voice_job_for_update(db, job.user_id, job.id)
    if locked_job is None:
        raise VoiceNotFound("voice job not found")
    pending_count = (
        db.query(ExtractedCandidate)
        .filter(
            ExtractedCandidate.job_id == locked_job.id,
            ExtractedCandidate.status == "PENDING",
        )
        .count()
    )
    if pending_count:
        locked_job.status = "REVIEW_READY"
        db.add(locked_job)
        db.commit()
        db.refresh(locked_job)
        return locked_job

    events = (
        db.query(ApprovedMemoryEvent)
        .filter(ApprovedMemoryEvent.job_id == locked_job.id)
        .all()
    )
    statuses = {event.index_status for event in events}
    if statuses & {"PENDING", "INDEXING"}:
        locked_job.status = "REVIEW_READY"
    elif "INDEX_FAILED" in statuses:
        locked_job.status = "COMPLETED_WITH_ERRORS"
    else:
        locked_job.status = "COMPLETED"
    db.add(locked_job)
    db.commit()
    db.refresh(locked_job)
    return locked_job
