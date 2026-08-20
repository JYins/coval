"""Typed persistence models for reviewed voice ingestion."""

from __future__ import annotations

import uuid

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from src.models.database import Base
from src.models.types import GUID


class VoiceIngestionJob(Base):
    __tablename__ = "voice_ingestion_jobs"
    __table_args__ = (
        UniqueConstraint("user_id", "idempotency_key", name="uq_voice_job_user_key"),
        CheckConstraint(
            "status IN ('RECEIVED', 'TRANSCRIBING', 'REVIEW_READY', 'COMPLETED', 'COMPLETED_WITH_ERRORS', 'FAILED', 'CANCELED')",
            name="ck_voice_job_status",
        ),
        CheckConstraint(
            "retention_policy = 'delete_after_processing'",
            name="ck_voice_job_retention",
        ),
    )

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False, index=True)
    person_id = Column(GUID(), ForeignKey("persons.id"), nullable=False, index=True)
    status = Column(String(30), nullable=False, default="RECEIVED")
    provider = Column(String(50), nullable=False)
    provider_version = Column(String(100), nullable=True)
    model_name = Column(String(255), nullable=True)
    model_revision = Column(String(255), nullable=True)
    fixture_name = Column(String(100), nullable=True)
    language = Column(String(20), nullable=False, default="zh")
    recorded_at = Column(DateTime(timezone=True), nullable=True)
    audio_sha256 = Column(String(64), nullable=False)
    audio_size_bytes = Column(Integer, nullable=False)
    audio_mime_type = Column(String(100), nullable=False)
    retention_policy = Column(
        String(50),
        nullable=False,
        default="delete_after_processing",
    )
    audio_deleted_at = Column(DateTime(timezone=True), nullable=True)
    idempotency_key = Column(String(255), nullable=False)
    request_fingerprint = Column(String(64), nullable=False)
    trace_id = Column(String(36), nullable=False, default=lambda: str(uuid.uuid4()))
    retry_count = Column(Integer, nullable=False, default=0)
    error_code = Column(String(100), nullable=True)
    version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __mapper_args__ = {"version_id_col": version}

    user = relationship("User")
    person = relationship("Person")
    segments = relationship(
        "AudioSegment",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="AudioSegment.segment_index",
    )
    turns = relationship(
        "SpeakerTurn",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="SpeakerTurn.turn_index",
    )
    candidates = relationship(
        "ExtractedCandidate",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="ExtractedCandidate.candidate_index",
    )
    approved_events = relationship(
        "ApprovedMemoryEvent",
        back_populates="job",
        cascade="all, delete-orphan",
    )


class AudioSegment(Base):
    __tablename__ = "audio_segments"
    __table_args__ = (
        UniqueConstraint("job_id", "segment_index", name="uq_audio_segment_job_index"),
        CheckConstraint("start_ms >= 0", name="ck_audio_segment_start"),
        CheckConstraint("end_ms > start_ms", name="ck_audio_segment_end"),
    )

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    job_id = Column(
        GUID(),
        ForeignKey("voice_ingestion_jobs.id"),
        nullable=False,
        index=True,
    )
    segment_index = Column(Integer, nullable=False)
    start_ms = Column(Integer, nullable=False)
    end_ms = Column(Integer, nullable=False)
    channel = Column(Integer, nullable=False, default=0)
    vad_confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    job = relationship("VoiceIngestionJob", back_populates="segments")
    turns = relationship(
        "SpeakerTurn",
        back_populates="segment",
        cascade="all, delete-orphan",
        order_by="SpeakerTurn.turn_index",
    )


class SpeakerTurn(Base):
    __tablename__ = "speaker_turns"
    __table_args__ = (
        UniqueConstraint("job_id", "turn_index", name="uq_speaker_turn_job_index"),
        CheckConstraint("start_ms >= 0", name="ck_speaker_turn_start"),
        CheckConstraint("end_ms > start_ms", name="ck_speaker_turn_end"),
    )

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    job_id = Column(
        GUID(),
        ForeignKey("voice_ingestion_jobs.id"),
        nullable=False,
        index=True,
    )
    segment_id = Column(GUID(), ForeignKey("audio_segments.id"), nullable=False)
    turn_index = Column(Integer, nullable=False)
    speaker_label = Column(String(50), nullable=False)
    start_ms = Column(Integer, nullable=False)
    end_ms = Column(Integer, nullable=False)
    diarization_confidence = Column(Float, nullable=True)
    overlap = Column(Boolean, nullable=False, default=False)
    needs_review = Column(Boolean, nullable=False, default=True)
    revision_count = Column(Integer, nullable=False, default=1)
    version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __mapper_args__ = {"version_id_col": version}

    job = relationship("VoiceIngestionJob", back_populates="turns")
    segment = relationship("AudioSegment", back_populates="turns")
    alternatives = relationship(
        "TranscriptAlternative",
        back_populates="turn",
        cascade="all, delete-orphan",
        order_by="TranscriptAlternative.rank",
    )
    revisions = relationship(
        "TranscriptRevision",
        back_populates="turn",
        cascade="all, delete-orphan",
        order_by="TranscriptRevision.revision_no",
    )


class TranscriptAlternative(Base):
    __tablename__ = "transcript_alternatives"
    __table_args__ = (
        UniqueConstraint("turn_id", "rank", name="uq_transcript_alternative_rank"),
        CheckConstraint("rank >= 1", name="ck_transcript_alternative_rank"),
    )

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    turn_id = Column(GUID(), ForeignKey("speaker_turns.id"), nullable=False, index=True)
    rank = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    provider = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    turn = relationship("SpeakerTurn", back_populates="alternatives")


class TranscriptRevision(Base):
    __tablename__ = "transcript_revisions"
    __table_args__ = (
        UniqueConstraint("turn_id", "revision_no", name="uq_transcript_revision_no"),
        CheckConstraint("revision_no >= 1", name="ck_transcript_revision_no"),
        CheckConstraint(
            "source IN ('PROVIDER', 'USER')",
            name="ck_transcript_revision_source",
        ),
    )

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    turn_id = Column(GUID(), ForeignKey("speaker_turns.id"), nullable=False, index=True)
    revision_no = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    raw_text = Column(Text, nullable=False)
    normalized_text = Column(Text, nullable=False)
    provenance = Column(JSON, nullable=False, default=dict)
    source = Column(String(20), nullable=False)
    base_alternative_id = Column(
        GUID(),
        ForeignKey("transcript_alternatives.id"),
        nullable=True,
    )
    editor_user_id = Column(GUID(), ForeignKey("users.id"), nullable=True)
    reason = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    turn = relationship("SpeakerTurn", back_populates="revisions")
    base_alternative = relationship("TranscriptAlternative")
    editor = relationship("User")


class ExtractedCandidate(Base):
    __tablename__ = "extracted_candidates"
    __table_args__ = (
        UniqueConstraint("job_id", "candidate_index", name="uq_candidate_job_index"),
        CheckConstraint(
            "candidate_type IN ('stated_fact', 'stated_preference', 'commitment', 'follow_up_action')",
            name="ck_extracted_candidate_type",
        ),
        CheckConstraint(
            "status IN ('PENDING', 'STALE', 'APPROVED', 'REJECTED')",
            name="ck_extracted_candidate_status",
        ),
    )

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    job_id = Column(
        GUID(),
        ForeignKey("voice_ingestion_jobs.id"),
        nullable=False,
        index=True,
    )
    revision_id = Column(
        GUID(),
        ForeignKey("transcript_revisions.id"),
        nullable=False,
        index=True,
    )
    candidate_index = Column(Integer, nullable=False)
    candidate_type = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    uncertainty = Column(JSON, nullable=False, default=dict)
    source_turn_id = Column(GUID(), ForeignKey("speaker_turns.id"), nullable=False)
    speaker_label = Column(String(50), nullable=False)
    source_start_ms = Column(Integer, nullable=False)
    source_end_ms = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default="PENDING")
    version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __mapper_args__ = {"version_id_col": version}

    job = relationship("VoiceIngestionJob", back_populates="candidates")
    revision = relationship("TranscriptRevision")
    source_turn = relationship("SpeakerTurn")
    decision = relationship(
        "ReviewDecision",
        back_populates="candidate",
        uselist=False,
        cascade="all, delete-orphan",
    )
    approved_event = relationship(
        "ApprovedMemoryEvent",
        back_populates="candidate",
        uselist=False,
        cascade="all, delete-orphan",
    )


class ReviewDecision(Base):
    __tablename__ = "review_decisions"
    __table_args__ = (
        UniqueConstraint("candidate_id", name="uq_review_decision_candidate"),
        UniqueConstraint(
            "candidate_id",
            "idempotency_key",
            name="uq_review_decision_candidate_key",
        ),
        CheckConstraint(
            "decision IN ('approve', 'reject')",
            name="ck_review_decision_value",
        ),
    )

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(
        GUID(),
        ForeignKey("extracted_candidates.id"),
        nullable=False,
        index=True,
    )
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False, index=True)
    decision = Column(String(20), nullable=False)
    edited_content = Column(Text, nullable=True)
    idempotency_key = Column(String(255), nullable=False)
    request_fingerprint = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    candidate = relationship("ExtractedCandidate", back_populates="decision")
    user = relationship("User")
    approved_event = relationship(
        "ApprovedMemoryEvent",
        back_populates="review_decision",
        uselist=False,
    )


class ApprovedMemoryEvent(Base):
    __tablename__ = "approved_memory_events"
    __table_args__ = (
        CheckConstraint(
            "index_status IN ('PENDING', 'INDEXING', 'INDEXED', 'INDEX_FAILED')",
            name="ck_approved_memory_index_status",
        ),
    )

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False, index=True)
    person_id = Column(GUID(), ForeignKey("persons.id"), nullable=False, index=True)
    job_id = Column(
        GUID(),
        ForeignKey("voice_ingestion_jobs.id"),
        nullable=False,
        index=True,
    )
    candidate_id = Column(
        GUID(),
        ForeignKey("extracted_candidates.id"),
        nullable=False,
        unique=True,
    )
    review_decision_id = Column(
        GUID(),
        ForeignKey("review_decisions.id"),
        nullable=False,
        unique=True,
    )
    conversation_id = Column(
        GUID(),
        ForeignKey("conversations.id"),
        nullable=False,
        unique=True,
    )
    content = Column(Text, nullable=False)
    source_turn_ids = Column(JSON, nullable=False, default=list)
    source_audio_sha256 = Column(String(64), nullable=False)
    source_spans = Column(JSON, nullable=False, default=list)
    provider = Column(String(50), nullable=False)
    model_name = Column(String(255), nullable=True)
    model_revision = Column(String(255), nullable=True)
    index_status = Column(String(30), nullable=False, default="PENDING")
    index_runner_id = Column(String(36), nullable=True)
    index_lease_expires_at = Column(DateTime(timezone=True), nullable=True)
    indexed_at = Column(DateTime(timezone=True), nullable=True)
    error_code = Column(String(100), nullable=True)
    version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    user = relationship("User")
    person = relationship("Person")
    job = relationship("VoiceIngestionJob", back_populates="approved_events")
    candidate = relationship("ExtractedCandidate", back_populates="approved_event")
    review_decision = relationship("ReviewDecision", back_populates="approved_event")
    conversation = relationship("Conversation")

    __mapper_args__ = {"version_id_col": version}
