"""End-to-end tests for Voice G0 review and approved memory."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.voice.service as voice_service
from src.analysis.personality import load_person_conversations
from src.api.app import app
from src.api.deps import get_current_user
from src.api.routes_voice import raise_voice_http_error
from src.models import Base
from src.models.chunk import Chunk
from src.models.conversation import Conversation
from src.models.database import get_db
from src.models.person import Person
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
    VoiceConflict,
    VoiceNotFound,
    add_transcript_revision,
    cancel_voice_job,
    create_voice_job,
    decide_candidate,
    finish_memory_index,
    get_voice_job,
    mark_audio_deleted,
)


FAKE_AUDIO = b"RIFF\x00\x00\x00\x00WAVEsynthetic-public-fixture"


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


def seed_person(db, email="voice-owner@example.com"):
    user = User(email=email, password_hash="test-hash")
    db.add(user)
    db.flush()
    person = Person(
        user_id=user.id,
        name="Lin",
        relationship_type="client",
        notes="Voice G0 synthetic fixture only.",
    )
    db.add(person)
    db.commit()
    db.refresh(user)
    db.refresh(person)
    return user, person


def create_job(db, user, person, key="voice-job-1", fixture="mandarin_two_speaker_v1"):
    job, created_by_request = create_voice_job(
        db,
        user_id=user.id,
        person_id=person.id,
        audio=FAKE_AUDIO,
        mime_type="audio/wav",
        language="zh",
        recorded_at=datetime(2026, 8, 19, 9, 0, tzinfo=timezone.utc),
        provider_name="fake",
        fixture_name=fixture,
        idempotency_key=key,
    )
    if created_by_request:
        mark_audio_deleted(db, job)
    return job


def test_fake_provider_builds_typed_review_job_and_deletes_audio(db, caplog):
    user, person = seed_person(db)
    job = create_job(db, user, person)

    assert job.status == "REVIEW_READY"
    assert job.retention_policy == "delete_after_processing"
    assert job.audio_deleted_at is not None
    assert job.audio_sha256
    assert job.audio_size_bytes == len(FAKE_AUDIO)
    assert job.provider_version == "1"
    assert job.model_name == "synthetic-fixture"
    assert job.model_revision == "mandarin_two_speaker_v1"
    assert job.trace_id
    assert job.retry_count == 0
    assert not hasattr(job, "audio_bytes")
    assert not hasattr(job, "audio_path")
    assert db.query(AudioSegment).count() == 2
    assert db.query(SpeakerTurn).count() == 2
    assert db.query(TranscriptAlternative).count() == 3
    assert db.query(TranscriptRevision).count() == 2
    assert db.query(ExtractedCandidate).count() == 2
    segment = db.query(AudioSegment).order_by(AudioSegment.segment_index).first()
    turn = db.query(SpeakerTurn).order_by(SpeakerTurn.turn_index).first()
    candidate = (
        db.query(ExtractedCandidate)
        .order_by(ExtractedCandidate.candidate_index)
        .first()
    )
    assert segment.channel == 0
    assert segment.vad_confidence == 0.95
    assert turn.overlap is False
    assert candidate.source_turn_id == turn.id
    assert candidate.speaker_label == "speaker_0"
    assert candidate.source_start_ms == 0
    assert candidate.source_end_ms == 4200
    assert "我下周二上午有时间" not in caplog.text
    assert "偏好周二上午开会" not in caplog.text


def test_voice_upload_is_idempotent_and_audio_bound(db):
    user, person = seed_person(db)
    first = create_job(db, user, person, key="same-upload")
    duplicate = create_job(db, user, person, key="same-upload")

    assert duplicate.id == first.id
    assert db.query(VoiceIngestionJob).count() == 1
    with pytest.raises(VoiceConflict):
        create_voice_job(
            db,
            user_id=user.id,
            person_id=person.id,
            audio=FAKE_AUDIO + b"changed",
            mime_type="audio/wav",
            language="zh",
            recorded_at=datetime(2026, 8, 19, 9, 0, tzinfo=timezone.utc),
            provider_name="fake",
            fixture_name="mandarin_two_speaker_v1",
            idempotency_key="same-upload",
        )


def test_approve_writes_one_reviewed_memory_and_chunks(db):
    user, person = seed_person(db)
    job = create_job(db, user, person)
    candidate = (
        db.query(ExtractedCandidate)
        .filter(ExtractedCandidate.job_id == job.id)
        .order_by(ExtractedCandidate.candidate_index.asc())
        .first()
    )

    decision = decide_candidate(
        db,
        user_id=user.id,
        job_id=job.id,
        candidate_id=candidate.id,
        decision="approve",
        edited_content="偏好周二上午开会，会议控制在二十分钟。",
        expected_version=candidate.version,
        idempotency_key="approve-1",
    )
    duplicate = decide_candidate(
        db,
        user_id=user.id,
        job_id=job.id,
        candidate_id=candidate.id,
        decision="approve",
        edited_content="偏好周二上午开会，会议控制在二十分钟。",
        expected_version=1,
        idempotency_key="approve-1",
    )

    assert duplicate.id == decision.id
    assert db.query(ReviewDecision).count() == 1
    assert db.query(ApprovedMemoryEvent).count() == 1
    assert db.query(Conversation).count() == 1
    assert db.query(Chunk).count() >= 1
    conversation = db.query(Conversation).one()
    assert conversation.source_type == "voice"
    assert conversation.raw_content == "偏好周二上午开会，会议控制在二十分钟。"
    event = db.query(ApprovedMemoryEvent).one()
    assert event.index_status == "INDEXED"
    assert event.conversation_id == conversation.id
    assert event.source_audio_sha256 == job.audio_sha256
    assert event.provider == "fake"
    assert event.source_spans[0]["speaker_label"] == "speaker_0"


def test_identical_approve_from_stale_session_returns_winner(db):
    user, person = seed_person(db)
    job = create_job(db, user, person)
    candidate = (
        db.query(ExtractedCandidate)
        .filter(ExtractedCandidate.job_id == job.id)
        .order_by(ExtractedCandidate.candidate_index.asc())
        .first()
    )
    other_db = sessionmaker(bind=db.bind)()
    try:
        other_candidate = other_db.get(ExtractedCandidate, candidate.id)
        first = decide_candidate(
            db,
            user_id=user.id,
            job_id=job.id,
            candidate_id=candidate.id,
            decision="approve",
            edited_content=None,
            expected_version=candidate.version,
            idempotency_key="approve-stale-session",
        )
        second = decide_candidate(
            other_db,
            user_id=user.id,
            job_id=job.id,
            candidate_id=other_candidate.id,
            decision="approve",
            edited_content=None,
            expected_version=other_candidate.version,
            idempotency_key="approve-stale-session",
        )

        assert second.id == first.id
        assert db.query(ReviewDecision).count() == 1
        assert db.query(ApprovedMemoryEvent).count() == 1
        assert db.query(Conversation).count() == 1
    finally:
        other_db.close()


def test_late_index_runner_cannot_overwrite_current_owner(db):
    user, person = seed_person(db)
    job = create_job(db, user, person)
    candidate = (
        db.query(ExtractedCandidate)
        .filter(ExtractedCandidate.job_id == job.id)
        .order_by(ExtractedCandidate.candidate_index.asc())
        .first()
    )
    decide_candidate(
        db,
        user_id=user.id,
        job_id=job.id,
        candidate_id=candidate.id,
        decision="approve",
        edited_content=None,
        expected_version=candidate.version,
        idempotency_key="approve-for-fencing",
    )
    event = db.query(ApprovedMemoryEvent).one()
    event.index_status = "INDEXING"
    event.index_runner_id = "runner-b"
    event.indexed_at = None
    db.commit()

    with pytest.raises(VoiceConflict, match="lease was lost"):
        finish_memory_index(
            db,
            event_id=event.id,
            runner_id="runner-a",
            index_status="INDEXED",
            error_code=None,
        )

    db.refresh(event)
    assert event.index_status == "INDEXING"
    assert event.index_runner_id == "runner-b"


def test_reject_has_no_crm_memory_side_effect(db):
    user, person = seed_person(db)
    job = create_job(db, user, person)
    candidate = db.query(ExtractedCandidate).filter_by(job_id=job.id).first()

    decide_candidate(
        db,
        user_id=user.id,
        job_id=job.id,
        candidate_id=candidate.id,
        decision="reject",
        edited_content=None,
        expected_version=candidate.version,
        idempotency_key="reject-1",
    )

    assert db.query(ReviewDecision).count() == 1
    assert db.query(ApprovedMemoryEvent).count() == 0
    assert db.query(Conversation).count() == 0
    assert db.query(Chunk).count() == 0


def test_review_idempotency_key_rejects_changed_content(db):
    user, person = seed_person(db)
    job = create_job(db, user, person)
    candidate = db.query(ExtractedCandidate).filter_by(job_id=job.id).first()
    decide_candidate(
        db,
        user_id=user.id,
        job_id=job.id,
        candidate_id=candidate.id,
        decision="approve",
        edited_content="周二上午开会。",
        expected_version=candidate.version,
        idempotency_key="decision-key",
    )

    with pytest.raises(VoiceConflict):
        decide_candidate(
            db,
            user_id=user.id,
            job_id=job.id,
            candidate_id=candidate.id,
            decision="approve",
            edited_content="周四上午开会。",
            expected_version=1,
            idempotency_key="decision-key",
        )


def test_cross_tenant_voice_records_are_hidden(db):
    owner, person = seed_person(db)
    stranger = User(email="voice-stranger@example.com", password_hash="test-hash")
    db.add(stranger)
    db.commit()
    db.refresh(stranger)
    job = create_job(db, owner, person)
    candidate = db.query(ExtractedCandidate).filter_by(job_id=job.id).first()

    assert get_voice_job(db, stranger.id, job.id) is None
    with pytest.raises(VoiceNotFound):
        decide_candidate(
            db,
            user_id=stranger.id,
            job_id=job.id,
            candidate_id=candidate.id,
            decision="reject",
            edited_content=None,
            expected_version=candidate.version,
            idempotency_key="stranger-decision",
        )


def test_transcript_revision_is_append_only_and_stales_old_candidate(db):
    user, person = seed_person(db)
    job = create_job(db, user, person)
    turn = (
        db.query(SpeakerTurn)
        .filter(SpeakerTurn.job_id == job.id)
        .order_by(SpeakerTurn.turn_index.asc())
        .first()
    )
    old_candidate = (
        db.query(ExtractedCandidate)
        .join(TranscriptRevision)
        .filter(
            ExtractedCandidate.job_id == job.id,
            TranscriptRevision.turn_id == turn.id,
        )
        .first()
    )

    revision = add_transcript_revision(
        db,
        user_id=user.id,
        job_id=job.id,
        turn_id=turn.id,
        text="我下周四上午有时间，会议控制在二十分钟。",
        reason="corrected weekday after listening again",
        expected_version=turn.version,
    )

    db.refresh(old_candidate)
    assert revision.revision_no == 2
    assert db.query(TranscriptRevision).filter_by(turn_id=turn.id).count() == 2
    assert old_candidate.status == "STALE"
    replacement = (
        db.query(ExtractedCandidate)
        .filter(
            ExtractedCandidate.revision_id == revision.id,
            ExtractedCandidate.status == "PENDING",
        )
        .one()
    )
    assert replacement.content == revision.text
    assert revision.raw_text != revision.normalized_text
    assert revision.provenance["source"] == "human_revision"
    db.refresh(turn)
    assert turn.needs_review is True
    with pytest.raises(VoiceConflict, match="stale"):
        decide_candidate(
            db,
            user_id=user.id,
            job_id=job.id,
            candidate_id=old_candidate.id,
            decision="approve",
            edited_content=None,
            expected_version=old_candidate.version,
            idempotency_key="stale-approve",
        )


def test_failed_index_retry_does_not_duplicate_memory(db, monkeypatch):
    user, person = seed_person(db)
    job = create_job(db, user, person)
    candidate = db.query(ExtractedCandidate).filter_by(job_id=job.id).first()
    calls = {"count": 0}
    commit_modes = []

    def fail_once(*args, **kwargs):
        calls["count"] += 1
        commit_modes.append(kwargs["commit"])
        if calls["count"] == 1:
            raise RuntimeError("transcript text must not appear in error")
        return [SimpleNamespace(id="chunk-1")]

    monkeypatch.setattr(voice_service, "save_chunks_for_conversation", fail_once)
    decision = decide_candidate(
        db,
        user_id=user.id,
        job_id=job.id,
        candidate_id=candidate.id,
        decision="approve",
        edited_content=None,
        expected_version=candidate.version,
        idempotency_key="retry-index-1",
    )
    event = db.query(ApprovedMemoryEvent).one()
    assert event.index_status == "INDEX_FAILED"
    assert event.error_code == "VOICE_MEMORY_INDEX_ERROR"

    second_candidate = (
        db.query(ExtractedCandidate)
        .filter(
            ExtractedCandidate.job_id == job.id,
            ExtractedCandidate.id != candidate.id,
        )
        .one()
    )
    decide_candidate(
        db,
        user_id=user.id,
        job_id=job.id,
        candidate_id=second_candidate.id,
        decision="reject",
        edited_content=None,
        expected_version=second_candidate.version,
        idempotency_key="reject-second-after-index-failure",
    )
    db.refresh(job)
    assert job.status == "COMPLETED_WITH_ERRORS"

    duplicate = decide_candidate(
        db,
        user_id=user.id,
        job_id=job.id,
        candidate_id=candidate.id,
        decision="approve",
        edited_content=None,
        expected_version=1,
        idempotency_key="retry-index-1",
    )
    db.refresh(event)
    assert duplicate.id == decision.id
    assert event.index_status == "INDEXED"
    db.refresh(job)
    assert job.status == "COMPLETED"
    assert job.retry_count == 1
    assert commit_modes == [False, False]
    assert db.query(Conversation).count() == 1
    assert db.query(ApprovedMemoryEvent).count() == 1


def test_cancel_stales_pending_candidates_without_memory(db):
    user, person = seed_person(db)
    job = create_job(db, user, person)

    canceled = cancel_voice_job(
        db,
        user_id=user.id,
        job_id=job.id,
        expected_version=job.version,
    )
    duplicate = cancel_voice_job(
        db,
        user_id=user.id,
        job_id=job.id,
        expected_version=1,
    )

    assert canceled.status == "CANCELED"
    assert duplicate.id == canceled.id
    assert {
        row.status for row in db.query(ExtractedCandidate).filter_by(job_id=job.id)
    } == {"STALE"}
    assert db.query(ReviewDecision).count() == 0
    assert db.query(ApprovedMemoryEvent).count() == 0
    assert db.query(Conversation).count() == 0


def test_cancel_during_transcription_discards_provider_result(db, monkeypatch):
    user, person = seed_person(db)
    fake_provider = voice_service.build_voice_provider("fake")

    class CancelingProvider:
        def transcribe(self, audio, *, language, fixture_name):
            other_db = sessionmaker(bind=db.bind)()
            try:
                current = other_db.query(VoiceIngestionJob).one()
                cancel_voice_job(
                    other_db,
                    user_id=user.id,
                    job_id=current.id,
                    expected_version=current.version,
                )
            finally:
                other_db.close()
            return fake_provider.transcribe(
                audio,
                language=language,
                fixture_name=fixture_name,
            )

    monkeypatch.setattr(
        voice_service,
        "build_voice_provider",
        lambda provider_name: CancelingProvider(),
    )
    job = create_job(db, user, person, key="cancel-during-transcription")

    assert job.status == "CANCELED", job.error_code
    assert job.audio_deleted_at is not None
    assert db.query(AudioSegment).count() == 0
    assert db.query(SpeakerTurn).count() == 0
    assert db.query(ExtractedCandidate).count() == 0
    assert db.query(Conversation).count() == 0


def test_reviewed_turn_cannot_be_revised_after_memory_decision(db):
    user, person = seed_person(db)
    job = create_job(db, user, person)
    candidate = (
        db.query(ExtractedCandidate)
        .filter(ExtractedCandidate.job_id == job.id)
        .order_by(ExtractedCandidate.candidate_index.asc())
        .first()
    )
    turn_id = candidate.revision.turn_id
    turn = db.query(SpeakerTurn).filter(SpeakerTurn.id == turn_id).one()
    decide_candidate(
        db,
        user_id=user.id,
        job_id=job.id,
        candidate_id=candidate.id,
        decision="approve",
        edited_content=None,
        expected_version=candidate.version,
        idempotency_key="approve-before-revision",
    )

    with pytest.raises(VoiceConflict, match="reviewed candidate"):
        add_transcript_revision(
            db,
            user_id=user.id,
            job_id=job.id,
            turn_id=turn.id,
            text="不允许在批准后悄悄改写证据。",
            reason="late correction",
            expected_version=turn.version,
        )


def test_provider_failure_keeps_sanitized_job_without_audio(db):
    user, person = seed_person(db)
    job = create_job(db, user, person, key="bad-fixture", fixture="missing")

    assert job.status == "FAILED"
    assert job.error_code == "ValueError"
    assert job.audio_deleted_at is not None
    assert db.query(TranscriptAlternative).count() == 0


def test_voice_memory_is_excluded_from_personality_input(db):
    user, person = seed_person(db)
    db.add_all(
        [
            Conversation(
                person_id=person.id,
                source_type="manual",
                raw_content="Manual note allowed for profile draft.",
                language="en",
            ),
            Conversation(
                person_id=person.id,
                source_type="voice",
                raw_content="Approved voice memory is not personality evidence.",
                language="en",
            ),
        ]
    )
    db.commit()

    rows = load_person_conversations(db, person.id)

    assert [row.source_type for row in rows] == ["manual"]
    assert user.id == person.user_id


def test_voice_api_upload_review_and_tenant_404(db):
    owner, person = seed_person(db)
    stranger = User(email="api-stranger@example.com", password_hash="test-hash")
    db.add(stranger)
    db.commit()
    db.refresh(stranger)
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: owner
    client = TestClient(app)
    try:
        response = client.post(
            "/api/voice/jobs",
            headers={"Idempotency-Key": "api-voice-1"},
            data={"person_id": str(person.id), "language": "zh"},
            files={"audio": ("synthetic.wav", FAKE_AUDIO, "audio/wav")},
        )
        assert response.status_code == 201
        job = response.json()
        assert job["status"] == "REVIEW_READY"
        assert job["audio_deleted_at"] is not None
        assert job["provider_version"] == "1"
        assert job["model_name"] == "synthetic-fixture"
        assert job["trace_id"]
        assert job["segments"][0]["vad_confidence"] == 0.95
        assert job["turns"][0]["overlap"] is False
        assert len(job["turns"][0]["alternatives"]) == 2
        candidate = job["candidates"][0]

        decision_response = client.post(
            f"/api/voice/jobs/{job['id']}/candidates/{candidate['id']}/decision",
            headers={"Idempotency-Key": "api-approve-1"},
            json={
                "decision": "approve",
                "edited_content": "周二上午适合短会。",
                "expected_version": candidate["version"],
            },
        )
        assert decision_response.status_code == 200
        reviewed = decision_response.json()["candidates"][0]
        assert reviewed["status"] == "APPROVED"
        assert reviewed["approved_memory_event"]["index_status"] == "INDEXED"

        app.dependency_overrides[get_current_user] = lambda: stranger
        hidden = client.get(f"/api/voice/jobs/{job['id']}")
        assert hidden.status_code == 404
        hidden_decision = client.post(
            f"/api/voice/jobs/{job['id']}/candidates/{candidate['id']}/decision",
            headers={"Idempotency-Key": "stranger-api-decision"},
            json={
                "decision": "reject",
                "expected_version": candidate["version"],
            },
        )
        assert hidden_decision.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_voice_api_cancel_and_safe_integrity_error(db):
    owner, person = seed_person(db)
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: owner
    client = TestClient(app)
    try:
        response = client.post(
            "/api/voice/jobs",
            headers={"Idempotency-Key": "api-cancel-voice"},
            data={"person_id": str(person.id), "language": "zh"},
            files={"audio": ("synthetic.wav", FAKE_AUDIO, "audio/wav")},
        )
        job = response.json()
        canceled = client.post(
            f"/api/voice/jobs/{job['id']}/cancel",
            json={"expected_version": job["version"]},
        )

        assert canceled.status_code == 200
        assert canceled.json()["status"] == "CANCELED"
        assert {row["status"] for row in canceled.json()["candidates"]} == {
            "STALE"
        }

        secret = "private transcript should not leak"
        with pytest.raises(HTTPException) as error:
            raise_voice_http_error(
                db,
                IntegrityError("insert revision", {"text": secret}, Exception()),
            )
        assert error.value.status_code == 409
        assert error.value.detail == "voice write conflict"
        assert secret not in error.value.detail
    finally:
        app.dependency_overrides.clear()
