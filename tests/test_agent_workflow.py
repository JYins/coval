"""State, approval, idempotency, and tenant tests for the follow-up agent."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import StaleDataError
from sqlalchemy.pool import StaticPool

import src.agent.tools as agent_tools
from src.agent.tools import (
    ImportCalendarContextParams,
    ToolLeaseLostError,
    ToolSpec,
    approve_and_execute_tool,
    execute_tool_call,
    get_tool_call,
)
from src.agent.workflow import (
    WorkflowConflict,
    WorkflowNotFound,
    claim_prepare_workflow,
    build_context,
    create_follow_up_workflow,
    decide_follow_up_workflow,
    get_user_workflow,
    prepare_follow_up_workflow,
    release_prepare_workflow,
)
from src.api.app import app
from src.api.deps import get_current_user
from src.models import Base
from src.models.agent_workflow import FollowUpTask, ToolCall
from src.models.chunk import Chunk
from src.models.conversation import Conversation
from src.models.database import get_db
from src.models.person import Person
from src.models.user import User
from src.rag.indexing import save_chunks_for_conversation


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


def seed_person(db, email="owner@example.com"):
    user = User(email=email, password_hash="test-hash")
    db.add(user)
    db.flush()
    person = Person(
        user_id=user.id,
        name="Ben",
        relationship_type="client",
        notes="Prefers concise updates.",
    )
    db.add(person)
    db.flush()
    conversation = Conversation(
        person_id=person.id,
        source_type="manual",
        raw_content="Ben prefers Tuesday morning meetings and short agendas.",
        language="en",
    )
    db.add(conversation)
    db.flush()
    db.add(
        Chunk(
            conversation_id=conversation.id,
            chunk_text=(
                "Ben: Ben prefers Tuesday morning meetings and short agendas."
            ),
            person_name_prefix="Ben",
            chunk_index=0,
            embedding_model="mock",
        )
    )
    db.commit()
    db.refresh(user)
    db.refresh(person)
    return user, person


def create_workflow(db, user, person, key="follow-up-1"):
    return create_follow_up_workflow(
        db,
        user_id=user.id,
        person_id=person.id,
        goal="Prepare a concise follow-up and make me a task.",
        due_at=None,
        idempotency_key=key,
    )


def test_approved_workflow_creates_one_task(db):
    user, person = seed_person(db)
    workflow = create_workflow(db, user, person)

    prepared = prepare_follow_up_workflow(
        db,
        user_id=user.id,
        workflow_id=workflow.id,
    )

    assert prepared.state == "AWAITING_APPROVAL"
    assert prepared.draft_text
    assert db.query(FollowUpTask).count() == 0
    assert [row.to_state for row in prepared.transitions] == [
        "INGESTED",
        "EXTRACTED",
        "IDENTITY_MATCHED",
        "MEMORY_REVIEW",
        "BRIEFING_OR_DRAFT_READY",
        "AWAITING_APPROVAL",
    ]
    pending = (
        db.query(ToolCall)
        .filter(ToolCall.workflow_id == workflow.id, ToolCall.approval_status == "PENDING")
        .one()
    )
    assert pending.tool_name == "create_follow_up_task"

    executed = decide_follow_up_workflow(
        db,
        user_id=user.id,
        workflow_id=workflow.id,
        decision="approve",
        expected_version=prepared.version,
    )
    duplicate = decide_follow_up_workflow(
        db,
        user_id=user.id,
        workflow_id=workflow.id,
        decision="approve",
        expected_version=prepared.version,
    )

    assert executed.state == "EXECUTED"
    assert duplicate.id == executed.id
    assert db.query(FollowUpTask).count() == 1
    task = db.query(FollowUpTask).one()
    assert task.user_id == user.id
    assert task.person_id == person.id
    assert task.workflow_id == workflow.id


def test_rejected_workflow_has_no_side_effect(db):
    user, person = seed_person(db)
    workflow = create_workflow(db, user, person, key="reject-1")
    prepared = prepare_follow_up_workflow(
        db,
        user_id=user.id,
        workflow_id=workflow.id,
    )

    rejected = decide_follow_up_workflow(
        db,
        user_id=user.id,
        workflow_id=workflow.id,
        decision="reject",
        expected_version=prepared.version,
    )

    assert rejected.state == "REJECTED"
    assert db.query(FollowUpTask).count() == 0
    tool_call = (
        db.query(ToolCall)
        .filter(ToolCall.workflow_id == workflow.id, ToolCall.tool_name == "create_follow_up_task")
        .one()
    )
    assert tool_call.approval_status == "REJECTED"


def test_workflow_request_is_idempotent_and_fingerprint_checked(db):
    user, person = seed_person(db)
    first = create_workflow(db, user, person, key="same-key")
    second = create_workflow(db, user, person, key="same-key")

    assert second.id == first.id
    with pytest.raises(WorkflowConflict):
        create_follow_up_workflow(
            db,
            user_id=user.id,
            person_id=person.id,
            goal="A different request",
            due_at=None,
            idempotency_key="same-key",
        )

    with pytest.raises(ValueError, match="Idempotency-Key"):
        create_follow_up_workflow(
            db,
            user_id=user.id,
            person_id=person.id,
            goal="Valid request",
            due_at=None,
            idempotency_key="x" * 256,
        )


def test_workflow_is_hidden_from_another_tenant(db):
    owner, person = seed_person(db)
    stranger = User(email="stranger@example.com", password_hash="test-hash")
    db.add(stranger)
    db.commit()
    db.refresh(stranger)
    workflow = create_workflow(db, owner, person)

    assert get_user_workflow(db, stranger.id, workflow.id) is None
    with pytest.raises(WorkflowNotFound):
        prepare_follow_up_workflow(
            db,
            user_id=stranger.id,
            workflow_id=workflow.id,
        )


def test_tool_failure_is_visible_in_workflow_and_trace(db, monkeypatch):
    user, person = seed_person(db)
    workflow = create_workflow(db, user, person, key="failure-1")

    def fail_retrieval(*args, **kwargs):
        raise RuntimeError("vector backend unavailable")

    monkeypatch.setattr(agent_tools, "retrieve_chunks", fail_retrieval)
    failed = prepare_follow_up_workflow(
        db,
        user_id=user.id,
        workflow_id=workflow.id,
    )

    assert failed.state == "FAILED"
    assert "vector backend unavailable" in failed.error
    tool_call = (
        db.query(ToolCall)
        .filter(ToolCall.workflow_id == workflow.id, ToolCall.tool_name == "search_person_memory")
        .one()
    )
    assert tool_call.error == "RuntimeError: vector backend unavailable"
    assert tool_call.trace_id == workflow.trace_id


def test_incomplete_read_tool_is_replayed_after_restart(db):
    user, person = seed_person(db)
    workflow = create_workflow(db, user, person, key="resume-read-1")
    db.add(
        ToolCall(
            workflow_id=workflow.id,
            tool_name="search_person_memory",
            input_json={"query": workflow.goal, "top_k": 5},
            idempotency_key="memory-search-v1",
            approval_status="NOT_REQUIRED",
            trace_id=workflow.trace_id,
        )
    )
    db.commit()

    prepared = prepare_follow_up_workflow(
        db,
        user_id=user.id,
        workflow_id=workflow.id,
    )

    assert prepared.state == "AWAITING_APPROVAL"
    replayed = (
        db.query(ToolCall)
        .filter(
            ToolCall.workflow_id == workflow.id,
            ToolCall.idempotency_key == "memory-search-v1",
        )
        .one()
    )
    assert replayed.executed_at is not None
    assert replayed.output_json["source_ids"]


def test_workflow_routes_pause_then_approve(db):
    user, person = seed_person(db)
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: user
    client = TestClient(app)
    try:
        created_response = client.post(
            "/api/workflows/follow-up",
            headers={"Idempotency-Key": "route-1"},
            json={
                "person_id": str(person.id),
                "goal": "Draft a follow-up and create a task after I approve.",
            },
        )
        assert created_response.status_code == 201
        workflow_id = created_response.json()["id"]

        prepared_response = client.post(f"/api/workflows/{workflow_id}/prepare")
        assert prepared_response.status_code == 200
        prepared = prepared_response.json()
        assert prepared["state"] == "AWAITING_APPROVAL"
        assert prepared["pending_action"]["tool_name"] == "create_follow_up_task"
        assert prepared["task"] is None

        decision_response = client.post(
            f"/api/workflows/{workflow_id}/decision",
            json={
                "decision": "approve",
                "expected_version": prepared["version"],
            },
        )
        assert decision_response.status_code == 200
        body = decision_response.json()
        assert body["state"] == "EXECUTED"
        assert body["task"]["status"] == "OPEN"
        assert body["trace_id"]
    finally:
        app.dependency_overrides.clear()


def test_conversation_indexing_rejects_wrong_tenant(db):
    user, person = seed_person(db)
    conversation = db.query(Conversation).filter(Conversation.person_id == person.id).one()

    with pytest.raises(ValueError, match="tenant user"):
        save_chunks_for_conversation(
            db,
            conversation,
            person.name,
            {"vector_backend": "memory"},
            user_id=uuid4(),
            person_id=person.id,
        )

    assert user.id != person.id


def test_approve_reject_race_uses_workflow_version(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    first_db = session_factory()
    second_db = session_factory()
    try:
        user, person = seed_person(first_db)
        user_id = user.id
        workflow = create_workflow(first_db, user, person, key="race-1")
        prepared = prepare_follow_up_workflow(
            first_db,
            user_id=user_id,
            workflow_id=workflow.id,
        )
        stale = get_user_workflow(second_db, user_id, workflow.id)
        assert stale is not None
        stale_version = stale.version

        decide_follow_up_workflow(
            first_db,
            user_id=user_id,
            workflow_id=workflow.id,
            decision="approve",
            expected_version=prepared.version,
        )

        with pytest.raises(StaleDataError):
            decide_follow_up_workflow(
                second_db,
                user_id=user_id,
                workflow_id=workflow.id,
                decision="reject",
                expected_version=stale_version,
            )
        second_db.rollback()
        final = get_user_workflow(first_db, user_id, workflow.id)
        first_db.refresh(final)
        assert final.state == "EXECUTED"
        assert first_db.query(FollowUpTask).count() == 1
    finally:
        first_db.close()
        second_db.close()
        Base.metadata.drop_all(engine)


def test_concurrent_prepare_cannot_mark_active_run_failed(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    first_db = session_factory()
    second_db = session_factory()
    try:
        user, person = seed_person(first_db)
        workflow = create_workflow(first_db, user, person, key="prepare-race-1")
        stale = get_user_workflow(second_db, user.id, workflow.id)
        assert stale is not None

        runner_id = claim_prepare_workflow(first_db, workflow)
        with pytest.raises(StaleDataError):
            prepare_follow_up_workflow(
                second_db,
                user_id=user.id,
                workflow_id=workflow.id,
            )
        second_db.rollback()

        first_db.refresh(workflow)
        assert workflow.state == "INGESTED"
        assert workflow.error is None
        assert first_db.query(ToolCall).count() == 0
        release_prepare_workflow(first_db, workflow, runner_id)
    finally:
        first_db.close()
        second_db.close()
        Base.metadata.drop_all(engine)


def test_fresh_prepare_request_sees_active_lease(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    first_db = session_factory()
    second_db = session_factory()
    try:
        user, person = seed_person(first_db)
        workflow = create_workflow(first_db, user, person, key="active-lease-1")
        runner_id = claim_prepare_workflow(first_db, workflow)

        with pytest.raises(WorkflowConflict, match="already being prepared"):
            prepare_follow_up_workflow(
                second_db,
                user_id=user.id,
                workflow_id=workflow.id,
            )

        first_db.refresh(workflow)
        release_prepare_workflow(first_db, workflow, runner_id)
    finally:
        first_db.close()
        second_db.close()
        Base.metadata.drop_all(engine)


def test_expired_prepare_lease_can_be_recovered(db):
    user, person = seed_person(db)
    workflow = create_workflow(db, user, person, key="expired-lease-1")
    workflow.runner_id = "dead-runner"
    workflow.lease_expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db.add(workflow)
    db.commit()

    prepared = prepare_follow_up_workflow(
        db,
        user_id=user.id,
        workflow_id=workflow.id,
    )

    assert prepared.state == "AWAITING_APPROVAL"
    assert prepared.runner_id is None
    assert prepared.lease_expires_at is None


def test_reject_cannot_override_approved_write_during_repair(db):
    user, person = seed_person(db)
    workflow = create_workflow(db, user, person, key="decision-repair-1")
    prepared = prepare_follow_up_workflow(
        db,
        user_id=user.id,
        workflow_id=workflow.id,
    )
    prepared.approval_decision = "approve"
    db.add(prepared)
    db.commit()
    db.refresh(prepared)
    tool_call = get_tool_call(db, prepared.id, "create-follow-up-task-v1")
    assert tool_call is not None
    context = build_context(db, prepared, person, None)
    approve_and_execute_tool(context, tool_call)

    with pytest.raises(WorkflowConflict, match="already approve"):
        decide_follow_up_workflow(
            db,
            user_id=user.id,
            workflow_id=prepared.id,
            decision="reject",
            expected_version=prepared.version,
        )

    repaired = decide_follow_up_workflow(
        db,
        user_id=user.id,
        workflow_id=prepared.id,
        decision="approve",
        expected_version=prepared.version,
    )
    assert repaired.state == "EXECUTED"
    assert db.query(FollowUpTask).count() == 1


def test_late_runner_cannot_overwrite_new_runner_result(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    first_db = session_factory()
    second_db = session_factory()
    try:
        user, person = seed_person(first_db)
        workflow = create_workflow(first_db, user, person, key="fence-1")
        runner_a = claim_prepare_workflow(first_db, workflow)
        row = ToolCall(
            workflow_id=workflow.id,
            tool_name="import_calendar_context",
            input_json={"days": 7},
            idempotency_key="fence-tool-v1",
            approval_status="NOT_REQUIRED",
            runner_id=runner_a,
            trace_id=workflow.trace_id,
        )
        first_db.add(row)
        first_db.commit()
        first_db.refresh(row)
        context = build_context(
            first_db,
            workflow,
            person,
            None,
            runner_id=runner_a,
        )

        def finish_with_new_runner(context, params):
            current = get_user_workflow(second_db, user.id, workflow.id)
            assert current is not None
            current.lease_expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
            second_db.add(current)
            second_db.commit()
            runner_b = claim_prepare_workflow(second_db, current)
            replacement = get_tool_call(second_db, workflow.id, "fence-tool-v1")
            assert replacement is not None
            assert replacement.runner_id == runner_b
            replacement.output_json = {"winner": "B"}
            replacement.executed_at = datetime.now(timezone.utc)
            second_db.add(replacement)
            second_db.commit()
            return {"winner": "A"}

        spec = ToolSpec(ImportCalendarContextParams, finish_with_new_runner)
        with pytest.raises(ToolLeaseLostError):
            execute_tool_call(
                context,
                row,
                spec,
                ImportCalendarContextParams(days=7),
            )
        first_db.rollback()

        saved = get_tool_call(second_db, workflow.id, "fence-tool-v1")
        second_db.refresh(saved)
        assert saved.output_json == {"winner": "B"}
    finally:
        first_db.close()
        second_db.close()
        Base.metadata.drop_all(engine)


def test_tool_call_version_blocks_result_after_runner_handoff(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    first_db = session_factory()
    second_db = session_factory()
    try:
        user, person = seed_person(first_db)
        workflow = create_workflow(first_db, user, person, key="tool-version-1")
        row = ToolCall(
            workflow_id=workflow.id,
            tool_name="import_calendar_context",
            input_json={"days": 7},
            idempotency_key="tool-version-v1",
            approval_status="NOT_REQUIRED",
            runner_id="runner-a",
            trace_id=workflow.trace_id,
        )
        first_db.add(row)
        first_db.commit()
        first_db.refresh(row)

        replacement = get_tool_call(second_db, workflow.id, "tool-version-v1")
        assert replacement is not None
        replacement.runner_id = "runner-b"
        second_db.add(replacement)
        second_db.commit()

        row.output_json = {"winner": "A"}
        row.executed_at = datetime.now(timezone.utc)
        first_db.add(row)
        with pytest.raises(StaleDataError):
            first_db.commit()
        first_db.rollback()

        second_db.refresh(replacement)
        assert replacement.runner_id == "runner-b"
        assert replacement.output_json is None
    finally:
        first_db.close()
        second_db.close()
        Base.metadata.drop_all(engine)
