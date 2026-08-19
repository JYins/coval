"""Durable follow-up workflow service."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.agent.states import (
    AWAITING_APPROVAL,
    BRIEFING_OR_DRAFT_READY,
    EXECUTED,
    EXTRACTED,
    FAILED,
    FINAL_STATES,
    IDENTITY_MATCHED,
    INGESTED,
    MEMORY_REVIEW,
    REJECTED,
    check_transition,
)
from src.agent.tools import (
    ToolContext,
    ToolExecutionError,
    ToolLeaseLostError,
    approve_and_execute_tool,
    get_tool_call,
    propose_tool,
    reject_tool,
    run_tool,
)
from src.models.agent_workflow import (
    AgentWorkflow,
    FollowUpTask,
    ToolCall,
    WorkflowTransition,
)
from src.models.person import Person
from src.rag.retriever import load_default_config


class WorkflowNotFound(ValueError):
    pass


class WorkflowConflict(ValueError):
    pass


PREPARE_LEASE_MINUTES = 10


def request_fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_user_person(db: Session, user_id: UUID, person_id: UUID) -> Person | None:
    return (
        db.query(Person)
        .filter(Person.id == person_id, Person.user_id == user_id)
        .first()
    )


def get_user_workflow(
    db: Session,
    user_id: UUID,
    workflow_id: UUID,
) -> AgentWorkflow | None:
    return (
        db.query(AgentWorkflow)
        .filter(AgentWorkflow.id == workflow_id, AgentWorkflow.user_id == user_id)
        .first()
    )


def create_follow_up_workflow(
    db: Session,
    *,
    user_id: UUID,
    person_id: UUID,
    goal: str,
    due_at: datetime | None,
    idempotency_key: str,
) -> AgentWorkflow:
    cleaned_goal = goal.strip()
    cleaned_key = idempotency_key.strip()
    if not cleaned_goal:
        raise ValueError("goal is required")
    if len(cleaned_goal) > 4000:
        raise ValueError("goal should be at most 4000 characters")
    if not cleaned_key:
        raise ValueError("Idempotency-Key is required")
    if len(cleaned_key) > 255:
        raise ValueError("Idempotency-Key should be at most 255 characters")

    person = get_user_person(db, user_id, person_id)
    if person is None:
        raise WorkflowNotFound("person not found")

    payload = {
        "person_id": str(person_id),
        "goal": cleaned_goal,
        "due_at": due_at.isoformat() if due_at else None,
    }
    fingerprint = request_fingerprint(payload)
    existing = (
        db.query(AgentWorkflow)
        .filter(
            AgentWorkflow.user_id == user_id,
            AgentWorkflow.idempotency_key == cleaned_key,
        )
        .first()
    )
    if existing is not None:
        check_idempotent_request(existing, fingerprint)
        return existing

    workflow = AgentWorkflow(
        user_id=user_id,
        person_id=person_id,
        state=INGESTED,
        goal=cleaned_goal,
        request_payload=payload,
        request_fingerprint=fingerprint,
        idempotency_key=cleaned_key,
        trace_id=str(uuid.uuid4()),
    )
    db.add(workflow)
    db.flush()
    db.add(
        WorkflowTransition(
            workflow_id=workflow.id,
            from_state=None,
            to_state=INGESTED,
            actor="user",
            reason="follow-up request accepted",
            trace_id=workflow.trace_id,
        )
    )
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = (
            db.query(AgentWorkflow)
            .filter(
                AgentWorkflow.user_id == user_id,
                AgentWorkflow.idempotency_key == cleaned_key,
            )
            .one()
        )
        check_idempotent_request(existing, fingerprint)
        return existing
    db.refresh(workflow)
    return workflow


def check_idempotent_request(workflow: AgentWorkflow, fingerprint: str) -> None:
    if workflow.request_fingerprint != fingerprint:
        raise WorkflowConflict("Idempotency-Key was already used with another request")


def transition_workflow(
    db: Session,
    workflow: AgentWorkflow,
    next_state: str,
    *,
    actor: str,
    reason: str,
) -> AgentWorkflow:
    check_transition(workflow.state, next_state)
    current = workflow.state
    workflow.state = next_state
    db.add(workflow)
    db.add(
        WorkflowTransition(
            workflow_id=workflow.id,
            from_state=current,
            to_state=next_state,
            actor=actor,
            reason=reason,
            trace_id=workflow.trace_id,
        )
    )
    db.commit()
    db.refresh(workflow)
    return workflow


def mark_failed(
    db: Session,
    workflow: AgentWorkflow,
    error: str,
) -> AgentWorkflow:
    if workflow.state in FINAL_STATES:
        return workflow
    workflow.error = error
    return transition_workflow(
        db,
        workflow,
        FAILED,
        actor="system",
        reason="tool execution failed",
    )


def normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def claim_prepare_workflow(
    db: Session,
    workflow: AgentWorkflow,
) -> str:
    now = datetime.now(timezone.utc)
    if (
        workflow.runner_id
        and workflow.lease_expires_at
        and normalize_utc(workflow.lease_expires_at) > now
    ):
        raise WorkflowConflict("workflow is already being prepared")

    runner_id = str(uuid.uuid4())
    workflow.runner_id = runner_id
    workflow.lease_expires_at = now + timedelta(minutes=PREPARE_LEASE_MINUTES)
    incomplete_calls = (
        db.query(ToolCall)
        .filter(
            ToolCall.workflow_id == workflow.id,
            ToolCall.approval_status == "NOT_REQUIRED",
            ToolCall.executed_at.is_(None),
        )
        .all()
    )
    for row in incomplete_calls:
        row.runner_id = runner_id
        db.add(row)
    db.add(workflow)
    db.commit()
    db.refresh(workflow)
    return runner_id


def release_prepare_workflow(
    db: Session,
    workflow: AgentWorkflow,
    runner_id: str,
) -> AgentWorkflow:
    if workflow.runner_id != runner_id:
        raise WorkflowConflict("workflow prepare lease changed")
    workflow.runner_id = None
    workflow.lease_expires_at = None
    db.add(workflow)
    db.commit()
    db.refresh(workflow)
    return workflow


def build_context(
    db: Session,
    workflow: AgentWorkflow,
    person: Person,
    config: dict[str, Any] | None,
    runner_id: str | None = None,
) -> ToolContext:
    return ToolContext(
        db=db,
        workflow=workflow,
        person=person,
        user_id=workflow.user_id,
        config=dict(config or load_default_config()),
        runner_id=runner_id,
    )


def require_output(row: ToolCall | None) -> dict[str, Any]:
    if row is None:
        raise ToolExecutionError("required tool call is missing")
    if row.error:
        raise ToolExecutionError(row.error)
    if row.output_json is None:
        raise ToolExecutionError(f"tool call has no output: {row.tool_name}")
    return dict(row.output_json)


def prepare_follow_up_workflow(
    db: Session,
    *,
    user_id: UUID,
    workflow_id: UUID,
    config: dict[str, Any] | None = None,
) -> AgentWorkflow:
    workflow = get_user_workflow(db, user_id, workflow_id)
    if workflow is None:
        raise WorkflowNotFound("workflow not found")
    if workflow.state in FINAL_STATES or workflow.state == AWAITING_APPROVAL:
        return workflow

    person = get_user_person(db, user_id, workflow.person_id)
    if person is None:
        raise WorkflowNotFound("person not found")
    runner_id = claim_prepare_workflow(db, workflow)
    context = build_context(db, workflow, person, config, runner_id=runner_id)

    try:
        while workflow.state not in FINAL_STATES | {AWAITING_APPROVAL}:
            if workflow.state == INGESTED:
                transition_workflow(
                    db,
                    workflow,
                    EXTRACTED,
                    actor="agent",
                    reason="request fields normalized",
                )
                continue

            if workflow.state == EXTRACTED:
                transition_workflow(
                    db,
                    workflow,
                    IDENTITY_MATCHED,
                    actor="agent",
                    reason="person ownership matched",
                )
                continue

            if workflow.state == IDENTITY_MATCHED:
                run_tool(
                    context,
                    "search_person_memory",
                    {"query": workflow.goal, "top_k": 5},
                    "memory-search-v1",
                )
                run_tool(
                    context,
                    "get_recent_interactions",
                    {"limit": 5},
                    "recent-interactions-v1",
                )
                run_tool(
                    context,
                    "import_calendar_context",
                    {"days": 7},
                    "calendar-context-v1",
                )
                transition_workflow(
                    db,
                    workflow,
                    MEMORY_REVIEW,
                    actor="agent",
                    reason="person memory and local context retrieved",
                )
                continue

            if workflow.state == MEMORY_REVIEW:
                memory = require_output(
                    get_tool_call(db, workflow.id, "memory-search-v1")
                )
                interactions = require_output(
                    get_tool_call(db, workflow.id, "recent-interactions-v1")
                )
                calendar = require_output(
                    get_tool_call(db, workflow.id, "calendar-context-v1")
                )
                draft_call = run_tool(
                    context,
                    "draft_follow_up",
                    {
                        "goal": workflow.goal,
                        "memory_chunks": memory["chunks"],
                        "recent_interactions": interactions["interactions"],
                        "calendar_context": calendar,
                    },
                    "draft-follow-up-v1",
                )
                draft = require_output(draft_call)["draft_text"]
                workflow.draft_text = str(draft)
                workflow.task_payload = {
                    "title": f"Follow up with {person.name}",
                    "details": str(draft),
                    "due_at": workflow.request_payload.get("due_at"),
                }
                transition_workflow(
                    db,
                    workflow,
                    BRIEFING_OR_DRAFT_READY,
                    actor="agent",
                    reason="grounded follow-up draft prepared",
                )
                continue

            if workflow.state == BRIEFING_OR_DRAFT_READY:
                propose_tool(
                    context,
                    "create_follow_up_task",
                    dict(workflow.task_payload),
                    "create-follow-up-task-v1",
                )
                transition_workflow(
                    db,
                    workflow,
                    AWAITING_APPROVAL,
                    actor="agent",
                    reason="state-changing tool is waiting for human approval",
                )
                continue

            raise ValueError(f"unsupported workflow state: {workflow.state}")
    except ToolExecutionError as exc:
        mark_failed(db, workflow, str(exc))
        return release_prepare_workflow(db, workflow, runner_id)
    except ToolLeaseLostError as exc:
        raise WorkflowConflict(str(exc)) from exc

    return release_prepare_workflow(db, workflow, runner_id)


def decide_follow_up_workflow(
    db: Session,
    *,
    user_id: UUID,
    workflow_id: UUID,
    decision: str,
    expected_version: int,
    config: dict[str, Any] | None = None,
) -> AgentWorkflow:
    workflow = get_user_workflow(db, user_id, workflow_id)
    if workflow is None:
        raise WorkflowNotFound("workflow not found")

    normalized = decision.strip().lower()
    if normalized not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject")
    if workflow.state == EXECUTED and normalized == "approve":
        return workflow
    if workflow.state == REJECTED and normalized == "reject":
        return workflow
    if workflow.state in FINAL_STATES:
        raise WorkflowConflict(f"workflow is already {workflow.state}")
    if workflow.state != AWAITING_APPROVAL:
        raise WorkflowConflict("workflow is not awaiting approval")

    person = get_user_person(db, user_id, workflow.person_id)
    if person is None:
        raise WorkflowNotFound("person not found")
    context = build_context(db, workflow, person, config)
    tool_call = get_tool_call(db, workflow.id, "create-follow-up-task-v1")
    if tool_call is None:
        raise WorkflowConflict("approval-gated tool call is missing")
    if (
        workflow.approval_decision is not None
        and workflow.approval_decision != normalized
    ):
        raise WorkflowConflict(
            f"workflow decision is already {workflow.approval_decision}"
        )

    # repair a delivery that completed its write before the final transition
    if (
        workflow.approval_decision == "approve"
        and tool_call.executed_at is not None
        and tool_call.error is None
    ):
        return transition_workflow(
            db,
            workflow,
            EXECUTED,
            actor="system",
            reason="completed approved tool call reconciled",
        )

    if workflow.approval_decision is None and workflow.version != expected_version:
        raise WorkflowConflict(
            f"workflow version changed: expected {expected_version}, got {workflow.version}"
        )
    else:
        # claim the decision with the workflow version before touching the tool call
        workflow.approval_decision = normalized
        db.add(workflow)
        db.commit()
        db.refresh(workflow)

    if normalized == "reject":
        reject_tool(db, tool_call)
        return transition_workflow(
            db,
            workflow,
            REJECTED,
            actor="user",
            reason="human rejected the proposed task",
        )

    try:
        approve_and_execute_tool(context, tool_call)
    except ToolExecutionError as exc:
        return mark_failed(db, workflow, str(exc))
    return transition_workflow(
        db,
        workflow,
        EXECUTED,
        actor="user",
        reason="human approved and the task was created",
    )


def load_follow_up_task(db: Session, workflow_id: UUID) -> FollowUpTask | None:
    return (
        db.query(FollowUpTask)
        .filter(FollowUpTask.workflow_id == workflow_id)
        .first()
    )
