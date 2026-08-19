"""API routes for durable, human-approved agent workflows."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from src.agent.workflow import (
    WorkflowConflict,
    WorkflowNotFound,
    create_follow_up_workflow,
    decide_follow_up_workflow,
    get_user_workflow,
    load_follow_up_task,
    prepare_follow_up_workflow,
)
from src.api.deps import get_current_user
from src.models.agent_workflow import AgentWorkflow, ToolCall
from src.models.database import get_db
from src.models.user import User


router = APIRouter(prefix="/api/workflows", tags=["workflows"])


class FollowUpWorkflowCreate(BaseModel):
    person_id: UUID
    goal: str = Field(min_length=1, max_length=4000)
    due_at: datetime | None = None


class WorkflowDecision(BaseModel):
    decision: Literal["approve", "reject"]
    expected_version: int = Field(ge=1)


class TransitionResponse(BaseModel):
    from_state: str | None
    to_state: str
    actor: str
    reason: str
    created_at: datetime


class ToolCallResponse(BaseModel):
    id: UUID
    tool_name: str
    input_json: dict
    output_json: dict | None
    approval_status: str
    latency_ms: int | None
    error: str | None
    executed_at: datetime | None


class FollowUpTaskResponse(BaseModel):
    id: UUID
    title: str
    details: str
    due_at: datetime | None
    status: str


class WorkflowResponse(BaseModel):
    id: UUID
    person_id: UUID
    workflow_type: str
    state: str
    goal: str
    version: int
    trace_id: str
    draft_text: str | None
    approval_decision: str | None
    error: str | None
    pending_action: ToolCallResponse | None
    tool_calls: list[ToolCallResponse]
    transitions: list[TransitionResponse]
    task: FollowUpTaskResponse | None
    created_at: datetime
    updated_at: datetime


def build_tool_response(row: ToolCall) -> ToolCallResponse:
    return ToolCallResponse(
        id=row.id,
        tool_name=row.tool_name,
        input_json=dict(row.input_json),
        output_json=dict(row.output_json) if row.output_json is not None else None,
        approval_status=row.approval_status,
        latency_ms=row.latency_ms,
        error=row.error,
        executed_at=row.executed_at,
    )


def build_workflow_response(db: Session, row: AgentWorkflow) -> WorkflowResponse:
    db.refresh(row)
    tool_calls = [build_tool_response(item) for item in row.tool_calls]
    pending_action = next(
        (
            item
            for item in tool_calls
            if item.approval_status in {"PENDING", "APPROVED"}
            and item.executed_at is None
        ),
        None,
    )
    task = load_follow_up_task(db, row.id)
    return WorkflowResponse(
        id=row.id,
        person_id=row.person_id,
        workflow_type=row.workflow_type,
        state=row.state,
        goal=row.goal,
        version=row.version,
        trace_id=row.trace_id,
        draft_text=row.draft_text,
        approval_decision=row.approval_decision,
        error=row.error,
        pending_action=pending_action,
        tool_calls=tool_calls,
        transitions=[
            TransitionResponse(
                from_state=item.from_state,
                to_state=item.to_state,
                actor=item.actor,
                reason=item.reason,
                created_at=item.created_at,
            )
            for item in row.transitions
        ],
        task=(
            FollowUpTaskResponse(
                id=task.id,
                title=task.title,
                details=task.details,
                due_at=task.due_at,
                status=task.status,
            )
            if task is not None
            else None
        ),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def raise_workflow_http_error(exc: Exception) -> None:
    if isinstance(exc, WorkflowNotFound):
        raise HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (WorkflowConflict, StaleDataError)):
        raise HTTPException(status_code=409, detail=str(exc))
    raise HTTPException(status_code=400, detail=str(exc))


@router.post(
    "/follow-up",
    response_model=WorkflowResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_follow_up(
    payload: FollowUpWorkflowCreate,
    idempotency_key: str = Header(
        alias="Idempotency-Key",
        min_length=1,
        max_length=255,
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> WorkflowResponse:
    try:
        workflow = create_follow_up_workflow(
            db,
            user_id=current_user.id,
            person_id=payload.person_id,
            goal=payload.goal,
            due_at=payload.due_at,
            idempotency_key=idempotency_key,
        )
    except (ValueError, IntegrityError, StaleDataError) as exc:
        raise_workflow_http_error(exc)
    return build_workflow_response(db, workflow)


@router.get("/{workflow_id}", response_model=WorkflowResponse)
def get_workflow(
    workflow_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> WorkflowResponse:
    workflow = get_user_workflow(db, current_user.id, workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    return build_workflow_response(db, workflow)


@router.post("/{workflow_id}/prepare", response_model=WorkflowResponse)
def prepare_workflow(
    workflow_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> WorkflowResponse:
    try:
        workflow = prepare_follow_up_workflow(
            db,
            user_id=current_user.id,
            workflow_id=workflow_id,
        )
    except (ValueError, IntegrityError, StaleDataError) as exc:
        raise_workflow_http_error(exc)
    return build_workflow_response(db, workflow)


@router.post("/{workflow_id}/decision", response_model=WorkflowResponse)
def decide_workflow(
    workflow_id: UUID,
    payload: WorkflowDecision,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> WorkflowResponse:
    try:
        workflow = decide_follow_up_workflow(
            db,
            user_id=current_user.id,
            workflow_id=workflow_id,
            decision=payload.decision,
            expected_version=payload.expected_version,
        )
    except (ValueError, IntegrityError, StaleDataError) as exc:
        raise_workflow_http_error(exc)
    return build_workflow_response(db, workflow)
