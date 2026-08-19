"""Typed tools and durable tool-call audit records."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.llm.client import build_llm_client
from src.models.agent_workflow import AgentWorkflow, FollowUpTask, ToolCall
from src.models.conversation import Conversation
from src.models.interaction import Interaction
from src.models.person import Person
from src.rag.retriever import retrieve_chunks


class SearchPersonMemoryParams(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class GetRecentInteractionsParams(BaseModel):
    limit: int = Field(default=5, ge=1, le=20)


class ImportCalendarContextParams(BaseModel):
    days: int = Field(default=7, ge=1, le=30)


class DraftFollowUpParams(BaseModel):
    goal: str = Field(min_length=1)
    memory_chunks: list[dict[str, Any]]
    recent_interactions: list[dict[str, Any]]
    calendar_context: dict[str, Any]


class CreateFollowUpTaskParams(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    details: str = Field(min_length=1)
    due_at: datetime | None = None


@dataclass
class ToolContext:
    db: Session
    workflow: AgentWorkflow
    person: Person
    user_id: UUID
    config: dict[str, Any]
    runner_id: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    params_model: type[BaseModel]
    handler: Callable[[ToolContext, BaseModel], dict[str, Any]]
    requires_approval: bool = False


class ToolExecutionError(RuntimeError):
    """A tool failed after its audit row was persisted."""


class ToolLeaseLostError(RuntimeError):
    """A late worker lost its fencing lease before saving a result."""


def search_person_memory(
    context: ToolContext,
    raw_params: BaseModel,
) -> dict[str, Any]:
    params = SearchPersonMemoryParams.model_validate(raw_params)
    has_conversations = (
        context.db.query(Conversation.id)
        .filter(Conversation.person_id == context.person.id)
        .first()
    )
    if has_conversations is None:
        return {"chunks": [], "source_ids": []}

    config = dict(context.config)
    config["top_k"] = params.top_k
    rows = retrieve_chunks(
        context.db,
        context.person,
        params.query,
        user_id=context.user_id,
        person_id=context.person.id,
        config=config,
    )
    chunks = [
        {
            "chunk_id": str(row["chunk_id"]),
            "chunk_text": str(row["chunk_text"]),
            "score": float(row.get("score", 0.0)),
            "rank": int(row["rank"]),
        }
        for row in rows
    ]
    return {
        "chunks": chunks,
        "source_ids": [row["chunk_id"] for row in chunks],
    }


def get_recent_interactions(
    context: ToolContext,
    raw_params: BaseModel,
) -> dict[str, Any]:
    params = GetRecentInteractionsParams.model_validate(raw_params)
    rows = (
        context.db.query(Interaction)
        .filter(Interaction.person_id == context.person.id)
        .order_by(Interaction.created_at.desc())
        .limit(params.limit)
        .all()
    )
    return {
        "interactions": [
            {
                "id": str(row.id),
                "type": row.interaction_type,
                "advice": row.ai_advice_given,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]
    }


def import_calendar_context(
    context: ToolContext,
    raw_params: BaseModel,
) -> dict[str, Any]:
    params = ImportCalendarContextParams.model_validate(raw_params)
    return {
        "provider": "local_demo",
        "available": False,
        "days": params.days,
        "events": [],
    }


def draft_follow_up(
    context: ToolContext,
    raw_params: BaseModel,
) -> dict[str, Any]:
    params = DraftFollowUpParams.model_validate(raw_params)
    memory_text = "\n".join(
        f"- {row['chunk_text']}" for row in params.memory_chunks
    ) or "- No saved conversation evidence."
    interaction_text = "\n".join(
        f"- {row['advice']}" for row in params.recent_interactions
    ) or "- No prior AI interactions."
    prompt = "\n\n".join(
        [
            f"Draft a concise follow-up plan for {context.person.name}.",
            f"User goal: {params.goal}",
            "Retrieved evidence:",
            memory_text,
            "Recent interaction notes:",
            interaction_text,
            "Calendar context:",
            str(params.calendar_context),
            "Return a practical draft only. Do not claim an external action was completed.",
        ]
    )
    client = build_llm_client(context.config)
    draft = client.generate(
        system_prompt=(
            "You are Coval's follow-up drafting tool. Ground the draft in supplied evidence."
        ),
        user_prompt=prompt,
    )
    return {"draft_text": draft.strip()}


def create_follow_up_task(
    context: ToolContext,
    raw_params: BaseModel,
) -> dict[str, Any]:
    params = CreateFollowUpTaskParams.model_validate(raw_params)
    existing = (
        context.db.query(FollowUpTask)
        .filter(FollowUpTask.workflow_id == context.workflow.id)
        .first()
    )
    if existing is not None:
        return task_result(existing)

    task = FollowUpTask(
        user_id=context.user_id,
        person_id=context.person.id,
        workflow_id=context.workflow.id,
        title=params.title,
        details=params.details,
        due_at=params.due_at,
    )
    context.db.add(task)
    try:
        context.db.commit()
    except IntegrityError:
        # unique workflow_id makes a repeated approved delivery safe
        context.db.rollback()
        task = (
            context.db.query(FollowUpTask)
            .filter(FollowUpTask.workflow_id == context.workflow.id)
            .one()
        )
    context.db.refresh(task)
    return task_result(task)


def task_result(task: FollowUpTask) -> dict[str, Any]:
    return {
        "task_id": str(task.id),
        "status": task.status,
        "title": task.title,
    }


TOOL_REGISTRY = {
    "search_person_memory": ToolSpec(SearchPersonMemoryParams, search_person_memory),
    "get_recent_interactions": ToolSpec(
        GetRecentInteractionsParams,
        get_recent_interactions,
    ),
    "import_calendar_context": ToolSpec(
        ImportCalendarContextParams,
        import_calendar_context,
    ),
    "draft_follow_up": ToolSpec(DraftFollowUpParams, draft_follow_up),
    "create_follow_up_task": ToolSpec(
        CreateFollowUpTaskParams,
        create_follow_up_task,
        requires_approval=True,
    ),
}


def get_tool_call(
    db: Session,
    workflow_id: UUID,
    idempotency_key: str,
) -> ToolCall | None:
    return (
        db.query(ToolCall)
        .filter(
            ToolCall.workflow_id == workflow_id,
            ToolCall.idempotency_key == idempotency_key,
        )
        .first()
    )


def run_tool(
    context: ToolContext,
    tool_name: str,
    params: dict[str, Any],
    idempotency_key: str,
) -> ToolCall:
    spec = TOOL_REGISTRY[tool_name]
    if spec.requires_approval:
        raise ValueError(f"{tool_name} must be proposed before execution")

    existing = get_tool_call(context.db, context.workflow.id, idempotency_key)
    if existing is not None:
        if existing.error:
            raise ToolExecutionError(existing.error)
        if existing.executed_at is None:
            existing.runner_id = context.runner_id
            context.db.add(existing)
            context.db.commit()
            context.db.refresh(existing)
            parsed = spec.params_model.model_validate(existing.input_json)
            return execute_tool_call(context, existing, spec, parsed)
        return existing

    parsed = spec.params_model.model_validate(params)
    row = ToolCall(
        workflow_id=context.workflow.id,
        tool_name=tool_name,
        input_json=parsed.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        approval_status="NOT_REQUIRED",
        runner_id=context.runner_id,
        trace_id=context.workflow.trace_id,
    )
    context.db.add(row)
    context.db.commit()
    context.db.refresh(row)
    return execute_tool_call(context, row, spec, parsed)


def propose_tool(
    context: ToolContext,
    tool_name: str,
    params: dict[str, Any],
    idempotency_key: str,
) -> ToolCall:
    spec = TOOL_REGISTRY[tool_name]
    if not spec.requires_approval:
        raise ValueError(f"{tool_name} does not require approval")

    existing = get_tool_call(context.db, context.workflow.id, idempotency_key)
    if existing is not None:
        return existing

    parsed = spec.params_model.model_validate(params)
    row = ToolCall(
        workflow_id=context.workflow.id,
        tool_name=tool_name,
        input_json=parsed.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        approval_status="PENDING",
        runner_id=context.runner_id,
        trace_id=context.workflow.trace_id,
    )
    context.db.add(row)
    context.db.commit()
    context.db.refresh(row)
    return row


def approve_and_execute_tool(context: ToolContext, row: ToolCall) -> ToolCall:
    if row.approval_status == "REJECTED":
        raise ValueError("rejected tool call cannot be executed")
    if row.executed_at is not None:
        return row
    if row.approval_status == "PENDING":
        row.approval_status = "APPROVED"
        context.db.add(row)
        context.db.commit()
        context.db.refresh(row)

    spec = TOOL_REGISTRY[row.tool_name]
    if not spec.requires_approval:
        raise ValueError(f"tool is not approval-gated: {row.tool_name}")
    parsed = spec.params_model.model_validate(row.input_json)
    return execute_tool_call(context, row, spec, parsed)


def reject_tool(db: Session, row: ToolCall) -> ToolCall:
    if row.executed_at is not None:
        raise ValueError("executed tool call cannot be rejected")
    row.approval_status = "REJECTED"
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def execute_tool_call(
    context: ToolContext,
    row: ToolCall,
    spec: ToolSpec,
    parsed: BaseModel,
) -> ToolCall:
    started = time.perf_counter()
    try:
        output = spec.handler(context, parsed)
    except Exception as exc:
        check_tool_lease(context)
        row.error = f"{type(exc).__name__}: {exc}"
        row.latency_ms = round((time.perf_counter() - started) * 1000)
        row.executed_at = datetime.now(timezone.utc)
        context.db.add(row)
        context.db.commit()
        raise ToolExecutionError(row.error) from exc

    check_tool_lease(context)

    row.output_json = output
    row.latency_ms = round((time.perf_counter() - started) * 1000)
    row.executed_at = datetime.now(timezone.utc)
    context.db.add(row)
    context.db.commit()
    context.db.refresh(row)
    return row


def check_tool_lease(context: ToolContext) -> None:
    if context.runner_id is None:
        return
    current_runner = (
        context.db.query(AgentWorkflow.runner_id)
        .filter(AgentWorkflow.id == context.workflow.id)
        .scalar()
    )
    if current_runner != context.runner_id:
        context.db.rollback()
        raise ToolLeaseLostError("workflow prepare lease was lost")
