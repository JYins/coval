"""Durable state for the follow-up agent."""

from __future__ import annotations

import uuid

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    DateTime,
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


WORKFLOW_STATES = (
    "INGESTED",
    "EXTRACTED",
    "IDENTITY_MATCHED",
    "MEMORY_REVIEW",
    "BRIEFING_OR_DRAFT_READY",
    "AWAITING_APPROVAL",
    "EXECUTED",
    "REJECTED",
    "FAILED",
)


class AgentWorkflow(Base):
    __tablename__ = "agent_workflows"
    __table_args__ = (
        UniqueConstraint("user_id", "idempotency_key", name="uq_workflow_user_key"),
        CheckConstraint(
            f"state IN {WORKFLOW_STATES}",
            name="ck_agent_workflow_state",
        ),
        CheckConstraint(
            "approval_decision IS NULL OR approval_decision IN ('approve', 'reject')",
            name="ck_agent_workflow_approval_decision",
        ),
    )

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False, index=True)
    person_id = Column(GUID(), ForeignKey("persons.id"), nullable=False, index=True)
    workflow_type = Column(String(50), nullable=False, default="follow_up")
    state = Column(String(50), nullable=False, default="INGESTED")
    goal = Column(Text, nullable=False)
    request_payload = Column(JSON, nullable=False, default=dict)
    request_fingerprint = Column(String(64), nullable=False)
    idempotency_key = Column(String(255), nullable=False)
    trace_id = Column(String(36), nullable=False, unique=True, index=True)
    draft_text = Column(Text, nullable=True)
    task_payload = Column(JSON, nullable=True)
    approval_decision = Column(String(20), nullable=True)
    runner_id = Column(String(36), nullable=True)
    lease_expires_at = Column(DateTime(timezone=True), nullable=True)
    error = Column(Text, nullable=True)
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
    transitions = relationship(
        "WorkflowTransition",
        back_populates="workflow",
        cascade="all, delete-orphan",
        order_by="WorkflowTransition.created_at",
    )
    tool_calls = relationship(
        "ToolCall",
        back_populates="workflow",
        cascade="all, delete-orphan",
        order_by="ToolCall.created_at",
    )
    task = relationship(
        "FollowUpTask",
        back_populates="workflow",
        uselist=False,
        cascade="all, delete-orphan",
    )


class WorkflowTransition(Base):
    __tablename__ = "workflow_transitions"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    workflow_id = Column(
        GUID(),
        ForeignKey("agent_workflows.id"),
        nullable=False,
        index=True,
    )
    from_state = Column(String(50), nullable=True)
    to_state = Column(String(50), nullable=False)
    actor = Column(String(50), nullable=False)
    reason = Column(Text, nullable=False)
    trace_id = Column(String(36), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    workflow = relationship("AgentWorkflow", back_populates="transitions")


class ToolCall(Base):
    __tablename__ = "tool_calls"
    __table_args__ = (
        UniqueConstraint(
            "workflow_id",
            "idempotency_key",
            name="uq_tool_call_workflow_key",
        ),
        CheckConstraint(
            "approval_status IN ('NOT_REQUIRED', 'PENDING', 'APPROVED', 'REJECTED')",
            name="ck_tool_call_approval_status",
        ),
    )

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    workflow_id = Column(
        GUID(),
        ForeignKey("agent_workflows.id"),
        nullable=False,
        index=True,
    )
    tool_name = Column(String(100), nullable=False)
    input_json = Column(JSON, nullable=False)
    output_json = Column(JSON, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)
    idempotency_key = Column(String(255), nullable=False)
    actor = Column(String(50), nullable=False, default="agent")
    approval_status = Column(String(20), nullable=False)
    runner_id = Column(String(36), nullable=True)
    version = Column(Integer, nullable=False, default=1)
    trace_id = Column(String(36), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    executed_at = Column(DateTime(timezone=True), nullable=True)

    workflow = relationship("AgentWorkflow", back_populates="tool_calls")

    __mapper_args__ = {"version_id_col": version}


class FollowUpTask(Base):
    __tablename__ = "follow_up_tasks"
    __table_args__ = (
        CheckConstraint(
            "status IN ('OPEN', 'DONE', 'CANCELLED')",
            name="ck_follow_up_task_status",
        ),
    )

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False, index=True)
    person_id = Column(GUID(), ForeignKey("persons.id"), nullable=False, index=True)
    workflow_id = Column(
        GUID(),
        ForeignKey("agent_workflows.id"),
        nullable=False,
        unique=True,
    )
    title = Column(String(255), nullable=False)
    details = Column(Text, nullable=False)
    due_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), nullable=False, default="OPEN")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    workflow = relationship("AgentWorkflow", back_populates="task")
    person = relationship("Person")
    user = relationship("User")
