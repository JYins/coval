"""Conversation model."""

from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import relationship

from src.models.database import Base
from src.models.types import GUID


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    person_id = Column(GUID(), ForeignKey("persons.id"), nullable=False, index=True)
    source_type = Column(String(50), nullable=False)
    raw_content = Column(Text, nullable=False)
    conversation_date = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    language = Column(String(20), nullable=False)

    person = relationship("Person", back_populates="conversations")
    chunks = relationship("Chunk", back_populates="conversation", cascade="all, delete-orphan")

