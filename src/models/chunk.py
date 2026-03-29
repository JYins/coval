"""Chunk model."""

from __future__ import annotations

import uuid

from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from src.models.database import Base
from src.models.types import GUID


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        GUID(),
        ForeignKey("conversations.id"),
        nullable=False,
        index=True,
    )
    chunk_text = Column(Text, nullable=False)
    person_name_prefix = Column(String(255), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    embedding_model = Column(String(255), nullable=False)

    conversation = relationship("Conversation", back_populates="chunks")

