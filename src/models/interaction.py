"""Interaction model."""

from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import relationship

from src.models.database import Base
from src.models.types import GUID


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    person_id = Column(GUID(), ForeignKey("persons.id"), nullable=False, index=True)
    interaction_type = Column(String(50), nullable=False)
    ai_advice_given = Column(Text, nullable=False)
    user_rating = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    person = relationship("Person", back_populates="interactions")

