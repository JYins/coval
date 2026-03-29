"""Personality profile model."""

from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import relationship

from src.models.database import Base
from src.models.types import GUID


class PersonalityProfile(Base):
    __tablename__ = "personality_profiles"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    person_id = Column(
        GUID(),
        ForeignKey("persons.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    big_five_scores = Column(JSON, nullable=True)
    communication_style = Column(JSON, nullable=True)
    preferences = Column(JSON, nullable=True)
    topics_of_interest = Column(JSON, nullable=True)
    last_updated = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    person = relationship("Person", back_populates="personality_profile")

