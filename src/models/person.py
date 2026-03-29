"""Person model."""

from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import relationship

from src.models.database import Base
from src.models.types import GUID


class Person(Base):
    __tablename__ = "persons"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    relationship_type = Column(String(50), nullable=False)
    notes = Column(Text, nullable=True)
    first_met = Column(DateTime(timezone=True), nullable=True)
    last_contact = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="persons")
    conversations = relationship(
        "Conversation",
        back_populates="person",
        cascade="all, delete-orphan",
    )
    personality_profile = relationship(
        "PersonalityProfile",
        back_populates="person",
        uselist=False,
        cascade="all, delete-orphan",
    )
    interactions = relationship(
        "Interaction",
        back_populates="person",
        cascade="all, delete-orphan",
    )

