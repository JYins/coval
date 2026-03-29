"""Person routes."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.analysis.briefing import generate_person_briefing
from src.api.deps import get_current_user
from src.models.database import get_db
from src.models.interaction import Interaction
from src.models.person import Person
from src.models.user import User


router = APIRouter(prefix="/api/persons", tags=["persons"])


class PersonCreate(BaseModel):
    name: str
    relationship_type: str
    notes: str | None = None
    first_met: datetime | None = None
    last_contact: datetime | None = None


class PersonResponse(BaseModel):
    id: UUID
    user_id: UUID
    name: str
    relationship_type: str
    notes: str | None = None
    first_met: datetime | None = None
    last_contact: datetime | None = None


class BriefingChunkResponse(BaseModel):
    chunk_id: str
    chunk_text: str
    score: float
    rank: int


class BriefingResponse(BaseModel):
    person_id: UUID
    briefing: str
    retrieved_chunks: list[BriefingChunkResponse]


class InteractionResponse(BaseModel):
    id: UUID
    person_id: UUID
    interaction_type: str
    ai_advice_given: str
    user_rating: int | None = None
    created_at: datetime


class InteractionRatingUpdate(BaseModel):
    user_rating: int


class InteractionSummaryResponse(BaseModel):
    person_id: UUID
    total_interactions: int
    rated_interactions: int
    average_rating: float | None = None
    rating_counts: dict[str, int]
    interaction_type_counts: dict[str, int]


def normalize_person_name(name: str) -> str:
    value = name.strip()
    if not value:
        raise HTTPException(status_code=400, detail="name is required")
    return value


def normalize_relationship_type(value: str) -> str:
    cleaned = value.strip().lower()
    if not cleaned:
        raise HTTPException(status_code=400, detail="relationship_type is required")
    return cleaned


def build_person_response(person: Person) -> PersonResponse:
    return PersonResponse(
        id=person.id,
        user_id=person.user_id,
        name=person.name,
        relationship_type=person.relationship_type,
        notes=person.notes,
        first_met=person.first_met,
        last_contact=person.last_contact,
    )


def create_person_row(db: Session, user: User, data: PersonCreate) -> Person:
    person = Person(
        user_id=user.id,
        name=normalize_person_name(data.name),
        relationship_type=normalize_relationship_type(data.relationship_type),
        notes=data.notes,
        first_met=data.first_met,
        last_contact=data.last_contact,
    )
    db.add(person)
    db.commit()
    db.refresh(person)
    return person


def list_user_persons(db: Session, user_id: UUID) -> list[Person]:
    return db.query(Person).filter(Person.user_id == user_id).order_by(Person.name.asc()).all()


def get_user_person(db: Session, user_id: UUID, person_id: UUID) -> Person | None:
    return (
        db.query(Person)
        .filter(Person.id == person_id, Person.user_id == user_id)
        .first()
    )


def list_person_interactions(
    db: Session,
    person_id: UUID,
    limit: int = 20,
) -> list[Interaction]:
    if limit <= 0:
        raise ValueError("limit should be > 0")
    return (
        db.query(Interaction)
        .filter(Interaction.person_id == person_id)
        .order_by(Interaction.created_at.desc())
        .limit(limit)
        .all()
    )


def build_interaction_response(row: Interaction) -> InteractionResponse:
    return InteractionResponse(
        id=row.id,
        person_id=row.person_id,
        interaction_type=row.interaction_type,
        ai_advice_given=row.ai_advice_given,
        user_rating=row.user_rating,
        created_at=row.created_at,
    )


def get_person_interaction(
    db: Session,
    person_id: UUID,
    interaction_id: UUID,
) -> Interaction | None:
    return (
        db.query(Interaction)
        .filter(Interaction.id == interaction_id, Interaction.person_id == person_id)
        .first()
    )


def update_interaction_rating(
    db: Session,
    interaction: Interaction,
    user_rating: int,
) -> Interaction:
    if user_rating < 1 or user_rating > 5:
        raise ValueError("user_rating should be between 1 and 5")
    interaction.user_rating = user_rating
    db.add(interaction)
    db.commit()
    db.refresh(interaction)
    return interaction


def summarize_interactions(
    person_id: UUID,
    rows: list[Interaction],
) -> InteractionSummaryResponse:
    rating_counts = {str(number): 0 for number in range(1, 6)}
    interaction_type_counts: dict[str, int] = {}
    ratings = []

    for row in rows:
        interaction_type_counts[row.interaction_type] = (
            interaction_type_counts.get(row.interaction_type, 0) + 1
        )
        if row.user_rating is not None:
            rating_counts[str(row.user_rating)] += 1
            ratings.append(row.user_rating)

    average_rating = None
    if ratings:
        average_rating = round(sum(ratings) / len(ratings), 2)

    return InteractionSummaryResponse(
        person_id=person_id,
        total_interactions=len(rows),
        rated_interactions=len(ratings),
        average_rating=average_rating,
        rating_counts=rating_counts,
        interaction_type_counts=interaction_type_counts,
    )


@router.post("", response_model=PersonResponse, status_code=status.HTTP_201_CREATED)
def create_person(
    data: PersonCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PersonResponse:
    person = create_person_row(db, current_user, data)
    return build_person_response(person)


@router.get("", response_model=list[PersonResponse])
def list_persons(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[PersonResponse]:
    persons = list_user_persons(db, current_user.id)
    return [build_person_response(person) for person in persons]


@router.get("/{person_id}", response_model=PersonResponse)
def get_person(
    person_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PersonResponse:
    person = get_user_person(db, current_user.id, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="person not found")
    return build_person_response(person)


@router.get("/{person_id}/briefing", response_model=BriefingResponse)
def get_person_briefing(
    person_id: UUID,
    top_k: int | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> BriefingResponse:
    person = get_user_person(db, current_user.id, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="person not found")

    try:
        result = generate_person_briefing(db, person, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return BriefingResponse(
        person_id=person.id,
        briefing=str(result["briefing"]),
        retrieved_chunks=[
            BriefingChunkResponse(
                chunk_id=str(row["chunk_id"]),
                chunk_text=str(row["chunk_text"]),
                score=float(row.get("score", 0.0)),
                rank=int(row["rank"]),
            )
            for row in list(result["retrieved_chunks"])
        ],
    )


@router.get("/{person_id}/interactions", response_model=list[InteractionResponse])
def get_person_interactions(
    person_id: UUID,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[InteractionResponse]:
    person = get_user_person(db, current_user.id, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="person not found")

    try:
        rows = list_person_interactions(db, person.id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return [build_interaction_response(row) for row in rows]


@router.get(
    "/{person_id}/interactions/summary",
    response_model=InteractionSummaryResponse,
)
def get_person_interaction_summary(
    person_id: UUID,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> InteractionSummaryResponse:
    person = get_user_person(db, current_user.id, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="person not found")

    try:
        rows = list_person_interactions(db, person.id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return summarize_interactions(person.id, rows)


@router.patch(
    "/{person_id}/interactions/{interaction_id}/rating",
    response_model=InteractionResponse,
)
def rate_person_interaction(
    person_id: UUID,
    interaction_id: UUID,
    payload: InteractionRatingUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> InteractionResponse:
    person = get_user_person(db, current_user.id, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="person not found")

    interaction = get_person_interaction(db, person.id, interaction_id)
    if interaction is None:
        raise HTTPException(status_code=404, detail="interaction not found")

    try:
        row = update_interaction_rating(db, interaction, payload.user_rating)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return build_interaction_response(row)

