"""Ask routes for RAG + LLM flow."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.deps import get_current_user
from src.llm.client import build_llm_client
from src.models.database import get_db
from src.models.interaction import Interaction
from src.models.person import Person
from src.models.user import User
from src.rag.retriever import load_default_config, run_retrieval


router = APIRouter(prefix="/api", tags=["ask"])


class AskRequest(BaseModel):
    person_id: UUID
    question: str = Field(min_length=1)
    top_k: int | None = None
    interaction_type: str = "question"


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    chunk_text: str
    chunk_index: int
    score: float
    rank: int
    conversation_id: str | None = None


class AskResponse(BaseModel):
    person_id: UUID
    question: str
    answer: str
    interaction_id: UUID
    retrieved_chunks: list[RetrievedChunkResponse]


def get_user_person(db: Session, user_id: UUID, person_id: UUID) -> Person | None:
    return (
        db.query(Person)
        .filter(Person.id == person_id, Person.user_id == user_id)
        .first()
    )


def save_interaction(
    db: Session,
    person_id: UUID,
    interaction_type: str,
    answer: str,
) -> Interaction:
    row = Interaction(
        person_id=person_id,
        interaction_type=interaction_type,
        ai_advice_given=answer,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def run_person_question(
    db: Session,
    person: Person,
    question: str,
    top_k: int | None = None,
) -> dict[str, object]:
    config = load_default_config()
    if top_k is not None:
        config["top_k"] = top_k

    rag_payload = run_retrieval(db, person, question, config=config)
    client = build_llm_client(config)
    answer = client.generate(
        system_prompt=str(rag_payload["system_prompt"]),
        user_prompt=str(rag_payload["user_prompt"]),
    )
    return {
        "answer": answer,
        "retrieved_chunks": rag_payload["retrieved_chunks"],
    }


def build_chunk_response(row: dict[str, object]) -> RetrievedChunkResponse:
    return RetrievedChunkResponse(
        chunk_id=str(row["chunk_id"]),
        chunk_text=str(row["chunk_text"]),
        chunk_index=int(row["chunk_index"]),
        score=float(row.get("score", 0.0)),
        rank=int(row["rank"]),
        conversation_id=str(row["conversation_id"]) if row.get("conversation_id") else None,
    )


@router.post("/ask", response_model=AskResponse)
def ask_question(
    payload: AskRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AskResponse:
    person = get_user_person(db, current_user.id, payload.person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="person not found")

    try:
        result = run_person_question(
            db=db,
            person=person,
            question=payload.question,
            top_k=payload.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    interaction = save_interaction(
        db=db,
        person_id=person.id,
        interaction_type=payload.interaction_type,
        answer=str(result["answer"]),
    )
    return AskResponse(
        person_id=person.id,
        question=payload.question,
        answer=str(result["answer"]),
        interaction_id=interaction.id,
        retrieved_chunks=[
            build_chunk_response(row) for row in list(result["retrieved_chunks"])
        ],
    )

