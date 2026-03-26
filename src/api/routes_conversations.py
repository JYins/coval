"""Conversation upload routes."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.analysis.personality import refresh_personality_profile
from src.api.deps import get_current_user
from src.ingestion.file_upload import build_file_upload_conversation
from src.ingestion.manual import build_manual_conversation
from src.ingestion.ocr import extract_text_from_image
from src.ingestion.voice import transcribe_voice
from src.models.conversation import Conversation
from src.models.database import get_db
from src.models.person import Person
from src.models.user import User
from src.rag.indexing import save_chunks_for_conversation


router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class ConversationResponse(BaseModel):
    id: UUID
    person_id: UUID
    source_type: str
    raw_content: str
    conversation_date: datetime
    language: str


def build_conversation_response(conversation: Conversation) -> ConversationResponse:
    return ConversationResponse(
        id=conversation.id,
        person_id=conversation.person_id,
        source_type=conversation.source_type,
        raw_content=conversation.raw_content,
        conversation_date=conversation.conversation_date,
        language=conversation.language,
    )


def get_user_person(db: Session, user_id: UUID, person_id: UUID) -> Person | None:
    return (
        db.query(Person)
        .filter(Person.id == person_id, Person.user_id == user_id)
        .first()
    )


def save_conversation(db: Session, person: Person, payload: dict[str, object]) -> Conversation:
    data = {
        "person_id": person.id,
        "source_type": payload["source_type"],
        "raw_content": payload["raw_content"],
        "language": payload["language"],
    }
    if payload.get("conversation_date") is not None:
        data["conversation_date"] = payload["conversation_date"]

    conversation = Conversation(**data)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def upload_conversation(
    person_id: UUID = Form(...),
    source_type: str = Form(...),
    language: str = Form("en"),
    conversation_date: datetime | None = Form(None),
    raw_content: str | None = Form(None),
    file: UploadFile | None = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ConversationResponse:
    person = get_user_person(db, current_user.id, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="person not found")

    kind = source_type.strip().lower()

    try:
        if kind == "manual":
            payload = build_manual_conversation(raw_content or "", language, conversation_date)
        elif kind == "file_upload":
            if file is None:
                raise ValueError("file is required for file_upload")
            payload = build_file_upload_conversation(
                file.filename or "upload.txt",
                await file.read(),
                language,
                conversation_date,
            )
        elif kind == "ocr":
            extract_text_from_image()
            raise ValueError("unreachable")
        elif kind == "voice":
            transcribe_voice()
            raise ValueError("unreachable")
        else:
            raise ValueError("unsupported source_type")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))

    conversation = save_conversation(db, person, payload)
    save_chunks_for_conversation(db, conversation, person.name)
    refresh_personality_profile(db, person)
    return build_conversation_response(conversation)

