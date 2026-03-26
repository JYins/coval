"""Tests for conversation upload routes."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

import src.api.routes_conversations as routes_conversations
from src.api.app import app
from src.api.deps import get_current_user
from src.models.database import get_db


client = TestClient(app)


def override_db():
    return object()


def override_user():
    return SimpleNamespace(id=uuid4(), email="me@example.com")


def setup_module():
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = override_user


def teardown_module():
    app.dependency_overrides.clear()


def test_upload_manual_conversation(monkeypatch):
    user = override_user()
    person = SimpleNamespace(id=uuid4(), user_id=user.id, name="Alice")
    seen = {}

    monkeypatch.setattr(routes_conversations, "get_user_person", lambda db, user_id, person_id: person)
    monkeypatch.setattr(
        routes_conversations,
        "save_chunks_for_conversation",
        lambda db, conversation, person_name: seen.update({"name": person_name, "conversation_id": conversation.id}),
    )
    monkeypatch.setattr(routes_conversations, "refresh_personality_profile", lambda db, person_row: None)

    def fake_save_conversation(db, person_row, payload):
        return SimpleNamespace(
            id=uuid4(),
            person_id=person_row.id,
            source_type=payload["source_type"],
            raw_content=payload["raw_content"],
            conversation_date=payload["conversation_date"] or datetime.now(timezone.utc),
            language=payload["language"],
        )

    monkeypatch.setattr(routes_conversations, "save_conversation", fake_save_conversation)
    app.dependency_overrides[get_current_user] = lambda: user

    response = client.post(
        "/api/conversations",
        data={
            "person_id": str(person.id),
            "source_type": "manual",
            "language": "en",
            "raw_content": "We talked about hiking",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["source_type"] == "manual"
    assert body["raw_content"] == "We talked about hiking"
    assert seen["name"] == "Alice"
    assert "conversation_id" in seen


def test_upload_txt_file_conversation(monkeypatch):
    user = override_user()
    person = SimpleNamespace(id=uuid4(), user_id=user.id, name="Bob")

    monkeypatch.setattr(routes_conversations, "get_user_person", lambda db, user_id, person_id: person)
    monkeypatch.setattr(routes_conversations, "save_chunks_for_conversation", lambda db, conversation, person_name: None)
    monkeypatch.setattr(routes_conversations, "refresh_personality_profile", lambda db, person_row: None)

    def fake_save_conversation(db, person_row, payload):
        return SimpleNamespace(
            id=uuid4(),
            person_id=person_row.id,
            source_type=payload["source_type"],
            raw_content=payload["raw_content"],
            conversation_date=payload["conversation_date"] or datetime.now(timezone.utc),
            language=payload["language"],
        )

    monkeypatch.setattr(routes_conversations, "save_conversation", fake_save_conversation)
    app.dependency_overrides[get_current_user] = lambda: user

    response = client.post(
        "/api/conversations",
        data={
            "person_id": str(person.id),
            "source_type": "file_upload",
            "language": "en",
        },
        files={"file": ("chat.txt", b"hello from txt", "text/plain")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["source_type"] == "file_upload"
    assert body["raw_content"] == "hello from txt"


def test_upload_voice_not_ready(monkeypatch):
    user = override_user()
    person = SimpleNamespace(id=uuid4(), user_id=user.id, name="Carol")

    monkeypatch.setattr(routes_conversations, "get_user_person", lambda db, user_id, person_id: person)
    monkeypatch.setattr(routes_conversations, "save_chunks_for_conversation", lambda db, conversation, person_name: None)
    monkeypatch.setattr(routes_conversations, "refresh_personality_profile", lambda db, person_row: None)
    app.dependency_overrides[get_current_user] = lambda: user

    response = client.post(
        "/api/conversations",
        data={
            "person_id": str(person.id),
            "source_type": "voice",
            "language": "en",
        },
    )

    assert response.status_code == 501
    assert response.json()["detail"] == "voice ingestion is not built yet"


def test_upload_file_missing_attachment(monkeypatch):
    user = override_user()
    person = SimpleNamespace(id=uuid4(), user_id=user.id, name="David")

    monkeypatch.setattr(routes_conversations, "get_user_person", lambda db, user_id, person_id: person)
    monkeypatch.setattr(routes_conversations, "save_chunks_for_conversation", lambda db, conversation, person_name: None)
    monkeypatch.setattr(routes_conversations, "refresh_personality_profile", lambda db, person_row: None)
    app.dependency_overrides[get_current_user] = lambda: user

    response = client.post(
        "/api/conversations",
        data={
            "person_id": str(person.id),
            "source_type": "file_upload",
            "language": "en",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "file is required for file_upload"

