"""Tests for ask route."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

import src.api.routes_ask as routes_ask
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


def test_ask_question_returns_answer(monkeypatch):
    user = override_user()
    person = SimpleNamespace(id=uuid4(), user_id=user.id, name="Alice")

    monkeypatch.setattr(routes_ask, "get_user_person", lambda db, user_id, person_id: person)
    monkeypatch.setattr(
        routes_ask,
        "run_person_question",
        lambda db, person, question, top_k=None: {
            "answer": "She seems to like calm places and music chats.",
            "retrieved_chunks": [
                {
                    "chunk_id": "chunk-1",
                    "chunk_text": "Alice: She likes jazz bars but not noisy ones.",
                    "chunk_index": 0,
                    "score": 0.91,
                    "rank": 1,
                    "conversation_id": "conv-1",
                }
            ],
        },
    )
    monkeypatch.setattr(
        routes_ask,
        "save_interaction",
        lambda db, person_id, interaction_type, answer: SimpleNamespace(id=uuid4()),
    )
    app.dependency_overrides[get_current_user] = lambda: user

    response = client.post(
        "/api/ask",
        json={
            "person_id": str(person.id),
            "question": "What kind of place should I pick for the next meetup?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["person_id"] == str(person.id)
    assert "calm places" in body["answer"]
    assert len(body["retrieved_chunks"]) == 1
    assert body["retrieved_chunks"][0]["chunk_id"] == "chunk-1"


def test_ask_question_person_not_found(monkeypatch):
    user = override_user()

    monkeypatch.setattr(routes_ask, "get_user_person", lambda db, user_id, person_id: None)
    app.dependency_overrides[get_current_user] = lambda: user

    response = client.post(
        "/api/ask",
        json={
            "person_id": str(uuid4()),
            "question": "Any good reminder before the meeting?",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "person not found"


def test_ask_question_passes_top_k(monkeypatch):
    user = override_user()
    person = SimpleNamespace(id=uuid4(), user_id=user.id, name="Ben")
    seen = {}

    monkeypatch.setattr(routes_ask, "get_user_person", lambda db, user_id, person_id: person)

    def fake_run_person_question(db, person, question, top_k=None):
        seen["top_k"] = top_k
        return {
            "answer": "Keep it short and direct.",
            "retrieved_chunks": [],
        }

    monkeypatch.setattr(routes_ask, "run_person_question", fake_run_person_question)
    monkeypatch.setattr(
        routes_ask,
        "save_interaction",
        lambda db, person_id, interaction_type, answer: SimpleNamespace(id=uuid4()),
    )
    app.dependency_overrides[get_current_user] = lambda: user

    response = client.post(
        "/api/ask",
        json={
            "person_id": str(person.id),
            "question": "How should I talk to Ben?",
            "top_k": 4,
        },
    )

    assert response.status_code == 200
    assert seen["top_k"] == 4


def test_ask_question_returns_bad_request(monkeypatch):
    user = override_user()
    person = SimpleNamespace(id=uuid4(), user_id=user.id, name="Emily")

    monkeypatch.setattr(routes_ask, "get_user_person", lambda db, user_id, person_id: person)
    monkeypatch.setattr(
        routes_ask,
        "run_person_question",
        lambda db, person, question, top_k=None: (_ for _ in ()).throw(
            ValueError("person has no conversations yet")
        ),
    )
    app.dependency_overrides[get_current_user] = lambda: user

    response = client.post(
        "/api/ask",
        json={
            "person_id": str(person.id),
            "question": "Any reminder before the catch-up?",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "person has no conversations yet"

