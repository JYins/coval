"""Tests for person routes."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

import src.api.routes_persons as routes_persons
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


def test_create_person(monkeypatch):
    user = override_user()

    def fake_create_person_row(db, current_user, data):
        return SimpleNamespace(
            id=uuid4(),
            user_id=current_user.id,
            name=data.name,
            relationship_type=data.relationship_type,
            notes=data.notes,
            first_met=data.first_met,
            last_contact=data.last_contact,
        )

    monkeypatch.setattr(routes_persons, "create_person_row", fake_create_person_row)
    app.dependency_overrides[get_current_user] = lambda: user

    response = client.post(
        "/api/persons",
        json={
            "name": "Alice",
            "relationship_type": "friend",
            "notes": "met at school",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Alice"
    assert body["relationship_type"] == "friend"
    assert body["user_id"] == str(user.id)


def test_list_persons(monkeypatch):
    user = override_user()

    def fake_list_user_persons(db, user_id):
        return [
            SimpleNamespace(
                id=uuid4(),
                user_id=user_id,
                name="Alice",
                relationship_type="friend",
                notes=None,
                first_met=None,
                last_contact=None,
            ),
            SimpleNamespace(
                id=uuid4(),
                user_id=user_id,
                name="Bob",
                relationship_type="client",
                notes="follow up later",
                first_met=None,
                last_contact=None,
            ),
        ]

    monkeypatch.setattr(routes_persons, "list_user_persons", fake_list_user_persons)
    app.dependency_overrides[get_current_user] = lambda: user

    response = client.get("/api/persons")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert body[0]["name"] == "Alice"
    assert body[1]["relationship_type"] == "client"


def test_get_person_not_found(monkeypatch):
    user = override_user()

    monkeypatch.setattr(routes_persons, "get_user_person", lambda db, user_id, person_id: None)
    app.dependency_overrides[get_current_user] = lambda: user

    response = client.get(f"/api/persons/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"] == "person not found"


def test_get_person_briefing(monkeypatch):
    user = override_user()
    person = SimpleNamespace(
        id=uuid4(),
        user_id=user.id,
        name="Alice",
        relationship_type="friend",
        notes="met at school",
        first_met=None,
        last_contact=None,
    )

    monkeypatch.setattr(routes_persons, "get_user_person", lambda db, user_id, person_id: person)
    monkeypatch.setattr(
        routes_persons,
        "generate_person_briefing",
        lambda db, person, top_k=None: {
            "briefing": "Start with music and keep the tone relaxed.",
            "retrieved_chunks": [
                {
                    "chunk_id": "chunk-1",
                    "chunk_text": "Alice: She likes jazz and quiet bars.",
                    "score": 0.88,
                    "rank": 1,
                }
            ],
        },
    )
    app.dependency_overrides[get_current_user] = lambda: user

    response = client.get(f"/api/persons/{person.id}/briefing")

    assert response.status_code == 200
    body = response.json()
    assert body["person_id"] == str(person.id)
    assert "music" in body["briefing"]
    assert body["retrieved_chunks"][0]["chunk_id"] == "chunk-1"


def test_get_person_interactions(monkeypatch):
    user = override_user()
    person = SimpleNamespace(
        id=uuid4(),
        user_id=user.id,
        name="Alice",
        relationship_type="friend",
        notes="met at school",
        first_met=None,
        last_contact=None,
    )

    monkeypatch.setattr(routes_persons, "get_user_person", lambda db, user_id, person_id: person)
    monkeypatch.setattr(
        routes_persons,
        "list_person_interactions",
        lambda db, person_id, limit=20: [
            SimpleNamespace(
                id=uuid4(),
                person_id=person_id,
                interaction_type="question",
                ai_advice_given="Ask about music first.",
                user_rating=4,
                created_at=datetime.now(timezone.utc),
            ),
            SimpleNamespace(
                id=uuid4(),
                person_id=person_id,
                interaction_type="briefing",
                ai_advice_given="Keep the tone relaxed.",
                user_rating=None,
                created_at=datetime.now(timezone.utc),
            ),
        ],
    )
    app.dependency_overrides[get_current_user] = lambda: user

    response = client.get(f"/api/persons/{person.id}/interactions?limit=2")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert body[0]["interaction_type"] == "question"
    assert body[1]["interaction_type"] == "briefing"


def test_get_person_interactions_bad_limit(monkeypatch):
    user = override_user()
    person = SimpleNamespace(
        id=uuid4(),
        user_id=user.id,
        name="Bob",
        relationship_type="client",
        notes=None,
        first_met=None,
        last_contact=None,
    )

    monkeypatch.setattr(routes_persons, "get_user_person", lambda db, user_id, person_id: person)
    monkeypatch.setattr(
        routes_persons,
        "list_person_interactions",
        lambda db, person_id, limit=20: (_ for _ in ()).throw(ValueError("limit should be > 0")),
    )
    app.dependency_overrides[get_current_user] = lambda: user

    response = client.get(f"/api/persons/{person.id}/interactions?limit=0")

    assert response.status_code == 400
    assert response.json()["detail"] == "limit should be > 0"


def test_rate_person_interaction(monkeypatch):
    user = override_user()
    person = SimpleNamespace(
        id=uuid4(),
        user_id=user.id,
        name="Alice",
        relationship_type="friend",
        notes=None,
        first_met=None,
        last_contact=None,
    )
    interaction = SimpleNamespace(
        id=uuid4(),
        person_id=person.id,
        interaction_type="question",
        ai_advice_given="Ask about music first.",
        user_rating=None,
        created_at=datetime.now(timezone.utc),
    )

    monkeypatch.setattr(routes_persons, "get_user_person", lambda db, user_id, person_id: person)
    monkeypatch.setattr(routes_persons, "get_person_interaction", lambda db, person_id, interaction_id: interaction)
    monkeypatch.setattr(
        routes_persons,
        "update_interaction_rating",
        lambda db, interaction_row, user_rating: SimpleNamespace(
            id=interaction_row.id,
            person_id=interaction_row.person_id,
            interaction_type=interaction_row.interaction_type,
            ai_advice_given=interaction_row.ai_advice_given,
            user_rating=user_rating,
            created_at=interaction_row.created_at,
        ),
    )
    app.dependency_overrides[get_current_user] = lambda: user

    response = client.patch(
        f"/api/persons/{person.id}/interactions/{interaction.id}/rating",
        json={"user_rating": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(interaction.id)
    assert body["user_rating"] == 5


def test_rate_person_interaction_bad_rating(monkeypatch):
    user = override_user()
    person = SimpleNamespace(
        id=uuid4(),
        user_id=user.id,
        name="Bob",
        relationship_type="client",
        notes=None,
        first_met=None,
        last_contact=None,
    )
    interaction = SimpleNamespace(
        id=uuid4(),
        person_id=person.id,
        interaction_type="briefing",
        ai_advice_given="Keep it short and clear.",
        user_rating=None,
        created_at=datetime.now(timezone.utc),
    )

    monkeypatch.setattr(routes_persons, "get_user_person", lambda db, user_id, person_id: person)
    monkeypatch.setattr(routes_persons, "get_person_interaction", lambda db, person_id, interaction_id: interaction)
    monkeypatch.setattr(
        routes_persons,
        "update_interaction_rating",
        lambda db, interaction_row, user_rating: (_ for _ in ()).throw(
            ValueError("user_rating should be between 1 and 5")
        ),
    )
    app.dependency_overrides[get_current_user] = lambda: user

    response = client.patch(
        f"/api/persons/{person.id}/interactions/{interaction.id}/rating",
        json={"user_rating": 9},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "user_rating should be between 1 and 5"

