"""Tests for person routes."""

from __future__ import annotations

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

