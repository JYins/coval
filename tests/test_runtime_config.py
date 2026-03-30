"""Tests for hosted runtime helpers."""

from __future__ import annotations

from src.api.app import allow_credentials, app, load_cors_origins
from src.models.database import normalize_database_url


def test_load_cors_origins_reads_list(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com, http://localhost:3000")

    assert load_cors_origins() == [
        "https://app.example.com",
        "http://localhost:3000",
    ]


def test_allow_credentials_disables_wildcard_mode():
    assert allow_credentials(["*"]) is False
    assert allow_credentials(["https://app.example.com"]) is True


def test_normalize_database_url_for_hosted_postgres():
    assert normalize_database_url("postgres://demo") == "postgresql+psycopg2://demo"
    assert normalize_database_url("postgresql://demo") == "postgresql+psycopg2://demo"


def test_health_endpoint_returns_runtime_shape():
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "database" in body
