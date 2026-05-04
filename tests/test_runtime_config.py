"""Tests for hosted runtime helpers."""

from __future__ import annotations

from src.api.app import allow_credentials, app, load_cors_origins
from src.models.database import normalize_database_url
from src.rag.retriever import apply_env_overrides


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


def test_apply_env_overrides_for_hosted_runtime(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "qdrant")
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example.com")
    monkeypatch.setenv("QDRANT_API_KEY", "secret-key")
    monkeypatch.setenv("LLM_PROVIDER", "kimi")
    monkeypatch.setenv("LLM_MODEL", "kimi-k2.6")
    monkeypatch.setenv("KIMI_API_KEY", "kimi-secret")

    config = apply_env_overrides({"vector_store": {}, "llm": {}})

    assert config["vector_backend"] == "qdrant"
    assert config["vector_store"]["url"] == "https://qdrant.example.com"
    assert config["vector_store"]["api_key"] == "secret-key"
    assert config["llm"]["provider"] == "kimi"
    assert config["llm"]["model"] == "kimi-k2.6"
    assert config["llm"]["api_key"] == "kimi-secret"


def test_apply_env_overrides_defaults_kimi_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "kimi")

    config = apply_env_overrides({"llm": {"model": "mock-relationship-v1"}})

    assert config["llm"]["provider"] == "kimi"
    assert config["llm"]["model"] == "kimi-k2.6"


def test_apply_env_overrides_reads_kimi_key_without_provider_change(monkeypatch):
    monkeypatch.setenv("APP_ENV", "hosted")
    monkeypatch.setenv("KIMI_API_KEY", "kimi-secret")

    config = apply_env_overrides({"llm": {"provider": "mock"}})

    assert config["llm"]["provider"] == "kimi"
    assert config["llm"]["model"] == "kimi-k2.6"
    assert config["llm"]["api_key"] == "kimi-secret"
