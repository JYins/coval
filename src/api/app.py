"""FastAPI app entrypoint."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes_ask import router as ask_router
from src.api.routes_conversations import router as conversations_router
from src.api.routes_persons import router as persons_router
from src.api.routes_users import router as users_router
from src.models import Base
from src.models.database import engine
from src.rag.retriever import load_default_config


def load_cors_origins() -> list[str]:
    value = os.getenv("CORS_ORIGINS", "*").strip()
    if not value or value == "*":
        return ["*"]
    return [item.strip() for item in value.split(",") if item.strip()]


def allow_credentials(cors_origins: list[str]) -> bool:
    return cors_origins != ["*"]


def detect_database_driver() -> str:
    database_url = os.getenv("DATABASE_URL", "").strip().lower()
    if database_url.startswith("sqlite"):
        return "sqlite"
    if database_url.startswith("postgres"):
        return "postgres"
    return "unknown"


def load_runtime_mode() -> str:
    return os.getenv("APP_ENV", "local").strip() or "local"


def load_host_name() -> str | None:
    value = os.getenv("QDRANT_URL", "").strip()
    if not value:
        return None
    return urlparse(value).hostname


def load_effective_llm_provider() -> str:
    config = load_default_config()
    llm_config = dict(config.get("llm", {}))
    return str(llm_config.get("provider", "mock"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    # import models here so metadata sees every table
    from src.models.chunk import Chunk  # noqa: F401
    from src.models.conversation import Conversation  # noqa: F401
    from src.models.interaction import Interaction  # noqa: F401
    from src.models.person import Person  # noqa: F401
    from src.models.personality_profile import PersonalityProfile  # noqa: F401
    from src.models.user import User  # noqa: F401

    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Coval API", lifespan=lifespan)
cors_origins = load_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials(cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users_router)
app.include_router(persons_router)
app.include_router(conversations_router)
app.include_router(ask_router)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "coval backend is up"}


@app.get("/health")
def read_health() -> dict[str, object]:
    return {
        "status": "ok",
        "app_env": load_runtime_mode(),
        "database": detect_database_driver(),
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "sentence-transformers").strip(),
        "llm_provider": load_effective_llm_provider(),
        "qdrant_host": load_host_name(),
    }

