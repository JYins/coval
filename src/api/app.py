"""FastAPI app entrypoint."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes_ask import router as ask_router
from src.api.routes_conversations import router as conversations_router
from src.api.routes_persons import router as persons_router
from src.api.routes_users import router as users_router
from src.models import Base
from src.models.database import engine


def load_cors_origins() -> list[str]:
    value = os.getenv("CORS_ORIGINS", "*").strip()
    if not value or value == "*":
        return ["*"]
    return [item.strip() for item in value.split(",") if item.strip()]


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=load_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users_router)
app.include_router(persons_router)
app.include_router(conversations_router)
app.include_router(ask_router)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "coval backend is starting up"}

