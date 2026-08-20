"""Vercel entrypoint for the FastAPI backend."""

from src.api.app import app

__all__ = ["app"]
