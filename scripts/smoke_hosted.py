"""Smoke test a hosted Coval API end to end."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import httpx


DEFAULT_PASSWORD = "smoke-test-password"


def build_email() -> str:
    return os.getenv("COVAL_SMOKE_EMAIL") or f"smoke+{int(time.time())}@coval.local"


def clean_base_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if not url:
        raise ValueError("base url is required")
    return url


def request_json(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    token: str | None = None,
    **kwargs: Any,
) -> Any:
    headers = dict(kwargs.pop("headers", {}))
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = client.request(method, path, headers=headers, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(f"{method} {path} failed: {response.status_code} {response.text}")
    return response.json()


def register_or_login(client: httpx.Client, email: str, password: str) -> str:
    payload = {"email": email, "password": password}
    response = client.post("/api/users/register", json=payload)
    if response.status_code not in {201, 400}:
        raise RuntimeError(f"register failed: {response.status_code} {response.text}")

    token_payload = request_json(client, "POST", "/api/users/login", json=payload)
    return str(token_payload["access_token"])


def run_smoke(base_url: str, email: str, password: str) -> dict[str, Any]:
    with httpx.Client(base_url=clean_base_url(base_url), timeout=30.0) as client:
        health = request_json(client, "GET", "/health")
        token = register_or_login(client, email, password)

        person = request_json(
            client,
            "POST",
            "/api/persons",
            token=token,
            json={
                "name": "Mina Chen",
                "relationship_type": "networking",
                "notes": "Met through a small founder dinner.",
            },
        )
        person_id = person["id"]

        request_json(client, "GET", "/api/persons", token=token)
        request_json(
            client,
            "POST",
            "/api/conversations",
            token=token,
            data={
                "person_id": person_id,
                "source_type": "manual",
                "language": "en",
                "raw_content": (
                    "Mina prefers direct updates, likes tea more than coffee, "
                    "and mentioned she is exploring AI tools for small teams."
                ),
            },
        )

        answer = request_json(
            client,
            "POST",
            "/api/ask",
            token=token,
            json={
                "person_id": person_id,
                "question": "What should I remember before following up with Mina?",
                "top_k": 3,
            },
        )
        briefing = request_json(
            client,
            "GET",
            f"/api/persons/{person_id}/briefing?top_k=3",
            token=token,
        )
        interaction_id = answer["interaction_id"]
        request_json(
            client,
            "PATCH",
            f"/api/persons/{person_id}/interactions/{interaction_id}/rating",
            token=token,
            json={"user_rating": 5},
        )
        summary = request_json(
            client,
            "GET",
            f"/api/persons/{person_id}/interactions/summary",
            token=token,
        )

    return {
        "health": health,
        "person_id": person_id,
        "ask_chunks": len(answer["retrieved_chunks"]),
        "briefing_chunks": len(briefing["retrieved_chunks"]),
        "rated_interactions": summary["rated_interactions"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test a hosted Coval API.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("COVAL_API_BASE_URL"),
        help="Hosted API base URL, for example https://coval-api.onrender.com",
    )
    parser.add_argument("--email", default=build_email())
    parser.add_argument(
        "--password",
        default=os.getenv("COVAL_SMOKE_PASSWORD", DEFAULT_PASSWORD),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base_url:
        raise SystemExit("Set COVAL_API_BASE_URL or pass --base-url.")

    result = run_smoke(args.base_url, args.email, args.password)
    print("hosted smoke test passed")
    print(f"health: {result['health']}")
    print(f"person_id: {result['person_id']}")
    print(f"ask_chunks: {result['ask_chunks']}")
    print(f"briefing_chunks: {result['briefing_chunks']}")
    print(f"rated_interactions: {result['rated_interactions']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"hosted smoke test failed: {exc}", file=sys.stderr)
        raise
