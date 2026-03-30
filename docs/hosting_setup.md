# Hosted Backend Setup

This file is the practical follow-up to the current Vercel demo backend.

The free-first target stack is:

- frontend on Vercel
- backend API on Render free web service
- PostgreSQL on Neon free
- vector store on Qdrant Cloud free

## Why this split

- Render is a better fit than Vercel for the main FastAPI backend here
- Neon gives a real hosted Postgres path without the short-lived free DB issue
- Qdrant Cloud keeps the vector-store story aligned with the repo design

## Render service

Use the repo root and the included `render.yaml`.

- build command: `pip install -r requirements-hosted.txt`
- start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- health check: `/health`

## Required env vars

Copy from `.env.hosted.example`.

- `DATABASE_URL` should come from Neon and include `sslmode=require`
- `SECRET_KEY` should be changed from the example value
- `CORS_ORIGINS` should include the live Vercel frontend plus localhost
- `QDRANT_URL` and `QDRANT_API_KEY` should come from Qdrant Cloud

## Phase 1 hosted mode

For the first hosted version, keep this simple:

- `EMBEDDING_PROVIDER=mock`
- `LLM_PROVIDER=mock`

That gives a more stable hosted backend first. Real model providers can come later after the storage and API path are stable.

## Cutover checklist

1. Create Neon project and copy `DATABASE_URL`
2. Create Qdrant Cloud free cluster and copy URL + API key
3. Create Render web service from this repo
4. Fill env vars from `.env.hosted.example`
5. Wait for `/health` to return `status: ok`
6. Change Vercel frontend `NEXT_PUBLIC_API_BASE_URL` to the Render service URL
7. Redeploy frontend and test register, login, person CRUD, upload, ask
