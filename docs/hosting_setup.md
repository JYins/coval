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
- if Qdrant gives a Python snippet, use the full URL form with `:6333`
- `VECTOR_BACKEND=qdrant` enables the hosted vector-store path

## Phase 1 hosted mode

For the first hosted version, keep this simple:

- `VECTOR_BACKEND=qdrant`
- `EMBEDDING_PROVIDER=mock`
- `LLM_PROVIDER=kimi`
- `LLM_MODEL=kimi-k2.6`
- `KIMI_BASE_URL=https://api.moonshot.cn/v1`
- `KIMI_API_KEY` from Kimi / Moonshot

That gives a stable hosted backend while still exercising the Qdrant integration and a real hosted LLM call. Embeddings stay mock for now to keep the free-first deployment light.

After the Render env value is live, `/health` should show:

```json
{
  "llm_provider": "kimi"
}
```

## Cutover checklist

1. Create Neon project and copy `DATABASE_URL`
2. Create Qdrant Cloud free cluster and copy URL + API key
3. Create Render web service from this repo
4. Fill env vars from `.env.hosted.example`
5. Wait for `/health` to return `status: ok`
6. Change Vercel frontend `NEXT_PUBLIC_API_BASE_URL` to the Render service URL
7. Redeploy frontend and test register, login, person CRUD, upload, ask

## Hosted smoke test

After Render is live, run:

```bash
python scripts/smoke_hosted.py --base-url https://YOUR-RENDER-SERVICE.onrender.com
```

This creates a throwaway account, adds one person, uploads one conversation, runs ask and briefing, rates the generated interaction, and checks the feedback summary.

Current hosted backend:

```bash
python scripts/smoke_hosted.py --base-url https://coval-tb2s.onrender.com
```
