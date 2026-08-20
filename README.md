# Coval

[![CI](https://github.com/JYins/coval/actions/workflows/ci.yml/badge.svg)](https://github.com/JYins/coval/actions/workflows/ci.yml)

Coval is an AI-powered relationship memory backend I am building step by step. The idea is a bit unusual, I know, but I think there is a real use case here for dating, sales, and active networking: if we can retrieve the right personal context at the right time, maybe we can show up a little better in real conversations. This repo is still backend-first at the core, but it now also includes a premium frontend demo so the product shape is easier to see.

## Frontend Demo

- Live UI demo: [https://web-tau-lake-89.vercel.app](https://web-tau-lake-89.vercel.app)
- Live hosted backend API: [https://coval-tb2s.onrender.com](https://coval-tb2s.onrender.com)
- Hosted health check: [https://coval-tb2s.onrender.com/health](https://coval-tb2s.onrender.com/health)
- Current status: hosted demo stack with durable PostgreSQL storage
- Important note: the hosted backend now runs on Render with Neon Postgres and Qdrant Cloud. Kimi K2.6 support is wired through the domestic Moonshot API endpoint; the live service will report `llm_provider: kimi` on `/health` once `KIMI_API_KEY` is present in Render. Embeddings stay in mock mode for the free-first demo path.

![Coval live demo screenshot](docs/images/coval-vercel-home.png)

## What Is Live Today

- register and login flow works on the hosted demo
- dashboard and person creation flow work
- the product shape, visual direction, and API surface are all visible online
- the frontend is deployed on Vercel and points to the Render backend
- PostgreSQL persistence is backed by Neon
- Qdrant Cloud is wired into the hosted retrieval path with mock embeddings
- hosted smoke test passes across register, login, person CRUD, upload, ask, briefing, rating, and summary
- Kimi K2.6 provider support is implemented through the OpenAI-compatible Moonshot API client, using `https://api.moonshot.cn/v1`

## Why This Repo Matters

This is the application layer of my earlier `rag-eval-pipeline` work.

- `rag-eval-pipeline`: figure out what actually improves retrieval
- `coval`: apply those findings to a product with real user flows

That is the portfolio narrative I wanted to make visible. First I benchmarked retrieval ideas, then I turned the good ones into a usable product backend.

## Why I Built This

I built this after spending time on `rag-eval-pipeline`, where I benchmarked chunking strategies, embedding models, and retrieval setups to understand what actually improves retrieval quality. That project taught me a very practical lesson: if retrieval is weak, the LLM cannot really save the answer.

So Coval is the next layer up. Instead of stopping at eval, I wanted to apply those retrieval lessons to a product shape that feels more personal and more concrete:

- ingest conversation notes
- organize them around a person
- retrieve the right context later
- generate grounded advice or a quick briefing before a meeting

The name comes from `covalent bond`. In chemistry, a covalent bond is about shared electrons. Here the metaphor is shared context: two people build a relationship through shared memories, shared details, shared information. A little nerdy maybe, but anyways I still think it fits.

## What It Does

- creates users and person profiles
- ingests conversation data from manual text or `txt/csv` upload
- chunks conversation text with person-name-aware prefixing
- runs dense retrieval over conversation chunks
- assembles prompts for Q&A and briefing generation
- stores lightweight personality profiles and communication-style summaries
- logs Q&A and briefing history per person
- supports simple 1-5 feedback on generated interactions
- exposes a small feedback summary per person
- runs a small retrieval eval set with `Recall@K` and `MRR`
- runs a durable follow-up workflow with typed tools and explicit state transitions
- pauses before creating a follow-up task and supports human approve/reject decisions
- records tool inputs, outputs, latency, errors, approval status, and one trace ID per run
- runs a reviewed Voice G0 pipeline with typed speaker turns, transcript alternatives, revisions, and memory candidates
- writes only human-approved Voice candidates into CRM conversations and retrieval chunks
- validates pinned Voice model/data licenses and scores normalized local baseline outputs

## Architecture Overview

The product loop is simple:

1. user data goes into structured storage and retrieval-friendly chunks
2. RAG retrieves relevant context and assembles a prompt
3. the LLM answers with that supplied context, not with hidden memory

Storage split:

- PostgreSQL stores users, persons, conversations, chunks, personality profiles, and interaction logs
- Qdrant is the vector-store target for chunk embeddings
- current local default config uses in-memory dense search for easier development, but the Qdrant wrapper is already in the repo

The follow-up agent is an explicit state machine:

```text
INGESTED -> EXTRACTED -> IDENTITY_MATCHED -> MEMORY_REVIEW
         -> BRIEFING_OR_DRAFT_READY -> AWAITING_APPROVAL
         -> EXECUTED | REJECTED | FAILED
```

Read tools can search person memory, load recent interactions, and inspect a deterministic local calendar stub. The only write tool creates an internal follow-up task, and it cannot run before approval. Workflow requests and task writes are idempotent, decisions use optimistic version checks, and every transition/tool call is stored for inspection. This is implemented locally; it does not claim to send email or modify an external calendar.

Qdrant payloads and queries are filtered by both `user_id` and `person_id`. Querying an existing index no longer recreates or rewrites the collection.

Voice uses a separate review state path because transcript uncertainty is different from Agent tool execution:

```text
audio request -> TRANSCRIBING -> REVIEW_READY
              -> revise / approve / reject
              -> ApprovedMemoryEvent -> Conversation + chunks
              -> CANCELED (only before any review decision)
```

The current provider is a deterministic synthetic fake for G0. It proves retention, review, idempotency, tenant isolation, and the approved-memory integration without pretending that ASR/diarization quality has been measured.

Hosted deployment target:

- `Vercel` keeps the frontend
- `Render` runs the FastAPI backend
- `Neon` holds the hosted PostgreSQL database
- `Qdrant Cloud` stores vectors when hosted semantic search is enabled

## Connection to RAG Eval Pipeline

| RAG Eval module | Coval module | Transfer |
|---|---|---|
| `cleaning.py` | `src/rag/cleaning.py` | adapted for conversation cleanup |
| `chunking.py` | `src/rag/chunking.py` | reused and adjusted for person-level chunking |
| `eval_metrics.py` | `src/rag/eval_metrics.py` | reused for chunk-level retrieval metrics |
| dense retrieval flow | `src/rag/retriever.py` | simplified product-side version |
| sermon title-aware insight | person-name prefix on chunks | key idea transfer |

The sermon experiments in `rag-eval-pipeline` showed me that title-aware chunking helps retrieval a lot. In this product, the "title" is basically the person's name.

## Project Structure

```text
coval/
|-- README.md
|-- requirements.txt
|-- .github/workflows/ci.yml
|-- configs/
|   |-- default.yaml
|   |-- eval.yaml
|   `-- prompts/
|-- src/
|   |-- api/
|   |-- analysis/
|   |-- ingestion/
|   |-- llm/
|   |-- models/
|   |-- agent/
|   |-- voice/
|   `-- rag/
|-- scripts/
|   |-- init_db.py
|   |-- run_eval.py
|   `-- seed_data.py
|-- tests/
|-- data/eval/
|-- results/
`-- docs/
```

## Quick Start

Local setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/init_db.py
python scripts/seed_data.py
uvicorn src.api.app:app --reload
```

Run retrieval eval:

```bash
python scripts/run_eval.py --config configs/eval.yaml
```

Smoke test a hosted API:

```bash
python scripts/smoke_hosted.py --base-url https://coval-tb2s.onrender.com
```

Latest hosted smoke test passed across register, login, person creation, conversation upload, ask, briefing, rating, and feedback summary.

Notes:

- `scripts/init_db.py` expects PostgreSQL from `DATABASE_URL`
- the current default retrieval backend in `configs/default.yaml` is `memory`
- switch to Qdrant by changing config and running a local Qdrant instance
- for hosted backend deployment, use `requirements-hosted.txt`, `.env.hosted.example`, and `render.yaml`

Frontend local run:

```bash
cd web
npm install
npm run dev
```

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/users/register` | create account |
| `POST` | `/api/users/login` | get JWT token |
| `POST` | `/api/persons` | create a person profile |
| `GET` | `/api/persons` | list persons for current user |
| `GET` | `/api/persons/{person_id}` | get person detail |
| `GET` | `/api/persons/{person_id}/briefing` | generate pre-meeting briefing |
| `GET` | `/api/persons/{person_id}/interactions` | inspect recent Q&A and briefing history |
| `GET` | `/api/persons/{person_id}/interactions/summary` | inspect rating summary for recent interactions |
| `PATCH` | `/api/persons/{person_id}/interactions/{interaction_id}/rating` | rate one generated interaction |
| `POST` | `/api/conversations` | upload manual or file-based conversation |
| `POST` | `/api/ask` | ask a question about a person with RAG |
| `POST` | `/api/workflows/follow-up` | create an idempotent follow-up workflow |
| `GET` | `/api/workflows/{workflow_id}` | inspect state, trace, tools, and result |
| `POST` | `/api/workflows/{workflow_id}/prepare` | retrieve context, draft, and pause before the write tool |
| `POST` | `/api/workflows/{workflow_id}/decision` | approve or reject the pending write tool |
| `POST` | `/api/voice/jobs` | process request-scoped audio with the Voice G0 fake provider |
| `GET` | `/api/voice/jobs/{job_id}` | inspect segments, turns, alternatives, revisions, and candidates |
| `POST` | `/api/voice/jobs/{job_id}/cancel` | cancel an unreviewed job and stale its pending candidates |
| `POST` | `/api/voice/jobs/{job_id}/turns/{turn_id}/revisions` | append a human transcript correction |
| `POST` | `/api/voice/jobs/{job_id}/candidates/{candidate_id}/decision` | approve, reject, or edit a candidate before CRM memory write |

## Configuration

Two YAML files matter right now:

- `configs/default.yaml`: main backend settings for chunking, embedding model, vector backend, and LLM provider
- `configs/eval.yaml`: eval dataset path, metric outputs, chunking config, and top-k settings

The backend is intentionally config-light for now. I wanted the pipeline logic to stay easy to explain in an interview before adding too many toggles.

Hosted env notes:

- `APP_ENV=hosted` marks the Render/Neon path
- `EMBEDDING_PROVIDER=mock` keeps hosted embeddings cheap for now
- `LLM_PROVIDER=kimi`, `LLM_MODEL=kimi-k2.6`, `KIMI_BASE_URL=https://api.moonshot.cn/v1`, and `KIMI_API_KEY` enable the live LLM path
- `CORS_ORIGINS` should be set to the live Vercel frontend plus localhost

Current hosted backend stack:

- `Vercel frontend`: `https://web-tau-lake-89.vercel.app`
- `Render API`: `https://coval-tb2s.onrender.com`
- `Neon Postgres`: durable relational storage for the 6-table schema
- `Qdrant Cloud`: hosted vector store target for chunk embeddings
- `Kimi K2.6`: real LLM provider is implemented through the domestic Moonshot API endpoint; Render needs a valid `KIMI_API_KEY` env value for this to become live

## Database Schema

| Table | Purpose |
|---|---|
| `users` | account records and auth identity |
| `persons` | one row per tracked relationship |
| `conversations` | raw conversation inputs with source and language |
| `chunks` | retrieval-ready text segments |
| `personality_profiles` | lightweight structured personality summary |
| `interactions` | Q&A / briefing history and future feedback hooks |
| `agent_workflows` | durable workflow state, request fingerprint, trace, and decision |
| `workflow_transitions` | append-only state transition history |
| `tool_calls` | typed tool inputs, outputs, approval status, latency, and errors |
| `follow_up_tasks` | approved internal CRM follow-up tasks |
| `voice_ingestion_jobs` | provider, retention, audio hash, state, and tenant context |
| `audio_segments` | timestamped logical audio ranges without stored audio bytes |
| `speaker_turns` | anonymous provider speaker labels and diarization uncertainty |
| `transcript_alternatives` | immutable ranked transcript hypotheses |
| `transcript_revisions` | append-only provider and human transcript revisions |
| `extracted_candidates` | versioned review candidates with inherited uncertainty |
| `review_decisions` | idempotent final approve/reject decisions |
| `approved_memory_events` | approved candidate to CRM conversation/index linkage |

## Evaluation

Current sample eval results from `results/eval_metrics.csv`:

- `MRR = 1.0`
- `Recall@1 = 0.8`
- `Recall@3 = 1.0`
- `Recall@5 = 1.0`

This eval is small on purpose right now. It is mainly there to prove that the retrieval layer is being checked, not just assumed.

Artifacts:

- `results/eval_metrics.csv`
- `results/eval_per_query.json`

## Design Decisions

- backend first: I care more about retrieval quality than UI at this stage
- simple route layer: FastAPI routes stay thin and call helper functions
- person-name-aware chunking: this is the most direct transfer from the sermon retrieval findings
- mock LLM mode: local wiring should still run before real API keys are plugged in
- explicit state machine: the reliability rules stay visible and interview-friendly without hiding them behind an agent framework
- approval-gated mutation: read/draft tools may run automatically, but the task write needs a human decision
- reviewed voice memory: unreviewed transcript text never enters CRM conversations, and approved Voice memory is excluded from personality profiling
- honest scope: Voice G0 uses a fake provider; local ASR and diarization are not claimed before measured baselines

More detail lives in `docs/design_decisions.md`.
Hosted setup notes live in `docs/hosting_setup.md`.

## Limitations

- Qdrant support exists, but the default local path still uses in-memory dense retrieval
- personality analysis is intentionally lightweight and still early
- the eval set is small and hand-labeled
- the live hosted stack is demo-grade and free-first, so Render cold starts can happen
- hosted embeddings still use mock vectors to keep deployment light and cheap
- the follow-up workflow currently writes only an internal CRM task; calendar import is a deterministic local stub
- durable stage checkpoints and duplicate-delivery safety are covered, but there is no background worker or automatic retry scheduler yet
- tool audit rows currently keep grounded context snapshots for replay, which increases the amount of relationship data retained
- PostgreSQL and Qdrant writes are recoverable through index rebuild, but they are not one atomic cross-database transaction
- Voice G0 uses a fixed synthetic fake provider and has no ASR/diarization accuracy or latency claim yet
- Voice G1 has an evaluation harness and license gate, but remains unmeasured until both local runtimes produce pinned public/synthetic artifacts
- the legacy `/api/conversations` voice branch stays disabled; reviewed audio uses `/api/voice/jobs`
- OCR is not implemented beyond a clear stub

## Future Work

- switch the main path from local memory retrieval to persistent Qdrant indexing
- add richer chunk persistence during ingestion instead of only runtime chunk building
- improve personality profile refresh logic with better prompts and stronger parsing
- add state-based workflow evaluation over approval, rejection, failure, and tenant-isolation cases
- measure the two local Voice runtime/pipeline baselines documented in `docs/voice_pipeline.md`
- test uncertainty-aware review routing before claiming confidence or n-best improvements
- support screenshot OCR
- build the separate medical-profile follow-up repo on top of the shared backend ideas

## License

This repo is licensed under the MIT License. See `LICENSE` for the full text.
