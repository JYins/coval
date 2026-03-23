# Coval

[![CI](https://github.com/JYins/coval/actions/workflows/ci.yml/badge.svg)](https://github.com/JYins/coval/actions/workflows/ci.yml)

> Shared electrons, shared context.

Coval is an AI relationship memory backend I am building step by step. The product idea is a little unconventional, I know, but I think there is a real use case here for dating, networking, and sales: if our tools can remember context better than we can, maybe that memory can help us show up a bit more thoughtfully too. I am keeping this backend-first on purpose. The main job right now is to make ingestion, retrieval, and grounded advice actually work before I worry about making it look polished.

## Why this name

`Coval` comes from **covalent bond**.

In chemistry, a covalent bond happens when two atoms share electrons. I liked that metaphor immediately. Human relationships are also built on shared things: shared context, shared memories, shared jokes, shared history, shared information that only makes sense between two people.

So the naming logic is basically:

- covalent bond -> shared electrons
- Coval -> shared context
- better shared context -> hopefully better conversations

Maybe a little nerdy, but anyways I think it fits.

## Why I built this

This repo comes directly after my RAG evaluation work in `rag-eval-pipeline`. In that project, I spent time benchmarking chunking strategies, embedding models, and retrieval setups because I wanted to understand what actually makes retrieval work instead of just building another shiny demo.

The biggest lesson from that repo was simple: if retrieval is weak, the LLM cannot really save you. Good context matters more than fancy prompting.

So Coval is the application layer of that work. Instead of stopping at evaluation, I want to apply those retrieval ideas to a real product shape:

- ingest conversations
- organize them around a person
- retrieve the right memory fragments later
- generate grounded advice or a quick briefing before you meet someone

## What works today

The backend is still early, but it is not just a skeleton anymore.

- SQLAlchemy models for users, persons, conversations, chunks, personality profiles, and interactions
- PostgreSQL connection setup and `scripts/init_db.py`
- FastAPI app entrypoint with register/login endpoints
- JWT auth helpers and password hashing
- person CRUD endpoints
- conversation upload route for manual text and `txt/csv` file input
- OCR and voice ingestion stubs with clear `NotImplementedError`

## Current Flow

The product loop I am building toward is:

1. take conversation input
2. store the structured record and retrieval-friendly chunks
3. retrieve the right context later and assemble a prompt for the LLM

The LLM has no memory by itself. The backend has to build that memory layer.

## Connection to RAG Eval

I already cloned my earlier RAG eval repo locally while building this one, and that connection is intentional.

- `cleaning.py` -> conversation text normalization here
- `chunking.py` -> person-name-aware chunking here
- `eval_metrics.py` -> retrieval quality tracking here
- config-driven experiments -> same style later in this repo

The sermon side of the eval project taught me that title-aware chunking matters a lot. In Coval, the "title" is basically the person's name.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/init_db.py
uvicorn src.api.app:app --reload
```

Once the server is running, current routes are:

- `POST /api/users/register`
- `POST /api/users/login`
- `POST /api/persons`
- `GET /api/persons`
- `GET /api/persons/{person_id}`
- `POST /api/conversations`

## Build Style

I do not want this repo to look like AI dumped a whole startup codebase in one shot.

So I am building it commit by commit, with a little more human trace:

- small steps
- clear folder growth
- simple code first
- honest TODOs when something is not built yet

## Limitations Right Now

- RAG retrieval is not wired in yet
- briefing and personality analysis are not wired in yet
- tests are still lightweight and focused on route shape
- OCR and voice are stubs for now
- the current local environment still needs full dependency setup before everything can be run end to end

That is okay for this stage. I would rather get the backbone right first.

