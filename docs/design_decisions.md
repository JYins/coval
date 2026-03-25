# Design Decisions

## Why FastAPI + Python

This project grows directly out of a Python RAG evaluation repo, so keeping the application layer in Python makes the transfer very clean. FastAPI also lets me keep the route layer small and readable, which matters because I want to be able to explain every file in an interview without doing framework gymnastics.

## Why Backend First

The product idea is only convincing if the retrieval loop is real. A pretty frontend would not help much if:

- ingestion is messy
- retrieval misses the useful memory
- the prompt is weak
- the generated advice is not grounded

So I intentionally built this as a backend repo first. The main thing I am trying to prove is that user conversation data can be turned into retrieval-ready context in a way that is actually helpful later.

## Why Person-Name-Aware Chunking

This is the most direct transfer from `rag-eval-pipeline`.

In the sermon experiments, title-aware chunking improved retrieval because the title carried high-value context. In Coval, the analogous "title" is the person's name. Prefixing the chunk with the person's name is a simple trick, but it helps make each chunk more identity-aware.

That is why chunk rows carry:

- `chunk_text`
- `person_name_prefix`
- `chunk_index`

It is simple, but easy to justify.

## Why PostgreSQL + Qdrant Split

I did not want to force everything into one storage layer.

- PostgreSQL is the source of truth for accounts, persons, conversations, profiles, and interactions
- Qdrant is the vector-search layer for semantic retrieval

Right now the default local config still uses in-memory dense search, mainly because it makes development easier on a laptop. But the Qdrant wrapper is already in place because I want the architecture to reflect the production direction, not just the easiest demo path.

## Why Mock LLM Mode Exists

I want the repo to be runnable even before real API keys are configured. So there is a mock provider in the LLM client that returns:

- regular answer text for `/api/ask`
- JSON-like payloads for personality analysis and communication style
- a short briefing response

This is not meant to pretend the mock output is "real AI quality". It is mainly there so the pipeline can be wired, tested, and demoed locally without blocking on external credentials.

## Why The Eval Set Is Small

The current eval set is intentionally small and hand-labeled. At this stage, I mainly want:

- a concrete `run_eval.py`
- retrieval metrics that are actually computed
- visible artifacts in `results/`

This is more honest than claiming large-scale evaluation before the product data shape has stabilized.

## Why Medical Profile Is Deferred

The medical-profile idea shares a lot of the same infrastructure, but it is not just a copy-paste problem.

Medical retrieval has different constraints:

- compliance matters a lot more
- keyword precision matters more
- hybrid retrieval likely matters more than purely semantic retrieval

So I kept the medical direction as a design doc for now. If Relationship AI cannot reach a clean, explainable architecture first, then the medical follow-up would be premature.

