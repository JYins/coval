# Medical Profile Design

## Status

Design doc only. No implementation in this repo.

The plan is still to keep Relationship AI as the first full product and move the medical version into a separate repo later, after the shared backend ideas are proven.

## Why A Separate Repo

The infrastructure overlap is real, but the product constraints are different enough that I do not want to blur them together too early.

Shared pieces:

- FastAPI app structure
- auth and user management
- PostgreSQL base models
- vector-store client pattern
- prompt assembly flow
- config-driven eval scripts

Different pieces:

- medical schema
- retrieval strategy emphasis
- compliance and data handling expectations
- prompt templates and output format

## Likely Medical Schema

Early candidate tables:

- `patients`
- `medical_records`
- `medications`
- `allergies`
- `appointments`
- `symptom_timeline`
- `interactions`

The exact shape is still open, but it would follow the same idea as Coval: structured rows in PostgreSQL, semantic search over chunked records, and prompt assembly on top.

## Retrieval Difference

Relationship AI can lean more on semantic similarity because phrasing is often loose and conversational.

Medical retrieval is different:

- keyword precision matters more
- terminology matters more
- false positives are riskier

So the likely retrieval path there is hybrid:

- dense retrieval for semantic similarity
- keyword / BM25 style retrieval for exact terms
- stronger reranking around medical entities

## Compliance Note

This is the biggest reason to defer implementation.

Even for a personal project, medical information changes the stakes. If I build that version later, I would want to think more carefully about:

- storage boundaries
- encryption expectations
- access control
- auditability
- whether local-only or self-hosted flows make more sense

## Migration Idea

If the current architecture is healthy, the later medical repo should be able to reuse most of the backbone with only moderate adaptation:

- new schema
- new prompt templates
- new eval set
- different retrieval defaults

If it requires a total rewrite, then this repo's architecture was probably not modular enough in the first place.

