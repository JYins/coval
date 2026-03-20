# Coval

> Shared electrons, shared context.

Coval is the starting repo for an AI relationship memory backend. The idea is a bit unusual, I know, but I think there is a real use case here for dating, networking, and sales: if our apps can remember everything, maybe we can use that memory a little more intentionally too. This repo is backend-first on purpose. I want to get the retrieval loop right before pretending the whole product is finished.

## Why this name

`Coval` comes from **covalent bond**.

In chemistry, a covalent bond happens when two atoms share electrons. I liked that metaphor a lot. Human relationships are also kind of built on shared things: shared context, shared memories, shared jokes, shared history, shared information that only makes sense between two people.

So the naming logic is basically:

- covalent bond -> shared electrons
- Coval -> shared context
- better shared context -> hopefully better conversations

Maybe a little nerdy, but anyways I think it fits.

## Why I am building this

This project comes right after my RAG eval work. In that repo, I spent time benchmarking chunking strategies, embedding models, and retrieval setups because I wanted to understand what actually makes retrieval work, not just build another demo that looks smart on the surface.

The main lesson I took away was simple: if retrieval is weak, the LLM cannot really save you. Good context matters more than fancy prompting.

So Coval is the next step. Instead of stopping at evaluation, I want to apply those retrieval ideas to a real product shape:

- ingest conversations
- organize them around a person
- retrieve the right memory fragments later
- generate grounded advice or a quick briefing before you meet someone

## What this repo is right now

Right now this is only the first commit skeleton.

I am setting up the repo structure first so the project can grow in a more natural way:

- FastAPI backend
- PostgreSQL models
- Qdrant-based retrieval
- config-driven RAG pipeline
- personality / briefing logic later

Some parts will stay simple for a while on purpose. OCR and voice input are in scope later, but not for the first step. I would rather get manual text input and retrieval working first than fake completeness.

## Build Style

I do not want this repo to look like AI dumped a whole startup codebase in one shot.

So I am building it commit by commit, with a little more human trace:

- small steps
- clear folder growth
- simple code first
- honest TODOs when something is not built yet

## Current Status

- repo bootstrapped
- README v1 written
- skeleton folders added
- backend implementation starts next

More technical detail will come later when the app pieces are real enough to deserve it.

