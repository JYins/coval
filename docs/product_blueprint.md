# Product Blueprint

## One-Line Idea

An AI relationship memory backend that ingests conversations, builds lightweight profiles, and helps me prepare for future interactions with better context.

## Why This Product

The motivation is practical more than philosophical.

When people are dating, networking aggressively, or handling lots of client conversations, context slips fast. Not because they do not care, but because memory is messy and fragmented. The product idea here is to treat those fragments as retrieval data:

- conversation notes
- uploaded chat exports
- later maybe voice notes
- later maybe screenshot OCR

Then the system should retrieve the right fragments when I ask:

- "what should I remember before I meet this person?"
- "what topics are safe to open with?"
- "what style works best with them?"

## Target Early Users

I do not think this is a universal product. It is probably strongest for people who already manage a lot of interpersonal context:

- dating users who want better continuity
- sales people who want sharper meeting prep
- frequent networkers who meet many people fast

There is an awkward side to the concept, and I do not want to hide that. But there is also a real productivity use case here, especially if the system stays grounded in actual conversation history instead of fake personality magic.

## Core Product Loop

1. ingest conversations
2. chunk and organize them around one person
3. retrieve relevant memory later
4. assemble prompt context
5. generate advice or a briefing

The key idea is that the LLM does not "remember" anything by itself. The memory layer is externalized through storage + retrieval.

## Input Methods

Implementation order:

1. manual text input
2. file upload for `txt/csv` chat exports
3. voice note transcription
4. screenshot OCR

The first two matter most for MVP because they are enough to prove the ingestion and retrieval loop.

## Main Entities

- `users`: auth and ownership
- `persons`: one tracked relationship target
- `conversations`: raw inputs by source and language
- `chunks`: retrieval-ready segments
- `personality_profiles`: lightweight structured summary
- `interactions`: asks, briefings, and later maybe feedback scores

## Product Outputs

The first output types are:

- direct Q&A about a person
- pre-meeting briefing
- lightweight personality summary
- communication-style hints

These outputs should stay practical. If they become too abstract or too "AI-ish", the product loses the point.

## Why This Connects Well To RAG Eval

This repo only makes sense because of the earlier retrieval project.

That earlier work answered:

- what chunking style helps retrieval
- which embedding models are usable
- why metadata and chunk structure matter

This repo asks the next question:

- can those same retrieval lessons support a real product workflow?

That is the portfolio story I want the code to support.

