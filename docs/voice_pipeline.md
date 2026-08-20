# Voice Pipeline

Voice is part of Coval's CRM memory flow, not a separate demo. The current G0 milestone proves the product and data path with a deterministic fake provider:

```text
request-scoped audio bytes
  -> fake multi-speaker transcript + timestamps + alternatives
  -> typed revisions and extracted candidates
  -> human approve / reject / edit
  -> ApprovedMemoryEvent
  -> Conversation(source_type="voice") + chunks
  -> existing RAG and follow-up agent
```

Only approved candidate content enters CRM memory. The unreviewed full transcript is never copied into `conversations`, and approved Voice memory is excluded from personality-profile generation.

## G0 scope

G0 includes:

- `VoiceIngestionJob`, `AudioSegment`, `SpeakerTurn`, `TranscriptAlternative`
- provider/model revision, trace ID, retry count, channel/VAD, overlap, and source spans
- append-only `TranscriptRevision`
- versioned `ExtractedCandidate`
- idempotent `ReviewDecision`
- `ApprovedMemoryEvent` linked to exactly one `Conversation`
- user/person tenant checks on every read or mutation
- retryable indexing without duplicate CRM conversations
- cancellation before any review decision; pending candidates become stale
- default `delete_after_processing` retention; no audio bytes or audio path in the database

Index writes acquire the approved-event row while SQL chunks and Qdrant vectors are
updated. This keeps an expired slow runner from racing a new owner. PostgreSQL and
Qdrant still cannot form one atomic transaction, so a failed cross-store write remains
recoverable through the existing index retry/rebuild path rather than being advertised
as distributed exactly-once delivery.

The fake provider uses a fixed, synthetic Mandarin two-speaker fixture. It is a pipeline test double, not an ASR benchmark. It does not infer identity, disease, emotion, or personality.

## Privacy rules

- use public or synthetic audio only during development
- never log audio, transcript text, candidate text, or provider raw responses
- store only audio SHA-256, byte size, MIME type, provider name, and lifecycle timestamps
- use provider speaker labels such as `speaker_0`; do not claim the real identity of a speaker
- require human review before any Voice candidate becomes CRM memory
- do not feed approved Voice memory into automated personality profiling

## Local baseline plan

G1 will compare two local runtime/pipeline baselines on named public data:

1. FunASR + SenseVoiceSmall + FSMN-VAD + CAM++
2. sherpa-onnx + the official SenseVoice ONNX conversion + a separately licensed diarization stack

These share upstream SenseVoice weights, so they must be described as two runtime/pipeline baselines, not two independent ASR models. CI will continue to run only the fake provider and JSON/database tests; model and dataset downloads stay opt-in.

Licenses must be recorded separately for source code, model weights, and evaluation data. FunASR code is MIT while its model weights use the FunASR Model License; sherpa-onnx runtime is Apache-2.0 while loaded weights keep their upstream terms. AISHELL-4's original specification and OpenSLR listing use CC BY-SA 4.0, so audio will not be committed or redistributed here.

## Claim gate

After G0, the honest claim is:

> Built a typed, auditable Voice ingestion/review/approval workflow with retention controls and fake-provider end-to-end tests.

G0 does not support claims about Chinese ASR accuracy, speaker diarization accuracy, calibrated confidence, n-best decoding, real-time performance, or production readiness. Those require measured G1/G2 artifacts such as CER, DER, speaker-attributed CER, RTF, review time, candidate F1, and false-approved-fact rate.
