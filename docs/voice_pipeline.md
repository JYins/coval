# Voice Pipeline

Voice is part of Coval's CRM memory flow, not a separate demo. G0 proves the product and data path with a deterministic fake provider, while optional local adapters now expose the same typed transcription contract for FunASR and sherpa-onnx:

```text
request-scoped audio bytes
  -> ASR/diarization provider: segments + anonymous speaker turns
  -> candidate extractor: typed review candidates (or manual selection)
  -> append-only transcript revisions
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

The provider and candidate extractor are deliberately separate. ASR code cannot write a
CRM fact directly. Every candidate records extractor provenance, and that provenance is
copied into the approved memory event. Human transcript corrections stale old candidates
and run extraction again instead of being relabeled as a 100%-confidence fact.

The fake provider uses a fixed, synthetic Mandarin two-speaker fixture. It is a pipeline test double, not an ASR benchmark. It does not infer identity, disease, emotion, or personality.

## Privacy rules

- use public or synthetic audio only during development
- never log audio, transcript text, candidate text, or provider raw responses
- store only audio SHA-256, byte size, MIME type, provider name, and lifecycle timestamps
- local adapters write one request to an OS temporary directory only while native inference
  runs; normal and error exits remove it through `TemporaryDirectory`. A killed process can
  still leave an orphan for the operating system to clean up, and this is not secure erase.
- use provider speaker labels such as `speaker_0`; do not claim the real identity of a speaker
- require human review before any Voice candidate becomes CRM memory
- do not feed approved Voice memory into automated personality profiling

## Local baseline plan

G1 will compare two local runtime/pipeline baselines on named public data:

1. FunASR + SenseVoiceSmall + FSMN-VAD + CAM++
2. sherpa-onnx + the official SenseVoice ONNX conversion + a separately licensed diarization stack

These share upstream SenseVoice weights, so they must be described as two runtime/pipeline baselines, not two independent ASR models. CI will continue to run only the fake provider and JSON/database tests; model and dataset downloads stay opt-in.

Licenses must be recorded separately for source code, model weights, and evaluation data. FunASR code is MIT while its model weights use the FunASR Model License; sherpa-onnx runtime is Apache-2.0 while loaded weights keep their upstream terms. AISHELL-4's original specification and OpenSLR listing use CC BY-SA 4.0, so audio will not be committed or redistributed here.

## Optional local product adapters

`src/voice/local_providers.py` contains strict adapters for:

- FunASR `1.4.2`: local SenseVoiceSmall + FSMN-VAD + CAM++ directories; consumes
  `sentence_info` with `text` or the documented `sentence` alias plus `start/end/spk`, and
  fails if neither transcript field is present.
- sherpa-onnx `1.13.6`: 16 kHz mono WAV, pyannote segmentation + ERes2Net clustering,
  then greedy SenseVoice decoding per anonymous speaker turn.

Neither adapter exposes emotion/audio-event tags, n-best output, or invented confidence.
Model objects are cached per process and inference is serialized because the optional
native runtimes are not treated as safely re-entrant.

Install each runtime in a separate environment. For FunASR, install a matching PyTorch and
torchaudio build first. Coval's preflight requires Python 3.12 for FunASR; the sherpa path
accepts Python 3.12 or newer:

```bash
pip install -r requirements-voice-funasr.txt
python scripts/check_voice_runtime.py --provider funasr

pip install -r requirements-voice-sherpa.txt
python scripts/check_voice_runtime.py --provider sherpa
```

The preflight checks the fixed FunASR/sherpa/SoundFile versions, a compatible installed
PyTorch/torchaudio pair, supported Python, local artifact paths, model revision metadata,
and a configurable working-space budget. Required variables:

```text
VOICE_THREADS
VOICE_FUNASR_SENSEVOICE_DIR
VOICE_FUNASR_SENSEVOICE_REVISION
VOICE_FUNASR_SENSEVOICE_SHA256
VOICE_FUNASR_VAD_DIR
VOICE_FUNASR_VAD_REVISION
VOICE_FUNASR_VAD_SHA256
VOICE_FUNASR_SPEAKER_DIR
VOICE_FUNASR_SPEAKER_REVISION
VOICE_FUNASR_SPEAKER_SHA256
VOICE_FUNASR_DEVICE

VOICE_SHERPA_SENSEVOICE_MODEL
VOICE_SHERPA_TOKENS
VOICE_SHERPA_SENSEVOICE_REVISION
VOICE_SHERPA_SENSEVOICE_SHA256
VOICE_SHERPA_SEGMENTATION_MODEL
VOICE_SHERPA_SEGMENTATION_REVISION
VOICE_SHERPA_SEGMENTATION_SHA256
VOICE_SHERPA_EMBEDDING_MODEL
VOICE_SHERPA_EMBEDDING_REVISION
VOICE_SHERPA_EMBEDDING_SHA256
VOICE_SHERPA_PROVIDER
VOICE_SHERPA_CLUSTER_THRESHOLD
```

Sherpa clustering always estimates the speaker count (`num_speakers=-1`). Using a reference
speaker count as an oracle is rejected because it would make DER comparisons unfair. Local
provider environment variables are startup-only; restart the API after changing them.

The preflight hashes each configured model file or deterministic directory bundle and checks
it against these startup values. Directory hashes cover sorted relative file names plus each
file SHA-256; the sherpa SenseVoice bundle covers the model and token file names and hashes.
It still does not replace `validate_voice_assets.py`: G1 evidence must pin the exact selected
PyTorch/torchaudio wheel identity and bind the same model hashes and runtime/config artifacts
in the local asset manifest. Provider audit rows record model artifacts, not an approved
runtime manifest, so the evaluation gate remains mandatory for benchmark claims.

Real providers default to the `manual` candidate extractor. The reviewer selects a typed
candidate from a transcript turn through the API, then separately approves or rejects it.
This keeps local speech inference useful before a separately evaluated automatic extractor
exists, without pretending that raw transcript text is already a trusted CRM fact.

## Claim gate

After G0, the honest claim is:

> Built a typed, auditable Voice ingestion/review/approval workflow with retention controls and fake-provider end-to-end tests.

G0 does not support claims about Chinese ASR accuracy, speaker diarization accuracy, calibrated confidence, n-best decoding, real-time performance, or production readiness. Those require measured G1/G2 artifacts such as CER, DER, speaker-attributed CER, RTF, review time, candidate F1, and false-approved-fact rate.

## G1 evaluation harness

The repository now has a provider-independent, fail-loud harness for the two local
runtime/pipeline baselines. It does not download a model or invent a result.

1. Copy `configs/voice/artifacts.example.yaml` to the ignored
   `configs/voice/artifacts.local.yaml` and replace every `REQUIRED/TODO` revision and
   SHA-256.
2. Validate the license and artifact selection:

   ```bash
   python scripts/validate_voice_assets.py --run-manifest configs/voice/artifacts.local.yaml
   ```

3. Make each local adapter write the frozen JSON contracts from
   `src/voice/eval_contracts.py`. Reference and prediction sample IDs must match exactly.
4. Point a copy of `configs/voice/eval.yaml` at those manifests, then run:

   ```bash
   python scripts/run_voice_eval.py --config configs/voice/eval.local.yaml
   ```

The evaluator reports duration/character-weighted CER, VAD precision/recall/F1,
frame-based DER with documented collar and overlap handling, JER, speaker-attributed CER,
proper-name recall, number/date/unit exact match, action-owner and speaker attribution,
structured candidate F1, false-approved-fact rate, review/abstention rate, RTF,
first-partial and completion latency percentiles, peak RAM/VRAM, and model size. Anonymous
speaker labels are aligned one-to-one by maximum temporal overlap before attribution
metrics are scored.

Speaker-attributed text is rebuilt from timestamped speaker turns; a provider cannot submit
a separate unbound speaker transcript. Candidate IDs are unique, review decisions reference
those IDs, and candidate spans must overlap the claimed speaker turn. Structured F1 requires
an exact normalized type/speaker/text match plus source-span IoU of at least 0.5. JER is a
macro average over all scored meeting-speaker pairs. Number/date/unit exact match is a macro
average over labeled meetings, while label-level precision/recall/F1 are also emitted.

G1 passes only after both named baselines have measured artifacts for both a public and a
fictional synthetic corpus:

```bash
python scripts/check_voice_g1.py --run-manifest configs/voice/artifacts.local.yaml results/voice/*.json
```

The evaluator calls the asset/license gate before scoring, hashes the actual reference and
prediction JSON files, and embeds those hashes plus the inventory/run-manifest hashes in the
result. The final G1 command reopens those files and recomputes every metric. It also rejects
duplicate prediction evidence, mismatched baseline pins/config hashes, different reference
sets between baselines, and reuse of one reference set as both public and synthetic.
The frozen synthetic split must contain at least six fictional meetings. Scoring parameters
such as DER frame, collar, and overlap handling are pinned in the run manifest rather than
chosen after results are visible.

As of this commit, the table is intentionally unfilled:

| Baseline | Public corpus | Synthetic corpus | Status |
|---|---|---|---|
| FunASR/SenseVoice/FSMN/CAM++ | — | — | not measured |
| sherpa-onnx/SenseVoice + separate diarization | — | — | not measured |

The two paths share SenseVoice acoustic weights. Results can support a runtime/pipeline
comparison, not a claim that two independent ASR models were compared. The license
inventory and redistribution boundaries are documented in `docs/voice_licenses.md`.
`false_approved_fact_rate` means consistency with the frozen reference annotations; it is
not a production safety guarantee. The gate recomputes evidence and blocks hand-edited
result JSON, but it is not cryptographic remote attestation that a named model executed.
