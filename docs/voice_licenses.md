# Voice assets and licenses

Voice G1 is opt-in: this repository does not download, commit, mirror, or redistribute
models or audio datasets. The authoritative inventory is
[`configs/voice/licenses.yaml`](../configs/voice/licenses.yaml). Every actual run must
copy `artifacts.example.yaml`, replace each `REQUIRED/TODO` revision and SHA-256, and
check the selected artifacts against that inventory.

## Why there are separate entries

Source code, model weights, and evaluation data have independent licenses. For example,
FunASR source is MIT, while FunASR model artifacts use the FunASR Model License v1.1.
SenseVoice source is MIT, but its weights follow its model-card/upstream FunASR terms.
Likewise, sherpa-onnx is Apache-2.0 as a runtime; that does not change the license of
the SenseVoice ONNX weights it loads.

Direct runtime dependencies are also explicit assets. The FunASR path records PyTorch,
torchaudio, and SoundFile; the sherpa path records SoundFile in addition to the sherpa
binary runtime. SoundFile's Python package is BSD-3-Clause, while bundled
libsndfile has separate LGPL terms that must remain with a redistributed wheel. A runtime
dependency cannot be omitted from the evidence manifest just because pip installs it.
The optional FunASR requirements file intentionally does not choose a universal CPU/CUDA
PyTorch wheel. Preflight checks that the installed PyTorch and torchaudio base versions
match; each measured G1 run must separately pin the exact selected wheel versions and hashes
in its local artifact manifest.

The intended G1 comparison contains two runtime/pipeline baselines, not two independent
ASR models: FunASR + SenseVoice versus sherpa-onnx + the official SenseVoice ONNX
conversion. They share upstream SenseVoice weights.

## Data policy

AISHELL-4's original specification and the OpenSLR SLR111 listing identify CC BY-SA 4.0.
Metadata from hosting mirrors can conflict, so this project resolves dataset terms against
the original specification before an experiment. No AISHELL-4 audio is committed or
redistributed from this repository. Keep downloaded data outside the repository and record
only its source revision/checksum in a local, ignored experiment manifest.

The current Hugging Face dataset metadata shows Apache-2.0, which conflicts with the
original specification and OpenSLR. It is recorded as conflicting evidence, not used to
relicense the data. Fictional CRM meeting audio has its own fail-closed inventory entry:
the G1 asset gate will reject it until the recording or TTS provenance, terms, revision,
and checksum are real.

The sherpa diarization plan lists the Apache-2.0 sherpa runtime for inference and clustering,
gated pyannote segmentation weights, and an Apache-2.0 3D-Speaker ERes2Net embedding
artifact. A segmentation model alone is not represented as a complete diarization stack.

## Before running a measured baseline

1. Read the exact current model card and license for every selected weight.
2. Copy `configs/voice/artifacts.example.yaml` to a local ignored file; fill revision and
   SHA-256 fields for code, weights, and data.
3. Record acceptance of gated terms where needed. `pyannote/segmentation-3.0` is MIT but
   access is gated; never mirror the weights.
4. Use the resulting pinned manifest in the experiment report. Without it, do not make
   accuracy, latency, confidence, or redistribution claims.
