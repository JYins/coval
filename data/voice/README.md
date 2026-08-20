# Voice evaluation data

Do not commit corpus audio or model weights here.

G1 expects two frozen JSON inputs:

- `manifests/reference.json`: licensed reference annotations and split metadata
- `predictions/<baseline>.json`: normalized output plus runtime/resource measurements

The JSON contracts live in `src/voice/eval_contracts.py`. A reference manifest must
identify whether each frozen run is public or synthetic, name its license, and carry the
real dataset source SHA-256. A prediction manifest must pin runtime/model revisions, config/model
hashes, hardware, and per-sample timing/resource measurements.

AISHELL-4 audio is not redistributed by this repository. Synthetic scenarios must use
fictional participants and separately audited recording or TTS assets.
