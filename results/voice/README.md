# Voice result artifacts

This directory intentionally contains no benchmark numbers yet.

`scripts/run_voice_eval.py` writes one JSON artifact per measured baseline only after
real reference and prediction manifests pass schema and checksum/license gates. Fake
provider tests never write resume evidence here.
