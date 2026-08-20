"""Contract tests for optional local Voice provider adapters."""

import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.voice.extractors import CandidateTurnInput, FakeCandidateExtractor
import src.voice.local_providers as local_providers
from src.voice.local_providers import (
    FunASRProviderConfig,
    FunASRVoiceProvider,
    SherpaProviderConfig,
    build_funasr_result,
    build_sherpa_result,
)
from src.voice.providers import FakeVoiceProvider


def test_transcription_and_candidate_extraction_are_separate():
    transcript = FakeVoiceProvider().transcribe(
        b"RIFF-test",
        language="zh",
        fixture_name="mandarin_two_speaker_v1",
    )
    assert not hasattr(transcript, "candidates")

    result = FakeCandidateExtractor().extract(
        [
            CandidateTurnInput(
                turn_index=row.turn_index,
                speaker_label=row.speaker_label,
                start_ms=row.start_ms,
                end_ms=row.end_ms,
                text=row.alternatives[0].text,
                alternatives=[item.text for item in row.alternatives],
                diarization_confidence=row.diarization_confidence,
                overlap=row.overlap,
            )
            for row in transcript.turns
        ],
        language="zh",
        fixture_name="mandarin_two_speaker_v1",
    )
    assert len(result.candidates) == 2
    assert result.extractor_name == "fake"


def test_funasr_adapter_maps_current_sentence_info_contract():
    result = build_funasr_result(
        {
            "sentence_info": [
                {
                    "start": 600,
                    "end": 1700,
                    "spk": 2,
                    "text": "<|zh|>你好",
                }
            ]
        },
        provider_version="1.4.2",
        model_revision="sensevoice-revision",
        clean_text=lambda value: value.replace("<|zh|>", ""),
    )
    assert result.provider_version == "1.4.2"
    assert result.turns[0].speaker_label == "speaker_2"
    assert result.turns[0].alternatives[0].text == "你好"
    assert result.turns[0].diarization_confidence is None


def test_funasr_adapter_accepts_official_sentence_alias():
    result = build_funasr_result(
        {
            "sentence_info": [
                {"start": 0, "end": 1000, "spk": 0, "sentence": "官方字段"}
            ]
        },
        provider_version="1.4.2",
        model_revision="sensevoice-revision",
    )

    assert result.turns[0].alternatives[0].text == "官方字段"


def test_funasr_adapter_fails_loud_on_unknown_shape():
    with pytest.raises(RuntimeError, match="invalid row"):
        build_funasr_result(
            {"sentence_info": [{"start": 0, "end": 1000, "spk": 0}]},
            provider_version="1.4.2",
            model_revision="sensevoice-revision",
        )


def test_funasr_provider_passes_the_requested_language(monkeypatch):
    model = SimpleNamespace(
        generate=Mock(
            return_value=[
                {
                    "sentence_info": [
                        {"start": 0, "end": 1000, "spk": 0, "text": "你好"}
                    ]
                }
            ]
        )
    )
    auto_model = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=auto_model))
    monkeypatch.setitem(sys.modules, "funasr.utils", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "funasr.utils.postprocess_utils",
        SimpleNamespace(rich_transcription_postprocess=lambda value: value),
    )
    monkeypatch.setattr(
        local_providers.importlib.metadata,
        "version",
        lambda name: "1.4.2",
    )
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
        root = Path(folder)
        provider = FunASRVoiceProvider(
            FunASRProviderConfig(
                sensevoice_dir=root,
                vad_dir=root,
                speaker_dir=root,
                artifact_provenance={
                    "sensevoice-weights": {
                        "revision": "test-revision",
                        "sha256": "a" * 64,
                    }
                },
                device="cpu",
                threads=2,
            )
        )
        provider.transcribe(
            b"RIFF-test",
            language="mixed",
            fixture_name="unused",
            mime_type="audio/wav",
        )

    assert model.generate.call_args.kwargs["language"] == "auto"


def test_sherpa_config_rejects_oracle_speaker_count(monkeypatch):
    monkeypatch.setenv("VOICE_SHERPA_NUM_SPEAKERS", "2")
    with pytest.raises(ValueError, match="oracle speaker counts are disabled"):
        SherpaProviderConfig.from_env()


def test_sherpa_adapter_keeps_anonymous_speakers_and_overlap():
    result = build_sherpa_result(
        [
            (SimpleNamespace(start=0.0, end=1.5, speaker=0), "第一句"),
            (SimpleNamespace(start=1.2, end=2.0, speaker=1), "第二句"),
        ],
        provider_version="1.13.4",
        model_revision="sherpa-artifact-revision",
    )
    assert [row.speaker_label for row in result.turns] == [
        "speaker_0",
        "speaker_1",
    ]
    assert all(row.overlap for row in result.turns)
    assert all(row.alternatives[0].confidence is None for row in result.turns)
