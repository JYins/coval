"""Optional local FunASR and sherpa-onnx Voice providers."""

from __future__ import annotations

import importlib.metadata
import hashlib
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from src.voice.providers import (
    VoiceAlternativeResult,
    VoiceSegmentResult,
    VoiceTranscriptionResult,
    VoiceTurnResult,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def required_path(name: str, *, directory: bool = False) -> Path:
    raw = os.getenv(name, "").strip()
    if not raw:
        raise ValueError(f"{name} is required for the local Voice provider")
    path = Path(raw).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise ValueError(f"{name} should point to an existing {kind}")
    return path


def required_value(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required for the local Voice provider")
    return value


def required_sha256(name: str) -> str:
    value = required_value(name).lower()
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} should be a lowercase SHA-256")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(row for row in path.rglob("*") if row.is_file())
    if not files:
        raise ValueError(f"Voice model directory is empty: {path}")
    for row in files:
        relative = row.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(row)))
    return digest.hexdigest()


def sha256_bundle(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        label = path.name.encode("utf-8")
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def check_artifact_hash(name: str, actual: str) -> str:
    expected = required_sha256(name)
    if actual != expected:
        raise ValueError(f"{name} does not match the local Voice artifact")
    return expected


def voice_threads() -> int:
    value = int(os.getenv("VOICE_THREADS", "4"))
    if value < 1:
        raise ValueError("VOICE_THREADS should be at least 1")
    return value


def audio_suffix(mime_type: str) -> str:
    suffixes = {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
    }
    suffix = suffixes.get(mime_type)
    if suffix is None:
        raise ValueError("local Voice providers require WAV, MP3, or M4A audio")
    return suffix


@dataclass(frozen=True)
class FunASRProviderConfig:
    sensevoice_dir: Path
    vad_dir: Path
    speaker_dir: Path
    artifact_provenance: dict[str, dict[str, str]]
    device: str
    threads: int

    @classmethod
    def from_env(cls):
        sensevoice_dir = required_path(
            "VOICE_FUNASR_SENSEVOICE_DIR",
            directory=True,
        )
        vad_dir = required_path("VOICE_FUNASR_VAD_DIR", directory=True)
        speaker_dir = required_path("VOICE_FUNASR_SPEAKER_DIR", directory=True)
        artifacts = {
            "sensevoice-weights": {
                "revision": required_value("VOICE_FUNASR_SENSEVOICE_REVISION"),
                "sha256": check_artifact_hash(
                    "VOICE_FUNASR_SENSEVOICE_SHA256",
                    sha256_directory(sensevoice_dir),
                ),
            },
            "fsmn-vad-weights": {
                "revision": required_value("VOICE_FUNASR_VAD_REVISION"),
                "sha256": check_artifact_hash(
                    "VOICE_FUNASR_VAD_SHA256",
                    sha256_directory(vad_dir),
                ),
            },
            "campplus-weights": {
                "revision": required_value("VOICE_FUNASR_SPEAKER_REVISION"),
                "sha256": check_artifact_hash(
                    "VOICE_FUNASR_SPEAKER_SHA256",
                    sha256_directory(speaker_dir),
                ),
            },
        }
        return cls(
            sensevoice_dir=sensevoice_dir,
            vad_dir=vad_dir,
            speaker_dir=speaker_dir,
            artifact_provenance=artifacts,
            device=os.getenv("VOICE_FUNASR_DEVICE", "cpu").strip() or "cpu",
            threads=voice_threads(),
        )

    @property
    def model_revision(self) -> str:
        return ";".join(
            f"{name}@{row['revision']}"
            for name, row in sorted(self.artifact_provenance.items())
        )


class FunASRVoiceProvider:
    name = "funasr"

    def __init__(self, config: FunASRProviderConfig):
        self.config = config
        self._model = None
        self._lock = threading.Lock()

    @property
    def request_identity(self) -> dict[str, object]:
        return {
            "artifacts": self.config.artifact_provenance,
            "device": self.config.device,
            "threads": self.config.threads,
        }

    @classmethod
    def from_env(cls):
        return cls(FunASRProviderConfig.from_env())

    def transcribe(
        self,
        audio: bytes,
        *,
        language: str,
        fixture_name: str,
        mime_type: str = "audio/wav",
    ) -> VoiceTranscriptionResult:
        del fixture_name
        if not audio:
            raise ValueError("audio should not be empty")
        if language not in {"zh", "mixed"}:
            raise ValueError("FunASR Voice supports zh or mixed")
        try:
            from funasr import AutoModel
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except ImportError as exc:
            raise RuntimeError(
                "FunASR Voice requires the optional funasr runtime"
            ) from exc

        suffix = audio_suffix(mime_type)
        with self._lock:
            if self._model is None:
                self._model = AutoModel(
                    model=str(self.config.sensevoice_dir),
                    vad_model=str(self.config.vad_dir),
                    spk_model=str(self.config.speaker_dir),
                    spk_mode="punc_segment",
                    device=self.config.device,
                    ncpu=self.config.threads,
                    disable_update=True,
                )
            with tempfile.TemporaryDirectory(prefix="coval-voice-") as folder:
                path = Path(folder) / f"input{suffix}"
                path.write_bytes(audio)
                rows = self._model.generate(
                    input=str(path),
                    language="zh" if language == "zh" else "auto",
                    batch_size_s=300,
                    sentence_timestamp=True,
                    return_spk_res=True,
                    use_itn=True,
                )
        if len(rows) != 1:
            raise RuntimeError("FunASR should return exactly one audio result")
        return build_funasr_result(
            rows[0],
            provider_version=importlib.metadata.version("funasr"),
            model_revision=self.config.model_revision,
            artifact_provenance=self.config.artifact_provenance,
            clean_text=rich_transcription_postprocess,
        )


def build_funasr_result(
    raw: dict,
    *,
    provider_version: str,
    model_revision: str,
    artifact_provenance: dict[str, dict[str, str]] | None = None,
    clean_text=lambda value: value,
) -> VoiceTranscriptionResult:
    rows = raw.get("sentence_info")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("FunASR result is missing sentence_info")
    segments = []
    turns = []
    for index, row in enumerate(rows):
        try:
            start_ms = int(row["start"])
            end_ms = int(row["end"])
            speaker = int(row["spk"])
            raw_text = row.get("text", row.get("sentence"))
            if raw_text is None:
                raise KeyError("text")
            text = clean_text(str(raw_text)).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("FunASR sentence_info has an invalid row") from exc
        if not text:
            raise RuntimeError("FunASR returned an empty speaker turn")
        segments.append(
            VoiceSegmentResult(
                segment_index=index,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
        turns.append(
            VoiceTurnResult(
                turn_index=index,
                segment_index=index,
                speaker_label=f"speaker_{speaker}",
                start_ms=start_ms,
                end_ms=end_ms,
                alternatives=[VoiceAlternativeResult(text=text)],
            )
        )
    return VoiceTranscriptionResult(
        provider_version=provider_version,
        model_name="SenseVoiceSmall+FSMN-VAD+CAM++",
        model_revision=model_revision,
        artifact_provenance=dict(artifact_provenance or {}),
        segments=segments,
        turns=turns,
    )


@dataclass(frozen=True)
class SherpaProviderConfig:
    sensevoice_model: Path
    tokens: Path
    segmentation_model: Path
    embedding_model: Path
    artifact_provenance: dict[str, dict[str, str]]
    provider: str
    threads: int
    num_speakers: int
    cluster_threshold: float

    @classmethod
    def from_env(cls):
        num_speakers = int(os.getenv("VOICE_SHERPA_NUM_SPEAKERS", "-1"))
        if num_speakers != -1:
            raise ValueError(
                "VOICE_SHERPA_NUM_SPEAKERS must be -1; oracle speaker counts are disabled"
            )
        threshold = float(os.getenv("VOICE_SHERPA_CLUSTER_THRESHOLD", "0.5"))
        if not 0 < threshold <= 1:
            raise ValueError("VOICE_SHERPA_CLUSTER_THRESHOLD should be in (0, 1]")
        sensevoice_model = required_path("VOICE_SHERPA_SENSEVOICE_MODEL")
        tokens = required_path("VOICE_SHERPA_TOKENS")
        segmentation_model = required_path("VOICE_SHERPA_SEGMENTATION_MODEL")
        embedding_model = required_path("VOICE_SHERPA_EMBEDDING_MODEL")
        artifacts = {
            "sherpa-sensevoice-onnx-weights": {
                "revision": required_value("VOICE_SHERPA_SENSEVOICE_REVISION"),
                "sha256": check_artifact_hash(
                    "VOICE_SHERPA_SENSEVOICE_SHA256",
                    sha256_bundle([sensevoice_model, tokens]),
                ),
            },
            "pyannote-segmentation-3-weights": {
                "revision": required_value("VOICE_SHERPA_SEGMENTATION_REVISION"),
                "sha256": check_artifact_hash(
                    "VOICE_SHERPA_SEGMENTATION_SHA256",
                    sha256_file(segmentation_model),
                ),
            },
            "eres2net-weights": {
                "revision": required_value("VOICE_SHERPA_EMBEDDING_REVISION"),
                "sha256": check_artifact_hash(
                    "VOICE_SHERPA_EMBEDDING_SHA256",
                    sha256_file(embedding_model),
                ),
            },
        }
        return cls(
            sensevoice_model=sensevoice_model,
            tokens=tokens,
            segmentation_model=segmentation_model,
            embedding_model=embedding_model,
            artifact_provenance=artifacts,
            provider=os.getenv("VOICE_SHERPA_PROVIDER", "cpu").strip() or "cpu",
            threads=voice_threads(),
            num_speakers=num_speakers,
            cluster_threshold=threshold,
        )

    @property
    def model_revision(self) -> str:
        return ";".join(
            f"{name}@{row['revision']}"
            for name, row in sorted(self.artifact_provenance.items())
        )


class SherpaVoiceProvider:
    name = "sherpa"

    def __init__(self, config: SherpaProviderConfig):
        self.config = config
        self._diarization = None
        self._recognizer = None
        self._lock = threading.Lock()

    @property
    def request_identity(self) -> dict[str, object]:
        return {
            "artifacts": self.config.artifact_provenance,
            "provider": self.config.provider,
            "threads": self.config.threads,
            "num_speakers": self.config.num_speakers,
            "cluster_threshold": self.config.cluster_threshold,
        }

    @classmethod
    def from_env(cls):
        return cls(SherpaProviderConfig.from_env())

    def transcribe(
        self,
        audio: bytes,
        *,
        language: str,
        fixture_name: str,
        mime_type: str = "audio/wav",
    ) -> VoiceTranscriptionResult:
        del fixture_name
        if not audio:
            raise ValueError("audio should not be empty")
        if language not in {"zh", "mixed"}:
            raise ValueError("sherpa Voice supports zh or mixed")
        if mime_type not in {"audio/wav", "audio/x-wav"}:
            raise ValueError("sherpa Voice currently requires 16 kHz mono WAV")
        try:
            import sherpa_onnx
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "sherpa Voice requires sherpa-onnx and soundfile"
            ) from exc

        with tempfile.TemporaryDirectory(prefix="coval-voice-") as folder:
            path = Path(folder) / "input.wav"
            path.write_bytes(audio)
            samples, sample_rate = sf.read(
                path,
                dtype="float32",
                always_2d=True,
            )
        if sample_rate != 16000 or samples.shape[1] != 1:
            raise ValueError("sherpa Voice currently requires 16 kHz mono WAV")
        audio_mono = samples[:, 0]
        with self._lock:
            if self._diarization is None:
                self._diarization = self._build_diarization(sherpa_onnx)
            if self._recognizer is None:
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                    model=str(self.config.sensevoice_model),
                    tokens=str(self.config.tokens),
                    num_threads=self.config.threads,
                    sample_rate=16000,
                    feature_dim=80,
                    decoding_method="greedy_search",
                    provider=self.config.provider,
                    language="zh" if language == "zh" else "auto",
                    use_itn=True,
                )
            if self._diarization.sample_rate != sample_rate:
                raise ValueError("speaker model sample rate does not match 16 kHz audio")
            diarized = self._diarization.process(audio_mono).sort_by_start_time()
            decoded = []
            for row in diarized:
                start_sample = max(0, round(row.start * sample_rate))
                end_sample = min(len(audio_mono), round(row.end * sample_rate))
                if end_sample <= start_sample:
                    continue
                stream = self._recognizer.create_stream()
                stream.accept_waveform(
                    sample_rate,
                    audio_mono[start_sample:end_sample],
                )
                self._recognizer.decode_stream(stream)
                text = stream.result.text.strip()
                if text:
                    decoded.append((row, text))
        return build_sherpa_result(
            decoded,
            provider_version=importlib.metadata.version("sherpa-onnx"),
            model_revision=self.config.model_revision,
            artifact_provenance=self.config.artifact_provenance,
        )

    def _build_diarization(self, sherpa_onnx):
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(self.config.segmentation_model),
                    window_shift_ratio=0.1,
                )
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self.config.embedding_model),
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=self.config.num_speakers,
                threshold=self.config.cluster_threshold,
            ),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        if not config.validate():
            raise ValueError("sherpa diarization configuration is invalid")
        return sherpa_onnx.OfflineSpeakerDiarization(config)


def build_sherpa_result(
    decoded: list[tuple[object, str]],
    *,
    provider_version: str,
    model_revision: str,
    artifact_provenance: dict[str, dict[str, str]] | None = None,
) -> VoiceTranscriptionResult:
    if not decoded:
        raise RuntimeError("sherpa returned no transcribed speaker turns")
    ranges = [(float(row.start), float(row.end)) for row, _ in decoded]
    segments = []
    turns = []
    for index, (row, text) in enumerate(decoded):
        start_ms = round(float(row.start) * 1000)
        end_ms = round(float(row.end) * 1000)
        overlap = any(
            other != index
            and min(float(row.end), ranges[other][1])
            > max(float(row.start), ranges[other][0])
            for other in range(len(ranges))
        )
        segments.append(
            VoiceSegmentResult(
                segment_index=index,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
        turns.append(
            VoiceTurnResult(
                turn_index=index,
                segment_index=index,
                speaker_label=f"speaker_{int(row.speaker)}",
                start_ms=start_ms,
                end_ms=end_ms,
                overlap=overlap,
                alternatives=[VoiceAlternativeResult(text=text)],
            )
        )
    return VoiceTranscriptionResult(
        provider_version=provider_version,
        model_name="sherpa-onnx SenseVoice+pyannote+ERes2Net",
        model_revision=model_revision,
        artifact_provenance=dict(artifact_provenance or {}),
        segments=segments,
        turns=turns,
    )
