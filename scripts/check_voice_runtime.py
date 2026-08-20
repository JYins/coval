"""Fail-loud preflight for optional local Voice runtimes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PINNED_VERSIONS = {
    "funasr": {
        "funasr": "1.4.2",
        "soundfile": "0.14.0",
    },
    "sherpa": {
        "sherpa-onnx": "1.13.6",
        "soundfile": "0.14.0",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a Coval Voice runtime")
    parser.add_argument("--provider", choices=sorted(PINNED_VERSIONS), required=True)
    parser.add_argument("--workspace", default=str(ROOT))
    parser.add_argument("--min-free-gb", type=float, default=25.0)
    return parser.parse_args()


def package_check(name: str, expected: str) -> dict:
    try:
        actual = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        actual = None
    return {
        "ok": actual == expected,
        "expected": expected,
        "actual": actual,
    }


def python_check(provider: str) -> dict:
    version = sys.version_info[:3]
    if provider == "funasr":
        supported = version[:2] == (3, 12)
        note = "Coval pins FunASR to Python 3.12"
    else:
        supported = version[:2] >= (3, 12)
        note = "sherpa-onnx publishes a Windows CPython 3.14 wheel"
    return {
        "ok": supported,
        "actual": platform.python_version(),
        "note": note,
    }


def disk_check(workspace: Path, min_free_gb: float) -> dict:
    if min_free_gb <= 0:
        raise ValueError("--min-free-gb should be positive")
    free_gb = shutil.disk_usage(workspace).free / (1024**3)
    return {
        "ok": free_gb >= min_free_gb,
        "workspace": str(workspace.resolve()),
        "free_gb": round(free_gb, 2),
        "minimum_gb": min_free_gb,
        "note": "working-space budget, not a benchmark measurement",
    }


def config_check(provider: str) -> dict:
    try:
        if provider == "funasr":
            from src.voice.local_providers import FunASRProviderConfig

            config = FunASRProviderConfig.from_env()
        else:
            from src.voice.local_providers import SherpaProviderConfig

            config = SherpaProviderConfig.from_env()
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "device": getattr(config, "device", getattr(config, "provider", "cpu")),
        "threads": config.threads,
        "model_revision": config.model_revision,
        "artifacts": config.artifact_provenance,
    }


def torch_check(provider: str) -> dict:
    if provider != "funasr":
        return {"ok": True, "required": False}
    try:
        import torch
    except ImportError:
        return {"ok": False, "required": True, "error": "torch is not installed"}
    try:
        torchaudio_version = importlib.metadata.version("torchaudio")
    except importlib.metadata.PackageNotFoundError:
        torchaudio_version = None
    torch_base = torch.__version__.split("+", 1)[0]
    audio_base = torchaudio_version.split("+", 1)[0] if torchaudio_version else None
    compatible = audio_base == torch_base
    return {
        "ok": compatible,
        "required": True,
        "version": torch.__version__,
        "torchaudio_version": torchaudio_version,
        "versions_match": compatible,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }


def build_report(provider: str, workspace: Path, min_free_gb: float) -> dict:
    checks = {
        "python": python_check(provider),
        "disk": disk_check(workspace, min_free_gb),
        "packages": {
            name: package_check(name, expected)
            for name, expected in PINNED_VERSIONS[provider].items()
        },
        "provider_config": config_check(provider),
        "torch": torch_check(provider),
    }
    if (
        provider == "funasr"
        and checks["provider_config"].get("ok")
        and str(checks["provider_config"].get("device", "cpu")).startswith("cuda")
        and not checks["torch"].get("cuda_available")
    ):
        checks["torch"]["ok"] = False
        checks["torch"]["error"] = "FunASR requested CUDA but PyTorch cannot see CUDA"
    package_ok = all(row["ok"] for row in checks["packages"].values())
    ok = (
        checks["python"]["ok"]
        and checks["disk"]["ok"]
        and package_ok
        and checks["provider_config"]["ok"]
        and checks["torch"]["ok"]
    )
    return {
        "provider": provider,
        "ok": ok,
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    report = build_report(
        args.provider,
        Path(args.workspace),
        args.min_free_gb,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
