"""Tests for the local Voice runtime preflight."""

from pathlib import Path
from types import SimpleNamespace

import scripts.check_voice_runtime as preflight


def test_python_support_is_provider_specific(monkeypatch):
    monkeypatch.setattr(preflight.sys, "version_info", (3, 14, 0))
    monkeypatch.setattr(preflight.platform, "python_version", lambda: "3.14.0")

    assert preflight.python_check("funasr")["ok"] is False
    assert preflight.python_check("sherpa")["ok"] is True


def test_report_fails_when_workspace_or_pins_are_not_ready(monkeypatch):
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=2 * 1024**3),
    )
    monkeypatch.setattr(
        preflight,
        "package_check",
        lambda name, expected: {"ok": False, "expected": expected, "actual": None},
    )
    monkeypatch.setattr(
        preflight,
        "config_check",
        lambda provider: {"ok": False, "error": "model path missing"},
    )
    monkeypatch.setattr(
        preflight,
        "torch_check",
        lambda provider: {"ok": True},
    )

    report = preflight.build_report("sherpa", Path("."), 25)

    assert report["ok"] is False
    assert report["checks"]["disk"]["free_gb"] == 2.0
