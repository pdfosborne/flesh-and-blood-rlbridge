"""Tests for unified training debug logging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fab_bridge import unified_training_debug as debug


@pytest.fixture(autouse=True)
def _reset_debug_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAB_UNIFIED_DEBUG", raising=False)
    debug._ENABLED = False  # noqa: SLF001
    debug._RUN_DIR = None  # noqa: SLF001
    debug._LOG_PATH = None  # noqa: SLF001


def test_configure_writes_jsonl(tmp_path: Path) -> None:
    log_path = debug.configure(run_dir=tmp_path, enabled=True)
    assert log_path == tmp_path / "unified_training_debug.jsonl"
    assert log_path.is_file()
    debug.log_event("test", "hello", shard="localhost:8081")
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
    record = json.loads(lines[-1])
    assert record["category"] == "test"
    assert record["details"]["shard"] == "localhost:8081"


def test_disabled_does_not_write(tmp_path: Path) -> None:
    assert debug.configure(run_dir=tmp_path, enabled=False) is None
    debug.log_event("test", "ignored")
    assert not (tmp_path / "unified_training_debug.jsonl").exists()


def test_connection_details_from_exception() -> None:
    exc = RuntimeError(
        "HTTPConnectionPool(host='localhost', port=8081): Max retries exceeded"
    )
    details = debug.connection_details_from_exception(exc)
    assert details["host"] == "localhost"
    assert details["port"] == 8081


def test_read_debug_from_manifest(tmp_path: Path) -> None:
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"debug_training": True}),
        encoding="utf-8",
    )
    assert debug.read_debug_from_manifest(tmp_path) is True
