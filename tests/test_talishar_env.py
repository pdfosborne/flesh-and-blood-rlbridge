"""Tests for persisted Talishar training env."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fab_bridge import talishar_env


def test_build_training_env_values_single_shard() -> None:
    values = talishar_env.build_training_env_values(shards=1, base_port=8080)
    assert values["TALISHAR_URL"] == "http://localhost:8080/game"
    assert "TALISHAR_URLS" not in values


def test_build_training_env_values_multi_shard() -> None:
    values = talishar_env.build_training_env_values(shards=3, base_port=8080)
    assert values["TALISHAR_URLS"] == (
        "http://localhost:8080/game,"
        "http://localhost:8081/game,"
        "http://localhost:8082/game"
    )


def test_write_and_apply_training_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "talishar-training.local.env"
    monkeypatch.setattr(talishar_env, "training_env_path", lambda: env_file)
    monkeypatch.delenv("TALISHAR_URLS", raising=False)

    written = talishar_env.write_training_env(shards=2, base_port=8080)
    assert env_file.is_file()
    assert "TALISHAR_URLS" in written

    monkeypatch.delenv("TALISHAR_URL", raising=False)
    loaded = talishar_env.apply_training_env()
    assert loaded["TALISHAR_URL"] == "http://localhost:8080/game"
    assert os.environ["TALISHAR_URLS"] == written["TALISHAR_URLS"]


def test_apply_training_env_respects_existing_shell_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / "talishar-training.local.env"
    monkeypatch.setattr(talishar_env, "training_env_path", lambda: env_file)
    talishar_env.write_training_env(shards=2, base_port=8080)

    monkeypatch.setenv("TALISHAR_URL", "http://override:9000/game")
    talishar_env.apply_training_env()
    assert os.environ["TALISHAR_URL"] == "http://override:9000/game"
