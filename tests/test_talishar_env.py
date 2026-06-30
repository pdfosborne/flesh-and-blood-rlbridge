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
    values = talishar_env.build_training_env_values(shards=4, base_port=8080)
    assert values["TALISHAR_URLS"] == (
        "http://localhost:8080/game,"
        "http://localhost:8081/game,"
        "http://localhost:8082/game"
    )
    assert values["TALISHAR_EVAL_URL"] == "http://localhost:8083/game"


def test_build_training_env_values_reserves_render_shard() -> None:
    values = talishar_env.build_training_env_values(
        shards=5,
        base_port=8080,
        reserve_eval_shard=True,
        reserve_render_shard=True,
    )
    assert values["TALISHAR_URLS"] == (
        "http://localhost:8080/game,"
        "http://localhost:8081/game,"
        "http://localhost:8082/game"
    )
    assert values["TALISHAR_EVAL_URL"] == "http://localhost:8083/game"
    assert values["TALISHAR_RENDER_URL"] == "http://localhost:8084/game"


def test_build_training_env_values_multi_shard_no_reserve() -> None:
    values = talishar_env.build_training_env_values(
        shards=3,
        base_port=8080,
        reserve_eval_shard=False,
    )
    assert values["TALISHAR_URLS"] == (
        "http://localhost:8080/game,"
        "http://localhost:8081/game,"
        "http://localhost:8082/game"
    )
    assert "TALISHAR_EVAL_URL" not in values


def test_write_and_apply_training_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "talishar-training.local.env"
    monkeypatch.setattr(talishar_env, "training_env_path", lambda: env_file)
    monkeypatch.delenv("TALISHAR_URLS", raising=False)

    written = talishar_env.write_training_env(shards=2, base_port=8080)
    assert env_file.is_file()
    assert "TALISHAR_URLS" not in written
    assert written["TALISHAR_EVAL_URL"] == "http://localhost:8081/game"
    assert written["TALISHAR_URL"] == "http://localhost:8080/game"

    monkeypatch.delenv("TALISHAR_URL", raising=False)
    loaded = talishar_env.apply_training_env()
    assert loaded["TALISHAR_URL"] == "http://localhost:8080/game"
    assert loaded["TALISHAR_EVAL_URL"] == "http://localhost:8081/game"


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
