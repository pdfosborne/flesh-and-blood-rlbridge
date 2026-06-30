"""Tests for eval shard lock coordination."""

from __future__ import annotations

from pathlib import Path

from fab_bridge.eval_shard_lock import (
    acquire_eval_shard_lock,
    is_eval_shard_busy,
    release_eval_shard_lock,
)


def test_eval_shard_lock_acquire_and_release(tmp_path: Path) -> None:
    assert not is_eval_shard_busy(tmp_path)
    acquire_eval_shard_lock(tmp_path, "checkpoint_eval", label="merged ep 50")
    assert is_eval_shard_busy(tmp_path)
    release_eval_shard_lock(tmp_path, "checkpoint_eval")
    assert not is_eval_shard_busy(tmp_path)
