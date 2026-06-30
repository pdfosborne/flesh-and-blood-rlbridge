"""Tests for optimal-policy live render companion locking."""

from __future__ import annotations

import os
from pathlib import Path

from fab_bridge.optimal_policy_render_lock import (
    optimal_policy_render_lock_holder,
    release_optimal_policy_render_lock,
    try_acquire_optimal_policy_render_lock,
)


def test_render_lock_allows_single_holder(tmp_path: Path) -> None:
    assert try_acquire_optimal_policy_render_lock(tmp_path) is True
    assert optimal_policy_render_lock_holder(tmp_path) == os.getpid()
    release_optimal_policy_render_lock(tmp_path)
    assert optimal_policy_render_lock_holder(tmp_path) is None


def test_render_lock_released_allows_new_holder(tmp_path: Path) -> None:
    assert try_acquire_optimal_policy_render_lock(tmp_path) is True
    release_optimal_policy_render_lock(tmp_path)
    assert try_acquire_optimal_policy_render_lock(tmp_path) is True
    release_optimal_policy_render_lock(tmp_path)


def test_pick_live_render_matchup_prefers_latest(tmp_path: Path) -> None:
    from scripts.eval.eval_phase3_checkpoint import _pick_live_render_matchup_dir  # noqa: PLC0415

    run_dir = tmp_path / "run"
    older = run_dir / "matchup_a"
    newer = run_dir / "matchup_b"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "matchup_label.json").write_text("{}", encoding="utf-8")
    (newer / "matchup_label.json").write_text("{}", encoding="utf-8")
    os.utime(newer, (2, 2))
    os.utime(older, (1, 1))

    picked = _pick_live_render_matchup_dir(run_dir, [older, newer])
    assert picked.resolve() == newer.resolve()
