"""Tests for unified random matchup result discovery helpers."""

from __future__ import annotations

import json
from pathlib import Path

from fab_bridge.unified_results import (
    has_unified_selfplay_checkpoints,
    is_unified_random_matchup_run,
    iter_unified_checkpoint_metadata,
    resolve_latest_unified_matchup_dir,
)


def _write_unified_checkpoint(matchup_dir: Path, episode: int) -> None:
    ckpt = (
        matchup_dir
        / "unified_selfplay"
        / "p1"
        / f"episode_{episode:06d}"
    )
    ckpt.mkdir(parents=True)
    (ckpt / "metadata.json").write_text("{}", encoding="utf-8")


def test_resolve_latest_matchup_prefers_scope_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260626_120000"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")

    older = run_dir / "match_a"
    newer = run_dir / "match_b"
    older.mkdir()
    newer.mkdir()
    (older / "matchup_label.json").write_text(json.dumps({"name": "a-vs-b"}), encoding="utf-8")
    (newer / "matchup_label.json").write_text(json.dumps({"name": "c-vs-d"}), encoding="utf-8")

    (run_dir / "checkpoint_eval_scope.json").write_text(
        json.dumps({"matchup_dir": "match_a"}),
        encoding="utf-8",
    )

    assert resolve_latest_unified_matchup_dir(run_dir) == older


def test_has_unified_selfplay_checkpoints(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
    matchup_dir = run_dir / "abc123"
    matchup_dir.mkdir()
    (matchup_dir / "matchup_label.json").write_text("{}", encoding="utf-8")
    assert not has_unified_selfplay_checkpoints(run_dir)

    _write_unified_checkpoint(matchup_dir, 100)
    assert has_unified_selfplay_checkpoints(run_dir)
    assert len(iter_unified_checkpoint_metadata(run_dir, "p1")) == 1


def test_is_unified_random_matchup_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert not is_unified_random_matchup_run(run_dir)
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
    assert is_unified_random_matchup_run(run_dir)
