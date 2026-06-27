"""Tests for unified random matchup result discovery helpers."""

from __future__ import annotations

import json
from pathlib import Path

from fab_bridge.unified_results import (
    find_latest_unified_checkpoint_metadata,
    has_unified_selfplay_checkpoints,
    is_unified_random_matchup_run,
    iter_unified_checkpoint_metadata,
    resolve_latest_unified_matchup_dir,
    resolve_unified_run_root,
)


def _write_unified_checkpoint(matchup_dir: Path, episode: int) -> Path:
    ckpt = (
        matchup_dir
        / "unified_selfplay"
        / "p1"
        / f"episode_{episode:06d}"
    )
    ckpt.mkdir(parents=True)
    meta = ckpt / "metadata.json"
    meta.write_text("{}", encoding="utf-8")
    return meta


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


def test_resolve_unified_run_root_from_matchup_subdir(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260626_215125"
    matchup_dir = run_dir / "deck_a-vs-deck_b"
    matchup_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
    (matchup_dir / "matchup_label.json").write_text("{}", encoding="utf-8")

    assert resolve_unified_run_root(matchup_dir) == run_dir


def test_find_latest_checkpoint_across_all_matchups(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260626_215125"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "checkpoint_eval_scope.json").write_text(
        json.dumps({"matchup_dir": "older_matchup"}),
        encoding="utf-8",
    )

    older = run_dir / "older_matchup"
    newer = run_dir / "newer_matchup"
    older.mkdir()
    newer.mkdir()
    (older / "matchup_label.json").write_text("{}", encoding="utf-8")
    (newer / "matchup_label.json").write_text("{}", encoding="utf-8")
    _write_unified_checkpoint(older, 500)
    latest_meta = _write_unified_checkpoint(newer, 900)

    scoped = iter_unified_checkpoint_metadata(
        run_dir,
        "p1",
        matchup_dir=resolve_latest_unified_matchup_dir(run_dir),
    )
    assert len(scoped) == 1
    assert scoped[0].parent.name == "episode_000500"

    all_paths = iter_unified_checkpoint_metadata(run_dir, "p1")
    assert all_paths[-1] == latest_meta
    assert find_latest_unified_checkpoint_metadata(run_dir, "p1") == latest_meta
