"""Tests for fab_tui results discovery helpers."""

from __future__ import annotations

import json
from pathlib import Path

from fab_tui.results import (
    _run_time_labels,
    discover_completed_training_runs,
    discover_evaluable_results,
)


def test_run_time_labels_from_folder_stamp(tmp_path: Path) -> None:
    run_dir = tmp_path / "briar_vs_dorinthea_20260621_153346"
    run_dir.mkdir()
    started, stamp = _run_time_labels(run_dir)
    assert started == "2026-06-21 15:33"
    assert stamp == "20260621_153346"


def test_run_time_labels_from_manifest_when_no_stamp(tmp_path: Path) -> None:
    run_dir = tmp_path / "custom_sideboard_run"
    run_dir.mkdir()
    (run_dir / "candidates_manifest.json").write_text(
        json.dumps({"started_at": "2026-06-21T16:45:00+00:00"}),
        encoding="utf-8",
    )
    started, stamp = _run_time_labels(run_dir)
    assert started.startswith("2026-06-21")
    assert "16:45" in started
    assert stamp == "20260621_164500"


def test_discover_includes_run_started(tmp_path: Path, monkeypatch) -> None:
    from fab_tui import results as results_mod

    root = tmp_path / "sideboard_compare"
    run_dir = root / "briar_vs_dorinthea_20260621_120000"
    candidate_dir = run_dir / "candidates" / "baseline"
    ckpt = candidate_dir / "p3_ab-vs-cd" / "p1" / "episode_000010"
    ckpt.mkdir(parents=True)
    (ckpt / "metadata.json").write_text("{}", encoding="utf-8")
    (run_dir / "candidates_manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        results_mod,
        "RESULT_CATEGORY_ROOTS",
        (("sideboard_compare", root),),
    )

    entries = discover_evaluable_results(limit=5)
    assert len(entries) == 1
    assert entries[0].run_started == "2026-06-21 12:00"
    assert entries[0].run_stamp == "20260621_120000"


def test_discover_completed_training_requires_completion_marker(
    tmp_path: Path, monkeypatch,
) -> None:
    from fab_tui import results as results_mod

    root = tmp_path / "sideboard_compare"
    run_dir = root / "briar_vs_briar_20260621_120000"
    candidate_dir = run_dir / "candidates" / "manual_01"
    ckpt = candidate_dir / "p3_ab-vs-cd" / "p1" / "episode_000100"
    ckpt.mkdir(parents=True)
    (ckpt / "metadata.json").write_text("{}", encoding="utf-8")
    (run_dir / "candidates_manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        results_mod,
        "RESULT_CATEGORY_ROOTS",
        (("sideboard_compare", root),),
    )

    assert discover_evaluable_results(limit=5)
    assert not discover_completed_training_runs(limit=5)

    (candidate_dir / "candidate_result.json").write_text(
        json.dumps(
            {
                "candidate_id": "manual_01",
                "play_win_rate": 0.58,
                "label": "Manual",
            }
        ),
        encoding="utf-8",
    )
    completed = discover_completed_training_runs(limit=5)
    assert len(completed) == 1
    assert completed[0].status_summary.startswith("trained")
    assert "manual_01" in completed[0].status_summary


def test_discover_unified_random_matchup_runs(tmp_path: Path, monkeypatch) -> None:
    from fab_tui import results as results_mod

    root = tmp_path / "unified_random_matchups"
    run_dir = root / "silver_age" / "20260626_120000"
    matchup_dir = run_dir / "abc12345"
    ckpt = matchup_dir / "unified_selfplay" / "p1" / "episode_000100"
    ckpt.mkdir(parents=True)
    (ckpt / "metadata.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"format": "silver_age", "matchups_sampled": ["a-vs-b"]}),
        encoding="utf-8",
    )
    (matchup_dir / "matchup_label.json").write_text(
        json.dumps({"name": "a-vs-b"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        results_mod,
        "RESULT_CATEGORY_ROOTS",
        (("unified_random_matchups", root),),
    )

    entries = discover_evaluable_results(limit=5)
    assert len(entries) == 1
    assert entries[0].category == "unified_random_matchups"
    assert "silver age" in entries[0].label.lower()
    assert entries[0].latest_episode == "episode_000100"
