"""Tests for unified random matchups HTML dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from fab_bridge.unified_dashboard import (
    UNIFIED_DASHBOARD_NAME,
    UNIFIED_LIVE_STATE,
    collect_unified_run_state,
    count_completed_matchups,
    render_unified_random_matchups_html,
    write_unified_random_matchups_dashboard,
)


def _write_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "20260626_215125"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "format": "silver_age",
                "matchups_requested": 3,
                "matchups_sampled": ["a-vs-b", "c-vs-d", "e-vs-f"],
                "episodes_per_matchup": 1000,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_collect_and_render_dashboard(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    matchup_dir = run_dir / "match_a"
    matchup_dir.mkdir()
    (matchup_dir / "matchup_label.json").write_text(
        json.dumps({"name": "a-vs-b"}),
        encoding="utf-8",
    )
    (run_dir / "checkpoint_eval_scope.json").write_text(
        json.dumps({"matchup": "a-vs-b", "matchup_dir": "match_a"}),
        encoding="utf-8",
    )
    (run_dir / UNIFIED_LIVE_STATE).write_text(
        json.dumps(
            {
                "current_matchup": "a-vs-b",
                "current_matchup_dir": "match_a",
                "matchups_total": 3,
                "matchups_completed": 1,
                "target_episodes": 1000,
                "episodes_completed": 250,
                "p1_win_rate": 0.52,
                "p2_win_rate": 0.48,
                "status": "training",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "checkpoint_eval_history.json").write_text(
        json.dumps(
            [
                {
                    "matchup": "a-vs-b",
                    "episodes_completed": 100,
                    "p1_win_rate": 0.45,
                    "p1_wins": 45,
                    "p2_wins": 55,
                },
                {
                    "matchup": "a-vs-b",
                    "episodes_completed": 200,
                    "p1_win_rate": 0.51,
                    "p1_wins": 51,
                    "p2_wins": 49,
                },
            ]
        ),
        encoding="utf-8",
    )

    state = collect_unified_run_state(run_dir)
    assert state["matchups_total"] == 3
    assert state["current_matchup"] == "a-vs-b"
    assert state["episodes_completed"] == 250
    assert state["train_p1_win_rate"] == 0.52
    assert len(state["checkpoint_points"]) == 2

    html = render_unified_random_matchups_html(state, auto_refresh_seconds=5.0)
    assert "Unified random matchups" in html
    assert "a-vs-b" in html
    assert "1/3" in html
    assert "250/1000" in html

    html_path = write_unified_random_matchups_dashboard(run_dir, auto_refresh_seconds=5.0)
    assert html_path == run_dir / UNIFIED_DASHBOARD_NAME
    assert html_path.is_file()


def test_count_completed_matchups(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    done = run_dir / "done_match"
    done.mkdir()
    (done / "matchup_label.json").write_text("{}", encoding="utf-8")
    ppo = done / "ppo_test"
    ppo.mkdir()
    (ppo / "training_results.json").write_text(
        json.dumps(
            {
                "n_episodes": 1000,
                "episode_rewards": [0.1] * 1000,
                "training_stats": {
                    "episodes": 1000,
                    "p1_win_rate": 0.55,
                    "p2_win_rate": 0.45,
                },
            }
        ),
        encoding="utf-8",
    )
    assert count_completed_matchups(run_dir, 1000) == 1
