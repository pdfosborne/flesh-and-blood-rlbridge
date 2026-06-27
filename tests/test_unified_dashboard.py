"""Tests for unified random matchups HTML dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
                    "timeout_rate": 0.02,
                    "logic_vs_logic": {"p1_win_rate": 0.48},
                    "vs_logic": {
                        "agent_p1_seat": {
                            "agent_win_rate": 0.62,
                            "p2_win_rate": 0.38,
                        },
                        "agent_p2_seat": {
                            "agent_win_rate": 0.58,
                            "p1_win_rate": 0.42,
                        },
                    },
                },
                {
                    "matchup": "a-vs-b",
                    "episodes_completed": 200,
                    "p1_win_rate": 0.51,
                    "p1_wins": 51,
                    "p2_wins": 49,
                    "timeout_rate": 0.01,
                    "logic_vs_logic": {"p1_win_rate": 0.52},
                    "vs_logic": {
                        "agent_p1_seat": {
                            "agent_win_rate": 0.65,
                            "p2_win_rate": 0.35,
                        },
                        "agent_p2_seat": {
                            "agent_win_rate": 0.61,
                            "p1_win_rate": 0.39,
                        },
                    },
                },
            ]
        ),
        encoding="utf-8",
    )
    done_dir = run_dir / "done_match"
    done_dir.mkdir()
    (done_dir / "matchup_label.json").write_text(
        json.dumps({"name": "c-vs-d"}),
        encoding="utf-8",
    )
    ppo = done_dir / "ppo_test"
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
    (done_dir / "checkpoint_eval_history.json").write_text(
        json.dumps(
            [
                {
                    "matchup": "c-vs-d",
                    "episodes_completed": 100,
                    "p1_win_rate": 0.40,
                    "vs_logic": {
                        "agent_p1_seat": {"agent_win_rate": 0.30},
                        "agent_p2_seat": {"agent_win_rate": 0.32},
                    },
                },
                {
                    "matchup": "c-vs-d",
                    "episodes_completed": 1000,
                    "p1_win_rate": 0.55,
                    "vs_logic": {
                        "agent_p1_seat": {"agent_win_rate": 0.42},
                        "agent_p2_seat": {"agent_win_rate": 0.41},
                    },
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
    latest = state["checkpoint_points"][-1]
    assert latest["vs_logic_agent_p1"] == 0.65
    assert latest["vs_logic_win_rate"] == pytest.approx(0.63)
    assert latest["logic_vs_logic_win_rate"] == 0.52
    assert latest["logic_vs_agent_win_rate"] == pytest.approx(0.37)
    assert latest["agent_vs_agent_win_rate"] == 0.51
    assert latest["timeout_rate"] == 0.01

    html = render_unified_random_matchups_html(state, auto_refresh_seconds=5.0)
    assert "Training AI agents with random matchups" in html
    assert html.count("<th>Matchup</th>") == 1
    assert "Agent win% vs logic" in html
    assert "Logic win% vs logic" in html
    assert "Logic vs agent win%" in html
    assert "Agent win% vs agent" in html
    assert "Timeout %" in html
    assert "Vs logic avg%" not in html
    assert "Self-play P1%" not in html
    assert "First self-play ckpt" in html
    assert "Train P1 win%" not in html
    assert ">Status<" not in html
    completed = state["completed_matchups"][0]
    assert completed["first_checkpoint_win_rate"] == 0.40
    assert completed["checkpoint_win_rate"] == 0.55
    assert completed["checkpoint_vs_logic_win_rate"] == pytest.approx(0.415)
    assert "c-vs-d" in html
    assert "a-vs-b" in html
    assert "1/3" in html
    assert "250/1000" in html

    from fab_bridge.cpp_eval_live_dashboard import CPP_EVAL_LIVE_DASHBOARD

    (run_dir / CPP_EVAL_LIVE_DASHBOARD).write_text("<html></html>", encoding="utf-8")
    state_with_live = collect_unified_run_state(run_dir)
    assert state_with_live["cpp_eval_live_dashboard_path"]
    html_with_live = render_unified_random_matchups_html(state_with_live)
    assert CPP_EVAL_LIVE_DASHBOARD in html_with_live

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
