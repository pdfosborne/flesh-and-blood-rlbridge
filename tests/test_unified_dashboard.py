"""Tests for unified random matchups HTML dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fab_bridge.unified_dashboard import (
    LOGIC_VS_LOGIC_BASELINE_NAME,
    UNIFIED_DASHBOARD_NAME,
    UNIFIED_LIVE_STATE,
    _centered_axis_label,
    _checkpoint_point_from_row,
    _merged_aggregate_points,
    _rows_from_latest_merged,
    _training_matchup_series,
    aggregate_checkpoint_points,
    collect_unified_run_state,
    count_completed_matchups,
    record_policy_weight_update,
    render_unified_random_matchups_html,
    update_unified_matchup_live,
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
        json.dumps({"name": "a-vs-b", "p1_hero": "viserai", "p2_hero": "kayo"}),
        encoding="utf-8",
    )
    (run_dir / "checkpoint_eval_scope.json").write_text(
        json.dumps({"matchup": "a-vs-b", "matchup_dir": "match_a"}),
        encoding="utf-8",
    )
    (matchup_dir / LOGIC_VS_LOGIC_BASELINE_NAME).write_text(
        json.dumps({"p1_win_rate": 0.48}),
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
                "batch_index": 1,
                "parallel_matchups": 1,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "match_a" / "checkpoint_eval_history.json").write_text(
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
                    "p2_win_rate": 0.49,
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
    assert len(state["active_checkpoint_rows"]) == 1
    latest = state["active_checkpoint_rows"][0]
    assert latest["vs_logic_agent_p1"] == 0.65
    assert latest["vs_logic_win_rate"] == pytest.approx(0.63)
    assert latest["logic_vs_agent_win_rate"] == pytest.approx(0.37)
    assert latest["agent_vs_agent_win_rate"] == 0.51
    assert latest["timeout_rate"] == 0.01

    html = render_unified_random_matchups_html(state, auto_refresh_seconds=5.0)
    assert "Training AI agents with random matchups" in html
    assert "Training progress" in html
    assert "Checkpoint eval (active batch)" in html
    assert html.count("<th>Matchup</th>") == 2
    assert "Vs logic win% (Hero 1)" in html
    assert "Logic win% vs agent (Hero 1)" in html
    assert "Agent vs agent (Hero 1 win%)" in html
    assert "Hero 2 win%" not in html
    assert "Fabrary all (n)" in html
    assert "Fabrary std" not in html
    assert "(Hero 1) vs" in html
    assert "Timeout %" in html
    assert "class=\"summary\"" not in html
    assert "class=\"metrics\"" not in html
    assert "Matchups 1/3" in html
    assert "250/1000" in html

    assert "cpp_eval_live_dashboard.html" not in html
    assert "checkpoint eval replay" not in html.lower()

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


def test_parallel_active_matchups_and_aggregate(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    match_a = run_dir / "match_a"
    match_b = run_dir / "match_b"
    match_a.mkdir()
    match_b.mkdir()
    for subdir, name in ((match_a, "a-vs-b"), (match_b, "c-vs-d")):
        (subdir / "matchup_label.json").write_text(
            json.dumps({"name": name}),
            encoding="utf-8",
        )
        (subdir / "checkpoint_eval_history.json").write_text(
            json.dumps(
                [
                    {
                        "matchup": name,
                        "episodes_completed": 100,
                        "p1_win_rate": 0.40 if name == "a-vs-b" else 0.60,
                        "vs_logic": {
                            "agent_p1_seat": {"agent_win_rate": 0.30},
                            "agent_p2_seat": {"agent_win_rate": 0.34},
                        },
                    },
                    {
                        "matchup": name,
                        "episodes_completed": 200,
                        "p1_win_rate": 0.50 if name == "a-vs-b" else 0.70,
                        "vs_logic": {
                            "agent_p1_seat": {"agent_win_rate": 0.40},
                            "agent_p2_seat": {"agent_win_rate": 0.44},
                        },
                    },
                ]
            ),
            encoding="utf-8",
        )

    update_unified_matchup_live(
        run_dir,
        "match_a",
        name="a-vs-b",
        episodes_completed=200,
        p1_win_rate=0.52,
        hero1_win_rate=0.52,
        status="training",
    )
    update_unified_matchup_live(
        run_dir,
        "match_b",
        name="c-vs-d",
        episodes_completed=180,
        p1_win_rate=0.61,
        hero1_win_rate=0.61,
        status="training",
    )
    (run_dir / UNIFIED_LIVE_STATE).write_text(
        json.dumps(
            {
                **json.loads((run_dir / UNIFIED_LIVE_STATE).read_text(encoding="utf-8")),
                "matchups_total": 3,
                "matchups_completed": 0,
                "target_episodes": 1000,
                "parallel_matchups": 2,
                "batch_index": 1,
                "status": "training",
            }
        ),
        encoding="utf-8",
    )

    state = collect_unified_run_state(run_dir)
    assert len(state["active_matchups"]) == 2
    assert len(state["active_checkpoint_rows"]) == 2
    assert len(state["checkpoint_matchup_series"]) == 2
    assert len(state["training_matchup_series"]) == 2
    aggregate = aggregate_checkpoint_points(
        {
            "match_a": json.loads((match_a / "checkpoint_eval_history.json").read_text()),
            "match_b": json.loads((match_b / "checkpoint_eval_history.json").read_text()),
        }
    )
    ep200 = next(row for row in aggregate if row["episode"] == 200)
    assert ep200["win_rate_mean"] == pytest.approx(0.60)
    assert ep200["vs_logic_mean"] == pytest.approx(0.42)
    assert ep200["n_matchups"] == 2

    html = render_unified_random_matchups_html(state)
    assert "(Hero 1) vs" in html
    assert "c-vs-d" in html or "(Hero 2)" in html
    assert "matchup-progress" in html
    assert "chart-matchup-line" in html
    assert "Training rollouts" in html
    assert "Checkpoint eval" in html
    assert "+50%" in html
    assert "agent vs agent mean" not in html


def test_centered_axis_label_and_training_series(tmp_path: Path) -> None:
    assert _centered_axis_label(0.5) == "0%"
    assert _centered_axis_label(0.6) == "+10%"
    assert _centered_axis_label(0.4) == "-10%"

    run_dir = _write_run(tmp_path)
    match_a = run_dir / "match_a"
    match_a.mkdir()
    (match_a / "matchup_label.json").write_text(
        json.dumps({"name": "a-vs-b", "p1_hero": "viserai", "p2_hero": "kayo"}),
        encoding="utf-8",
    )
    update_unified_matchup_live(
        run_dir,
        "match_a",
        name="a-vs-b",
        episodes_completed=50,
        hero1_win_rate=0.55,
    )
    update_unified_matchup_live(
        run_dir,
        "match_a",
        name="a-vs-b",
        episodes_completed=100,
        hero1_win_rate=0.60,
    )
    live = json.loads((run_dir / UNIFIED_LIVE_STATE).read_text(encoding="utf-8"))
    series = _training_matchup_series(run_dir, live["active_matchups"])
    assert len(series) == 1
    assert series[0]["points"][-1]["win_rate"] == pytest.approx(0.60)
    assert series[0]["points"][-1]["episode"] == 100


def test_merged_checkpoint_eval_dashboard(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    merged_record = {
        "eval_mode": "merged",
        "episodes_completed": 100,
        "target_episodes": 1000,
        "eval_episodes_total": 100,
        "matchups_evaluated": 2,
        "checkpoint_bucket": "periodic",
        "aggregate": {
            "self_play_win_rate_mean": 0.55,
            "self_play_win_rate_se": 0.05,
            "vs_logic_win_rate_mean": 0.42,
            "vs_logic_win_rate_se": 0.04,
            "total_eval_games": 100,
        },
        "per_matchup": {
            "match_a": {
                "matchup": "a-vs-b",
                "episodes_completed": 100,
                "eval_episodes": 50,
                "p1_win_rate": 0.50,
                "timeout_rate": 0.02,
                "vs_logic": {
                    "agent_p1_seat": {"agent_win_rate": 0.40},
                    "agent_p2_seat": {"agent_win_rate": 0.44},
                },
            },
            "match_b": {
                "matchup": "c-vs-d",
                "episodes_completed": 100,
                "eval_episodes": 50,
                "p1_win_rate": 0.60,
                "timeout_rate": 0.01,
                "vs_logic": {
                    "agent_p1_seat": {"agent_win_rate": 0.38},
                    "agent_p2_seat": {"agent_win_rate": 0.46},
                },
            },
        },
    }
    (run_dir / "checkpoint_eval_history.json").write_text(
        json.dumps([merged_record]),
        encoding="utf-8",
    )
    update_unified_matchup_live(
        run_dir,
        "match_a",
        name="a-vs-b",
        episodes_completed=100,
        status="training",
    )
    update_unified_matchup_live(
        run_dir,
        "match_b",
        name="c-vs-d",
        episodes_completed=100,
        status="training",
    )

    aggregate = _merged_aggregate_points([merged_record])
    assert aggregate[0]["win_rate_mean"] == pytest.approx(0.55)
    assert aggregate[0]["n_matchups"] == 2

    rows = _rows_from_latest_merged(
        [merged_record],
        {
            "match_a": {"name": "a-vs-b"},
            "match_b": {"name": "c-vs-d"},
        },
        run_dir=run_dir,
    )
    assert len(rows) == 2
    assert rows[0]["eval_episodes"] == 50

    state = collect_unified_run_state(run_dir)
    assert state["checkpoint_aggregate_points"][0]["win_rate_mean"] == pytest.approx(0.55)
    assert len(state["active_checkpoint_rows"]) == 2
    assert len(state["checkpoint_matchup_series"]) == 2
    assert state["active_checkpoint_rows"][0]["eval_episodes"] == 50

    html = render_unified_random_matchups_html(state)
    assert "Eval games" in html
    assert "50" in html
    assert "chart-matchup-line" in html


def test_render_policy_weights_card(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    record_policy_weight_update(
        run_dir,
        {
            "initialized": True,
            "obs_dim": 128,
            "n_actions": 64,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "param_count": 12345,
            "l2_norm": 42.5,
            "fingerprint": "abc123def456",
            "update_count": 3,
        },
    )
    record_policy_weight_update(
        run_dir,
        {
            "initialized": True,
            "obs_dim": 128,
            "n_actions": 64,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "param_count": 12345,
            "l2_norm": 43.1,
            "fingerprint": "def456abc789",
            "update_count": 4,
        },
    )

    state = collect_unified_run_state(run_dir)
    html = render_unified_random_matchups_html(state)
    assert "Agent weights" in html
    assert "abc123def456" in html
    assert "def456abc789" in html
    assert "PPO updates" in html
    assert "L2 norm trend" in html


def test_dashboard_shows_logic_vs_logic_baseline_before_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path)
    matchup_dir = run_dir / "match_a"
    matchup_dir.mkdir()
    (matchup_dir / "matchup_label.json").write_text(
        json.dumps({"name": "a-vs-b"}),
        encoding="utf-8",
    )
    (matchup_dir / LOGIC_VS_LOGIC_BASELINE_NAME).write_text(
        json.dumps({"episodes": 20, "p1_win_rate": 0.48, "timeout_rate": 0.0}),
        encoding="utf-8",
    )
    (run_dir / UNIFIED_LIVE_STATE).write_text(
        json.dumps(
            {
                "matchups_total": 1,
                "matchups_completed": 0,
                "target_episodes": 1000,
                "status": "training",
                "batch_index": 1,
                "parallel_matchups": 1,
                "active_matchups": {
                    "match_a": {
                        "name": "a-vs-b",
                        "episodes_completed": 12,
                        "status": "training",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    state = collect_unified_run_state(run_dir)
    assert len(state["active_checkpoint_rows"]) == 1
    row = state["active_checkpoint_rows"][0]
    assert row["episode_label"] == "baseline"
    assert row["logic_vs_logic_win_rate"] == pytest.approx(0.48)
    assert row["eval_episodes"] == 20

    html = render_unified_random_matchups_html(state)
    assert "Logic vs logic (Hero 1 win%)" in html
    assert "48.0%" in html
    assert "baseline" in html


def test_chart_uses_self_play_when_merged_aggregate_null(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    match_a = run_dir / "match_a"
    match_b = run_dir / "match_b"
    for subdir, name in (("match_a", "a-vs-b"), ("match_b", "c-vs-d")):
        subdir_path = run_dir / subdir
        subdir_path.mkdir()
        (subdir_path / "matchup_label.json").write_text(
            json.dumps({"name": name}),
            encoding="utf-8",
        )
        (subdir_path / "checkpoint_eval_history.json").write_text(
            json.dumps(
                [
                    {
                        "matchup": name,
                        "episodes_completed": 50,
                        "eval_episodes": 5,
                        "p1_win_rate": 0.6 if subdir == "match_a" else 0.4,
                        "eval_mode": "self_play",
                    }
                ]
            ),
            encoding="utf-8",
        )
    (run_dir / "checkpoint_eval_history.json").write_text(
        json.dumps(
            [
                {
                    "eval_mode": "merged",
                    "status": "error",
                    "episodes_completed": 50,
                    "aggregate": {},
                    "per_matchup": {},
                }
            ]
        ),
        encoding="utf-8",
    )
    update_unified_matchup_live(
        run_dir,
        "match_a",
        name="a-vs-b",
        episodes_completed=50,
        status="training",
    )
    update_unified_matchup_live(
        run_dir,
        "match_b",
        name="c-vs-d",
        episodes_completed=50,
        status="training",
    )

    state = collect_unified_run_state(run_dir)
    points = state["checkpoint_aggregate_points"]
    assert points
    assert points[0]["win_rate_mean"] == pytest.approx(0.5)
    assert len(state["checkpoint_matchup_series"]) == 2
    html = render_unified_random_matchups_html(state)
    assert "chart-matchup-line" in html
    assert "agent vs agent mean" not in html
    assert "agent vs logic mean" not in html


def test_chart_hides_vs_logic_columns_without_data(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    matchup_dir = run_dir / "match_a"
    matchup_dir.mkdir()
    (matchup_dir / "matchup_label.json").write_text(
        json.dumps({"name": "a-vs-b"}),
        encoding="utf-8",
    )
    (matchup_dir / "checkpoint_eval_history.json").write_text(
        json.dumps(
            [
                {
                    "matchup": "a-vs-b",
                    "episodes_completed": 50,
                    "eval_episodes": 5,
                    "p1_win_rate": 0.55,
                    "eval_mode": "self_play",
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "format": "silver_age",
                "matchups_requested": 1,
                "episodes_per_matchup": 1000,
                "checkpoint_eval_logic_vs_logic": False,
                "checkpoint_eval_agent_vs_logic": False,
            }
        ),
        encoding="utf-8",
    )
    update_unified_matchup_live(
        run_dir,
        "match_a",
        name="a-vs-b",
        episodes_completed=50,
        status="training",
    )

    html = render_unified_random_matchups_html(collect_unified_run_state(run_dir))
    assert "Agent vs agent (Hero 1 win%)" in html
    assert "Vs logic win% (Hero 1)" not in html
    assert "Logic vs logic (Hero 1 win%)" not in html
    assert "Hero 2 win%" not in html


def test_checkpoint_win_rates_exclude_timeouts() -> None:
    row = {
        "episodes_completed": 100,
        "p1_wins": 4,
        "p2_wins": 4,
        "timeouts": 2,
        "p1_win_rate": 0.4,
        "win_rate_decided": 0.5,
        "logic_vs_logic": {
            "p1_wins": 3,
            "p2_wins": 5,
            "timeouts": 2,
            "p1_win_rate": 0.3,
            "win_rate_decided": 0.375,
        },
    }
    point = _checkpoint_point_from_row(row)
    assert point["agent_vs_agent_hero1_win_rate"] == pytest.approx(0.5)
    assert point["logic_vs_logic_hero1_win_rate"] == pytest.approx(0.375)
