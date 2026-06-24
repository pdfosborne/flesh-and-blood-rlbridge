"""Tests for the sideboard compare HTML dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts" / "eval"))

from datetime import datetime, timedelta, timezone

from sideboard_compare_dashboard import (  # noqa: E402
    _chart_engine_label,
    _chart_title,
    _estimate_sideboard_compare_eta,
    _slug_to_display_name,
    collect_sideboard_compare_state,
    render_sideboard_compare_html,
    write_sideboard_compare_dashboard,
)


def test_chart_engine_labels() -> None:
    assert _chart_engine_label("C++ engine") == "cpp engine"
    assert _chart_engine_label("HTTP Talishar") == "talishar engine"
    assert _chart_title("Training win rate", "cpp engine") == (
        "Training win rate (cpp engine)"
    )

def test_slug_to_display_name() -> None:
    assert "Red" in _slug_to_display_name("aurora_rousing_aurora_red")
    assert _slug_to_display_name("briar") == "Briar"


def test_collect_and_render_dashboard(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    candidates_dir = out_dir / "candidates"
    baseline_dir = candidates_dir / "baseline"
    swap_dir = candidates_dir / "swap_01"
    baseline_dir.mkdir(parents=True)
    swap_dir.mkdir(parents=True)

    manifest = {
        "started_at": "2026-06-21T12:00:00+00:00",
        "format": "silver_age",
        "hero_id": "aurora",
        "opponent_hero_id": "briar",
        "opponent_deck": "BriarSAGEPrecon",
        "play_episodes": 100,
        "final_eval_episodes": 20,
        "skip_final_eval": False,
        "max_parallel": 2,
        "checkpoint_interval": 10,
        "checkpoint_eval_episodes": 50,
        "cpp_engine_dir": "/tmp/cpp_engine",
        "candidates": [
            {
                "candidate_id": "baseline",
                "label": "Default deck",
                "game_deck": {"a_red": 40},
                "swaps": [],
            },
            {
                "candidate_id": "swap_01",
                "label": "Swap blockers",
                "game_deck": {"a_red": 39, "b_red": 1},
                "swaps": [["slow_red", "block_red"]],
                "guide_margin": 1.25,
            },
        ],
    }
    (out_dir / "candidates_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    meta_dir = swap_dir / "p3_abcd1234-vs-efgh5678" / "p1" / "episode_000050"
    meta_dir.mkdir(parents=True)
    (meta_dir / "metadata.json").write_text(
        json.dumps({
            "episodes_completed": 50,
            "target_episodes": 100,
            "win_rate": 0.54,
            "wins": 27,
            "losses": 23,
            "draws": 0,
            "runtime_backend": "C++ engine",
            "cpp_engine_dir": "/tmp/cpp_engine",
        }),
        encoding="utf-8",
    )
    (swap_dir / "play_training_live.json").write_text(
        json.dumps({
            "episodes_completed": 50,
            "target_episodes": 100,
            "win_rate": 0.54,
            "wins": 27,
            "losses": 23,
            "draws": 0,
            "runtime_backend": "C++ engine",
        }),
        encoding="utf-8",
    )
    (swap_dir / "play_training_history.json").write_text(
        json.dumps([{
            "episodes_completed": 50,
            "target_episodes": 100,
            "win_rate": 0.54,
            "wins": 27,
            "losses": 23,
            "draws": 0,
            "runtime_backend": "C++ engine",
        }]),
        encoding="utf-8",
    )
    (meta_dir / "checkpoint_eval.json").write_text(
        json.dumps({
            "episodes_completed": 50,
            "p1_win_rate": 0.62,
            "p1_wins": 31,
            "losses": 19,
            "draws": 0,
            "runtime_backend": "C++ engine",
        }),
        encoding="utf-8",
    )
    (swap_dir / "checkpoint_eval_history.json").write_text(
        json.dumps([
            {"episodes_completed": 10, "p1_win_rate": 0.54, "eval_episodes": 50, "runtime_backend": "C++ engine"},
            {"episodes_completed": 25, "p1_win_rate": 0.55, "eval_episodes": 50, "runtime_backend": "C++ engine"},
            {"episodes_completed": 50, "p1_win_rate": 0.56, "eval_episodes": 50, "runtime_backend": "C++ engine"},
            {"episodes_completed": 75, "p1_win_rate": 0.55, "eval_episodes": 50, "runtime_backend": "C++ engine"},
            {"episodes_completed": 90, "p1_win_rate": 0.55, "eval_episodes": 50, "runtime_backend": "C++ engine"},
        ]),
        encoding="utf-8",
    )
    (swap_dir / "candidate_result.json").write_text(
        json.dumps({
            "candidate_id": "swap_01",
            "play_win_rate": 0.58,
            "final_eval_win_rate": 0.60,
        }),
        encoding="utf-8",
    )
    (baseline_dir / "candidate_result.json").write_text(
        json.dumps({
            "candidate_id": "baseline",
            "play_win_rate": 0.52,
            "final_eval_win_rate": 0.50,
        }),
        encoding="utf-8",
    )

    state = collect_sideboard_compare_state(out_dir)
    assert state["hero_id"] == "aurora"
    assert len(state["candidates"]) == 2
    swap_row = next(c for c in state["candidates"] if c["candidate_id"] == "swap_01")
    assert swap_row["train_done"] == 50
    assert swap_row["play_win_rate"] == pytest.approx(0.58)
    assert swap_row["train_chart_points"][-1]["win_rate"] == pytest.approx(0.54)
    assert swap_row["train_chart_points"][-1]["stderr"] == pytest.approx(0.0704, rel=0.01)
    assert swap_row["chart_points"][-1]["stderr"] == pytest.approx(0.0704, rel=0.01)
    assert swap_row["train_engine_label"] == "cpp engine"
    assert swap_row["eval_engine_label"] == "cpp engine"
    assert swap_row["latest_checkpoint_win_rate"] == pytest.approx(0.55)
    assert swap_row["final_eval_delta"] == pytest.approx(0.10)
    stability = swap_row.get("eval_stability") or {}
    assert stability.get("status") == "converged"
    assert stability.get("converged") is True

    # Parallel seeds: dashboard picks best seed win% per checkpoint.
    parallel_dir = candidates_dir / "parallel_swap"
    for seed_idx, rate in [(0, 0.52), (1, 0.61)]:
        seed_dir = parallel_dir / "parallel_seeds" / f"seed_{seed_idx}"
        seed_dir.mkdir(parents=True)
        (seed_dir / "checkpoint_eval_history.json").write_text(
            json.dumps([{
                "episodes_completed": 50,
                "p1_win_rate": rate,
                "p1_wins": int(rate * 100),
                "losses": 100 - int(rate * 100),
                "draws": 0,
                "timeouts": 0,
                "eval_episodes": 50,
            }]),
            encoding="utf-8",
        )
    manifest["candidates"].append({
        "candidate_id": "parallel_swap",
        "label": "Parallel seeds",
        "game_deck": {"a_red": 40},
        "swaps": [],
    })
    manifest["parallel_seeds"] = 5
    (out_dir / "candidates_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    state_parallel = collect_sideboard_compare_state(out_dir)
    parallel_row = next(
        c for c in state_parallel["candidates"] if c["candidate_id"] == "parallel_swap"
    )
    assert parallel_row["latest_checkpoint_win_rate"] == pytest.approx(0.61)
    assert parallel_row["latest_checkpoint_best_seed"] == 1

    staged_dir = candidates_dir / "staged_swap"
    for seed_idx in range(4):
        seed_dir = staged_dir / "parallel_seeds" / f"seed_{seed_idx}"
        seed_dir.mkdir(parents=True)
        if seed_idx < 3:
            live = {"episodes_completed": 20, "target_episodes": 20}
        else:
            live = {"episodes_completed": 30, "target_episodes": 80}
        (seed_dir / "play_training_live.json").write_text(
            json.dumps(live),
            encoding="utf-8",
        )
    (staged_dir / "candidate_result.json").write_text(
        json.dumps({
            "candidate_id": "staged_swap",
            "play_win_rate": 0.57,
        }),
        encoding="utf-8",
    )
    manifest["candidates"].append({
        "candidate_id": "staged_swap",
        "label": "Staged seeds",
        "game_deck": {"a_red": 40},
        "swaps": [],
    })
    manifest["parallel_seeds"] = 4
    manifest["parallel_seeds_until_first_checkpoint"] = True
    manifest["checkpoint_interval"] = 20
    (out_dir / "candidates_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    state_staged = collect_sideboard_compare_state(out_dir)
    staged_row = next(
        c for c in state_staged["candidates"] if c["candidate_id"] == "staged_swap"
    )
    assert staged_row["train_done"] == 50
    assert staged_row["train_target"] == 100
    assert staged_row["train_pct"] == pytest.approx(50.0)
    assert staged_row["training_stage"] == "Best-seed continuation"
    assert staged_row["status"] == "training"

    html = render_sideboard_compare_html(state, auto_refresh_seconds=5.0)
    assert "Sideboard comparison dashboard" in html
    assert "Eval stability" in html
    assert "Converged" in html
    assert "chart-error-bar" in html
    assert "chart-error-bar-train" in html
    assert "Training win rate (cpp engine)" in html
    assert "Checkpoint eval win rate (cpp engine)" in html
    assert "Swap blockers" in html
    assert "slow" in html.lower() or "Slow" in html
    assert 'http-equiv="refresh"' in html

    html_path = write_sideboard_compare_dashboard(out_dir, auto_refresh_seconds=None)
    assert html_path.is_file()
    assert "sideboard_compare_dashboard.html" in html_path.name


def test_training_phase_eta_accounts_for_slow_final_eval() -> None:
    started = datetime.now(timezone.utc) - timedelta(minutes=30)
    eta_seconds, _ = _estimate_sideboard_compare_eta(
        started_at=started,
        train_done=50,
        train_total=100,
        final_done_episodes=0,
        final_total_episodes=20,
        final_eval_weight=25,
        active_final_lives=[],
        in_final_eval_phase=False,
    )
    assert eta_seconds is not None
    # Weighted remaining work is much larger than unfinished training alone.
    assert eta_seconds > 3600


def test_final_eval_phase_eta_uses_observed_talishar_rate() -> None:
    started = datetime.now(timezone.utc) - timedelta(hours=2)
    eta_seconds, _ = _estimate_sideboard_compare_eta(
        started_at=started,
        train_done=200,
        train_total=200,
        final_done_episodes=5,
        final_total_episodes=40,
        final_eval_weight=25,
        active_final_lives=[{
            "episodes_completed": 5,
            "target_episodes": 20,
            "phase": "episodes",
            "episode_rate": 0.001,
        }],
        in_final_eval_phase=True,
    )
    assert eta_seconds is not None
    assert eta_seconds > 10_000


def test_collect_final_eval_eta_from_live_progress(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    candidate_dir = out_dir / "candidates" / "swap_01"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "final_eval").mkdir()
    (out_dir / "candidates_manifest.json").write_text(
        json.dumps({
            "started_at": (
                datetime.now(timezone.utc) - timedelta(hours=1)
            ).isoformat(),
            "play_episodes": 100,
            "final_eval_episodes": 20,
            "skip_final_eval": False,
            "candidates": [{
                "candidate_id": "swap_01",
                "label": "Swap",
                "game_deck": {"a_red": 40},
                "swaps": [],
            }],
        }),
        encoding="utf-8",
    )
    (candidate_dir / "candidate_result.json").write_text(
        json.dumps({
            "candidate_id": "swap_01",
            "play_win_rate": 0.55,
        }),
        encoding="utf-8",
    )
    (candidate_dir / "final_eval" / "final_eval_live.json").write_text(
        json.dumps({
            "episodes_completed": 4,
            "target_episodes": 20,
            "phase": "episodes",
            "episode_rate": 0.002,
            "runtime_backend": "HTTP Talishar",
        }),
        encoding="utf-8",
    )

    state = collect_sideboard_compare_state(out_dir)
    assert state["candidates"][0]["status"] == "final_eval"
    assert state["eta_seconds"] is not None
    assert state["eta_seconds"] > 3600
