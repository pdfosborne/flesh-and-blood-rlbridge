"""Tests for play episode outcome classification."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts" / "training"))

from play_outcome_stats import (  # noqa: E402
    classify_p1_episode_outcome,
    compute_eval_stability,
    summarize_p1_outcomes,
    win_rate_standard_error,
    win_rate_standard_error_from_rate,
)


def test_classify_outcomes() -> None:
    assert classify_p1_episode_outcome(p1_hp=20, p2_hp=0, terminated=True) == "win"
    assert classify_p1_episode_outcome(p1_hp=0, p2_hp=20, terminated=True) == "loss"
    assert classify_p1_episode_outcome(p1_hp=0, p2_hp=0, terminated=True) == "draw"
    assert classify_p1_episode_outcome(p1_hp=10, p2_hp=10, terminated=True) == "draw"
    assert classify_p1_episode_outcome(p1_hp=20, p2_hp=10, truncated=True) == "timeout"
    assert classify_p1_episode_outcome(p1_hp=20, p2_hp=10, truncated=True, terminated=True) == "timeout"
    assert classify_p1_episode_outcome(p1_hp=20, p2_hp=10, terminated=True) == "draw"
    assert classify_p1_episode_outcome(skipped=True) == "timeout"


def test_summarize_includes_draws_and_timeouts_in_denominator() -> None:
    summary = summarize_p1_outcomes(
        ["win", "loss", "draw", "timeout", "timeout"],
        episodes=5,
    )
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["draws"] == 1
    assert summary["timeouts"] == 2
    assert summary["win_rate"] == 0.2


def test_eval_stability_converged() -> None:
    stability = compute_eval_stability(
        [0.55, 0.56, 0.55, 0.56],
        window=3,
        max_std=0.02,
    )
    assert stability["status"] == "converged"
    assert stability["converged"] is True
    assert stability["rolling_std"] is not None
    assert stability["rolling_std"] <= 0.02


def test_eval_stability_learning() -> None:
    stability = compute_eval_stability([0.50, 0.60, 0.70], window=3, max_std=0.03)
    assert stability["status"] == "learning"
    assert stability["converged"] is False


def test_eval_stability_insufficient_points() -> None:
    stability = compute_eval_stability([0.62], min_points=2)
    assert stability["status"] == "insufficient"
    assert stability["converged"] is False


def test_eval_stability_marks_sufficient_when_budget_complete() -> None:
    stability = compute_eval_stability(
        [0.50, 0.70],
        episodes_completed=10000,
        target_episodes=10000,
    )
    assert stability["episodes_complete"] is True
    assert stability["sufficient"] is True


def test_win_rate_standard_error() -> None:
    assert win_rate_standard_error(50, 100) == pytest.approx(0.05)
    assert win_rate_standard_error(0, 0) is None
    assert win_rate_standard_error_from_rate(0.5, 100) == pytest.approx(0.05)
    assert win_rate_standard_error_from_rate(0.5, 0) is None
