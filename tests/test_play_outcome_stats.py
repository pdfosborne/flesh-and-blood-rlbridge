"""Tests for play episode outcome classification."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts" / "training"))

from play_outcome_stats import (  # noqa: E402
    OutcomeCounters,
    absolute_p1_p2_deck_from_env,
    absolute_p1_p2_hp_from_env,
    absolute_p1_p2_hp_from_obs,
    classify_p1_episode_outcome,
    compute_eval_stability,
    legacy_hero_rates_from_seat_summary,
    nominal_hero_slot,
    summarize_p1_outcomes,
    winning_hero_id_from_seat_outcome,
    win_rate_standard_error,
    win_rate_standard_error_from_rate,
)


def test_classify_outcomes() -> None:
    assert classify_p1_episode_outcome(p1_hp=20, p2_hp=0, terminated=True) == "win"
    assert classify_p1_episode_outcome(p1_hp=20, p2_hp=0) == "win"
    assert classify_p1_episode_outcome(p1_hp=0, p2_hp=20, terminated=True) == "loss"
    assert classify_p1_episode_outcome(p1_hp=0, p2_hp=20) == "loss"
    assert classify_p1_episode_outcome(p1_hp=0, p2_hp=0, terminated=True) == "draw"
    assert classify_p1_episode_outcome(p1_hp=10, p2_hp=10, terminated=True) == "timeout"
    assert (
        classify_p1_episode_outcome(
            p1_hp=10,
            p2_hp=10,
            p1_deck=0,
            p2_deck=0,
            terminated=True,
        )
        == "timeout"
    )
    assert classify_p1_episode_outcome(p1_hp=20, p2_hp=10, truncated=True) == "timeout"
    assert classify_p1_episode_outcome(p1_hp=20, p2_hp=0, truncated=True, terminated=True) == "win"
    assert classify_p1_episode_outcome(p1_hp=20, p2_hp=10, truncated=True, terminated=True) == "timeout"
    assert classify_p1_episode_outcome(p1_hp=20, p2_hp=10, terminated=True) == "timeout"
    assert classify_p1_episode_outcome(skipped=True) == "timeout"


def test_absolute_p1_p2_hp_from_obs_swaps_when_p2_acting() -> None:
    p1_hp, p2_hp = absolute_p1_p2_hp_from_obs(
        {
            "actingPlayerID": 2,
            "playerHealth": 0,
            "opponentHealth": 14,
        }
    )
    assert p1_hp == 14.0
    assert p2_hp == 0.0
    assert classify_p1_episode_outcome(
        p1_hp=p1_hp,
        p2_hp=p2_hp,
        terminated=True,
    ) == "win"


def test_absolute_p1_p2_hp_from_env_uses_acting_player_view() -> None:
    class _Env:
        _acting_player_id = 2
        _player_hp = 0
        _opp_hp = 18
        _last_state = {"playerHealth": 0, "opponentHealth": 18}

    p1_hp, p2_hp = absolute_p1_p2_hp_from_env(_Env())
    assert p1_hp == 18
    assert p2_hp == 0
    assert classify_p1_episode_outcome(
        p1_hp=p1_hp,
        p2_hp=p2_hp,
        terminated=True,
    ) == "win"

    p1_deck, p2_deck = absolute_p1_p2_deck_from_env(
        type(
            "_DeckEnv",
            (),
            {
                "_acting_player_id": 2,
                "_last_state": {
                    "playerDeckCount": 3,
                    "opponentDeckCount": 11,
                },
            },
        )()
    )
    assert p1_deck == 11
    assert p2_deck == 3


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
    assert summary["win_rate_decided"] == pytest.approx(1 / 3)


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


def test_fast_outcome_requires_lethal_hp() -> None:
    from play_outcome_stats import classify_p1_fast_episode_outcome  # noqa: PLC0415

    outcome, anomaly = classify_p1_fast_episode_outcome(
        {
            "terminated": True,
            "truncated": False,
            "winner": 0,
            "p1_health": 18,
            "p2_health": 12,
        }
    )
    assert outcome == "timeout"
    assert anomaly is not None
    assert "disagrees" in anomaly

    outcome, anomaly = classify_p1_fast_episode_outcome(
        {
            "terminated": True,
            "truncated": False,
            "winner": 0,
            "p1_health": 20,
            "p2_health": 0,
        }
    )
    assert outcome == "win"
    assert anomaly is None

    outcome, _ = classify_p1_fast_episode_outcome(
        {
            "terminated": True,
            "truncated": False,
            "winner": 1,
            "p1_health": 0,
            "p2_health": 15,
        }
    )
    assert outcome == "loss"

    _, stale_winner_anomaly = classify_p1_fast_episode_outcome(
        {
            "terminated": True,
            "truncated": False,
            "winner": 0,
            "p1_health": 0,
            "p2_health": 12,
        }
    )
    assert stale_winner_anomaly is not None

    _, fixed_winner = classify_p1_fast_episode_outcome(
        {
            "terminated": True,
            "truncated": False,
            "winner": 1,
            "p1_health": 0,
            "p2_health": 12,
        }
    )
    assert fixed_winner is None


def test_winning_hero_id_from_seat_outcome() -> None:
    assert winning_hero_id_from_seat_outcome(
        "win", active_p1_hero="a", active_p2_hero="b"
    ) == "a"
    assert winning_hero_id_from_seat_outcome(
        "loss", active_p1_hero="a", active_p2_hero="b"
    ) == "b"
    assert winning_hero_id_from_seat_outcome(
        "timeout", active_p1_hero="a", active_p2_hero="b"
    ) is None


def test_outcome_counters_no_swap_seat_equals_hero() -> None:
    counters = OutcomeCounters()
    for outcome in ("win", "loss", "win"):
        counters.record_seat_outcome(
            outcome,
            active_p1_hero="hero1",
            active_p2_hero="hero2",
            nominal_hero1="hero1",
            nominal_hero2="hero2",
        )
    summary = counters.to_summary(3)
    assert summary["seats"]["p1_wins"] == 2
    assert summary["heroes"]["hero1_wins"] == 2
    assert summary["seats"]["p2_wins"] == 1
    assert summary["heroes"]["hero2_wins"] == 1


def test_outcome_counters_swap_inverts_hero_attribution() -> None:
    """All P1-seat losses with alternating swap → 50/50 nominal heroes."""
    counters = OutcomeCounters()
    for ep in range(10):
        swapped = ep % 2 == 1
        if swapped:
            active_p1, active_p2 = "hero2", "hero1"
        else:
            active_p1, active_p2 = "hero1", "hero2"
        counters.record_seat_outcome(
            "loss",
            active_p1_hero=active_p1,
            active_p2_hero=active_p2,
            nominal_hero1="hero1",
            nominal_hero2="hero2",
        )
    summary = counters.to_summary(10, deck_swap_eval=True)
    assert summary["seats"]["p1_wins"] == 0
    assert summary["seats"]["p2_wins"] == 10
    assert summary["heroes"]["hero1_wins"] == 5
    assert summary["heroes"]["hero2_wins"] == 5


def test_outcome_counters_draw_timeout_no_hero_win() -> None:
    counters = OutcomeCounters()
    counters.record_seat_outcome(
        "draw",
        active_p1_hero="h1",
        active_p2_hero="h2",
        nominal_hero1="h1",
        nominal_hero2="h2",
    )
    counters.record_seat_outcome(
        "timeout",
        active_p1_hero="h1",
        active_p2_hero="h2",
        nominal_hero1="h1",
        nominal_hero2="h2",
    )
    summary = counters.to_summary(2)
    assert summary["heroes"]["hero1_wins"] == 0
    assert summary["heroes"]["hero2_wins"] == 0
    assert summary["draws"] == 1
    assert summary["timeouts"] == 1


def test_nominal_hero_slot() -> None:
    assert nominal_hero_slot("a", hero1_id="a", hero2_id="b") == "hero1"
    assert nominal_hero_slot("b", hero1_id="a", hero2_id="b") == "hero2"
    assert nominal_hero_slot("c", hero1_id="a", hero2_id="b") is None


def test_legacy_hero_rates_from_seat_summary() -> None:
    h1, h2 = legacy_hero_rates_from_seat_summary(
        {"p1_win_rate": 0.0, "p2_win_rate": 1.0},
        deck_swap_eval=False,
    )
    assert h1 == 0.0
    assert h2 == 1.0
    h1, h2 = legacy_hero_rates_from_seat_summary(
        {
            "heroes": {"hero1_win_rate": 0.6, "hero2_win_rate": 0.4},
        },
    )
    assert h1 == 0.6
    assert h2 == 0.4
