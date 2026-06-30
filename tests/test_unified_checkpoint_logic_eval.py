"""Tests for unified checkpoint eval vs C++ logic policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts" / "training"))
sys.path.insert(0, str(_REPO / "src"))

from flesh_and_blood_rlbridge.player_observation import PLAYER_OBS_DIM  # noqa: E402
from rl_agents.ppo import PPOAgent  # noqa: E402
from train_dual_agent_common import Matchup, _CheckpointEvalTracker  # noqa: E402


def _sample_metrics(*, p1_wr: float, p2_wr: float) -> dict:
    episodes = 10
    p1_wins = int(round(p1_wr * episodes))
    p2_wins = int(round(p2_wr * episodes))
    return {
        "episodes": episodes,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": 0,
        "timeouts": 0,
        "errors": 0,
        "p1_win_rate": p1_wins / episodes,
        "p2_win_rate": p2_wins / episodes,
        "draw_rate": 0.0,
        "timeout_rate": 0.0,
        "runtime_backend": "C++ engine fast sampled eval",
    }


def test_evaluate_logic_vs_logic(monkeypatch) -> None:
    from train_play import LOGIC_POLICY, evaluate_logic_vs_logic  # noqa: PLC0415

    calls: list[tuple[object, object]] = []

    def _fake_evaluate_policy_matchup(
        matchup: Matchup,
        p1_policy: object,
        p2_policy: object,
        **kwargs: object,
    ) -> dict:
        del matchup, kwargs
        calls.append((p1_policy, p2_policy))
        return _sample_metrics(p1_wr=0.55, p2_wr=0.45)

    monkeypatch.setattr(
        "train_play.evaluate_policy_matchup",
        _fake_evaluate_policy_matchup,
    )
    matchup = Matchup(
        name="a-vs-b",
        p1_deck="deck_a",
        p2_deck="deck_b",
        description="test",
        p1_hero="a",
        p2_hero="b",
        cpp_engine_dir="/tmp/engine",
    )

    result = evaluate_logic_vs_logic(
        matchup,
        base_url="http://localhost:8080/game",
        game_format="silver_age",
        max_steps=100,
        episodes=10,
        seed=42,
    )

    assert calls == [(LOGIC_POLICY, LOGIC_POLICY)]
    assert result["p1_win_rate"] == pytest.approx(0.55, abs=0.06)
    assert result["p1_policy"] == "logic"
    assert result["p2_policy"] == "logic"


def test_evaluate_agent_vs_logic_both_seats(monkeypatch) -> None:
    from train_play import evaluate_agent_vs_logic_both_seats  # noqa: PLC0415

    calls: list[tuple[str, str]] = []

    def _fake_evaluate_policy_matchup(
        matchup: Matchup,
        p1_policy: object,
        p2_policy: object,
        **kwargs: object,
    ) -> dict:
        del matchup, kwargs
        from train_play import LOGIC_POLICY  # noqa: PLC0415

        if p1_policy is LOGIC_POLICY:
            calls.append(("logic", "agent"))
            return _sample_metrics(p1_wr=0.4, p2_wr=0.6)
        calls.append(("agent", "logic"))
        return _sample_metrics(p1_wr=0.7, p2_wr=0.3)

    monkeypatch.setattr(
        "train_play.evaluate_policy_matchup",
        _fake_evaluate_policy_matchup,
    )
    matchup = Matchup(
        name="a-vs-b",
        p1_deck="deck_a",
        p2_deck="deck_b",
        description="test",
        p1_hero="a",
        p2_hero="b",
        cpp_engine_dir="/tmp/engine",
    )
    policy = PPOAgent(n_actions=8, obs_dim=PLAYER_OBS_DIM)
    policy._init_nets(PLAYER_OBS_DIM)

    result = evaluate_agent_vs_logic_both_seats(
        matchup,
        policy,
        base_url="http://localhost:8080/game",
        game_format="silver_age",
        max_steps=100,
        episodes=10,
        seed=42,
    )

    assert calls == [("agent", "logic"), ("logic", "agent")]
    assert result["agent_p1_seat"]["agent_win_rate"] == 0.7
    assert result["agent_p2_seat"]["agent_win_rate"] == 0.6
    assert result["agent_p1_seat"]["p2_policy"] == "logic"
    assert result["agent_p2_seat"]["p1_policy"] == "logic"


def test_checkpoint_eval_tracker_records_vs_logic(monkeypatch, tmp_path: Path) -> None:
    policy = PPOAgent(n_actions=8, obs_dim=PLAYER_OBS_DIM)
    policy._init_nets(PLAYER_OBS_DIM)
    obs = np.zeros(PLAYER_OBS_DIM, dtype=np.float64)
    policy.predict_policy_value(obs)

    matchup = Matchup(
        name="a-vs-b",
        p1_deck="deck_a",
        p2_deck="deck_b",
        description="test",
        p1_hero="a",
        p2_hero="b",
        cpp_engine_dir="/tmp/engine",
    )

    logic_calls = 0

    def _fake_self_play(*_args: object, **kwargs: object) -> dict:
        label = str(kwargs.get("eval_label", ""))
        if "logic win% vs logic" in label.lower():
            nonlocal logic_calls
            logic_calls += 1
            return _sample_metrics(p1_wr=0.54, p2_wr=0.46)
        assert kwargs.get("live_progress_path") is not None
        return _sample_metrics(p1_wr=0.5, p2_wr=0.5)

    def _fake_vs_logic(*_args: object, **kwargs: object) -> dict:
        assert kwargs.get("live_progress_path") is not None
        return {
            "agent_p1_seat": {**_sample_metrics(p1_wr=0.6, p2_wr=0.4), "agent_win_rate": 0.6},
            "agent_p2_seat": {**_sample_metrics(p1_wr=0.3, p2_wr=0.7), "agent_win_rate": 0.7},
        }

    monkeypatch.setattr(
        "train_play._evaluate_p1_vs_fixed_opponent",
        _fake_self_play,
    )
    monkeypatch.setattr(
        "train_play.evaluate_agent_vs_logic_both_seats",
        _fake_vs_logic,
    )
    monkeypatch.setattr(
        "train_play.evaluate_logic_vs_logic",
        _fake_self_play,
    )
    monkeypatch.setattr(
        "train_dual_agent_common._save_unified_selfplay_checkpoint",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "train_dual_agent_common._write_unified_checkpoint_eval_scope",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "train_dual_agent_common.maybe_refresh_unified_dashboard",
        lambda *_args, **_kwargs: None,
    )

    tracker = _CheckpointEvalTracker(
        matchup=matchup,
        base_url="http://localhost:8080/game",
        game_format="silver_age",
        max_steps=100,
        n_episodes=100,
        checkpoint_interval=10,
        checkpoint_eval_episodes=10,
        p1_policy=policy,
        p2_policy=policy,
        seed=0,
        out_dir=tmp_path,
        eval_agent_vs_logic=True,
        eval_logic_vs_logic=False,
    )
    tracker.on_parallel_progress(10)
    tracker.on_parallel_progress(20)

    assert len(tracker.log) == 2
    record = tracker.log[0]
    assert record["p1_win_rate"] == 0.5
    assert record["vs_logic"]["agent_p1_seat"]["agent_win_rate"] == 0.6
    assert record["vs_logic"]["agent_p2_seat"]["agent_win_rate"] == 0.7
    assert "logic_vs_logic" not in record
    assert logic_calls == 0
    ckpt_json = (
        tmp_path
        / matchup.name
        / "checkpoint_eval"
        / "episode_000010"
        / "checkpoint_eval.json"
    )
    assert ckpt_json.is_file()
    from fab_bridge.cpp_eval_live_dashboard import UNIFIED_CHECKPOINT_EVAL_LIVE

    assert (tmp_path / UNIFIED_CHECKPOINT_EVAL_LIVE).parent == tmp_path.resolve()
