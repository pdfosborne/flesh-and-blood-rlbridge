"""Tests for fast rollout mode dispatch and concurrent stepping."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from train_dual_agent_common import (  # noqa: E402
    Matchup,
    _FastRolloutSlot,
    _batched_fast_rollout_step,
    _batched_fast_rollout_step_concurrent,
    _EnvRolloutWorker,
    _run_parallel_batched_fast_episodes,
    _run_parallel_fast_episodes_threaded,
)
from runtime_defaults import normalize_rollout_mode  # noqa: E402


class _SleepingFastEnv:
    """Mock env whose fast_step_index sleeps to expose concurrency wins."""

    def __init__(self, *, delay: float = 0.05) -> None:
        self.delay = delay
        self._step = 0
        self._acting = 1

    def fast_reset(self, seed=None, starting_player_id: int = 1) -> dict:
        self._step = 0
        self._acting = int(starting_player_id)
        return self._state()

    def fast_step_index(self, action_index: int) -> dict:
        time.sleep(self.delay)
        self._step += 1
        self._acting = 2 if self._acting == 1 else 1
        terminated = self._step >= 3
        return {
            **self._state(),
            "reward": 0.0,
            "terminated": terminated,
            "truncated": False,
        }

    def _state(self) -> dict:
        obs = np.zeros(8, dtype=np.float64)
        return {
            "obs_vec": obs,
            "legal_count": 2,
            "acting_player_id": self._acting,
            "p1_health": 20,
            "p2_health": 20,
            "p1_deck": 40,
            "p2_deck": 40,
            "turn_no": 1,
        }


def _make_slot(delay: float = 0.05) -> _FastRolloutSlot:
    env = _SleepingFastEnv(delay=delay)
    slot = _FastRolloutSlot(
        env=env,  # type: ignore[arg-type]
        state=env.fast_reset(),
        p1_rng=np.random.default_rng(0),
        p2_rng=np.random.default_rng(1),
    )
    return slot


def _mock_policy() -> SimpleNamespace:
    policy = SimpleNamespace(n_actions=8)

    def predict_batch(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch = obs.shape[0]
        return (
            np.zeros((batch, policy.n_actions), dtype=np.float64),
            np.zeros(batch, dtype=np.float64),
        )

    policy.predict_batch = predict_batch
    return policy


def test_normalize_rollout_mode_accepts_known_values() -> None:
    assert normalize_rollout_mode("batched") == "batched"
    assert normalize_rollout_mode("batched_concurrent") == "batched_concurrent"
    with pytest.raises(ValueError):
        normalize_rollout_mode("invalid-mode")


def test_batched_concurrent_faster_than_sequential_for_multiple_slots() -> None:
    delay = 0.04
    slots = [_make_slot(delay=delay) for _ in range(4)]
    policy = _mock_policy()

    t0 = time.perf_counter()
    _batched_fast_rollout_step(
        slots,
        policy,
        policy,
        warmup=False,
        max_steps=10,
    )
    sequential_elapsed = time.perf_counter() - t0

    slots = [_make_slot(delay=delay) for _ in range(4)]
    workers = [_EnvRolloutWorker(slot) for slot in slots]
    try:
        t0 = time.perf_counter()
        _batched_fast_rollout_step_concurrent(
            slots,
            workers,
            policy,
            policy,
            warmup=False,
            max_steps=10,
        )
        concurrent_elapsed = time.perf_counter() - t0
    finally:
        for worker in workers:
            worker.shutdown()

    assert concurrent_elapsed < sequential_elapsed * 0.75


def test_run_parallel_batched_fast_episodes_respects_rollout_mode(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_step(slots, p1, p2, *, warmup, max_steps):
        calls.append("batched")

    def _fake_step_concurrent(slots, workers, p1, p2, *, warmup, max_steps):
        calls.append("concurrent")

    monkeypatch.setattr(
        "train_dual_agent_common._batched_fast_rollout_step",
        _fake_step,
    )
    monkeypatch.setattr(
        "train_dual_agent_common._batched_fast_rollout_step_concurrent",
        _fake_step_concurrent,
    )

    env = MagicMock()
    env.fast_reset.return_value = {
        "obs_vec": np.zeros(4),
        "legal_count": 1,
        "acting_player_id": 1,
        "p1_health": 20,
        "p2_health": 20,
        "p1_deck": 40,
        "p2_deck": 40,
        "turn_no": 1,
    }
    env.fast_step_index.return_value = {
        "obs_vec": np.zeros(4),
        "legal_count": 1,
        "acting_player_id": 2,
        "reward": 0.0,
        "terminated": True,
        "truncated": False,
        "p1_health": 20,
        "p2_health": 20,
        "p1_deck": 40,
        "p2_deck": 40,
        "turn_no": 1,
    }
    policy = _mock_policy()

    _run_parallel_batched_fast_episodes(
        [env],
        policy,
        policy,
        max_steps=1,
        warmup=False,
        episode_indices=[0],
        seed_base=0,
        rollout_mode="batched",
    )
    _run_parallel_batched_fast_episodes(
        [env],
        policy,
        policy,
        max_steps=1,
        warmup=False,
        episode_indices=[0],
        seed_base=0,
        rollout_mode="batched_concurrent",
    )
    assert calls == ["batched", "concurrent"]


def test_threaded_episodes_honors_swap_envs() -> None:
    matchup = Matchup(
        name="a-vs-b",
        p1_deck="deck_a",
        p2_deck="deck_b",
        description="test",
        p1_hero="hero_a",
        p2_hero="hero_b",
    )
    env = MagicMock()
    swap_env = MagicMock()
    for mock_env in (env, swap_env):
        mock_env.fast_reset.return_value = {
            "obs_vec": np.zeros(4),
            "legal_count": 1,
            "acting_player_id": 1,
            "p1_health": 20,
            "p2_health": 0,
            "p1_deck": 40,
            "p2_deck": 40,
            "turn_no": 1,
        }
        mock_env.fast_step_index.return_value = {
            "obs_vec": np.zeros(4),
            "legal_count": 1,
            "acting_player_id": 2,
            "reward": 0.0,
            "terminated": True,
            "truncated": False,
            "p1_health": 20,
            "p2_health": 0,
            "p1_deck": 40,
            "p2_deck": 40,
            "turn_no": 1,
        }
    policy = _mock_policy()
    results = _run_parallel_fast_episodes_threaded(
        [env],
        policy,
        policy,
        max_steps=1,
        warmup=True,
        episode_indices=[1],
        seed_base=0,
        swap_envs=[swap_env],
        max_workers=1,
        matchup=matchup,
    )
    assert results[0]["active_p1_hero"] == "hero_b"
    assert results[0]["active_p2_hero"] == "hero_a"
    swap_env.fast_reset.assert_called_once()
    env.fast_reset.assert_not_called()
