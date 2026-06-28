"""Tests for parallel unified matchup shared buffer."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "training"))

from runtime_defaults import DEFAULT_PPO_ROLLOUT_BATCH  # noqa: E402


@pytest.fixture
def mock_policy() -> MagicMock:
    policy = MagicMock()
    policy._shared = MagicMock()
    return policy


def test_shared_buffer_flush_at_rollout_batch(mock_policy: MagicMock) -> None:
    from unified_parallel_training import UnifiedSharedExperienceBuffer  # noqa: PLC0415

    buffer = UnifiedSharedExperienceBuffer(rollout_batch=4)
    buffer.bind_policy(mock_policy)
    transitions = [
        {
            "obs_vec": [0.0],
            "next_obs_vec": [0.0],
            "step_order": idx,
        }
        for idx in range(4)
    ]

    with pytest.MonkeyPatch.context() as mp:
        flush_calls: list[int] = []

        def _fake_flush(policy, batch):
            flush_calls.append(len(batch))

        mp.setattr(
            "train_dual_agent_common._flush_unified_merged_transitions",
            _fake_flush,
        )
        buffer.extend_ppo(transitions)
        buffer.maybe_flush_ppo(mock_policy)

    assert flush_calls == [4]


def test_shared_buffer_thread_safe_extend(mock_policy: MagicMock) -> None:
    from unified_parallel_training import UnifiedSharedExperienceBuffer  # noqa: PLC0415

    buffer = UnifiedSharedExperienceBuffer(rollout_batch=DEFAULT_PPO_ROLLOUT_BATCH)
    buffer.bind_policy(mock_policy)
    errors: list[Exception] = []

    def _worker(start: int) -> None:
        try:
            for idx in range(20):
                buffer.extend_ppo(
                    [
                        {
                            "obs_vec": [float(start + idx)],
                            "next_obs_vec": [0.0],
                            "step_order": start + idx,
                        }
                    ]
                )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i * 100,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(buffer._ppo_accum) == 80


def test_workers_per_parallel_matchup() -> None:
    from unified_parallel_training import workers_per_parallel_matchup  # noqa: PLC0415

    assert workers_per_parallel_matchup(16, 4) == 4
    assert workers_per_parallel_matchup(16, 1) == 16
    assert workers_per_parallel_matchup(3, 4) == 1
