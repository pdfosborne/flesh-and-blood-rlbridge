"""Smoke tests for merged P1+P2 transition updates."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts" / "training"))
sys.path.insert(0, str(_REPO / "src"))

from flesh_and_blood_rlbridge.player_observation import PLAYER_OBS_DIM  # noqa: E402
from rl_agents.ppo import PPOAgent  # noqa: E402
from train_dual_agent_common import (  # noqa: E402
    _flush_unified_ppo_buffers,
    _merge_episode_transitions,
)


def test_merge_episode_transitions_chronological() -> None:
    p1 = [{"step_order": 0, "reward": 1.0}, {"step_order": 2, "reward": 1.0}]
    p2 = [{"step_order": 1, "reward": -1.0}]
    merged = _merge_episode_transitions(p1, p2)
    assert [t["step_order"] for t in merged] == [0, 1, 2]


def test_unified_ppo_flush_single_update(monkeypatch) -> None:
    policy = PPOAgent(n_actions=8, obs_dim=PLAYER_OBS_DIM)
    policy._init_nets(PLAYER_OBS_DIM)
    calls: list[int] = []

    def _fake_ppo(agent, buf, next_vec):  # noqa: ANN001
        del buf, next_vec
        calls.append(id(agent))

    monkeypatch.setattr("train_dual_agent_common._ppo_update", _fake_ppo)

    obs = np.zeros(PLAYER_OBS_DIM, dtype=np.float64)
    p1_trans = [
        {
            "obs_vec": obs,
            "action": 0,
            "reward": 1.0,
            "value": 0.0,
            "log_prob": 0.0,
            "done": 0.0,
            "n_legal": 2,
            "next_obs_vec": obs,
            "step_order": 0,
        }
    ]
    p2_trans = [
        {
            "obs_vec": obs,
            "action": 1,
            "reward": -1.0,
            "value": 0.0,
            "log_prob": 0.0,
            "done": 1.0,
            "n_legal": 2,
            "next_obs_vec": obs,
            "step_order": 1,
        }
    ]
    _flush_unified_ppo_buffers(policy, p1_trans, p2_trans)
    assert len(calls) == 1
    assert calls[0] == id(policy)
