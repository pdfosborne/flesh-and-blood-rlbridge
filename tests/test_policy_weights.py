"""Tests for unified policy weight summaries."""

from __future__ import annotations

from flesh_and_blood_rlbridge.player_observation import PLAYER_OBS_DIM
from fab_bridge.policy_weights import summarize_policy_weights
from rl_agents.ppo import PPOAgent


def test_summarize_policy_weights_reports_architecture_and_fingerprint() -> None:
    agent = PPOAgent(n_actions=8, obs_dim=PLAYER_OBS_DIM)
    agent._init_nets(PLAYER_OBS_DIM)

    summary = summarize_policy_weights(agent)

    assert summary["initialized"] is True
    assert summary["obs_dim"] == PLAYER_OBS_DIM
    assert summary["n_actions"] == 8
    assert summary["param_count"] > 0
    assert summary["l2_norm"] > 0
    assert len(summary["fingerprint"]) == 12


def test_summarize_policy_weights_changes_after_update() -> None:
    agent = PPOAgent(n_actions=8, obs_dim=PLAYER_OBS_DIM)
    agent._init_nets(PLAYER_OBS_DIM)
    before = summarize_policy_weights(agent)

    import torch

    with torch.no_grad():
        for param in agent._shared.parameters():  # type: ignore[union-attr]
            param.add_(0.01)

    after = summarize_policy_weights(agent)
    assert after["fingerprint"] != before["fingerprint"]
    assert after["l2_norm"] != before["l2_norm"]
