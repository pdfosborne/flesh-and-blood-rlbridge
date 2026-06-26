import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flesh_and_blood_rlbridge.player_observation import ACTION_CAPACITY, PLAYER_OBS_DIM
from rl_agents.attention_policy import _AttentionPolicyValue


def test_attention_forward_shapes() -> None:
    model = _AttentionPolicyValue(ACTION_CAPACITY, d_model=64, n_layers=2, n_heads=4, seed=0)
    obs = torch.randn(4, PLAYER_OBS_DIM, dtype=torch.float32)
    logits = model.forward_logits(obs)
    values = model.forward_value(obs)
    assert logits.shape == (4, ACTION_CAPACITY)
    assert values.shape == (4, 1)


def test_attention_gradient_flow() -> None:
    model = _AttentionPolicyValue(8, d_model=32, n_layers=1, n_heads=4, seed=1)
    obs = torch.randn(2, PLAYER_OBS_DIM, dtype=torch.float32, requires_grad=False)
    logits = model.forward_logits(obs)
    loss = logits.sum()
    loss.backward()
    assert model.hero_embed.weight.grad is not None
    assert model.cross_attn.in_proj_weight.grad is not None


def test_predict_policy_value_numpy() -> None:
    model = _AttentionPolicyValue(16, d_model=32, n_layers=1, n_heads=4, seed=2)
    x = np.random.randn(PLAYER_OBS_DIM).astype(np.float64)
    logits, values = model.predict_policy_value(x)
    assert logits.shape == (1, 16)
    assert values.shape == (1, 1)


def test_state_dict_json_round_trip() -> None:
    model = _AttentionPolicyValue(8, d_model=32, n_layers=1, n_heads=4, seed=3)
    other = _AttentionPolicyValue(8, d_model=32, n_layers=1, n_heads=4, seed=3)
    payload = model.state_dict_json()
    other.load_state_dict_json(payload, torch.device("cpu"))
    model.eval()
    other.eval()
    obs = torch.randn(1, PLAYER_OBS_DIM, dtype=torch.float32)
    assert torch.allclose(model.forward_logits(obs), other.forward_logits(obs), rtol=1e-5, atol=1e-5)
