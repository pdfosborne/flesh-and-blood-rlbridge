import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flesh_and_blood_rlbridge.card_text import TEXT_EMBED_VERSION, load_text_embedding_table, text_embed_dim
from flesh_and_blood_rlbridge.card_vocab import card_index, vocab_size
from flesh_and_blood_rlbridge.player_observation import ACTION_CAPACITY, PLAYER_OBS_DIM
from rl_agents.attention_policy_v2 import _AttentionPolicyValueV2


def test_text_embedding_table_shape() -> None:
    table = load_text_embedding_table()
    assert table.shape[0] == vocab_size()
    assert table.shape[1] == text_embed_dim()
    assert float(table[0].sum()) == 0.0


def test_text_embedding_row_matches_card_index() -> None:
    table = load_text_embedding_table()
    idx = card_index("surging_strike_red")
    assert idx > 0
    assert float(np.linalg.norm(table[idx])) > 0.0


def test_attention_v2_forward_shapes() -> None:
    model = _AttentionPolicyValueV2(ACTION_CAPACITY, d_model=64, n_layers=2, n_heads=4, seed=0)
    obs = torch.randn(4, PLAYER_OBS_DIM, dtype=torch.float32)
    logits = model.forward_logits(obs)
    values = model.forward_value(obs)
    assert logits.shape == (4, ACTION_CAPACITY)
    assert values.shape == (4, 1)


def test_attention_v2_frozen_embed_no_grad() -> None:
    model = _AttentionPolicyValueV2(8, d_model=32, n_layers=1, n_heads=4, seed=1)
    assert model.text_embed_table.requires_grad is False
    obs = torch.randn(2, PLAYER_OBS_DIM, dtype=torch.float32)
    logits = model.forward_logits(obs)
    loss = logits.sum()
    loss.backward()
    assert model.text_proj.weight.grad is not None
    assert model.hero_embed.weight.grad is not None


def test_attention_v2_state_dict_omits_frozen_table() -> None:
    model = _AttentionPolicyValueV2(8, d_model=32, n_layers=1, n_heads=4, seed=2)
    payload = model.state_dict_json()
    assert "text_embed_table" not in payload
    assert model.text_embed_version == TEXT_EMBED_VERSION
