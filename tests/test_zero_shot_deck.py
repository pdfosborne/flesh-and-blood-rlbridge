import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flesh_and_blood_rlbridge.card_vocab import card_index
from flesh_and_blood_rlbridge.deck_context import EpisodeContext
from flesh_and_blood_rlbridge.player_observation import (
    ACTION_CAPACITY,
    DECK_OFF,
    HAND_OFF,
    PLAYER_OBS_DIM,
    player_observation_vector,
)
from rl_agents.attention_policy_v2 import _AttentionPolicyValueV2


def test_zero_shot_deck_forward_pass() -> None:
    """Held-out card IDs in deck/hand should still produce finite policy logits."""
    held_out = "a_drop_in_the_ocean_blue"
    idx = card_index(held_out)
    assert idx > 0

    ctx = EpisodeContext(
        self_hero_id="ira_crimson_haze",
        opp_hero_id="briar",
        format="silver_age",
        self_deck_counts={held_out: 3},
        first_player=1,
    )
    state = {
        "actingPlayerID": 1,
        "playerHealth": 20,
        "opponentHealth": 20,
        "turnPhase": "m",
        "playerHand": [
            {
                "cardNumber": held_out,
                "cost": 0,
                "pitch": 3,
                "power": 0,
                "defense": 0,
            }
        ],
    }
    vec = player_observation_vector(state, [{"label": "pass"}], episode_context=ctx)
    assert vec.shape == (PLAYER_OBS_DIM,)
    assert float(vec[DECK_OFF]) > 0.0
    assert float(vec[HAND_OFF]) > 0.0

    model = _AttentionPolicyValueV2(ACTION_CAPACITY, d_model=32, n_layers=1, n_heads=4, seed=0)
    obs = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
    logits = model.forward_logits(obs)
    assert logits.shape == (1, ACTION_CAPACITY)
    assert np.isfinite(logits.detach().cpu().numpy()).all()
