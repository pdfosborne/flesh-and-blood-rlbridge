import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.card_vocab import hero_index_normalized, hero_vocab_size, vocab_size
from flesh_and_blood_rlbridge.deck_context import EpisodeContext
from flesh_and_blood_rlbridge.obs_tokenizer import ObsTokenLayout, build_token_features, denorm_hero_index
from flesh_and_blood_rlbridge.player_observation import (
    CONTEXT_DIM,
    DECK_OFF,
    EFFECT_OFF,
    HAND_OFF,
    HAND_SLOT_DIM,
    HAND_SLOTS,
    LAYER_OFF,
    PLAYER_OBS_DIM,
    PLAYED_SELF_OFF,
    SCALAR_OFF,
    player_observation_vector,
)


def _minimal_state() -> dict:
    return {
        "actingPlayerID": 1,
        "playerHealth": 20,
        "opponentHealth": 20,
        "turnNo": 1,
        "turnPhase": "m",
        "playerHand": [],
        "playerHandSize": 0,
        "opponentHandSize": 0,
        "playerDeckCount": 40,
        "opponentDeckCount": 40,
        "playerPitchCount": 0,
        "legalActions": [{"label": "pass", "action_code": 0}],
    }


def test_obs_layout_matches_player_obs_dim() -> None:
    layout = ObsTokenLayout()
    assert layout.obs_dim == PLAYER_OBS_DIM
    assert layout.board_token_count == (
        1 + HAND_SLOTS + layout.zone_slots_total + 1 + 8 + layout.played_slots_total + layout.effect_slots + layout.layer_slots
    )


def test_tokenizer_slices_match_obs_vector() -> None:
    ctx = EpisodeContext(
        self_hero_id="hero_kayo",
        opp_hero_id="hero_iyslander",
        format="silver_age",
        first_player=1,
    )
    vec = player_observation_vector(_minimal_state(), [{"label": "pass"}], episode_context=ctx)
    obs = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
    batch = build_token_features(obs)

    assert batch.hero_self_ids.shape == (1,)
    assert batch.hero_opp_ids.shape == (1,)
    assert batch.deck_card_ids.shape[1] == CONTEXT_DIM - DECK_OFF
    assert batch.scalars.shape == (1, 26)
    assert batch.hand.shape == (1, HAND_SLOTS, HAND_SLOT_DIM)
    assert batch.zones.shape[1] == ObsTokenLayout().zone_slots_total
    assert batch.combat_chain.shape == (1, 8, 3)
    assert batch.played_history.shape == (1, 24, 2)
    assert batch.turn_effects.shape == (1, 12, 4)
    assert batch.layers.shape == (1, 8, 4)
    assert batch.board_padding_mask.shape == (1, ObsTokenLayout().board_token_count)

    assert float(obs[0, SCALAR_OFF].item()) == float(batch.scalars[0, 0].item())
    hand_dims = HAND_SLOTS * HAND_SLOT_DIM
    assert torch.allclose(obs[0, HAND_OFF : HAND_OFF + hand_dims], batch.hand[0].reshape(-1))


def test_empty_hand_slots_are_padded() -> None:
    vec = player_observation_vector(_minimal_state(), [{"label": "pass"}])
    obs = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
    batch = build_token_features(obs)
    assert not batch.board_padding_mask[0, 0].item()
    assert batch.board_padding_mask[0, 1:9].all().item()


def test_hero_id_round_trip() -> None:
    norm = torch.tensor([hero_index_normalized("hero_kayo")], dtype=torch.float32)
    ids = denorm_hero_index(norm)
    assert ids.item() > 0
    assert ids.item() < hero_vocab_size()


def test_deck_indices_match_obs_slice() -> None:
    ctx = EpisodeContext(
        self_hero_id="hero_kayo",
        opp_hero_id="hero_iyslander",
        format="silver_age",
        first_player=1,
    )
    indices = ctx.self_deck_indices()
    vec = player_observation_vector(_minimal_state(), [{"label": "pass"}], episode_context=ctx)
    obs = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
    batch = build_token_features(obs)
    deck_slice = obs[0, DECK_OFF:CONTEXT_DIM]
    if indices:
        assert torch.any(batch.deck_card_ids[0] > 0)
    assert batch.deck_card_ids.max().item() <= vocab_size()
    assert torch.allclose(
        batch.deck_card_ids[0].float(),
        torch.round(deck_slice * float(vocab_size())),
        atol=1.0,
    )
