"""Schema v3 observation block encoding tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.card_vocab import card_index_normalized
from flesh_and_blood_rlbridge.deck_context import EpisodeContext
from flesh_and_blood_rlbridge.player_observation import (
    EFFECT_OFF,
    LAYER_OFF,
    PLAYER_OBS_DIM,
    PLAYED_OPP_OFF,
    PLAYED_SELF_OFF,
    player_observation_vector,
)


def _base_state() -> dict:
    return {
        "actingPlayerID": 1,
        "playerHealth": 20,
        "opponentHealth": 20,
        "turnNo": 3,
        "turnPhase": "m",
        "playerHand": [],
        "playerHandSize": 0,
        "opponentHandSize": 0,
        "playerDeckCount": 40,
        "opponentDeckCount": 40,
        "playerPitchCount": 0,
    }


def test_play_history_preserves_order() -> None:
    cards = ["snatch_red", "wounding_blow_red", "surging_strike_blue"]
    state = {
        **_base_state(),
        "playHistory": {
            "player": {"namesOfCardsPlayed": cards},
            "opponent": {"namesOfCardsPlayed": ["enchanting_melody_blue"]},
        },
    }
    vec = player_observation_vector(state, [])
    self_block = vec[PLAYED_SELF_OFF:PLAYED_OPP_OFF]
    assert float(self_block[0]) == card_index_normalized(cards[0])
    assert float(self_block[2]) == card_index_normalized(cards[1])
    assert float(self_block[4]) == card_index_normalized(cards[2])
    assert float(self_block[1]) > 0.0
    opp_block = vec[PLAYED_OPP_OFF:EFFECT_OFF]
    assert float(opp_block[0]) == card_index_normalized("enchanting_melody_blue")


def test_turn_effects_and_layers_encode() -> None:
    state = {
        **_base_state(),
        "currentTurnEffects": [
            {
                "effectId": "dominate_red",
                "player": 1,
                "usesRemaining": 2,
                "isCombatEffect": True,
            }
        ],
        "layers": [
            {
                "layerType": "TRIGGER",
                "cardId": "snatch_red",
                "player": 2,
            }
        ],
    }
    vec = player_observation_vector(state, [])
    effect_block = vec[EFFECT_OFF : EFFECT_OFF + 4]
    assert float(effect_block[0]) == card_index_normalized("dominate_red")
    assert float(effect_block[1]) == 0.5
    assert float(effect_block[2]) > 0.0
    assert float(effect_block[3]) == 1.0

    layer_block = vec[LAYER_OFF : LAYER_OFF + 4]
    assert float(layer_block[0]) > 0.0
    assert float(layer_block[1]) == card_index_normalized("snatch_red")
    assert float(layer_block[2]) == 1.0
    assert float(layer_block[3]) == 1.0


def test_empty_v3_blocks_are_zero() -> None:
    ctx = EpisodeContext(format="silver_age")
    vec = player_observation_vector(_base_state(), [], episode_context=ctx)
    assert vec.shape == (PLAYER_OBS_DIM,)
    tail = vec[PLAYED_SELF_OFF:]
    assert float(np.sum(np.abs(tail))) == 0.0
