"""Talishar-shaped state encoding tests for player-fair observations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.deck_context import EpisodeContext  # noqa: E402
from flesh_and_blood_rlbridge.card_vocab import card_index
from flesh_and_blood_rlbridge.player_observation import (  # noqa: E402
    PLAYER_OBS_DIM,
    player_observation_vector,
)


def _minimal_state(*, acting: int = 1) -> dict:
    return {
        "actingPlayerID": acting,
        "playerHealth": 20,
        "opponentHealth": 18,
        "turnNo": 2,
        "turnPhase": "M",
        "playerHandSize": 1,
        "opponentHandSize": 4,
        "playerDeckCount": 35,
        "opponentDeckCount": 36,
        "playerPitchCount": 1,
        "playerHand": [
            {
                "cardID": "WTR078",
                "action": 27,
                "cost": 0,
                "pitch": 3,
                "power": 3,
                "defense": 3,
            }
        ],
        "havePriority": True,
    }


def test_opponent_hand_not_leaked_in_vector() -> None:
    card_id = "absorb_in_aether_blue"
    assert card_index(card_id) > 0
    episode = EpisodeContext(
        self_hero_id="hero_briar",
        opp_hero_id="hero_kayo",
        format="silver_age",
        self_deck_counts={card_id: 3},
        first_player=1,
    )
    state = _minimal_state()
    state["playerHand"] = [
        {
            "cardID": card_id,
            "action": 27,
            "cost": 0,
            "pitch": 3,
            "power": 3,
            "defense": 3,
        }
    ]
    raw = {
        **state,
        "opponentHand": [{"cardID": "SECRET_CARD"}],
        "playerEquipment": [],
        "opponentEquipment": [],
    }
    vec = player_observation_vector(
        state,
        [{"index": 0}, {"index": 1}],
        episode_context=episode,
        raw_talishar_state=raw,
    )
    assert vec.shape == (PLAYER_OBS_DIM,)
    hand_start = 4 + 80 + 26
    assert vec[hand_start] > 0.0


def test_context_block_includes_deck_slots() -> None:
    card_a, card_b = "absorb_in_aether_blue", "absorb_in_aether_red"
    assert card_index(card_a) > 0 and card_index(card_b) > 0
    episode = EpisodeContext(
        self_hero_id="hero_briar",
        opp_hero_id="hero_kayo",
        format="silver_age",
        self_deck_counts={card_a: 2, card_b: 1},
        first_player=2,
    )
    vec = player_observation_vector(
        _minimal_state(),
        [],
        episode_context=episode,
    )
    assert vec[3] == 2.0
    assert np.count_nonzero(vec[4:84]) >= 2
