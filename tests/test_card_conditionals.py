import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.card_conditionals import evaluate_conditional_active
from flesh_and_blood_rlbridge.card_vocab import card_index_normalized
from flesh_and_blood_rlbridge.deck_context import EpisodeContext
from flesh_and_blood_rlbridge.player_observation import (
    CONTEXT_DIM,
    HAND_SLOT_DIM,
    PLAYER_OBS_DIM,
    SCALAR_COUNT,
    ZONE_OFF,
    ZONE_SLOT_DIM,
    player_observation_vector,
)


def test_conditional_defaults_unconditional() -> None:
    assert evaluate_conditional_active(
        "surging_strike_red",
        {"turnPhase": "m"},
        zone="hand",
        side="self",
        phase="m",
        card_visible=True,
    ) == 1.0


def test_hidden_card_conditional_is_zero() -> None:
    assert (
        evaluate_conditional_active(
            "surging_strike_red",
            {},
            zone="hand",
            side="self",
            card_visible=False,
        )
        == 0.0
    )


def test_self_arsenal_face_down_reveals_card_idx() -> None:
    ctx = EpisodeContext(format="silver_age")
    state = {
        "actingPlayerID": 1,
        "playerHealth": 20,
        "opponentHealth": 20,
        "turnPhase": "m",
        "playerArse": [
            {
                "cardNumber": "surging_strike_red",
                "facing": "DOWN",
                "cost": 0,
                "pitch": 1,
                "power": 3,
                "defense": 2,
            }
        ],
    }
    vec = player_observation_vector(state, [], episode_context=ctx)
    # Self arsenal is zone index 1 after equipment (4 slots)
    arsenal_offset = ZONE_OFF + 4 * ZONE_SLOT_DIM
    card_idx = float(vec[arsenal_offset])
    face_down = float(vec[arsenal_offset + 3])
    assert card_idx > 0.0
    assert face_down == 1.0


def test_opponent_arsenal_hidden_card_idx() -> None:
    ctx = EpisodeContext(format="silver_age")
    state = {
        "actingPlayerID": 1,
        "playerHealth": 20,
        "opponentHealth": 20,
        "turnPhase": "m",
        "opponentArse": [{"cardNumber": "card_back", "facing": "DOWN"}],
    }
    vec = player_observation_vector(state, [], episode_context=ctx)
    # Opponent zones start after self zones (43 slots * 5 dims)
    opp_zone_off = ZONE_OFF + 43 * ZONE_SLOT_DIM
    opp_arsenal_offset = opp_zone_off + 4 * ZONE_SLOT_DIM
    assert float(vec[opp_arsenal_offset]) == 0.0


def test_hand_slot_includes_conditional_active_dim() -> None:
    ctx = EpisodeContext(format="silver_age")
    state = {
        "actingPlayerID": 1,
        "playerHealth": 20,
        "opponentHealth": 20,
        "turnPhase": "m",
        "playerHand": [
            {
                "cardNumber": "surging_strike_red",
                "cost": 0,
                "pitch": 1,
                "power": 3,
                "defense": 2,
            }
        ],
    }
    vec = player_observation_vector(state, [], episode_context=ctx)
    assert vec.shape == (PLAYER_OBS_DIM,)
    hand_start = CONTEXT_DIM + SCALAR_COUNT
    assert float(vec[hand_start + HAND_SLOT_DIM - 1]) == 1.0
