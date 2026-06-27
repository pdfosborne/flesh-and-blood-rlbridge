import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.card_vocab import card_index
from flesh_and_blood_rlbridge.deck_context import EpisodeContext
from flesh_and_blood_rlbridge.player_observation import (
    DECK_SLOTS,
    HAND_SLOTS,
    PLAYER_OBS_DIM,
    PLAYER_OBS_SCHEMA_VERSION,
    player_observation_vector,
)


def test_schema_version_is_two() -> None:
    assert PLAYER_OBS_SCHEMA_VERSION == 2


def test_observation_dim_is_626() -> None:
    assert PLAYER_OBS_DIM == 626


def test_unknown_card_index_is_zero() -> None:
    assert card_index("") == 0
    assert card_index("not_a_real_card_xyz") == 0


def test_player_observation_vector_shape() -> None:
    ctx = EpisodeContext(
        self_hero_id="ira_crimson_haze",
        opp_hero_id="briar",
        format="silver_age",
        self_deck_counts={"surging_strike_red": 3},
        first_player=1,
    )
    state = {
        "actingPlayerID": 1,
        "playerHealth": 20,
        "opponentHealth": 19,
        "turnNo": 2,
        "turnPhase": "M",
        "playerHandSize": 1,
        "opponentHandSize": 4,
        "playerDeckCount": 30,
        "opponentDeckCount": 31,
        "playerPitchCount": 0,
        "opponentPitchCount": 0,
        "playerHand": [
            {
                "cardID": "surging_strike_red",
                "cardNumber": "surging_strike_red",
                "cost": 0,
                "pitch": 1,
                "power": 3,
                "defense": 2,
                "action": 27,
            }
        ],
        "playerEquipment": [{"cardNumber": "ira_crimson_haze"}],
        "opponentEquipment": [{"cardNumber": "briar"}],
    }
    vec = player_observation_vector(state, [{"label": "pass"}], episode_context=ctx)
    assert vec.shape == (PLAYER_OBS_DIM,)
    assert vec.dtype.name == "float64"


def test_deck_context_indices_capped_at_80() -> None:
    counts = {f"card_{i}": 2 for i in range(100)}
    ctx = EpisodeContext(self_deck_counts=counts)
    assert len(ctx.self_deck_indices()) <= DECK_SLOTS


def test_hand_slots_do_not_leak_opponent_hand() -> None:
    ctx = EpisodeContext(format="silver_age")
    state = {
        "actingPlayerID": 1,
        "playerHealth": 20,
        "opponentHealth": 20,
        "turnPhase": "M",
        "playerHand": [],
        "opponentHand": [{"cardNumber": "secret_card_red"} for _ in range(4)],
        "opponentHandSize": 4,
    }
    vec = player_observation_vector(state, [], episode_context=ctx)
    assert vec.shape == (PLAYER_OBS_DIM,)
    # Hand block starts after context + scalars
    from flesh_and_blood_rlbridge.player_observation import CONTEXT_DIM, SCALAR_COUNT, HAND_SLOT_DIM

    hand_start = CONTEXT_DIM + SCALAR_COUNT
    hand_block = vec[hand_start : hand_start + HAND_SLOTS * HAND_SLOT_DIM]
    assert float(hand_block.max()) == 0.0
