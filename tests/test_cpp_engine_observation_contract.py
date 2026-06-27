import json
import sys
from pathlib import Path

# Allow tests from repo root without package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.cpp_engine_environment import CppEngineEnvironment
from flesh_and_blood_rlbridge.deck_context import EpisodeContext
from flesh_and_blood_rlbridge.player_observation import (
    PLAYER_OBS_DIM,
    PLAYER_OBS_SCHEMA_VERSION,
)


class _FakePlayer:
    def __init__(self, *, hand=None, pitch_size=0) -> None:
        self.health = 20
        self.hand = hand or []
        self.deck = []
        self.discard = []
        self.equipment = []
        self.arsenal = []
        self.pitch_zone = [object()] * pitch_size
        self.hero_card_id = "hero_ira"


class _FakeGS:
    def __init__(self, *, phase: int = 1) -> None:
        self.phase = phase
        self.p1_health = 20
        self.p2_health = 19
        self.turn_no = 3
        self.p1_hand_size = 2
        self.p2_hand_size = 3
        self.p1_deck_size = 32
        self.p2_deck_size = 31
        self.p1_pitch_size = 1
        self.p2_pitch_size = 0
        self.game_over = False
        self.p1_hand = [
            _FakeCard(card_id="WTR001", name="Blue Attack", cost=0, pitch=3),
            _FakeCard(card_id="WTR002", name="Too Expensive", cost=5, pitch=2),
        ]
        self.p2_hand = []
        self.players = [
            _FakePlayer(hand=self.p1_hand, pitch_size=1),
            _FakePlayer(hand=self.p2_hand, pitch_size=0),
        ]


class _FakeCard:
    def __init__(
        self,
        *,
        card_id: str,
        name: str,
        cost: int = 0,
        pitch: int = 0,
        power: int = 0,
        defense: int = 0,
    ) -> None:
        self.card_id = card_id
        self.name = name
        self.cost = cost
        self.pitch = pitch

        self.power = power
        self.defense = defense


class _FakeAction:
    def __init__(
        self,
        *,
        action_code: int,
        button_input: str,
        card_id: str,
        zone: str,
        label: str,
    ) -> None:
        self.action_code = action_code
        self.button_input = button_input
        self.card_id = card_id
        self.zone = zone
        self.label = label


def _build_env_for_encode(phase: int = 1) -> CppEngineEnvironment:
    # Construct without invoking __init__ (which requires a compiled module).
    env = object.__new__(CppEngineEnvironment)
    env._gs = _FakeGS(phase=phase)
    env._acting_player = 1
    env._talishar_overlay = None
    env._talishar_raw_state = None
    env._flow_phase = ""
    env._hand_playability = {}
    env._turn_no_override = None
    env._p1_episode_context = EpisodeContext(
        self_hero_id="hero_ira",
        opp_hero_id="hero_ira",
        format="silver_age",
        self_deck_counts={"WTR001": 1},
        first_player=1,
    )
    env._p2_episode_context = EpisodeContext(
        self_hero_id="hero_ira",
        opp_hero_id="hero_ira",
        format="silver_age",
        self_deck_counts={"WTR001": 1},
        first_player=1,
    )
    return env


def test_cpp_observation_contract_matches_talishar_field_shapes() -> None:
    env = _build_env_for_encode(phase=1)
    legal = [
        _FakeAction(
            action_code=27,
            button_input="0",
            card_id="WTR001",
            zone="hand",
            label="Blue Attack",
        ),
        _FakeAction(
            action_code=99,
            button_input="",
            card_id="",
            zone="button",
            label="Pass",
        ),
    ]

    encoded = env._encode_observation(legal)
    obs = json.loads(encoded)

    expected_keys = {
        "actingPlayerID",
        "selfPlay",
        "playerHealth",
        "opponentHealth",
        "turnNo",
        "turnPhase",
        "havePriority",
        "playerHandSize",
        "opponentHandSize",
        "playerDeckCount",
        "opponentDeckCount",
        "playerPitchCount",
        "playerHand",
        "legalActions",
        "observationVec",
        "obsSchemaVersion",
    }
    assert expected_keys.issubset(set(obs.keys()))

    assert obs["actingPlayerID"] == 1
    assert obs["turnPhase"] == "M"
    assert obs["havePriority"] is True
    assert obs["obsSchemaVersion"] == PLAYER_OBS_SCHEMA_VERSION
    assert len(obs["observationVec"]) == PLAYER_OBS_DIM

    assert obs["playerHand"] == [
        {
            "cardID": "WTR001",
            "action": 27,
            "actionDataOverride": "0",
            "label": "Blue Attack",
        },
        {
            "cardID": "WTR002",
            "action": 0,
            "actionDataOverride": "1",
            "label": "",
        }
    ]

    assert obs["legalActions"] == [
        {"index": 0, "label": "Blue Attack", "zone": "hand"},
        {"index": 1, "label": "Pass", "zone": "button"},
    ]


def test_cpp_phase_mapping_exposes_over_state() -> None:
    env = _build_env_for_encode(phase=7)
    legal = [
        _FakeAction(
            action_code=99,
            button_input="",
            card_id="",
            zone="button",
            label="Pass",
        )
    ]
    obs = json.loads(env._encode_observation(legal))
    assert obs["turnPhase"] == "OVER"


def test_raw_state_from_gs_uses_bindings_not_players_attr() -> None:
    """Regression: pybind11 GameState exposes p1_* / p2_* but not .players."""

    class _BindingStyleGS:
        p1_health = 20
        p2_health = 19
        p1_hand_size = 2
        p2_hand_size = 3
        p1_deck_size = 32
        p2_deck_size = 31
        p1_pitch_size = 1
        p2_pitch_size = 0
        phase = 1
        turn_no = 1
        game_over = False
        first_player = 1
        p1_hand = []
        p2_hand = []
        p1_equipment = [_FakeCard(card_id="WTR001", name="Sword")]
        p2_equipment = []
        p1_arsenal = []
        p2_arsenal = []
        p1_pitch = [_FakeCard(card_id="WTR002", name="Blue")]
        p2_pitch = []
        p1_discard = []
        p2_discard = []

    env = _build_env_for_encode(phase=1)
    env._gs = _BindingStyleGS()
    raw = env._raw_state_from_gs()
    assert raw["opponentPitchCount"] == 0
    assert len(raw["playerEquipment"]) == 1
    assert raw["playerEquipment"][0]["cardID"] == "WTR001"
    assert len(raw["playerPitch"]) == 1

    legal = [
        _FakeAction(
            action_code=99,
            button_input="",
            card_id="",
            zone="button",
            label="Pass",
        )
    ]
    encoded = env._encode_observation(legal)
    obs = json.loads(encoded)
    assert len(obs["observationVec"]) == PLAYER_OBS_DIM
