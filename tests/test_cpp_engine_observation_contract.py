import json
import sys
from pathlib import Path

# Allow tests from repo root without package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.cpp_engine_environment import CppEngineEnvironment


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
            _FakeCard(card_id="WTR001", name="Blue Attack"),
            _FakeCard(card_id="WTR002", name="Too Expensive"),
        ]
        self.p2_hand = []


class _FakeCard:
    def __init__(self, *, card_id: str, name: str) -> None:
        self.card_id = card_id
        self.name = name


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
    env._flow_phase = ""
    env._hand_playability = {}
    env._turn_no_override = None
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
    }
    assert set(obs.keys()) == expected_keys

    assert obs["actingPlayerID"] == 1
    assert obs["turnPhase"] == "M"
    assert obs["havePriority"] is True
    assert obs["playerPitchCount"] == 1
    assert obs["playerHandSize"] == 2

    # Talishar-compatible playerHand entry shape for all cards in hand:
    # playable card gets action=27 + actionDataOverride; non-playable action=0.
    assert obs["playerHand"] == [
        {
            "cardID": "WTR001",
            "action": 27,
            "actionDataOverride": "0",
            "label": "",
        },
        {
            "cardID": "WTR002",
            "action": 0,
            "actionDataOverride": "1",
            "label": "",
        }
    ]

    # Talishar-compatible legalActions shape (index/label/zone only)
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
