import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.cpp_engine_environment import CppEngineEnvironment
from flesh_and_blood_rlbridge.talishar_default_policy import (
    _apply_block_phase_filter,
    _card_cost,
    _is_affordable_hand_play,
    _strip_revert_actions,
)
from flesh_and_blood_rlbridge.talishar_engine_environment import TalisharEngineEnvironment


class _FakeCard:
    def __init__(
        self,
        *,
        card_id: str,
        name: str,
        cost: int = 0,
        pitch: int = 0,
        defense: int = 0,
    ) -> None:
        self.card_id = card_id
        self.name = name
        self.cost = cost
        self.pitch = pitch
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


def _build_talishar_env() -> TalisharEngineEnvironment:
    return object.__new__(TalisharEngineEnvironment)


def _build_cpp_env(hand: list[_FakeCard]) -> CppEngineEnvironment:
    env = object.__new__(CppEngineEnvironment)
    env._gs = type("_GS", (), {"p1_hand": hand, "p2_hand": []})()
    env._acting_player = 1
    return env


def test_card_cost_uses_card_db_when_talishar_omits_cost() -> None:
    card = {"cardNumber": "infecting_shot_red", "action": 27, "actionDataOverride": "0"}
    assert _card_cost(card) == 1


def test_talishar_filters_unaffordable_single_card_play() -> None:
    env = _build_talishar_env()
    state = {
        "turnPhase": {"turnPhase": "M"},
        "playerHand": [
            {
                "action": 27,
                "actionDataOverride": "0",
                "cardNumber": "infecting_shot_red",
                "label": "Infecting Shot",
            }
        ],
    }
    legal = [
        {
            "action_code": 27,
            "button_input": "0",
            "zone": "hand",
            "card_id": "infecting_shot_red",
            "label": "Infecting Shot",
        },
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "label": "Pass",
        },
    ]

    filtered = env._filter_legal_actions(state, legal)

    assert len(filtered) == 1
    assert filtered[0]["action_code"] == 99


def test_talishar_keeps_affordable_play_when_other_cards_can_pitch() -> None:
    env = _build_talishar_env()
    state = {
        "turnPhase": {"turnPhase": "M"},
        "playerHand": [
            {
                "action": 27,
                "actionDataOverride": "0",
                "cardNumber": "infecting_shot_red",
                "label": "Infecting Shot",
            },
            {
                "action": 27,
                "actionDataOverride": "1",
                "cardNumber": "nimblism_red",
                "label": "Nimblism",
            },
        ],
    }
    legal = [
        {
            "action_code": 27,
            "button_input": "0",
            "zone": "hand",
            "card_id": "infecting_shot_red",
            "label": "Infecting Shot",
        },
        {
            "action_code": 27,
            "button_input": "1",
            "zone": "hand",
            "card_id": "nimblism_red",
            "label": "Nimblism",
        },
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "label": "Pass",
        },
    ]

    filtered = env._filter_legal_actions(state, legal)
    play_labels = {
        a["label"]
        for a in filtered
        if a["action_code"] == 27 and a["zone"] == "hand"
    }

    assert "Infecting Shot" in play_labels
    assert "Nimblism" in play_labels


def test_talishar_empty_pitch_window_offers_cancel_only() -> None:
    env = _build_talishar_env()
    state = {"turnPhase": {"turnPhase": "P"}, "playerHand": []}
    legal = [
        {
            "action_code": 10000,
            "button_input": "",
            "zone": "button",
            "label": "Cancel",
        },
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "label": "Pass",
        },
    ]

    filtered = env._filter_legal_actions(state, legal)

    assert len(filtered) == 1
    assert filtered[0]["action_code"] == 10000


def test_is_affordable_hand_play_helper() -> None:
    state = {
        "playerHand": [
            {
                "action": 27,
                "actionDataOverride": "0",
                "cardNumber": "infecting_shot_red",
            }
        ]
    }
    action = {
        "action_code": 27,
        "button_input": "0",
        "zone": "hand",
        "card_id": "infecting_shot_red",
    }
    assert not _is_affordable_hand_play(action, state)


def test_cpp_filters_unaffordable_play_using_card_db() -> None:
    env = _build_cpp_env(
        [
            _FakeCard(card_id="infecting_shot_red", name="Infecting Shot"),
        ]
    )
    legal = [
        _FakeAction(
            action_code=27,
            button_input="0",
            card_id="infecting_shot_red",
            zone="hand",
            label="Infecting Shot",
        ),
        _FakeAction(
            action_code=99,
            button_input="",
            card_id="",
            zone="button",
            label="Pass",
        ),
    ]

    filtered = env._filter_legal_actions(legal)

    assert len(filtered) == 1
    assert filtered[0].action_code == 99


def test_cpp_observation_excludes_unaffordable_hand_play() -> None:
    env = object.__new__(CppEngineEnvironment)
    env._gs = type(
        "_GS",
        (),
        {
            "phase": 1,
            "p1_health": 20,
            "p2_health": 19,
            "turn_no": 3,
            "p1_hand_size": 1,
            "p2_hand_size": 0,
            "p1_deck_size": 32,
            "p2_deck_size": 31,
            "p1_pitch_size": 0,
            "p2_pitch_size": 0,
            "game_over": False,
            "p1_hand": [_FakeCard(card_id="infecting_shot_red", name="Infecting Shot")],
            "p2_hand": [],
        },
    )()
    env._acting_player = 1
    legal = [
        _FakeAction(
            action_code=27,
            button_input="0",
            card_id="infecting_shot_red",
            zone="hand",
            label="Infecting Shot",
        ),
        _FakeAction(
            action_code=99,
            button_input="",
            card_id="",
            zone="button",
            label="Pass",
        ),
    ]

    obs = json.loads(env._encode_observation(legal))

    assert obs["legalActions"] == [{"index": 0, "label": "Pass", "zone": "button"}]


def test_talishar_block_phase_pass_only_when_only_attackers_remain() -> None:
    env = _build_talishar_env()
    hand = [
        {
            "action": 27,
            "actionDataOverride": "0",
            "cardNumber": "swift_shot_red",
            "label": "Swift Shot",
            "defense": 2,
        },
        {
            "action": 27,
            "actionDataOverride": "1",
            "cardNumber": "searing_shot_red",
            "label": "Searing Shot",
            "defense": 2,
        },
    ]
    state = {"turnPhase": {"turnPhase": "B"}, "playerHand": hand}
    legal = [
        {
            "action_code": 27,
            "button_input": "0",
            "zone": "hand",
            "card_id": "swift_shot_red",
            "label": "Swift Shot",
        },
        {
            "action_code": 27,
            "button_input": "1",
            "zone": "hand",
            "card_id": "searing_shot_red",
            "label": "Searing Shot",
        },
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "label": "Pass",
        },
        {
            "action_code": 10001,
            "button_input": "",
            "zone": "button",
            "label": "Undo Block",
        },
    ]

    filtered = env._filter_legal_actions(state, legal)

    assert len(filtered) == 1
    assert filtered[0]["action_code"] == 99


def test_apply_block_phase_filter_keeps_real_blockers() -> None:
    hand = [
        {
            "action": 27,
            "actionDataOverride": "0",
            "cardNumber": "nimblism_red",
            "label": "Nimblism",
            "defense": 3,
        },
        {
            "action": 27,
            "actionDataOverride": "1",
            "cardNumber": "swift_shot_red",
            "label": "Swift Shot",
            "defense": 2,
        },
    ]
    state = {"turnPhase": {"turnPhase": "B"}, "playerHand": hand}
    legal = [
        {
            "action_code": 27,
            "button_input": "0",
            "zone": "hand",
            "card_id": "nimblism_red",
            "label": "Nimblism",
        },
        {
            "action_code": 27,
            "button_input": "1",
            "zone": "hand",
            "card_id": "swift_shot_red",
            "label": "Swift Shot",
        },
        {"action_code": 99, "button_input": "", "zone": "button", "label": "Pass"},
    ]

    filtered = _apply_block_phase_filter(state, legal)
    block_labels = {
        a["label"] for a in filtered if a["action_code"] == 27 and a["zone"] == "hand"
    }

    assert block_labels == {"Nimblism"}
    assert any(a["action_code"] == 99 for a in filtered)


def test_cpp_block_phase_pass_only_for_attack_cards() -> None:
    env = object.__new__(CppEngineEnvironment)
    env._gs = type(
        "_GS",
        (),
        {
            "phase": 4,
            "p1_health": 20,
            "p2_health": 19,
            "turn_no": 3,
            "p1_hand_size": 2,
            "p2_hand_size": 0,
            "p1_deck_size": 32,
            "p2_deck_size": 31,
            "p1_pitch_size": 0,
            "p2_pitch_size": 0,
            "game_over": False,
            "p1_hand": [
                _FakeCard(card_id="swift_shot_red", name="Swift Shot", defense=2),
                _FakeCard(card_id="searing_shot_red", name="Searing Shot", defense=2),
            ],
            "p2_hand": [],
        },
    )()
    env._acting_player = 1
    legal = [
        _FakeAction(
            action_code=27,
            button_input="0",
            card_id="swift_shot_red",
            zone="hand",
            label="Swift Shot",
        ),
        _FakeAction(
            action_code=27,
            button_input="1",
            card_id="searing_shot_red",
            zone="hand",
            label="Searing Shot",
        ),
        _FakeAction(
            action_code=99,
            button_input="",
            card_id="",
            zone="button",
            label="Pass",
        ),
    ]

    filtered = env._filter_legal_actions(legal)

    assert len(filtered) == 1
    assert filtered[0].action_code == 99


def test_strip_revert_actions_removes_undo_block_in_block_phase() -> None:
    legal = [
        {
            "action_code": 27,
            "button_input": "0",
            "zone": "hand",
            "label": "Blocker",
        },
        {
            "action_code": 10001,
            "button_input": "",
            "zone": "button",
            "label": "Undo Block",
        },
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "label": "Pass",
        },
    ]
    stripped = _strip_revert_actions("b", legal)
    codes = {_a["action_code"] for _a in stripped}
    assert 10001 not in codes
    assert 27 in codes
    assert 99 in codes


def test_talishar_block_phase_strips_undo_even_with_viable_blockers() -> None:
    env = _build_talishar_env()
    hand = [
        {
            "action": 27,
            "actionDataOverride": "0",
            "cardNumber": "nimblism_red",
            "label": "Nimblism",
            "defense": 3,
        },
    ]
    state = {"turnPhase": {"turnPhase": "B"}, "playerHand": hand}
    legal = [
        {
            "action_code": 27,
            "button_input": "0",
            "zone": "hand",
            "card_id": "nimblism_red",
            "label": "Nimblism",
        },
        {
            "action_code": 10001,
            "button_input": "",
            "zone": "button",
            "label": "Undo Block",
        },
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "label": "Pass",
        },
    ]
    filtered = env._filter_legal_actions(state, legal)
    codes = {_a["action_code"] for _a in filtered}
    assert 10001 not in codes
    assert 27 in codes
    assert 99 in codes
