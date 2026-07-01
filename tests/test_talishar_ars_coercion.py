"""Tests for ARS-phase action coercion in TalisharEngineEnvironment."""

from __future__ import annotations

from flesh_and_blood_rlbridge.talishar_engine_environment import TalisharEngineEnvironment


def _bare_env_with_state(state: dict) -> TalisharEngineEnvironment:
    env = object.__new__(TalisharEngineEnvironment)
    env._last_state = state
    return env


def test_coerce_ars_action_replaces_invalid_hand_arsenal_with_pass() -> None:
    state = {
        "turnPhase": {"turnPhase": "ARS"},
        "playerHand": [
            {
                "cardNumber": "swiftstrike_bracers",
                "action": 4,
                "actionDataOverride": "swiftstrike_bracers",
            },
        ],
        "playerEquipment": [],
    }
    legal = [
        {
            "action_code": 4,
            "button_input": "swiftstrike_bracers",
            "zone": "hand",
            "card_id": "swiftstrike_bracers",
            "label": "Swiftstrike Bracers",
        },
        {"action_code": 99, "button_input": "", "zone": "button", "label": "Pass"},
    ]
    env = _bare_env_with_state(state)

    mode, button = env._coerce_ars_action(4, "swiftstrike_bracers", legal)

    assert mode == 99
    assert button == ""


def test_coerce_ars_action_keeps_valid_attack_arsenal() -> None:
    state = {
        "turnPhase": {"turnPhase": "ARS"},
        "playerHand": [
            {
                "cardNumber": "nimblism_red",
                "action": 4,
                "actionDataOverride": "nimblism_red",
            },
        ],
        "playerEquipment": [],
    }
    legal = [
        {
            "action_code": 4,
            "button_input": "nimblism_red",
            "zone": "hand",
            "card_id": "nimblism_red",
            "label": "Nimblism",
        },
        {"action_code": 99, "button_input": "", "zone": "button", "label": "Pass"},
    ]
    env = _bare_env_with_state(state)

    mode, button = env._coerce_ars_action(4, "nimblism_red", legal)

    assert mode == 4
    assert button == "nimblism_red"
