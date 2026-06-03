import sys
from pathlib import Path

# Allow tests from repo root without package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.talishar_default_policy import choose_talishar_action_index


def test_defense_window_prefers_highest_block() -> None:
    state = {
        "turnPhase": "defense",
        "playerHand": [
            {
                "action": 10,
                "actionDataOverride": "0",
                "label": "Blue Card",
                "defense": 2,
                "power": 1,
            },
            {
                "action": 10,
                "actionDataOverride": "1",
                "label": "Red Card",
                "defense": 3,
                "power": 3,
            },
        ],
    }
    legal = [
        {
            "action_code": 10,
            "button_input": "0",
            "zone": "hand",
            "label": "Blue Card",
        },
        {
            "action_code": 10,
            "button_input": "1",
            "zone": "hand",
            "label": "Red Card",
        },
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "label": "Pass",
        },
    ]
    assert choose_talishar_action_index(legal, state) == 1


def test_offense_window_prefers_highest_attack() -> None:
    state = {
        "turnPhase": "main",
        "playerHand": [
            {
                "action": 10,
                "actionDataOverride": "0",
                "label": "Swing Small",
                "power": 2,
                "defense": 3,
                "cost": 0,
            },
            {
                "action": 10,
                "actionDataOverride": "1",
                "label": "Swing Big",
                "power": 5,
                "defense": 2,
                "cost": 1,
            },
        ],
    }
    legal = [
        {
            "action_code": 10,
            "button_input": "0",
            "zone": "hand",
            "label": "Swing Small",
        },
        {
            "action_code": 10,
            "button_input": "1",
            "zone": "hand",
            "label": "Swing Big",
        },
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "label": "Pass",
        },
    ]
    assert choose_talishar_action_index(legal, state) == 1


def test_popup_prefers_yes_like_confirmation() -> None:
    legal = [
        {
            "action_code": 1,
            "button_input": "n",
            "zone": "popup",
            "label": "No",
        },
        {
            "action_code": 1,
            "button_input": "y",
            "zone": "popup",
            "label": "Yes",
        },
    ]
    assert choose_talishar_action_index(legal, {}) == 1


def test_only_pass_returns_zero() -> None:
    legal = [{"action_code": 99, "button_input": "", "zone": "button", "label": "Pass"}]
    assert choose_talishar_action_index(legal, {}) == 0
