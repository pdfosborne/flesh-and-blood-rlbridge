import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.cpp_engine_environment import CppEngineEnvironment
from flesh_and_blood_rlbridge.legal_action_filter import filter_legal_actions
from flesh_and_blood_rlbridge.talishar_default_policy import (
    _apply_block_phase_filter,
    _card_cost,
    _is_affordable_arsenal_play,
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


def _build_cpp_env(hand: list[_FakeCard], *, phase: int = 1) -> CppEngineEnvironment:
    env = object.__new__(CppEngineEnvironment)
    env._gs = type("_GS", (), {"p1_hand": hand, "p2_hand": [], "phase": phase})()
    env._acting_player = 1
    env._talishar_overlay = None
    env._flow_phase = ""
    return env


def test_is_pass_only_helper() -> None:
    from flesh_and_blood_rlbridge.legal_action_filter import is_pass_only

    assert is_pass_only([{"action_code": 99, "label": "Pass"}])
    assert not is_pass_only([
        {"action_code": 27, "label": "Attack"},
        {"action_code": 99, "label": "Pass"},
    ])
    assert is_pass_only([])


def test_observation_fingerprint_matches_string_hash() -> None:
    from flesh_and_blood_rlbridge.obs_encoding import observation_fingerprint

    obs = '{"actingPlayerID":1,"turnNo":3}'
    vec = observation_fingerprint(obs)
    assert vec.shape == (1,)
    assert 0.0 <= float(vec[0]) <= 1.0
    assert np.allclose(vec, observation_fingerprint(obs))


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


def test_talishar_empty_pitch_window_offers_pass_only() -> None:
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

    filtered = filter_legal_actions(state, legal)

    assert len(filtered) == 1
    assert filtered[0]["action_code"] == 99


def test_cpp_empty_pitch_window_offers_pass_only() -> None:
    env = _build_cpp_env([], phase=2)
    legal = [
        _FakeAction(
            action_code=10000,
            button_input="",
            card_id="",
            zone="button",
            label="Cancel",
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


def test_is_affordable_arsenal_play_requires_hand_pitch() -> None:
    state = {
        "playerPitchCount": 0,
        "playerHand": [],
        "playerArse": [
            {
                "action": 5,
                "actionDataOverride": "0",
                "cardNumber": "adrenaline_rush_red",
            }
        ],
    }
    action = {
        "action_code": 5,
        "button_input": "0",
        "zone": "arsenal",
        "card_id": "adrenaline_rush_red",
        "label": "Adrenaline Rush",
    }
    assert not _is_affordable_arsenal_play(action, state)


def test_is_affordable_arsenal_play_with_enough_hand_pitch() -> None:
    state = {
        "playerPitchCount": 0,
        "playerHand": [
            {
                "action": 27,
                "actionDataOverride": "0",
                "cardNumber": "nimblism_blue",
            },
            {
                "action": 27,
                "actionDataOverride": "1",
                "cardNumber": "nimblism_blue",
            },
        ],
        "playerArse": [
            {
                "action": 5,
                "actionDataOverride": "0",
                "cardNumber": "adrenaline_rush_red",
            }
        ],
    }
    action = {
        "action_code": 5,
        "button_input": "0",
        "zone": "arsenal",
        "card_id": "adrenaline_rush_red",
        "label": "Adrenaline Rush",
    }
    assert _is_affordable_arsenal_play(action, state)


def test_filter_strips_unaffordable_arsenal_play_with_empty_hand() -> None:
    state = {
        "turnPhase": {"turnPhase": "M"},
        "playerPitchCount": 0,
        "playerHand": [],
        "playerArse": [
            {
                "action": 5,
                "actionDataOverride": "0",
                "cardNumber": "adrenaline_rush_red",
            }
        ],
    }
    legal = [
        {
            "action_code": 5,
            "button_input": "0",
            "zone": "arsenal",
            "card_id": "adrenaline_rush_red",
            "label": "Adrenaline Rush",
        },
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "label": "Pass",
        },
    ]

    filtered = filter_legal_actions(state, legal)
    codes = {a["action_code"] for a in filtered}

    assert 5 not in codes
    assert 99 in codes


def test_filter_keeps_affordable_arsenal_play_when_hand_can_pitch() -> None:
    state = {
        "turnPhase": {"turnPhase": "M"},
        "playerPitchCount": 0,
        "playerHand": [
            {
                "action": 27,
                "actionDataOverride": "0",
                "cardNumber": "nimblism_blue",
            },
            {
                "action": 27,
                "actionDataOverride": "1",
                "cardNumber": "nimblism_blue",
            },
        ],
        "playerArse": [
            {
                "action": 5,
                "actionDataOverride": "0",
                "cardNumber": "adrenaline_rush_red",
            }
        ],
    }
    legal = [
        {
            "action_code": 5,
            "button_input": "0",
            "zone": "arsenal",
            "card_id": "adrenaline_rush_red",
            "label": "Adrenaline Rush",
        },
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "label": "Pass",
        },
    ]

    filtered = filter_legal_actions(state, legal)
    codes = {a["action_code"] for a in filtered}

    assert 5 in codes


def test_is_affordable_hand_play_counts_floating_resources() -> None:
    state = {
        "playerPitchCount": 1,
        "playerHand": [
            {
                "action": 27,
                "actionDataOverride": "0",
                "cardNumber": "infecting_shot_red",
            },
            {
                "action": 27,
                "actionDataOverride": "1",
                "cardNumber": "nimblism_red",
            },
        ],
    }
    action = {
        "action_code": 27,
        "button_input": "0",
        "zone": "hand",
        "card_id": "infecting_shot_red",
    }
    assert _is_affordable_hand_play(action, state)


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
    env._talishar_overlay = None
    env._flow_phase = ""
    env._turn_no_override = None
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

    legal = env._filter_legal_actions(legal)
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
    env._talishar_overlay = None
    env._flow_phase = ""
    env._turn_no_override = None
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


def _bloodrot_yesno_legal() -> list[dict]:
    return [
        {
            "action_code": 20,
            "button_input": "YES",
            "card_id": "",
            "zone": "popup",
            "label": "Yes",
        },
        {
            "action_code": 20,
            "button_input": "NO",
            "card_id": "",
            "zone": "popup",
            "label": "No",
        },
    ]


def test_yesno_strips_yes_when_hand_pitch_cannot_pay_bloodrot() -> None:
    state = {
        "turnPhase": {"turnPhase": "YESNO"},
        "playerPitchCount": 0,
        "playerHand": [
            {
                "cardNumber": "nimblism_red",
                "action": 27,
                "actionDataOverride": "0",
                "resource": 1,
            }
        ],
        "playerPrompt": {
            "helpText": "Choose if you want to pay 3 to avoid taking 2 damage",
        },
        "playerInputPopUp": {
            "active": True,
            "popup": {
                "title": "Choose if you want to pay 3 to avoid taking 2 damage",
            },
        },
    }
    filtered = filter_legal_actions(state, _bloodrot_yesno_legal())

    assert len(filtered) == 1
    assert filtered[0]["button_input"] == "NO"


def test_yesno_keeps_yes_when_hand_pitch_can_pay_bloodrot() -> None:
    state = {
        "turnPhase": {"turnPhase": "YESNO"},
        "playerPitchCount": 0,
        "playerHand": [
            {
                "cardNumber": "evergreen_blue",
                "action": 27,
                "actionDataOverride": "0",
            }
        ],
        "playerPrompt": {
            "helpText": "Choose if you want to pay 3 to avoid taking 2 damage",
        },
    }
    filtered = filter_legal_actions(state, _bloodrot_yesno_legal())
    buttons = {action["button_input"] for action in filtered}

    assert buttons == {"YES", "NO"}


def test_yesno_strips_yes_when_only_pool_resources_are_insufficient() -> None:
    state = {
        "turnPhase": {"turnPhase": "YESNO"},
        "playerPitchCount": 2,
        "playerHand": [],
        "playerPrompt": {
            "helpText": "Choose if you want to pay 3 to avoid taking 2 damage",
        },
    }
    filtered = filter_legal_actions(state, _bloodrot_yesno_legal())

    assert len(filtered) == 1
    assert filtered[0]["button_input"] == "NO"


def test_filter_strips_undo_but_keeps_equipment_outside_main_phase() -> None:
    """Undo/cancel stall loops must not be offered; equipment plays are allowed."""
    state = {"turnPhase": {"turnPhase": "INSTANT"}}
    legal = [
        {
            "action_code": 3,
            "button_input": "4",
            "zone": "equipment",
            "card_id": "boltn_boots",
            "label": "Bolt'n Boots",
        },
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

    filtered = filter_legal_actions(state, legal)
    codes = {_a["action_code"] for _a in filtered}

    assert 3 in codes
    assert 10000 not in codes
    assert 99 in codes


def test_filter_strips_all_talishar_undo_modes() -> None:
    """Every Talishar undo/revert mode must be removed from agent legal actions."""
    state = {"turnPhase": {"turnPhase": "M"}, "playerHand": []}
    undo_modes = (10000, 10001, 10003, 100016, 100017, 100018, 100019)
    for mode in undo_modes:
        legal = [
            {
                "action_code": 3,
                "button_input": "0",
                "zone": "equipment",
                "card_id": "blossom_of_spring",
                "label": "Blossom of Spring",
            },
            {
                "action_code": mode,
                "button_input": "",
                "zone": "button",
                "label": f"Undo mode {mode}",
            },
            {
                "action_code": 99,
                "button_input": "",
                "zone": "button",
                "label": "Pass",
            },
        ]
        filtered = filter_legal_actions(state, legal)
        codes = {_a["action_code"] for _a in filtered}
        assert mode not in codes, f"mode {mode} should be stripped"
        assert 3 in codes


def test_filter_strips_revert_to_prior_turn() -> None:
    state = {"turnPhase": {"turnPhase": "M"}}
    legal = [
        {
            "action_code": 10003,
            "button_input": "beginTurnGamestate.txt",
            "zone": "button",
            "label": "Revert",
        },
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "label": "Pass",
        },
    ]

    filtered = filter_legal_actions(state, legal)
    codes = {_a["action_code"] for _a in filtered}

    assert 10003 not in codes
    assert 99 in codes


def test_filter_blacklists_arsenal_play_after_abort() -> None:
    state = {"turnPhase": {"turnPhase": "M"}}
    legal = [
        {
            "action_code": 5,
            "button_input": "0",
            "zone": "arsenal",
            "card_id": "spellblade_assault_red",
            "label": "Spellblade Assault",
        },
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "label": "Pass",
        },
    ]

    filtered = filter_legal_actions(
        state,
        legal,
        block_blacklist=frozenset({"Spellblade Assault"}),
    )
    codes = {_a["action_code"] for _a in filtered}

    assert 5 not in codes
    assert 99 in codes


def test_talishar_sanitize_blocks_undo_submission() -> None:
    env = _build_talishar_env()
    state = {"turnPhase": {"turnPhase": "INSTANT"}, "playerHand": []}
    legal = [
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "label": "Pass",
        },
    ]
    mode, button = env._sanitize_revert_submission(10000, "", legal, state)
    assert mode == 99
    assert button == ""
