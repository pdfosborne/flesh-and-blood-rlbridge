import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.cpp_engine_environment import CppEngineEnvironment
from flesh_and_blood_rlbridge.legal_action_filter import filter_legal_actions
from flesh_and_blood_rlbridge.talishar_default_policy import (
    _apply_block_phase_filter,
    _card_cost,
    _equipment_activation_cost,
    _is_affordable_arsenal_play,
    _is_affordable_equipment_play,
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
    env._gs = type(
        "_GS",
        (),
        {
            "p1_hand": hand,
            "p2_hand": [],
            "phase": phase,
            "p1_health": 20,
            "p2_health": 20,
            "turn_no": 1,
            "p1_hand_size": len(hand),
            "p2_hand_size": 0,
            "p1_deck_size": 30,
            "p2_deck_size": 30,
            "p1_pitch_size": 0,
            "p2_pitch_size": 0,
            "game_over": False,
            "instant_window": False,
            "pending_attack_power": 0,
            "pending_block_value": 0,
            "p1_equipment": [],
            "p2_equipment": [],
            "p1_arsenal": [],
            "p2_arsenal": [],
            "p1_pitch": [],
            "p2_pitch": [],
            "p1_discard": [],
            "p2_discard": [],
        },
    )()
    env._acting_player = 1
    env._talishar_overlay = None
    env._talishar_raw_state = None
    env._talishar_parity_extra = None
    env._flow_phase = ""
    env._turn_no_override = None
    env._strict_simulation = False
    env._steps = 0
    env._hand_playability = {}
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
    state = {"turnPhase": {"turnPhase": "P"}, "playerHand": [], "canPassPhase": True}
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


def test_talishar_empty_pitch_window_offers_cancel_when_pass_ignored() -> None:
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
    assert filtered[0]["action_code"] == 10000


def test_instant_strips_unaffordable_hand_play_from_active_layers() -> None:
    state = {
        "turnPhase": {"turnPhase": "INSTANT"},
        "playerPitchCount": 0,
        "playerHand": [
            {
                "action": 27,
                "actionDataOverride": "0",
                "cardNumber": "cosmic_flare_red",
                "cost": 0,
                "resource": 1,
            },
            {
                "action": 27,
                "actionDataOverride": "1",
                "cardNumber": "nebula_duality_red",
                "cost": 2,
                "resource": 2,
            },
        ],
        "activeLayers": [
            {"caption": "Choose an instant"},
        ],
    }
    legal = [
        {
            "action_code": 27,
            "button_input": "1",
            "zone": "hand",
            "card_id": "nebula_duality_red",
            "label": "Nebula Duality",
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

    assert 27 not in codes
    assert 99 in codes


def test_materialize_filtered_actions_injects_synthesized_pass() -> None:
    from flesh_and_blood_rlbridge.legal_action_filter import materialize_filtered_actions

    original = [
        {
            "action_code": 27,
            "button_input": "0",
            "zone": "hand",
            "card_id": "WTR001",
            "label": "Attack",
        }
    ]
    filtered = [
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "card_id": "",
            "label": "Pass",
        }
    ]
    made: list[dict[str, Any]] = []

    def _make(action: dict[str, Any]) -> dict[str, Any]:
        made.append(action)
        return action

    out = materialize_filtered_actions(original, filtered, make_action=_make)
    assert len(out) == 1
    assert out[0]["action_code"] == 99
    assert len(made) == 1


def test_cpp_empty_pitch_window_offers_cancel_when_pass_ignored() -> None:
    env = _build_cpp_env([], phase=2)
    env._strict_simulation = False
    env._talishar_parity_extra = None
    env._steps = 0
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
    assert filtered[0].action_code == 10000


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


def test_equipment_activation_cost_from_talishar() -> None:
    assert _equipment_activation_cost({"cardNumber": "reaping_blade"}, None) == 1
    assert _equipment_activation_cost({"cardNumber": "dawnblade"}, None) == 1
    assert _equipment_activation_cost({"cardNumber": "romping_club"}, None) == 2


def test_is_affordable_equipment_play_requires_hand_pitch() -> None:
    state = {
        "playerPitchCount": 0,
        "playerHand": [],
        "playerEquipment": [
            {
                "action": 3,
                "actionDataOverride": "0",
                "cardNumber": "reaping_blade",
            }
        ],
    }
    action = {
        "action_code": 3,
        "button_input": "0",
        "zone": "equipment",
        "card_id": "reaping_blade",
        "label": "Reaping Blade",
    }
    assert not _is_affordable_equipment_play(action, state)


def test_is_affordable_equipment_play_with_floating_resources() -> None:
    state = {
        "playerPitchCount": 1,
        "playerHand": [],
        "playerEquipment": [
            {
                "action": 3,
                "actionDataOverride": "0",
                "cardNumber": "reaping_blade",
            }
        ],
    }
    action = {
        "action_code": 3,
        "button_input": "0",
        "zone": "equipment",
        "card_id": "reaping_blade",
        "label": "Reaping Blade",
    }
    assert _is_affordable_equipment_play(action, state)


def test_filter_strips_unaffordable_equipment_play_with_empty_hand() -> None:
    state = {
        "turnPhase": {"turnPhase": "M"},
        "playerPitchCount": 0,
        "playerHand": [],
        "playerEquipment": [
            {
                "action": 3,
                "actionDataOverride": "0",
                "cardNumber": "reaping_blade",
            }
        ],
    }
    legal = [
        {
            "action_code": 3,
            "button_input": "0",
            "zone": "equipment",
            "card_id": "reaping_blade",
            "label": "Reaping Blade",
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

    assert 3 not in codes
    assert 99 in codes


def test_filter_keeps_affordable_equipment_play_when_hand_can_pitch() -> None:
    state = {
        "turnPhase": {"turnPhase": "M"},
        "playerPitchCount": 0,
        "playerHand": [
            {
                "action": 27,
                "actionDataOverride": "0",
                "cardNumber": "nimblism_blue",
            }
        ],
        "playerEquipment": [
            {
                "action": 3,
                "actionDataOverride": "0",
                "cardNumber": "reaping_blade",
            }
        ],
    }
    legal = [
        {
            "action_code": 3,
            "button_input": "0",
            "zone": "equipment",
            "card_id": "reaping_blade",
            "label": "Reaping Blade",
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

    assert 3 in codes


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


def test_yesno_popup_keeps_both_buttons() -> None:
    """YESNO prompts are not regex-filtered; loop guard handles stalls instead."""
    state = {
        "turnPhase": {"turnPhase": "YESNO"},
        "playerPitchCount": 0,
        "playerHand": [],
        "playerPrompt": {
            "helpText": "Choose if you want to pay 3 to avoid taking 2 damage",
        },
    }
    legal = [
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
    filtered = filter_legal_actions(state, legal)
    buttons = {action["button_input"] for action in filtered}

    assert buttons == {"YES", "NO"}


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


def test_no_priority_waiting_prompt_collapses_to_pass_only() -> None:
    state = {
        "turnPhase": {"turnPhase": "INSTANT"},
        "havePriority": False,
        "playerPrompt": {
            "helpText": "Waiting for other player to choose an instant",
        },
        "playerHand": [
            {
                "action": 27,
                "actionDataOverride": "0",
                "cardNumber": "stroke_of_foresight_yellow",
                "cost": 0,
                "resource": 2,
            },
        ],
    }
    legal = [
        {
            "action_code": 27,
            "button_input": "0",
            "zone": "hand",
            "card_id": "stroke_of_foresight_yellow",
            "label": "Stroke of Foresight",
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


def test_have_priority_false_without_waiting_text_still_pass_only() -> None:
    state = {
        "turnPhase": {"turnPhase": "INSTANT"},
        "havePriority": False,
        "playerHand": [],
    }
    legal = [
        {
            "action_code": 27,
            "button_input": "0",
            "zone": "hand",
            "card_id": "some_card",
            "label": "Some Card",
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


def test_choosetop_strips_pass_when_top_available() -> None:
    """CHOOSETOP Pass is a no-op when a Top action exists (Azalea stall case)."""
    state = {
        "turnPhase": {"turnPhase": "CHOOSETOP"},
        "canPassPhase": True,
        "playerHand": [],
    }
    legal = [
        {
            "action_code": 8,
            "button_input": "widowmaker_yellow",
            "zone": "popup",
            "label": "Top",
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
    assert filtered[0]["action_code"] == 8
    assert filtered[0]["label"] == "Top"


def test_instant_keeps_pass_with_optional_equipment() -> None:
    """Optional INSTANT equipment windows must not force activation at filter layer."""
    state = {
        "turnPhase": {"turnPhase": "INSTANT"},
        "canPassPhase": True,
        "playerHand": [],
        "playerEquipment": [{"cardNumber": "topsy_turvy", "action": 3}],
    }
    legal = [
        {
            "action_code": 3,
            "button_input": "120",
            "zone": "equipment",
            "label": "topsy_turvy",
        },
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "label": "Pass",
        },
    ]

    filtered = filter_legal_actions(state, legal)

    assert len(filtered) == 2
    assert any(a["action_code"] == 3 for a in filtered)
    assert any(a["action_code"] == 99 for a in filtered)


def test_prefer_non_pass_index_from_legal_actions() -> None:
    from flesh_and_blood_rlbridge.legal_action_filter import prefer_non_pass_index

    obs = {
        "legal_actions": [
            {"index": 0, "label": "Top", "zone": "popup"},
            {"index": 1, "label": "Pass", "zone": "button"},
        ],
    }
    assert prefer_non_pass_index(obs, fallback_action=1) == 0


def test_is_mandatory_progress_phase_choosetop() -> None:
    from flesh_and_blood_rlbridge.legal_action_filter import is_mandatory_progress_phase

    assert is_mandatory_progress_phase({"turnPhase": "CHOOSETOP"})
    assert not is_mandatory_progress_phase({"turnPhase": "INSTANT"})


def test_is_mandatory_progress_phase_ars() -> None:
    from flesh_and_blood_rlbridge.legal_action_filter import is_mandatory_progress_phase

    assert not is_mandatory_progress_phase({"turnPhase": "ARS"})
    assert is_mandatory_progress_phase({"turnPhase": "CHOOSEARSENAL"})


def test_ars_keeps_pass_when_hand_mode4_available() -> None:
    """ARS arsenaling is optional; Pass skips to end-of-turn finalize."""
    state = {
        "turnPhase": {"turnPhase": "ARS"},
        "canPassPhase": True,
        "playerHand": [{"cardNumber": "sink_below_red"}],
    }
    legal = [
        {
            "action_code": 4,
            "button_input": "sink_below_red",
            "zone": "hand",
            "label": "sink_below_red",
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

    assert codes == {4, 99}


def test_ars_keeps_pass_when_no_arsenal_actions() -> None:
    state = {
        "turnPhase": {"turnPhase": "ARS"},
        "canPassPhase": True,
        "playerHand": [],
    }
    legal = [
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


def test_choosearsenal_strips_pass_when_pick_exists() -> None:
    state = {
        "turnPhase": {"turnPhase": "CHOOSEARSENAL"},
        "canPassPhase": False,
    }
    legal = [
        {
            "action_code": 16,
            "button_input": "0",
            "zone": "arsenal",
            "label": "arsenal_card",
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
    assert filtered[0]["action_code"] == 16


def test_main_strips_pass_when_hand_play_available() -> None:
    state = {
        "turnPhase": {"turnPhase": "M"},
        "canPassPhase": True,
        "playerPitchCount": 2,
        "playerHand": [
            {
                "action": 27,
                "actionDataOverride": "0",
                "cardNumber": "nimblism_blue",
                "cost": 0,
            }
        ],
    }
    legal = [
        {
            "action_code": 27,
            "button_input": "0",
            "zone": "hand",
            "card_id": "nimblism_blue",
            "label": "Nimblism",
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

    assert 27 in codes
    assert 99 not in codes


def test_main_keeps_pass_when_no_plays() -> None:
    state = {
        "turnPhase": {"turnPhase": "M"},
        "canPassPhase": True,
        "playerHand": [],
    }
    legal = [
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
