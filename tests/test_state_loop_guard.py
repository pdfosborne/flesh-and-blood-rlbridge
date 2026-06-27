from __future__ import annotations

from flesh_and_blood_rlbridge.state_loop_guard import (
    TurnLoopGuard,
    decision_point_fingerprint,
    legal_actions_fingerprint,
)


def _sample_state() -> dict:
    return {
        "turnPhase": {"turnPhase": "M"},
        "turnNo": 3,
        "playerPitchCount": 0,
        "playerHand": [],
    }


def _sample_legal() -> list[dict]:
    return [
        {
            "action_code": 3,
            "button_input": "0",
            "zone": "equipment",
            "label": "Crow's Nest",
        },
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "label": "Pass",
        },
    ]


def test_legal_actions_fingerprint_is_stable() -> None:
    legal = _sample_legal()
    assert legal_actions_fingerprint(legal) == legal_actions_fingerprint(list(legal))


def test_loop_guard_detects_repeated_decision_point() -> None:
    guard = TurnLoopGuard(max_steps_per_turn=100, loop_repeat_threshold=4)
    state = _sample_state()
    legal = _sample_legal()

    for _ in range(3):
        result = guard.check(
            state,
            legal,
            turn_no=3,
            acting_player_id=1,
        )
        assert not result.force_pass

    result = guard.check(state, legal, turn_no=3, acting_player_id=1)
    assert result.force_pass
    assert result.reason == "decision_loop"
    assert result.loop_streak == 4


def test_loop_guard_resets_on_turn_change() -> None:
    guard = TurnLoopGuard(max_steps_per_turn=100, loop_repeat_threshold=3)
    state = _sample_state()
    legal = _sample_legal()

    for _ in range(2):
        guard.check(state, legal, turn_no=3, acting_player_id=1)

    guard.check(state, legal, turn_no=4, acting_player_id=1)
    result = guard.check(state, legal, turn_no=4, acting_player_id=1)
    assert not result.force_pass
    assert result.loop_streak == 2


def test_loop_guard_enforces_per_turn_step_cap() -> None:
    guard = TurnLoopGuard(max_steps_per_turn=3, loop_repeat_threshold=99)
    state = _sample_state()
    legal_a = _sample_legal()
    legal_b = [
        {
            "action_code": 27,
            "button_input": "0",
            "zone": "hand",
            "label": "Attack",
        },
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "label": "Pass",
        },
    ]

    guard.check(state, legal_a, turn_no=3, acting_player_id=1)
    guard.check(state, legal_b, turn_no=3, acting_player_id=1)
    guard.check(state, legal_a, turn_no=3, acting_player_id=1)
    result = guard.check(state, legal_b, turn_no=3, acting_player_id=1)

    assert result.force_pass
    assert result.reason == "turn_step_cap"
    assert result.turn_steps == 4


def test_decision_point_fingerprint_changes_with_legal_set() -> None:
    state = _sample_state()
    fp_a = decision_point_fingerprint(state, _sample_legal())
    fp_b = decision_point_fingerprint(
        state,
        [
            {
                "action_code": 99,
                "button_input": "",
                "zone": "button",
                "label": "Pass",
            },
        ],
    )
    assert fp_a != fp_b
