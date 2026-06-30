from __future__ import annotations

from flesh_and_blood_rlbridge.state_loop_guard import (
    TurnLoopGuard,
    board_state_fingerprint,
    decision_point_fingerprint,
    legal_actions_fingerprint,
    resolve_forced_submission,
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
    base = _sample_state()
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

    guard.check({**base, "playerPitchCount": 0}, legal_a, turn_no=3, acting_player_id=1)
    guard.check({**base, "playerPitchCount": 1}, legal_b, turn_no=3, acting_player_id=1)
    guard.check({**base, "playerPitchCount": 2}, legal_a, turn_no=3, acting_player_id=1)
    result = guard.check({**base, "playerPitchCount": 3}, legal_b, turn_no=3, acting_player_id=1)

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


def test_board_state_fingerprint_ignores_legal_actions() -> None:
    state = {
        "turnPhase": {"turnPhase": "M"},
        "playerHealth": 20,
        "opponentHealth": 18,
        "playerPitchCount": 1,
        "playerHand": [{"cardNumber": "WTR001"}, {"cardNumber": "WTR002"}],
    }
    fp_a = board_state_fingerprint(state)
    fp_b = board_state_fingerprint(dict(state))
    assert fp_a == fp_b
    assert board_state_fingerprint(
        {**state, "playerHand": [{"cardNumber": "WTR001"}]}
    ) != fp_a


def test_loop_guard_detects_board_state_revert_chain() -> None:
    guard = TurnLoopGuard(max_steps_per_turn=100, loop_repeat_threshold=99)
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
    board_a = {
        "turnPhase": {"turnPhase": "M"},
        "turnNo": 3,
        "playerHealth": 20,
        "opponentHealth": 20,
        "playerPitchCount": 0,
        "playerHand": [{"cardNumber": "WTR001"}, {"cardNumber": "WTR002"}],
    }
    board_b = {
        **board_a,
        "playerPitchCount": 1,
        "playerHand": [{"cardNumber": "WTR001"}],
    }

    assert not guard.check(board_a, legal_a, turn_no=3, acting_player_id=1).force_pass
    assert not guard.check(board_b, legal_b, turn_no=3, acting_player_id=1).force_pass
    result = guard.check(board_a, legal_a, turn_no=3, acting_player_id=1)
    assert result.force_pass
    assert result.reason == "board_revert"


def test_loop_guard_board_revert_resets_on_turn_change() -> None:
    guard = TurnLoopGuard(max_steps_per_turn=100, loop_repeat_threshold=99)
    board = {
        "turnPhase": {"turnPhase": "M"},
        "playerHealth": 20,
        "opponentHealth": 20,
        "playerPitchCount": 0,
        "playerHand": [{"cardNumber": "WTR001"}],
    }
    legal = _sample_legal()

    guard.check(board, legal, turn_no=3, acting_player_id=1)
    result = guard.check(board, legal, turn_no=4, acting_player_id=1)
    assert not result.force_pass
    assert result.reason != "board_revert"


def test_loop_guard_detects_revert_despite_changing_legal_set() -> None:
    """Chain loops can change prompts while landing on the same board snapshot."""
    guard = TurnLoopGuard(max_steps_per_turn=100, loop_repeat_threshold=99)
    board = {
        "turnPhase": {"turnPhase": "M"},
        "playerHealth": 20,
        "opponentHealth": 20,
        "playerPitchCount": 0,
        "playerHand": [{"cardNumber": "WTR001"}],
        "playerEquipment": [{"cardNumber": "crow's_nest", "action": 3}],
    }
    legal_prompt = [
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
    legal_pass_only = [
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "label": "Pass",
        },
    ]
    board_after_toggle = {
        **board,
        "playerEquipment": [{"cardNumber": "crow's_nest", "action": 3, "tapped": True}],
    }

    guard.check(board, legal_prompt, turn_no=2, acting_player_id=1)
    guard.check(board_after_toggle, legal_pass_only, turn_no=2, acting_player_id=1)
    result = guard.check(board, legal_pass_only, turn_no=2, acting_player_id=1)
    assert result.force_pass
    assert result.reason == "board_revert"


def test_decision_loop_forces_non_pass_when_available() -> None:
    guard = TurnLoopGuard(max_steps_per_turn=100, loop_repeat_threshold=4)
    state = {
        "turnPhase": {"turnPhase": "CHOOSETOP"},
        "turnNo": 2,
        "playerPitchCount": 0,
        "playerHand": [{"cardNumber": "widowmaker_yellow"}],
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

    result = None
    for _ in range(4):
        result = guard.check(state, legal, turn_no=2, acting_player_id=1)

    assert result is not None
    assert result.force_pass
    assert result.reason == "decision_loop"
    assert result.forced_action is not None
    assert result.forced_action["action_code"] == 8
    mode, button = resolve_forced_submission(legal, result)
    assert mode == 8
    assert button == "widowmaker_yellow"
