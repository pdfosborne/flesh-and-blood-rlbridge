"""Tests for Talishar gamestate-revert anti-stuck recovery."""

from __future__ import annotations

from unittest.mock import MagicMock

from flesh_and_blood_rlbridge.combat_log_tracker import (
    extract_talishar_chat_log_lines,
    talishar_gamestate_revert_detected,
)
from flesh_and_blood_rlbridge.talishar_engine_environment import TalisharEngineEnvironment


def test_talishar_gamestate_revert_detected_matches_declaration_message() -> None:
    state = {
        "chatLog": (
            "Player 1 played a card.<br>"
            "<p style='background: brown;'>"
            "<span style='color:azure;'>"
            "You have resources to pay for, but have no cards to pitch. "
            "Reverting gamestate prior to that declaration."
            "</span></p>"
        ),
    }
    assert talishar_gamestate_revert_detected(state)


def test_talishar_gamestate_revert_detected_ignores_unrelated_log() -> None:
    state = {"chatLog": "Player 1 played Nimble Strike."}
    assert not talishar_gamestate_revert_detected(state)


def test_extract_talishar_chat_log_lines_strips_html() -> None:
    lines = extract_talishar_chat_log_lines(
        "Line one<br><span>Line two</span>",
    )
    assert lines == ["Line one", "Line two"]


def _bare_env(*, using_cpp: bool = False) -> TalisharEngineEnvironment:
    env = object.__new__(TalisharEngineEnvironment)
    env._cpp_env = MagicMock() if using_cpp else None
    env._legal_actions = MagicMock(  # type: ignore[method-assign]
        return_value=[
            {"action_code": 99, "button_input": "", "label": "Pass", "zone": "button"},
        ],
    )
    env._force_pass_submission = MagicMock(  # type: ignore[method-assign]
        return_value=(99, ""),
    )
    env._submit_action_and_sync = MagicMock(  # type: ignore[method-assign]
        return_value={"havePriority": True, "chatLog": ""},
    )
    return env


def test_recover_from_gamestate_revert_forces_pass() -> None:
    env = _bare_env()
    reverted = {
        "havePriority": True,
        "chatLog": "Reverting gamestate prior to that declaration.",
    }

    recovered = env._recover_from_gamestate_revert_if_needed(
        reverted,
        submitted_mode=27,
        submitted_button="0",
    )

    env._force_pass_submission.assert_called_once()
    env._submit_action_and_sync.assert_called_once_with(99, "")
    assert recovered == {"havePriority": True, "chatLog": ""}


def test_recover_from_gamestate_revert_skips_when_already_passing() -> None:
    env = _bare_env()
    reverted = {
        "havePriority": True,
        "chatLog": "Reverting gamestate prior to that effect.",
    }

    recovered = env._recover_from_gamestate_revert_if_needed(
        reverted,
        submitted_mode=99,
        submitted_button="",
    )

    env._submit_action_and_sync.assert_not_called()
    assert recovered is reverted


def test_recover_from_gamestate_revert_skips_for_cpp_engine() -> None:
    env = _bare_env(using_cpp=True)
    reverted = {
        "havePriority": True,
        "chatLog": "Reverting gamestate prior to that declaration.",
    }

    recovered = env._recover_from_gamestate_revert_if_needed(
        reverted,
        submitted_mode=27,
        submitted_button="0",
    )

    env._submit_action_and_sync.assert_not_called()
    assert recovered is reverted
