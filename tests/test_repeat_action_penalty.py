"""Tests for repeat-action penalty tracking."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.talishar_default_policy import (
    RepeatActionTracker,
    _REPEAT_ACTION_PENALTY,
    _REPEAT_ACTION_THRESHOLD,
)


def _update(
    tracker: RepeatActionTracker,
    action_key: tuple[int, str],
    *,
    turn_no: int = 1,
    acting_player_id: int = 1,
) -> float:
    return tracker.update(
        action_key,
        turn_no=turn_no,
        acting_player_id=acting_player_id,
    )


def test_exact_repeat_penalty() -> None:
    tracker = RepeatActionTracker()
    play = (27, "0")
    assert _update(tracker, play) == 0.0
    assert _update(tracker, play) == 0.0
    assert _update(tracker, play) == _REPEAT_ACTION_PENALTY
    assert tracker.repeat_streak == _REPEAT_ACTION_THRESHOLD


def test_play_undo_oscillation_penalty() -> None:
    tracker = RepeatActionTracker()
    play = (27, "0")
    undo = (10000, "")
    assert _update(tracker, play) == 0.0
    assert _update(tracker, undo) == 0.0
    assert _update(tracker, play) == 0.0
    assert _update(tracker, undo) == _REPEAT_ACTION_PENALTY
    assert tracker.repeat_streak == _REPEAT_ACTION_THRESHOLD


def test_unrelated_actions_reset_streak() -> None:
    tracker = RepeatActionTracker()
    assert _update(tracker, (27, "0")) == 0.0
    assert _update(tracker, (27, "1")) == 0.0
    assert _update(tracker, (99, "")) == 0.0
    assert tracker.repeat_streak == 1


def test_turn_change_resets_streak() -> None:
    tracker = RepeatActionTracker()
    play = (27, "0")
    _update(tracker, play, turn_no=1)
    _update(tracker, play, turn_no=1)
    assert _update(tracker, play, turn_no=2) == 0.0
    assert tracker.repeat_streak == 1
