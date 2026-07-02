"""Tests for cross-turn macro stall detection."""

from __future__ import annotations

from flesh_and_blood_rlbridge.macro_stall_guard import MacroStallConfig, MacroStallGuard


def _state(
    *,
    turn_no: int = 1,
    phase: str = "M",
    acting: int = 1,
    p1_hp: int = 20,
    p2_hp: int = 20,
    hand_size: int = 4,
) -> dict:
    return {
        "turnNo": turn_no,
        "turnPhase": phase,
        "actingPlayerID": acting,
        "playerHealth": p1_hp if acting == 1 else p2_hp,
        "opponentHealth": p2_hp if acting == 1 else p1_hp,
        "playerHandSize": hand_size,
    }


def test_turns_without_damage_increment_and_reset() -> None:
    guard = MacroStallGuard(
        MacroStallConfig(stall_no_damage_turns=3, stall_pass_only_turns=99)
    )
    pass_only = [{"action_code": 99, "label": "Pass", "zone": "button"}]

    for turn in range(1, 4):
        result = guard.observe(
            _state(turn_no=turn, acting=1 if turn % 2 else 2),
            pass_only,
            p1_hp=20,
            p2_hp=20,
        )
        assert not result.should_truncate

    result = guard.observe(_state(turn_no=4, acting=1), pass_only, p1_hp=20, p2_hp=20)
    assert result.should_truncate
    assert result.reason == "no_damage_turns"
    assert result.turns_without_damage == 3

    guard.reset()
    guard.observe(_state(turn_no=1), pass_only, p1_hp=20, p2_hp=20)
    guard.observe(_state(turn_no=2), pass_only, p1_hp=18, p2_hp=20)
    result = guard.observe(_state(turn_no=3), pass_only, p1_hp=18, p2_hp=20)
    assert result.turns_without_damage == 1
    assert not result.should_truncate


def test_pass_only_main_streak_truncates() -> None:
    guard = MacroStallGuard(
        MacroStallConfig(stall_no_damage_turns=99, stall_pass_only_turns=2)
    )
    pass_only = [{"action_code": 99, "label": "Pass", "zone": "button"}]
    playable = [
        {"action_code": 27, "label": "Attack", "zone": "hand"},
        {"action_code": 99, "label": "Pass", "zone": "button"},
    ]

    result = guard.observe(_state(turn_no=1, acting=1), pass_only, p1_hp=20, p2_hp=20)
    assert not result.should_truncate
    assert result.pass_only_main_streak == 1

    result = guard.observe(_state(turn_no=1, acting=2), pass_only, p1_hp=20, p2_hp=20)
    assert result.should_truncate
    assert result.reason == "pass_only_main"
    assert result.pass_only_main_streak == 2

    guard.reset()
    guard.observe(_state(turn_no=1, acting=1), pass_only, p1_hp=20, p2_hp=20)
    result = guard.observe(_state(turn_no=1, acting=2), playable, p1_hp=20, p2_hp=20)
    assert result.pass_only_main_streak == 0
    assert not result.should_truncate


def test_low_hand_gate_blocks_no_damage_truncation() -> None:
    guard = MacroStallGuard(
        MacroStallConfig(
            stall_no_damage_turns=2,
            stall_pass_only_turns=99,
            stall_no_damage_requires_low_hand=True,
            stall_low_hand_turns=2,
            stall_max_single_low_hand_turns=99,
            stall_min_attack_hand=2,
        )
    )
    pass_only = [{"action_code": 99, "label": "Pass", "zone": "button"}]

    guard.observe(_state(turn_no=1, acting=1, hand_size=4), pass_only, p1_hp=20, p2_hp=20)
    result = guard.observe(
        _state(turn_no=2, acting=2, hand_size=4),
        pass_only,
        p1_hp=20,
        p2_hp=20,
    )
    assert not result.should_truncate

    guard.reset()
    guard.observe(_state(turn_no=1, acting=1, hand_size=1), pass_only, p1_hp=20, p2_hp=20)
    guard.observe(_state(turn_no=1, acting=2, hand_size=1), pass_only, p1_hp=20, p2_hp=20)
    guard.observe(_state(turn_no=2, acting=1, hand_size=1), pass_only, p1_hp=20, p2_hp=20)
    guard.observe(_state(turn_no=2, acting=2, hand_size=1), pass_only, p1_hp=20, p2_hp=20)
    result = guard.observe(
        _state(turn_no=3, acting=1, hand_size=1),
        pass_only,
        p1_hp=20,
        p2_hp=20,
    )
    assert result.should_truncate
    assert result.reason == "no_damage_turns"


def test_no_damage_with_real_plays_does_not_truncate() -> None:
    """Flat HP alone shouldn't stall-truncate when real non-pass plays existed
    every turn — only genuinely pass-only flat-HP turns should count."""
    guard = MacroStallGuard(
        MacroStallConfig(stall_no_damage_turns=3, stall_pass_only_turns=99)
    )
    playable = [
        {"action_code": 27, "label": "battalion_barque_red", "zone": "hand"},
        {"action_code": 5, "label": "saltwater_swell_red", "zone": "arsenal"},
        {"action_code": 99, "label": "Pass", "zone": "button"},
    ]

    for turn in range(1, 6):
        result = guard.observe(
            _state(turn_no=turn, acting=1 if turn % 2 else 2),
            playable,
            p1_hp=20,
            p2_hp=20,
        )
        assert not result.should_truncate

    # Contrast: the same flat HP with genuinely pass-only turns still truncates.
    guard.reset()
    pass_only = [{"action_code": 99, "label": "Pass", "zone": "button"}]
    result = None
    for turn in range(1, 5):
        result = guard.observe(
            _state(turn_no=turn, acting=1 if turn % 2 else 2),
            pass_only,
            p1_hp=20,
            p2_hp=20,
        )
    assert result is not None
    assert result.should_truncate
    assert result.reason == "no_damage_turns"


def test_disabled_guard_never_truncates() -> None:
    guard = MacroStallGuard(MacroStallConfig(enabled=False, stall_no_damage_turns=1))
    pass_only = [{"action_code": 99, "label": "Pass", "zone": "button"}]
    for turn in range(1, 5):
        result = guard.observe(_state(turn_no=turn), pass_only, p1_hp=20, p2_hp=20)
    assert not result.should_truncate


def test_mutual_pass_stall_sets_mutual_loss_flag() -> None:
    guard = MacroStallGuard(
        MacroStallConfig(
            stall_no_damage_turns=99,
            stall_pass_only_turns=99,
            stall_mutual_pass_turns=2,
        )
    )
    pass_only = [{"action_code": 99, "label": "Pass", "zone": "button"}]

    guard.observe(_state(turn_no=1, acting=1), pass_only, p1_hp=20, p2_hp=20)
    guard.observe(_state(turn_no=1, acting=2), pass_only, p1_hp=20, p2_hp=20)
    guard.observe(_state(turn_no=2, acting=1), pass_only, p1_hp=20, p2_hp=20)
    result = guard.observe(_state(turn_no=2, acting=2), pass_only, p1_hp=20, p2_hp=20)

    assert result.should_truncate
    assert result.reason == "mutual_pass_stall"
    assert result.mutual_stall_loss is True
