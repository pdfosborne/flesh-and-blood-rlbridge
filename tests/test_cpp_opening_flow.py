"""Regression tests for C++ engine opening-flow progression."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.cpp_engine_environment import CppEngineEnvironment


def _build_flow_env() -> CppEngineEnvironment:
    env = object.__new__(CppEngineEnvironment)
    env._gs = SimpleNamespace(
        turn_no=0,
        priority=0,
        p1_health=20,
        p2_health=20,
        p1_hand_size=4,
        p2_hand_size=4,
        p1_deck_size=30,
        p2_deck_size=30,
        p1_pitch_size=0,
        p2_pitch_size=0,
    )
    env._acting_player = 1
    env._flow_phase = "OPENING_MAIN"
    env._arsenal_complete = set()
    env._turn_no_override = None
    env._talishar_overlay = None
    env._talishar_parity_extra = None
    env._talishar_mirror_state = None
    env._hand_playability = {}
    return env


def test_opening_main_pass_enters_ars() -> None:
    env = _build_flow_env()
    assert env._handle_flow_pass() is True
    assert env._flow_phase == "ARS"
    assert env._acting_player == 1


def test_ars_pass_advances_to_opponent_then_main() -> None:
    env = _build_flow_env()
    env._flow_phase = "ARS"

    assert env._handle_flow_pass() is True
    assert env._flow_phase == "ARS"
    assert env._acting_player == 2
    assert env._turn_no_override == 1

    assert env._handle_flow_pass() is True
    assert env._flow_phase == "M"
    assert env._acting_player == 1
    assert env._turn_no_override is None


def test_ars_arsenal_advances_like_pass() -> None:
    """Arsenal (action code 4) must not be a no-op during ARS."""
    env = _build_flow_env()
    env._flow_phase = "ARS"
    arsenal = SimpleNamespace(action_code=4, button_input="card_1", card_id="card_1", zone="hand", label="Card")

    assert env._handle_flow_hand_action(arsenal) is True
    assert env._flow_phase == "ARS"
    assert env._acting_player == 2

    assert env._handle_flow_hand_action(arsenal) is True
    assert env._flow_phase == "M"
    assert env._acting_player == 1


def test_obs_turn_no_uses_cpp_after_main_phase() -> None:
    env = _build_flow_env()
    env._flow_phase = "M"
    env._turn_no_override = None
    env._gs.turn_no = 7
    assert env._obs_turn_no() == 7


def test_always_action_zero_progresses_with_real_engine() -> None:
    """Untrained policies often pick action 0; must not stall in ARS forever."""
    from flesh_and_blood_rlbridge.cpp_engine_environment import (
        get_engine_dir,
        is_cpp_engine_available,
    )

    engine_dir = get_engine_dir("briar", "riptide")
    if not is_cpp_engine_available(engine_dir):
        return

    env = CppEngineEnvironment(engine_dir=engine_dir, max_turns=200)
    env.reset()

    for _ in range(200):
        out = env.step("0")
        if out.terminated or out.truncated:
            break

    assert env._flow_phase != "ARS", "episode stuck in ARS opening phase"
    assert env._gs.turn_no > 0 or env._gs.p1_health < 20 or env._gs.p2_health < 20
