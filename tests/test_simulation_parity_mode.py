"""Offline tests for simulation parity mode routing."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "cpp"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.cpp_engine_environment import CppEngineEnvironment  # noqa: E402
from check_cpp_vs_talishar_parity import (  # noqa: E402
    ParityReport,
    _configure_simulation_cpp_env,
    run_parity_episode,
)


def test_strict_simulation_blocks_mirror_state() -> None:
    env = object.__new__(CppEngineEnvironment)
    env._strict_simulation = True
    env._talishar_mirror_state = None
    try:
        env.set_talishar_mirror_state({"state": {}})
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "strict simulation" in str(exc).lower()


def test_configure_simulation_cpp_env_sets_flag() -> None:
    wrapper = SimpleNamespace(_cpp_env=SimpleNamespace(_strict_simulation=False))
    _configure_simulation_cpp_env(wrapper, parity_mode="simulation")
    assert wrapper._cpp_env._strict_simulation is True


def test_simulation_episode_does_not_call_mirror() -> None:
    report = ParityReport()
    env_tal = MagicMock()
    env_tal._acting_player_id = 1
    env_tal._last_state = {
        "actingPlayerID": 1,
        "playerHealth": 20,
        "opponentHealth": 20,
        "turnNo": 1,
        "turnPhase": {"turnPhase": "M"},
        "playerHand": [],
        "playerDeckCount": 30,
        "opponentDeckCount": 30,
    }
    env_tal._fetch_state = MagicMock(return_value=env_tal._last_state)
    env_tal._legal_actions = MagicMock(return_value=[])
    env_tal._encode_observation = MagicMock(return_value="{}")
    env_tal.sample_action = MagicMock(return_value="0")

    inner_cpp = MagicMock()
    inner_cpp._strict_simulation = True
    inner_cpp.export_game_state = MagicMock(
        return_value={
            "p1_health": 20,
            "p2_health": 20,
            "turn_no": 1,
            "phase": "M",
            "acting_player_id": 1,
            "priority_player": 1,
            "p1_hand_size": 0,
            "p2_hand_size": 0,
            "p1_deck_count": 30,
            "p2_deck_count": 30,
            "p1_pitch_count": 0,
            "p2_pitch_count": 0,
            "p1_hand": [],
            "p2_hand": [],
            "p1_deck": [],
            "p2_deck": [],
            "p1_discard": [],
            "p2_discard": [],
            "p1_equipment": [],
            "p2_equipment": [],
            "p1_arsenal": [],
            "p2_arsenal": [],
            "p1_pitch": [],
            "p2_pitch": [],
            "p1_banish": [],
            "p2_banish": [],
            "combat_chain": [],
            "pending_attack_power": 0,
            "pending_block_value": 0,
            "p1_resources": 0,
            "p2_resources": 0,
        }
    )
    inner_cpp.apply_initial_sync_from_talishar = MagicMock()
    inner_cpp.set_talishar_mirror_state = MagicMock()

    env_cpp = MagicMock()
    env_cpp._cpp_env = inner_cpp
    env_cpp.reset = MagicMock(
        return_value=SimpleNamespace(observation="{}", info={"legal_actions": []})
    )
    env_cpp.step = MagicMock(
        return_value=SimpleNamespace(
            observation="{}",
            reward=0.0,
            terminated=True,
            truncated=False,
            info={"legal_actions": []},
        )
    )

    with patch("check_cpp_vs_talishar_parity._reset_talishar_for_parity"), patch(
        "check_cpp_vs_talishar_parity._opening_hands_from_talishar",
        return_value={1: [], 2: []},
    ), patch(
        "check_cpp_vs_talishar_parity._hand_playability_from_talishar",
        return_value={},
    ), patch(
        "check_cpp_vs_talishar_parity._build_talishar_reset_snapshot",
        return_value=SimpleNamespace(observation="{}", info={}),
    ), patch(
        "check_cpp_vs_talishar_parity._choose_action",
        return_value=(0, "Pass"),
    ), patch(
        "check_cpp_vs_talishar_parity._talishar_action_descriptor",
        return_value={"action_code": 99},
    ):
        env_tal.step = MagicMock(
            return_value=SimpleNamespace(
                observation='{"actingPlayerID":1,"playerHealth":20,"opponentHealth":20,"turnNo":1,"turnPhase":"M","havePriority":true,"playerHandSize":0,"opponentHandSize":0,"playerDeckCount":30,"opponentDeckCount":30,"playerPitchCount":0,"playerHand":[],"legalActions":[]}',
                reward=0.0,
                terminated=True,
                truncated=False,
                info={"legal_actions": []},
            )
        )
        run_parity_episode(
            env_tal,
            env_cpp,
            report,
            episode=1,
            max_steps=1,
            stress=False,
            parity_mode="simulation",
        )

    inner_cpp.set_talishar_mirror_state.assert_not_called()
