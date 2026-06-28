"""Tests for Talishar fast backend wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from flesh_and_blood_rlbridge.player_observation import ACTION_CAPACITY, PLAYER_OBS_DIM
from flesh_and_blood_rlbridge.talishar_engine_environment import TalisharEngineEnvironment


def test_fast_backend_disables_cpp_obs_alignment() -> None:
    env = TalisharEngineEnvironment.__new__(TalisharEngineEnvironment)
    env._cpp_env = None
    env._talishar_backend_requested = "fast"
    env._base_url = "http://localhost:8080/game"
    env._request_timeout = 5.0
    env._combat_tracker = MagicMock()
    env._cpp_obs_alignment = True

    with patch("flesh_and_blood_rlbridge.talishar_engine_environment.TalisharFastClient") as mock_client:
        instance = mock_client.return_value
        instance.session = MagicMock()
        instance.probe_rlstep.return_value = False
        env._finalize_talishar_backend(use_cpp_engine=False)

    assert env._cpp_obs_alignment is False


def test_supports_fast_training_when_fast_backend_resolved() -> None:
    env = TalisharEngineEnvironment.__new__(TalisharEngineEnvironment)
    env._cpp_env = None
    env._resolved_talishar_backend = "fast"
    assert env.supports_fast_training is True


def test_fast_action_capacity_matches_schema() -> None:
    env = TalisharEngineEnvironment.__new__(TalisharEngineEnvironment)
    env._cpp_env = None
    assert env.fast_action_capacity() == ACTION_CAPACITY


def test_build_fast_step_result_shape() -> None:
    env = TalisharEngineEnvironment.__new__(TalisharEngineEnvironment)
    env._acting_player_id = 1
    env._player_hp = 18
    env._opp_hp = 20
    env._last_state = {"playerHealth": 18, "opponentHealth": 20, "turnNo": 2, "playerDeckCount": 40, "opponentDeckCount": 39}
    env._last_observation_vec = np.zeros(PLAYER_OBS_DIM, dtype=np.float64)
    env._legal_actions = lambda _s: []  # type: ignore[method-assign]
    env._filter_legal_actions = lambda _s, actions: actions  # type: ignore[method-assign]
    result = env._build_fast_step_result(reward=-0.001, terminated=False, truncated=False)
    assert result["obs_vec"].shape == (PLAYER_OBS_DIM,)
    assert result["legal_count"] == 0
    assert result["p1_health"] == 18
    assert result["p2_health"] == 20


def test_finalize_talishar_backend_fast() -> None:
    env = TalisharEngineEnvironment.__new__(TalisharEngineEnvironment)
    env._cpp_env = None
    env._talishar_backend_requested = "fast"
    env._base_url = "http://localhost:8080/game"
    env._request_timeout = 5.0
    env._combat_tracker = MagicMock()
    env._cpp_obs_alignment = True

    with patch("flesh_and_blood_rlbridge.talishar_engine_environment.TalisharFastClient") as mock_client:
        instance = mock_client.return_value
        instance.session = MagicMock()
        instance.probe_rlstep.return_value = True
        env._finalize_talishar_backend(use_cpp_engine=False)

    assert env._resolved_talishar_backend == "fast"
    assert env._rlstep_available is True
    assert env._cpp_obs_alignment is False


def test_fast_logic_policy_action_index_on_talishar() -> None:
    env = TalisharEngineEnvironment.__new__(TalisharEngineEnvironment)
    env._cpp_env = None
    env._last_state = {"turnPhase": "M", "playerHealth": 20, "opponentHealth": 20}
    env._block_max_pitch_value = 3
    env._block_min_resource_cost = 0
    env._legal_actions = lambda _s: [  # type: ignore[method-assign]
        {"action_code": 27, "button_input": "0", "label": "Attack", "zone": "hand"},
        {"action_code": 99, "button_input": "", "label": "Pass", "zone": "button"},
    ]
    env._filter_legal_actions = lambda _s, actions: actions  # type: ignore[method-assign]
    env._loop_guard_for_step = lambda _s, _a: MagicMock(force_pass=False)  # type: ignore[method-assign]

    with patch(
        "flesh_and_blood_rlbridge.talishar_engine_environment.choose_talishar_action_index",
        return_value=0,
    ) as choose:
        idx = env.fast_logic_policy_action_index()
    assert idx == 0
    choose.assert_called_once()


def test_apply_rlstep_states_picks_priority_holder() -> None:
    env = TalisharEngineEnvironment.__new__(TalisharEngineEnvironment)
    env._acting_player_id = 1
    env._auth_key = "k1"
    env._p1_auth_key = "k1"
    env._p2_auth_key = "k2"
    env._last_state = {}

    def adopt(state: dict, pid: int) -> dict:
        env._acting_player_id = pid
        return state

    env._adopt_player_state = adopt  # type: ignore[method-assign]
    env._is_game_over = lambda _s: False  # type: ignore[method-assign]
    env._auth_key_for = lambda pid: "k1" if pid == 1 else "k2"  # type: ignore[method-assign]

    resp = {
        "success": True,
        "states": {
            "1": {"havePriority": False, "playerHealth": 20},
            "2": {"havePriority": True, "playerHealth": 19},
        },
    }
    state = env._apply_rlstep_states(resp)
    assert state["playerHealth"] == 19
    assert env._acting_player_id == 2
