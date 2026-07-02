"""Tests for RLStep training-mode payload and parity helpers."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from flesh_and_blood_rlbridge.talishar_engine_environment import TalisharEngineEnvironment
from flesh_and_blood_rlbridge.talishar_oracle import TalisharConnectionError


def test_rlstep_payload_extras_profile_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAB_RLSTEP_PROFILE", "1")
    env = TalisharEngineEnvironment.__new__(TalisharEngineEnvironment)
    extras = env._rlstep_payload_extras()
    assert extras.get("profileTimings") is True


def test_rlstep_payload_extras_debug_gamestate_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAB_RLSTEP_DEBUG_GAMESTATE", "1")
    env = TalisharEngineEnvironment.__new__(TalisharEngineEnvironment)
    extras = env._rlstep_payload_extras()
    assert extras.get("debugGamestate") is True


def test_rlstep_training_payload_includes_training_mode() -> None:
    env = TalisharEngineEnvironment.__new__(TalisharEngineEnvironment)
    env._cpp_env = None
    env._resolved_talishar_backend = "fast"
    env._rlstep_available = True
    env._fast_client = MagicMock()
    env._fast_client.post_rlstep.return_value = {
        "success": True,
        "states": {"1": {"havePriority": True, "playerHealth": 20, "opponentHealth": 20}},
    }
    env._game_name = "12345"
    env._acting_player_id = 1
    env._last_update = 0
    env._auth_key_for = lambda pid: "k1" if pid == 1 else "k2"  # type: ignore[method-assign]
    env._rl_training_mode = True
    env._rl_slim_response = True
    env._rl_use_min_gamestate = True
    env._rl_parity_checked = True
    env._apply_rlstep_states = lambda resp: resp["states"]["1"]  # type: ignore[method-assign]
    env._rlstep_payload_extras = lambda: {}  # type: ignore[method-assign]
    env._maybe_check_rlstep_parity = lambda _resp: None  # type: ignore[method-assign]

    env._submit_action_and_sync(99, "")

    payload = env._fast_client.post_rlstep.call_args.args[0]
    assert payload["trainingMode"] is True
    assert payload["slimResponse"] is True
    assert "useRlGameState" not in payload


def test_rlstep_parity_check_compares_legal_actions() -> None:
    env = TalisharEngineEnvironment.__new__(TalisharEngineEnvironment)
    env._rl_training_mode = True
    env._rl_parity_checked = False
    env._extract_legal_actions = lambda _s: [  # type: ignore[method-assign]
        {"action_code": 99, "button_input": "", "zone": "button", "label": "Pass"},
    ]
    env._filter_legal_actions = lambda _s, actions: actions  # type: ignore[method-assign]
    env._encode_observation = lambda _s, _a: "{}"  # type: ignore[method-assign]
    env._last_observation_vec = np.zeros(4, dtype=np.float64)

    state = {
        "havePriority": True,
        "playerHealth": 20,
        "opponentHealth": 20,
        "turnPhase": {"turnPhase": "M"},
        "playerHand": [],
    }
    with patch.dict(os.environ, {"FAB_RLSTEP_PARITY_CHECK": "1"}):
        env._maybe_check_rlstep_parity(
            {"compareStates": {"rl": state, "full": dict(state)}}
        )
    assert env._rl_parity_checked is True


@pytest.mark.talishar
def test_rlstep_training_live_parity_smoke() -> None:
    """Compare RL vs full gamestate build on the first training step."""
    base_url = os.environ.get("TALISHAR_URL", "http://localhost:8080/game")
    env = TalisharEngineEnvironment(
        base_url=base_url,
        local_deck_name="Ira",
        opponent_deck_name="Ira",
        self_play=True,
        talishar_backend="fast",
        rl_training_mode=True,
    )
    if not env.supports_fast_training:
        pytest.skip("Talishar fast backend / RLStep overlay not available")
    try:
        with patch.dict(os.environ, {"FAB_RLSTEP_PARITY_CHECK": "1"}):
            env.fast_reset()
            env.fast_step_index(0)
    except TalisharConnectionError as exc:
        pytest.skip(f"Talishar server not reachable: {exc}")
    finally:
        env.close()
