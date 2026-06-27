"""Tests for Talishar self-play priority inference and resync."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.talishar_engine_environment import TalisharEngineEnvironment


def _env() -> TalisharEngineEnvironment:
    env = object.__new__(TalisharEngineEnvironment)
    env._acting_player_id = 1
    env._player_hp = 20
    env._opp_hp = 20
    env._self_play = True
    env._verbose = False
    env._deck_nonzero_ever_seen = False
    env._last_update = 0
    return env


def test_infer_priority_from_waiting_prompt() -> None:
    env = _env()
    states = {
        1: {
            "havePriority": False,
            "playerPrompt": {
                "helpText": "Waiting for other player to choose an instant",
            },
            "turnPhase": {
                "turnPhase": "INSTANT",
                "caption": "Your opponent is choosing an instant",
            },
        },
        2: {
            "havePriority": False,
            "playerPrompt": {"helpText": "Choose an instant from your hand"},
            "turnPhase": {"turnPhase": "INSTANT", "caption": "Choose an instant"},
        },
    }

    assert env._infer_priority_player(states) == 2


def test_infer_priority_from_have_priority_flag() -> None:
    env = _env()
    states = {
        1: {"havePriority": False},
        2: {"havePriority": True},
    }

    assert env._infer_priority_player(states) == 2


def test_resolve_priority_holder_adopts_waiting_opponent() -> None:
    env = _env()
    env._fetch_both_player_states = lambda: {  # type: ignore[method-assign]
        1: {
            "havePriority": False,
            "playerHealth": 20,
            "opponentHealth": 20,
            "playerPrompt": {
                "helpText": "Waiting for other player to choose an instant",
            },
            "turnPhase": {"turnPhase": "INSTANT"},
            "lastUpdate": 10,
        },
        2: {
            "havePriority": True,
            "playerHealth": 20,
            "opponentHealth": 20,
            "playerPrompt": {"helpText": "Choose an instant from your hand"},
            "turnPhase": {"turnPhase": "INSTANT", "caption": "Choose an instant"},
            "lastUpdate": 11,
        },
    }
    env._fetch_state = lambda player_id=1, last_update=0: env._fetch_both_player_states()[player_id]  # type: ignore[method-assign]
    env._auth_key_for = lambda _pid: "auth"  # type: ignore[method-assign]

    state = env._resolve_priority_holder()

    assert env._acting_player_id == 2
    assert state.get("havePriority") is True
