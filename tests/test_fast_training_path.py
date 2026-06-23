"""Fast C++ training path availability."""

from __future__ import annotations

from pathlib import Path

import pytest

from flesh_and_blood_rlbridge.cpp_engine_environment import (
    CppEngineEnvironment,
    is_cpp_engine_available,
)


def _ira_engine_dir() -> Path | None:
    root = Path(__file__).resolve().parents[1] / "results" / "cpp_engines"
    for candidate in sorted(root.glob("Ira_vs_Ira*")):
        if is_cpp_engine_available(candidate):
            return candidate
    return None


def test_combat_tracker_does_not_disable_fast_training() -> None:
    engine_dir = _ira_engine_dir()
    if engine_dir is None:
        pytest.skip("Ira vs Ira C++ engine not built")

    env = CppEngineEnvironment(
        engine_dir=engine_dir,
        enable_combat_tracker=True,
    )
    assert env.supports_fast_training
    assert env.fast_training_unavailable_reasons() == []
    state = env.fast_reset(seed=0)
    next_state = env.fast_step_index(0)
    assert "obs_vec" in state
    assert "obs_vec" in next_state


def test_fast_training_unavailable_reasons_when_module_incomplete() -> None:
    env = object.__new__(CppEngineEnvironment)
    env._gs = None

    class _Fab:
        class GameState:
            pass

    env._fab = _Fab()
    reasons = env.fast_training_unavailable_reasons()
    assert any("fast_step_index" in reason for reason in reasons)
    assert not env.supports_fast_training
