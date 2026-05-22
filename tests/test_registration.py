"""Smoke tests for flesh-and-blood-rlbridge registration."""

from __future__ import annotations

from rlbridge.environments.registry import EnvironmentRegistry

from flesh_and_blood_rlip import ALL_FAB_FACTORIES, register_environments


def test_register_environments() -> None:
    reg = EnvironmentRegistry()
    count = register_environments(reg)
    assert count == len(ALL_FAB_FACTORIES)
    for factory in ALL_FAB_FACTORIES:
        assert factory.env_info.env_id in reg


def test_create_talishar_env() -> None:
    reg = EnvironmentRegistry()
    register_environments(reg)
    env = reg.create("FleshAndBlood-Talishar-v0", render_mode=None, format="silver_age")
    try:
        result = env.reset(seed=0)
        assert isinstance(result.observation, dict)
        assert "agent" in result.observation
    finally:
        env.close()
