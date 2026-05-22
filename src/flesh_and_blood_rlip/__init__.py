"""Flesh and Blood TCG environments for RL Bridge."""

from __future__ import annotations

from typing import Any

from .environment import (
    ALL_FAB_FACTORIES,
    FLESH_AND_BLOOD_DECKBUILD_V0,
    FLESH_AND_BLOOD_SELFPLAY_V0,
    FLESH_AND_BLOOD_TALISHAR_V0,
    FleshAndBloodEnvironment,
    FleshAndBloodFactory,
    register_mcp_tools,
)

__all__ = [
    "ALL_FAB_FACTORIES",
    "FLESH_AND_BLOOD_DECKBUILD_V0",
    "FLESH_AND_BLOOD_SELFPLAY_V0",
    "FLESH_AND_BLOOD_TALISHAR_V0",
    "FleshAndBloodEnvironment",
    "FleshAndBloodFactory",
    "register_environments",
    "register_mcp_tools",
]


def register_environments(registry: Any = None) -> int:
    """Register FaB environment factories with an RLIP registry."""
    if registry is None:
        from rlbridge.environments.registry import registry as default_registry

        registry = default_registry

    for factory in ALL_FAB_FACTORIES:
        registry.register(factory)
    return len(ALL_FAB_FACTORIES)
