"""Flesh and Blood TCG environments for RL Bridge."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__all__ = [
    "ALL_FAB_FACTORIES",
    "FLESH_AND_BLOOD_DECKBUILD_V0",
    "FLESH_AND_BLOOD_SELFPLAY_V0",
    "FLESH_AND_BLOOD_TALISHAR_V0",
    "FleshAndBloodDeckBuilderEnvironment",
    "FleshAndBloodEnvironment",
    "FleshAndBloodFactory",
    "FleshAndBloodGameplayEnvironment",
    "register_environments",
    "register_mcp_tools",
]


def register_environments(registry: Any = None) -> int:
    """Register FaB environment factories with an rlbridge registry."""
    # Import the environment module first so the internal environment <-> factory
    # import cycle is fully resolved before ALL_FAB_FACTORIES is accessed.
    from . import environment as _environment  # noqa: F401
    from .environment_factory import ALL_FAB_FACTORIES

    if registry is None:
        from rlbridge.environments.registry import registry as default_registry

        registry = default_registry

    for factory in ALL_FAB_FACTORIES:
        registry.register(factory)
    return len(ALL_FAB_FACTORIES)


def register_mcp_tools(**kwargs: Any) -> int:
    """Register FaB-specific MCP tools (delegates to the implementation)."""
    from .mcp_tools import register_mcp_tools as _impl

    return _impl(**kwargs)


# Submodule attribute → (module, attribute) used for lazy access below.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "FleshAndBloodEnvironment": ("environment", "FleshAndBloodEnvironment"),
    "ALL_FAB_FACTORIES": ("environment_factory", "ALL_FAB_FACTORIES"),
    "FLESH_AND_BLOOD_DECKBUILD_V0": ("environment_factory", "FLESH_AND_BLOOD_DECKBUILD_V0"),
    "FLESH_AND_BLOOD_SELFPLAY_V0": ("environment_factory", "FLESH_AND_BLOOD_SELFPLAY_V0"),
    "FLESH_AND_BLOOD_TALISHAR_V0": ("environment_factory", "FLESH_AND_BLOOD_TALISHAR_V0"),
    "FleshAndBloodFactory": ("environment_factory", "FleshAndBloodFactory"),
    "FleshAndBloodDeckBuilderEnvironment": (
        "deck_builder_environment",
        "FleshAndBloodDeckBuilderEnvironment",
    ),
    "FleshAndBloodGameplayEnvironment": (
        "gameplay_environment",
        "FleshAndBloodGameplayEnvironment",
    ),
}


def __getattr__(name: str) -> Any:
    """Lazily import public classes/factories (PEP 562).

    Keeping these imports lazy means ``import flesh_and_blood_rlbridge`` does not
    eagerly import ``rlbridge.environments``; otherwise the rlbridge registry's
    import-time plugin loader would re-enter this module before its attributes
    are defined, causing a circular-import warning.
    """
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{target[0]}", __name__)
    return getattr(module, target[1])


if TYPE_CHECKING:  # Static type-checkers and IDEs still see the real symbols.
    from .deck_builder_environment import FleshAndBloodDeckBuilderEnvironment
    from .environment import FleshAndBloodEnvironment
    from .environment_factory import (
        ALL_FAB_FACTORIES,
        FLESH_AND_BLOOD_DECKBUILD_V0,
        FLESH_AND_BLOOD_SELFPLAY_V0,
        FLESH_AND_BLOOD_TALISHAR_V0,
        FleshAndBloodFactory,
    )
    from .gameplay_environment import FleshAndBloodGameplayEnvironment
