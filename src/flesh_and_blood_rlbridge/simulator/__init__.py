"""Legacy Python simulator environments for Flesh and Blood.

These environments use a custom Python game engine rather than the live
Talishar server.  Prefer ``TalisharEngineEnvironment`` for new work.
"""

from .deck_builder_environment import FleshAndBloodDeckBuilderEnvironment
from .environment import FleshAndBloodEnvironment
from .gameplay_environment import FleshAndBloodGameplayEnvironment

__all__ = [
    "FleshAndBloodDeckBuilderEnvironment",
    "FleshAndBloodEnvironment",
    "FleshAndBloodGameplayEnvironment",
]
