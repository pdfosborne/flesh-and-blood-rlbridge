from __future__ import annotations

from typing import Any, Optional

from .environment import FleshAndBloodEnvironment


class FleshAndBloodGameplayEnvironment(FleshAndBloodEnvironment):
    """Gameplay-only FaB environment (no deck-selection stage)."""

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        agent_hero_id: str = "hero_dorinthea_ironsong",
        opponent_hero_id: str = "hero_rhinar_reckless_rampage",
        max_turns: int = 2000,
        deck_size: int = 36,
        agent_deck_style: str = "balanced",
        opponent_deck_style: str = "balanced",
        format: str = "classic_constructed",
        self_play: bool = False,
        opponent_type: str = "preset_logic",
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__(
            seed=seed,
            agent_hero_id=agent_hero_id,
            opponent_hero_id=opponent_hero_id,
            max_turns=max_turns,
            deck_size=deck_size,
            agent_deck_style=agent_deck_style,
            opponent_deck_style=opponent_deck_style,
            format=format,
            two_phase_deckbuild=False,
            self_play=self_play,
            opponent_type=opponent_type,
            render_mode=render_mode,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Any:
        opts = dict(options or {})
        opts["two_phase_deckbuild"] = False
        return super().reset(seed=seed, options=opts)
