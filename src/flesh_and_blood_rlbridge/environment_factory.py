from __future__ import annotations

from typing import Any, Optional

from rlbridge.environments.base import rlbridgeEnvironmentFactory
from rlbridge.protocol.messages import EnvironmentInfo, SuggestedHyperparameters

from .deck_builder_environment import FleshAndBloodDeckBuilderEnvironment
from .gameplay_environment import FleshAndBloodGameplayEnvironment


class FleshAndBloodFactory(rlbridgeEnvironmentFactory):
    def __init__(
        self,
        env_id: str,
        *,
        agent_hero_id: str = "hero_dorinthea_ironsong",
        opponent_hero_id: str = "hero_rhinar_reckless_rampage",
        max_turns: int = 60,
        deck_size: int = 36,
        format: str = "classic_constructed",
        two_phase_deckbuild: bool = False,
        self_play: bool = False,
        opponent_type: str = "preset_logic",
    ) -> None:
        self._env_id = env_id
        self._agent_hero_id = agent_hero_id
        self._opponent_hero_id = opponent_hero_id
        self._max_turns = max_turns
        self._deck_size = deck_size
        self._format = format
        self._two_phase_deckbuild = two_phase_deckbuild
        self._self_play = self_play
        self._opponent_type = opponent_type

    @property
    def env_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            env_id=self._env_id,
            description=(
                "Talishar-inspired Flesh and Blood simulation with structured card "
                "database, hand/deck/combat phases, and scripted opponent policy."
                if not self._self_play
                else "Talishar-inspired Flesh and Blood self-play simulation where "
                "one policy controls both heroes across alternating turns."
            ),
            tags=[
                "tcg",
                "flesh-and-blood",
                "card-game",
                "turn-based",
                "simulator",
                *(["deck-selection"] if self._two_phase_deckbuild else []),
                *(["self-play"] if self._self_play else []),
            ],
            namespace="flesh_and_blood",
            render_modes=["ansi", "rgb_array"],
            max_episode_steps=self._max_turns,
            suggested_hyperparameters=SuggestedHyperparameters(
                agent_type="ppo",
                n_episodes=300,
                max_steps=self._max_turns,
                alpha=0.1,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.05,
                epsilon_decay=0.997,
                sub_goal_threshold=0.5,
                top_k=3,
                min_episode_visits=2,
            ),
        )

    def create(
        self,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        _ = render_mode
        if bool(kwargs.get("two_phase_deckbuild", self._two_phase_deckbuild)):
            return FleshAndBloodDeckBuilderEnvironment(
                seed=kwargs.get("seed"),
                agent_hero_id=kwargs.get("agent_hero_id", self._agent_hero_id),
                opponent_hero_id=kwargs.get("opponent_hero_id", self._opponent_hero_id),
                max_turns=int(kwargs.get("max_turns", self._max_turns)),
                deck_size=int(kwargs.get("deck_size", self._deck_size)),
                format=str(kwargs.get("format", self._format)),
                self_play=bool(kwargs.get("self_play", self._self_play)),
                opponent_type=str(kwargs.get("opponent_type", self._opponent_type)),
                render_mode=render_mode,
                quality_eval_episodes=int(kwargs.get("quality_eval_episodes", 1)),
                quality_max_steps=int(kwargs.get("quality_max_steps", 60)),
            )

        return FleshAndBloodGameplayEnvironment(
            seed=kwargs.get("seed"),
            agent_hero_id=kwargs.get("agent_hero_id", self._agent_hero_id),
            opponent_hero_id=kwargs.get("opponent_hero_id", self._opponent_hero_id),
            max_turns=int(kwargs.get("max_turns", self._max_turns)),
            deck_size=int(kwargs.get("deck_size", self._deck_size)),
            format=str(kwargs.get("format", self._format)),
            self_play=bool(kwargs.get("self_play", self._self_play)),
            opponent_type=str(kwargs.get("opponent_type", self._opponent_type)),
            render_mode=render_mode,
        )


FLESH_AND_BLOOD_TALISHAR_V0 = FleshAndBloodFactory("FleshAndBlood-Talishar-v0")
FLESH_AND_BLOOD_SELFPLAY_V0 = FleshAndBloodFactory(
    "FleshAndBlood-SelfPlay-v0",
    self_play=True,
)
FLESH_AND_BLOOD_DECKBUILD_V0 = FleshAndBloodFactory(
    "FleshAndBlood-DeckBuild-v0",
    two_phase_deckbuild=True,
)
ALL_FAB_FACTORIES: list[FleshAndBloodFactory] = [
    FLESH_AND_BLOOD_TALISHAR_V0,
    FLESH_AND_BLOOD_SELFPLAY_V0,
    FLESH_AND_BLOOD_DECKBUILD_V0,
]
