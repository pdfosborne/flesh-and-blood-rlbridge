from __future__ import annotations

from typing import Any, Optional

from rlbridge.environments.base import rlbridgeEnvironment, rlbridgeEnvironmentFactory
from rlbridge.protocol.messages import EnvironmentInfo, SuggestedHyperparameters

_TALISHAR_DEFAULT_DECK = "https://fabrary.net/decks/01GJG7Z4WGWSZ95FY74KX4M557"


class TalisharEngineFactory(rlbridgeEnvironmentFactory):
    """Factory that creates :class:`TalisharEngineEnvironment` instances."""

    def __init__(
        self,
        env_id: str,
        *,
        deck_link: str = _TALISHAR_DEFAULT_DECK,
        game_format: str = "silver_age",
        max_turns: int = 2000,
        self_play: bool = True,
    ) -> None:
        self._env_id = env_id
        self._deck_link = deck_link
        self._game_format = game_format
        self._max_turns = max_turns
        self._self_play = self_play

    @property
    def env_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            env_id=self._env_id,
            description=(
                "Flesh and Blood TCG environment backed by a live Talishar server "
                "(https://github.com/Talishar/Talishar).  "
                + (
                    "One policy controls both players (self-play; learns from both sides).  "
                    if self._self_play
                    else "The agent plays as player 1 against the built-in CombatDummy AI.  "
                )
                + "Requires a running Talishar Docker instance (set TALISHAR_URL env var)."
            ),
            tags=[
                "tcg",
                "flesh-and-blood",
                "card-game",
                "turn-based",
                "talishar",
                *(["self-play"] if self._self_play else []),
            ],
            namespace="flesh_and_blood",
            render_modes=["human", "ansi", "rgb_array"],
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
    ) -> rlbridgeEnvironment:
        from .talishar_engine_environment import TalisharEngineEnvironment

        local_deck = kwargs.get("local_deck_name", "Ira")
        opponent_deck = kwargs.get("opponent_deck_name", None)
        engine_keys = (
            "max_steps_per_turn",
            "loop_repeat_threshold",
            "step_penalty",
            "truncation_penalty",
            "repeat_action_threshold",
            "repeat_action_penalty",
            "damage_reward_scale",
            "max_consecutive_passes",
            "request_timeout",
        )
        engine_kw = {key: kwargs[key] for key in engine_keys if key in kwargs}
        return TalisharEngineEnvironment(
            base_url=kwargs.get("base_url"),
            frontend_url=kwargs.get("frontend_url"),
            deck_link=str(kwargs.get("deck_link", self._deck_link)),
            game_format=str(kwargs.get("game_format", self._game_format)),
            max_turns=int(kwargs.get("max_turns", self._max_turns)),
            render_mode=render_mode,
            local_deck_name=local_deck if local_deck is not None else None,
            opponent_deck_name=opponent_deck,
            self_play=bool(kwargs.get("self_play", self._self_play)),
            render_width=kwargs.get("render_width"),
            render_height=kwargs.get("render_height"),
            use_cpp_engine=bool(kwargs.get("use_cpp_engine", False)),
            talishar_backend=str(kwargs.get("talishar_backend", "fast")),
            cpp_obs_alignment=False,
            cpp_engine_cache_dir=kwargs.get("cpp_engine_cache_dir"),
            enable_combat_tracker=bool(kwargs.get("enable_combat_tracker", False)),
            **engine_kw,
        )


class TalisharDeckBuilderFactory(rlbridgeEnvironmentFactory):
    """Factory that creates :class:`TalisharDeckBuilderEnvironment` instances."""

    def __init__(
        self,
        env_id: str,
        *,
        hero_id: str = "ira_crimson_haze",
        hero_class: str = "Ninja",
        game_format: str = "silver_age",
        num_eval_games: int = 5,
        opponent_deck_name: str = "Ira",
        opponent_hero_id: str = "dorinthea_ironsong",
        num_sideboard_episodes: int = 10,
        max_build_steps: int = 200,
        starting_deck: Optional[dict[str, int]] = None,
    ) -> None:
        self._env_id = env_id
        self._hero_id = hero_id
        self._hero_class = hero_class
        self._game_format = game_format
        self._num_eval_games = num_eval_games
        self._opponent_deck_name = opponent_deck_name
        self._opponent_hero_id = opponent_hero_id
        self._num_sideboard_episodes = num_sideboard_episodes
        self._max_build_steps = max_build_steps
        self._starting_deck: Optional[dict[str, int]] = starting_deck

    @property
    def env_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            env_id=self._env_id,
            description=(
                "Flesh and Blood TCG deck-building environment.  The agent "
                "constructs a deck card-by-card; reward is based on win rate "
                "against the Talishar CombatDummy AI.  Requires a running "
                "Talishar Docker instance (set TALISHAR_URL env var)."
            ),
            tags=[
                "tcg",
                "flesh-and-blood",
                "card-game",
                "deck-building",
                "talishar",
            ],
            namespace="flesh_and_blood",
            render_modes=["ansi"],
            max_episode_steps=self._max_build_steps,
            suggested_hyperparameters=SuggestedHyperparameters(
                agent_type="ppo",
                n_episodes=500,
                max_steps=self._max_build_steps,
                alpha=0.1,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.05,
                epsilon_decay=0.995,
                sub_goal_threshold=0.5,
                top_k=3,
                min_episode_visits=2,
            ),
        )

    def create(
        self,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> rlbridgeEnvironment:
        from .talishar_deckbuilder_environment import TalisharDeckBuilderEnvironment

        return TalisharDeckBuilderEnvironment(
            hero_id=str(kwargs.get("hero_id", self._hero_id)),
            hero_class=str(kwargs.get("hero_class", self._hero_class)),
            game_format=str(kwargs.get("game_format", self._game_format)),
            num_eval_games=int(kwargs.get("num_eval_games", self._num_eval_games)),
            opponent_deck_name=str(kwargs.get("opponent_deck_name", self._opponent_deck_name)),
            opponent_hero_id=str(kwargs.get("opponent_hero_id", self._opponent_hero_id)),
            num_sideboard_episodes=int(
                kwargs.get("num_sideboard_episodes", self._num_sideboard_episodes)
            ),
            sideboard_agent=kwargs.get("sideboard_agent"),
            base_url=kwargs.get("base_url"),
            talishar_assets_path=kwargs.get("talishar_assets_path"),
            max_build_steps=int(kwargs.get("max_build_steps", self._max_build_steps)),
            starting_deck=kwargs.get("starting_deck", self._starting_deck),
            render_mode=render_mode,
        )


FLESH_AND_BLOOD_TALISHAR_V0 = TalisharEngineFactory(
    "FleshAndBlood-Talishar-v0",
    self_play=True,
)
FLESH_AND_BLOOD_TALISHAR_SELFPLAY_V0 = TalisharEngineFactory(
    "FleshAndBlood-Talishar-SelfPlay-v0",
    self_play=True,
)
FLESH_AND_BLOOD_TALISHAR_VS_AI_V0 = TalisharEngineFactory(
    "FleshAndBlood-Talishar-VsAI-v0",
    self_play=False,
)
FLESH_AND_BLOOD_DECKBUILD_TALISHAR_V0 = TalisharDeckBuilderFactory(
    "FleshAndBlood-DeckBuild-Talishar-v0"
)


class TalisharSideboardFactory(rlbridgeEnvironmentFactory):
    """Factory that creates :class:`TalisharSideboardEnvironment` instances.

    Phase 2 of the three-phase FaB pipeline: sideboard selection.  The factory
    must be seeded with the card pool built in Phase 1 before ``create()`` is
    called (pass ``card_pool`` and ``pool_by_id`` as ``create()`` kwargs or
    pre-set them via :meth:`set_pool`).
    """

    def __init__(
        self,
        env_id: str,
        *,
        hero_id: str = "ira_crimson_haze",
        hero_class: str = "Ninja",
        game_format: str = "silver_age",
        opponent_hero_id: str = "dorinthea_ironsong",
        num_eval_games: int = 5,
        max_sideboard_steps: int = 100,
    ) -> None:
        self._env_id = env_id
        self._hero_id = hero_id
        self._hero_class = hero_class
        self._game_format = game_format
        self._opponent_hero_id = opponent_hero_id
        self._num_eval_games = num_eval_games
        self._max_sideboard_steps = max_sideboard_steps
        # Pool set externally after Phase 1 completes
        self._card_pool: dict[str, int] = {}
        self._pool_by_id: dict[str, Any] = {}

    def set_pool(
        self,
        card_pool: dict[str, int],
        pool_by_id: dict[str, Any],
    ) -> None:
        """Provide the card pool built by the deckbuilder (Phase 1 output)."""
        self._card_pool = card_pool
        self._pool_by_id = pool_by_id

    @property
    def env_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            env_id=self._env_id,
            description=(
                "Flesh and Blood TCG sideboard environment (Phase 2 of 3).  "
                "The agent selects which cards from a pre-built pool to include "
                "in the active game deck for a specific opponent matchup.  "
                "Reward is based on win rate vs. the configured opponent.  "
                "Requires a running Talishar Docker instance (set TALISHAR_URL)."
            ),
            tags=[
                "tcg",
                "flesh-and-blood",
                "card-game",
                "sideboard",
                "talishar",
            ],
            namespace="flesh_and_blood",
            render_modes=["ansi"],
            max_episode_steps=self._max_sideboard_steps,
            suggested_hyperparameters=SuggestedHyperparameters(
                agent_type="ppo",
                n_episodes=300,
                max_steps=self._max_sideboard_steps,
                alpha=0.1,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.05,
                epsilon_decay=0.995,
                sub_goal_threshold=0.5,
                top_k=3,
                min_episode_visits=2,
            ),
        )

    def create(
        self,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> rlbridgeEnvironment:
        from .talishar_sideboard_environment import TalisharSideboardEnvironment

        card_pool = kwargs.get("card_pool", self._card_pool)
        pool_by_id = kwargs.get("pool_by_id", self._pool_by_id)
        return TalisharSideboardEnvironment(
            card_pool=card_pool,
            pool_by_id=pool_by_id,
            opponent_hero_id=str(kwargs.get("opponent_hero_id", self._opponent_hero_id)),
            hero_id=str(kwargs.get("hero_id", self._hero_id)),
            game_format=str(kwargs.get("game_format", self._game_format)),
            num_eval_games=int(kwargs.get("num_eval_games", self._num_eval_games)),
            base_url=kwargs.get("base_url"),
            talishar_assets_path=kwargs.get("talishar_assets_path"),
            max_sideboard_steps=int(kwargs.get("max_sideboard_steps", self._max_sideboard_steps)),
            render_mode=render_mode,
        )


FLESH_AND_BLOOD_SIDEBOARD_TALISHAR_V0 = TalisharSideboardFactory(
    "FleshAndBlood-Sideboard-Talishar-v0"
)
ALL_FAB_FACTORIES: list[rlbridgeEnvironmentFactory] = [
    FLESH_AND_BLOOD_TALISHAR_V0,
    FLESH_AND_BLOOD_TALISHAR_SELFPLAY_V0,
    FLESH_AND_BLOOD_TALISHAR_VS_AI_V0,
    FLESH_AND_BLOOD_DECKBUILD_TALISHAR_V0,
    FLESH_AND_BLOOD_SIDEBOARD_TALISHAR_V0,
]
