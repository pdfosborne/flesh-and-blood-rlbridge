from __future__ import annotations

import random
from typing import Any, Optional

from .environment import FleshAndBloodEnvironment
from .gameplay_environment import FleshAndBloodGameplayEnvironment


class FleshAndBloodDeckBuilderEnvironment(FleshAndBloodEnvironment):
    """Deck-builder environment that scores deck quality via gameplay rollouts."""

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        agent_hero_id: str = "hero_dorinthea_ironsong",
        opponent_hero_id: str = "hero_rhinar_reckless_rampage",
        max_turns: int = 60,
        deck_size: int = 36,
        agent_deck_style: str = "balanced",
        opponent_deck_style: str = "balanced",
        format: str = "classic_constructed",
        self_play: bool = False,
        opponent_type: str = "preset_logic",
        render_mode: Optional[str] = None,
        quality_eval_episodes: int = 1,
        quality_max_steps: int = 60,
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
            two_phase_deckbuild=True,
            self_play=self_play,
            opponent_type=opponent_type,
            render_mode=render_mode,
        )
        self._quality_eval_episodes = max(1, int(quality_eval_episodes))
        self._quality_max_steps = max(1, int(quality_max_steps))
        self._deck_quality_cache: dict[str, float] = {}
        self._deck_quality_rng = random.Random(seed)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Any:
        self._deck_quality_cache = {}
        opts = dict(options or {})
        opts["two_phase_deckbuild"] = True
        if "quality_eval_episodes" in opts:
            self._quality_eval_episodes = max(1, int(opts["quality_eval_episodes"]))
        if "quality_max_steps" in opts:
            self._quality_max_steps = max(1, int(opts["quality_max_steps"]))
        return super().reset(seed=seed, options=opts)

    def _deck_options_for_format(self) -> list[dict[str, Any]]:
        options = super()._deck_options_for_format()
        scored: list[dict[str, Any]] = []
        for option in options:
            item = dict(option)
            item["estimated_quality"] = round(self._estimate_deck_quality(item), 4)
            scored.append(item)
        scored.sort(key=lambda o: float(o.get("estimated_quality", 0.0)), reverse=True)
        return scored

    def _estimate_deck_quality(self, deck_option: dict[str, Any]) -> float:
        cache_key = f"{self._format}:{deck_option.get('key', '')}"
        cached = self._deck_quality_cache.get(cache_key)
        if cached is not None:
            return cached

        probe = FleshAndBloodGameplayEnvironment(
            seed=self._deck_quality_rng.randint(0, 10_000_000),
            format=self._format,
            deck_size=int(deck_option.get("deck_size", self._deck_size) or self._deck_size),
            max_turns=self._quality_max_steps,
            opponent_type=self._opponent_type,
            render_mode=None,
        )
        try:
            all_options = probe._deck_options_for_format()
        finally:
            probe.close()

        opponents = [o for o in all_options if str(o.get("hero_id")) != str(deck_option.get("hero_id"))]
        if not opponents:
            opponents = list(all_options)
        if not opponents:
            self._deck_quality_cache[cache_key] = 0.5
            return 0.5

        total = 0.0
        for ep in range(self._quality_eval_episodes):
            opponent_option = dict(self._deck_quality_rng.choice(opponents))
            total += self._run_quality_episode(deck_option=deck_option, opponent_option=opponent_option, ep=ep)

        score = total / float(self._quality_eval_episodes)
        self._deck_quality_cache[cache_key] = score
        return score

    def _run_quality_episode(
        self,
        *,
        deck_option: dict[str, Any],
        opponent_option: dict[str, Any],
        ep: int,
    ) -> float:
        env = FleshAndBloodGameplayEnvironment(
            seed=self._deck_quality_rng.randint(0, 10_000_000),
            agent_hero_id=str(deck_option.get("hero_id", self._agent_hero_id)),
            opponent_hero_id=str(opponent_option.get("hero_id", self._opponent_hero_id)),
            max_turns=self._quality_max_steps,
            deck_size=int(deck_option.get("deck_size", self._deck_size) or self._deck_size),
            agent_deck_style=str(deck_option.get("style", "balanced")),
            opponent_deck_style=str(opponent_option.get("style", "balanced")),
            format=self._format,
            self_play=False,
            opponent_type=self._opponent_type,
            render_mode=None,
        )
        try:
            seed = self._deck_quality_rng.randint(0, 10_000_000) + ep
            env.reset(seed=seed)

            agent_card_ids = deck_option.get("_card_ids")
            opponent_card_ids = opponent_option.get("_card_ids")
            if isinstance(agent_card_ids, list) or isinstance(opponent_card_ids, list):
                env._start_match(  # noqa: SLF001
                    agent_hero_id=str(deck_option.get("hero_id", self._agent_hero_id)),
                    opponent_hero_id=str(opponent_option.get("hero_id", self._opponent_hero_id)),
                    agent_deck_style=str(deck_option.get("style", "balanced")),
                    opponent_deck_style=str(opponent_option.get("style", "balanced")),
                    agent_deck_ids=agent_card_ids if isinstance(agent_card_ids, list) else None,
                    opponent_deck_ids=opponent_card_ids if isinstance(opponent_card_ids, list) else None,
                )

            for _ in range(self._quality_max_steps):
                action = env.sample_action()
                out = env.step(action)
                if out.terminated or out.truncated:
                    break

            if not env._players:  # noqa: SLF001
                return 0.5
            agent_life = env._players[0].life  # noqa: SLF001
            opp_life = env._players[1].life  # noqa: SLF001
            if opp_life <= 0 < agent_life:
                return 1.0
            if agent_life <= 0 < opp_life:
                return 0.0
            return 0.5
        finally:
            env.close()
