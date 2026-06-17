"""Four-tier PPO agent cache for dual-agent Talishar training.

Tiers (highest priority first for warm-starting the policy agent):
  1. deck_vs_deck     — player deck + opponent deck
  2. deck_vs_opp_hero — player deck + opponent hero (any opponent list)
  3. hero_vs_hero     — player hero vs opponent hero
  4. hero             — player hero only

During a matchup, tier-1 agents drive actions.  Each PPO update is applied to
all four tier agents for that player.  Weights are persisted under *cache_root*.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from rl_agents.ppo import PPOAgent

TIER_NAMES = ("deck_vs_deck", "deck_vs_opp_hero", "hero_vs_hero", "hero")

# (lr_scale, clip_eps_scale, entropy_scale) by how many tiers below the best cache hit
_SOFT_RESET_BY_GAP: list[tuple[float, float, float]] = [
    (1.0, 1.0, 1.0),
    (0.5, 0.85, 1.15),
    (0.25, 0.7, 1.3),
    (0.1, 0.55, 1.45),
]


def _safe_key(part: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", part)


@dataclass(frozen=True)
class PlayerCacheContext:
    player_deck: str
    player_hero: str
    opponent_deck: str
    opponent_hero: str

    def tier_keys(self) -> tuple[str, str, str, str]:
        return (
            f"{_safe_key(self.player_deck)}__vs__{_safe_key(self.opponent_deck)}",
            f"{_safe_key(self.player_deck)}__vs_hero__{_safe_key(self.opponent_hero)}",
            f"{_safe_key(self.player_hero)}__vs__{_safe_key(self.opponent_hero)}",
            _safe_key(self.player_hero),
        )


@dataclass
class PlayerTierAgents:
    """Four cached agents for one player; index 0 is tier 1 (policy)."""

    context: PlayerCacheContext
    agents: list[PPOAgent]  # length 4, tier 1 .. 4
    init_sources: list[str]  # human-readable init description per tier

    @property
    def policy(self) -> PPOAgent:
        return self.agents[0]


def apply_soft_reset(agent: PPOAgent, specificity_gap: int) -> None:
    """Scale learning hyperparameters after loading a less-specific cache tier."""
    gap = min(max(specificity_gap, 0), len(_SOFT_RESET_BY_GAP) - 1)
    lr_s, clip_s, ent_s = _SOFT_RESET_BY_GAP[gap]
    agent.lr_actor *= lr_s
    agent.lr_critic *= lr_s
    agent.clip_eps = min(0.5, agent.clip_eps * clip_s)
    agent.c_ent *= ent_s


def clone_agent_weights(src: PPOAgent, dst: PPOAgent) -> None:
    if src._actor is None or src.obs_dim <= 0:
        return
    dst.n_actions = src.n_actions
    dst._mask_actions = src._mask_actions
    dst.obs_dim = src.obs_dim
    dst._init_nets(src.obs_dim)
    dst._actor.from_dict(src._actor.to_dict())  # type: ignore[union-attr]
    dst._critic.from_dict(src._critic.to_dict())  # type: ignore[union-attr]


class AgentCacheStore:
    def __init__(self, cache_root: Path, game_format: str) -> None:
        self.cache_root = cache_root / _safe_key(game_format)
        for name in TIER_NAMES:
            (self.cache_root / name).mkdir(parents=True, exist_ok=True)

    def _tier_path(self, tier: int, key: str) -> Path:
        return self.cache_root / TIER_NAMES[tier - 1] / f"{key}.json"

    def load_if_exists(self, tier: int, key: str) -> Optional[PPOAgent]:
        path = self._tier_path(tier, key)
        if not path.is_file():
            return None
        agent = PPOAgent()
        try:
            agent.load(path)
            return agent
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def save(self, tier: int, key: str, agent: PPOAgent) -> None:
        if agent._actor is None:
            return
        path = self._tier_path(tier, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(path)

    def bootstrap_player(
        self,
        ctx: PlayerCacheContext,
        make_agent: Callable[[], PPOAgent],
    ) -> PlayerTierAgents:
        keys = ctx.tier_keys()
        loaded: dict[int, PPOAgent] = {}
        for tier in range(1, 5):
            agent = self.load_if_exists(tier, keys[tier - 1])
            if agent is not None:
                loaded[tier] = agent

        best_tier = min(loaded) if loaded else None
        agents: list[PPOAgent] = []
        init_sources: list[str] = []

        for tier in range(1, 5):
            if tier in loaded:
                agent = loaded[tier]
                gap = abs(tier - best_tier) if best_tier is not None else 0
                apply_soft_reset(agent, gap)
                init_sources.append(f"tier{tier}:cache({TIER_NAMES[tier - 1]},gap={gap})")
            elif tier == 1 or not agents:
                agent = make_agent()
                if best_tier is not None:
                    clone_agent_weights(loaded[best_tier], agent)
                    gap = abs(tier - best_tier)
                    apply_soft_reset(agent, gap)
                    init_sources.append(
                        f"tier{tier}:clone(tier{best_tier},{TIER_NAMES[best_tier - 1]},gap={gap})"
                    )
                else:
                    init_sources.append(f"tier{tier}:fresh")
            else:
                agent = make_agent()
                clone_agent_weights(agents[0], agent)
                apply_soft_reset(agent, tier - 1)
                init_sources.append(f"tier{tier}:clone(policy,gap={tier - 1})")
            agents.append(agent)

        return PlayerTierAgents(context=ctx, agents=agents, init_sources=init_sources)

    def persist_player(self, bundle: PlayerTierAgents) -> None:
        keys = bundle.context.tier_keys()
        for tier, agent in enumerate(bundle.agents, start=1):
            self.save(tier, keys[tier - 1], agent)
