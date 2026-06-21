"""Four-tier PPO agent cache for dual-agent Talishar training.

Tiers (highest priority first for warm-starting the policy agent):
  1. deck_vs_deck     — player deck + opponent deck (content fingerprint)
  2. deck_vs_opp_hero — player deck + opponent hero (any opponent list)
  3. hero_vs_hero     — player hero vs opponent hero
  4. hero             — player hero only

During a matchup, tier-1 agents drive actions.  Each PPO update is applied to
all four tier agents for that player.  Weights are persisted under *cache_root*.

Exact deck-vs-deck matchups are tracked in ``deck_matchup_registry.json`` so
fully trained pairings can skip re-training on subsequent runs.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from rl_agents.ppo import PPOAgent

TIER_NAMES = ("deck_vs_deck", "deck_vs_opp_hero", "hero_vs_hero", "hero")
REGISTRY_FILENAME = "deck_matchup_registry.json"

# (lr_scale, clip_eps_scale, entropy_scale) by how many tiers below the best cache hit
_SOFT_RESET_BY_GAP: list[tuple[float, float, float]] = [
    (1.0, 1.0, 1.0),
    (0.5, 0.85, 1.15),
    (0.25, 0.7, 1.3),
    (0.1, 0.55, 1.45),
]


def _safe_key(part: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", part)


def deck_content_fingerprint(
    deck: Mapping[str, int],
    *,
    equipment_header: str = "",
) -> str:
    """Stable hash for an rlbridge card-pool deck + equipment header."""
    payload = {
        "deck": {str(k): int(deck[k]) for k in sorted(deck)},
        "equipment": equipment_header.strip(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def talishar_asset_deck_fingerprint(assets_path: str, deck_stem: str) -> str:
    """Stable hash for a Talishar ``Assets/<stem>.txt`` deck file."""
    from flesh_and_blood_rlbridge.opponent_deck import normalize_talishar_asset_name

    normalized = normalize_talishar_asset_name(deck_stem, assets_path)
    asset_file = Path(assets_path) / f"{normalized}.txt"
    if asset_file.is_file():
        text = asset_file.read_text(encoding="utf-8", errors="replace").strip()
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def deck_matchup_key(p1_fingerprint: str, p2_fingerprint: str) -> str:
    """Registry / tier-1 key for P1 deck content vs P2 deck content."""
    return f"{p1_fingerprint}__vs__{p2_fingerprint}"


@dataclass(frozen=True)
class MatchupTrainingRecord:
    """Training status for an exact deck-vs-deck pairing (P1 perspective)."""

    matchup_key: str
    p1_deck_fingerprint: str
    p2_deck_fingerprint: str
    p1_hero: str
    p2_hero: str
    converged: bool
    episodes_completed: int
    target_episodes: int
    p1_win_rate: Optional[float] = None
    checkpoint_eval_win_rate: Optional[float] = None
    trained_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatchupTrainingRecord:
        return cls(
            matchup_key=str(data.get("matchup_key", "")),
            p1_deck_fingerprint=str(data.get("p1_deck_fingerprint", "")),
            p2_deck_fingerprint=str(data.get("p2_deck_fingerprint", "")),
            p1_hero=str(data.get("p1_hero", "")),
            p2_hero=str(data.get("p2_hero", "")),
            converged=bool(data.get("converged", False)),
            episodes_completed=int(data.get("episodes_completed", 0)),
            target_episodes=int(data.get("target_episodes", 0)),
            p1_win_rate=(
                float(data["p1_win_rate"])
                if data.get("p1_win_rate") is not None
                else None
            ),
            checkpoint_eval_win_rate=(
                float(data["checkpoint_eval_win_rate"])
                if data.get("checkpoint_eval_win_rate") is not None
                else None
            ),
            trained_at=str(data.get("trained_at", "")),
        )


class MatchupConvergenceRegistry:
    """Persistent hashmap of exact deck-vs-deck training outcomes."""

    def __init__(self, cache_root: Path, game_format: str) -> None:
        self._path = cache_root / _safe_key(game_format) / REGISTRY_FILENAME
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(raw, dict):
            self._entries = raw

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._entries, indent=2), encoding="utf-8")

    def lookup(self, matchup_key: str) -> Optional[MatchupTrainingRecord]:
        row = self._entries.get(matchup_key)
        if not row:
            return None
        return MatchupTrainingRecord.from_dict(row)

    def is_converged(
        self,
        matchup_key: str,
        *,
        min_episodes: int,
    ) -> bool:
        record = self.lookup(matchup_key)
        if record is None or not record.converged:
            return False
        return record.episodes_completed >= min_episodes

    def record_training(
        self,
        record: MatchupTrainingRecord,
        *,
        converged: bool,
    ) -> None:
        payload = asdict(record)
        payload["converged"] = converged
        if not payload.get("trained_at"):
            payload["trained_at"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._entries[record.matchup_key] = payload
            self._save()


@dataclass(frozen=True)
class PlayerCacheContext:
    player_deck: str
    player_hero: str
    opponent_deck: str
    opponent_hero: str
    player_deck_fingerprint: Optional[str] = None
    opponent_deck_fingerprint: Optional[str] = None

    def tier_keys(self) -> tuple[str, str, str, str]:
        p1_fp = self.player_deck_fingerprint
        p2_fp = self.opponent_deck_fingerprint
        if p1_fp and p2_fp:
            deck_vs_deck = deck_matchup_key(p1_fp, p2_fp)
        else:
            deck_vs_deck = (
                f"{_safe_key(self.player_deck)}__vs__{_safe_key(self.opponent_deck)}"
            )

        if p1_fp:
            deck_vs_opp_hero = (
                f"{p1_fp}__vs_hero__{_safe_key(self.opponent_hero)}"
            )
        else:
            deck_vs_opp_hero = (
                f"{_safe_key(self.player_deck)}__vs_hero__{_safe_key(self.opponent_hero)}"
            )

        return (
            deck_vs_deck,
            deck_vs_opp_hero,
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
        self.user_cache_root = cache_root
        self.cache_root = cache_root / _safe_key(game_format)
        self.game_format = game_format
        for name in TIER_NAMES:
            (self.cache_root / name).mkdir(parents=True, exist_ok=True)
        self.matchup_registry = MatchupConvergenceRegistry(cache_root, game_format)

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

    def has_tier1_agent(self, ctx: PlayerCacheContext) -> bool:
        tier1_key = ctx.tier_keys()[0]
        return self.load_if_exists(1, tier1_key) is not None

    def should_skip_training(
        self,
        *,
        p1_fingerprint: str,
        p2_fingerprint: str,
        target_episodes: int,
        require_p2_agent: bool = False,
        p2_context: Optional[PlayerCacheContext] = None,
    ) -> Optional[MatchupTrainingRecord]:
        """Return cached record when this exact matchup need not be re-trained."""
        matchup_key = deck_matchup_key(p1_fingerprint, p2_fingerprint)
        record = self.matchup_registry.lookup(matchup_key)
        if record is None or not self.matchup_registry.is_converged(
            matchup_key,
            min_episodes=target_episodes,
        ):
            return None

        p1_ctx = PlayerCacheContext(
            player_deck="cached",
            player_hero="",
            opponent_deck="cached",
            opponent_hero="",
            player_deck_fingerprint=p1_fingerprint,
            opponent_deck_fingerprint=p2_fingerprint,
        )
        if not self.has_tier1_agent(p1_ctx):
            return None

        if require_p2_agent:
            if p2_context is None:
                p2_context = PlayerCacheContext(
                    player_deck="cached",
                    player_hero="",
                    opponent_deck="cached",
                    opponent_hero="",
                    player_deck_fingerprint=p2_fingerprint,
                    opponent_deck_fingerprint=p1_fingerprint,
                )
            if not self.has_tier1_agent(p2_context):
                return None

        return record

    def mark_matchup_converged(
        self,
        *,
        p1_fingerprint: str,
        p2_fingerprint: str,
        p1_hero: str,
        p2_hero: str,
        episodes_completed: int,
        target_episodes: int,
        p1_win_rate: Optional[float] = None,
        checkpoint_eval_win_rate: Optional[float] = None,
    ) -> None:
        matchup_key = deck_matchup_key(p1_fingerprint, p2_fingerprint)
        record = MatchupTrainingRecord(
            matchup_key=matchup_key,
            p1_deck_fingerprint=p1_fingerprint,
            p2_deck_fingerprint=p2_fingerprint,
            p1_hero=p1_hero,
            p2_hero=p2_hero,
            converged=episodes_completed >= target_episodes,
            episodes_completed=episodes_completed,
            target_episodes=target_episodes,
            p1_win_rate=p1_win_rate,
            checkpoint_eval_win_rate=checkpoint_eval_win_rate,
            trained_at=datetime.now(timezone.utc).isoformat(),
        )
        self.matchup_registry.record_training(
            record,
            converged=record.converged,
        )

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
