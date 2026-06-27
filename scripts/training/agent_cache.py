"""Unified PPO agent cache — one shared policy per (game_format, weight_version).

Weights: ``{cache_root}/{format}/unified_agent_v{weight_version}.json``
Metadata: ``{cache_root}/{format}/unified_agent.meta.json`` including ``training_history``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from rl_agents.ppo import ARCHITECTURE, PPOAgent, UNIFIED_AGENT_WEIGHT_VERSION

UNIFIED_META_FILENAME = "unified_agent.meta.json"
LEGACY_REGISTRY_FILENAME = "deck_matchup_registry.json"


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
    return f"{p1_fingerprint}__vs__{p2_fingerprint}"


@dataclass(frozen=True)
class MatchupTrainingRecord:
    """One converged (or in-progress) deck-vs-deck training session."""

    matchup_key: str
    p1_deck_fingerprint: str
    p2_deck_fingerprint: str
    p1_hero: str
    p2_hero: str
    converged: bool
    episodes_completed: int
    target_episodes: int
    p1_win_rate: Optional[float] = None
    p2_win_rate: Optional[float] = None
    checkpoint_eval_win_rate: Optional[float] = None
    first_checkpoint_win_rate: Optional[float] = None
    final_checkpoint_win_rate: Optional[float] = None
    trained_at: str = ""
    matchup_name: str = ""
    training_stats: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatchupTrainingRecord:
        stats = data.get("training_stats")
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
            p2_win_rate=(
                float(data["p2_win_rate"])
                if data.get("p2_win_rate") is not None
                else None
            ),
            checkpoint_eval_win_rate=(
                float(data["checkpoint_eval_win_rate"])
                if data.get("checkpoint_eval_win_rate") is not None
                else None
            ),
            first_checkpoint_win_rate=(
                float(data["first_checkpoint_win_rate"])
                if data.get("first_checkpoint_win_rate") is not None
                else None
            ),
            final_checkpoint_win_rate=(
                float(data["final_checkpoint_win_rate"])
                if data.get("final_checkpoint_win_rate") is not None
                else None
            ),
            trained_at=str(data.get("trained_at", "")),
            matchup_name=str(data.get("matchup_name", "")),
            training_stats=stats if isinstance(stats, dict) else None,
        )


@dataclass
class UnifiedPolicyBundle:
    """Single shared policy used for both seats in self-play."""

    policy: PPOAgent
    init_sources: list[str]
    _shared_tiers: list[PPOAgent] | None = None

    def shared_tiers(self) -> list[PPOAgent]:
        """Return one cached tier list — use for both P1 and P2 in self-play."""
        if self._shared_tiers is None:
            self._shared_tiers = [self.policy]
        return self._shared_tiers

    @property
    def agents(self) -> list[PPOAgent]:
        return self.shared_tiers()


def clone_agent_weights(src: PPOAgent, dst: PPOAgent) -> None:
    if src._shared is None or src.obs_dim <= 0:
        raise RuntimeError(
            "Cannot clone agent weights: source policy has no initialized networks "
            f"(obs_dim={getattr(src, 'obs_dim', 0)!r})"
        )
    dst.n_actions = src.n_actions
    dst._mask_actions = src._mask_actions
    dst.obs_dim = src.obs_dim
    dst.hidden_size = src.hidden_size
    dst.n_layers = src.n_layers
    dst.n_heads = src.n_heads
    dst._init_nets(src.obs_dim)
    assert src._shared is not None and dst._shared is not None
    dst._shared.load_state_dict(src._shared.state_dict())


class AgentCacheStore:
    """Single unified agent cache per (game_format, obs_schema_version)."""

    def __init__(
        self,
        cache_root: Path,
        game_format: str,
        *,
        obs_schema_version: int = 2,
    ) -> None:
        self.user_cache_root = cache_root
        self.cache_root = cache_root / _safe_key(game_format)
        self.game_format = game_format
        self.obs_schema_version = int(obs_schema_version)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _weights_path(self) -> Path:
        return self.cache_root / f"unified_agent_v{UNIFIED_AGENT_WEIGHT_VERSION}.json"

    def _meta_path(self) -> Path:
        return self.cache_root / UNIFIED_META_FILENAME

    def _legacy_registry_path(self) -> Path:
        return self.cache_root / LEGACY_REGISTRY_FILENAME

    def load_meta(self) -> dict[str, Any]:
        path = self._meta_path()
        if not path.is_file():
            meta: dict[str, Any] = {}
            self._migrate_legacy_registry(meta)
            return meta
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        meta = raw if isinstance(raw, dict) else {}
        self._migrate_legacy_registry(meta)
        return meta

    def save_meta(self, meta: dict[str, Any]) -> None:
        history = meta.get("training_history")
        if not isinstance(history, list):
            meta["training_history"] = []
        self._meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _migrate_legacy_registry(self, meta: dict[str, Any]) -> None:
        """Import old deck_matchup_registry.json into training_history once."""
        if meta.get("training_history"):
            return
        legacy_path = self._legacy_registry_path()
        if not legacy_path.is_file():
            return
        try:
            raw = json.loads(legacy_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(raw, dict):
            return
        history: list[dict[str, Any]] = []
        for row in raw.values():
            if isinstance(row, dict):
                history.append(dict(row))
        if history:
            meta["training_history"] = history
            self.save_meta(meta)

    def _training_history(self, meta: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        if meta is None:
            meta = self.load_meta()
        history = meta.get("training_history", [])
        return list(history) if isinstance(history, list) else []

    def _lookup_history(
        self,
        matchup_key: str,
        *,
        meta: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        for row in reversed(self._training_history(meta)):
            if str(row.get("matchup_key", "")) == matchup_key:
                return row
        return None

    def load_if_exists(self) -> Optional[PPOAgent]:
        path = self._weights_path()
        if not path.is_file():
            return None
        agent = PPOAgent()
        try:
            agent.load(path)
            return agent
        except (json.JSONDecodeError, KeyError, OSError, ValueError):
            return None

    def has_agent(self) -> bool:
        return self.load_if_exists() is not None

    def load_required(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        mask_actions: bool = True,
    ) -> PPOAgent:
        """Load unified policy weights from cache or raise if missing/incompatible."""
        path = self._weights_path()
        agent = self.load_if_exists()
        meta = self.load_meta()
        if agent is None or agent._actor is None:
            raise FileNotFoundError(
                f"Unified agent weights not found at {path} "
                f"(format={self.game_format}, schema=v{self.obs_schema_version})"
            )
        schema = int(meta.get("obs_schema_version", -1))
        if schema not in (-1, self.obs_schema_version):
            raise RuntimeError(
                f"Unified agent schema mismatch at {path}: "
                f"cache v{schema}, expected v{self.obs_schema_version}"
            )
        if agent.obs_dim != obs_dim:
            cached = int(meta.get("obs_dim", agent.obs_dim))
            raise RuntimeError(
                f"Unified agent obs_dim mismatch at {path}: "
                f"loaded={agent.obs_dim}, meta={cached}, expected={obs_dim}"
            )
        agent.n_actions = n_actions
        agent._mask_actions = mask_actions
        return agent

    def should_skip_training(
        self,
        *,
        p1_fingerprint: str,
        p2_fingerprint: str,
        target_episodes: int,
    ) -> Optional[MatchupTrainingRecord]:
        if not self.has_agent():
            return None
        key = deck_matchup_key(p1_fingerprint, p2_fingerprint)
        row = self._lookup_history(key)
        if row is None:
            return None
        record = MatchupTrainingRecord.from_dict(row)
        if not record.converged:
            return None
        if record.episodes_completed < target_episodes:
            return None
        return record

    def load_or_create(
        self,
        make_agent: Callable[[], PPOAgent],
        *,
        obs_dim: int,
        n_actions: int,
        mask_actions: bool,
    ) -> tuple[PPOAgent, str]:
        agent = self.load_if_exists()
        meta = self.load_meta()
        if agent is not None and agent._actor is not None:
            if int(meta.get("obs_schema_version", -1)) not in (-1, self.obs_schema_version):
                agent = None
            elif int(meta.get("obs_dim", agent.obs_dim)) not in (0, obs_dim):
                agent = None
        if agent is not None:
            agent.n_actions = n_actions
            agent._mask_actions = mask_actions
            if agent.obs_dim != obs_dim:
                agent.obs_dim = obs_dim
                agent._init_nets(obs_dim)
            return agent, "cache"

        agent = make_agent()
        agent.n_actions = n_actions
        agent._mask_actions = mask_actions
        agent.obs_dim = obs_dim
        agent._init_nets(obs_dim)
        return agent, "fresh"

    def persist(
        self,
        agent: PPOAgent,
        *,
        episodes_delta: int = 0,
        training_summary: Optional[dict[str, Any]] = None,
    ) -> None:
        """Save weights and update meta, optionally appending a training_history entry."""
        if agent._actor is None:
            return
        path = self._weights_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(path)

        meta = self.load_meta()
        meta.update(
            {
                "obs_schema_version": self.obs_schema_version,
                "weight_version": UNIFIED_AGENT_WEIGHT_VERSION,
                "architecture": ARCHITECTURE,
                "obs_dim": agent.obs_dim,
                "n_actions": agent.n_actions,
                "d_model": agent.hidden_size,
                "n_layers": agent.n_layers,
                "n_heads": agent.n_heads,
                "game_format": self.game_format,
                "total_episodes_trained": int(meta.get("total_episodes_trained", 0))
                + int(episodes_delta),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
        )
        if training_summary:
            self._append_training_history(meta, training_summary)
            meta["last_matchup"] = str(
                training_summary.get("matchup_name")
                or meta.get("last_matchup", "")
            )
        self.save_meta(meta)

    def _append_training_history(
        self,
        meta: dict[str, Any],
        summary: dict[str, Any],
    ) -> None:
        p1_fp = str(summary.get("p1_fingerprint", "") or summary.get("p1_deck_fingerprint", ""))
        p2_fp = str(summary.get("p2_fingerprint", "") or summary.get("p2_deck_fingerprint", ""))
        if not p1_fp or not p2_fp:
            return

        episodes_completed = int(summary.get("episodes_completed", 0))
        target_episodes = int(summary.get("target_episodes", episodes_completed))
        entry: dict[str, Any] = {
            "matchup_key": deck_matchup_key(p1_fp, p2_fp),
            "matchup_name": str(summary.get("matchup_name", "")),
            "p1_deck_fingerprint": p1_fp,
            "p2_deck_fingerprint": p2_fp,
            "p1_hero": str(summary.get("p1_hero", "")),
            "p2_hero": str(summary.get("p2_hero", "")),
            "converged": bool(summary.get("converged", episodes_completed >= target_episodes)),
            "episodes_completed": episodes_completed,
            "target_episodes": target_episodes,
            "trained_at": str(
                summary.get("trained_at")
                or datetime.now(timezone.utc).isoformat()
            ),
        }
        for key in (
            "p1_win_rate",
            "p2_win_rate",
            "checkpoint_eval_win_rate",
            "first_checkpoint_win_rate",
            "final_checkpoint_win_rate",
        ):
            if summary.get(key) is not None:
                entry[key] = summary[key]
        final_ckpt = summary.get("final_checkpoint_win_rate")
        if final_ckpt is not None and entry.get("checkpoint_eval_win_rate") is None:
            entry["checkpoint_eval_win_rate"] = final_ckpt
        stats = summary.get("training_stats")
        if isinstance(stats, dict) and stats:
            entry["training_stats"] = stats

        history = self._training_history(meta)
        key = entry["matchup_key"]
        replaced = False
        for idx, row in enumerate(history):
            if str(row.get("matchup_key", "")) == key:
                history[idx] = entry
                replaced = True
                break
        if not replaced:
            history.append(entry)
        meta["training_history"] = history

    def record_matchup_training(
        self,
        *,
        p1_fingerprint: str,
        p2_fingerprint: str,
        p1_hero: str,
        p2_hero: str,
        episodes_completed: int,
        target_episodes: int,
        p1_win_rate: Optional[float] = None,
        p2_win_rate: Optional[float] = None,
        checkpoint_eval_win_rate: Optional[float] = None,
        matchup_name: str = "",
        training_stats: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append or update a training_history entry without saving weights."""
        meta = self.load_meta()
        self._append_training_history(
            meta,
            {
                "p1_fingerprint": p1_fingerprint,
                "p2_fingerprint": p2_fingerprint,
                "p1_hero": p1_hero,
                "p2_hero": p2_hero,
                "episodes_completed": episodes_completed,
                "target_episodes": target_episodes,
                "p1_win_rate": p1_win_rate,
                "p2_win_rate": p2_win_rate,
                "checkpoint_eval_win_rate": checkpoint_eval_win_rate,
                "matchup_name": matchup_name,
                "training_stats": training_stats,
            },
        )
        if matchup_name:
            meta["last_matchup"] = matchup_name
        meta["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.save_meta(meta)

    def training_history(self) -> list[MatchupTrainingRecord]:
        return [MatchupTrainingRecord.from_dict(row) for row in self._training_history()]


# Backward-compatible alias
UnifiedAgentStore = AgentCacheStore
