"""Persistent episode experience cache for Talishar dual-agent training.

Stores complete (terminated, non-truncated) episodes to disk, keyed by the
exact deck matchup played.  Cached transitions are replayed as PPO warm-start
data on subsequent training runs, and if sufficient episodes exist the
default-policy warmup phase is skipped entirely.

Storage layout
--------------
``<cache_root>/episodes/<game_format>/<p1_deck>__vs__<p2_deck>/episodes.pkl.gz``

Each file holds a list of episode dicts::

    {
        "obs_dim":        int,           # obs vector length (compatibility guard)
        "p1_transitions": [...],         # list of transition dicts (see below)
        "p2_transitions": [...],
        "p1_reward":      float,
        "p2_reward":      float,
        "steps":          int,
    }

Transition dict keys: ``obs_vec`` (list[float]), ``action`` (int),
``reward`` (float), ``value`` (float), ``log_prob`` (float),
``done`` (float), ``n_legal`` (int), ``next_obs_vec`` (list[float]).

Thread safety
-------------
Reads are unsynchronised (safe for concurrent readers).  Appends use a per-key
``threading.Lock`` so parallel workers can safely call ``add_episode``
simultaneously for the same matchup.
"""

from __future__ import annotations

import gzip
import pickle
import re
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ── constants ────────────────────────────────────────────────────────────────

# Default maximum stored episodes per matchup. Oldest episodes are evicted
# when the limit is exceeded so the cache stays bounded on disk.
DEFAULT_MAX_EPISODES: int = 500

# Minimum cached episodes required before the default-policy warmup is skipped.
DEFAULT_WARMUP_SKIP_THRESHOLD: int = 50


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_key(part: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", part)


def _matchup_key(p1_deck: str, p2_deck: str) -> str:
    return f"{_safe_key(p1_deck)}__vs__{_safe_key(p2_deck)}"


def _trans_to_serialisable(t: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy arrays inside a transition dict to plain Python lists."""
    out = dict(t)
    for k in ("obs_vec", "next_obs_vec"):
        v = out.get(k)
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        # already a list → leave as-is
    return out


def _trans_to_numpy(t: dict[str, Any]) -> dict[str, Any]:
    """Restore numpy arrays from a deserialised transition dict."""
    out = dict(t)
    for k in ("obs_vec", "next_obs_vec"):
        v = out.get(k)
        if isinstance(v, list):
            out[k] = np.array(v, dtype=np.float64)
    return out


# ── main class ───────────────────────────────────────────────────────────────

class EpisodeCache:
    """Persistent, bounded store of complete game episodes per deck matchup.

    Parameters
    ----------
    cache_root:
        Root directory for episode storage (``results/agent_cache``).
    game_format:
        Talishar game format string (e.g. ``"silver_age"``).
    max_episodes_per_matchup:
        Maximum number of episodes to retain per matchup key.
        Oldest episodes are dropped when this limit is exceeded.
    warmup_skip_threshold:
        If at least this many compatible episodes exist for a matchup, the
        caller may skip default-policy warmup entirely.
    """

    def __init__(
        self,
        cache_root: Path,
        game_format: str,
        max_episodes_per_matchup: int = DEFAULT_MAX_EPISODES,
        warmup_skip_threshold: int = DEFAULT_WARMUP_SKIP_THRESHOLD,
    ) -> None:
        self._root = cache_root / "episodes" / _safe_key(game_format)
        self._root.mkdir(parents=True, exist_ok=True)
        self.max_episodes = max_episodes_per_matchup
        self.warmup_skip_threshold = warmup_skip_threshold
        self._locks: dict[str, threading.Lock] = {}
        self._lock_map_lock = threading.Lock()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _cache_path(self, p1_deck: str, p2_deck: str) -> Path:
        key = _matchup_key(p1_deck, p2_deck)
        return self._root / key / "episodes.pkl.gz"

    def _key_lock(self, key: str) -> threading.Lock:
        with self._lock_map_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def _load_raw(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        try:
            with gzip.open(path, "rb") as fh:
                return pickle.load(fh)  # noqa: S301
        except Exception:
            return []

    def _save_raw(self, path: Path, episodes: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pkl.gz.tmp")
        with gzip.open(tmp, "wb", compresslevel=6) as fh:
            pickle.dump(episodes, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    # ── public API ────────────────────────────────────────────────────────────

    def count(self, p1_deck: str, p2_deck: str, *, obs_dim: Optional[int] = None) -> int:
        """Return the number of cached episodes for this matchup.

        If *obs_dim* is given, only episodes whose stored obs_dim matches are
        counted (incompatible episodes are ignored during warm-start).
        """
        path = self._cache_path(p1_deck, p2_deck)
        episodes = self._load_raw(path)
        if obs_dim is None:
            return len(episodes)
        return sum(1 for ep in episodes if ep.get("obs_dim") == obs_dim)

    def should_skip_warmup(
        self,
        p1_deck: str,
        p2_deck: str,
        *,
        obs_dim: Optional[int] = None,
    ) -> bool:
        """Return True if the cache has enough episodes to skip warmup."""
        return self.count(p1_deck, p2_deck, obs_dim=obs_dim) >= self.warmup_skip_threshold

    def add_episode(
        self,
        p1_deck: str,
        p2_deck: str,
        *,
        obs_dim: int,
        p1_transitions: list[dict[str, Any]],
        p2_transitions: list[dict[str, Any]],
        p1_reward: float,
        p2_reward: float,
        steps: int,
    ) -> None:
        """Append one completed episode to the cache (thread-safe).

        *transitions* lists are serialised by converting numpy arrays to plain
        Python lists so ``pickle`` storage stays portable.
        """
        key = _matchup_key(p1_deck, p2_deck)
        path = self._cache_path(p1_deck, p2_deck)
        episode: dict[str, Any] = {
            "obs_dim": obs_dim,
            "p1_transitions": [_trans_to_serialisable(t) for t in p1_transitions],
            "p2_transitions": [_trans_to_serialisable(t) for t in p2_transitions],
            "p1_reward": float(p1_reward),
            "p2_reward": float(p2_reward),
            "steps": int(steps),
        }
        with self._key_lock(key):
            episodes = self._load_raw(path)
            episodes.append(episode)
            # Evict oldest if over limit
            if len(episodes) > self.max_episodes:
                episodes = episodes[-self.max_episodes :]
            self._save_raw(path, episodes)

    def load_episodes(
        self,
        p1_deck: str,
        p2_deck: str,
        *,
        obs_dim: int,
        max_load: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Load compatible episodes, restoring numpy arrays.

        Episodes whose ``obs_dim`` does not match *obs_dim* are silently
        skipped (stale entries from a different feature encoding).

        Parameters
        ----------
        max_load:
            If set, return at most this many episodes (the most recent ones).
        """
        path = self._cache_path(p1_deck, p2_deck)
        raw = self._load_raw(path)
        compatible = [ep for ep in raw if ep.get("obs_dim") == obs_dim]
        if max_load is not None:
            compatible = compatible[-max_load:]
        # Restore numpy arrays in all transitions
        result: list[dict[str, Any]] = []
        for ep in compatible:
            result.append({
                "obs_dim":        ep["obs_dim"],
                "p1_transitions": [_trans_to_numpy(t) for t in ep["p1_transitions"]],
                "p2_transitions": [_trans_to_numpy(t) for t in ep["p2_transitions"]],
                "p1_reward":      ep["p1_reward"],
                "p2_reward":      ep["p2_reward"],
                "steps":          ep["steps"],
            })
        return result

    def info(self, p1_deck: str, p2_deck: str) -> dict[str, Any]:
        """Return a summary dict for logging (episode count, obs_dims seen, etc.)."""
        path = self._cache_path(p1_deck, p2_deck)
        raw = self._load_raw(path)
        if not raw:
            return {"total_episodes": 0, "obs_dims": [], "path": str(path)}
        dims: set[int] = {ep.get("obs_dim", -1) for ep in raw}
        total_steps = sum(ep.get("steps", 0) for ep in raw)
        return {
            "total_episodes": len(raw),
            "obs_dims": sorted(dims),
            "total_steps": total_steps,
            "path": str(path),
        }
