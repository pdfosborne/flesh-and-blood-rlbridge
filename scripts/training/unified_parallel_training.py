"""Parallel unified matchup batch training — shared PPO experience buffer."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from parallel_seed_training import workers_per_parallel_seed
from runtime_defaults import DEFAULT_PPO_ROLLOUT_BATCH

if TYPE_CHECKING:
    from agent_cache import AgentCacheStore
    from rl_agents.ppo import PPOAgent
    from train_dual_agent_common import Matchup


class UnifiedSharedExperienceBuffer:
    """Thread-safe transition accumulator for cross-matchup unified PPO training."""

    def __init__(self, *, rollout_batch: int = DEFAULT_PPO_ROLLOUT_BATCH) -> None:
        self._lock = threading.Lock()
        self._rollout_batch = max(1, int(rollout_batch))
        self._ppo_accum: list[dict[str, Any]] = []
        self._warmup_accum: list[dict[str, Any]] = []
        self._policy: Optional[PPOAgent] = None

    def extend_ppo(self, transitions: list[dict[str, Any]]) -> None:
        if not transitions:
            return
        with self._lock:
            self._ppo_accum.extend(transitions)

    def extend_warmup(self, transitions: list[dict[str, Any]]) -> None:
        if not transitions:
            return
        with self._lock:
            self._warmup_accum.extend(transitions)

    def maybe_flush_ppo(
        self,
        policy: "PPOAgent",
        *,
        force: bool = False,
    ) -> None:
        with self._lock:
            if not self._ppo_accum:
                return
            if len(self._ppo_accum) < self._rollout_batch and not force:
                return
            self._flush_ppo_locked(policy)

    def flush_warmup(
        self,
        policy: "PPOAgent",
        *,
        force: bool = False,
    ) -> None:
        with self._lock:
            if not self._warmup_accum:
                return
            if not force:
                return
            self._flush_warmup_locked(policy)

    def flush_remaining(self, policy: "PPOAgent") -> None:
        with self._lock:
            if self._warmup_accum:
                self._flush_warmup_locked(policy)
            if self._ppo_accum:
                self._flush_ppo_locked(policy)

    def clone_policy_snapshot(self) -> "PPOAgent":
        from agent_cache import clone_agent_weights  # noqa: PLC0415
        from rl_agents.ppo import PPOAgent  # noqa: PLC0415

        with self._lock:
            src = self._policy
            if src is None or src._shared is None:
                raise RuntimeError(
                    "Cannot clone policy snapshot: shared buffer has no policy reference"
                )
            dst = PPOAgent()
            clone_agent_weights(src, dst)
            return dst

    def bind_policy(self, policy: "PPOAgent") -> None:
        with self._lock:
            self._policy = policy

    def _flush_ppo_locked(self, policy: "PPOAgent") -> None:
        from train_dual_agent_common import _flush_unified_merged_transitions  # noqa: PLC0415

        batch = self._ppo_accum
        self._ppo_accum = []
        _flush_unified_merged_transitions(policy, batch)

    def _flush_warmup_locked(self, policy: "PPOAgent") -> None:
        from train_dual_agent_common import (  # noqa: PLC0415
            _flush_unified_merged_warmup_transitions,
        )

        batch = self._warmup_accum
        self._warmup_accum = []
        _flush_unified_merged_warmup_transitions(policy, batch)


def run_parallel_matchup_batch(
    matchups: list["Matchup"],
    *,
    train_fn: Callable[["Matchup"], dict[str, Any]],
    label: str = "parallel matchup batch",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run *matchups* concurrently via *train_fn*; return meta rows and failures."""
    if not matchups:
        return [], []

    rows: list[dict[str, Any]] = []
    failed: list[str] = []
    n_workers = len(matchups)
    print(
        f"\n  [{label}] starting {n_workers} matchup(s) in parallel…",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(train_fn, matchup): matchup for matchup in matchups}
        for fut in as_completed(futures):
            matchup = futures[fut]
            try:
                meta = fut.result()
                rows.append(meta)
            except Exception as exc:
                print(f"\n  ERROR training {matchup.name}: {exc}", flush=True)
                failed.append(matchup.name)

    print(
        f"  [{label}] finished — "
        f"{len(rows)} ok, {len(failed)} failed",
        flush=True,
    )
    return rows, failed


def workers_per_parallel_matchup(
    total_workers: int,
    parallel_matchups: int,
) -> int:
    """Split rollout worker budget across concurrent matchups."""
    return workers_per_parallel_seed(total_workers, parallel_matchups)


def persist_batch_to_cache(
    cache_store: "AgentCacheStore",
    policy: "PPOAgent",
    persist_payloads: list[dict[str, Any]],
) -> None:
    """Save unified weights once and record per-matchup training history."""
    if not persist_payloads:
        return

    total_episodes = sum(int(p.get("episodes_completed", 0) or 0) for p in persist_payloads)
    cache_store.persist(
        policy,
        episodes_delta=total_episodes,
        training_summary=None,
    )
    for payload in persist_payloads:
        if payload.get("skipped_training"):
            continue
        cache_store.record_matchup_training(
            p1_fingerprint=str(payload.get("p1_fingerprint", "")),
            p2_fingerprint=str(payload.get("p2_fingerprint", "")),
            p1_hero=str(payload.get("p1_hero", "")),
            p2_hero=str(payload.get("p2_hero", "")),
            episodes_completed=int(payload.get("episodes_completed", 0) or 0),
            target_episodes=int(payload.get("target_episodes", 0) or 0),
            p1_win_rate=payload.get("p1_win_rate"),
            p2_win_rate=payload.get("p2_win_rate"),
            checkpoint_eval_win_rate=payload.get("checkpoint_eval_win_rate"),
            matchup_name=str(payload.get("matchup_name", "")),
            training_stats=payload.get("training_stats"),
        )
