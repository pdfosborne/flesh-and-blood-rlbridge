"""Subprocess rollout workers for fast Talishar training."""

from __future__ import annotations

import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
_TRAINING_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from runtime_defaults import (  # noqa: E402
    envs_per_rollout_process,
    normalize_rollout_mode,
    resolve_rollout_processes,
)


def _process_pool_entry(payload: dict[str, Any]) -> dict[str, Any]:
    handler = str(payload.get("handler", ""))
    if handler == "rollout_worker":
        return execute_rollout_worker_job(payload)
    raise ValueError(f"unknown rollout worker handler: {handler!r}")


def _resolve_backend_pool(
    *,
    base_url: str,
    backend_pool_urls: list[str] | None,
):
    from flesh_and_blood_rlbridge.talishar_backend_pool import (  # noqa: PLC0415
        TalisharBackendPool,
    )

    if backend_pool_urls:
        return TalisharBackendPool.from_urls(backend_pool_urls)
    return TalisharBackendPool.from_runtime(fallback_url=base_url)


def execute_rollout_worker_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Process-pool entry: collect rollout episodes and return transition buffers."""
    import _bootstrap  # noqa: E402, PLC0415

    _bootstrap.configure_paths()

    from train_dual_agent_common import (  # noqa: E402, PLC0415
        Matchup,
        _run_parallel_fast_episode_batch,
        make_env,
        swapped_matchup,
    )
    from rl_agents.ppo import PPOAgent  # noqa: E402, PLC0415

    matchup_data = dict(payload["matchup"])
    matchup_data.setdefault("tags", [])
    matchup = Matchup(**matchup_data)
    swap = swapped_matchup(matchup)
    base_url = str(payload["base_url"])
    backend_pool_urls = payload.get("backend_pool_urls")
    urls_list = list(backend_pool_urls) if backend_pool_urls else None
    talishar_pool = _resolve_backend_pool(
        base_url=base_url,
        backend_pool_urls=urls_list,
    )
    game_format = str(payload["game_format"])
    max_steps = int(payload["max_steps"])
    warmup = bool(payload.get("warmup", False))
    rollout_mode = normalize_rollout_mode(payload.get("rollout_mode"))
    n_envs = max(1, int(payload.get("n_envs", 1)))
    episode_indices = [int(x) for x in payload.get("episode_indices") or []]
    if len(episode_indices) != n_envs:
        episode_indices = list(range(n_envs))
    seed_base = payload.get("seed_base")
    seed_base_i = int(seed_base) if seed_base is not None else None

    p1_path = Path(str(payload["p1_policy_path"]))
    p2_path = Path(str(payload["p2_policy_path"]))
    p1_policy = PPOAgent()
    p1_policy.load(p1_path)
    p2_policy = PPOAgent()
    p2_policy.load(p2_path)

    envs = []
    swap_envs = []
    for _ in range(n_envs):
        worker_url = talishar_pool.allocate_url()
        envs.append(
            make_env(
                matchup,
                base_url=worker_url,
                game_format=game_format,
                max_turns=max_steps,
            )
        )
        swap_envs.append(
            make_env(
                swap,
                base_url=worker_url,
                game_format=game_format,
                max_turns=max_steps,
            )
        )
    try:
        results = _run_parallel_fast_episode_batch(
            envs[:n_envs],
            p1_policy,
            p2_policy,
            max_steps=max_steps,
            warmup=warmup,
            episode_indices=episode_indices,
            seed_base=seed_base_i,
            swap_envs=swap_envs[:n_envs],
            rollout_mode=rollout_mode,
            max_workers=n_envs,
        )
    finally:
        for env in envs + swap_envs:
            try:
                env.close()
            except Exception:
                pass

    return {
        "worker_index": int(payload.get("worker_index", 0)),
        "results": results,
    }


def _serialize_policy(agent: Any, staging_dir: Path, name: str) -> str:
    staging_dir.mkdir(parents=True, exist_ok=True)
    path = staging_dir / f"{name}.json"
    agent.save(path)
    return str(path)


def collect_rollout_batch(
    *,
    p1_policy: Any,
    p2_policy: Any,
    matchup: Any,
    n_episodes: int,
    n_workers: int,
    max_steps: int,
    base_url: str,
    game_format: str,
    rollout_mode: str,
    rollout_processes: int,
    seed_base: Optional[int],
    warmup: bool,
    staging_dir: Path,
    backend_pool_urls: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Collect one rollout batch using subprocess workers when ``rollout_processes > 1``."""
    processes = resolve_rollout_processes(
        rollout_processes,
        default=1,
    )
    talishar_pool = _resolve_backend_pool(
        base_url=base_url,
        backend_pool_urls=backend_pool_urls,
    )
    if processes <= 1:
        from train_dual_agent_common import (  # noqa: PLC0415
            _run_parallel_fast_episode_batch,
            make_env,
            swapped_matchup,
        )

        envs = []
        swap_envs = []
        for _ in range(n_workers):
            worker_url = talishar_pool.allocate_url()
            envs.append(
                make_env(
                    matchup,
                    base_url=worker_url,
                    game_format=game_format,
                    max_turns=max_steps,
                )
            )
            swap_envs.append(
                make_env(
                    swapped_matchup(matchup),
                    base_url=worker_url,
                    game_format=game_format,
                    max_turns=max_steps,
                )
            )
        try:
            return _run_parallel_fast_episode_batch(
                envs[:n_workers],
                p1_policy,
                p2_policy,
                max_steps=max_steps,
                warmup=warmup,
                episode_indices=list(range(n_workers)),
                seed_base=seed_base,
                swap_envs=swap_envs[:n_workers],
                rollout_mode=rollout_mode,
                max_workers=n_workers,
            )
        finally:
            for env in envs + swap_envs:
                try:
                    env.close()
                except Exception:
                    pass

    envs_per_proc = envs_per_rollout_process(n_workers, processes)
    p1_path = _serialize_policy(p1_policy, staging_dir, "p1")
    p2_path = _serialize_policy(p2_policy, staging_dir, "p2")
    matchup_payload = {
        "name": matchup.name,
        "p1_deck": matchup.p1_deck,
        "p2_deck": matchup.p2_deck,
        "description": matchup.description,
        "tags": list(matchup.tags),
        "p1_hero": matchup.p1_hero,
        "p2_hero": matchup.p2_hero,
        "cpp_engine_dir": matchup.cpp_engine_dir,
        "p1_fabrary_entry": matchup.p1_fabrary_entry,
        "p2_fabrary_entry": matchup.p2_fabrary_entry,
    }

    jobs: list[dict[str, Any]] = []
    assigned = 0
    for worker_index in range(processes):
        batch_n = min(envs_per_proc, n_episodes - assigned)
        if batch_n <= 0:
            break
        jobs.append(
            {
                "handler": "rollout_worker",
                "worker_index": worker_index,
                "matchup": matchup_payload,
                "base_url": talishar_pool.url_for_worker(worker_index),
                "backend_pool_urls": list(talishar_pool.urls),
                "game_format": game_format,
                "max_steps": max_steps,
                "warmup": warmup,
                "rollout_mode": normalize_rollout_mode(rollout_mode),
                "n_envs": batch_n,
                "episode_indices": list(range(assigned, assigned + batch_n)),
                "seed_base": seed_base,
                "p1_policy_path": p1_path,
                "p2_policy_path": p2_path,
            }
        )
        assigned += batch_n

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        for row in pool.map(_process_pool_entry, jobs):
            results.extend(row.get("results") or [])
    return results
