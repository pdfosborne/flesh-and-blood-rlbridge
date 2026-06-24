"""Phase 3 play-agent training for the FaB RL pipeline."""

from __future__ import annotations

import argparse
import glob
import html
import json
import math
import os
import statistics
import sys
import threading
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_ROOT))
import _bootstrap  # noqa: E402

REPO_ROOT = _bootstrap.configure_paths()
_RL_SRC = Path("~/Documents/RL/rlbridge/src").expanduser()
if str(_RL_SRC) not in sys.path:
    sys.path.insert(0, str(_RL_SRC))

from flesh_and_blood_rlbridge import TalisharEngineEnvironment  # noqa: E402
from flesh_and_blood_rlbridge.opponent_deck import normalize_talishar_asset_name  # noqa: E402

from train_pipeline_common import (  # noqa: E402
    DEFAULT_AGENT_CACHE_DIR,
    DEFAULT_EQUIPMENT_HEADER,
    DEFAULT_FORMAT,
    DEFAULT_HERO_ID,
    DEFAULT_OPPONENT_DECK,
    DEFAULT_OPPONENT_HERO,
    OUT_DIR,
    PhaseAgents,
    _load_agent,
    _load_starting_deck,
    _runtime_backend_label,
    _save_agent,
    _save_all_agents,
    _write_deck_file,
    _write_results_json,
    apply_deck_state,
    build_matchup_deck_export,
    greedy_game_deck_cut,
    load_deck_state,
    min_deck_size_for_format,
    resolve_assets_path,
    save_deck_state,
)
from cpp_engine_matchup import (  # noqa: E402
    ensure_cpp_engine_for_matchup,
    resolve_cpp_lookup_decks,
)
from parallel_seed_training import (  # noqa: E402
    DEFAULT_PARALLEL_SEEDS,
    derive_training_seed,
    merge_parallel_seed_checkpoint_history,
    run_parallel_seed_jobs,
    select_best_agents_by_win_rate,
    sync_parallel_seed_dashboard_artifacts,
    workers_per_parallel_seed,
)
from checkpoint_eval_async import (  # noqa: E402
    submit_checkpoint_eval,
    shutdown_checkpoint_eval_executor,
    wait_for_checkpoint_evals,
)
from play_outcome_stats import (  # noqa: E402
    absolute_p1_p2_deck_from_env,
    absolute_p1_p2_deck_from_obs,
    absolute_p1_p2_hp_from_env,
    absolute_p1_p2_hp_from_obs,
    classify_p1_episode_outcome,
    compute_eval_stability,
    summarize_p1_outcomes,
)
from runtime_defaults import (  # noqa: E402
    DEFAULT_CHECKPOINT_EVAL_EPISODES,
    DEFAULT_CHECKPOINT_INTERVAL_PCT,
    RUNTIME,
)

try:
    from train_dual_agent_common import (  # noqa: E402
        make_agent,
        make_env,
        Matchup,
        train_agents_from_both_perspectives,
        train_agents_from_both_perspectives_parallel,
        _evaluate_policy_pair,
        _env_supports_fast_training,
        _announce_training_backend,
        _mask_logits_to_legal,
        _player_context,
        _save_warmup_handoff_checkpoint,
        DEFAULT_N_EPISODES,
        DEFAULT_WARMUP_EPISODES,
        DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
    )
    from agent_cache import (  # noqa: E402
        AgentCacheStore,
        deck_content_fingerprint,
        talishar_asset_deck_fingerprint,
    )
    from episode_cache import EpisodeCache  # noqa: E402
    _DUAL_AGENT_AVAILABLE = True
except ImportError:
    _DUAL_AGENT_AVAILABLE = False
    DEFAULT_N_EPISODES = RUNTIME.dual_matchup.episodes
    DEFAULT_WARMUP_EPISODES = RUNTIME.play.warmup_episodes
    DEFAULT_WARMUP_BASELINE_EVAL_EPISODES = RUNTIME.play.warmup_baseline_eval_episodes


def resolve_play_checkpoint_interval(
    n_episodes: int,
    *,
    checkpoint_interval: Optional[int] = None,
    checkpoint_interval_pct: float = DEFAULT_CHECKPOINT_INTERVAL_PCT,
) -> int:
    """Resolve checkpoint cadence — fixed interval or a %% of total episodes."""
    if checkpoint_interval is not None and checkpoint_interval > 0:
        return int(checkpoint_interval)
    pct = max(0.1, float(checkpoint_interval_pct))
    return max(1, int(math.ceil(n_episodes * pct / 100.0)))


def _resolve_phase3_deck_fingerprints(
    *,
    opponent_mode: str,
    p1_game_deck: dict[str, int],
    p1_equipment_header: str,
    p1_opponent_deck_name: str,
    assets_path: str,
    p2_game_deck: Optional[dict[str, int]] = None,
    p2_equipment_header: str = "",
) -> tuple[str, str]:
    """Content fingerprints for exact deck-vs-deck cache lookup."""
    p1_fp = deck_content_fingerprint(
        p1_game_deck,
        equipment_header=p1_equipment_header,
    )
    if opponent_mode == "preset":
        p2_fp = talishar_asset_deck_fingerprint(assets_path, p1_opponent_deck_name)
    elif opponent_mode == "mirror":
        p2_fp = p1_fp
    else:
        if not p2_game_deck:
            raise ValueError("dual play requires p2_game_deck for fingerprinting")
        p2_fp = deck_content_fingerprint(
            p2_game_deck,
            equipment_header=p2_equipment_header,
        )
    return p1_fp, p2_fp


def _load_converged_play_agent(
    cache_store: "AgentCacheStore",
    *,
    matchup: "Matchup",
    p1_deck_fingerprint: str,
    p2_deck_fingerprint: str,
    seed: Optional[int],
) -> tuple[Any, Any]:
    """Bootstrap P1 tiers from converged deck-vs-deck cache."""
    def _make_p1() -> Any:
        return make_agent(seed=seed)

    p1_bundle = cache_store.bootstrap_player(
        _player_context(
            matchup,
            as_p1=True,
            p1_deck_fingerprint=p1_deck_fingerprint,
            p2_deck_fingerprint=p2_deck_fingerprint,
        ),
        _make_p1,
    )
    return p1_bundle.agents[0], p1_bundle


def _write_play_cache_hit(
    out_dir: Path,
    *,
    cached_record: Any,
    p1_deck_fingerprint: str,
    p2_deck_fingerprint: str,
) -> None:
    from dataclasses import asdict

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "skipped_training": True,
        "reason": "converged_deck_matchup_cache",
        "p1_deck_fingerprint": p1_deck_fingerprint,
        "p2_deck_fingerprint": p2_deck_fingerprint,
        **asdict(cached_record),
    }
    (out_dir / "play_cache_hit.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

def run_phase3_play_preset(
    agents: PhaseAgents,
    opponent_hero_id: str,
    *,
    game_format: str,
    opponent_deck_name: str,
    equipment_header: str,
    n_episodes: int,
    max_play_steps: int,
    assets_path: str,
    base_url: str,
    render: bool,
) -> float:
    """Legacy preset play — delegates to run_phase3_play."""
    return run_phase3_play(
        agents, None,
        opponent_mode="preset",
        game_format=game_format,
        p1_hero_id="",
        p2_hero_id="",
        p1_equipment_header=equipment_header,
        p2_equipment_header="",
        p1_opponent_hero_id=opponent_hero_id,
        p1_opponent_deck_name=opponent_deck_name,
        n_episodes=n_episodes,
        max_play_steps=max_play_steps,
        warmup_episodes=DEFAULT_WARMUP_EPISODES,
        warmup_baseline_eval_episodes=DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
        n_workers=None,
        assets_path=assets_path,
        base_url=base_url,
        out_dir=Path(assets_path).parent,
        cache_dir=None,
        seed=None,
    )[0]


def run_phase3_play_mirror(
    agents: PhaseAgents,
    *,
    game_format: str,
    equipment_header: str,
    n_episodes: int,
    max_play_steps: int,
    assets_path: str,
    base_url: str,
    render: bool,
) -> float:
    """Legacy mirror play — delegates to run_phase3_play."""
    return run_phase3_play(
        agents, None,
        opponent_mode="mirror",
        game_format=game_format,
        p1_hero_id="",
        p2_hero_id="",
        p1_equipment_header=equipment_header,
        p2_equipment_header=equipment_header,
        p1_opponent_hero_id="",
        p1_opponent_deck_name="",
        n_episodes=n_episodes,
        max_play_steps=max_play_steps,
        warmup_episodes=DEFAULT_WARMUP_EPISODES,
        warmup_baseline_eval_episodes=DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
        n_workers=None,
        assets_path=assets_path,
        base_url=base_url,
        out_dir=Path(assets_path).parent,
        cache_dir=None,
        seed=None,
    )[0]


def run_phase3_play_dual(
    p1: PhaseAgents,
    p2: PhaseAgents,
    *,
    game_format: str,
    p1_equipment_header: str,
    p2_equipment_header: str,
    n_episodes: int,
    max_play_steps: int,
    assets_path: str,
    base_url: str,
    render: bool,
) -> tuple[float, float]:
    """Legacy dual play — delegates to run_phase3_play."""
    return run_phase3_play(
        p1, p2,
        opponent_mode="dual",
        game_format=game_format,
        p1_hero_id="",
        p2_hero_id="",
        p1_equipment_header=p1_equipment_header,
        p2_equipment_header=p2_equipment_header,
        p1_opponent_hero_id="",
        p1_opponent_deck_name="",
        n_episodes=n_episodes,
        max_play_steps=max_play_steps,
        warmup_episodes=DEFAULT_WARMUP_EPISODES,
        warmup_baseline_eval_episodes=DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
        n_workers=None,
        assets_path=assets_path,
        base_url=base_url,
        out_dir=Path(assets_path).parent,
        cache_dir=None,
        seed=None,
    )


# ---------------------------------------------------------------------------
# Phase 3 — Play  (unified, uses train_dual_agent_common warmup infrastructure)
# ---------------------------------------------------------------------------


def _clone_phase_agents_for_seed(
    p1: PhaseAgents,
    p2: Optional[PhaseAgents],
    *,
    seed: Optional[int],
) -> tuple[PhaseAgents, Optional[PhaseAgents]]:
    """Copy deck state for an independent parallel-seed training run."""
    from agent_cache import clone_agent_weights  # noqa: PLC0415

    def _clone_play(agent: Any) -> Optional[Any]:
        if agent is None:
            return None
        cloned = make_agent(seed=seed)
        clone_agent_weights(agent, cloned)
        return cloned

    p1_copy = PhaseAgents(
        player=p1.player,
        deckbuilder=p1.deckbuilder,
        sideboard=p1.sideboard,
        play=_clone_play(p1.play),
        equipment_header=p1.equipment_header,
        card_pool=dict(p1.card_pool),
        pool_by_id=dict(p1.pool_by_id),
        active_decks={k: dict(v) for k, v in p1.active_decks.items()},
    )
    p2_copy: Optional[PhaseAgents] = None
    if p2 is not None:
        p2_copy = PhaseAgents(
            player=p2.player,
            deckbuilder=p2.deckbuilder,
            sideboard=p2.sideboard,
            play=_clone_play(p2.play),
            equipment_header=p2.equipment_header,
            card_pool=dict(p2.card_pool),
            pool_by_id=dict(p2.pool_by_id),
            active_decks={k: dict(v) for k, v in p2.active_decks.items()},
        )
    return p1_copy, p2_copy


def _serialize_phase_agent_for_process(
    pa: PhaseAgents,
    staging_dir: Path,
    *,
    role: str,
) -> dict[str, Any]:
    """Pickle-safe PhaseAgents payload for ProcessPoolExecutor workers."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    play_path: Optional[str] = None
    if pa.play is not None:
        play_path = str(staging_dir / f"{role}_play.json")
        pa.play.save(play_path)
    return {
        "player": pa.player,
        "equipment_header": pa.equipment_header,
        "card_pool": dict(pa.card_pool),
        "pool_by_id": dict(pa.pool_by_id),
        "active_decks": {k: dict(v) for k, v in pa.active_decks.items()},
        "deck_asset_name": pa.deck_asset_name,
        "play_path": play_path,
    }


def _deserialize_phase_agent_from_process(
    data: dict[str, Any],
    *,
    seed: Optional[int],
) -> PhaseAgents:
    play = None
    play_path = data.get("play_path")
    if play_path:
        play = make_agent(seed=seed)
        play.load(play_path)
    return PhaseAgents(
        player=str(data.get("player", "p1")),
        equipment_header=str(data.get("equipment_header", "")),
        card_pool=dict(data.get("card_pool") or {}),
        pool_by_id=dict(data.get("pool_by_id") or {}),
        active_decks={
            str(k): dict(v) for k, v in (data.get("active_decks") or {}).items()
        },
        deck_asset_name=str(data.get("deck_asset_name", "")),
        play=play,
    )


def _load_play_agent_from_path(
    path: Optional[str],
    *,
    seed: Optional[int] = None,
) -> Optional[Any]:
    if not path:
        return None
    agent = make_agent(seed=seed)
    agent.load(path)
    return agent


def execute_play_parallel_seed_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Process-pool entry: run one parallel play seed from a serialized payload."""
    seed_index = int(payload["seed_index"])
    seed_i = payload.get("seed")
    seed_out = Path(str(payload["seed_out"]))
    p1 = _deserialize_phase_agent_from_process(payload["p1"], seed=seed_i)
    p2_data = payload.get("p2")
    p2 = (
        _deserialize_phase_agent_from_process(p2_data, seed=seed_i)
        if isinstance(p2_data, dict)
        else None
    )
    capture: dict[str, Any] = {}
    run_kwargs = dict(payload.get("run_kwargs") or {})
    sync_dir = payload.get("parallel_seed_sync_dir")
    p1_wr, p2_wr = run_phase3_play(
        p1,
        p2,
        out_dir=seed_out,
        seed=seed_i,
        _seed_run_capture=capture,
        _retain_temp_decks=True,
        _skip_cache_converge=True,
        _force_train=True,
        _suppress_train_progress=bool(payload.get("suppress_train_progress", False)),
        _parallel_seed_sync_dir=Path(sync_dir) if sync_dir else None,
        **run_kwargs,
    )
    return {
        **capture,
        "seed_index": seed_index,
        "seed": seed_i,
        "out_dir": str(seed_out),
        "p1_win_rate": p1_wr,
        "p2_win_rate": p2_wr,
    }


def _run_phase3_play_parallel_seeds(
    p1: PhaseAgents,
    p2: Optional[PhaseAgents],
    *,
    parallel_seeds: int,
    opponent_mode: str,
    game_format: str,
    p1_hero_id: str,
    p2_hero_id: str,
    p1_equipment_header: str,
    p2_equipment_header: str,
    p1_opponent_hero_id: str,
    p1_opponent_deck_name: str,
    n_episodes: int,
    max_play_steps: int,
    warmup_episodes: int,
    warmup_baseline_eval_episodes: int,
    n_workers: Optional[int],
    assets_path: str,
    base_url: str,
    out_dir: Path,
    cache_dir: Optional[Path],
    seed: Optional[int],
    cpp_engine_dir: Optional[str],
    checkpoint_interval: Optional[int],
    checkpoint_interval_pct: float,
    checkpoint_eval_episodes: int,
    parallel_seeds_until_first_checkpoint: bool,
    parallel_progress_label: Optional[str],
) -> tuple[float, float]:
    workers_per_seed = workers_per_parallel_seed(n_workers, parallel_seeds)
    if n_workers is not None and workers_per_seed != n_workers:
        print(
            f"  Parallel seeds: {parallel_seeds} × {workers_per_seed} worker(s)/seed "
            f"(total rollout budget {n_workers})"
        )

    shared_kwargs = dict(
        opponent_mode=opponent_mode,
        game_format=game_format,
        p1_hero_id=p1_hero_id,
        p2_hero_id=p2_hero_id,
        p1_equipment_header=p1_equipment_header,
        p2_equipment_header=p2_equipment_header,
        p1_opponent_hero_id=p1_opponent_hero_id,
        p1_opponent_deck_name=p1_opponent_deck_name,
        n_episodes=n_episodes,
        max_play_steps=max_play_steps,
        warmup_episodes=warmup_episodes,
        warmup_baseline_eval_episodes=warmup_baseline_eval_episodes,
        n_workers=workers_per_seed,
        assets_path=assets_path,
        base_url=base_url,
        cache_dir=cache_dir,
        cpp_engine_dir=cpp_engine_dir,
        checkpoint_interval=checkpoint_interval,
        checkpoint_interval_pct=checkpoint_interval_pct,
        parallel_seeds=1,
        checkpoint_eval_episodes=checkpoint_eval_episodes,
    )

    def _build_process_jobs(
        *,
        run_kwargs: dict[str, Any],
        seed_offset: int = 0,
        initial_p1: Optional[PhaseAgents] = None,
        initial_p2: Optional[PhaseAgents] = None,
        seed_indices: Optional[list[int]] = None,
    ) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        indices = seed_indices if seed_indices is not None else list(range(parallel_seeds))
        for seed_index in indices:
            seed_i = derive_training_seed(seed, seed_index)
            if seed_i is not None:
                seed_i += seed_offset
            source_p1 = initial_p1 if initial_p1 is not None else p1
            source_p2 = initial_p2 if initial_p2 is not None else p2
            p1_copy, p2_copy = _clone_phase_agents_for_seed(source_p1, source_p2, seed=seed_i)
            seed_staging = staging_root / f"seed_{seed_index}_offset_{seed_offset}"
            job: dict[str, Any] = {
                "handler": "play_parallel_seed",
                "seed_index": seed_index,
                "n_seeds": parallel_seeds,
                "workers_per_seed": workers_per_seed,
                "seed": seed_i,
                "seed_out": str(seeds_root / f"seed_{seed_index}"),
                "parallel_seed_sync_dir": str(out_dir),
                "suppress_train_progress": True,
                "p1": _serialize_phase_agent_for_process(
                    p1_copy, seed_staging / "p1", role="p1"
                ),
                "run_kwargs": run_kwargs,
            }
            if opponent_mode == "dual" and p2_copy is not None:
                job["p2"] = _serialize_phase_agent_for_process(
                    p2_copy, seed_staging / "p2", role="p2"
                )
            jobs.append(job)
        return jobs

    def _sync_parent_dashboard() -> None:
        sync_parallel_seed_dashboard_artifacts(out_dir)

    def _read_checkpoint_history(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return [row for row in raw if isinstance(row, dict)] if isinstance(raw, list) else []

    seeds_root = out_dir / "parallel_seeds"
    staging_root = seeds_root / "_agent_staging"
    staged_temp_deck_files: list[str] = []

    if parallel_seeds_until_first_checkpoint and parallel_seeds > 1:
        first_checkpoint = min(
            n_episodes,
            resolve_play_checkpoint_interval(
                n_episodes,
                checkpoint_interval=checkpoint_interval,
                checkpoint_interval_pct=checkpoint_interval_pct,
            ),
        )
        remaining_episodes = max(0, n_episodes - first_checkpoint)
        if remaining_episodes > 0:
            print(
                f"  Parallel seeds until first checkpoint: {parallel_seeds} seed(s) "
                f"for {first_checkpoint} episode(s), then continue best seed for "
                f"{remaining_episodes} episode(s)"
            )
            first_kwargs = dict(shared_kwargs)
            first_kwargs["n_episodes"] = first_checkpoint
            first_kwargs["warmup_episodes"] = min(warmup_episodes, first_checkpoint)
            first_kwargs["checkpoint_interval"] = first_checkpoint
            first_jobs = _build_process_jobs(run_kwargs=first_kwargs)
            try:
                first_summary = run_parallel_seed_jobs(
                    parallel_seeds,
                    seed,
                    out_dir,
                    process_jobs=first_jobs,
                    workers_per_seed=workers_per_seed,
                    label=(
                        f"{parallel_progress_label} · first checkpoint"
                        if parallel_progress_label
                        else "play training first checkpoint"
                    ),
                    on_seed_complete=lambda _row: _sync_parent_dashboard(),
                )
            finally:
                shutdown_checkpoint_eval_executor(wait=True)

            for row in first_summary.seed_rows:
                staged_temp_deck_files.extend(str(p) for p in row.get("temp_deck_files") or [])
                row.setdefault(
                    "p1_agent",
                    _load_play_agent_from_path(row.get("p1_play_path"), seed=row.get("seed")),
                )
                row.setdefault(
                    "p2_agent",
                    _load_play_agent_from_path(row.get("p2_play_path"), seed=row.get("seed")),
                )
            best_p1, best_p2, best_p1_idx, best_p2_idx = select_best_agents_by_win_rate(
                first_summary.seed_rows
            )
            first_checkpoint_eval = merge_parallel_seed_checkpoint_history(out_dir, write=True)
            if first_checkpoint_eval:
                eval_best_p1_idx = first_checkpoint_eval[-1].get("best_p1_seed_index")
                if eval_best_p1_idx is not None:
                    try:
                        best_p1_idx = int(eval_best_p1_idx)
                        best_p1 = next(
                            row["p1_agent"] for row in first_summary.seed_rows
                            if int(row.get("seed_index", -1)) == best_p1_idx
                        )
                    except (KeyError, StopIteration, TypeError, ValueError):
                        pass
            first_p1_row = next(
                row for row in first_summary.seed_rows
                if int(row.get("seed_index", -1)) == best_p1_idx
            )
            first_p2_row = next(
                row for row in first_summary.seed_rows
                if int(row.get("seed_index", -1)) == best_p2_idx
            )
            print(
                f"  Continuing only best seed after first checkpoint: "
                f"P1 seed {best_p1_idx}, P2 seed {best_p2_idx}"
            )
            selected_seed_history_path = (
                seeds_root / f"seed_{best_p1_idx}" / "checkpoint_eval_history.json"
            )
            selected_first_checkpoint_history = _read_checkpoint_history(
                selected_seed_history_path
            )

            continuation_p1 = PhaseAgents(
                player=p1.player,
                play=best_p1,
                equipment_header=p1.equipment_header,
                card_pool=dict(p1.card_pool),
                pool_by_id=dict(p1.pool_by_id),
                active_decks={k: dict(v) for k, v in p1.active_decks.items()},
                deck_asset_name=p1.deck_asset_name,
            )
            continuation_p2: Optional[PhaseAgents] = None
            if opponent_mode == "dual" and p2 is not None:
                continuation_p2 = PhaseAgents(
                    player=p2.player,
                    play=best_p2,
                    equipment_header=p2.equipment_header,
                    card_pool=dict(p2.card_pool),
                    pool_by_id=dict(p2.pool_by_id),
                    active_decks={k: dict(v) for k, v in p2.active_decks.items()},
                    deck_asset_name=p2.deck_asset_name,
                )
            elif p2 is not None:
                continuation_p2 = PhaseAgents(
                    player=p2.player,
                    play=best_p2,
                    equipment_header=p2.equipment_header,
                    card_pool=dict(p2.card_pool),
                    pool_by_id=dict(p2.pool_by_id),
                    active_decks={k: dict(v) for k, v in p2.active_decks.items()},
                    deck_asset_name=p2.deck_asset_name,
                )

            continuation_kwargs = dict(shared_kwargs)
            continuation_kwargs["n_episodes"] = remaining_episodes
            continuation_kwargs["warmup_episodes"] = max(0, warmup_episodes - first_checkpoint)
            continuation_kwargs["checkpoint_interval"] = first_checkpoint
            continuation_jobs = _build_process_jobs(
                run_kwargs=continuation_kwargs,
                seed_offset=first_checkpoint,
                initial_p1=continuation_p1,
                initial_p2=continuation_p2,
                seed_indices=[best_p1_idx],
            )
            try:
                continuation_summary = run_parallel_seed_jobs(
                    1,
                    seed,
                    out_dir,
                    process_jobs=continuation_jobs,
                    workers_per_seed=workers_per_seed,
                    label=(
                        f"{parallel_progress_label} · best-seed continuation"
                        if parallel_progress_label
                        else "play training best-seed continuation"
                    ),
                    on_seed_complete=lambda _row: _sync_parent_dashboard(),
                )
            finally:
                shutdown_checkpoint_eval_executor(wait=True)
            continuation_history = _read_checkpoint_history(selected_seed_history_path)
            if continuation_history:
                adjusted_history = list(selected_first_checkpoint_history)
                for row in continuation_history:
                    adjusted = dict(row)
                    adjusted["episodes_completed"] = (
                        int(adjusted.get("episodes_completed", 0) or 0)
                        + first_checkpoint
                    )
                    adjusted["target_episodes"] = n_episodes
                    adjusted_history.append(adjusted)
                selected_seed_history_path.write_text(
                    json.dumps(adjusted_history, indent=2),
                    encoding="utf-8",
                )
                merge_parallel_seed_checkpoint_history(out_dir, write=True)
            summary = continuation_summary
            final_row = summary.seed_rows[0]
            combined_p1_wr = (
                float(first_p1_row.get("p1_win_rate", 0.0)) * first_checkpoint
                + float(final_row.get("p1_win_rate", 0.0)) * remaining_episodes
            ) / max(1, n_episodes)
            combined_p2_wr = (
                float(first_p2_row.get("p2_win_rate", 0.0)) * first_checkpoint
                + float(final_row.get("p2_win_rate", 0.0)) * remaining_episodes
            ) / max(1, n_episodes)
            final_row["p1_win_rate"] = combined_p1_wr
            final_row["p2_win_rate"] = combined_p2_wr
            final_row["first_checkpoint_p1_seed_index"] = best_p1_idx
            final_row["first_checkpoint_p2_seed_index"] = best_p2_idx
            final_row["first_checkpoint_episodes"] = first_checkpoint
            final_row["continued_episodes"] = remaining_episodes
            final_row["temp_deck_files"] = list(final_row.get("temp_deck_files") or []) + staged_temp_deck_files
            summary.avg_p1_win_rate = combined_p1_wr
            summary.avg_p2_win_rate = combined_p2_wr
            summary.best_p1_seed_index = best_p1_idx
            summary.best_p2_seed_index = best_p2_idx
            summary.n_seeds = 1
            (out_dir / "parallel_seeds_summary.json").write_text(
                json.dumps(summary.to_dict(), indent=2),
                encoding="utf-8",
            )
        else:
            process_jobs = _build_process_jobs(run_kwargs=shared_kwargs)
            try:
                summary = run_parallel_seed_jobs(
                    parallel_seeds,
                    seed,
                    out_dir,
                    process_jobs=process_jobs,
                    workers_per_seed=workers_per_seed,
                    label=parallel_progress_label or "play training",
                    on_seed_complete=lambda _row: _sync_parent_dashboard(),
                )
            finally:
                shutdown_checkpoint_eval_executor(wait=True)
    else:
        process_jobs = _build_process_jobs(run_kwargs=shared_kwargs)

        try:
            summary = run_parallel_seed_jobs(
                parallel_seeds,
                seed,
                out_dir,
                process_jobs=process_jobs,
                workers_per_seed=workers_per_seed,
                label=parallel_progress_label or "play training",
                on_seed_complete=lambda _row: _sync_parent_dashboard(),
            )
        finally:
            shutdown_checkpoint_eval_executor(wait=True)

    for row in summary.seed_rows:
        if "p1_agent" not in row:
            row["p1_agent"] = _load_play_agent_from_path(
                row.get("p1_play_path"),
                seed=row.get("seed"),
            )
        if "p2_agent" not in row:
            row["p2_agent"] = _load_play_agent_from_path(
                row.get("p2_play_path"),
                seed=row.get("seed"),
            )
    best_p1, best_p2, best_p1_idx, best_p2_idx = select_best_agents_by_win_rate(
        summary.seed_rows
    )
    best_p1_row = next(
        (
            row for row in summary.seed_rows
            if int(row.get("seed_index", -1)) == best_p1_idx
        ),
        summary.seed_rows[0],
    )

    p1.play = best_p1
    if opponent_mode == "dual" and p2 is not None:
        p2.play = best_p2

    p1.win_rates.append(summary.avg_p1_win_rate)
    if opponent_mode == "dual" and p2 is not None:
        p2.win_rates.append(summary.avg_p2_win_rate)

    checkpoint_eval_log = merge_parallel_seed_checkpoint_history(out_dir, write=True)
    latest_ckpt_wr: Optional[float] = None
    if checkpoint_eval_log:
        latest = checkpoint_eval_log[-1]
        latest_ckpt_wr = float(latest.get("p1_win_rate", 0.0))
        best_seed = latest.get("best_p1_seed_index")
        print(
            f"  [p1] Latest merged checkpoint eval (best seed {best_seed}): "
            f"win%={latest_ckpt_wr:.1%} @ ep {latest.get('episodes_completed')}"
        )

    if cache_dir is not None and p1.play is not None:
        cache_store = AgentCacheStore(cache_dir, game_format)
        p1_deck_fp = best_p1_row.get("p1_deck_fingerprint")
        p2_deck_fp = best_p1_row.get("p2_deck_fingerprint")
        if p1_deck_fp and p2_deck_fp:
            latest_ckpt_wr = (
                float(checkpoint_eval_log[-1]["p1_win_rate"])
                if checkpoint_eval_log
                else None
            )
            cache_store.mark_matchup_converged(
                p1_fingerprint=str(p1_deck_fp),
                p2_fingerprint=str(p2_deck_fp),
                p1_hero=p1_hero_id.replace("_", "-"),
                p2_hero=(p2_hero_id or p1_opponent_hero_id).replace("_", "-"),
                episodes_completed=n_episodes,
                target_episodes=n_episodes,
                p1_win_rate=summary.avg_p1_win_rate,
                checkpoint_eval_win_rate=latest_ckpt_wr,
            )

    for row in summary.seed_rows:
        for deck_file in row.get("temp_deck_files") or []:
            try:
                Path(deck_file).unlink(missing_ok=True)
            except Exception:
                pass

    print(
        f"\n  Parallel-seed train win% (avg over {summary.n_seeds} active seed(s)): "
        f"p1={summary.avg_p1_win_rate:.1%}  p2={summary.avg_p2_win_rate:.1%}"
    )
    return summary.avg_p1_win_rate, summary.avg_p2_win_rate


def run_phase3_play(
    p1: PhaseAgents,
    p2: Optional[PhaseAgents],
    *,
    opponent_mode: str,               # "preset" | "mirror" | "dual"
    game_format: str,
    p1_hero_id: str,
    p2_hero_id: str,
    p1_equipment_header: str,
    p2_equipment_header: str,
    p1_opponent_hero_id: str,         # used to pick the right sideborded deck
    p1_opponent_deck_name: str,       # Talishar Assets deck name (preset mode)
    n_episodes: int,
    max_play_steps: int,
    warmup_episodes: int,             # default-policy warmup before PPO
    warmup_baseline_eval_episodes: int,
    n_workers: Optional[int],
    assets_path: str,
    base_url: str,
    out_dir: Path,
    cache_dir: Optional[Path],
    seed: Optional[int] = None,
    cpp_engine_dir: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
    checkpoint_interval_pct: float = DEFAULT_CHECKPOINT_INTERVAL_PCT,
    checkpoint_eval_episodes: int = DEFAULT_CHECKPOINT_EVAL_EPISODES,
    parallel_seeds: int = 1,
    parallel_seeds_until_first_checkpoint: bool = RUNTIME.play.parallel_seeds_until_first_checkpoint,
    parallel_progress_label: Optional[str] = None,
    _seed_run_capture: Optional[dict[str, Any]] = None,
    _retain_temp_decks: bool = False,
    _skip_cache_converge: bool = False,
    _force_train: bool = False,
    _suppress_train_progress: bool = False,
    _parallel_seed_sync_dir: Optional[Path] = None,
) -> tuple[float, float]:
    """Co-evolution play using train_dual_agent_common warmup + episode-cache infrastructure.

    Both players always get PPO updates (in preset/mirror mode the "opponent"
    agent is discarded afterwards; only ``p1.play`` is updated).

    When ``parallel_seeds`` > 1, runs that many independent trainings in parallel,
    averages their training win rates for reporting, and uses the best P1/P2
    agents (by per-seed training win rate) for checkpoint evaluation.

    Returns ``(p1_win_rate, p2_win_rate)``.
    """
    if parallel_seeds > 1 and _seed_run_capture is None:
        return _run_phase3_play_parallel_seeds(
            p1,
            p2,
            parallel_seeds=parallel_seeds,
            opponent_mode=opponent_mode,
            game_format=game_format,
            p1_hero_id=p1_hero_id,
            p2_hero_id=p2_hero_id,
            p1_equipment_header=p1_equipment_header,
            p2_equipment_header=p2_equipment_header,
            p1_opponent_hero_id=p1_opponent_hero_id,
            p1_opponent_deck_name=p1_opponent_deck_name,
            n_episodes=n_episodes,
            max_play_steps=max_play_steps,
            warmup_episodes=warmup_episodes,
            warmup_baseline_eval_episodes=warmup_baseline_eval_episodes,
            n_workers=n_workers,
            assets_path=assets_path,
            base_url=base_url,
            out_dir=out_dir,
            cache_dir=cache_dir,
            seed=seed,
            cpp_engine_dir=cpp_engine_dir,
            checkpoint_interval=checkpoint_interval,
            checkpoint_interval_pct=checkpoint_interval_pct,
            checkpoint_eval_episodes=checkpoint_eval_episodes,
            parallel_seeds_until_first_checkpoint=parallel_seeds_until_first_checkpoint,
            parallel_progress_label=parallel_progress_label,
        )
    print(
        f"\n{'='*62}\n"
        f"  PHASE 3 — Play ({opponent_mode})  [{p1.player}"
        + (f" vs {p2.player}" if p2 else "")
        + f"]\n{'='*62}"
    )

    if not _DUAL_AGENT_AVAILABLE:
        print("  WARNING: train_dual_agent_common not available — using fallback loop")
        return _run_phase3_fallback(
            p1, p2,
            opponent_mode=opponent_mode,
            game_format=game_format,
            p1_equipment_header=p1_equipment_header,
            p2_equipment_header=p2_equipment_header,
            p1_opponent_hero_id=p1_opponent_hero_id,
            p1_opponent_deck_name=p1_opponent_deck_name,
            n_episodes=n_episodes,
            max_play_steps=max_play_steps,
            assets_path=assets_path,
            base_url=base_url,
        )

    # ── select game decks ─────────────────────────────────────────────────────
    p1_game_deck = (
        p1.active_decks.get(p1_opponent_hero_id)
        or next(iter(p1.active_decks.values()), {})
        or p1.card_pool
    )
    if not p1_game_deck:
        print(f"  [{p1.player}] No deck available — skipping play phase")
        return 0.0, 0.0

    if opponent_mode == "dual" and p2 is not None:
        p2_game_deck = (
            p2.active_decks.get(p1_hero_id)
            or next(iter(p2.active_decks.values()), {})
            or p2.card_pool
        )
        if not p2_game_deck:
            print(f"  [{p2.player}] No deck available — skipping play phase")
            return 0.0, 0.0
    elif opponent_mode == "mirror":
        p2_game_deck = p1_game_deck
    else:
        p2_game_deck = None  # preset mode: p2_deck_name refers to assets file

    p1_deck_fp, p2_deck_fp = _resolve_phase3_deck_fingerprints(
        opponent_mode=opponent_mode,
        p1_game_deck=p1_game_deck,
        p1_equipment_header=p1_equipment_header,
        p1_opponent_deck_name=p1_opponent_deck_name,
        assets_path=assets_path,
        p2_game_deck=p2_game_deck if opponent_mode == "dual" else None,
        p2_equipment_header=p2_equipment_header,
    )

    cache_root = cache_dir or DEFAULT_AGENT_CACHE_DIR
    cache_store = AgentCacheStore(cache_root, game_format)

    if p1.play is None and not _force_train:
        cached_record = cache_store.should_skip_training(
            p1_fingerprint=p1_deck_fp,
            p2_fingerprint=p2_deck_fp,
            target_episodes=n_episodes,
        )
        if cached_record is not None:
            stub_matchup = Matchup(
                name="cached",
                p1_deck="cached",
                p2_deck=(
                    p1_opponent_deck_name
                    if opponent_mode == "preset"
                    else "cached"
                ),
                description="cached deck-vs-deck agent",
                p1_hero=p1_hero_id.replace("_", "-"),
                p2_hero=(p2_hero_id or p1_opponent_hero_id).replace("_", "-"),
            )
            p1_agent, _ = _load_converged_play_agent(
                cache_store,
                matchup=stub_matchup,
                p1_deck_fingerprint=p1_deck_fp,
                p2_deck_fingerprint=p2_deck_fp,
                seed=seed,
            )
            p1.play = p1_agent
            p1_wr = float(cached_record.p1_win_rate or 0.0)
            out_dir.mkdir(parents=True, exist_ok=True)
            _write_play_cache_hit(
                out_dir,
                cached_record=cached_record,
                p1_deck_fingerprint=p1_deck_fp,
                p2_deck_fingerprint=p2_deck_fp,
            )
            p1.win_rates.append(p1_wr)
            print(
                f"\n  Cache hit — converged deck-vs-deck matchup "
                f"({cached_record.episodes_completed}/{cached_record.target_episodes} ep) "
                f"— skipping training\n"
                f"  Cached P1 win rate: {p1_wr:.1%}"
            )
            return p1_wr, 0.0

    # ── write deck files ──────────────────────────────────────────────────────
    p1_deck_name = f"rl_p3_p1_{uuid.uuid4().hex[:8]}"
    p1_deck_file = _write_deck_file(p1_game_deck, p1_equipment_header, p1_deck_name, assets_path)
    p1.deck_asset_name = p1_deck_name

    if opponent_mode == "preset":
        p2_deck_name = p1_opponent_deck_name
        p2_deck_file: Optional[Path] = None
    elif opponent_mode == "mirror":
        p2_deck_name = p1_deck_name          # same file — mirror match
        p2_deck_file = None
    else:                                     # dual
        p2_deck_name = f"rl_p3_p2_{uuid.uuid4().hex[:8]}"
        p2_deck_file = _write_deck_file(
            p2_game_deck, p2_equipment_header, p2_deck_name, assets_path  # type: ignore[arg-type]
        )
    if p2 is not None:
        p2.deck_asset_name = p2_deck_name

    # ── C++ engine lookup key (resolved Assets stems, not UUID deck names) ─────
    _cpp_deck1, _cpp_deck2 = resolve_cpp_lookup_decks(
        assets_path, p1_hero_id, p2_hero_id or p1_hero_id
    )

    # ── backend visibility (C++ vs HTTP) ────────────────────────────────────
    probe_env = TalisharEngineEnvironment(
        base_url=base_url,
        game_format=game_format,
        local_deck_name=p1_deck_name,
        opponent_deck_name=p2_deck_name,
        max_turns=max_play_steps,
        self_play=True,
        render_mode=None,
        cpp_engine_deck1=_cpp_deck1,
        cpp_engine_deck2=_cpp_deck2,
        cpp_engine_dir=cpp_engine_dir,
    )
    try:
        print(f"  Runtime backend (Phase 3): {_runtime_backend_label(probe_env)}")
        _announce_training_backend(probe_env, label="Phase 3 training")
        use_cpp_backend = bool(getattr(probe_env, "_using_cpp", False))
        train_runtime_backend = (
            "C++ engine" if use_cpp_backend else "HTTP Talishar"
        )
        if cpp_engine_dir and not use_cpp_backend:
            raise RuntimeError(
                f"C++ engine required (--cpp-engine-dir={cpp_engine_dir}) but "
                f"failed to load for Python {sys.version_info.major}.{sys.version_info.minor}. "
                "Rebuild with:\n"
                f"  python scripts/cpp/build_cpp_engine_for_matchup.py "
                f"--deck1 {_cpp_deck1} --deck2 {_cpp_deck2} "
                f"--deck1-json <p1.json> --deck2-json <p2.json> --no-server"
            )
    finally:
        probe_env.close()

    if n_workers is None:
        n_workers = auto_detect_workers(
            hero_id=p1_hero_id,
            p2_hero_id=p2_hero_id or p1_hero_id,
            cpp_engine_dir=cpp_engine_dir,
            assets_path=assets_path,
        )

    if use_cpp_backend:
        print(
            f"  Parallel play: {n_episodes} episodes, "
            f"{n_workers} worker(s)"
        )
    elif n_workers > 1:
        print(
            f"  WARNING: HTTP Talishar cannot run {n_workers} parallel game sessions — "
            "capping workers to 1."
        )
        n_workers = 1

    # ── Matchup + EpisodeCache ────────────────────────────────────────────────
    matchup = Matchup(
        name=f"p3_{p1_deck_name[-8:]}-vs-{p2_deck_name[-8:]}",
        p1_deck=p1_deck_name,
        p2_deck=p2_deck_name,
        description=f"Phase 3 play ({opponent_mode}): {p1.player} vs "
                    + (p2.player if p2 else p1_opponent_deck_name),
        p1_hero=p1_hero_id.replace("_", "-"),
        p2_hero=p2_hero_id.replace("_", "-"),
        cpp_engine_deck1=_cpp_deck1,
        cpp_engine_deck2=_cpp_deck2,
        cpp_engine_dir=cpp_engine_dir,
    )

    episode_cache = EpisodeCache(cache_root=cache_root, game_format=game_format)

    _ep_cache_info = episode_cache.info(p1_deck_name, p2_deck_name)
    print(f"  Agent cache: {cache_root / game_format}")
    print(
        f"  Episode cache: {_ep_cache_info['total_episodes']} stored episode(s) "
        f"(skip threshold: {episode_cache.warmup_skip_threshold})"
    )

    # ── create / reuse play agents (four-tier cache when bootstrapping) ───────
    p1_bundle = None
    p2_bundle = None
    p2_seed = (seed + 1) if seed is not None else None

    if p1.play is not None:
        p1_agent = p1.play
        p1_tiers: list[Any] = [p1_agent]
    else:
        def _make_p1() -> Any:
            return make_agent(seed=seed)

        p1_bundle = cache_store.bootstrap_player(
            _player_context(
                matchup,
                as_p1=True,
                p1_deck_fingerprint=p1_deck_fp,
                p2_deck_fingerprint=p2_deck_fp,
            ),
            _make_p1,
        )
        print("  P1 cache init:", ", ".join(p1_bundle.init_sources))
        p1_agent = p1_bundle.agents[0]
        p1_tiers = p1_bundle.agents

    if opponent_mode == "dual" and p2 is not None:
        if p2.play is not None:
            p2_agent = p2.play
            p2_tiers: list[Any] = [p2_agent]
        else:
            def _make_p2() -> Any:
                return make_agent(seed=p2_seed)

            p2_bundle = cache_store.bootstrap_player(
                _player_context(
                    matchup,
                    as_p1=False,
                    p1_deck_fingerprint=p1_deck_fp,
                    p2_deck_fingerprint=p2_deck_fp,
                ),
                _make_p2,
            )
            print("  P2 cache init:", ", ".join(p2_bundle.init_sources))
            p2_agent = p2_bundle.agents[0]
            p2_tiers = p2_bundle.agents
    else:
        p2_agent = make_agent(seed=p2_seed)
        p2_tiers = [p2_agent]

    live_path: Optional[Path] = out_dir / "play_live_state.png"
    out_dir.mkdir(parents=True, exist_ok=True)

    effective_checkpoint_interval = resolve_play_checkpoint_interval(
        n_episodes,
        checkpoint_interval=checkpoint_interval,
        checkpoint_interval_pct=checkpoint_interval_pct,
    )
    if checkpoint_interval is None:
        print(
            f"  Checkpoint interval: every {effective_checkpoint_interval} episode(s) "
            f"({checkpoint_interval_pct:g}% of {n_episodes})"
        )
    checkpoint_eval_log: list[dict[str, Any]] = []
    _ckpt_eval_lock = threading.Lock()

    def _snapshot_play_agent(agent: Any) -> Any:
        from agent_cache import clone_agent_weights  # noqa: PLC0415

        snap = make_agent(seed=seed)
        clone_agent_weights(agent, snap)
        return snap

    def _record_checkpoint_eval(completed: int) -> None:
        if (
            checkpoint_eval_episodes <= 0
            or not use_cpp_backend
            or opponent_mode != "preset"
        ):
            return

        eval_p1 = _snapshot_play_agent(p1_agent)
        eval_p2 = _snapshot_play_agent(p2_agent) if p2_agent is not None else None
        ckpt_dir = out_dir / matchup.name / "p1" / f"episode_{completed:06d}"

        def _run_eval() -> None:
            eval_metrics = _evaluate_p1_vs_fixed_opponent(
                matchup,
                eval_p1,
                p2_agent=eval_p2,
                base_url=base_url,
                game_format=game_format,
                max_steps=max_play_steps,
                episodes=checkpoint_eval_episodes,
                seed=(seed + completed) if seed is not None else None,
            )
            eval_record = {
                "episodes_completed": completed,
                "target_episodes": n_episodes,
                "eval_episodes": checkpoint_eval_episodes,
                "opponent_policy": "fixed_policy_sample",
                **eval_metrics,
            }
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "checkpoint_eval.json").write_text(
                json.dumps(eval_record, indent=2),
                encoding="utf-8",
            )
            with _ckpt_eval_lock:
                checkpoint_eval_log.append(eval_record)
                history_path = out_dir / "checkpoint_eval_history.json"
                history_path.write_text(
                    json.dumps(checkpoint_eval_log, indent=2),
                    encoding="utf-8",
                )
            if _parallel_seed_sync_dir is not None:
                sync_parallel_seed_dashboard_artifacts(_parallel_seed_sync_dir)
            print(
                f"  [p1] Checkpoint eval @ ep {completed}: "
                f"win%={eval_metrics['p1_win_rate']:.1%} "
                f"({eval_metrics['p1_wins']}W/"
                f"{eval_metrics['losses']}L/{eval_metrics['draws']}D/"
                f"{eval_metrics['timeouts']}T "
                f"over {checkpoint_eval_episodes} games, fixed sampled policies)"
            )
            _write_play_training_progress(completed)

        submit_checkpoint_eval(_run_eval, label=f"ep {completed}")

    def _after_play_checkpoint(completed: int) -> None:
        _save_phase3_play_checkpoints(
            out_dir=out_dir,
            matchup=matchup,
            game_format=game_format,
            p1_agent=p1_agent,
            p2_agent=p2_agent,
            p1_rewards=p1_rewards,
            p2_rewards=p2_rewards,
            episodes_completed=completed,
            total_target_episodes=n_episodes,
            opponent_mode=opponent_mode,
            p1_deck_cards=p1_game_deck,
            p2_deck_cards=p2_game_deck,
            p1_equipment_header=p1_equipment_header,
            p2_equipment_header=p2_equipment_header,
            p1_opponent_deck_name=p1_opponent_deck_name,
            p1_outcomes=p1_outcomes,
            runtime_backend=train_runtime_backend,
        )
        _record_checkpoint_eval(completed)
        _write_play_training_progress(completed)

    def _write_play_training_progress(completed: int) -> None:
        summary = summarize_p1_outcomes(
            p1_outcomes[:completed],
            episodes=completed,
        )
        eval_rates = [
            float(row.get("p1_win_rate", 0.0))
            for row in checkpoint_eval_log
            if row.get("p1_win_rate") is not None
        ]
        stability = compute_eval_stability(
            eval_rates,
            episodes_completed=completed,
            target_episodes=n_episodes,
        )
        point = {
            "episodes_completed": completed,
            "target_episodes": n_episodes,
            "updated_at": datetime.now().isoformat(),
            "runtime_backend": train_runtime_backend,
            **summary,
            "eval_stability": stability,
        }
        live_path = out_dir / "play_training_live.json"
        live_path.write_text(json.dumps(point, indent=2), encoding="utf-8")
        history_path = out_dir / "play_training_history.json"
        history: list[dict[str, Any]] = []
        if history_path.is_file():
            try:
                loaded = json.loads(history_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    history = [row for row in loaded if isinstance(row, dict)]
            except Exception:
                history = []
        if not history or int(history[-1].get("episodes_completed", -1)) != completed:
            history.append(point)
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if _parallel_seed_sync_dir is not None:
            sync_parallel_seed_dashboard_artifacts(_parallel_seed_sync_dir)

    # ── training ──────────────────────────────────────────────────────────────
    p1_rewards: list[float] = []
    p2_rewards: list[float] = []
    p1_outcomes: list[str] = []
    baseline_saved = False
    last_checkpoint_at = 0
    progress_t0 = datetime.now()

    def _write_play_training_live(
        completed: int,
        p1_r: list[float],
        p2_r: list[float],
        outcomes: Optional[list[str]] = None,
    ) -> None:
        nonlocal p1_outcomes
        if outcomes is not None:
            p1_outcomes = outcomes
        summary = summarize_p1_outcomes(
            p1_outcomes[:completed],
            episodes=completed,
        )
        elapsed = max((datetime.now() - progress_t0).total_seconds(), 1e-9)
        ep_rate = completed / elapsed
        eta_seconds = (
            (n_episodes - completed) / ep_rate
            if ep_rate > 0
            else float("inf")
        )
        point = {
            "episodes_completed": completed,
            "target_episodes": n_episodes,
            "updated_at": datetime.now().isoformat(),
            "runtime_backend": train_runtime_backend,
            "elapsed_seconds": elapsed,
            "episode_rate": ep_rate,
            "eta_seconds": eta_seconds,
            "warmup": completed < warmup_episodes,
            "p1_avg": float(np.mean(p1_r)) if p1_r else 0.0,
            "p2_avg": float(np.mean(p2_r)) if p2_r else 0.0,
            **summary,
        }
        live_path = out_dir / "play_training_live.json"
        live_path.write_text(json.dumps(point, indent=2), encoding="utf-8")
        if _parallel_seed_sync_dir is not None:
            sync_parallel_seed_dashboard_artifacts(_parallel_seed_sync_dir)

    def _on_episodes_progress(
        completed: int,
        p1_r: list[float],
        p2_r: list[float],
        outcomes: Optional[list[str]] = None,
    ) -> None:
        nonlocal baseline_saved, last_checkpoint_at, p1_rewards, p2_rewards, p1_outcomes
        p1_rewards = p1_r
        p2_rewards = p2_r
        _write_play_training_live(completed, p1_r, p2_r, outcomes)
        if (
            not baseline_saved
            and warmup_episodes > 0
            and warmup_baseline_eval_episodes > 0
            and completed >= warmup_episodes
        ):
            _run_warmup_baseline(
                matchup, p1_agent, p2_agent,
                base_url=base_url, game_format=game_format,
                max_steps=max_play_steps, out_dir=out_dir,
                episodes=warmup_baseline_eval_episodes, seed=seed,
            )
            baseline_saved = True
        if effective_checkpoint_interval <= 0 or completed <= 0:
            return
        if completed == n_episodes or completed >= last_checkpoint_at + effective_checkpoint_interval:
            last_checkpoint_at = completed
            _after_play_checkpoint(completed)

    try:
        if use_cpp_backend:
            p1_rewards, p2_rewards, train_stats = train_agents_from_both_perspectives_parallel(
                matchup=matchup,
                base_url=base_url,
                game_format=game_format,
                p1_tiers=p1_tiers,
                p2_tiers=p2_tiers,
                n_episodes=n_episodes,
                max_steps=max_play_steps,
                seed=seed,
                warmup_episodes=min(warmup_episodes, n_episodes),
                n_workers=n_workers,
                live_state_image_path=live_path,
                episode_cache=episode_cache,
                on_episodes_progress=_on_episodes_progress,
                suppress_train_progress=_suppress_train_progress,
            )
            p1_outcomes = list(train_stats.get("p1_outcomes") or p1_outcomes)
            if (
                not baseline_saved
                and warmup_episodes > 0
                and warmup_baseline_eval_episodes > 0
                and len(p1_rewards) >= warmup_episodes
            ):
                _run_warmup_baseline(
                    matchup, p1_agent, p2_agent,
                    base_url=base_url, game_format=game_format,
                    max_steps=max_play_steps, out_dir=out_dir,
                    episodes=warmup_baseline_eval_episodes, seed=seed,
                )
            if effective_checkpoint_interval > 0 and p1_rewards and len(p1_rewards) != last_checkpoint_at:
                _after_play_checkpoint(len(p1_rewards))
        else:
            # Serial HTTP fallback — chunked for checkpointing.
            env = make_env(
                matchup, base_url=base_url, game_format=game_format,
                max_turns=max_play_steps,
            )
            total_completed = 0
            warmup_remaining = min(warmup_episodes, n_episodes)
            try:
                while total_completed < n_episodes:
                    remaining = n_episodes - total_completed
                    chunk_size = remaining
                    if effective_checkpoint_interval > 0:
                        chunk_size = min(chunk_size, effective_checkpoint_interval)
                    chunk_warmup = min(warmup_remaining, chunk_size)
                    chunk_seed = (seed + total_completed) if seed is not None else None
                    if chunk_warmup == chunk_size:
                        print(
                            f"  Warmup chunk: {chunk_size} episode(s) "
                            f"starting at ep {total_completed + 1}…"
                        )
                    elif chunk_warmup > 0:
                        print(
                            f"  Mixed chunk: {chunk_size} episode(s) "
                            f"({chunk_warmup} warmup) starting at ep {total_completed + 1}…"
                        )
                    else:
                        print(
                            f"  PPO chunk: {chunk_size} episode(s) "
                            f"starting at ep {total_completed + 1}…"
                        )

                    c_p1, c_p2, chunk_stats = train_agents_from_both_perspectives(
                        env, p1_tiers, p2_tiers,
                        n_episodes=chunk_size,
                        max_steps=max_play_steps,
                        seed=chunk_seed,
                        warmup_episodes=chunk_warmup,
                        live_state_image_path=live_path,
                        episode_cache=episode_cache,
                        p1_deck=p1_deck_name,
                        p2_deck=p2_deck_name,
                        suppress_train_progress=_suppress_train_progress,
                    )
                    p1_rewards.extend(c_p1)
                    p2_rewards.extend(c_p2)
                    p1_outcomes.extend(chunk_stats.get("p1_outcomes") or [])
                    total_completed += chunk_size
                    warmup_remaining -= chunk_warmup

                    if (
                        not baseline_saved
                        and warmup_episodes > 0
                        and warmup_baseline_eval_episodes > 0
                        and warmup_remaining <= 0
                    ):
                        _run_warmup_baseline(
                            matchup, p1_agent, p2_agent,
                            base_url=base_url, game_format=game_format,
                            max_steps=max_play_steps, out_dir=out_dir,
                            episodes=warmup_baseline_eval_episodes, seed=seed,
                        )
                        baseline_saved = True
                    if effective_checkpoint_interval > 0 and total_completed % effective_checkpoint_interval == 0:
                        _after_play_checkpoint(total_completed)
            finally:
                env.close()

            if effective_checkpoint_interval > 0 and p1_rewards:
                _after_play_checkpoint(len(p1_rewards))

    finally:
        wait_for_checkpoint_evals()
        # Clean up temp deck files (retained for parallel-seed eval orchestration)
        if not _retain_temp_decks:
            for f in [p1_deck_file] + ([p2_deck_file] if p2_deck_file else []):
                try:
                    if f and f.exists():
                        f.unlink(missing_ok=True)
                except Exception:
                    pass

    # ── persist shared agent cache + update in-memory agents ─────────────────
    if p1_bundle is not None:
        cache_store.persist_player(p1_bundle)
    if p2_bundle is not None:
        cache_store.persist_player(p2_bundle)

    p1.play = p1_agent
    if opponent_mode == "dual" and p2 is not None:
        p2.play = p2_agent

    # ── win rates from episode outcomes (draws/timeouts count in denominator) ─
    train_summary = summarize_p1_outcomes(
        p1_outcomes,
        episodes=max(len(p1_outcomes), len(p1_rewards)),
    )
    p1_wr = float(train_summary["win_rate"])
    p2_wr = (
        sum(1 for r in p2_rewards if r > 0) / max(1, len(p2_rewards))
        if p2_rewards else 0.0
    )
    if _seed_run_capture is None:
        p1.win_rates.append(p1_wr)
        if opponent_mode == "dual" and p2 is not None:
            p2.win_rates.append(p2_wr)

    if _seed_run_capture is None:
        print(
            f"\n  Win rates: p1={p1_wr:.1%}  p2={p2_wr:.1%}  "
            f"({train_summary['wins']}W/{train_summary['losses']}L/"
            f"{train_summary['draws']}D/{train_summary['timeouts']}T)"
        )
    if checkpoint_eval_log:
        history_path = out_dir / "checkpoint_eval_history.json"
        history_path.write_text(
            json.dumps(checkpoint_eval_log, indent=2),
            encoding="utf-8",
        )
        print(f"  Checkpoint eval history → {history_path}")

    latest_ckpt_wr: Optional[float] = None
    if checkpoint_eval_log:
        latest_ckpt_wr = float(checkpoint_eval_log[-1].get("p1_win_rate", 0.0))

    if p1_bundle is not None and len(p1_rewards) >= n_episodes and not _skip_cache_converge:
        cache_store.mark_matchup_converged(
            p1_fingerprint=p1_deck_fp,
            p2_fingerprint=p2_deck_fp,
            p1_hero=p1_hero_id.replace("_", "-"),
            p2_hero=(p2_hero_id or p1_opponent_hero_id).replace("_", "-"),
            episodes_completed=len(p1_rewards),
            target_episodes=n_episodes,
            p1_win_rate=p1_wr,
            checkpoint_eval_win_rate=latest_ckpt_wr,
        )

    if _seed_run_capture is not None:
        temp_decks: list[str] = []
        for f in [p1_deck_file] + ([p2_deck_file] if p2_deck_file else []):
            if f is not None:
                temp_decks.append(str(f))
        p1_play_path = out_dir / "result_p1_play.json"
        p2_play_path = out_dir / "result_p2_play.json"
        p1_agent.save(p1_play_path)
        if p2_agent is not None:
            p2_agent.save(p2_play_path)
        _seed_run_capture.update(
            {
                "p1_agent": p1_agent,
                "p2_agent": p2_agent,
                "p1_play_path": str(p1_play_path),
                "p2_play_path": str(p2_play_path) if p2_agent is not None else None,
                "matchup": matchup,
                "use_cpp_backend": use_cpp_backend,
                "p1_deck_fingerprint": p1_deck_fp,
                "p2_deck_fingerprint": p2_deck_fp,
                "temp_deck_files": temp_decks,
            }
        )

    return p1_wr, p2_wr


def _agent_action_for_eval(agent: Any, observation: Any) -> Any:
    if hasattr(agent, "act"):
        return agent.act(observation)
    if hasattr(agent, "act_greedy"):
        return agent.act_greedy(observation)
    raise TypeError("play agent missing act/act_greedy")


def _softmax_logits(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(exp, 1.0 / max(1, exp.size), dtype=np.float64)
    return exp / total


def _fast_sample_action_index(
    agent: Any,
    obs_vec: np.ndarray,
    n_legal: int,
    rng: np.random.Generator,
) -> int:
    if getattr(agent, "_actor", None) is None:
        return 0
    logits = agent._actor.predict(obs_vec[None, :])
    logits = _mask_logits_to_legal(logits, max(1, int(n_legal)))
    probs = _softmax_logits(np.asarray(logits[0], dtype=np.float64))
    action = int(rng.choice(len(probs), p=probs))
    if action >= n_legal:
        action = max(0, n_legal - 1)
    return action


def _evaluate_fast_p1_vs_fixed_opponent(
    env: Any,
    p1_agent: Any,
    *,
    p2_agent: Any,
    max_steps: int,
    episodes: int,
    seed: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    if not _env_supports_fast_training(env):
        return None

    wins = losses = draws = timeouts = 0
    eval_rng = np.random.default_rng(seed)
    for ep in range(episodes):
        ep_seed = (seed + ep) if seed is not None else None
        state = env.fast_reset(seed=ep_seed, starting_player_id=1 + (ep % 2))
        terminated = truncated = False
        steps = 0
        p1_hp = int(state.get("p1_health", 0) or 0)
        p2_hp = int(state.get("p2_health", 0) or 0)
        p1_deck = int(state.get("p1_deck", 0) or 0)
        p2_deck = int(state.get("p2_deck", 0) or 0)

        while steps < max_steps:
            acting = int(state.get("acting_player_id", 1) or 1)
            agent = p1_agent if acting == 1 else p2_agent
            obs_vec = np.asarray(state["obs_vec"], dtype=np.float64)
            n_legal = max(1, int(state.get("legal_count", 1) or 1))
            action = _fast_sample_action_index(agent, obs_vec, n_legal, eval_rng)
            state = env.fast_step_index(action)
            terminated = bool(state.get("terminated", False))
            truncated = bool(state.get("truncated", False))
            steps += 1
            p1_hp = int(state.get("p1_health", p1_hp))
            p2_hp = int(state.get("p2_health", p2_hp))
            p1_deck = int(state.get("p1_deck", p1_deck))
            p2_deck = int(state.get("p2_deck", p2_deck))
            if terminated or truncated:
                break

        if not terminated and not truncated and steps >= max_steps:
            truncated = True

        outcome = classify_p1_episode_outcome(
            p1_hp=p1_hp,
            p2_hp=p2_hp,
            p1_deck=p1_deck,
            p2_deck=p2_deck,
            terminated=terminated,
            truncated=truncated,
        )
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        elif outcome == "draw":
            draws += 1
        else:
            timeouts += 1

    total = max(1, episodes)
    return {
        "episodes": episodes,
        "p1_wins": wins,
        "p2_wins": losses,
        "draws": draws,
        "timeouts": timeouts,
        "losses": losses,
        "p1_win_rate": wins / total,
        "p2_win_rate": losses / total,
        "draw_rate": draws / total,
        "timeout_rate": timeouts / total,
        "runtime_backend": "C++ engine fast sampled eval",
    }


def _evaluate_p1_vs_fixed_opponent(
    matchup: "Matchup",
    p1_agent: Any,
    *,
    p2_agent: Any,
    base_url: str,
    game_format: str,
    max_steps: int,
    episodes: int,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Evaluate frozen P1/P2 policies against each other (C++ when available)."""
    empty = {
        "episodes": 0,
        "p1_wins": 0,
        "p2_wins": 0,
        "draws": 0,
        "timeouts": 0,
        "losses": 0,
        "p1_win_rate": 0.0,
        "p2_win_rate": 0.0,
        "draw_rate": 0.0,
        "timeout_rate": 0.0,
    }
    if not _DUAL_AGENT_AVAILABLE or episodes <= 0:
        return empty

    env = make_env(
        matchup,
        base_url=base_url,
        game_format=game_format,
        max_turns=max_steps,
    )
    fast_metrics = _evaluate_fast_p1_vs_fixed_opponent(
        env,
        p1_agent,
        p2_agent=p2_agent,
        max_steps=max_steps,
        episodes=episodes,
        seed=seed,
    )
    if fast_metrics is not None:
        env.close()
        print("  Checkpoint eval backend: C++ engine fast sampled eval")
        return fast_metrics

    wins = 0
    losses = 0
    draws = 0
    timeouts = 0
    runtime_backend: Optional[str] = None
    backend_printed = False
    try:
        for ep in range(episodes):
            ep_seed = (seed + ep) if seed is not None else None
            result = env.reset(
                seed=ep_seed,
                options={"acting_player_id": 1 + (ep % 2)},
            )
            obs = result.observation
            if not backend_printed:
                runtime_backend = _runtime_backend_label(env)
                print(f"  Checkpoint eval backend: {runtime_backend}")
                backend_printed = True
            done = False
            steps = 0
            terminated = False
            truncated = False
            while not done and steps < max_steps:
                obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
                acting = int(obs_data.get("actingPlayerID", 1) or 1)
                if acting == 1:
                    action = _agent_action_for_eval(p1_agent, obs)
                else:
                    action = _agent_action_for_eval(p2_agent, obs)
                step = env.step(action)
                obs = step.observation
                terminated = bool(step.terminated)
                truncated = bool(step.truncated)
                done = terminated or truncated
                steps += 1

            obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
            p1_hp, p2_hp = absolute_p1_p2_hp_from_env(env)
            p1_deck, p2_deck = absolute_p1_p2_deck_from_env(env)
            if p1_hp is None or p2_hp is None:
                p1_hp_f, p2_hp_f = absolute_p1_p2_hp_from_obs(obs_data)
                p1_hp = int(p1_hp_f) if p1_hp_f is not None else None
                p2_hp = int(p2_hp_f) if p2_hp_f is not None else None
            if p1_deck is None or p2_deck is None:
                p1_deck, p2_deck = absolute_p1_p2_deck_from_obs(obs_data)
            outcome = classify_p1_episode_outcome(
                p1_hp=p1_hp,
                p2_hp=p2_hp,
                p1_deck=p1_deck,
                p2_deck=p2_deck,
                terminated=terminated,
                truncated=truncated,
            )
            if outcome == "win":
                wins += 1
            elif outcome == "loss":
                losses += 1
            elif outcome == "draw":
                draws += 1
            else:
                timeouts += 1
    finally:
        env.close()

    total = max(1, episodes)
    return {
        "episodes": episodes,
        "p1_wins": wins,
        "p2_wins": losses,
        "draws": draws,
        "timeouts": timeouts,
        "losses": losses,
        "p1_win_rate": wins / total,
        "p2_win_rate": losses / total,
        "draw_rate": draws / total,
        "timeout_rate": timeouts / total,
        "runtime_backend": runtime_backend or "HTTP Talishar",
    }


def _run_warmup_baseline(
    matchup: "Matchup",
    p1_agent: Any,
    p2_agent: Any,
    *,
    base_url: str,
    game_format: str,
    max_steps: int,
    out_dir: Path,
    episodes: int,
    seed: Optional[int],
) -> None:
    """Evaluate P1/P2 policies after warmup and save a handoff checkpoint."""
    print(f"  Warmup baseline eval: {episodes} episode(s)…")
    baseline = _evaluate_policy_pair(
        matchup,
        base_url=base_url,
        game_format=game_format,
        max_steps=max_steps,
        p1_policy=p1_agent,
        p2_policy=p2_agent,
        episodes=episodes,
        seed=(seed + 100_000) if seed is not None else None,
    )
    ckpt_dir = _save_warmup_handoff_checkpoint(
        out_dir=out_dir,
        matchup=matchup,
        p1_policy=p1_agent,
        p2_policy=p2_agent,
        baseline=baseline,
    )
    print(
        f"  Warmup baseline: P1 win%={baseline['p1_win_rate'] * 100:.1f}  "
        f"P2 win%={baseline['p2_win_rate'] * 100:.1f}  "
        f"draw%={baseline['draw_rate'] * 100:.1f}"
    )
    print(f"  Warmup checkpoint → {ckpt_dir}")


def _run_phase3_fallback(
    p1: PhaseAgents,
    p2: Optional[PhaseAgents],
    *,
    opponent_mode: str,
    game_format: str,
    p1_equipment_header: str,
    p2_equipment_header: str,
    p1_opponent_hero_id: str,
    p1_opponent_deck_name: str,
    n_episodes: int,
    max_play_steps: int,
    assets_path: str,
    base_url: str,
) -> tuple[float, float]:
    """Simple fallback play loop used when train_dual_agent_common is unavailable."""
    p1_game_deck = (
        p1.active_decks.get(p1_opponent_hero_id)
        or next(iter(p1.active_decks.values()), {})
        or p1.card_pool
    )
    if not p1_game_deck:
        return 0.0, 0.0

    if opponent_mode == "dual" and p2 is not None:
        p2_game_deck = next(iter(p2.active_decks.values()), {}) or p2.card_pool
    elif opponent_mode == "mirror":
        p2_game_deck = p1_game_deck
    else:
        p2_game_deck = None

    p1_deck_name = f"rl_fb_p1_{uuid.uuid4().hex[:8]}"
    p1_deck_file = _write_deck_file(p1_game_deck, p1_equipment_header, p1_deck_name, assets_path)
    if opponent_mode == "preset":
        p2_deck_name = p1_opponent_deck_name
        p2_deck_file = None
    elif opponent_mode == "mirror":
        p2_deck_name = p1_deck_name
        p2_deck_file = None
    else:
        p2_deck_name = f"rl_fb_p2_{uuid.uuid4().hex[:8]}"
        p2_deck_file = _write_deck_file(
            p2_game_deck, p2_equipment_header, p2_deck_name, assets_path  # type: ignore[arg-type]
        )

    p1_wins = 0
    backend_printed = False
    try:
        for ep in range(1, n_episodes + 1):
            env = TalisharEngineEnvironment(
                base_url=base_url,
                game_format=game_format,
                local_deck_name=p1_deck_name,
                opponent_deck_name=p2_deck_name,
                max_turns=max_play_steps,
                self_play=True,
            )
            try:
                if not backend_printed:
                    print(f"  Runtime backend (fallback play): {_runtime_backend_label(env)}")
                    backend_printed = True
                result = env.reset(options={"acting_player_id": 1 + ((ep - 1) % 2)})
                done = False
                while not done:
                    obs_data = json.loads(result.observation)
                    acting_player = obs_data.get("actingPlayerID", 1)
                    agent = (p1.play if acting_player == 1 else (p2.play if p2 else None))
                    if agent is not None and hasattr(agent, "act"):
                        action = agent.act(result.observation)
                    else:
                        action = env.sample_action()
                    step = env.step(action)
                    done = step.terminated or step.truncated
                    result.observation = step.observation

                obs_data = json.loads(result.observation)
                if obs_data.get("playerHealth", 0) > 0 and obs_data.get("opponentHealth", 0) <= 0:
                    p1_wins += 1
            finally:
                env.close()
    finally:
        for f in [p1_deck_file] + ([p2_deck_file] if p2_deck_file else []):
            try:
                if f and f.exists():
                    f.unlink(missing_ok=True)
            except Exception:
                pass

    p1_wr = p1_wins / max(1, n_episodes)
    p2_wr = 1.0 - p1_wr
    p1.win_rates.append(p1_wr)
    if opponent_mode == "dual" and p2 is not None:
        p2.win_rates.append(p2_wr)
    print(f"\n  Fallback win rates: p1={p1_wr:.1%}  p2={p2_wr:.1%}")
    return p1_wr, p2_wr


def _save_play_checkpoint_package(
    *,
    agent: Any,
    out_dir: Path,
    matchup: "Matchup",
    game_format: str,
    role: str,
    episodes_completed: int,
    total_target_episodes: int,
    reward_history: list[float],
    deck_cards: dict[str, int],
    equipment_header: str,
    opponent_mode: str,
    opponent_deck_name: str,
    p1_outcomes: Optional[list[str]] = None,
    runtime_backend: Optional[str] = None,
) -> Optional[Path]:
    """Persist a discoverable phase-3 checkpoint package under results/."""
    if not hasattr(agent, "save"):
        return None

    checkpoint_dir = out_dir / matchup.name / role / f"episode_{episodes_completed:06d}"
    weights_dir = checkpoint_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    try:
        agent.save(weights_dir / "agent_weights.json")
    except Exception as exc:
        print(f"  [{role}] WARNING: could not save play checkpoint at ep {episodes_completed}: {exc}")
        return None

    rewards = reward_history[:episodes_completed]
    avg_reward = float(sum(rewards) / len(rewards)) if rewards else 0.0
    if p1_outcomes is not None:
        outcome_stats = summarize_p1_outcomes(
            p1_outcomes[:episodes_completed],
            episodes=episodes_completed,
        )
    else:
        wins = sum(1 for r in rewards if r > 0)
        losses = sum(1 for r in rewards if r < 0)
        draws = len(rewards) - wins - losses
        total = max(1, len(rewards)) if rewards else 1
        outcome_stats = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "timeouts": 0,
            "win_rate": wins / total,
            "loss_rate": losses / total,
            "draw_rate": draws / total,
            "timeout_rate": 0.0,
        }
    metadata = {
        "checkpoint_type": "phase3_play",
        "created_at": datetime.now().isoformat(),
        "matchup": matchup.name,
        "role": role,
        "game_format": game_format,
        "weights_file": "agent_weights.json",
        "episodes_completed": episodes_completed,
        "target_episodes": total_target_episodes,
        "p1_deck": matchup.p1_deck,
        "p2_deck": matchup.p2_deck,
        "p1_hero": matchup.p1_hero,
        "p2_hero": matchup.p2_hero,
        "cpp_engine_deck1": matchup.cpp_engine_deck1,
        "cpp_engine_deck2": matchup.cpp_engine_deck2,
        "cpp_engine_dir": matchup.cpp_engine_dir,
        "runtime_backend": runtime_backend,
        "avg_reward": avg_reward,
        "win_rate": outcome_stats["win_rate"],
        "wins": outcome_stats["wins"],
        "losses": outcome_stats["losses"],
        "draws": outcome_stats["draws"],
        "timeouts": outcome_stats["timeouts"],
        "loss_rate": outcome_stats["loss_rate"],
        "draw_rate": outcome_stats["draw_rate"],
        "timeout_rate": outcome_stats["timeout_rate"],
        "opponent_mode": opponent_mode,
        "opponent_deck_name": opponent_deck_name,
        "deck_spec": {
            "equipment_header": equipment_header,
            "cards": deck_cards,
        },
    }
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return checkpoint_dir


def _ensure_hero_in_header(equipment_header: str, hero_id: str) -> str:
    """Guarantee ``hero_id`` is the first token of ``equipment_header``.

    Talishar's deck file parser requires the hero card ID as the very first
    token on line 1.  When it is absent the hero portrait is never loaded.
    ``hero_id`` uses underscores; convert dashes just in case.
    """
    hero = hero_id.replace("-", "_").strip()
    header = (equipment_header or "").strip()
    if hero and not header.startswith(hero):
        header = (hero + " " + header).strip()
    return header


def _save_phase3_play_checkpoints(
    *,
    out_dir: Path,
    matchup: "Matchup",
    game_format: str,
    p1_agent: Any,
    p2_agent: Any,
    p1_rewards: list[float],
    p2_rewards: list[float],
    episodes_completed: int,
    total_target_episodes: int,
    opponent_mode: str,
    p1_deck_cards: dict[str, int],
    p2_deck_cards: Optional[dict[str, int]],
    p1_equipment_header: str,
    p2_equipment_header: str,
    p1_opponent_deck_name: str,
    p1_outcomes: Optional[list[str]] = None,
    runtime_backend: Optional[str] = None,
) -> None:
    # Always store the hero ID as the first token so eval scripts can
    # reconstruct a valid deck file without needing external hero metadata.
    _p1_header = _ensure_hero_in_header(p1_equipment_header, matchup.p1_hero)
    _p2_header = _ensure_hero_in_header(p2_equipment_header, matchup.p2_hero)

    p1_ckpt = _save_play_checkpoint_package(
        agent=p1_agent,
        out_dir=out_dir,
        matchup=matchup,
        game_format=game_format,
        role="p1",
        episodes_completed=episodes_completed,
        total_target_episodes=total_target_episodes,
        reward_history=p1_rewards,
        deck_cards=p1_deck_cards,
        equipment_header=_p1_header,
        opponent_mode=opponent_mode,
        opponent_deck_name=p1_opponent_deck_name,
        p1_outcomes=p1_outcomes,
        runtime_backend=runtime_backend,
    )
    if p1_ckpt is not None:
        print(f"  [p1] Phase-3 checkpoint → {p1_ckpt}")

    p2_ckpt = _save_play_checkpoint_package(
        agent=p2_agent,
        out_dir=out_dir,
        matchup=matchup,
        game_format=game_format,
        role="p2",
        episodes_completed=episodes_completed,
        total_target_episodes=total_target_episodes,
        reward_history=p2_rewards,
        deck_cards=p2_deck_cards or {},
        equipment_header=_p2_header,
        opponent_mode=opponent_mode,
        opponent_deck_name=(matchup.p1_deck if opponent_mode == "dual" else p1_opponent_deck_name),
    )
    if p2_ckpt is not None:
        print(f"  [p2] Phase-3 checkpoint → {p2_ckpt}")


# ---------------------------------------------------------------------------
# Final evaluation — eval games + optimal-policy render + GIF
# ---------------------------------------------------------------------------


def _ensure_playwright() -> None:
    """Install Playwright + Chromium if not already available."""
    try:
        import playwright  # noqa: F401
    except ImportError:
        import subprocess, sys  # noqa: PLC0415
        print("  [render] Installing playwright Python package…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
    # Ensure Chromium browser binaries are present
    try:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415
        with sync_playwright() as pw:
            pw.chromium.launch(headless=True)  # will throw if binaries missing
    except Exception:
        import subprocess, sys  # noqa: PLC0415
        print("  [render] Installing Playwright Chromium browser…")
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])


def _prepare_render_dir(render_dir: Path) -> None:
    """Delete stale frames from a prior rollout before writing new ones."""
    import shutil  # noqa: PLC0415

    if render_dir.exists():
        shutil.rmtree(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)


def _render_game_with_talishar_frontend(
    *,
    agents: Any,
    opponent_agents: Optional[Any],
    opponent_mode: str,
    base_url: str,
    fe_url: str,
    game_format: str,
    deck_name: str,
    opp_name: str,
    max_steps: int,
    render_dir: Path,
    player_label: str,
) -> tuple[list[Path], str]:
    """Play one game via the HTTP Talishar backend and screenshot the live
    Talishar frontend after every step.

    Uses ``render_mode='rgb_array'`` on the environment so that
    ``TalisharEngineEnvironment`` manages its own Playwright browser thread —
    the same approach used (and confirmed working) in
    ``train_eval_render_pipeline.py``.  On ``reset()`` the engine navigates to
    the frontend with ``domcontentloaded`` + a 5-second settle wait, then
    queues a screenshot (with a 1.5 s render delay) on every ``env.render()``
    call, which gives equipment card art time to load.

    Returns:
        ``(frame_paths, outcome)`` where *outcome* is ``win`` / ``loss`` /
        ``draw`` / ``timeout`` from P1's perspective.  *frame_paths* includes a
        final annotated end-state frame when capture succeeds.
    """
    _ensure_playwright()
    _prepare_render_dir(render_dir)
    frame_paths: list[Path] = []
    outcome = "timeout"

    try:
        env = TalisharEngineEnvironment(
            base_url=base_url,
            frontend_url=fe_url,           # passed directly to env — no manual browser
            game_format=game_format,
            local_deck_name=deck_name,
            opponent_deck_name=opp_name,
            max_turns=max_steps,
            self_play=True,
            render_mode="rgb_array",       # engine owns the Playwright worker
            use_cpp_engine=False,          # HTTP backend required so FE can connect
            enable_combat_tracker=True,
        )
        try:
            result = env.reset()           # _open_playwright_page() runs here
            obs = result.observation

            # Frame 0 — board state after reset (equipment is visible here)
            frame_path = render_dir / "frame_0000_reset.png"
            if _save_state_image(env, obs, frame_path):
                frame_paths.append(frame_path)
                print(f"  [{player_label}] Frame 0 saved (reset)")

            done = False
            step_no = 0
            terminated = False
            truncated = False
            while not done and step_no < max_steps:
                step_no += 1

                obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
                acting = obs_data.get("actingPlayerID", 1)

                # Route action to the correct agent
                active_agents = agents
                if opponent_mode == "dual" and opponent_agents is not None and acting != 1:
                    active_agents = opponent_agents

                if active_agents.play is not None and hasattr(active_agents.play, "act_greedy"):
                    action = active_agents.play.act_greedy(obs)
                elif active_agents.play is not None and hasattr(active_agents.play, "act"):
                    action = active_agents.play.act(obs)
                else:
                    action = env.sample_action()

                step = env.step(action)
                obs = step.observation
                terminated = bool(step.terminated)
                truncated = bool(step.truncated)
                done = terminated or truncated

                fname = f"frame_{step_no:04d}_p{acting}.png"
                fpath = render_dir / fname
                if _save_state_image(env, obs, fpath):
                    frame_paths.append(fpath)

            outcome = _infer_render_outcome(
                obs, terminated=terminated, truncated=truncated, env=env,
            )
            end_path = render_dir / f"frame_{step_no + 1:04d}_end_{outcome}.png"
            if _save_end_state_frame(env, obs, end_path, outcome=outcome, steps=step_no):
                frame_paths.append(end_path)
                print(f"  [{player_label}] End frame saved ({outcome})")

        finally:
            env.close()

    except Exception as exc:
        print(f"  [{player_label}] Render error: {exc}")

    print(f"  [{player_label}] Saved {len(frame_paths)} frames → {render_dir}  ({outcome})")
    return frame_paths, outcome


def _frames_to_gif(frame_paths: list[Path], gif_path: Path, fps: float = 3.0) -> None:
    """Assemble PNG frame paths into an animated GIF (requires Pillow)."""
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        print("  WARNING: Pillow not installed — skipping GIF assembly.")
        print("           Install with: pip install Pillow")
        return
    frames: list[Any] = []
    for p in frame_paths:
        try:
            frames.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    if not frames:
        return
    duration_ms = max(1, int(1000.0 / fps))
    # Pause longer on the final end-state frame so the outcome banner is readable.
    end_hold_ms = max(duration_ms * 3, 2000)
    durations = [duration_ms] * len(frames)
    if len(durations) > 1:
        durations[-1] = end_hold_ms
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=durations,
    )
    print(f"  GIF saved ({len(frames)} frames, {fps} fps) → {gif_path}")


def _infer_render_outcome(
    obs: Any,
    *,
    terminated: bool,
    truncated: bool,
    env: Any = None,
) -> str:
    """Classify a rendered rollout as win/loss/draw/timeout from P1's perspective."""
    p1_hp = p2_hp = None
    p1_deck = p2_deck = None
    if env is not None:
        p1_hp, p2_hp = absolute_p1_p2_hp_from_env(env)
        p1_deck, p2_deck = absolute_p1_p2_deck_from_env(env)
    if p1_hp is None or p2_hp is None:
        p1_hp_f, p2_hp_f = absolute_p1_p2_hp_from_obs(obs)
        p1_hp = p1_hp_f
        p2_hp = p2_hp_f
    if p1_deck is None or p2_deck is None:
        p1_deck, p2_deck = absolute_p1_p2_deck_from_obs(obs)
    return classify_p1_episode_outcome(
        p1_hp=p1_hp,
        p2_hp=p2_hp,
        p1_deck=p1_deck,
        p2_deck=p2_deck,
        terminated=terminated,
        truncated=truncated,
    )


def _save_end_state_frame(
    env: Any,
    obs: Any,
    out_path: Path,
    *,
    outcome: str,
    steps: int = 0,
) -> bool:
    """Save a final board screenshot with a game-end outcome banner."""
    import base64  # noqa: PLC0415
    import io  # noqa: PLC0415

    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415
    except ImportError:
        return _save_state_image(env, obs, out_path)

    img = None
    try:
        rr = env.render()
        b64 = getattr(rr, "data", None)
        if b64:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception:
        pass

    if img is None:
        tmp = out_path.with_suffix(".tmp.png")
        if not _save_state_image(env, obs, tmp):
            return False
        try:
            img = Image.open(tmp).convert("RGB")
        finally:
            tmp.unlink(missing_ok=True)

    obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
    p1_hp = obs_data.get("playerHealth", "?")
    p2_hp = obs_data.get("opponentHealth", "?")

    labels: dict[str, tuple[str, tuple[int, int, int]]] = {
        "win": ("WIN", (34, 197, 94)),
        "loss": ("LOSS", (239, 68, 68)),
        "draw": ("DRAW", (250, 204, 21)),
        "timeout": ("TIMEOUT", (249, 115, 22)),
        "stall_timeout": ("STALL TIMEOUT", (249, 115, 22)),
    }
    label, color = labels.get(outcome, (outcome.upper().replace("_", " "), (200, 200, 200)))

    draw = ImageDraw.Draw(img)
    width, height = img.size
    banner_h = max(72, height // 8)
    draw.rectangle([(0, height - banner_h), (width, height)], fill=(16, 16, 16))
    try:
        title_font = ImageFont.truetype("arial.ttf", max(28, banner_h // 3))
        sub_font = ImageFont.truetype("arial.ttf", max(16, banner_h // 5))
    except Exception:
        title_font = ImageFont.load_default()
        sub_font = title_font

    draw.text((24, height - banner_h + 10), label, fill=color, font=title_font)
    draw.text(
        (24, height - banner_h + 44),
        f"P1 {p1_hp} HP  |  P2 {p2_hp} HP  |  {steps} steps",
        fill=(220, 220, 220),
        font=sub_font,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return True


def _save_state_image(env: Any, obs: Any, out_path: Path) -> bool:
    """Save one render frame.  Falls back to a text dump image if needed."""
    import base64  # noqa: PLC0415

    # Try rgb_array render (returns base64-encoded PNG via env.render())
    try:
        rr = env.render()
        b64 = getattr(rr, "data", None)
        if b64:
            out_path.write_bytes(base64.b64decode(b64))
            return True
    except Exception:
        pass

    # Text-based fallback using Pillow
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415
        obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
        lines = [
            f"Turn {obs_data.get('turnNo', '?')}  Phase {obs_data.get('turnPhase', '?')}",
            f"Acting player: {obs_data.get('actingPlayerID', '?')}",
            f"P1 HP: {obs_data.get('playerHealth', '?')}   "
            f"P2 HP: {obs_data.get('opponentHealth', '?')}",
            f"Legal actions: {len(obs_data.get('legalActions', []) or [])}",
            f"Prompt: {obs_data.get('prompt', '')}",
        ]
        font = ImageFont.load_default()
        img = Image.new("RGB", (800, 200), color=(18, 18, 18))
        draw = ImageDraw.Draw(img)
        for i, line in enumerate(lines):
            draw.text((10, 10 + i * 30), line, fill=(235, 235, 235), font=font)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        return True
    except Exception:
        return False


def _parse_obs_hp(obs: Any) -> tuple[int, int, int]:
    """Return ``(turn_no, p1_hp, p2_hp)`` from an observation."""
    obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
    turn_no = int(obs_data.get("turnNo", 0) or 0)
    p1_hp_f, p2_hp_f = absolute_p1_p2_hp_from_obs(obs_data)
    p1_hp = int(p1_hp_f or 0)
    p2_hp = int(p2_hp_f or 0)
    return turn_no, p1_hp, p2_hp


def _track_turn_hp(
    turn_hp: dict[int, tuple[int, int]],
    turn_no: int,
    p1_hp: int,
    p2_hp: int,
) -> None:
    """Record the latest HP snapshot seen for each turn number."""
    key = max(1, turn_no)
    turn_hp[key] = (p1_hp, p2_hp)


def _aggregate_hp_by_turn(
    trajectories: list[dict[int, tuple[int, int]]],
) -> list[dict[str, Any]]:
    """Compute mean/std HP per turn across evaluation episodes."""
    p1_by_turn: dict[int, list[int]] = defaultdict(list)
    p2_by_turn: dict[int, list[int]] = defaultdict(list)
    for traj in trajectories:
        for turn, (p1_hp, p2_hp) in traj.items():
            p1_by_turn[turn].append(p1_hp)
            p2_by_turn[turn].append(p2_hp)

    rows: list[dict[str, Any]] = []
    for turn in sorted(set(p1_by_turn) | set(p2_by_turn)):
        p1_vals = p1_by_turn[turn]
        p2_vals = p2_by_turn[turn]
        n = max(len(p1_vals), len(p2_vals))
        rows.append(
            {
                "turn": turn,
                "n": n,
                "p1_hp_mean": round(statistics.mean(p1_vals), 2) if p1_vals else 0.0,
                "p1_hp_std": round(
                    statistics.pstdev(p1_vals) if len(p1_vals) > 1 else 0.0, 2
                ),
                "p2_hp_mean": round(statistics.mean(p2_vals), 2) if p2_vals else 0.0,
                "p2_hp_std": round(
                    statistics.pstdev(p2_vals) if len(p2_vals) > 1 else 0.0, 2
                ),
            }
        )
    return rows


def _build_eval_outcome_summary(
    *,
    episodes: int,
    wins: int,
    losses: int,
    draws: int,
    timeouts: int = 0,
    episode_log: list[dict[str, Any]],
) -> dict[str, Any]:
    """Win/loss/draw/timeout summary plus simple episode aggregates."""
    total = max(1, episodes)
    steps = [int(ep.get("steps", 0) or 0) for ep in episode_log]
    final_p1 = [float(ep.get("p1_hp", 0) or 0) for ep in episode_log]
    final_p2 = [float(ep.get("p2_hp", 0) or 0) for ep in episode_log]
    return {
        "episodes": episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "timeouts": timeouts,
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "draw_rate": draws / total,
        "timeout_rate": timeouts / total,
        "win_pct": round(100.0 * wins / total, 1),
        "loss_pct": round(100.0 * losses / total, 1),
        "draw_pct": round(100.0 * draws / total, 1),
        "timeout_pct": round(100.0 * timeouts / total, 1),
        "avg_steps": round(statistics.mean(steps), 1) if steps else 0.0,
        "avg_final_player_hp": round(statistics.mean(final_p1), 2) if final_p1 else 0.0,
        "avg_final_opponent_hp": round(statistics.mean(final_p2), 2) if final_p2 else 0.0,
    }


def _plot_final_eval_hp_chart(
    *,
    hp_by_turn: list[dict[str, Any]],
    chart_path: Path,
    player_label: str,
    opponent_label: str,
    summary: dict[str, Any],
) -> bool:
    """Plot average HP per turn (±1 std) for player vs opponent."""
    if not hp_by_turn:
        return False
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return False

    turns = [int(row["turn"]) for row in hp_by_turn]
    p1_mean = [float(row["p1_hp_mean"]) for row in hp_by_turn]
    p1_std = [float(row["p1_hp_std"]) for row in hp_by_turn]
    p2_mean = [float(row["p2_hp_mean"]) for row in hp_by_turn]
    p2_std = [float(row["p2_hp_std"]) for row in hp_by_turn]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(turns, p1_mean, marker="o", linewidth=2, label=player_label, color="#22c55e")
    ax.fill_between(
        turns,
        [m - s for m, s in zip(p1_mean, p1_std)],
        [m + s for m, s in zip(p1_mean, p1_std)],
        alpha=0.2,
        color="#22c55e",
    )
    ax.plot(turns, p2_mean, marker="s", linewidth=2, label=opponent_label, color="#ef4444")
    ax.fill_between(
        turns,
        [m - s for m, s in zip(p2_mean, p2_std)],
        [m + s for m, s in zip(p2_mean, p2_std)],
        alpha=0.2,
        color="#ef4444",
    )

    ax.set_xlabel("Turn")
    ax.set_ylabel("Hit points (avg ± std)")
    ax.set_title(
        f"Final eval HP by turn  ·  "
        f"Win {summary.get('win_pct', 0):.1f}%  "
        f"Loss {summary.get('loss_pct', 0):.1f}%  "
        f"Draw {summary.get('draw_pct', 0):.1f}%  "
        f"({summary.get('wins', 0)}W/"
        f"{summary.get('losses', 0)}L/"
        f"{summary.get('draws', 0)}D)"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(chart_path), dpi=120)
    plt.close(fig)
    return True


def _deck_sheet_card_cell(card: dict[str, Any]) -> str:
    """One card tile for the matchup deck sheet HTML."""
    name = html.escape(str(card.get("name") or card.get("id") or ""))
    count = int(card.get("count") or 1)
    image_url = html.escape(str(card.get("image_url") or ""))
    count_badge = f'<span class="count">×{count}</span>' if count > 1 else ""
    pitch = card.get("pitch")
    pitch_label = f'<span class="pitch">P{pitch}</span>' if pitch is not None else ""
    return (
        f'<div class="card">'
        f'<img src="{image_url}" alt="{name}" loading="lazy" '
        f'onerror="this.classList.add(\'missing\');" />'
        f'<div class="label">{name}{pitch_label}{count_badge}</div>'
        f"</div>"
    )


def _build_matchup_deck_sheet_html(
    deck_export: dict[str, Any],
    *,
    eval_summary: Optional[dict[str, Any]] = None,
) -> str:
    """HTML page that loads Talishar WebP card art for deck + sideboard."""
    hero = html.escape(str(deck_export.get("hero_id") or ""))
    opponent = html.escape(str(deck_export.get("opponent_hero_id") or ""))
    fmt = html.escape(str(deck_export.get("format") or ""))
    matchup = html.escape(str(deck_export.get("matchup") or f"{hero} vs {opponent}"))

    game_deck = deck_export.get("game_deck") or {}
    sideboard = deck_export.get("sideboard") or {}
    equipment = deck_export.get("equipment") or []
    deck_cards = game_deck.get("cards") or []
    sb_cards = sideboard.get("cards") or []

    eval_line = ""
    if eval_summary:
        eval_line = (
            f"Win {eval_summary.get('win_pct', 0):.1f}% · "
            f"Loss {eval_summary.get('loss_pct', 0):.1f}% · "
            f"Draw {eval_summary.get('draw_pct', 0):.1f}%"
        )

    equip_html = "".join(_deck_sheet_card_cell(card) for card in equipment)
    deck_html = "".join(_deck_sheet_card_cell(card) for card in deck_cards)
    sb_html = "".join(_deck_sheet_card_cell(card) for card in sb_cards)

    deck_total = int(game_deck.get("total_cards") or deck_export.get("game_deck_size") or 0)
    sb_total = int(sideboard.get("total_cards") or deck_export.get("sideboard_size") or 0)
    pool_total = int(deck_export.get("pool_size") or 0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{matchup} deck sheet</title>
<style>
  body {{
    margin: 0;
    padding: 20px 24px 32px;
    background: #111827;
    color: #f3f4f6;
    font-family: "Segoe UI", system-ui, sans-serif;
  }}
  h1 {{ margin: 0 0 4px; font-size: 1.6rem; }}
  .meta {{ color: #9ca3af; margin-bottom: 18px; font-size: 0.95rem; }}
  .zones {{ display: flex; gap: 20px; align-items: flex-start; }}
  .zone {{
    flex: 1;
    background: #1f2937;
    border-radius: 12px;
    padding: 14px;
    min-width: 0;
  }}
  .zone h2 {{
    margin: 0 0 12px;
    font-size: 1.05rem;
    color: #e5e7eb;
  }}
  .equipment {{
    margin-bottom: 18px;
    background: #1f2937;
    border-radius: 12px;
    padding: 14px;
  }}
  .equipment h2 {{ margin: 0 0 10px; font-size: 1rem; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(92px, 1fr));
    gap: 10px;
  }}
  .card {{
    position: relative;
    text-align: center;
  }}
  .card img {{
    width: 100%;
    aspect-ratio: 5 / 7;
    object-fit: cover;
    border-radius: 8px;
    background: #374151;
    display: block;
    box-shadow: 0 2px 8px rgba(0,0,0,0.35);
  }}
  .card img.missing {{ display: none; }}
  .label {{
    margin-top: 4px;
    font-size: 0.62rem;
    line-height: 1.2;
    color: #d1d5db;
    word-break: break-word;
  }}
  .count {{
    display: inline-block;
    margin-left: 4px;
    padding: 0 4px;
    border-radius: 4px;
    background: #111827;
    color: #fbbf24;
    font-weight: 700;
  }}
  .pitch {{
    display: inline-block;
    margin-left: 3px;
    color: #93c5fd;
  }}
</style>
</head>
<body>
  <h1>{matchup}</h1>
  <div class="meta">
    {hero} vs {opponent} · {fmt} · pool {pool_total} cards
    {f" · {html.escape(eval_line)}" if eval_line else ""}
  </div>
  {"<section class='equipment'><h2>Equipment</h2><div class='grid'>" + equip_html + "</div></section>" if equip_html else ""}
  <div class="zones">
    <section class="zone">
      <h2>Game deck ({deck_total})</h2>
      <div class="grid">{deck_html}</div>
    </section>
    <section class="zone">
      <h2>Sideboard ({sb_total})</h2>
      <div class="grid">{sb_html}</div>
    </section>
  </div>
</body>
</html>"""


def _render_matchup_deck_sheet(
    deck_export: dict[str, Any],
    *,
    out_path: Path,
    html_path: Optional[Path] = None,
    eval_summary: Optional[dict[str, Any]] = None,
) -> bool:
    """Screenshot a deck/sideboard sheet using Talishar card WebP images."""
    page_html = _build_matchup_deck_sheet_html(deck_export, eval_summary=eval_summary)
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(page_html, encoding="utf-8")

    try:
        _ensure_playwright()
        from playwright.sync_api import sync_playwright  # noqa: PLC0415
    except Exception:
        return False

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            page.set_content(page_html, wait_until="domcontentloaded")
            page.wait_for_timeout(2500)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(out_path), full_page=True)
            browser.close()
        return out_path.is_file()
    except Exception:
        return False


def _export_matchup_deck_artifacts(
    agents: PhaseAgents,
    *,
    hero_id: str,
    opponent_hero_id: str,
    game_format: str,
    equipment_header: str,
    game_deck: dict[str, int],
    image_base_url: str,
    out_dir: Path,
    player: str,
    eval_summary: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Write matchup deck JSON + Talishar card-art sheet image."""
    deck_export = build_matchup_deck_export(
        agents,
        hero_id=hero_id,
        opponent_hero_id=opponent_hero_id,
        game_format=game_format,
        equipment_header=equipment_header,
        game_deck=game_deck,
        image_base_url=image_base_url,
    )
    if eval_summary:
        deck_export["eval_summary"] = {
            k: eval_summary[k]
            for k in (
                "episodes", "wins", "losses", "draws",
                "win_pct", "loss_pct", "draw_pct", "win_rate",
            )
            if k in eval_summary
        }

    json_path = out_dir / f"{player}_matchup_deck.json"
    image_path = out_dir / f"{player}_matchup_deck.png"
    html_path = out_dir / f"{player}_matchup_deck.html"

    json_path.write_text(json.dumps(deck_export, indent=2), encoding="utf-8")
    image_ok = _render_matchup_deck_sheet(
        deck_export,
        out_path=image_path,
        html_path=html_path,
        eval_summary=eval_summary,
    )

    print(f"  [{player}] Matchup deck JSON  → {json_path}")
    if image_ok:
        print(f"  [{player}] Matchup deck sheet → {image_path}")
    else:
        print(f"  [{player}] Matchup deck sheet skipped (Playwright/images unavailable)")
        if html_path.is_file():
            print(f"  [{player}] Matchup deck HTML   → {html_path}")

    return {
        "json": str(json_path),
        "image": str(image_path) if image_ok else None,
        "html": str(html_path) if html_path.is_file() else None,
        "game_deck_size": deck_export.get("game_deck_size"),
        "sideboard_size": deck_export.get("sideboard_size"),
        "pool_size": deck_export.get("pool_size"),
    }


def _print_final_eval_analysis(
    player: str,
    *,
    hero_id: str,
    opponent_hero_id: str,
    summary: dict[str, Any],
    hp_by_turn: list[dict[str, Any]],
    chart_path: Optional[Path],
    matchup_deck: Optional[dict[str, Any]] = None,
) -> None:
    """Print a concise final-eval analysis block to the console."""
    print(f"\n  [{player}] Final eval analysis")
    print(
        f"    Win rate   : {summary['win_pct']:.1f}%  "
        f"({summary['wins']}W / {summary['losses']}L / {summary['draws']}D "
        f"over {summary['episodes']} games)"
    )
    print(
        f"    Loss rate  : {summary['loss_pct']:.1f}%   "
        f"Draw rate : {summary['draw_pct']:.1f}%"
    )
    print(
        f"    Avg steps  : {summary['avg_steps']:.1f}   "
        f"Final HP  {hero_id}: {summary['avg_final_player_hp']:.1f}  "
        f"{opponent_hero_id}: {summary['avg_final_opponent_hp']:.1f}"
    )
    if hp_by_turn:
        sample = hp_by_turn[:5]
        print(f"    HP by turn : {len(hp_by_turn)} turn(s) tracked")
        for row in sample:
            print(
                f"      T{row['turn']:>2}  "
                f"{hero_id}: {row['p1_hp_mean']:.1f}±{row['p1_hp_std']:.1f}  "
                f"{opponent_hero_id}: {row['p2_hp_mean']:.1f}±{row['p2_hp_std']:.1f}  "
                f"(n={row['n']})"
            )
        if len(hp_by_turn) > 8:
            print("      …")
            for row in hp_by_turn[-3:]:
                print(
                    f"      T{row['turn']:>2}  "
                    f"{hero_id}: {row['p1_hp_mean']:.1f}±{row['p1_hp_std']:.1f}  "
                    f"{opponent_hero_id}: {row['p2_hp_mean']:.1f}±{row['p2_hp_std']:.1f}  "
                    f"(n={row['n']})"
                )
    if chart_path is not None:
        print(f"    HP chart   : {chart_path}")
    if matchup_deck:
        if matchup_deck.get("json"):
            print(
                f"    Matchup deck : {matchup_deck['game_deck_size']} main / "
                f"{matchup_deck['sideboard_size']} sideboard"
            )
            print(f"    Deck JSON    : {matchup_deck['json']}")
        if matchup_deck.get("image"):
            print(f"    Deck sheet   : {matchup_deck['image']}")


def run_final_evaluation(
    agents: PhaseAgents,
    opponent_agents: Optional[PhaseAgents],
    *,
    hero_id: str,
    equipment_header: str,
    opponent_equipment_header: str = "",
    game_format: str,
    opponent_deck_name: str,
    opponent_hero_id: str,
    opponent_mode: str,
    num_eval_episodes: int,
    max_steps: int,
    assets_path: str,
    base_url: str,
    fe_url: str,
    out_dir: Path,
    render_gif: bool = False,
    gif_fps: float = 3.0,
) -> dict[str, Any]:
    """Full final evaluation for one player's pipeline.

    Steps
    -----
    1. Select the best sidebaorded game deck (or fall back to card pool).
    2. Write a temporary deck file to Talishar Assets.
    3. Run ``num_eval_episodes`` games with the trained sampled play policy,
       recording win / loss / draw for each episode.
    4. Optionally render one full rollout via the Talishar frontend
       (Playwright screenshots + GIF) when ``render_gif=True``.
    5. Write ``final_eval.json`` to ``out_dir``.

    Returns the summary dict.
    """
    player = agents.player
    print(
        f"\n{'='*62}\n"
        f"  FINAL EVALUATION  [{player} / {hero_id}]\n"
        f"{'='*62}"
    )

    # ── select game deck ──────────────────────────────────────────────────────
    game_deck = (
        agents.active_decks.get(opponent_hero_id)
        or next(iter(agents.active_decks.values()), {})
        or agents.card_pool
    )
    if not game_deck:
        print(f"  [{player}] No game deck available — skipping final eval")
        return {"skipped": True, "reason": "no_game_deck"}

    deck_total = sum(game_deck.values())
    print(f"  [{player}] Game deck: {deck_total} cards  (vs {opponent_hero_id})")

    # ── write deck file ───────────────────────────────────────────────────────
    deck_name = f"rl_final_{player}_{uuid.uuid4().hex[:8]}"
    deck_file = _write_deck_file(game_deck, equipment_header, deck_name, assets_path)

    # For dual mode, use the opponent's sidebaorded deck; otherwise use preset.
    if opponent_mode == "dual" and opponent_agents is not None:
        opp_deck = (
            opponent_agents.active_decks.get(hero_id)
            or next(iter(opponent_agents.active_decks.values()), {})
            or opponent_agents.card_pool
        )
        opp_name = f"rl_final_{opponent_agents.player}_{uuid.uuid4().hex[:8]}"
        # Resolve the opponent equipment header: explicit param > PhaseAgents
        # field > fall back to P1's header so the deck file is always valid.
        _opp_equip = (
            opponent_equipment_header
            or getattr(opponent_agents, "equipment_header", "")
            or equipment_header
        )
        opp_file = _write_deck_file(
            opp_deck,
            _opp_equip,
            opp_name,
            assets_path,
        )
    else:
        opp_name = normalize_talishar_asset_name(opponent_deck_name, assets_path)
        opp_file = None

    # ── evaluation games ──────────────────────────────────────────────────────
    wins = 0
    losses = 0
    draws = 0
    timeouts = 0
    episode_log: list[dict[str, Any]] = []
    turn_trajectories: list[dict[int, tuple[int, int]]] = []

    progress_t0 = datetime.now()

    def _write_final_eval_live(
        *,
        completed: int,
        phase: str = "episodes",
    ) -> None:
        elapsed = max((datetime.now() - progress_t0).total_seconds(), 1e-9)
        episode_rate = completed / elapsed if completed > 0 else 0.0
        remaining_eps = max(0, num_eval_episodes - completed)
        eta_seconds: Optional[float] = None
        render_eta_seconds: Optional[float] = None
        if phase == "episodes" and episode_rate > 0:
            eta_seconds = remaining_eps / episode_rate
        elif phase == "render":
            render_eta_seconds = 180.0
        payload = {
            "episodes_completed": completed,
            "target_episodes": num_eval_episodes,
            "phase": phase,
            "updated_at": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "episode_rate": episode_rate,
            "eta_seconds": eta_seconds,
            "render_eta_seconds": render_eta_seconds,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "timeouts": timeouts,
            "runtime_backend": "HTTP Talishar",
        }
        live_path = out_dir / "final_eval_live.json"
        live_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    backend_printed = False
    try:
        for ep in range(1, num_eval_episodes + 1):
            env = TalisharEngineEnvironment(
                base_url=base_url,
                game_format=game_format,
                local_deck_name=deck_name,
                opponent_deck_name=opp_name,
                max_turns=max_steps,
                self_play=True,
                render_mode=None,
                enable_combat_tracker=True,
            )
            try:
                if not backend_printed:
                    print(f"  [{player}] Runtime backend (final eval): {_runtime_backend_label(env)}")
                    backend_printed = True
                result = env.reset(options={"acting_player_id": 1 + ((ep - 1) % 2)})
                obs = result.observation
                turn_hp: dict[int, tuple[int, int]] = {}
                t0, p1_hp0, p2_hp0 = _parse_obs_hp(obs)
                _track_turn_hp(turn_hp, t0, p1_hp0, p2_hp0)
                done = False
                steps = 0
                terminated = False
                truncated = False
                while not done:
                    if agents.play is not None and hasattr(agents.play, "act"):
                        action = agents.play.act(obs)
                    elif agents.play is not None and hasattr(agents.play, "act_greedy"):
                        action = agents.play.act_greedy(obs)
                    else:
                        action = env.sample_action()
                    step = env.step(action)
                    obs = step.observation
                    terminated = bool(step.terminated)
                    truncated = bool(step.truncated)
                    done = terminated or truncated
                    steps += 1
                    turn_no, p1_hp, p2_hp = _parse_obs_hp(obs)
                    _track_turn_hp(turn_hp, turn_no, p1_hp, p2_hp)

                obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
                p1_hp, p2_hp = absolute_p1_p2_hp_from_env(env)
                p1_deck, p2_deck = absolute_p1_p2_deck_from_env(env)
                if p1_hp is None or p2_hp is None:
                    p1_hp_f, p2_hp_f = absolute_p1_p2_hp_from_obs(obs_data)
                    p1_hp = int(p1_hp_f) if p1_hp_f is not None else None
                    p2_hp = int(p2_hp_f) if p2_hp_f is not None else None
                if p1_deck is None or p2_deck is None:
                    p1_deck, p2_deck = absolute_p1_p2_deck_from_obs(obs_data)
                outcome = classify_p1_episode_outcome(
                    p1_hp=p1_hp,
                    p2_hp=p2_hp,
                    p1_deck=p1_deck,
                    p2_deck=p2_deck,
                    terminated=terminated,
                    truncated=truncated and not terminated,
                )
                if outcome == "win":
                    wins += 1
                elif outcome == "loss":
                    losses += 1
                elif outcome == "draw":
                    draws += 1
                else:
                    timeouts += 1

                turn_trajectories.append(turn_hp)
                episode_log.append({
                    "episode": ep,
                    "outcome": outcome,
                    "steps": steps,
                    "p1_hp": p1_hp,
                    "p2_hp": p2_hp,
                    "turns_played": max(turn_hp) if turn_hp else 0,
                })
                wr = wins / ep
                print(
                    f"  [{player}] Ep {ep:>3}/{num_eval_episodes}  "
                    f"{outcome:<4}  steps={steps:3d}  win_rate={wr:.1%}"
                )
                _write_final_eval_live(completed=ep, phase="episodes")
            finally:
                env.close()

    except Exception as exc:
        print(f"  [{player}] Eval error: {exc}")

    total = max(1, num_eval_episodes)
    win_rate = wins / total
    print(
        f"\n  [{player}] Final win rate: {win_rate:.1%}  "
        f"({wins}W / {losses}L / {draws}D / {timeouts}T  over {num_eval_episodes} games)"
    )

    outcome_summary = _build_eval_outcome_summary(
        episodes=num_eval_episodes,
        wins=wins,
        losses=losses,
        draws=draws,
        timeouts=timeouts,
        episode_log=episode_log,
    )
    hp_by_turn = _aggregate_hp_by_turn(turn_trajectories)
    player_label = hero_id.replace("_", " ").title()
    opponent_label = opponent_hero_id.replace("_", " ").title()
    hp_chart_path = out_dir / f"{player}_final_eval_hp_by_turn.png"
    chart_ok = _plot_final_eval_hp_chart(
        hp_by_turn=hp_by_turn,
        chart_path=hp_chart_path,
        player_label=player_label,
        opponent_label=opponent_label,
        summary=outcome_summary,
    )
    matchup_deck_artifacts = _export_matchup_deck_artifacts(
        agents,
        hero_id=hero_id,
        opponent_hero_id=opponent_hero_id,
        game_format=game_format,
        equipment_header=equipment_header,
        game_deck=game_deck,
        image_base_url=base_url,
        out_dir=out_dir,
        player=player,
        eval_summary=outcome_summary,
    )
    _print_final_eval_analysis(
        player,
        hero_id=hero_id,
        opponent_hero_id=opponent_hero_id,
        summary=outcome_summary,
        hp_by_turn=hp_by_turn,
        chart_path=hp_chart_path if chart_ok else None,
        matchup_deck=matchup_deck_artifacts,
    )

    # ── optional render rollout (Playwright + Talishar frontend) ─────────────
    render_dir = out_dir / f"{player}_final_render"
    gif_path = out_dir / f"{player}_optimal_policy.gif"
    frame_paths: list[Path] = []
    render_outcome = "skipped"
    render_steps = 0

    if render_gif:
        print(f"\n  [{player}] Rendering optimal-policy rollout via Talishar FE → {render_dir}")
        _write_final_eval_live(completed=num_eval_episodes, phase="render")
        frame_paths, render_outcome = _render_game_with_talishar_frontend(
            agents=agents,
            opponent_agents=opponent_agents,
            opponent_mode=opponent_mode,
            base_url=base_url,
            fe_url=fe_url,
            game_format=game_format,
            deck_name=deck_name,
            opp_name=opp_name,
            max_steps=max_steps,
            render_dir=render_dir,
            player_label=player,
        )
        render_steps = max(0, len(frame_paths) - 1)
        if frame_paths:
            _frames_to_gif(frame_paths, gif_path, fps=gif_fps)
    else:
        print(
            f"\n  [{player}] Skipping rollout GIF render "
            f"(use --render-gif or eval_phase3_checkpoint.py --render-only to generate later)"
        )

    # ── write final_eval.json ─────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "player": player,
        "hero_id": hero_id,
        "opponent_hero_id": opponent_hero_id,
        "opponent_mode": opponent_mode,
        "format": game_format,
        "game_deck_size": deck_total,
        "eval": {
            "episodes": num_eval_episodes,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "loss_rate": outcome_summary["loss_rate"],
            "draw_rate": outcome_summary["draw_rate"],
            "win_pct": outcome_summary["win_pct"],
            "loss_pct": outcome_summary["loss_pct"],
            "draw_pct": outcome_summary["draw_pct"],
            "avg_steps": outcome_summary["avg_steps"],
            "avg_final_player_hp": outcome_summary["avg_final_player_hp"],
            "avg_final_opponent_hp": outcome_summary["avg_final_opponent_hp"],
            "episode_log": episode_log,
        },
        "analysis": {
            "summary": outcome_summary,
            "hp_by_turn": hp_by_turn,
            "charts": {
                "hp_by_turn": str(hp_chart_path) if chart_ok else None,
            },
        },
        "matchup_deck": matchup_deck_artifacts,
        "render": {
            "frames_dir": str(render_dir),
            "frames_saved": len(frame_paths),
            "steps": render_steps,
            "outcome": render_outcome,
            "terminated": render_outcome in ("win", "loss", "draw"),
            "truncated": render_outcome == "timeout",
            "gif": str(gif_path) if (render_gif and frame_paths) else None,
        },
    }
    eval_json_path = out_dir / f"{player}_final_eval.json"
    eval_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  [{player}] Final eval written → {eval_json_path}")

    # ── cleanup temp deck files ───────────────────────────────────────────────
    for f in [deck_file] + ([opp_file] if opp_file else []):
        try:
            if f.exists():
                f.unlink()
        except Exception:
            pass

    return summary


def auto_detect_workers(
    *,
    hero_id: str,
    p2_hero_id: str,
    cpp_engine_dir: Optional[str],
    assets_path: str,
) -> int:
    """Return parallel worker count for play training (C++ vs HTTP)."""
    explicit = os.environ.get("FAB_PLAY_WORKERS")
    if explicit:
        try:
            workers = max(1, int(explicit))
            print(f"  [auto] FAB_PLAY_WORKERS={workers}")
            return workers
        except ValueError:
            print(f"  [auto] Ignoring invalid FAB_PLAY_WORKERS={explicit!r}")

    max_cap = max(1, int(os.environ.get("FAB_MAX_PLAY_WORKERS", "32")))

    if cpp_engine_dir:
        try:
            if str(REPO_ROOT / "src") not in sys.path:
                sys.path.insert(0, str(REPO_ROOT / "src"))
            from flesh_and_blood_rlbridge.cpp_engine_environment import (  # noqa: PLC0415
                is_cpp_engine_available,
                load_fab_engine,
            )
            if is_cpp_engine_available(cpp_engine_dir):
                load_fab_engine(cpp_engine_dir)
                workers = min(max(1, os.cpu_count() or 4), max_cap)
                print(f"  [auto] C++ engine ({cpp_engine_dir}) -> {workers} workers")
                return workers
        except Exception:
            pass

    lookup1, lookup2 = resolve_cpp_lookup_decks(assets_path, hero_id, p2_hero_id)
    _cpp_cache = os.path.join(str(REPO_ROOT), "results", "cpp_engines")
    _cpp_key = f"{lookup1}_vs_{lookup2}"

    if cpp_engine_dir and os.path.isdir(cpp_engine_dir):
        _cpp_dir = cpp_engine_dir
    else:
        _exact = os.path.join(_cpp_cache, _cpp_key)
        if os.path.isdir(_exact):
            _cpp_dir = _exact
        else:
            _candidates = sorted(
                glob.glob(os.path.join(_cpp_cache, f"{_cpp_key}-*")),
                key=os.path.getmtime,
                reverse=True,
            )
            _cpp_dir = _candidates[0] if _candidates else _exact

    _has_cpp = False
    if os.path.isdir(_cpp_dir):
        try:
            if str(REPO_ROOT / "src") not in sys.path:
                sys.path.insert(0, str(REPO_ROOT / "src"))
            from flesh_and_blood_rlbridge.cpp_engine_environment import (  # noqa: PLC0415
                is_cpp_engine_available,
                load_fab_engine,
            )
            if is_cpp_engine_available(_cpp_dir):
                load_fab_engine(_cpp_dir)
                _has_cpp = True
        except Exception:
            _has_cpp = False

    if _has_cpp:
        workers = min(max(1, os.cpu_count() or 4), max_cap)
        print(f"  [auto] C++ engine detected -> {workers} parallel workers")
        return workers
    print("  [auto] No C++ engine found -> 1 worker (HTTP Talishar)")
    return 1


def _seed_agents_from_starting_decks(
    p1: PhaseAgents,
    p2: Optional[PhaseAgents],
    *,
    p1_starting_deck: Optional[dict[str, int]],
    p2_starting_deck: Optional[dict[str, int]],
    min_size: int,
    p1_opponent_key: str,
    p2_opponent_key: str,
) -> None:
    if p1_starting_deck and not p1.card_pool:
        p1.card_pool = dict(p1_starting_deck)
        p1.active_decks[p1_opponent_key] = greedy_game_deck_cut(p1_starting_deck, min_size)
        print(f"  [p1] Cold-start deck: {sum(p1.active_decks[p1_opponent_key].values())} cards")
    if p2 is not None and p2_starting_deck and not p2.card_pool:
        p2.card_pool = dict(p2_starting_deck)
        p2.active_decks[p2_opponent_key] = greedy_game_deck_cut(p2_starting_deck, min_size)
        print(f"  [p2] Cold-start deck: {sum(p2.active_decks[p2_opponent_key].values())} cards")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train play agents (Phase 3) for the FaB RL pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--format", default=DEFAULT_FORMAT,
        choices=["silver_age", "classic_constructed", "blitz", "upf"])
    parser.add_argument("--hero-id", default=DEFAULT_HERO_ID)
    parser.add_argument("--equipment-header", default=DEFAULT_EQUIPMENT_HEADER)
    parser.add_argument("--opponent-mode", default="preset",
        choices=["preset", "mirror", "dual"])
    parser.add_argument("--opponent-deck", default=DEFAULT_OPPONENT_DECK)
    parser.add_argument("--opponent-hero-id", default=DEFAULT_OPPONENT_HERO)
    parser.add_argument("--p2-hero-id", default="dorinthea_ironsong")
    parser.add_argument("--p2-equipment-header",
        default="dorinthea_ironsong dori_equipment_sword dori_equipment_sword "
                "helm_of_avarice gauntlet_of_might ironrot_legs valor_boots")
    parser.add_argument("--play-episodes", type=int, default=RUNTIME.full_pipeline.play_episodes)
    parser.add_argument("--play-checkpoint-interval", type=int, default=None,
        help="Fixed checkpoint interval in episodes (default: 10%% of --play-episodes)")
    parser.add_argument("--checkpoint-interval-pct", type=float,
        default=DEFAULT_CHECKPOINT_INTERVAL_PCT,
        help="Checkpoint every N%% of play episodes when interval is unset")
    parser.add_argument("--checkpoint-eval-episodes", type=int,
        default=DEFAULT_CHECKPOINT_EVAL_EPISODES,
        help="C++ eval games vs fixed sampled opponent policy at each checkpoint (0=off)")
    parser.add_argument("--max-play-steps", type=int, default=RUNTIME.play.max_play_steps)
    parser.add_argument("--warmup-episodes", type=int, default=DEFAULT_WARMUP_EPISODES)
    parser.add_argument("--warmup-baseline-eval-episodes", type=int,
        default=DEFAULT_WARMUP_BASELINE_EVAL_EPISODES)
    parser.add_argument("--workers", type=int, default=None,
        help="Parallel C++ game sessions for play training (auto-detected when omitted)")
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_AGENT_CACHE_DIR),
        help="Shared PPO + episode cache root",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--parallel-seeds",
        type=int,
        default=DEFAULT_PARALLEL_SEEDS,
        help=(
            "Independent RNG seeds to train in parallel; training win%% is "
            "averaged and the best P1/P2 agents are used for eval (default: "
            f"{DEFAULT_PARALLEL_SEEDS}; use 1 to disable)"
        ),
    )
    parser.add_argument(
        "--parallel-seeds-until-first-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=RUNTIME.play.parallel_seeds_until_first_checkpoint,
        help=(
            "Run all parallel seeds only through the first checkpoint, then "
            "continue training only the best seed"
        ),
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--p1-play", default=None)
    parser.add_argument("--p2-play", default=None)
    parser.add_argument("--p1-starting-deck", default=None)
    parser.add_argument("--p2-starting-deck", default=None)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--results-json", default=None)
    parser.add_argument("--talishar-url",
        default=os.environ.get("TALISHAR_URL", "http://localhost:8080/game"))
    parser.add_argument("--talishar-fe-url",
        default=os.environ.get("TALISHAR_FE_URL", "http://localhost:5173"))
    parser.add_argument("--cpp-engine-dir", default=None)
    parser.add_argument("--assets-path", default=os.environ.get("TALISHAR_ASSETS_PATH", ""))
    parser.add_argument(
        "--no-build-cpp-engine",
        action="store_true",
        help="Do not auto-build a C++ engine when none is cached",
    )
    parser.add_argument("--deck-state", default=None,
        help="Path to deck_state.json (default: <out-dir>/deck_state.json)")
    parser.add_argument("--final-eval-episodes", type=int, default=RUNTIME.full_pipeline.final_eval_episodes)
    parser.add_argument("--final-eval-max-steps", type=int, default=RUNTIME.full_pipeline.final_eval_max_steps)
    parser.add_argument(
        "--render-gif",
        action="store_true",
        help="After final eval, render a Talishar FE rollout GIF (slow; off by default)",
    )
    parser.add_argument("--gif-fps", type=float, default=RUNTIME.play.gif_fps)
    parser.add_argument("--skip-final-eval", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_path = resolve_assets_path(args.assets_path or None)
    args.opponent_deck = normalize_talishar_asset_name(args.opponent_deck, assets_path)
    if args.opponent_mode == "mirror":
        args.opponent_hero_id = args.hero_id

    cpp_opponent = (
        args.p2_hero_id if args.opponent_mode == "dual" else args.opponent_deck
    )
    if not args.no_build_cpp_engine:
        _use_existing = False
        if args.cpp_engine_dir:
            try:
                if str(REPO_ROOT / "src") not in sys.path:
                    sys.path.insert(0, str(REPO_ROOT / "src"))
                from flesh_and_blood_rlbridge.cpp_engine_environment import (  # noqa: PLC0415
                    is_cpp_engine_available,
                )
                _use_existing = is_cpp_engine_available(args.cpp_engine_dir)
            except Exception:
                _use_existing = False
        if not _use_existing:
            args.cpp_engine_dir = ensure_cpp_engine_for_matchup(
                args.hero_id,
                cpp_opponent,
                assets_path=assets_path,
                talishar_url=args.talishar_url,
                deck1_json=Path(args.p1_starting_deck) if args.p1_starting_deck else None,
                deck2_json=Path(args.p2_starting_deck) if args.p2_starting_deck else None,
                build=True,
            ) or args.cpp_engine_dir

    if args.workers is None:
        args.workers = auto_detect_workers(
            hero_id=args.hero_id,
            p2_hero_id=args.p2_hero_id,
            cpp_engine_dir=args.cpp_engine_dir,
            assets_path=assets_path,
        )

    min_warmup = max(1, math.ceil(args.play_episodes / 10))
    warmup_eps = min(max(args.warmup_episodes, min_warmup), args.play_episodes)

    results_json = Path(args.results_json) if args.results_json else out_dir / "results.json"
    min_size = min_deck_size_for_format(args.format)

    p1 = PhaseAgents(player="p1", play=_load_agent(args.p1_play), equipment_header=args.equipment_header)
    p2: Optional[PhaseAgents] = None
    if args.opponent_mode == "dual":
        p2 = PhaseAgents(player="p2", play=_load_agent(args.p2_play), equipment_header=args.p2_equipment_header)

    deck_state_file = Path(args.deck_state) if args.deck_state else out_dir / "deck_state.json"
    if deck_state_file.is_file():
        state = json.loads(deck_state_file.read_text(encoding="utf-8"))
        apply_deck_state(p1, p2, state)
        print(f"  Loaded deck state from {deck_state_file}")
    else:
        p1_start = _load_starting_deck(args.p1_starting_deck)
        p2_start = _load_starting_deck(args.p2_starting_deck)
        p1_opp = args.p2_hero_id if args.opponent_mode == "dual" else args.opponent_hero_id
        _seed_agents_from_starting_decks(
            p1, p2,
            p1_starting_deck=p1_start,
            p2_starting_deck=p2_start,
            min_size=min_size,
            p1_opponent_key=p1_opp,
            p2_opponent_key=args.hero_id,
        )

    print(
        f"\n{'='*62}\n"
        f"  Play Training (Phase 3)\n"
        f"  Format: {args.format}  |  Mode: {args.opponent_mode}\n"
        f"  Iterations: {args.iterations}\n"
        f"{'='*62}"
    )

    for iteration in range(1, args.iterations + 1):
        print(f"\n\n{'#'*62}\n  PLAY ITERATION {iteration} / {args.iterations}\n{'#'*62}")

        p1_wr, p2_wr = run_phase3_play(
            p1,
            p2 if args.opponent_mode == "dual" else None,
            opponent_mode=args.opponent_mode,
            game_format=args.format,
            p1_hero_id=args.hero_id,
            p2_hero_id=args.p2_hero_id,
            p1_equipment_header=args.equipment_header,
            p2_equipment_header=args.p2_equipment_header,
            p1_opponent_hero_id=(
                args.p2_hero_id if args.opponent_mode == "dual" else args.opponent_hero_id
            ),
            p1_opponent_deck_name=args.opponent_deck,
            n_episodes=args.play_episodes,
            max_play_steps=args.max_play_steps,
            warmup_episodes=warmup_eps,
            warmup_baseline_eval_episodes=args.warmup_baseline_eval_episodes,
            n_workers=args.workers,
            assets_path=assets_path,
            base_url=args.talishar_url,
            out_dir=out_dir,
            cache_dir=Path(args.cache_dir),
            seed=args.seed,
            cpp_engine_dir=args.cpp_engine_dir,
            checkpoint_interval=args.play_checkpoint_interval,
            checkpoint_interval_pct=args.checkpoint_interval_pct,
            checkpoint_eval_episodes=args.checkpoint_eval_episodes,
            parallel_seeds=args.parallel_seeds,
            parallel_seeds_until_first_checkpoint=args.parallel_seeds_until_first_checkpoint,
        )
        p1.last_play_win_rate = p1_wr
        if p2 is not None:
            p2.last_play_win_rate = p2_wr

        _save_all_agents(p1, out_dir)
        if p2 is not None:
            _save_all_agents(p2, out_dir)
        save_deck_state(out_dir, p1=p1, p2=p2, game_format=args.format, opponent_mode=args.opponent_mode)
        _write_results_json(
            results_json,
            game_format=args.format,
            opponent_mode=args.opponent_mode,
            p1=p1,
            p2=p2,
            iterations=iteration,
        )

    if not args.skip_final_eval:
        final_eval_dir = out_dir / "final_eval"
        final_eval_dir.mkdir(parents=True, exist_ok=True)
        p1_eval = run_final_evaluation(
            p1,
            p2 if args.opponent_mode == "dual" else None,
            hero_id=args.hero_id,
            equipment_header=args.equipment_header,
            opponent_equipment_header=args.p2_equipment_header,
            game_format=args.format,
            opponent_deck_name=args.opponent_deck,
            opponent_hero_id=(
                args.opponent_hero_id if args.opponent_mode != "dual" else args.p2_hero_id
            ),
            opponent_mode=args.opponent_mode,
            num_eval_episodes=args.final_eval_episodes,
            max_steps=args.final_eval_max_steps,
            assets_path=assets_path,
            base_url=args.talishar_url,
            fe_url=args.talishar_fe_url,
            out_dir=final_eval_dir,
            render_gif=args.render_gif,
            gif_fps=args.gif_fps,
        )
        p2_eval = None
        if args.opponent_mode == "dual" and p2 is not None:
            p2_eval = run_final_evaluation(
                p2, p1,
                hero_id=args.p2_hero_id,
                equipment_header=args.p2_equipment_header,
                opponent_equipment_header=args.equipment_header,
                game_format=args.format,
                opponent_deck_name=args.opponent_deck,
                opponent_hero_id=args.hero_id,
                opponent_mode=args.opponent_mode,
                num_eval_episodes=args.final_eval_episodes,
                max_steps=args.final_eval_max_steps,
                assets_path=assets_path,
                base_url=args.talishar_url,
                fe_url=args.talishar_fe_url,
                out_dir=final_eval_dir,
                render_gif=args.render_gif,
                gif_fps=args.gif_fps,
            )
        _write_results_json(
            results_json,
            game_format=args.format,
            opponent_mode=args.opponent_mode,
            p1=p1,
            p2=p2,
            iterations=args.iterations,
            p1_final_eval=p1_eval,
            p2_final_eval=p2_eval,
        )

    print(f"\n  Play training complete → {out_dir}")


if __name__ == "__main__":
    main()
