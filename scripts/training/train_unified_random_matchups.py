#!/usr/bin/env python3
"""Train the unified agent on randomly sampled fabrary deck matchups.

Each completed matchup appends to ``unified_agent.meta.json`` ``training_history``
with training win rates plus first/final checkpoint eval win rates.

Usage:
    TALISHAR_URL=http://localhost:8080/game python scripts/training/train_unified_random_matchups.py
    TALISHAR_URL=http://localhost:8080/game python scripts/training/train_unified_random_matchups.py \\
        --format classic_constructed --matchups 5 --episodes 500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402

_bootstrap.configure_paths()

from train_dual_agent_common import (  # noqa: E402
    FABRARY_ENV_SUFFIX,
    REPO_ROOT,
    build_fabrary_eval_env_ids,
    materialize_fabrary_decks,
    print_training_summary,
    run_matchup_training,
    sample_random_fabrary_matchups,
)
from runtime_defaults import (  # noqa: E402
    DEFAULT_UNIFIED_CHECKPOINT_EVAL_EPISODES,
    DEFAULT_UNIFIED_CHECKPOINT_INTERVAL_PCT,
    DEFAULT_UNIFIED_EPISODES,
    DEFAULT_UNIFIED_MATCHUPS,
    DEFAULT_UNIFIED_MAX_STEPS,
    DEFAULT_UNIFIED_WARMUP_EPISODES,
    DEFAULT_UNIFIED_WORKERS,
    DEFAULT_UNIFIED_PARALLEL_MATCHUPS,
    DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
    DEFAULT_ROLLOUT_MODE,
    DEFAULT_ROLLOUT_PROCESSES,
    RUNTIME,
    resolve_talishar_backend_urls,
)
from flesh_and_blood_rlbridge.talishar_backend_pool import TalisharBackendPool  # noqa: E402

DEFAULT_MAX_STEPS = DEFAULT_UNIFIED_MAX_STEPS
_UR = RUNTIME.unified_random_matchups


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the unified PPO agent on random fabrary deck matchups "
            "for a chosen format."
        )
    )
    parser.add_argument(
        "--format",
        default="silver_age",
        choices=sorted(FABRARY_ENV_SUFFIX.keys()),
        help="Deck pool format (fabrary_decks.json filter)",
    )
    parser.add_argument(
        "--matchups",
        type=int,
        default=DEFAULT_UNIFIED_MATCHUPS,
        help="Number of random deck pairs to train this run",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_UNIFIED_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--warmup-episodes",
        type=int,
        default=DEFAULT_UNIFIED_WARMUP_EPISODES,
    )
    parser.add_argument(
        "--warmup-baseline-eval-episodes",
        type=int,
        default=DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
    )
    parser.add_argument(
        "--checkpoint-interval-pct",
        type=float,
        default=DEFAULT_UNIFIED_CHECKPOINT_INTERVAL_PCT,
    )
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument(
        "--checkpoint-eval-episodes",
        type=int,
        default=DEFAULT_UNIFIED_CHECKPOINT_EVAL_EPISODES,
    )
    parser.add_argument(
        "--optimal-policy-live-render",
        action=argparse.BooleanOptionalAction,
        default=_UR.optimal_policy_live_render,
        help="Continuous optimal-policy PNG render on the render shard during training.",
    )
    parser.add_argument(
        "--debug-training",
        action=argparse.BooleanOptionalAction,
        default=_UR.debug_training,
        help="Log detailed errors to unified_training_debug.jsonl in the run folder.",
    )
    parser.add_argument(
        "--anti-stuck-logging",
        action=argparse.BooleanOptionalAction,
        default=RUNTIME.eval_dashboard.anti_stuck_logging,
        help="Log eval/render stuck-game diagnostics to anti_stuck_reports.jsonl.",
    )
    parser.add_argument("--seed", type=int, default=_UR.seed)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_UNIFIED_WORKERS,
        help="Total parallel Talishar fast worker sessions (split across parallel matchups)",
    )
    parser.add_argument(
        "--parallel-matchups",
        type=int,
        default=DEFAULT_UNIFIED_PARALLEL_MATCHUPS,
        help="Number of matchups to train concurrently per batch",
    )
    parser.add_argument(
        "--safe-parallel",
        action=argparse.BooleanOptionalAction,
        default=_UR.safe_parallel,
        help=(
            "Cap concurrent training games to one session per Talishar shard "
            "(reduces parallel_matchups/workers when needed)"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=_UR.cache_dir or str(REPO_ROOT / "results" / "agent_cache"),
    )
    parser.add_argument(
        "--out-dir",
        default=_UR.out_dir or str(REPO_ROOT / "results" / "unified_random_matchups"),
        help="Base directory when --run-dir is not set",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Exact run output directory (skips format/timestamp subdirs)",
    )
    parser.add_argument(
        "--skip-converged",
        action=argparse.BooleanOptionalAction,
        default=_UR.skip_converged,
        help="Skip deck pairs already converged in unified agent cache",
    )
    parser.add_argument(
        "--allow-repeat-pairs",
        action="store_true",
        help="Allow the same deck pair to be sampled more than once per run",
    )
    parser.add_argument(
        "--show-frontend",
        action="store_true",
        help="Write live training-state PNG during each matchup",
    )
    parser.add_argument("--frontend-url", default=None)
    parser.add_argument("--talishar-url", default=None)
    parser.add_argument(
        "--talishar-urls",
        default=None,
        help="Comma-separated Talishar backend URLs for multi-shard rollouts",
    )
    parser.add_argument(
        "--no-build-cpp-engine",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--require-cpp-engine",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-require-cpp-engine",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rollout-mode",
        default=DEFAULT_ROLLOUT_MODE,
        choices=sorted({"batched", "threaded_episodes", "batched_concurrent"}),
        help="Fast rollout collection strategy for parallel training",
    )
    parser.add_argument(
        "--rollout-processes",
        type=int,
        default=DEFAULT_ROLLOUT_PROCESSES,
        help="Subprocess rollout workers (1 = in-process only)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    format_name = str(args.format)
    env_suffix = FABRARY_ENV_SUFFIX.get(format_name, "SA")

    decks = materialize_fabrary_decks(format_name)
    if len(decks) < 2:
        print(f"Need at least two {format_name} decks in fabrary_decks.json")
        sys.exit(1)

    rng = random.Random(args.seed)
    try:
        selected = sample_random_fabrary_matchups(
            decks,
            int(args.matchups),
            rng,
            format_name,
            unique_pairs=not args.allow_repeat_pairs,
        )
    except RuntimeError as exc:
        print(str(exc))
        sys.exit(1)

    eval_env_ids = build_fabrary_eval_env_ids(selected, decks, env_suffix)
    base_url = args.talishar_url or os.environ.get(
        "TALISHAR_URL",
        "http://localhost:8080/game",
    )
    if args.talishar_urls:
        os.environ["TALISHAR_URLS"] = str(args.talishar_urls)
    backend_urls = resolve_talishar_backend_urls(fallback_url=base_url)
    backend_pool = TalisharBackendPool(urls=backend_urls)
    base_url = backend_pool.primary_url
    from flesh_and_blood_rlbridge.talishar_backend_pool import (  # noqa: PLC0415
        resolve_eval_backend_url,
        resolve_render_backend_url,
    )

    eval_url = resolve_eval_backend_url(fallback_url=base_url)
    render_url = resolve_render_backend_url(fallback_url=base_url)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.run_dir:
        out_dir = Path(args.run_dir)
    else:
        out_dir = Path(args.out_dir) / format_name / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    from fab_bridge.unified_training_debug import (  # noqa: PLC0415
        audit_matchup_decks,
        configure as configure_unified_debug,
        log_event as unified_debug_event,
    )

    debug_log = configure_unified_debug(
        run_dir=out_dir,
        enabled=bool(args.debug_training),
    )
    if debug_log is not None:
        print(f"Debug log    : {debug_log}")

    manifest = {
        "format": format_name,
        "matchups_requested": int(args.matchups),
        "matchups_sampled": [m.name for m in selected],
        "episodes_per_matchup": int(args.episodes),
        "parallel_matchups": max(1, int(args.parallel_matchups)),
        "safe_parallel": bool(args.safe_parallel),
        "debug_training": bool(args.debug_training),
        "anti_stuck_logging": bool(args.anti_stuck_logging),
        "optimal_policy_live_render": bool(args.optimal_policy_live_render),
        "talishar_backends": list(backend_pool.urls),
        "talishar_eval_url": eval_url,
        "talishar_render_url": render_url,
        "seed": args.seed,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Talishar URL : {backend_pool.format_log_label()}")
    if eval_url != base_url:
        print(f"Eval backend : {eval_url} (checkpoint eval)")
    if args.optimal_policy_live_render and render_url != eval_url:
        print(f"Render backend: {render_url} (optimal-policy live render)")
    elif args.optimal_policy_live_render:
        print(f"Render backend: {render_url} (shares eval shard — set TALISHAR_RENDER_URL)")
    print(f"Backend      : Talishar fast (fast_reset / fast_step_index)")
    print(f"Rollout mode : {args.rollout_mode} ({args.rollout_processes} process(es))")
    print(f"Format       : {format_name}")
    print(f"Deck pool    : {len(decks)} fabrary decks")
    print(f"Matchups     : {len(selected)} random pairs")
    print(f"Parallel     : {max(1, int(args.parallel_matchups))} matchup(s) per batch")
    print(f"Safe parallel: {'yes' if args.safe_parallel else 'no'}")
    for matchup in selected:
        print(f"  - {matchup.name}")
        audit_matchup_decks(matchup, game_format=format_name)
    print(f"Output dir   : {out_dir}")
    print(f"Cache dir    : {args.cache_dir}")

    from fab_bridge.unified_dashboard import (  # noqa: PLC0415
        UNIFIED_DASHBOARD_NAME,
        write_unified_random_matchups_dashboard,
    )

    write_unified_random_matchups_dashboard(out_dir, auto_refresh_seconds=5.0)
    print(f"Dashboard    : {out_dir / UNIFIED_DASHBOARD_NAME}")

    build_cpp_engine = _UR.build_cpp_engine
    if args.no_build_cpp_engine:
        build_cpp_engine = False
    require_cpp_engine = _UR.require_cpp_engine
    if args.require_cpp_engine:
        require_cpp_engine = True
    if args.no_require_cpp_engine:
        require_cpp_engine = False

    dashboard_proc = None
    eval_dashboard_proc = None
    summary: list = []
    failed: list = []
    try:
        from runscripts._common import (  # noqa: PLC0415
            start_unified_random_matchups_eval_dashboard,
            start_unified_random_matchups_train_dashboard,
            stop_background_process,
        )

        dashboard_proc = start_unified_random_matchups_train_dashboard(out_dir)
        assets_path = os.environ.get(
            "TALISHAR_ASSETS_PATH",
            str(REPO_ROOT / "Talishar" / "Assets"),
        )
        if args.optimal_policy_live_render:
            eval_dashboard_proc = start_unified_random_matchups_eval_dashboard(
                out_dir,
                assets=assets_path,
                talishar_url_value=eval_url,
                talishar_render_url_value=render_url,
                cache_dir=args.cache_dir,
                live_render=True,
                debug_training=bool(args.debug_training),
                anti_stuck_logging=bool(args.anti_stuck_logging),
            )
        unified_debug_event(
            "matchup_load",
            "Starting unified matchup training",
            matchups=[m.name for m in selected],
            workers=int(args.workers),
            parallel_matchups=int(args.parallel_matchups),
            backends=list(backend_pool.urls),
        )
        summary, failed = run_matchup_training(
            selected,
            eval_env_ids,
            base_url=base_url,
            n_episodes=int(args.episodes),
            max_steps=int(args.max_steps),
            out_dir=out_dir,
            seed=args.seed,
            game_format=format_name,
            cache_dir=Path(args.cache_dir),
            warmup_episodes=int(args.warmup_episodes),
            warmup_baseline_eval_episodes=int(args.warmup_baseline_eval_episodes),
            show_frontend=bool(args.show_frontend),
            frontend_url=args.frontend_url,
            n_workers=max(1, int(args.workers)),
            parallel_matchups=max(1, int(args.parallel_matchups)),
            checkpoint_interval_pct=float(args.checkpoint_interval_pct),
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_eval_episodes=int(args.checkpoint_eval_episodes),
            skip_converged=bool(args.skip_converged),
            build_cpp_engine=build_cpp_engine,
            require_cpp_engine=require_cpp_engine,
            rollout_mode=str(args.rollout_mode),
            rollout_processes=int(args.rollout_processes),
            backend_pool=backend_pool,
            safe_parallel=bool(args.safe_parallel),
            debug_training=bool(args.debug_training),
        )
    finally:
        if eval_dashboard_proc is not None:
            from runscripts._common import stop_background_process  # noqa: PLC0415

            stop_background_process(eval_dashboard_proc)
        if dashboard_proc is not None:
            from runscripts._common import stop_background_process  # noqa: PLC0415

            stop_background_process(dashboard_proc)
    print_training_summary(summary, failed, out_dir)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
