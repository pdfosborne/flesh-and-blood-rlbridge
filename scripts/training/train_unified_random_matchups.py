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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_dual_agent_common import (  # noqa: E402
    DEFAULT_N_EPISODES,
    DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
    DEFAULT_WARMUP_EPISODES,
    FABRARY_ENV_SUFFIX,
    REPO_ROOT,
    build_fabrary_eval_env_ids,
    materialize_fabrary_decks,
    print_training_summary,
    run_matchup_training,
    sample_random_fabrary_matchups,
)
from runtime_defaults import (  # noqa: E402
    DEFAULT_CHECKPOINT_EVAL_EPISODES,
    DEFAULT_CHECKPOINT_INTERVAL_PCT,
    RUNTIME,
)

DEFAULT_MAX_STEPS = RUNTIME.dual_matchup.max_steps


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
        default=3,
        help="Number of random deck pairs to train this run",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_N_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--warmup-episodes",
        type=int,
        default=DEFAULT_WARMUP_EPISODES,
    )
    parser.add_argument(
        "--warmup-baseline-eval-episodes",
        type=int,
        default=DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
    )
    parser.add_argument(
        "--checkpoint-interval-pct",
        type=float,
        default=DEFAULT_CHECKPOINT_INTERVAL_PCT,
    )
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument(
        "--checkpoint-eval-episodes",
        type=int,
        default=DEFAULT_CHECKPOINT_EVAL_EPISODES,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=RUNTIME.dual_matchup.workers,
        help="Parallel Talishar/C++ worker sessions per matchup",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "results" / "agent_cache"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "results" / "unified_random_matchups"),
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
        default=True,
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
        "--no-build-cpp-engine",
        action="store_true",
        help="Do not build a C++ engine when one is missing",
    )
    parser.add_argument(
        "--no-require-cpp-engine",
        action="store_true",
        help="Allow HTTP Talishar fallback when no C++ engine is available",
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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.run_dir:
        out_dir = Path(args.run_dir)
    else:
        out_dir = Path(args.out_dir) / format_name / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": format_name,
        "matchups_requested": int(args.matchups),
        "matchups_sampled": [m.name for m in selected],
        "episodes_per_matchup": int(args.episodes),
        "seed": args.seed,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Talishar URL : {base_url}")
    print(f"Format       : {format_name}")
    print(f"Deck pool    : {len(decks)} fabrary decks")
    print(f"Matchups     : {len(selected)} random pairs")
    for matchup in selected:
        print(f"  - {matchup.name}")
    print(f"Output dir   : {out_dir}")
    print(f"Cache dir    : {args.cache_dir}")

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
        checkpoint_interval_pct=float(args.checkpoint_interval_pct),
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_eval_episodes=int(args.checkpoint_eval_episodes),
        skip_converged=bool(args.skip_converged),
        build_cpp_engine=not args.no_build_cpp_engine,
        require_cpp_engine=not args.no_require_cpp_engine,
    )
    print_training_summary(summary, failed, out_dir)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
