#!/usr/bin/env python3
"""Train PPO agents for every Silver Age fabrary-deck cross-matchup using dual-agent self-play.

Deck lists come from ``fabrary_decks.json`` (format ``silver_age``).

Usage:
    TALISHAR_URL=http://localhost:8080/game python scripts/train_silver_age_decks.py
    TALISHAR_URL=http://localhost:8080/game python scripts/train_silver_age_decks.py \\
        --episodes 500 --matchup ira_crimson_haze_sage_aggro-vs-fai_sage_chain
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_dual_agent_common import (  # noqa: E402
    DEFAULT_N_EPISODES,
    DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
    DEFAULT_WARMUP_EPISODES,
    REPO_ROOT,
    init_fabrary_training,
    print_training_summary,
    run_matchup_training,
)

DEFAULT_MAX_STEPS = 200
FORMAT_NAME = "silver_age"
ENV_SUFFIX = "SA"


def main() -> None:
    decks, matchups, matchup_index, eval_env_ids = init_fabrary_training(
        FORMAT_NAME, ENV_SUFFIX
    )

    parser = argparse.ArgumentParser(
        description=(
            "Train PPO agents across all Silver Age fabrary deck cross-matchups, "
            "learning from both perspectives."
        )
    )
    parser.add_argument(
        "--matchup",
        default="all",
        help=(
            "Which matchup to train (default: all).  "
            "Use 'all' or a slug like 'ira_crimson_haze_sage_aggro-vs-fai_sage_chain'.  "
            f"Available: {len(matchup_index)} matchups across {len(decks)} decks"
        ),
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_N_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--warmup-episodes",
        type=int,
        default=DEFAULT_WARMUP_EPISODES,
        help="Episodes to bootstrap with Talishar default heuristic policy.",
    )
    parser.add_argument(
        "--warmup-baseline-eval-episodes",
        type=int,
        default=DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
        help="Episodes to evaluate baseline win% at warmup handoff.",
    )
    parser.add_argument("--format", default=FORMAT_NAME)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "results" / "silver_age_agents"),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "results" / "agent_cache"),
        help="Root directory for four-tier agent weight cache.",
    )
    args = parser.parse_args()

    if args.matchup == "all":
        selected = matchups
    elif args.matchup in matchup_index:
        selected = [matchup_index[args.matchup]]
    else:
        sample = ", ".join(list(matchup_index)[:5])
        print(f"Unknown matchup '{args.matchup}'. Examples: {sample} ...")
        sys.exit(1)

    base_url = os.environ.get("TALISHAR_URL", "http://localhost:8080/game")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Talishar URL : {base_url}")
    print(f"Output dir   : {out_dir}")
    print(f"Decks        : {len(decks)} {FORMAT_NAME} decks from fabrary_decks.json")
    print(f"Matchups     : {len(selected)} selected")

    summary, failed = run_matchup_training(
        selected,
        eval_env_ids,
        base_url=base_url,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        out_dir=out_dir,
        seed=args.seed,
        game_format=args.format,
        cache_dir=Path(args.cache_dir),
        warmup_episodes=args.warmup_episodes,
        warmup_baseline_eval_episodes=args.warmup_baseline_eval_episodes,
    )
    print_training_summary(summary, failed, out_dir)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
