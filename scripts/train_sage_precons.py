#!/usr/bin/env python3
"""Train PPO agents for every SAGE precon cross-matchup using dual-agent self-play.

"Self-play" here means both sides of every game are controlled by learning agents —
not a mirror match.  Each matchup pits two different SAGE precon decks (P1 vs P2).
The Talishar server runs the game with both players AI-controlled (no CombatDummy),
and a separate PPO agent trains from each player's perspective simultaneously.

45 unique matchups are generated across 10 SAGE precon decks (C(10,2) pairs).

Usage:
    Linux / macOS:
        TALISHAR_URL=http://localhost:8080/game python scripts/train_sage_precons.py
        TALISHAR_URL=http://localhost:8080/game python scripts/train_sage_precons.py \\
            --episodes 500 --matchup briar-vs-dorinthea

    Windows (PowerShell):
        $env:TALISHAR_URL="http://localhost:8080/game"; python scripts/train_sage_precons.py
        $env:TALISHAR_URL="http://localhost:8080/game"; python scripts/train_sage_precons.py `
            --episodes 500 --matchup briar-vs-dorinthea

    Windows (Command Prompt):
        set "TALISHAR_URL=http://localhost:8080/game" && python scripts/train_sage_precons.py
        set "TALISHAR_URL=http://localhost:8080/game" && python scripts/train_sage_precons.py ^
            --episodes 500 --matchup briar-vs-dorinthea
        Note: always use quoted ``set "VAR=value"`` syntax — unquoted ``set VAR=value &&``
        captures the space before ``&&`` as part of the value, causing URL path errors.
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
    Matchup,
    REPO_ROOT,
    print_training_summary,
    run_matchup_training,
)

# All available SAGE precon decks (hero_slug, deck_name)
SAGE_DECKS: list[tuple[str, str]] = [
    ("briar", "BriarSAGEPrecon"),
    ("dorinthea", "DorintheSAGEPrecon"),
    ("kayo", "KayoSAGEPrecon"),
    ("viserai", "ViseraiSAGEPrecon"),
    ("iyslander", "IyslanderSAGEPrecon"),
    ("dash", "DashSAGEPrecon"),
    ("fai", "FaiSAGEPrecon"),
    ("azalea", "AzaleaSAGEPrecon"),
    ("boltyn", "BoltynSAGEPrecon"),
    ("enigma", "EnigmaSAGEPrecon"),
]

SAGE_MATCHUPS: list[Matchup] = [
    Matchup(
        name=f"{h1}-vs-{h2}",
        p1_deck=d1,
        p2_deck=d2,
        description=(
            f"{h1.capitalize()} (P1) vs {h2.capitalize()} (P2) — "
            "dual-agent training, both perspectives"
        ),
        tags=[h1, h2, "cross"],
        p1_hero=h1,
        p2_hero=h2,
    )
    for i, (h1, d1) in enumerate(SAGE_DECKS)
    for h2, d2 in SAGE_DECKS[i + 1 :]
]

MATCHUP_INDEX = {m.name: m for m in SAGE_MATCHUPS}

_DECK_TO_HERO: dict[str, str] = {deck: hero for hero, deck in SAGE_DECKS}

EVAL_ENV_IDS: dict[str, str] = {
    m.name: (
        f"FaB-{_DECK_TO_HERO[m.p1_deck].capitalize()}"
        f"-vs-{_DECK_TO_HERO[m.p2_deck].capitalize()}-SAGE-v0"
    )
    for m in SAGE_MATCHUPS
}

DEFAULT_MAX_STEPS = 100


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train PPO agents across all SAGE precon cross-matchups, "
            "learning from both perspectives."
        )
    )
    parser.add_argument(
        "--matchup",
        default="all",
        help=(
            "Which matchup to train (default: all).  "
            "Use 'all' or a name like 'briar-vs-dorinthea'.  "
            f"Available: {len(MATCHUP_INDEX)} total"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_N_EPISODES,
        help=f"Training episodes per matchup (default: {DEFAULT_N_EPISODES}).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Max steps per episode (default: {DEFAULT_MAX_STEPS}).",
    )
    parser.add_argument(
        "--warmup-episodes",
        type=int,
        default=DEFAULT_WARMUP_EPISODES,
        help=(
            "Episodes to run with Talishar default heuristic actions before "
            "PPO takes full control (default: 100)."
        ),
    )
    parser.add_argument(
        "--warmup-baseline-eval-episodes",
        type=int,
        default=DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
        help=(
            "Episodes to evaluate at warmup->PPO handoff for baseline win%% "
            "benchmarking."
        ),
    )
    parser.add_argument(
        "--format",
        default="sage",
        help="Talishar game format (default: sage).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (optional).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "results" / "sage_precon_agents"),
        help="Directory to save trained agents.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "results" / "agent_cache"),
        help="Root directory for four-tier agent weight cache.",
    )
    parser.add_argument(
        "--show-frontend",
        action="store_true",
        help="Write a live training-state image during training (no browser tabs).",
    )
    parser.add_argument(
        "--frontend-url",
        default=None,
        help="Talishar FE URL override (default: TALISHAR_URL host:port when --show-frontend is set).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel game sessions for training (default: 1). "
             "Set to 2-4 to saturate Talishar HTTP throughput.",
    )
    args = parser.parse_args()

    if args.matchup == "all":
        matchups = SAGE_MATCHUPS
    elif args.matchup in MATCHUP_INDEX:
        matchups = [MATCHUP_INDEX[args.matchup]]
    else:
        sample = ", ".join(list(MATCHUP_INDEX)[:10])
        print(f"Unknown matchup '{args.matchup}'. Available: {sample} ...")
        sys.exit(1)

    base_url = os.environ.get("TALISHAR_URL", "http://localhost:8080/game")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Talishar URL : {base_url}")
    print(f"Output dir   : {out_dir}")
    print(f"Matchups     : {[m.name for m in matchups]}")

    summary, failed = run_matchup_training(
        matchups,
        EVAL_ENV_IDS,
        base_url=base_url,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        out_dir=out_dir,
        seed=args.seed,
        game_format=args.format,
        cache_dir=Path(args.cache_dir),
        warmup_episodes=args.warmup_episodes,
        warmup_baseline_eval_episodes=args.warmup_baseline_eval_episodes,
        show_frontend=args.show_frontend,
        frontend_url=args.frontend_url,
        n_workers=args.workers,
    )
    print_training_summary(summary, failed, out_dir)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
