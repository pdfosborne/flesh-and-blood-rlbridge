#!/usr/bin/env python3
"""Systematically train PPO agents for all SAGE precon matchups using self-play.

Each matchup trains a single PPO policy that controls BOTH players in a
self-play game (one policy, alternating turns).  Training against yourself is
more sample-efficient than training against a fixed AI: the policy generates
its own curriculum on both sides of the board simultaneously.

Matchups trained:
  1. briar-mirror      – BriarSAGEPrecon (P1) vs BriarSAGEPrecon (P2)
  2. dorinthea-mirror  – DorintheSAGEPrecon (P1) vs DorintheSAGEPrecon (P2)
  3. briar-vs-dorinthea – BriarSAGEPrecon (P1) vs DorintheSAGEPrecon (P2)
  4. dorinthea-vs-briar – DorintheSAGEPrecon (P1) vs BriarSAGEPrecon (P2)

Saved agents land in results/sage_precon_agents/<matchup-name>/.

Usage:
    TALISHAR_URL=http://localhost:8080/game python scripts/train_sage_precons.py
    TALISHAR_URL=http://localhost:8080/game python scripts/train_sage_precons.py \\
        --episodes 500 --matchup briar-vs-dorinthea
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── path setup ──────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_FAB_SRC = _REPO_ROOT / "src"
_RL_SRC = Path("~/Documents/RL-IP/src").expanduser()

for p in (_FAB_SRC, _RL_SRC):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from flesh_and_blood_rlbridge.talishar_engine_environment import TalisharEngineEnvironment  # noqa: E402
from rlbridge.rl_agents.ppo import PPOAgent  # noqa: E402

# ── matchup definitions ──────────────────────────────────────────────────────

@dataclass
class Matchup:
    name: str
    p1_deck: str
    p2_deck: str
    description: str
    tags: list[str] = field(default_factory=list)


SAGE_MATCHUPS: list[Matchup] = [
    Matchup(
        name="briar-mirror",
        p1_deck="BriarSAGEPrecon",
        p2_deck="BriarSAGEPrecon",
        description="Briar SAGE precon mirror self-play",
        tags=["briar", "mirror"],
    ),
    Matchup(
        name="dorinthea-mirror",
        p1_deck="DorintheSAGEPrecon",
        p2_deck="DorintheSAGEPrecon",
        description="Dorinthea SAGE precon mirror self-play",
        tags=["dorinthea", "mirror"],
    ),
    Matchup(
        name="briar-vs-dorinthea",
        p1_deck="BriarSAGEPrecon",
        p2_deck="DorintheSAGEPrecon",
        description="Briar (P1) vs Dorinthea (P2) cross-matchup self-play",
        tags=["briar", "dorinthea", "cross"],
    ),
    Matchup(
        name="dorinthea-vs-briar",
        p1_deck="DorintheSAGEPrecon",
        p2_deck="BriarSAGEPrecon",
        description="Dorinthea (P1) vs Briar (P2) cross-matchup self-play",
        tags=["dorinthea", "briar", "cross"],
    ),
]

MATCHUP_INDEX = {m.name: m for m in SAGE_MATCHUPS}

# Cached rlbridge environment used for evaluation / policy render after training.
# Must be loaded via rl_load_cached_environments before running eval/render tools.
EVAL_ENV_IDS: dict[str, str] = {
    "briar-vs-dorinthea": "FaB-Briar-vs-Dorinthea-SAGE-v0",
    "briar-mirror":       "FaB-Briar-vs-Dorinthea-SAGE-v0",
}

# ── PPO defaults ─────────────────────────────────────────────────────────────
DEFAULT_N_EPISODES = 300
DEFAULT_MAX_STEPS = 60
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_LR = 3e-4
DEFAULT_GAMMA = 0.99
DEFAULT_LAM = 0.95
DEFAULT_CLIP_EPS = 0.2
DEFAULT_N_STEPS = 256
DEFAULT_PPO_EPOCHS = 4
DEFAULT_MINI_BATCH = 64

# ── helpers ───────────────────────────────────────────────────────────────────

def make_env(matchup: Matchup, base_url: str, game_format: str = "blitz") -> TalisharEngineEnvironment:
    """Create a self-play Talishar environment for the given matchup."""
    # Mirror matches: p2 deck = p1 deck — omit opponent_deck_name so the
    # engine uses TalisharEngineEnvironment's default (same as local_deck_name)
    opponent = None if matchup.p1_deck == matchup.p2_deck else matchup.p2_deck
    return TalisharEngineEnvironment(
        base_url=base_url,
        local_deck_name=matchup.p1_deck,
        opponent_deck_name=opponent,
        game_format=game_format,
        self_play=True,
        max_turns=DEFAULT_MAX_STEPS,
    )


def make_agent(seed: Optional[int] = None) -> PPOAgent:
    return PPOAgent(
        hidden_size=DEFAULT_HIDDEN_SIZE,
        lr_actor=DEFAULT_LR,
        lr_critic=DEFAULT_LR,
        gamma=DEFAULT_GAMMA,
        lam=DEFAULT_LAM,
        clip_eps=DEFAULT_CLIP_EPS,
        n_steps=DEFAULT_N_STEPS,
        ppo_epochs=DEFAULT_PPO_EPOCHS,
        mini_batch_size=DEFAULT_MINI_BATCH,
        seed=seed,
    )


def train_matchup(
    matchup: Matchup,
    *,
    base_url: str,
    n_episodes: int,
    max_steps: int,
    out_dir: Path,
    seed: Optional[int] = None,
    game_format: str = "blitz",
) -> dict:
    """Train a PPO agent for one matchup and save weights + metadata."""
    print(f"\n{'='*60}")
    print(f"  Matchup : {matchup.name}")
    print(f"  Decks   : {matchup.p1_deck} (P1) vs {matchup.p2_deck} (P2)")
    print(f"  Mode    : self-play | {n_episodes} episodes | max {max_steps} steps")
    print(f"{'='*60}")

    env = make_env(matchup, base_url=base_url, game_format=game_format)
    agent = make_agent(seed=seed)

    t0 = time.time()
    result = agent.train(env, n_episodes=n_episodes, max_steps=max_steps, seed=seed)
    elapsed = time.time() - t0

    avg_reward = float(result.avg_reward) if hasattr(result, "avg_reward") else 0.0
    best_reward = float(result.best_reward) if hasattr(result, "best_reward") else 0.0

    print(f"  Done in {elapsed:.1f}s — avg_reward={avg_reward:.3f}  best={best_reward:.3f}")

    # ── save in rlbridge package format ───────────────────────────────────────
    agent_id = f"{matchup.name}-{uuid.uuid4().hex[:8]}"
    eval_env_id = EVAL_ENV_IDS.get(matchup.name, "")

    package_dir = out_dir / matchup.name / f"ppo_{agent_id}"
    weights_dir = package_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    weights_path = weights_dir / "agent_weights.json"
    agent.save(weights_path)

    episode_rewards = list(result.episode_rewards) if hasattr(result, "episode_rewards") else []
    (package_dir / "metadata.json").write_text(json.dumps({
        "agent_id":       agent_id,
        "agent_type":     "ppo",
        "env_id":         eval_env_id,
        "created_at":     datetime.now().isoformat(),
        "weights_file":   "agent_weights.json",
        "training_config": {"hidden_size": DEFAULT_HIDDEN_SIZE},
        "use_language_state": False,
        "matchup":        matchup.name,
        "p1_deck":        matchup.p1_deck,
        "p2_deck":        matchup.p2_deck,
        "self_play":      True,
        "game_format":    game_format,
    }, indent=2))
    (package_dir / "training_results.json").write_text(json.dumps({
        "agent_name":      "ppo",
        "n_episodes":      n_episodes,
        "episode_rewards": episode_rewards,
        "final_epsilon":   0.0,
        "eval_mean":       avg_reward,
        "eval_std":        0.0,
        "eval_rewards":    [],
    }, indent=2))

    meta = {
        "matchup":      matchup.name,
        "agent_id":     agent_id,
        "eval_env_id":  eval_env_id,
        "package_dir":  str(package_dir),
        "elapsed_secs": round(elapsed, 1),
        "avg_reward":   avg_reward,
        "best_reward":  best_reward,
    }
    print(f"  Saved   → {package_dir}")
    print(f"  agent_id: {agent_id}")

    return meta


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO self-play agents for SAGE precon matchups."
    )
    parser.add_argument(
        "--matchup",
        choices=list(MATCHUP_INDEX) + ["all"],
        default="all",
        help="Which matchup to train (default: all).",
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
        "--format",
        default="blitz",
        help="Talishar game format, e.g. blitz or cc (default: blitz).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (optional).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_REPO_ROOT / "results" / "sage_precon_agents"),
        help="Directory to save trained agents.",
    )
    args = parser.parse_args()

    base_url = os.environ.get("TALISHAR_URL", "http://localhost:8080/game")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matchups = SAGE_MATCHUPS if args.matchup == "all" else [MATCHUP_INDEX[args.matchup]]

    print(f"Talishar URL : {base_url}")
    print(f"Output dir   : {out_dir}")
    print(f"Matchups     : {[m.name for m in matchups]}")

    summary: list[dict] = []
    failed: list[str] = []

    for matchup in matchups:
        try:
            meta = train_matchup(
                matchup,
                base_url=base_url,
                n_episodes=args.episodes,
                max_steps=args.max_steps,
                out_dir=out_dir,
                seed=args.seed,
                game_format=args.format,
            )
            summary.append(meta)
        except Exception as exc:
            print(f"\n  ERROR training {matchup.name}: {exc}")
            failed.append(matchup.name)

    # ── final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  TRAINING SUMMARY")
    print(f"{'='*60}")
    for m in summary:
        print(
            f"  {m['matchup']:<28} avg={m['avg_reward']:+.3f}  "
            f"best={m['best_reward']:+.3f}  ({m['elapsed_secs']:.0f}s)\n"
            f"  {'agent_id':<28} {m['agent_id']}\n"
            f"  {'package_dir':<28} {m['package_dir']}"
        )
    for name in failed:
        print(f"  {name:<28} FAILED")

    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\n  Summary saved → {out_dir / 'training_summary.json'}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
