#!/usr/bin/env python3
"""Split checkpoint self-play eval by seat vs deck to diagnose one-sided win rates.

Usage:
    python scripts/eval/diagnose_seat_bias.py \\
        --checkpoint-dir results/.../unified_selfplay/p1/episode_000400 \\
        --episodes 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS / "training"))
import _bootstrap  # noqa: E402

_bootstrap.configure_paths()

from agent_cache import clone_agent_weights  # noqa: E402
from flesh_and_blood_rlbridge.player_observation import ACTION_CAPACITY, PLAYER_OBS_DIM  # noqa: E402
from play_outcome_stats import classify_p1_fast_episode_outcome  # noqa: E402
from rl_agents.ppo import PPOAgent  # noqa: E402
from train_dual_agent_common import (  # noqa: E402
    Matchup,
    _env_supports_fast_training,
    make_env,
    swapped_matchup,
)
from train_play import _fast_action_for_policy  # noqa: E402
def _matchup_from_metadata(meta: dict) -> Matchup:
    return Matchup(
        name=str(meta.get("matchup", "checkpoint_matchup")),
        p1_deck=str(meta.get("p1_deck", "")),
        p2_deck=str(meta.get("p2_deck", "")),
        description=str(meta.get("matchup", "checkpoint eval")),
        p1_hero=str(meta.get("p1_hero", "")),
        p2_hero=str(meta.get("p2_hero", "")),
        cpp_engine_deck1=meta.get("cpp_engine_deck1"),
        cpp_engine_deck2=meta.get("cpp_engine_deck2"),
        cpp_engine_dir=meta.get("cpp_engine_dir"),
    )


def _load_policy(checkpoint_dir: Path) -> PPOAgent:
    meta_path = checkpoint_dir / "metadata.json"
    weights_path = checkpoint_dir / "weights" / "agent_weights.json"
    if not weights_path.is_file():
        weights_path = checkpoint_dir / "agent_weights.json"
    policy = PPOAgent(n_actions=ACTION_CAPACITY, obs_dim=PLAYER_OBS_DIM)
    policy._init_nets(PLAYER_OBS_DIM)
    policy.load(weights_path)
    return policy


def _summarize(label: str, outcomes: list[str]) -> dict[str, float | int | str]:
    wins = sum(1 for o in outcomes if o == "win")
    losses = sum(1 for o in outcomes if o == "loss")
    total = max(1, len(outcomes))
    return {
        "label": label,
        "games": len(outcomes),
        "p1_seat_wins": wins,
        "p1_seat_losses": losses,
        "p1_seat_win_rate": wins / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose seat vs deck win-rate skew.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--game-format", default="silver_age")
    parser.add_argument("--base-url", default="http://localhost:8080/game")
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir.expanduser().resolve()
    meta_path = ckpt_dir / "metadata.json"
    if not meta_path.is_file():
        raise SystemExit(f"Missing metadata.json in {ckpt_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    matchup = _matchup_from_metadata(meta)
    policy = _load_policy(ckpt_dir)
    eval_p2 = PPOAgent(n_actions=ACTION_CAPACITY, obs_dim=PLAYER_OBS_DIM)
    eval_p2._init_nets(PLAYER_OBS_DIM)
    clone_agent_weights(policy, eval_p2)

    env = make_env(
        matchup,
        base_url=args.base_url,
        game_format=args.game_format,
        max_turns=args.max_steps,
        use_cpp_engine=True,
        require_fast_training=True,
    )
    swap_env = make_env(
        swapped_matchup(matchup),
        base_url=args.base_url,
        game_format=args.game_format,
        max_turns=args.max_steps,
        use_cpp_engine=True,
        require_fast_training=True,
    )
    if not _env_supports_fast_training(env):
        raise SystemExit("C++ fast training path unavailable for this matchup")

    p1_hero = meta.get("p1_hero", matchup.p1_hero)
    p2_hero = meta.get("p2_hero", matchup.p2_hero)

    even_outcomes: list[str] = []  # Boltyn (nominal p1 deck) in P1 seat
    odd_outcomes: list[str] = []   # Dorinthea in P1 seat (swapped)
    p1_starts: list[str] = []
    p2_starts: list[str] = []

    import numpy as np

    for ep in range(args.episodes):
        ep_seed = ep
        use_swap = ep % 2 == 1
        starting_player_id = 1 + (ep % 2)
        active_env = swap_env if use_swap else env
        p1_rng = np.random.default_rng((ep_seed * 31 + 7))
        p2_rng = np.random.default_rng((ep_seed * 31 + 13))
        state = active_env.fast_reset(seed=ep_seed, starting_player_id=starting_player_id)
        steps = 0
        terminated = truncated = False
        while steps < args.max_steps:
            acting = int(state.get("acting_player_id", 1) or 1)
            seat_policy = policy if acting == 1 else eval_p2
            seat_rng = p1_rng if acting == 1 else p2_rng
            obs_vec = np.asarray(state["obs_vec"], dtype=np.float64)
            n_legal = max(1, int(state.get("legal_count", 1) or 1))
            action = _fast_action_for_policy(
                seat_policy,
                active_env,
                obs_vec=obs_vec,
                n_legal=n_legal,
                rng=seat_rng,
            )
            state = active_env.fast_step_index(action)
            terminated = bool(state.get("terminated", False))
            truncated = bool(state.get("truncated", False))
            steps += 1
            if terminated or truncated:
                break
        max_steps_reached = not terminated and not truncated and steps >= args.max_steps
        outcome, _ = classify_p1_fast_episode_outcome(
            state,
            max_steps_reached=max_steps_reached,
        )
        bucket = odd_outcomes if use_swap else even_outcomes
        bucket.append(outcome)
        if starting_player_id == 1:
            p1_starts.append(outcome)
        else:
            p2_starts.append(outcome)

    report = {
        "checkpoint_dir": str(ckpt_dir),
        "matchup": matchup.name,
        "p1_hero_nominal": p1_hero,
        "p2_hero_nominal": p2_hero,
        "episodes": args.episodes,
        "interpretation": {
            "even_episodes": f"{p1_hero} in P1 seat (no deck swap)",
            "odd_episodes": f"{p2_hero} in P1 seat (deck swapped)",
        },
        "splits": [
            _summarize(f"even — {p1_hero} @ P1 seat", even_outcomes),
            _summarize(f"odd — {p2_hero} @ P1 seat", odd_outcomes),
            _summarize("P1 started", p1_starts),
            _summarize("P2 started", p2_starts),
            _summarize("all games", even_outcomes + odd_outcomes),
        ],
    }

    even_wr = report["splits"][0]["p1_seat_win_rate"]
    odd_wr = report["splits"][1]["p1_seat_win_rate"]
    all_wr = report["splits"][-1]["p1_seat_win_rate"]
    if even_wr < 0.15 and odd_wr < 0.15:
        report["diagnosis"] = (
            "SEAT_BIAS: both decks lose from P1 seat — policy plays poorly as engine P1 "
            "regardless of hero/deck (not a deck-strength or eval-metric artifact)."
        )
    elif even_wr > 0.85 and odd_wr > 0.85:
        report["diagnosis"] = (
            "ENGINE_ASYMMETRY: P1 seat wins with both decks — likely C++ engine / deck load "
            "issue (not learned policy bias). Random play would show the same skew."
        )
    elif even_wr < 0.15 and odd_wr > 0.85:
        report["diagnosis"] = (
            f"DECK_STRENGTH: {p2_hero} beats {p1_hero} when sides are nominal; "
            "P1 seat win rate recovers when the stronger deck sits in P1."
        )
    elif even_wr > 0.85 and odd_wr < 0.15:
        report["diagnosis"] = (
            f"DECK_STRENGTH: {p1_hero} beats {p2_hero} in nominal seats; "
            "skew flips when decks swap."
        )
    else:
        report["diagnosis"] = "MIXED: no clean seat-only or deck-only pattern in this sample."

    print(json.dumps(report, indent=2))
    env.close()
    swap_env.close()


if __name__ == "__main__":
    main()
