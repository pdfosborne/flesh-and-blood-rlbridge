#!/usr/bin/env python3
"""Evaluate two trained PPO agents against each other in Talishar self-play mode.

P1 (Briar) and P2 (Dorinthea) each use their own trained policy, routed by
actingPlayerID at each step.  Results: win rates, avg reward, episode logs.

Usage:
    TALISHAR_URL=http://localhost:8080/game python scripts/eval_agent_vs_agent.py \\
        --p1-weights results/sage_precon_agents/briar-vs-dorinthea/ppo_.../weights/agent_weights.json \\
        --p2-weights results/sage_precon_agents/dorinthea-vs-briar/ppo_.../weights/agent_weights.json \\
        --episodes 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAB_SRC = _REPO_ROOT / "src"
_RL_SRC = Path("~/Documents/RL/rlbridge/src").expanduser()

for p in (_FAB_SRC, _REPO_ROOT, _RL_SRC):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from flesh_and_blood_rlbridge.talishar_engine_environment import (
    TalisharEngineEnvironment,
    parse_acting_player_id,
    run_talishar_eval_episode,
    talishar_deck_player_won,
)
from rl_agents.ppo import PPOAgent


def load_agent(weights_path: str) -> PPOAgent:
    agent = PPOAgent()
    agent.load(weights_path)
    return agent


def run_episode(
    env: TalisharEngineEnvironment,
    p1_agent: PPOAgent,
    p2_agent: PPOAgent,
    max_steps: int,
    seed: int | None = None,
    verbose: bool = False,
) -> dict:
    if not verbose:
        out = run_talishar_eval_episode(
            env,
            p1_agent,
            max_steps,
            seed,
            p2_agent=p2_agent,
            deck_player_id=1,
        )
        return {
            "total_reward": out["total_reward"],
            "steps": out["steps"],
            "terminated": out["terminated"],
            "truncated": out["truncated"],
            "p1_won": out.get("deck_player_won"),
        }

    result = env.reset(seed=seed)
    obs = result.observation
    total_reward = 0.0
    steps = 0
    step_result = None

    while steps < max_steps:
        acting = parse_acting_player_id(env, obs)
        agent = p1_agent if acting == 1 else p2_agent
        action = agent.act_greedy(obs)

        step_result = env.step(action)
        total_reward += step_result.reward
        steps += 1
        obs = step_result.observation

        info = step_result.info
        print(
            f"  step {steps:3d}  P{acting} acts  "
            f"r={step_result.reward:+.3f}  "
            f"P1 HP={info.get('player_hp','?')}  "
            f"P2 HP={info.get('opponent_hp','?')}"
        )

        if step_result.terminated or step_result.truncated:
            break

    return {
        "total_reward": total_reward,
        "steps": steps,
        "terminated": step_result.terminated if step_result else False,
        "truncated": step_result.truncated if step_result else False,
        "p1_won": talishar_deck_player_won(
            obs,
            deck_player_id=1,
            terminated=bool(step_result and step_result.terminated),
            truncated=bool(step_result and step_result.truncated),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate two PPO agents head-to-head in Talishar self-play."
    )
    parser.add_argument("--p1-weights", required=True, help="Briar agent weights path.")
    parser.add_argument("--p2-weights", required=True, help="Dorinthea agent weights path.")
    parser.add_argument("--p1-deck", default="BriarSAGEPrecon")
    parser.add_argument("--p2-deck", default="DorintheSAGEPrecon")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--format", default="sage")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", default=None, help="Save JSON results to this path.")
    args = parser.parse_args()

    base_url = os.environ.get("TALISHAR_URL", "http://localhost:8080/game")

    print(f"Loading P1 agent from {args.p1_weights}")
    p1 = load_agent(args.p1_weights)
    print(f"Loading P2 agent from {args.p2_weights}")
    p2 = load_agent(args.p2_weights)

    env = TalisharEngineEnvironment(
        base_url=base_url,
        local_deck_name=args.p1_deck,
        opponent_deck_name=args.p2_deck,
        game_format=args.format,
        self_play=True,
        max_turns=args.max_steps,
    )

    print(f"\nEvaluating {args.p1_deck} (P1) vs {args.p2_deck} (P2)")
    print(f"{args.episodes} episodes | max {args.max_steps} steps | format={args.format}")
    print("-" * 60)

    episodes = []
    p1_wins = p2_wins = draws = 0

    for ep in range(1, args.episodes + 1):
        seed = args.seed + ep if args.seed is not None else None
        if args.verbose:
            print(f"\nEpisode {ep}:")
        ep_result = run_episode(env, p1, p2, args.max_steps, seed=seed, verbose=args.verbose)
        episodes.append(ep_result)

        p1_won = ep_result.get("p1_won")
        r = ep_result["total_reward"]
        if p1_won is True:
            p1_wins += 1
            outcome = "P1 WIN"
        elif p1_won is False:
            p2_wins += 1
            outcome = "P2 WIN"
        else:
            draws += 1
            outcome = "draw"

        print(
            f"  ep {ep:3d}  {outcome:<8}  "
            f"reward={r:+.3f}  steps={ep_result['steps']:3d}"
        )

    env.close()

    total = args.episodes
    print(f"\n{'='*60}")
    print(f"  P1 ({args.p1_deck}) wins : {p1_wins}/{total}  ({100*p1_wins/total:.0f}%)")
    print(f"  P2 ({args.p2_deck}) wins : {p2_wins}/{total}  ({100*p2_wins/total:.0f}%)")
    print(f"  Draws                : {draws}/{total}  ({100*draws/total:.0f}%)")
    avg_r = sum(e["total_reward"] for e in episodes) / total
    print(f"  Avg reward (P1 POV)  : {avg_r:+.3f}")

    if args.out:
        summary = {
            "p1_deck": args.p1_deck,
            "p2_deck": args.p2_deck,
            "episodes": total,
            "p1_wins": p1_wins,
            "p2_wins": p2_wins,
            "draws": draws,
            "avg_reward": avg_r,
            "episode_log": episodes,
        }
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"  Results saved → {args.out}")


if __name__ == "__main__":
    main()
