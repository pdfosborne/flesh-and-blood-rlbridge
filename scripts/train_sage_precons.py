#!/usr/bin/env python3
"""Train PPO agents for every SAGE precon cross-matchup using dual-agent self-play.

"Self-play" here means both sides of every game are controlled by learning agents —
not a mirror match.  Each matchup pits two different SAGE precon decks (P1 vs P2).
The Talishar server runs the game with both players AI-controlled (no CombatDummy),
and a separate PPO agent trains from each player's perspective simultaneously.
Every episode produces gradient updates for both the P1 agent and the P2 agent.

45 unique matchups are generated across 10 SAGE precon decks (C(10,2) pairs).
Each matchup saves two trained agents:
  results/sage_precon_agents/<p1>-vs-<p2>/ppo_<matchup>-p1-<id>/
  results/sage_precon_agents/<p1>-vs-<p2>/ppo_<matchup>-p2-<id>/

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

import numpy as np  # noqa: E402

from flesh_and_blood_rlbridge.talishar_engine_environment import TalisharEngineEnvironment  # noqa: E402
from rlbridge.rl_agents.ppo import (  # noqa: E402
    PPOAgent,
    _gae,
    _get,
    _log_softmax,
    _n_legal_of,
    _softmax,
    _to_env_action,
    _infer_action_capacity,
    _discrete_n,
)

# ── matchup definitions ──────────────────────────────────────────────────────

@dataclass
class Matchup:
    name: str
    p1_deck: str
    p2_deck: str
    description: str
    tags: list[str] = field(default_factory=list)


# All available SAGE precon decks (hero_slug, deck_name)
SAGE_DECKS: list[tuple[str, str]] = [
    ("briar",     "BriarSAGEPrecon"),
    ("dorinthea", "DorintheSAGEPrecon"),
    ("kayo",      "KayoSAGEPrecon"),
    ("viserai",   "ViseraiSAGEPrecon"),
    ("iyslander", "IyslanderSAGEPrecon"),
    ("dash",      "DashSAGEPrecon"),
    ("fai",       "FaiSAGEPrecon"),
    ("azalea",    "AzaleaSAGEPrecon"),
    ("boltyn",    "BoltynSAGEPrecon"),
    ("enigma",    "EnigmaSAGEPrecon"),
]

# Every unique ordered pair (45 matchups across 10 heroes)
SAGE_MATCHUPS: list[Matchup] = [
    Matchup(
        name=f"{h1}-vs-{h2}",
        p1_deck=d1,
        p2_deck=d2,
        description=f"{h1.capitalize()} (P1) vs {h2.capitalize()} (P2) — dual-agent training, both perspectives",
        tags=[h1, h2, "cross"],
    )
    for i, (h1, d1) in enumerate(SAGE_DECKS)
    for h2, d2 in SAGE_DECKS[i + 1:]
]

MATCHUP_INDEX = {m.name: m for m in SAGE_MATCHUPS}

# Cached rlbridge environment used for evaluation / policy render after training.
# Must be loaded via rl_load_cached_environments before running eval/render tools.
# Derived from SAGE_MATCHUPS — each matchup maps to a canonical rlbridge env ID.
# The env ID follows the pattern FaB-{P1Hero}-vs-{P2Hero}-SAGE-v0.
# Cached environments must exist under this ID for rl_load_agent eval/render to work;
# agents save this ID in their metadata regardless so it's ready when the env is registered.
_DECK_TO_HERO: dict[str, str] = {deck: hero for hero, deck in SAGE_DECKS}

EVAL_ENV_IDS: dict[str, str] = {
    m.name: (
        f"FaB-{_DECK_TO_HERO[m.p1_deck].capitalize()}"
        f"-vs-{_DECK_TO_HERO[m.p2_deck].capitalize()}-SAGE-v0"
    )
    for m in SAGE_MATCHUPS
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
    """Create a Talishar environment with both sides AI-controlled (no CombatDummy).

    self_play=True disables the server-side CombatDummy so our agents drive both
    P1 and P2.  The two decks are always different cross-matchup heroes.
    """
    return TalisharEngineEnvironment(
        base_url=base_url,
        local_deck_name=matchup.p1_deck,
        opponent_deck_name=matchup.p2_deck,
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


def _ppo_update(agent: PPOAgent, buf: dict, next_obs_vec: np.ndarray) -> None:
    """Run one PPO update cycle on `buf` using the agent's internal actor/critic."""
    T = len(buf["obs"])
    if T == 0:
        return

    obs_arr     = np.array(buf["obs"],       dtype=np.float64)
    act_arr     = np.array(buf["actions"],   dtype=np.int64)
    values_arr  = np.array(buf["values"],    dtype=np.float64)
    log_old_arr = np.array(buf["log_probs"], dtype=np.float64)
    dones_arr   = np.array(buf["dones"],     dtype=np.float64)
    nlegal_arr  = np.array(buf["n_legal"],   dtype=np.int64)

    next_val = float(agent._critic.predict(next_obs_vec[None, :]).flatten()[0])
    next_vals_arr = np.append(values_arr[1:], next_val)

    advantages, returns = _gae(
        np.array(buf["rewards"], dtype=np.float64),
        values_arr, next_vals_arr, dones_arr,
        agent.gamma, agent.lam,
    )
    if T > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    indices = np.arange(T)
    for _ in range(agent.ppo_epochs):
        agent._rng_np.shuffle(indices)
        for start in range(0, T, agent.mini_batch_size):
            mb_idx = indices[start: start + agent.mini_batch_size]
            if len(mb_idx) == 0:
                continue
            mb_obs      = obs_arr[mb_idx]
            mb_acts     = act_arr[mb_idx]
            mb_adv      = advantages[mb_idx]
            mb_ret      = returns[mb_idx]
            mb_lp_old   = log_old_arr[mb_idx]
            mb_nlegal   = nlegal_arr[mb_idx]

            logits_new = agent._actor.forward(mb_obs)
            B = mb_obs.shape[0]
            legal_mask = None
            if agent._mask_actions:
                legal_mask = np.arange(agent.n_actions)[None, :] < mb_nlegal[:, None]
                logits_new = np.where(legal_mask, logits_new, -1e9)
            log_probs_new = _log_softmax(logits_new)
            probs_new     = _softmax(logits_new)
            lp_new = log_probs_new[np.arange(B), mb_acts]

            ratio  = np.exp(lp_new - mb_lp_old)
            clip_r = np.clip(ratio, 1.0 - agent.clip_eps, 1.0 + agent.clip_eps)

            used_ratio = np.where(ratio <= clip_r, ratio, clip_r)
            grad_logp  = -(used_ratio * mb_adv) / B
            grad_logits = np.zeros_like(logits_new)
            grad_logits[np.arange(B), mb_acts] += grad_logp
            ent_grad = probs_new * (log_probs_new + 1.0) - \
                       (probs_new * (log_probs_new + 1.0)).sum(axis=1, keepdims=True)
            grad_logits += agent.c_ent * ent_grad
            if legal_mask is not None:
                grad_logits = np.where(legal_mask, grad_logits, 0.0)
            agent._actor.backward(grad_logits)

            val_pred = agent._critic.forward(mb_obs).flatten()
            grad_val = 2.0 * (val_pred - mb_ret) / B
            agent._critic.backward(agent.c_vf * grad_val[:, None])


def _empty_buf() -> dict:
    return {"obs": [], "actions": [], "rewards": [], "values": [],
            "log_probs": [], "dones": [], "n_legal": []}


def train_agents_from_both_perspectives(
    env: TalisharEngineEnvironment,
    p1_agent: PPOAgent,
    p2_agent: PPOAgent,
    n_episodes: int,
    max_steps: int,
    seed: Optional[int] = None,
) -> tuple[list[float], list[float]]:
    """Train two agents simultaneously, each learning from its own player's perspective.

    The environment runs with both sides AI-controlled (self_play=True — no CombatDummy).
    At each step we check which player has priority and route the decision to the
    corresponding agent.  Rewards are split by perspective: P1 agent receives the
    environment reward directly; P2 agent receives the negated reward (zero-sum).
    Each agent maintains its own rollout buffer and fires PPO updates independently
    when its buffer fills.  Every game episode advances learning for both agents.
    """
    n_actions_p1, mask_p1 = _infer_action_capacity(env, seed=seed)
    n_actions_p2, mask_p2 = _infer_action_capacity(env, seed=seed)

    for agent, n_actions, mask in [
        (p1_agent, n_actions_p1, mask_p1),
        (p2_agent, n_actions_p2, mask_p2),
    ]:
        agent.n_actions    = n_actions
        agent._mask_actions = mask

    p1_ep_rewards: list[float] = []
    p2_ep_rewards: list[float] = []
    p1_buf = _empty_buf()
    p2_buf = _empty_buf()

    ep_seed = seed
    reset_out = env.reset(seed=ep_seed)
    obs = _get(reset_out, "observation", reset_out)

    # Lazy-init both agents on first observation
    obs_vec = p1_agent._obs_to_vec(obs)
    p1_agent._init_nets(obs_vec.shape[0])
    p2_agent._init_nets(obs_vec.shape[0])

    completed = 0
    cur_p1_r = cur_p2_r = 0.0

    total_steps = n_episodes * max_steps
    global_step = 0

    while completed < n_episodes and global_step < total_steps:
        acting = env._acting_player_id
        agent  = p1_agent if acting == 1 else p2_agent
        buf    = p1_buf   if acting == 1 else p2_buf

        obs_vec = agent._obs_to_vec(obs)
        logits  = agent._masked_logits(agent._actor.forward(obs_vec[None, :]), obs)
        lp_all  = _log_softmax(logits)[0]
        probs   = _softmax(logits)[0]
        action  = int(agent._rng_np.choice(agent.n_actions, p=probs))
        value   = float(agent._critic.predict(obs_vec[None, :]).flatten()[0])
        n_legal = _n_legal_of(obs)
        env_action = _to_env_action(obs, action, agent._mask_actions)

        step_out   = env.step(env_action)
        env_reward = float(_get(step_out, "reward", 0.0))
        terminated = bool(_get(step_out, "terminated", False))
        truncated  = bool(_get(step_out, "truncated", False))
        done = terminated or truncated

        # P1 gets env reward directly; P2 gets negated (zero-sum)
        agent_reward = env_reward if acting == 1 else -env_reward
        if acting == 1:
            cur_p1_r += env_reward
        else:
            cur_p2_r += -env_reward  # from P2's own perspective

        buf["obs"].append(obs_vec)
        buf["actions"].append(action)
        buf["rewards"].append(agent_reward)
        buf["values"].append(value)
        buf["log_probs"].append(float(lp_all[action]))
        buf["dones"].append(float(done))
        buf["n_legal"].append(n_legal if n_legal is not None else agent.n_actions)

        global_step += 1

        if done:
            p1_ep_rewards.append(cur_p1_r)
            p2_ep_rewards.append(cur_p2_r)
            completed += 1
            cur_p1_r = cur_p2_r = 0.0
            ep_seed = (seed + completed) if seed is not None else None
            reset_out = env.reset(seed=ep_seed)
            obs = _get(reset_out, "observation", reset_out)
        else:
            obs = _get(step_out, "observation", obs)

        # Fire independent PPO updates when a buffer is full
        for agt, buf_ref in [(p1_agent, p1_buf), (p2_agent, p2_buf)]:
            if len(buf_ref["obs"]) >= agt.n_steps:
                next_vec = agt._obs_to_vec(_get(step_out, "observation", obs))
                _ppo_update(agt, buf_ref, next_vec)
                buf_ref.clear()
                buf_ref.update(_empty_buf())

    # Final update on any remaining buffer data
    for agt, buf_ref in [(p1_agent, p1_buf), (p2_agent, p2_buf)]:
        if len(buf_ref["obs"]) > 0:
            next_vec = agt._obs_to_vec(obs)
            _ppo_update(agt, buf_ref, next_vec)

    return p1_ep_rewards, p2_ep_rewards


def _save_agent(
    agent: PPOAgent,
    agent_id: str,
    out_dir: Path,
    matchup: Matchup,
    episode_rewards: list[float],
    n_episodes: int,
    elapsed: float,
    game_format: str,
    role: str,
) -> dict:
    """Save a trained agent in rlbridge package format."""
    eval_env_id = EVAL_ENV_IDS.get(matchup.name, "")
    avg_reward  = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    best_reward = float(max(episode_rewards)) if episode_rewards else 0.0

    package_dir = out_dir / matchup.name / f"ppo_{agent_id}"
    weights_dir = package_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    agent.save(weights_dir / "agent_weights.json")
    (package_dir / "metadata.json").write_text(json.dumps({
        "agent_id":        agent_id,
        "agent_type":      "ppo",
        "env_id":          eval_env_id,
        "created_at":      datetime.now().isoformat(),
        "weights_file":    "agent_weights.json",
        "training_config": {"hidden_size": DEFAULT_HIDDEN_SIZE},
        "use_language_state": False,
        "matchup":         matchup.name,
        "role":            role,
        "p1_deck":         matchup.p1_deck,
        "p2_deck":         matchup.p2_deck,
        "training_mode":   "both_perspectives",
        "game_format":     game_format,
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

    print(f"  [{role}] Saved → {package_dir}  (avg={avg_reward:+.3f}  best={best_reward:+.3f})")
    return {
        "matchup":      matchup.name,
        "role":         role,
        "agent_id":     agent_id,
        "eval_env_id":  eval_env_id,
        "package_dir":  str(package_dir),
        "elapsed_secs": round(elapsed, 1),
        "avg_reward":   avg_reward,
        "best_reward":  best_reward,
    }


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
    """Train a P1 and P2 agent from both perspectives of the same games and save both."""
    print(f"\n{'='*60}")
    print(f"  Matchup : {matchup.name}")
    print(f"  Decks   : {matchup.p1_deck} (P1) vs {matchup.p2_deck} (P2)")
    print(f"  Mode    : both-perspectives training | {n_episodes} episodes | max {max_steps} steps")
    print(f"{'='*60}")

    # SHMOP auto-recovery: restart Talishar Docker if CreateGame returns empty.
    for attempt in range(2):
        env = make_env(matchup, base_url=base_url, game_format=game_format)
        try:
            env.reset()
            break
        except Exception as exc:
            if attempt == 0:
                print(f"  CreateGame failed ({exc}), restarting Talishar Docker...")
                import subprocess
                subprocess.run(
                    ["docker", "compose", "restart"],
                    cwd=_REPO_ROOT / "Talishar", capture_output=True, check=False,
                )
                time.sleep(5)
            else:
                raise

    p1_agent = make_agent(seed=seed)
    p2_agent = make_agent(seed=(seed + 1) if seed is not None else None)

    t0 = time.time()
    p1_rewards, p2_rewards = train_agents_from_both_perspectives(
        env, p1_agent, p2_agent, n_episodes=n_episodes, max_steps=max_steps, seed=seed,
    )
    elapsed = time.time() - t0
    env.close()

    print(f"  Done in {elapsed:.1f}s — "
          f"P1 avg={np.mean(p1_rewards):+.3f}  "
          f"P2 avg={np.mean(p2_rewards):+.3f}")

    p1_id = f"{matchup.name}-p1-{uuid.uuid4().hex[:8]}"
    p2_id = f"{matchup.name}-p2-{uuid.uuid4().hex[:8]}"

    meta_p1 = _save_agent(p1_agent, p1_id, out_dir, matchup, p1_rewards,
                           n_episodes, elapsed, game_format, role="p1")
    meta_p2 = _save_agent(p2_agent, p2_id, out_dir, matchup, p2_rewards,
                           n_episodes, elapsed, game_format, role="p2")

    return {"p1": meta_p1, "p2": meta_p2}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO agents across all SAGE precon cross-matchups, learning from both perspectives."
    )
    parser.add_argument(
        "--matchup",
        default="all",
        help=(
            "Which matchup to train (default: all).  "
            "Use 'all' or a name like 'briar-vs-dorinthea'.  "
            f"Available: {', '.join(list(MATCHUP_INDEX)[:5])} ... ({len(MATCHUP_INDEX)} total)"
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

    if args.matchup == "all":
        matchups = SAGE_MATCHUPS
    elif args.matchup in MATCHUP_INDEX:
        matchups = [MATCHUP_INDEX[args.matchup]]
    else:
        print(f"Unknown matchup '{args.matchup}'. Available: {', '.join(list(MATCHUP_INDEX)[:10])} ...")
        sys.exit(1)

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
        for role in ("p1", "p2"):
            r = m[role]
            print(
                f"  {r['matchup']}/{role:<4}  avg={r['avg_reward']:+.3f}  "
                f"best={r['best_reward']:+.3f}  ({r['elapsed_secs']:.0f}s)\n"
                f"  {'agent_id':<28} {r['agent_id']}\n"
                f"  {'package_dir':<28} {r['package_dir']}"
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
