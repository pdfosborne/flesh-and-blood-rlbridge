"""Shared dual-agent PPO self-play training utilities for Talishar scripts."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
FAB_SRC = REPO_ROOT / "src"
RL_SRC = Path("~/Documents/RL/rlbridge/src").expanduser()
FAB_DB_DIR = FAB_SRC / "flesh_and_blood_rlbridge" / "card_db"
FABRARY_DECKS_PATH = FAB_DB_DIR / "fabrary_decks.json"
CARDS_DB_PATH = FAB_DB_DIR / "cards.json"
TALISHAR_ID_RE = re.compile(r"^[a-z][a-z0-9_]+$")

for p in (FAB_SRC, RL_SRC):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

import numpy as np  # noqa: E402

from flesh_and_blood_rlbridge.talishar_engine_environment import (  # noqa: E402
    TalisharEngineEnvironment,
    run_talishar_eval_episode,
)
from rlbridge.rl_agents.ppo import (  # noqa: E402
    PPOAgent,
    _gae,
    _get,
    _infer_action_capacity,
    _log_softmax,
    _n_legal_of,
    _softmax,
    _to_env_action,
)

FORMAT_DECK_RULES: dict[str, dict[str, int]] = {
    "silver_age": {"max_copies": 2, "deck_size": 40},
    "classic_constructed": {"max_copies": 3, "deck_size": 60},
}

DEFAULT_N_EPISODES = 300
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_LR = 3e-4
DEFAULT_GAMMA = 0.99
DEFAULT_LAM = 0.95
DEFAULT_CLIP_EPS = 0.2
DEFAULT_N_STEPS = 256
DEFAULT_PPO_EPOCHS = 4
DEFAULT_MINI_BATCH = 64
DEFAULT_WARMUP_EPISODES = 100
DEFAULT_WARMUP_BASELINE_EVAL_EPISODES = 20


@dataclass
class Matchup:
    name: str
    p1_deck: str
    p2_deck: str
    description: str
    tags: list[str] = field(default_factory=list)
    p1_hero: str = ""
    p2_hero: str = ""


def make_env(
    matchup: Matchup,
    base_url: str,
    game_format: str,
    max_turns: int,
) -> TalisharEngineEnvironment:
    return TalisharEngineEnvironment(
        base_url=base_url,
        local_deck_name=matchup.p1_deck,
        opponent_deck_name=matchup.p2_deck,
        game_format=game_format,
        self_play=True,
        max_turns=max_turns,
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
    T = len(buf["obs"])
    if T == 0:
        return

    obs_arr = np.array(buf["obs"], dtype=np.float64)
    act_arr = np.array(buf["actions"], dtype=np.int64)
    values_arr = np.array(buf["values"], dtype=np.float64)
    log_old_arr = np.array(buf["log_probs"], dtype=np.float64)
    dones_arr = np.array(buf["dones"], dtype=np.float64)
    nlegal_arr = np.array(buf["n_legal"], dtype=np.int64)

    next_val = float(agent._critic.predict(next_obs_vec[None, :]).flatten()[0])
    next_vals_arr = np.append(values_arr[1:], next_val)

    advantages, returns = _gae(
        np.array(buf["rewards"], dtype=np.float64),
        values_arr,
        next_vals_arr,
        dones_arr,
        agent.gamma,
        agent.lam,
    )
    if T > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    indices = np.arange(T)
    for _ in range(agent.ppo_epochs):
        agent._rng_np.shuffle(indices)
        for start in range(0, T, agent.mini_batch_size):
            mb_idx = indices[start : start + agent.mini_batch_size]
            if len(mb_idx) == 0:
                continue
            mb_obs = obs_arr[mb_idx]
            mb_acts = act_arr[mb_idx]
            mb_adv = advantages[mb_idx]
            mb_ret = returns[mb_idx]
            mb_lp_old = log_old_arr[mb_idx]
            mb_nlegal = nlegal_arr[mb_idx]

            logits_new = agent._actor.forward(mb_obs)
            B = mb_obs.shape[0]
            legal_mask = None
            if agent._mask_actions:
                legal_mask = np.arange(agent.n_actions)[None, :] < mb_nlegal[:, None]
                logits_new = np.where(legal_mask, logits_new, -1e9)
            log_probs_new = _log_softmax(logits_new)
            probs_new = _softmax(logits_new)
            lp_new = log_probs_new[np.arange(B), mb_acts]

            ratio = np.exp(lp_new - mb_lp_old)
            clip_r = np.clip(ratio, 1.0 - agent.clip_eps, 1.0 + agent.clip_eps)

            used_ratio = np.where(ratio <= clip_r, ratio, clip_r)
            grad_logp = -(used_ratio * mb_adv) / B
            grad_logits = np.zeros_like(logits_new)
            grad_logits[np.arange(B), mb_acts] += grad_logp
            ent_grad = probs_new * (log_probs_new + 1.0) - (
                probs_new * (log_probs_new + 1.0)
            ).sum(axis=1, keepdims=True)
            grad_logits += agent.c_ent * ent_grad
            if legal_mask is not None:
                grad_logits = np.where(legal_mask, grad_logits, 0.0)
            agent._actor.backward(grad_logits)

            val_pred = agent._critic.forward(mb_obs).flatten()
            grad_val = 2.0 * (val_pred - mb_ret) / B
            agent._critic.backward(agent.c_vf * grad_val[:, None])


def _empty_buf() -> dict:
    return {
        "obs": [],
        "actions": [],
        "rewards": [],
        "values": [],
        "log_probs": [],
        "dones": [],
        "n_legal": [],
    }


def _sync_tier_agent_config(tier_agents: list[PPOAgent], n_actions: int, mask: bool) -> None:
    for agent in tier_agents:
        agent.n_actions = n_actions
        agent._mask_actions = mask


def _init_tier_nets(tier_agents: list[PPOAgent], obs_dim: int) -> None:
    for agent in tier_agents:
        agent._init_nets(obs_dim)


def _ppo_update_all_tiers(
    tier_agents: list[PPOAgent],
    buf: dict,
    next_obs_vec: np.ndarray,
) -> None:
    for agent in tier_agents:
        _ppo_update(agent, buf, next_obs_vec)


def train_agents_from_both_perspectives(
    env: TalisharEngineEnvironment,
    p1_tiers: list[PPOAgent],
    p2_tiers: list[PPOAgent],
    n_episodes: int,
    max_steps: int,
    seed: Optional[int] = None,
    warmup_episodes: int = DEFAULT_WARMUP_EPISODES,
) -> tuple[list[float], list[float]]:
    p1_policy = p1_tiers[0]
    p2_policy = p2_tiers[0]

    n_actions_p1, mask_p1 = _infer_action_capacity(env, seed=seed)
    n_actions_p2, mask_p2 = _infer_action_capacity(env, seed=seed)

    _sync_tier_agent_config(p1_tiers, n_actions_p1, mask_p1)
    _sync_tier_agent_config(p2_tiers, n_actions_p2, mask_p2)

    p1_ep_rewards: list[float] = []
    p2_ep_rewards: list[float] = []
    p1_buf = _empty_buf()
    p2_buf = _empty_buf()

    ep_seed = seed
    reset_out = env.reset(seed=ep_seed)
    obs = _get(reset_out, "observation", reset_out)

    obs_vec = p1_policy._obs_to_vec(obs)
    _init_tier_nets(p1_tiers, obs_vec.shape[0])
    _init_tier_nets(p2_tiers, obs_vec.shape[0])

    completed = 0
    cur_p1_r = cur_p2_r = 0.0
    total_steps = n_episodes * max_steps
    global_step = 0
    progress_every = max(1, n_episodes // 100)  # ~1% cadence after warmup
    progress_t0 = time.time()

    while completed < n_episodes and global_step < total_steps:
        acting = env._acting_player_id
        policy = p1_policy if acting == 1 else p2_policy
        tier_agents = p1_tiers if acting == 1 else p2_tiers
        buf = p1_buf if acting == 1 else p2_buf
        in_warmup = completed < warmup_episodes

        obs_vec = policy._obs_to_vec(obs)
        logits = policy._masked_logits(policy._actor.forward(obs_vec[None, :]), obs)
        lp_all = _log_softmax(logits)[0]
        probs = _softmax(logits)[0]
        if in_warmup:
            env_action = str(env.sample_action())
            try:
                action = int(env_action)
            except (TypeError, ValueError):
                action = 0
            action = max(0, min(action, policy.n_actions - 1))
        else:
            action = int(policy._rng_np.choice(policy.n_actions, p=probs))
            env_action = _to_env_action(obs, action, policy._mask_actions)
        value = float(policy._critic.predict(obs_vec[None, :]).flatten()[0])
        n_legal = _n_legal_of(obs)

        step_out = env.step(env_action)
        env_reward = float(_get(step_out, "reward", 0.0))
        terminated = bool(_get(step_out, "terminated", False))
        truncated = bool(_get(step_out, "truncated", False))
        done = terminated or truncated

        agent_reward = env_reward if acting == 1 else -env_reward
        if acting == 1:
            cur_p1_r += env_reward
        else:
            cur_p2_r += -env_reward

        buf["obs"].append(obs_vec)
        buf["actions"].append(action)
        buf["rewards"].append(agent_reward)
        buf["values"].append(value)
        buf["log_probs"].append(float(lp_all[action]))
        buf["dones"].append(float(done))
        buf["n_legal"].append(n_legal if n_legal is not None else policy.n_actions)

        global_step += 1

        if done:
            p1_ep_rewards.append(cur_p1_r)
            p2_ep_rewards.append(cur_p2_r)
            completed += 1

            if (
                completed <= 10  # dense startup visibility
                or completed == n_episodes
                or completed % progress_every == 0
            ):
                elapsed = time.time() - progress_t0
                pct = (completed / max(1, n_episodes)) * 100.0
                p1_avg = float(np.mean(p1_ep_rewards)) if p1_ep_rewards else 0.0
                p2_avg = float(np.mean(p2_ep_rewards)) if p2_ep_rewards else 0.0
                ep_rate = completed / max(elapsed, 1e-9)
                eta_secs = (n_episodes - completed) / ep_rate if ep_rate > 0 else float("inf")
                print(
                    f"  [train-progress] episodes={completed}/{n_episodes} "
                    f"({pct:6.2f}%) elapsed={elapsed:.1f}s "
                    f"rate={ep_rate:.3f}ep/s eta={eta_secs/60:.1f}m "
                    f"warmup={'yes' if completed < warmup_episodes else 'no '} "
                    f"p1_avg={p1_avg:+.3f} p2_avg={p2_avg:+.3f}"
                )

            cur_p1_r = cur_p2_r = 0.0
            ep_seed = (seed + completed) if seed is not None else None
            reset_out = env.reset(seed=ep_seed)
            obs = _get(reset_out, "observation", reset_out)
        else:
            obs = _get(step_out, "observation", obs)

        for tiers, buf_ref in [(p1_tiers, p1_buf), (p2_tiers, p2_buf)]:
            if len(buf_ref["obs"]) >= tiers[0].n_steps:
                next_vec = tiers[0]._obs_to_vec(_get(step_out, "observation", obs))
                _ppo_update_all_tiers(tiers, buf_ref, next_vec)
                buf_ref.clear()
                buf_ref.update(_empty_buf())

    for tiers, buf_ref in [(p1_tiers, p1_buf), (p2_tiers, p2_buf)]:
        if len(buf_ref["obs"]) > 0:
            next_vec = tiers[0]._obs_to_vec(obs)
            _ppo_update_all_tiers(tiers, buf_ref, next_vec)

    return p1_ep_rewards, p2_ep_rewards


def save_agent(
    agent: PPOAgent,
    agent_id: str,
    out_dir: Path,
    matchup: Matchup,
    episode_rewards: list[float],
    n_episodes: int,
    elapsed: float,
    game_format: str,
    role: str,
    eval_env_ids: dict[str, str],
    warmup_baseline: Optional[dict[str, Any]] = None,
) -> dict:
    eval_env_id = eval_env_ids.get(matchup.name, "")
    avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    best_reward = float(max(episode_rewards)) if episode_rewards else 0.0

    package_dir = out_dir / matchup.name / f"ppo_{agent_id}"
    weights_dir = package_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    agent.save(weights_dir / "agent_weights.json")
    (package_dir / "metadata.json").write_text(
        json.dumps(
            {
                "agent_id": agent_id,
                "agent_type": "ppo",
                "env_id": eval_env_id,
                "created_at": datetime.now().isoformat(),
                "weights_file": "agent_weights.json",
                "training_config": {"hidden_size": DEFAULT_HIDDEN_SIZE},
                "use_language_state": False,
                "matchup": matchup.name,
                "role": role,
                "p1_deck": matchup.p1_deck,
                "p2_deck": matchup.p2_deck,
                "training_mode": "both_perspectives",
                "game_format": game_format,
                "warmup_baseline": warmup_baseline,
            },
            indent=2,
        )
    )
    (package_dir / "training_results.json").write_text(
        json.dumps(
            {
                "agent_name": "ppo",
                "n_episodes": n_episodes,
                "episode_rewards": episode_rewards,
                "final_epsilon": 0.0,
                "eval_mean": avg_reward,
                "eval_std": 0.0,
                "eval_rewards": [],
                "warmup_baseline": warmup_baseline,
            },
            indent=2,
        )
    )

    print(
        f"  [{role}] Saved → {package_dir}  "
        f"(avg={avg_reward:+.3f}  best={best_reward:+.3f})"
    )
    return {
        "matchup": matchup.name,
        "role": role,
        "agent_id": agent_id,
        "eval_env_id": eval_env_id,
        "package_dir": str(package_dir),
        "elapsed_secs": round(elapsed, 1),
        "avg_reward": avg_reward,
        "best_reward": best_reward,
        "warmup_baseline": warmup_baseline,
    }


def _evaluate_policy_pair(
    matchup: Matchup,
    *,
    base_url: str,
    game_format: str,
    max_steps: int,
    p1_policy: PPOAgent,
    p2_policy: PPOAgent,
    episodes: int,
    seed: Optional[int],
) -> dict[str, Any]:
    """Evaluate current P1/P2 policies head-to-head and return win-rate summary."""
    if episodes <= 0:
        return {
            "episodes": 0,
            "p1_wins": 0,
            "p2_wins": 0,
            "draws": 0,
            "p1_win_rate": 0.0,
            "p2_win_rate": 0.0,
            "draw_rate": 0.0,
            "max_steps": max_steps,
        }

    env = make_env(matchup, base_url=base_url, game_format=game_format, max_turns=max_steps)
    p1_wins = 0
    p2_wins = 0
    draws = 0
    try:
        for ep in range(episodes):
            ep_seed = (seed + ep) if seed is not None else None
            out = run_talishar_eval_episode(
                env,
                p1_policy,
                max_steps=max_steps,
                seed=ep_seed,
                p2_agent=p2_policy,
                deck_player_id=1,
            )
            won = out.get("deck_player_won")
            if won is True:
                p1_wins += 1
            elif won is False:
                p2_wins += 1
            else:
                draws += 1
    finally:
        env.close()

    total = max(1, episodes)
    return {
        "episodes": episodes,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": draws,
        "p1_win_rate": p1_wins / total,
        "p2_win_rate": p2_wins / total,
        "draw_rate": draws / total,
        "max_steps": max_steps,
    }


def _save_warmup_handoff_checkpoint(
    *,
    out_dir: Path,
    matchup: Matchup,
    p1_policy: PPOAgent,
    p2_policy: PPOAgent,
    baseline: dict[str, Any],
) -> Path:
    """Persist policy weights + baseline evaluation right before PPO handoff."""
    ckpt_dir = out_dir / matchup.name / "warmup_handoff_baseline"
    p1_dir = ckpt_dir / "p1"
    p2_dir = ckpt_dir / "p2"
    p1_dir.mkdir(parents=True, exist_ok=True)
    p2_dir.mkdir(parents=True, exist_ok=True)

    p1_policy.save(p1_dir / "agent_weights.json")
    p2_policy.save(p2_dir / "agent_weights.json")
    (ckpt_dir / "baseline_metrics.json").write_text(
        json.dumps(baseline, indent=2),
        encoding="utf-8",
    )
    return ckpt_dir


def _player_context(matchup: Matchup, *, as_p1: bool) -> "PlayerCacheContext":
    from agent_cache import PlayerCacheContext

    if as_p1:
        return PlayerCacheContext(
            player_deck=matchup.p1_deck,
            player_hero=matchup.p1_hero,
            opponent_deck=matchup.p2_deck,
            opponent_hero=matchup.p2_hero,
        )
    return PlayerCacheContext(
        player_deck=matchup.p2_deck,
        player_hero=matchup.p2_hero,
        opponent_deck=matchup.p1_deck,
        opponent_hero=matchup.p1_hero,
    )


def train_matchup(
    matchup: Matchup,
    *,
    base_url: str,
    n_episodes: int,
    max_steps: int,
    out_dir: Path,
    eval_env_ids: dict[str, str],
    cache_store: Optional["AgentCacheStore"] = None,
    seed: Optional[int] = None,
    game_format: str = "sage",
    warmup_episodes: int = DEFAULT_WARMUP_EPISODES,
    warmup_baseline_eval_episodes: int = DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
) -> dict:
    from agent_cache import AgentCacheStore
    print(f"\n{'=' * 60}")
    print(f"  Matchup : {matchup.name}")
    print(f"  Decks   : {matchup.p1_deck} (P1) vs {matchup.p2_deck} (P2)")
    print(
        f"  Mode    : both-perspectives training | {n_episodes} episodes | "
        f"max {max_steps} steps | warmup {warmup_episodes} episodes"
    )
    print(f"{'=' * 60}")

    for attempt in range(2):
        env = make_env(matchup, base_url=base_url, game_format=game_format, max_turns=max_steps)
        try:
            env.reset()
            break
        except Exception as exc:
            if attempt == 0:
                print(f"  CreateGame failed ({exc}), restarting Talishar Docker...")
                subprocess.run(
                    ["docker", "compose", "restart"],
                    cwd=REPO_ROOT / "Talishar",
                    capture_output=True,
                    check=False,
                )
                time.sleep(5)
            else:
                raise

    if cache_store is None:
        cache_store = AgentCacheStore(REPO_ROOT / "results" / "agent_cache", game_format)

    def _make_p1() -> PPOAgent:
        return make_agent(seed=seed)

    def _make_p2() -> PPOAgent:
        return make_agent(seed=(seed + 1) if seed is not None else None)

    p1_bundle = cache_store.bootstrap_player(_player_context(matchup, as_p1=True), _make_p1)
    p2_bundle = cache_store.bootstrap_player(_player_context(matchup, as_p1=False), _make_p2)

    print("  P1 cache init:", ", ".join(p1_bundle.init_sources))
    print("  P2 cache init:", ", ".join(p2_bundle.init_sources))

    t0 = time.time()
    warmup_baseline: Optional[dict[str, Any]] = None
    warmup_count = max(0, min(warmup_episodes, n_episodes))
    p1_rewards: list[float] = []
    p2_rewards: list[float] = []

    if warmup_count > 0:
        print(f"  Warmup  : training first {warmup_count} episode(s) with Talishar default policy")
        warm_p1, warm_p2 = train_agents_from_both_perspectives(
            env,
            p1_bundle.agents,
            p2_bundle.agents,
            n_episodes=warmup_count,
            max_steps=max_steps,
            seed=seed,
            warmup_episodes=warmup_count,
        )
        p1_rewards.extend(warm_p1)
        p2_rewards.extend(warm_p2)

        print(
            f"  Warmup baseline eval: {warmup_baseline_eval_episodes} episode(s) before PPO handoff"
        )
        baseline = _evaluate_policy_pair(
            matchup,
            base_url=base_url,
            game_format=game_format,
            max_steps=max_steps,
            p1_policy=p1_bundle.policy,
            p2_policy=p2_bundle.policy,
            episodes=warmup_baseline_eval_episodes,
            seed=(seed + 100_000) if seed is not None else None,
        )
        ckpt_dir = _save_warmup_handoff_checkpoint(
            out_dir=out_dir,
            matchup=matchup,
            p1_policy=p1_bundle.policy,
            p2_policy=p2_bundle.policy,
            baseline=baseline,
        )
        warmup_baseline = {
            **baseline,
            "checkpoint_dir": str(ckpt_dir),
        }
        print(
            "  Warmup baseline: "
            f"P1 win%={baseline['p1_win_rate'] * 100:.1f} "
            f"P2 win%={baseline['p2_win_rate'] * 100:.1f} "
            f"draw%={baseline['draw_rate'] * 100:.1f}"
        )
        print(f"  Warmup checkpoint saved → {ckpt_dir}")

    remaining_episodes = n_episodes - warmup_count
    if remaining_episodes > 0:
        print(f"  PPO     : training remaining {remaining_episodes} episode(s) with agent policy")
        rem_seed = (seed + warmup_count) if seed is not None else None
        rem_p1, rem_p2 = train_agents_from_both_perspectives(
            env,
            p1_bundle.agents,
            p2_bundle.agents,
            n_episodes=remaining_episodes,
            max_steps=max_steps,
            seed=rem_seed,
            warmup_episodes=0,
        )
        p1_rewards.extend(rem_p1)
        p2_rewards.extend(rem_p2)

    elapsed = time.time() - t0
    env.close()

    cache_store.persist_player(p1_bundle)
    cache_store.persist_player(p2_bundle)

    print(
        f"  Done in {elapsed:.1f}s — "
        f"P1 avg={np.mean(p1_rewards):+.3f}  "
        f"P2 avg={np.mean(p2_rewards):+.3f}"
    )

    p1_id = f"{matchup.name}-p1-{uuid.uuid4().hex[:8]}"
    p2_id = f"{matchup.name}-p2-{uuid.uuid4().hex[:8]}"

    meta_p1 = save_agent(
        p1_bundle.policy, p1_id, out_dir, matchup, p1_rewards,
        n_episodes, elapsed, game_format, "p1", eval_env_ids,
        warmup_baseline=warmup_baseline,
    )
    meta_p2 = save_agent(
        p2_bundle.policy, p2_id, out_dir, matchup, p2_rewards,
        n_episodes, elapsed, game_format, "p2", eval_env_ids,
        warmup_baseline=warmup_baseline,
    )
    return {"p1": meta_p1, "p2": meta_p2}


def print_training_summary(summary: list[dict], failed: list[str], out_dir: Path) -> None:
    print(f"\n{'=' * 60}")
    print("  TRAINING SUMMARY")
    print(f"{'=' * 60}")
    for m in summary:
        baseline = m.get("p1", {}).get("warmup_baseline")
        for role in ("p1", "p2"):
            r = m[role]
            print(
                f"  {r['matchup']}/{role:<4}  avg={r['avg_reward']:+.3f}  "
                f"best={r['best_reward']:+.3f}  ({r['elapsed_secs']:.0f}s)\n"
                f"  {'agent_id':<28} {r['agent_id']}\n"
                f"  {'package_dir':<28} {r['package_dir']}"
            )
        if isinstance(baseline, dict) and baseline.get("episodes", 0) > 0:
            print(
                "  "
                f"warmup baseline ({baseline['episodes']} ep): "
                f"P1 win%={baseline.get('p1_win_rate', 0.0) * 100:.1f} "
                f"P2 win%={baseline.get('p2_win_rate', 0.0) * 100:.1f} "
                f"draw%={baseline.get('draw_rate', 0.0) * 100:.1f}"
            )
            ckpt = baseline.get("checkpoint_dir")
            if ckpt:
                print(f"  {'warmup_checkpoint':<28} {ckpt}")
    for name in failed:
        print(f"  {name:<28} FAILED")

    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary saved → {out_dir / 'training_summary.json'}")


def run_matchup_training(
    matchups: list[Matchup],
    eval_env_ids: dict[str, str],
    *,
    base_url: str,
    n_episodes: int,
    max_steps: int,
    out_dir: Path,
    seed: Optional[int],
    game_format: str,
    cache_dir: Optional[Path] = None,
    warmup_episodes: int = DEFAULT_WARMUP_EPISODES,
    warmup_baseline_eval_episodes: int = DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
) -> tuple[list[dict], list[str]]:
    from agent_cache import AgentCacheStore

    cache_root = cache_dir or (REPO_ROOT / "results" / "agent_cache")
    cache_store = AgentCacheStore(cache_root, game_format)

    summary: list[dict] = []
    failed: list[str] = []
    for matchup in matchups:
        try:
            meta = train_matchup(
                matchup,
                base_url=base_url,
                n_episodes=n_episodes,
                max_steps=max_steps,
                out_dir=out_dir,
                eval_env_ids=eval_env_ids,
                cache_store=cache_store,
                seed=seed,
                game_format=game_format,
                warmup_episodes=warmup_episodes,
                warmup_baseline_eval_episodes=warmup_baseline_eval_episodes,
            )
            summary.append(meta)
        except Exception as exc:
            print(f"\n  ERROR training {matchup.name}: {exc}")
            failed.append(matchup.name)
    return summary, failed


# ── fabrary deck helpers ──────────────────────────────────────────────────────

def talishar_assets_path() -> Path:
    return Path(
        os.environ.get(
            "TALISHAR_ASSETS_PATH",
            str(REPO_ROOT / "Talishar" / "Assets"),
        )
    )


def load_fabrary_decks() -> list[dict]:
    data = json.loads(FABRARY_DECKS_PATH.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return list(data.get("decks", []))


def load_fabrary_decks_for_format(format_name: str) -> list[dict]:
    return [
        d for d in load_fabrary_decks()
        if str(d.get("format", "")) == format_name and d.get("id")
    ]


def _build_assets_hero_map(assets_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not assets_path.is_dir():
        return result
    for txt_file in sorted(assets_path.glob("*.txt")):
        try:
            first_line = txt_file.read_text(encoding="utf-8").splitlines()[0].strip()
            if not first_line:
                continue
            hero_id = first_line.split()[0]
            result[hero_id] = first_line
        except OSError:
            continue
    return result


def _get_hero_header(hero_id: str, assets_map: dict[str, str]) -> str:
    base = hero_id.removeprefix("hero_")
    return assets_map.get(hero_id) or assets_map.get(base) or base


def resolve_fabrary_deck_cards(deck_entry: dict, format_name: str) -> list[str]:
    rules = FORMAT_DECK_RULES[format_name]
    max_copies = rules["max_copies"]
    deck_size = rules["deck_size"]

    cards_data = json.loads(CARDS_DB_PATH.read_text(encoding="utf-8"))
    name_map: dict[str, list[dict]] = {}
    for c in cards_data:
        cid = c.get("id", "")
        if not TALISHAR_ID_RE.match(cid):
            continue
        name = c.get("name", "").lower()
        name_map.setdefault(name, []).append(c)
    for entry_list in name_map.values():
        entry_list.sort(key=lambda c: c.get("pitch") or 99)

    result: list[str] = []
    unresolved: list[str] = []
    for card_entry in deck_entry.get("cards", []):
        name = str(card_entry.get("name", "")).lower()
        count = min(int(card_entry.get("count", 1)), max_copies)
        candidates = name_map.get(name)
        if candidates:
            result.extend([candidates[0]["id"]] * count)
        else:
            unresolved.append(card_entry.get("name", name))
    if unresolved:
        print(
            f"  Warning: {len(unresolved)} card(s) not resolved for "
            f"{deck_entry.get('id', '?')}: {', '.join(str(u) for u in unresolved[:5])}"
            + (" ..." if len(unresolved) > 5 else "")
        )
    if len(result) > deck_size:
        result = result[:deck_size]
    return result


def write_fabrary_deck_file(
    deck_entry: dict,
    assets_path: Path,
    assets_map: dict[str, str],
    format_name: str,
) -> Optional[str]:
    deck_id = str(deck_entry["id"])
    hero_id = str(deck_entry.get("hero_id", ""))
    hero_header = _get_hero_header(hero_id, assets_map)
    card_ids = resolve_fabrary_deck_cards(deck_entry, format_name)
    if not card_ids:
        print(f"  Error: deck {deck_id} resolved to zero cards; skipping.")
        return None
    out_path = assets_path / f"{deck_id}.txt"
    out_path.write_text(f"{hero_header}\n{' '.join(card_ids)}\n", encoding="utf-8")
    return deck_id


def materialize_fabrary_decks(format_name: str) -> list[tuple[str, str, dict]]:
    """Write fabrary decks for *format_name* to Talishar Assets.

    Returns list of (matchup_slug, deck_stem, deck_entry).
    """
    assets_path = talishar_assets_path()
    assets_path.mkdir(parents=True, exist_ok=True)
    assets_map = _build_assets_hero_map(assets_path)

    decks: list[tuple[str, str, dict]] = []
    for entry in load_fabrary_decks_for_format(format_name):
        deck_stem = write_fabrary_deck_file(entry, assets_path, assets_map, format_name)
        if deck_stem is None:
            continue
        slug = deck_stem.removeprefix("fab_")
        decks.append((slug, deck_stem, entry))
    return decks


def build_fabrary_matchups(
    decks: list[tuple[str, str, dict]],
    format_name: str,
) -> list[Matchup]:
    matchups: list[Matchup] = []
    for i, (slug1, stem1, entry1) in enumerate(decks):
        for slug2, stem2, entry2 in decks[i + 1 :]:
            matchups.append(
                Matchup(
                    name=f"{slug1}-vs-{slug2}",
                    p1_deck=stem1,
                    p2_deck=stem2,
                    description=(
                        f"{entry1.get('name', slug1)} (P1) vs "
                        f"{entry2.get('name', slug2)} (P2) — dual-agent training"
                    ),
                    tags=[slug1, slug2, format_name],
                    p1_hero=hero_slug(str(entry1.get("hero_id", slug1))),
                    p2_hero=hero_slug(str(entry2.get("hero_id", slug2))),
                )
            )
    return matchups


def hero_slug(hero_id: str) -> str:
    return hero_id.removeprefix("hero_").replace("_", "-")


def build_fabrary_eval_env_ids(
    matchups: list[Matchup],
    decks: list[tuple[str, str, dict]],
    env_suffix: str,
) -> dict[str, str]:
    deck_entry_by_stem = {stem: entry for _, stem, entry in decks}
    eval_env_ids: dict[str, str] = {}
    for m in matchups:
        e1 = deck_entry_by_stem[m.p1_deck]
        e2 = deck_entry_by_stem[m.p2_deck]
        h1 = hero_slug(str(e1.get("hero_id", "p1")))
        h2 = hero_slug(str(e2.get("hero_id", "p2")))
        eval_env_ids[m.name] = f"FaB-{h1}-vs-{h2}-{env_suffix}-v0"
    return eval_env_ids


def init_fabrary_training(
    format_name: str,
    env_suffix: str,
) -> tuple[list[tuple[str, str, dict]], list[Matchup], dict[str, Matchup], dict[str, str]]:
    decks = materialize_fabrary_decks(format_name)
    if not decks:
        print(f"No {format_name} decks found in {FABRARY_DECKS_PATH}")
        sys.exit(1)
    matchups = build_fabrary_matchups(decks, format_name)
    matchup_index = {m.name: m for m in matchups}
    eval_env_ids = build_fabrary_eval_env_ids(matchups, decks, env_suffix)
    return decks, matchups, matchup_index, eval_env_ids
