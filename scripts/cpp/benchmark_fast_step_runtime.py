#!/usr/bin/env python3
"""Micro-benchmark C++ fast path vs Python wrapper vs PPO inference."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

from flesh_and_blood_rlbridge.cpp_engine_environment import CppEngineEnvironment
from flesh_and_blood_rlbridge.fast_observation import FAST_OBS_DIM
from rl_agents.ppo import PPOAgent
from runtime_defaults import DEFAULT_HIDDEN_SIZE
from train_dual_agent_common import _mask_logits_to_legal, _run_one_fast_episode

ENGINE_DIR = REPO_ROOT / "results" / "cpp_engines" / "Ira_vs_Ira-8386dab645e583fc"
WARMUP = 200
ITERS = 5000
EPISODE_STEPS = 500


def _bench(label: str, fn, *, iters: int = ITERS) -> float:
    for _ in range(WARMUP):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - t0
    per_us = elapsed / iters * 1e6
    print(f"  {label:42s}  {per_us:8.2f} us/step  ({iters / elapsed:,.0f} ops/s)")
    return per_us


def main() -> None:
    if not ENGINE_DIR.is_dir():
        raise SystemExit(f"Engine not found: {ENGINE_DIR}")

    env = CppEngineEnvironment(engine_dir=ENGINE_DIR, max_turns=2000)
    gs = env._fab.GameState()
    gs.register_all_cards()
    gs.init_standard_decks()

    print("=" * 72)
    print(f"  Fast-step runtime benchmark — Ira vs Ira")
    print(f"  Engine: {ENGINE_DIR.name}")
    print(f"  Warmup={WARMUP}  microbench_iters={ITERS}  obs_dim={FAST_OBS_DIM}")
    print("=" * 72)

    # ── C++ raw (no Python env wrapper) ─────────────────────────────────────
    print("\n[C++ GameState — direct pybind]")
    action_idx = 0

    def cpp_fast_step() -> None:
        nonlocal action_idx, gs
        r = gs.fast_step_index(action_idx)
        action_idx = (action_idx + 1) % max(1, int(r.legal_count))

    def cpp_legal_count() -> None:
        gs.fast_legal_count()

    def cpp_obs_only() -> None:
        gs.fast_observation_vector(gs.fast_legal_count())

    # fresh gs for isolated benches
    gs_lc = env._fab.GameState()
    gs_lc.register_all_cards()
    gs_lc.init_standard_decks()

    def cpp_legal_only() -> None:
        gs_lc.fast_legal_count()

    t_cpp_step = _bench("fast_step_index (full)", cpp_fast_step)
    t_cpp_legal = _bench("fast_legal_count only", cpp_legal_only, iters=ITERS * 2)
    t_cpp_obs = _bench("fast_observation_vector only", cpp_obs_only, iters=ITERS * 2)

    # ── Python fast wrapper ─────────────────────────────────────────────────
    print("\n[Python CppEngineEnvironment — fast path]")
    state = env.fast_reset(seed=42)
    py_action = 0

    def py_fast_step() -> None:
        nonlocal state, py_action
        state = env.fast_step_index(py_action)
        py_action = (py_action + 1) % max(1, int(state["legal_count"]))

    t_py_fast = _bench("fast_step_index wrapper", py_fast_step)

    env2 = CppEngineEnvironment(engine_dir=ENGINE_DIR, max_turns=2000)
    state2 = env2.fast_reset(seed=42)
    py_action2 = 0

    def py_fast_step_minimal() -> None:
        nonlocal state2, py_action2
        r = env2._gs.fast_step_index(py_action2)
        py_action2 = (py_action2 + 1) % max(1, int(r.legal_count))
        np.asarray(r.obs_vec, dtype=np.float64)

    t_py_raw_plus_asarray = _bench("raw C++ + np.asarray(obs)", py_fast_step_minimal)

    # ── Slow path comparison ────────────────────────────────────────────────
    print("\n[Python CppEngineEnvironment — slow step() path]")
    env_slow = CppEngineEnvironment(engine_dir=ENGINE_DIR, max_turns=2000)
    env_slow.reset(seed=42)
    slow_action = 0

    def slow_step() -> None:
        nonlocal slow_action, env_slow
        legal = env_slow._filter_legal_actions(env_slow._legal_actions())
        out = env_slow.step(str(slow_action % max(1, len(legal))))
        if out.terminated or out.truncated:
            env_slow.reset(seed=42)
            slow_action = 0
        else:
            slow_action += 1

    t_slow = _bench("step() + JSON obs", slow_step, iters=500)

    # combat tracker slow path
    env_track = CppEngineEnvironment(
        engine_dir=ENGINE_DIR, max_turns=2000, enable_combat_tracker=True
    )
    env_track.reset(seed=42)
    track_action = 0

    def tracked_step() -> None:
        nonlocal track_action, env_track
        legal = env_track._filter_legal_actions(env_track._legal_actions())
        out = env_track.step(str(track_action % max(1, len(legal))))
        if out.terminated or out.truncated:
            env_track.reset(seed=42)
            track_action = 0
        else:
            track_action += 1

    t_tracked = _bench("step() + combat tracker", tracked_step, iters=200)

    # ── PPO inference ───────────────────────────────────────────────────────
    print("\n[PPO inference — hidden_size=%d, n_actions=32]" % DEFAULT_HIDDEN_SIZE)
    policy = PPOAgent(hidden_size=DEFAULT_HIDDEN_SIZE, n_actions=32, obs_dim=FAST_OBS_DIM)
    policy._init_nets(FAST_OBS_DIM)
    obs_vec = np.asarray(state["obs_vec"], dtype=np.float64)
    n_legal = max(1, int(state["legal_count"]))

    def actor_only() -> None:
        policy._actor.predict(obs_vec[None, :])

    def critic_only() -> None:
        policy._critic.predict(obs_vec[None, :])

    def actor_critic_separate() -> None:
        logits = policy._actor.predict(obs_vec[None, :])
        policy._critic.predict(obs_vec[None, :])
        _mask_logits_to_legal(logits, n_legal)

    def actor_critic_fused() -> None:
        logits, _value = policy.predict_policy_value(obs_vec)
        _mask_logits_to_legal(logits, n_legal)

    t_actor = _bench("actor.predict", actor_only, iters=ITERS * 2)
    t_critic = _bench("critic.predict", critic_only, iters=ITERS * 2)
    t_both = _bench("actor + critic + mask (separate)", actor_critic_separate, iters=ITERS * 2)
    t_fused = _bench("predict_policy_value + mask (fused)", actor_critic_fused, iters=ITERS * 2)

    # transition dict (training storage)
    next_obs = obs_vec.copy()

    def store_transition() -> None:
        _ = {
            "obs_vec": obs_vec,
            "action": 0,
            "reward": 0.0,
            "value": 0.0,
            "log_prob": 0.0,
            "done": 0.0,
            "n_legal": n_legal,
            "next_obs_vec": next_obs,
        }

    t_store = _bench("transition dict (obs+next_obs)", store_transition, iters=ITERS * 2)

    # ── Full simulated training step ────────────────────────────────────────
    print("\n[Simulated training step — env + policy]")

    def full_training_step() -> None:
        nonlocal state, py_action, obs_vec, n_legal, next_obs
        acting_legal = max(1, int(state["legal_count"]))
        obs_vec = np.asarray(state["obs_vec"], dtype=np.float64)
        logits, _value = policy.predict_policy_value(obs_vec)
        logits = _mask_logits_to_legal(logits, acting_legal)
        state = env.fast_step_index(py_action % acting_legal)
        next_obs = np.asarray(state["obs_vec"], dtype=np.float64)
        py_action += 1
        if state["terminated"] or state["truncated"]:
            state = env.fast_reset(seed=42)
            py_action = 0

    state = env.fast_reset(seed=42)
    py_action = 0
    t_full = _bench("fast env + actor + critic + copies", full_training_step, iters=2000)

    def full_warmup_step() -> None:
        nonlocal state, py_action
        acting_legal = max(1, int(state["legal_count"]))
        state = env.fast_step_index(py_action % acting_legal)
        py_action += 1
        if state["terminated"] or state["truncated"]:
            state = env.fast_reset(seed=42)
            py_action = 0

    state = env.fast_reset(seed=42)
    py_action = 0
    t_warmup = _bench("fast env only (warmup-style)", full_warmup_step, iters=ITERS)

    # ── Episode throughput ────────────────────────────────────────────────────
    print("\n[End-to-end episode throughput]")
    policy_p1 = PPOAgent(hidden_size=DEFAULT_HIDDEN_SIZE, n_actions=32, obs_dim=FAST_OBS_DIM)
    policy_p1._init_nets(FAST_OBS_DIM)
    policy_p2 = PPOAgent(hidden_size=DEFAULT_HIDDEN_SIZE, n_actions=32, obs_dim=FAST_OBS_DIM)
    policy_p2._init_nets(FAST_OBS_DIM)
    from flesh_and_blood_rlbridge.talishar_engine_environment import TalisharEngineEnvironment

    tal_env = TalisharEngineEnvironment(
        local_deck_name="Ira",
        opponent_deck_name="Ira",
        use_cpp_engine=True,
        cpp_engine_dir=str(ENGINE_DIR),
        max_turns=EPISODE_STEPS,
    )
    rng = np.random.default_rng(0)

    t0 = time.perf_counter()
    ep_count = 20
    total_steps = 0
    for i in range(ep_count):
        result = _run_one_fast_episode(
            tal_env, policy_p1, policy_p2, EPISODE_STEPS, seed=i,
            warmup=False, p1_rng=rng, p2_rng=rng,
        )
        total_steps += int(result["steps"])
    ep_elapsed = time.perf_counter() - t0
    print(f"  {ep_count} episodes × up to {EPISODE_STEPS} steps: {total_steps} total steps")
    print(f"  {total_steps / ep_elapsed:,.0f} steps/s  |  {ep_count / ep_elapsed:.1f} episodes/s")

    # ── Summary ranking ───────────────────────────────────────────────────────
    overhead_py = t_py_fast - t_cpp_step
    overhead_asarray = t_py_raw_plus_asarray - t_cpp_step
    overhead_wrapper_rest = t_py_fast - t_py_raw_plus_asarray

    rows = [
        ("C++ fast_step_index", t_cpp_step, "baseline engine step"),
        ("+ fast_legal_count (duplicate scan est.)", t_cpp_legal, "~extra if not merged"),
        ("+ fast_observation_vector alone", t_cpp_obs, "subset of fast_step"),
        ("Python wrapper overhead (total)", overhead_py, "pybind dict + properties"),
        ("  - np.asarray(obs_vec) copy", overhead_asarray, "zero-copy target"),
        ("  - dict + deck/turn fields", overhead_wrapper_rest, "slim return dict"),
        ("Slow step() JSON path", t_slow, "avoid in training"),
        ("step() + combat tracker", t_tracked, "disables fast path"),
        ("PPO actor", t_actor, "fused forward candidate"),
        ("PPO critic", t_critic, "fused forward candidate"),
        ("PPO actor+critic+mask (separate)", t_both, "pre-fusion baseline"),
        ("PPO predict_policy_value+mask (fused)", t_fused, "shared trunk rollout"),
        ("Transition storage", t_store, "compact buffer candidate"),
        ("Full training step (simulated)", t_full, "current hot path"),
        ("Warmup step (env only)", t_warmup, "no NN"),
    ]
    rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)

    print("\n" + "=" * 72)
    print("  RANKED BY TIME PER STEP (higher us = bigger optimization target)")
    print("=" * 72)
    print(f"  {'Component':<42s}  {'us/step':>8s}  Notes")
    print("  " + "-" * 68)
    for name, us, note in rows_sorted:
        pct = 100.0 * us / t_full if t_full > 0 else 0.0
        print(f"  {name:<42s}  {us:8.2f}  {note} ({pct:.0f}% of full step)")

    savings = {
        "fuse_actor_critic": t_both - t_fused,
        "zero_copy_obs": overhead_asarray,
        "slim_fast_return": max(0.0, overhead_wrapper_rest),
        "cpp_legal_cache": t_cpp_legal * 0.5,
        "avoid_slow_path": t_slow - t_py_fast,
        "disable_combat_tracker": t_tracked - t_slow,
    }
    print("\n  Estimated savings if implemented (us/step, rough):")
    for k, v in sorted(savings.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {k:<28s}  ~{v:6.1f} us")


if __name__ == "__main__":
    main()
