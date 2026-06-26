"""Shared dual-agent PPO self-play training utilities for Talishar scripts."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.parse
import uuid
import base64
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from fab_bridge.paths import configure_import_paths, prepend_sys_path, repo_root, src_dir  # noqa: E402

configure_import_paths()
REPO_ROOT = repo_root()
FAB_SRC = src_dir()
RL_SRC = Path("~/Documents/RL/rlbridge/src").expanduser()
prepend_sys_path(RL_SRC)
FAB_DB_DIR = FAB_SRC / "flesh_and_blood_rlbridge" / "card_db"
FABRARY_DECKS_PATH = FAB_DB_DIR / "fabrary_decks.json"
CARDS_DB_PATH = FAB_DB_DIR / "cards.json"
TALISHAR_ID_RE = re.compile(r"^[a-z][a-z0-9_]+$")

from fab_tui.deck_cards import assign_pitch_variants  # noqa: E402
from play_outcome_stats import (  # noqa: E402
    absolute_p1_p2_deck_from_env,
    absolute_p1_p2_hp_from_env,
    classify_p1_episode_outcome,
    summarize_p1_outcomes,
)
import numpy as np  # noqa: E402
try:
    import torch
    import torch.nn.functional as _F
    _TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    _TORCH_DEVICE = None  # type: ignore[assignment]

from flesh_and_blood_rlbridge.talishar_engine_environment import (  # noqa: E402
    TalisharEngineEnvironment,
    run_talishar_eval_episode,
)
from flesh_and_blood_rlbridge.player_observation import (  # noqa: E402
    ACTION_CAPACITY,
    PLAYER_OBS_SCHEMA_VERSION,
)
from rl_agents.ppo import (  # noqa: E402
    PPOAgent,
    _gae,
    _get,
    _infer_action_capacity,
    _log_softmax,
    _n_legal_of,
    _softmax,
    _to_env_action,
)
from episode_cache import (  # noqa: E402
    EpisodeCache,
    DEFAULT_WARMUP_SKIP_THRESHOLD,
)
from runtime_defaults import (  # noqa: E402
    DEFAULT_CLIP_EPS,
    DEFAULT_GAMMA,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LAM,
    DEFAULT_LR,
    DEFAULT_MINI_BATCH,
    DEFAULT_N_EPISODES,
    DEFAULT_N_HEADS,
    DEFAULT_N_LAYERS,
    DEFAULT_N_STEPS,
    DEFAULT_PARALLEL_SEEDS,
    DEFAULT_PPO_EPOCHS,
    DEFAULT_PPO_ROLLOUT_BATCH,
    DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
    DEFAULT_WARMUP_EPISODES,
    DEFAULT_CHECKPOINT_INTERVAL_PCT,
    DEFAULT_CHECKPOINT_EVAL_EPISODES,
)
from parallel_seed_training import (  # noqa: E402
    run_parallel_seed_jobs,
    select_best_agents_by_win_rate,
    workers_per_parallel_seed,
)

FORMAT_DECK_RULES: dict[str, dict[str, int]] = {
    "silver_age": {"max_copies": 2, "deck_size": 40},
    "classic_constructed": {"max_copies": 3, "deck_size": 60},
}

FABRARY_ENV_SUFFIX: dict[str, str] = {
    "silver_age": "SA",
    "classic_constructed": "CC",
    "blitz": "BL",
    "upf": "UPF",
}


def resolve_checkpoint_interval(
    n_episodes: int,
    *,
    checkpoint_interval: Optional[int] = None,
    checkpoint_interval_pct: float = DEFAULT_CHECKPOINT_INTERVAL_PCT,
) -> int:
    """Resolve checkpoint cadence — fixed interval or a % of total episodes."""
    if checkpoint_interval is not None and checkpoint_interval > 0:
        return int(checkpoint_interval)
    pct = max(0.1, float(checkpoint_interval_pct))
    return max(1, int(math.ceil(n_episodes * pct / 100.0)))

_TORCH_COMPUTE_DTYPE = torch.float32 if _TORCH_AVAILABLE else None


@dataclass
class Matchup:
    name: str
    p1_deck: str
    p2_deck: str
    description: str
    tags: list[str] = field(default_factory=list)
    p1_hero: str = ""
    p2_hero: str = ""
    # Optional override deck names for C++ engine cache lookup.
    # Set these to the original hero/asset IDs when p1_deck/p2_deck are
    # UUID-based (Phase 3) but the compiled engine was built for the hero IDs.
    cpp_engine_deck1: Optional[str] = None
    cpp_engine_deck2: Optional[str] = None
    # Explicit engine directory — bypasses key/cache lookup entirely.
    cpp_engine_dir: Optional[str] = None


def make_env(
    matchup: Matchup,
    base_url: str,
    game_format: str,
    max_turns: int,
    *,
    show_frontend: bool = False,
    frontend_url: Optional[str] = None,
    request_timeout: float = 30.0,
    use_cpp_engine: bool = True,
    cpp_engine_cache_dir: Optional[str] = None,
    enable_combat_tracker: bool = False,
    require_fast_training: Optional[bool] = None,
) -> TalisharEngineEnvironment:
    """Create a :class:`TalisharEngineEnvironment` for *matchup*.

    When ``use_cpp_engine=True`` (default) and a compiled C++ engine exists in
    the cache for this matchup, the environment will use it instead of HTTP
    Talishar — roughly 100× faster per step.  Falls back to HTTP silently if
    no compiled module is found.

    Generate a C++ engine for a matchup with::

        python scripts/generate_cpp_engine.py \\
            --talishar-src Talishar \\
            --deck1 {p1_deck} --deck2 {p2_deck}

    Then implement the stubs in ``results/cpp_engines/<matchup>/cards.h`` and
    build with ``cmake``.

    When ``require_fast_training`` is omitted, it defaults to ``use_cpp_engine``.
    If a C++ engine is active but lacks the fast training API, a
    :class:`RuntimeError` is raised instead of silently using the slow
    ``step()`` + JSON path.
    """
    resolved_frontend_url = frontend_url
    if show_frontend and not resolved_frontend_url:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme and parsed.netloc:
            resolved_frontend_url = os.environ.get("TALISHAR_FE_URL")
            if not resolved_frontend_url:
                # Default FE dev URL for local training visualisation.
                resolved_frontend_url = "http://localhost:5173"

    # Resolve the default cache dir relative to the repo root
    effective_cache_dir = cpp_engine_cache_dir
    if effective_cache_dir is None:
        effective_cache_dir = str(REPO_ROOT / "results" / "cpp_engines")

    env = TalisharEngineEnvironment(
        base_url=base_url,
        frontend_url=resolved_frontend_url,
        local_deck_name=matchup.p1_deck,
        opponent_deck_name=matchup.p2_deck,
        game_format=game_format,
        self_play=True,
        max_turns=max_turns,
        render_mode=("rgb_array" if show_frontend else None),
        request_timeout=request_timeout,
        use_cpp_engine=use_cpp_engine,
        cpp_engine_cache_dir=effective_cache_dir,
        cpp_engine_deck1=matchup.cpp_engine_deck1,
        cpp_engine_deck2=matchup.cpp_engine_deck2,
        cpp_engine_dir=matchup.cpp_engine_dir,
        enable_combat_tracker=enable_combat_tracker,
    )
    if require_fast_training is None:
        require_fast_training = use_cpp_engine or bool(matchup.cpp_engine_dir)
    if (
        require_fast_training
        and getattr(env, "_using_cpp", False)
        and not _env_supports_fast_training(env)
    ):
        reasons = "; ".join(_fast_training_unavailable_reasons(env)) or "unknown"
        raise RuntimeError(
            "C++ engine is loaded but fast training is unavailable: "
            f"{reasons}"
        )
    return env


def make_agent(seed: Optional[int] = None) -> PPOAgent:
    return PPOAgent(
        hidden_size=DEFAULT_HIDDEN_SIZE,
        n_layers=DEFAULT_N_LAYERS,
        n_heads=DEFAULT_N_HEADS,
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


def _probe_training_dims(
    env: Any,
    seed: Optional[int] = None,
) -> tuple[int, int, bool]:
    """Return (obs_dim, n_actions, mask_actions) from a live environment."""
    if _env_supports_fast_training(env):
        reset_info = env.fast_reset(seed=seed)
        obs_vec = np.asarray(reset_info["obs_vec"], dtype=np.float64)
        return int(obs_vec.shape[0]), int(env.fast_action_capacity()), True
    reset_out = env.reset(seed=seed)
    obs = _get(reset_out, "observation", reset_out)
    n_actions, mask_actions = _infer_action_capacity(env, seed=seed)
    probe = make_agent(seed=seed)
    obs_dim = int(probe._obs_to_vec(obs).shape[0])
    return obs_dim, int(n_actions), bool(mask_actions)


def _bootstrap_unified_policy(
    cache_store: "AgentCacheStore",
    env: Any,
    seed: Optional[int],
) -> "UnifiedPolicyBundle":
    from agent_cache import UnifiedPolicyBundle  # noqa: PLC0415

    obs_dim, n_actions, mask_actions = _probe_training_dims(env, seed=seed)

    def _make() -> PPOAgent:
        return make_agent(seed=seed)

    policy, init_src = cache_store.load_or_create(
        _make,
        obs_dim=obs_dim,
        n_actions=n_actions,
        mask_actions=mask_actions,
    )
    return UnifiedPolicyBundle(policy=policy, init_sources=[init_src])


def _ppo_update(agent: PPOAgent, buf: dict, next_obs_vec: np.ndarray) -> None:
    T = len(buf["obs"])
    if T == 0:
        return

    if _TORCH_AVAILABLE and hasattr(agent._actor, "_net"):
        # ── GPU-native PPO update (all math stays on _TORCH_DEVICE) ──────────
        dev = _TORCH_DEVICE

        def _t(x, dtype=_TORCH_COMPUTE_DTYPE):
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=dev)

        obs_t     = _t(buf["obs"])                            # (T, D)
        act_t     = _t(buf["actions"], torch.int64)           # (T,)
        rew_t     = _t(buf["rewards"])                        # (T,)
        done_t    = _t(buf["dones"])                          # (T,)
        lp_old_t  = _t(buf["log_probs"])                      # (T,)
        nlegal_t  = _t(buf["n_legal"], torch.int64)           # (T,)

        # Bootstrap value from current critic (no grad needed)
        with torch.no_grad():
            vals_t      = agent._critic._net(obs_t).squeeze(-1)   # (T,)
            next_val_t  = agent._critic._net(
                _t(next_obs_vec[None, :])
            ).squeeze(-1).squeeze(0)                              # scalar
            next_vals_t = torch.cat([vals_t[1:], next_val_t.unsqueeze(0)])  # (T,)

        # GAE on GPU
        gam, lam = agent.gamma, agent.lam
        adv_t = torch.zeros(T, dtype=_TORCH_COMPUTE_DTYPE, device=dev)
        last_gae = torch.tensor(0.0, dtype=_TORCH_COMPUTE_DTYPE, device=dev)
        for i in range(T - 1, -1, -1):
            delta     = rew_t[i] + gam * next_vals_t[i] * (1.0 - done_t[i]) - vals_t[i]
            last_gae  = delta + gam * lam * (1.0 - done_t[i]) * last_gae
            adv_t[i]  = last_gae
        ret_t = adv_t + vals_t
        if T > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        indices = np.arange(T)
        for _ in range(agent.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, agent.mini_batch_size):
                mb_idx = indices[start : start + agent.mini_batch_size]
                if len(mb_idx) == 0:
                    continue
                mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.int64, device=dev)
                mb_obs    = obs_t[mb_idx_t]
                mb_acts   = act_t[mb_idx_t]
                mb_adv    = adv_t[mb_idx_t]
                mb_ret    = ret_t[mb_idx_t]
                mb_lp_old = lp_old_t[mb_idx_t]
                mb_nlegal = nlegal_t[mb_idx_t]
                B = mb_obs.shape[0]

                # ── Actor ──────────────────────────────────────────────────────
                agent._actor._opt.zero_grad()
                logits = agent._actor._net(mb_obs)              # (B, A)
                if agent._mask_actions:
                    A = logits.shape[1]
                    mask = (
                        torch.arange(A, device=dev).unsqueeze(0)
                        < mb_nlegal.unsqueeze(1)
                    )
                    logits = logits.masked_fill(~mask, -1e9)
                lp_new = _F.log_softmax(logits, dim=-1)
                probs  = _F.softmax(logits, dim=-1)
                lp_a   = lp_new[torch.arange(B, device=dev), mb_acts]  # (B,)
                ratio  = torch.exp(lp_a - mb_lp_old)
                clip_r = torch.clamp(ratio, 1.0 - agent.clip_eps, 1.0 + agent.clip_eps)
                actor_loss = -torch.minimum(ratio * mb_adv, clip_r * mb_adv).mean()
                entropy = -(probs * lp_new).sum(dim=-1).mean()
                loss_a = actor_loss - agent.c_ent * entropy
                loss_a.backward()
                agent._actor._opt.step()

                # ── Critic ────────────────────────────────────────────────────
                agent._critic._opt.zero_grad()
                val_pred = agent._critic._net(mb_obs).squeeze(-1)  # (B,)
                loss_c = agent.c_vf * _F.mse_loss(val_pred, mb_ret)
                loss_c.backward()
                agent._critic._opt.step()
        return

    # ── Fallback: numpy PPO (no torch available) ──────────────────────────────
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


def _bc_update(agent: PPOAgent, buf: dict, next_obs_vec: np.ndarray) -> None:
    """Behavioural-cloning warm-start update.

    Trains the actor via cross-entropy loss on cached (obs, action) pairs so
    that the policy imitates the demonstrated actions directly — bypassing the
    stale log-prob problem that makes PPO ratio ≈ 1 when replaying old episodes.

    Also updates the critic via MSE against GAE returns recomputed from the
    *current* critic (fresh bootstraps), so the value function is warmed up
    without relying on stale value estimates stored in the cache.

    Uses PyTorch autograd on _TORCH_DEVICE (GPU when available) to keep all
    tensors on device throughout.  Falls back to numpy when torch is absent.
    """
    T = len(buf["obs"])
    if T == 0:
        return

    if _TORCH_AVAILABLE and hasattr(agent._actor, "_net"):
        # ── GPU-native BC update ──────────────────────────────────────────────
        dev = _TORCH_DEVICE

        def _t(x, dtype=_TORCH_COMPUTE_DTYPE):
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=dev)

        obs_t    = _t(buf["obs"])                        # (T, D)
        act_t    = _t(buf["actions"], torch.int64)       # (T,)
        rew_t    = _t(buf["rewards"])                    # (T,)
        done_t   = _t(buf["dones"])                      # (T,)
        nlegal_t = _t(buf["n_legal"], torch.int64)       # (T,)

        # Fresh value estimates from current critic (no grad)
        with torch.no_grad():
            vals_t     = agent._critic._net(obs_t).squeeze(-1)       # (T,)
            next_val_t = agent._critic._net(
                _t(next_obs_vec[None, :])
            ).squeeze(-1).squeeze(0)                                  # scalar
            next_vals_t = torch.cat([vals_t[1:], next_val_t.unsqueeze(0)])  # (T,)

        # GAE returns on GPU (only returns needed for BC critic update)
        gam, lam = agent.gamma, agent.lam
        adv_t = torch.zeros(T, dtype=_TORCH_COMPUTE_DTYPE, device=dev)
        last_gae = torch.tensor(0.0, dtype=_TORCH_COMPUTE_DTYPE, device=dev)
        for i in range(T - 1, -1, -1):
            delta    = rew_t[i] + gam * next_vals_t[i] * (1.0 - done_t[i]) - vals_t[i]
            last_gae = delta + gam * lam * (1.0 - done_t[i]) * last_gae
            adv_t[i] = last_gae
        ret_t = adv_t + vals_t

        indices = np.arange(T)
        for _ in range(agent.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, agent.mini_batch_size):
                mb_idx = indices[start : start + agent.mini_batch_size]
                if len(mb_idx) == 0:
                    continue
                mb_idx_t  = torch.as_tensor(mb_idx, dtype=torch.int64, device=dev)
                mb_obs    = obs_t[mb_idx_t]
                mb_acts   = act_t[mb_idx_t]
                mb_nlegal = nlegal_t[mb_idx_t]
                mb_ret    = ret_t[mb_idx_t]
                B = mb_obs.shape[0]

                # ── Actor: cross-entropy (behavioural cloning) ────────────────
                agent._actor._opt.zero_grad()
                logits = agent._actor._net(mb_obs)          # (B, A)
                if agent._mask_actions:
                    A = logits.shape[1]
                    mask = (
                        torch.arange(A, device=dev).unsqueeze(0)
                        < mb_nlegal.unsqueeze(1)
                    )
                    logits = logits.masked_fill(~mask, -1e9)
                loss_ce = _F.cross_entropy(
                    logits.float(), mb_acts,
                    reduction="mean",
                )
                loss_ce.backward()
                agent._actor._opt.step()

                # ── Critic: MSE against GAE returns ───────────────────────────
                agent._critic._opt.zero_grad()
                val_pred = agent._critic._net(mb_obs).squeeze(-1)  # (B,)
                loss_c = agent.c_vf * _F.mse_loss(val_pred, mb_ret)
                loss_c.backward()
                agent._critic._opt.step()
        return

    # ── Fallback: numpy BC (no torch available) ───────────────────────────────
    obs_arr    = np.array(buf["obs"],     dtype=np.float64)
    act_arr    = np.array(buf["actions"], dtype=np.int64)
    dones_arr  = np.array(buf["dones"],   dtype=np.float64)
    nlegal_arr = np.array(buf["n_legal"], dtype=np.int64)

    # Recompute value estimates from the *current* critic (not stale cache values)
    values_arr = agent._critic.predict(obs_arr).flatten()
    next_val   = float(agent._critic.predict(next_obs_vec[None, :]).flatten()[0])
    next_vals  = np.append(values_arr[1:], next_val)

    _, returns = _gae(
        np.array(buf["rewards"], dtype=np.float64),
        values_arr,
        next_vals,
        dones_arr,
        agent.gamma,
        agent.lam,
    )

    indices = np.arange(T)
    for _ in range(agent.ppo_epochs):
        agent._rng_np.shuffle(indices)
        for start in range(0, T, agent.mini_batch_size):
            mb_idx = indices[start : start + agent.mini_batch_size]
            if len(mb_idx) == 0:
                continue
            mb_obs    = obs_arr[mb_idx]
            mb_acts   = act_arr[mb_idx]
            mb_nlegal = nlegal_arr[mb_idx]
            mb_ret    = returns[mb_idx]
            B = mb_obs.shape[0]

            # ── Actor: cross-entropy (behavioural cloning) ────────────────────
            logits = agent._actor.forward(mb_obs)
            legal_mask = None
            if agent._mask_actions:
                legal_mask = (
                    np.arange(agent.n_actions)[None, :] < mb_nlegal[:, None]
                )
                logits = np.where(legal_mask, logits, -1e9)
            probs = _softmax(logits)
            # CE gradient w.r.t. logits: softmax(logits) - one_hot(action)
            grad_logits = probs.copy()
            grad_logits[np.arange(B), mb_acts] -= 1.0
            grad_logits /= B
            if legal_mask is not None:
                grad_logits = np.where(legal_mask, grad_logits, 0.0)
            agent._actor.backward(grad_logits)

            # ── Critic: MSE against GAE returns ───────────────────────────────
            val_pred = agent._critic.forward(mb_obs).flatten()
            grad_val = 2.0 * (val_pred - mb_ret) / B
            agent._critic.backward(agent.c_vf * grad_val[:, None])


def _bc_update_all_tiers(
    tier_agents: list[PPOAgent],
    buf: dict,
    next_obs_vec: np.ndarray,
) -> None:
    for agent in tier_agents:
        _bc_update(agent, buf, next_obs_vec)


def _env_action_to_index(obs: Any, env_action: str, policy: PPOAgent) -> int:
    """Convert a Talishar mode-code string to a compact policy action index.

    The PPO actor uses dense indices 0..n_actions-1 where index i maps to the
    i-th entry in the legal-actions list the environment exposed at that step.
    Raw mode codes (e.g. "99", "27", "10000") are NOT valid indices — storing
    them directly corrupts the BC buffer with wrong obs->action mappings.

    Strategy:
      1. Parse legalActions from the observation.
      2. Find the position of env_action's mode code in that ordered list.
      3. That position is the correct policy index.
      4. Fall back to 0 if not found (e.g. sample_action chose a filtered-out
         action) — index 0 is safe; it just weakly reinforces the first legal
         action rather than injecting a garbage index.
    """
    try:
        mode_code = int(env_action)
    except (TypeError, ValueError):
        return 0
    try:
        raw = json.loads(obs) if isinstance(obs, str) else obs
        if not isinstance(raw, dict):
            return 0
        legal: list[int] = []
        for entry in raw.get("legalActions", []) or []:
            try:
                code = int(
                    entry.get("actionCode", entry)
                    if isinstance(entry, dict) else entry
                )
                legal.append(code)
            except (TypeError, ValueError):
                pass
        if mode_code in legal:
            return min(legal.index(mode_code), policy.n_actions - 1)
    except Exception:
        pass
    return 0


def _obs_to_text(obs: Any) -> str:
    if isinstance(obs, str):
        try:
            parsed = json.loads(obs)
        except json.JSONDecodeError:
            parsed = {"raw": obs}
    elif isinstance(obs, dict):
        parsed = obs
    else:
        parsed = {"raw": repr(obs)}

    lines: list[str] = ["Talishar Training State"]
    lines.append(f"turnNo: {parsed.get('turnNo', '?')}")
    lines.append(f"turnPhase: {parsed.get('turnPhase', '?')}")
    lines.append(f"actingPlayerID: {parsed.get('actingPlayerID', '?')}")
    lines.append(f"playerHealth: {parsed.get('playerHealth', '?')}")
    lines.append(f"opponentHealth: {parsed.get('opponentHealth', '?')}")
    lines.append(f"legalActions: {len(parsed.get('legalActions', []) or [])}")
    lines.append("")
    lines.append("Observation JSON:")
    lines.append(json.dumps(parsed, indent=2, ensure_ascii=False))
    return "\n".join(lines)


def _write_state_image(obs: Any, out_path: Path, header: str = "") -> None:
    text = _obs_to_text(obs)
    if header:
        text = f"{header}\n\n{text}"
    try:
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.load_default()
        lines = text.splitlines()
        line_height = 14
        width = 1800
        height = max(300, 20 + line_height * (len(lines) + 2))
        img = Image.new("RGB", (width, height), color=(18, 18, 18))
        draw = ImageDraw.Draw(img)
        y = 10
        for line in lines:
            draw.text((10, y), line, fill=(235, 235, 235), font=font)
            y += line_height
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
    except Exception:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.with_suffix(".txt").write_text(text, encoding="utf-8")


def _resolve_obs_vec(obs: Any, info: Any, policy: PPOAgent) -> np.ndarray:
    """Use env-precomputed observation_vec when available (C++ fast path)."""
    if isinstance(info, dict):
        cached = info.get("observation_vec")
        if cached is not None:
            vec = np.asarray(cached, dtype=np.float64)
            if policy.obs_dim > 0 and vec.shape[0] == policy.obs_dim:
                return vec
    return policy._obs_to_vec(obs)


def _policy_forward(
    policy: PPOAgent,
    obs: Any,
    obs_vec: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Actor logits, critic value, and softmax probs for one observation."""
    logits, value = policy.predict_policy_value(obs_vec)
    logits = policy._masked_logits(logits, obs)
    lp_all = _log_softmax(logits)[0]
    probs = _softmax(logits)[0]
    return logits, value, probs


def _mask_logits_to_legal(logits: np.ndarray, n_legal: int) -> np.ndarray:
    masked = np.asarray(logits, dtype=np.float64).copy()
    if 0 < n_legal < masked.shape[-1]:
        masked[..., n_legal:] = -1e9
    return masked


def _mask_logits_batch(logits: np.ndarray, n_legal: np.ndarray) -> np.ndarray:
    """Mask illegal action columns for each row of batched logits."""
    masked = np.asarray(logits, dtype=np.float64).copy()
    if masked.ndim == 1:
        masked = masked[None, :]
    n_actions = masked.shape[-1]
    n_legal_arr = np.asarray(n_legal, dtype=np.int64).reshape(-1)
    cols = np.arange(n_actions, dtype=np.int64)
    illegal = (n_legal_arr[:, None] > 0) & (n_legal_arr[:, None] < n_actions) & (
        cols[None, :] >= n_legal_arr[:, None]
    )
    masked[illegal] = -1e9
    return masked


def _sample_actions_from_logits_batch(
    logits: np.ndarray,
    n_legal: np.ndarray,
    rngs: list[np.random.Generator],
    n_actions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sampled action indices and log-probs for a batch of masked logits."""
    masked = _mask_logits_batch(logits, n_legal)
    lp_all = _log_softmax(masked)
    probs = _softmax(masked)
    actions = np.empty(masked.shape[0], dtype=np.int64)
    log_probs = np.empty(masked.shape[0], dtype=np.float64)
    for row, rng in enumerate(rngs):
        nl = max(1, int(n_legal[row]))
        action = int(rng.choice(n_actions, p=probs[row]))
        if action >= nl:
            action = nl - 1
        actions[row] = action
        log_probs[row] = float(lp_all[row, action])
    return actions, log_probs


def _int_fast_state(state_dict: dict[str, Any], key: str, default: int) -> int:
    value = state_dict.get(key, default)
    return default if value is None else int(value)


def _finalize_fast_episode_transitions(
    slot: "_FastRolloutSlot",
    *,
    max_steps: int,
) -> None:
    """Apply terminal-reward transitions and truncation flags after an episode ends."""
    if not slot.truncated and slot.steps >= max_steps:
        slot.truncated = True
    if not slot.terminated:
        return
    if slot.final_p1_hp > slot.final_p2_hp and slot.p2_trans:
        last = slot.p2_trans[-1]
        slot.p2_trans.append({
            "obs_vec": last["next_obs_vec"],
            "action": last["action"],
            "reward": -1.0,
            "value": 0.0,
            "log_prob": last["log_prob"],
            "done": 1.0,
            "n_legal": last["n_legal"],
            "next_obs_vec": np.zeros_like(last["next_obs_vec"]),
        })
        slot.cur_p2_r -= 1.0
    elif slot.final_p2_hp > slot.final_p1_hp and slot.p1_trans:
        last = slot.p1_trans[-1]
        slot.p1_trans.append({
            "obs_vec": last["next_obs_vec"],
            "action": last["action"],
            "reward": -1.0,
            "value": 0.0,
            "log_prob": last["log_prob"],
            "done": 1.0,
            "n_legal": last["n_legal"],
            "next_obs_vec": np.zeros_like(last["next_obs_vec"]),
        })
        slot.cur_p1_r -= 1.0


def _fast_episode_result_from_slot(
    slot: "_FastRolloutSlot",
    *,
    warmup: bool,
    max_steps: int,
) -> dict[str, Any]:
    _finalize_fast_episode_transitions(slot, max_steps=max_steps)
    return {
        "p1_transitions": slot.p1_trans,
        "p2_transitions": slot.p2_trans,
        "p1_reward": slot.cur_p1_r,
        "p2_reward": slot.cur_p2_r,
        "terminated": slot.terminated,
        "truncated": slot.truncated,
        "warmup": warmup,
        "steps": slot.steps,
        "p1_hp": slot.final_p1_hp,
        "p2_hp": slot.final_p2_hp,
        "turn_no": slot.final_turn_no,
        "p1_deck": slot.final_p1_deck,
        "p2_deck": slot.final_p2_deck,
    }


@dataclass
class _FastRolloutSlot:
    env: TalisharEngineEnvironment
    state: dict[str, Any]
    p1_rng: np.random.Generator
    p2_rng: np.random.Generator
    p1_trans: list[dict[str, Any]] = field(default_factory=list)
    p2_trans: list[dict[str, Any]] = field(default_factory=list)
    cur_p1_r: float = 0.0
    cur_p2_r: float = 0.0
    steps: int = 0
    active: bool = True
    terminated: bool = False
    truncated: bool = False
    final_p1_hp: int = 0
    final_p2_hp: int = 0
    final_p1_deck: int = 0
    final_p2_deck: int = 0
    final_turn_no: int = 0


def _reset_fast_rollout_slot(
    slot: _FastRolloutSlot,
    *,
    seed: Optional[int],
    starting_player_id: int,
) -> None:
    slot.state = slot.env.fast_reset(seed=seed, starting_player_id=starting_player_id)
    slot.p1_trans = []
    slot.p2_trans = []
    slot.cur_p1_r = 0.0
    slot.cur_p2_r = 0.0
    slot.steps = 0
    slot.active = True
    slot.terminated = False
    slot.truncated = False
    slot.final_p1_hp = _int_fast_state(slot.state, "p1_health", 0)
    slot.final_p2_hp = _int_fast_state(slot.state, "p2_health", 0)
    slot.final_p1_deck = _int_fast_state(slot.state, "p1_deck", 0)
    slot.final_p2_deck = _int_fast_state(slot.state, "p2_deck", 0)
    slot.final_turn_no = _int_fast_state(slot.state, "turn_no", 0)


def _batched_fast_rollout_step(
    slots: list[_FastRolloutSlot],
    p1_policy: PPOAgent,
    p2_policy: PPOAgent,
    *,
    warmup: bool,
    max_steps: int,
) -> None:
    """Advance all active fast-path rollout slots by one env step with batched policy inference."""
    active_indices = [index for index, slot in enumerate(slots) if slot.active]
    if not active_indices:
        return

    if warmup:
        for index in active_indices:
            slot = slots[index]
            state = slot.state
            n_legal = max(
                1,
                int(state.get("legal_count", p1_policy.n_actions) or 1),
            )
            acting = int(state["acting_player_id"])
            rng = slot.p1_rng if acting == 1 else slot.p2_rng
            action = int(rng.integers(n_legal))
            value = 0.0
            log_prob = 0.0
            _apply_fast_rollout_action(
                slot,
                action=action,
                value=value,
                log_prob=log_prob,
                acting=acting,
                max_steps=max_steps,
            )
        return

    p1_indices: list[int] = []
    p2_indices: list[int] = []
    for index in active_indices:
        acting = int(slots[index].state["acting_player_id"])
        if acting == 1:
            p1_indices.append(index)
        else:
            p2_indices.append(index)

    p1_logits = p1_values = p2_logits = p2_values = None
    if p1_indices:
        p1_obs = np.stack([
            np.asarray(slots[index].state["obs_vec"], dtype=np.float64)
            for index in p1_indices
        ])
        p1_logits, p1_values = p1_policy.predict_batch(p1_obs)
    if p2_indices:
        p2_obs = np.stack([
            np.asarray(slots[index].state["obs_vec"], dtype=np.float64)
            for index in p2_indices
        ])
        p2_logits, p2_values = p2_policy.predict_batch(p2_obs)

    if p1_indices:
        p1_nlegal = np.array([
            max(1, int(slots[index].state.get("legal_count", p1_policy.n_actions) or 1))
            for index in p1_indices
        ], dtype=np.int64)
        p1_actions, p1_log_probs = _sample_actions_from_logits_batch(
            np.asarray(p1_logits, dtype=np.float64),
            p1_nlegal,
            [slots[index].p1_rng for index in p1_indices],
            p1_policy.n_actions,
        )
    if p2_indices:
        p2_nlegal = np.array([
            max(1, int(slots[index].state.get("legal_count", p2_policy.n_actions) or 1))
            for index in p2_indices
        ], dtype=np.int64)
        p2_actions, p2_log_probs = _sample_actions_from_logits_batch(
            np.asarray(p2_logits, dtype=np.float64),
            p2_nlegal,
            [slots[index].p2_rng for index in p2_indices],
            p2_policy.n_actions,
        )

    p1_cursor = 0
    p2_cursor = 0
    for index in active_indices:
        slot = slots[index]
        acting = int(slot.state["acting_player_id"])
        if acting == 1:
            action = int(p1_actions[p1_cursor])
            log_prob = float(p1_log_probs[p1_cursor])
            value = float(p1_values[p1_cursor])
            p1_cursor += 1
        else:
            action = int(p2_actions[p2_cursor])
            log_prob = float(p2_log_probs[p2_cursor])
            value = float(p2_values[p2_cursor])
            p2_cursor += 1
        _apply_fast_rollout_action(
            slot,
            action=action,
            value=value,
            log_prob=log_prob,
            acting=acting,
            max_steps=max_steps,
        )


def _apply_fast_rollout_action(
    slot: _FastRolloutSlot,
    *,
    action: int,
    value: float,
    log_prob: float,
    acting: int,
    max_steps: int,
) -> None:
    obs_vec = np.asarray(slot.state["obs_vec"], dtype=np.float64)
    n_legal = max(
        1,
        int(slot.state.get("legal_count", 32) or 1),
    )
    next_state = slot.env.fast_step_index(action)
    env_reward = float(next_state.get("reward", 0.0) or 0.0)
    slot.terminated = bool(next_state.get("terminated", False))
    slot.truncated = bool(next_state.get("truncated", False))
    done = slot.terminated or slot.truncated
    slot.steps += 1

    next_obs_vec = np.asarray(next_state["obs_vec"], dtype=np.float64)
    agent_reward = env_reward if acting == 1 else -env_reward
    trans = {
        "obs_vec": obs_vec,
        "action": action,
        "reward": agent_reward,
        "value": value,
        "log_prob": log_prob,
        "done": float(done),
        "n_legal": n_legal,
        "next_obs_vec": next_obs_vec,
    }
    if acting == 1:
        slot.p1_trans.append(trans)
        slot.cur_p1_r += env_reward
    else:
        slot.p2_trans.append(trans)
        slot.cur_p2_r += -env_reward

    slot.final_p1_hp = _int_fast_state(next_state, "p1_health", slot.final_p1_hp)
    slot.final_p2_hp = _int_fast_state(next_state, "p2_health", slot.final_p2_hp)
    slot.final_p1_deck = _int_fast_state(next_state, "p1_deck", slot.final_p1_deck)
    slot.final_p2_deck = _int_fast_state(next_state, "p2_deck", slot.final_p2_deck)
    slot.final_turn_no = _int_fast_state(next_state, "turn_no", slot.final_turn_no)
    slot.state = next_state
    if done or slot.steps >= max_steps:
        slot.active = False


def _run_parallel_batched_fast_episodes(
    envs: list[TalisharEngineEnvironment],
    p1_policy: PPOAgent,
    p2_policy: PPOAgent,
    *,
    max_steps: int,
    warmup: bool,
    episode_indices: list[int],
    seed_base: Optional[int],
) -> list[dict[str, Any]]:
    """Run one episode per env slot with batched PPO inference across active slots."""
    slots: list[_FastRolloutSlot] = []
    for worker, env in enumerate(envs):
        ep_index = episode_indices[worker]
        ep_seed = (seed_base + worker) if seed_base is not None else None
        slot = _FastRolloutSlot(
            env=env,
            state={},
            p1_rng=np.random.default_rng((ep_seed * 31 + 7) if ep_seed is not None else None),
            p2_rng=np.random.default_rng((ep_seed * 31 + 13) if ep_seed is not None else None),
        )
        _reset_fast_rollout_slot(
            slot,
            seed=ep_seed,
            starting_player_id=1 + (ep_index % 2),
        )
        slots.append(slot)

    for _ in range(max_steps):
        if not any(slot.active for slot in slots):
            break
        _batched_fast_rollout_step(
            slots,
            p1_policy,
            p2_policy,
            warmup=warmup,
            max_steps=max_steps,
        )

    return [
        _fast_episode_result_from_slot(slot, warmup=warmup, max_steps=max_steps)
        for slot in slots
    ]


def _run_parallel_fast_episodes_threaded(
    envs: list[TalisharEngineEnvironment],
    p1_policy: PPOAgent,
    p2_policy: PPOAgent,
    *,
    max_steps: int,
    warmup: bool,
    episode_indices: list[int],
    seed_base: Optional[int],
    max_workers: int,
) -> list[dict[str, Any]]:
    """Run one episode per env using threads and per-step policy inference."""
    from concurrent.futures import ThreadPoolExecutor

    def _run_worker(worker: int) -> dict[str, Any]:
        ep_index = episode_indices[worker]
        ep_seed = (seed_base + worker) if seed_base is not None else None
        p1_rng = np.random.default_rng((ep_seed * 31 + 7) if ep_seed is not None else None)
        p2_rng = np.random.default_rng((ep_seed * 31 + 13) if ep_seed is not None else None)
        return _run_one_fast_episode(
            envs[worker],
            p1_policy,
            p2_policy,
            max_steps,
            ep_seed,
            warmup,
            p1_rng,
            p2_rng,
            starting_player_id=1 + (ep_index % 2),
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run_worker, worker) for worker in range(len(envs))]
        return [future.result() for future in futures]


def _env_supports_fast_training(env: Any) -> bool:
    return bool(
        getattr(env, "supports_fast_training", False)
        and hasattr(env, "fast_reset")
        and hasattr(env, "fast_step_index")
    )


def _fast_training_unavailable_reasons(env: Any) -> list[str]:
    """Collect blockers for the C++ numeric training path."""
    inner = getattr(env, "_cpp_env", None)
    if inner is not None and hasattr(inner, "fast_training_unavailable_reasons"):
        return list(inner.fast_training_unavailable_reasons())
    if hasattr(env, "fast_training_unavailable_reasons"):
        return list(env.fast_training_unavailable_reasons())
    if not getattr(env, "_using_cpp", False):
        return ["not using C++ engine"]
    return ["fast training API not exposed on environment wrapper"]


def _announce_training_backend(
    env: Any,
    *,
    require_fast_training: bool = False,
    label: str = "training",
) -> None:
    """Log which rollout path is active and optionally require the fast path."""
    using_cpp = bool(getattr(env, "_using_cpp", False))
    fast = _env_supports_fast_training(env)
    if fast:
        print(
            f"  [fast] {label}: C++ numeric path "
            "(fast_reset / fast_step_index)",
            flush=True,
        )
        return

    if using_cpp:
        reasons = "; ".join(_fast_training_unavailable_reasons(env)) or "unknown"
        msg = (
            f"C++ engine is loaded but fast {label} is unavailable: {reasons}"
        )
        if require_fast_training:
            raise RuntimeError(msg)
        print(f"  [WARN] {msg}", flush=True)
        print(
            f"  [slow] {label}: falling back to step() + JSON observations",
            flush=True,
        )
        return

    print(
        f"  [slow] {label}: HTTP Talishar path (no compiled C++ engine)",
        flush=True,
    )


def _merge_episode_transitions(
    p1_trans: list[dict[str, Any]],
    p2_trans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(p1_trans) + list(p2_trans)
    merged.sort(key=lambda t: int(t.get("step_order", 0)))
    return merged


def _flush_unified_ppo_buffers(
    policy: PPOAgent,
    p1_trans: list[dict[str, Any]],
    p2_trans: list[dict[str, Any]],
) -> None:
    merged = _merge_episode_transitions(p1_trans, p2_trans)
    if merged:
        buf = _transitions_to_buf(merged)
        _ppo_update(policy, buf, merged[-1]["next_obs_vec"])


def _flush_unified_warmup_buffers(
    policy: PPOAgent,
    p1_trans: list[dict[str, Any]],
    p2_trans: list[dict[str, Any]],
) -> None:
    merged = _merge_episode_transitions(p1_trans, p2_trans)
    if merged:
        buf = _transitions_to_buf(merged)
        _bc_update(policy, buf, merged[-1]["next_obs_vec"])


def _uses_unified_policy(p1_tiers: list[PPOAgent], p2_tiers: list[PPOAgent]) -> bool:
    return bool(p1_tiers and p2_tiers and p1_tiers[0] is p2_tiers[0])


def _flush_ppo_buffers_auto(
    p1_tiers: list[PPOAgent],
    p2_tiers: list[PPOAgent],
    p1_trans: list[dict[str, Any]],
    p2_trans: list[dict[str, Any]],
) -> None:
    if _uses_unified_policy(p1_tiers, p2_tiers):
        _flush_unified_ppo_buffers(p1_tiers[0], p1_trans, p2_trans)
    else:
        _flush_ppo_buffers(p1_tiers, p2_tiers, p1_trans, p2_trans)


def _flush_warmup_buffers_auto(
    p1_tiers: list[PPOAgent],
    p2_tiers: list[PPOAgent],
    p1_trans: list[dict[str, Any]],
    p2_trans: list[dict[str, Any]],
) -> None:
    if _uses_unified_policy(p1_tiers, p2_tiers):
        _flush_unified_warmup_buffers(p1_tiers[0], p1_trans, p2_trans)
    else:
        _flush_warmup_buffers(p1_tiers, p2_tiers, p1_trans, p2_trans)


def _flush_ppo_buffers(
    p1_tiers: list[PPOAgent],
    p2_tiers: list[PPOAgent],
    p1_trans: list[dict[str, Any]],
    p2_trans: list[dict[str, Any]],
) -> None:
    if p1_trans:
        p1_buf = _transitions_to_buf(p1_trans)
        _ppo_update_all_tiers(p1_tiers, p1_buf, p1_trans[-1]["next_obs_vec"])
    if p2_trans:
        p2_buf = _transitions_to_buf(p2_trans)
        _ppo_update_all_tiers(p2_tiers, p2_buf, p2_trans[-1]["next_obs_vec"])


def _flush_warmup_buffers(
    p1_tiers: list[PPOAgent],
    p2_tiers: list[PPOAgent],
    p1_trans: list[dict[str, Any]],
    p2_trans: list[dict[str, Any]],
) -> None:
    """One behavioural-cloning update from all warmup rollouts."""
    if p1_trans:
        p1_buf = _transitions_to_buf(p1_trans)
        _bc_update_all_tiers(p1_tiers, p1_buf, p1_trans[-1]["next_obs_vec"])
    if p2_trans:
        p2_buf = _transitions_to_buf(p2_trans)
        _bc_update_all_tiers(p2_tiers, p2_buf, p2_trans[-1]["next_obs_vec"])


def _write_live_state_snapshot(
    env: TalisharEngineEnvironment,
    obs: Any,
    out_path: Path,
    header: str = "",
) -> None:
    try:
        render_out = env.render()
        b64_data = getattr(render_out, "data", None)
        if isinstance(b64_data, str) and b64_data.strip():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(base64.b64decode(b64_data))
            if header:
                out_path.with_suffix(".meta.txt").write_text(header, encoding="utf-8")
            return
    except Exception:
        pass

    _write_state_image(obs, out_path, header=header)


def _run_one_fast_episode(
    env: TalisharEngineEnvironment,
    p1_policy: PPOAgent,
    p2_policy: PPOAgent,
    max_steps: int,
    seed: Optional[int],
    warmup: bool,
    p1_rng: np.random.Generator,
    p2_rng: np.random.Generator,
    starting_player_id: int = 1,
) -> dict[str, Any]:
    """Run one episode through the generated C++ numeric fast path."""
    def _int_state(state_dict: dict[str, Any], key: str, default: int) -> int:
        value = state_dict.get(key, default)
        return default if value is None else int(value)

    p1_trans: list[dict[str, Any]] = []
    p2_trans: list[dict[str, Any]] = []
    cur_p1_r = cur_p2_r = 0.0

    state = env.fast_reset(seed=seed, starting_player_id=starting_player_id)
    terminated = truncated = False
    steps_taken = 0
    final_p1_hp = _int_state(state, "p1_health", 0)
    final_p2_hp = _int_state(state, "p2_health", 0)
    final_p1_deck = _int_state(state, "p1_deck", 0)
    final_p2_deck = _int_state(state, "p2_deck", 0)
    final_turn_no = _int_state(state, "turn_no", 0)
    step_order = 0

    for _ in range(max_steps):
        acting = int(state["acting_player_id"])
        policy = p1_policy if acting == 1 else p2_policy
        rng = p1_rng if acting == 1 else p2_rng
        obs_vec = np.asarray(state["obs_vec"], dtype=np.float64)
        n_legal = max(1, int(state.get("legal_count", policy.n_actions) or 1))

        if warmup:
            action = int(rng.integers(n_legal))
            value = 0.0
            lp_all_action = 0.0
        else:
            logits, value = policy.predict_policy_value(obs_vec)
            logits = _mask_logits_to_legal(logits, n_legal)
            lp_all = _log_softmax(logits)[0]
            probs = _softmax(logits)[0]
            action = int(rng.choice(policy.n_actions, p=probs))
            if action >= n_legal:
                action = n_legal - 1
            lp_all_action = float(lp_all[action])

        next_state = env.fast_step_index(action)
        env_reward = float(next_state.get("reward", 0.0) or 0.0)
        terminated = bool(next_state.get("terminated", False))
        truncated = bool(next_state.get("truncated", False))
        done = terminated or truncated
        steps_taken += 1

        next_obs_vec = np.asarray(next_state["obs_vec"], dtype=np.float64)
        agent_reward = env_reward if acting == 1 else -env_reward
        trans = {
            "obs_vec": obs_vec,
            "action": action,
            "reward": agent_reward,
            "value": value,
            "log_prob": lp_all_action,
            "done": float(done),
            "n_legal": n_legal,
            "next_obs_vec": next_obs_vec,
            "step_order": step_order,
        }
        step_order += 1
        if acting == 1:
            p1_trans.append(trans)
            cur_p1_r += env_reward
        else:
            p2_trans.append(trans)
            cur_p2_r += -env_reward

        final_p1_hp = _int_state(next_state, "p1_health", final_p1_hp)
        final_p2_hp = _int_state(next_state, "p2_health", final_p2_hp)
        final_p1_deck = _int_state(next_state, "p1_deck", final_p1_deck)
        final_p2_deck = _int_state(next_state, "p2_deck", final_p2_deck)
        final_turn_no = _int_state(next_state, "turn_no", final_turn_no)
        state = next_state
        if done:
            break

    if not terminated and not truncated and steps_taken >= max_steps:
        truncated = True

    if terminated:
        if final_p1_hp > final_p2_hp and p2_trans:
            last = p2_trans[-1]
            p2_trans.append({
                "obs_vec": last["next_obs_vec"],
                "action": last["action"],
                "reward": -1.0,
                "value": 0.0,
                "log_prob": last["log_prob"],
                "done": 1.0,
                "n_legal": last["n_legal"],
                "next_obs_vec": np.zeros_like(last["next_obs_vec"]),
            })
            cur_p2_r -= 1.0
        elif final_p2_hp > final_p1_hp and p1_trans:
            last = p1_trans[-1]
            p1_trans.append({
                "obs_vec": last["next_obs_vec"],
                "action": last["action"],
                "reward": -1.0,
                "value": 0.0,
                "log_prob": last["log_prob"],
                "done": 1.0,
                "n_legal": last["n_legal"],
                "next_obs_vec": np.zeros_like(last["next_obs_vec"]),
            })
            cur_p1_r -= 1.0

    return {
        "p1_transitions": p1_trans,
        "p2_transitions": p2_trans,
        "p1_reward": cur_p1_r,
        "p2_reward": cur_p2_r,
        "terminated": terminated,
        "truncated": truncated,
        "warmup": warmup,
        "steps": steps_taken,
        "p1_hp": final_p1_hp,
        "p2_hp": final_p2_hp,
        "turn_no": final_turn_no,
        "p1_deck": final_p1_deck,
        "p2_deck": final_p2_deck,
    }


def _run_one_episode(
    env: TalisharEngineEnvironment,
    p1_policy: PPOAgent,
    p2_policy: PPOAgent,
    max_steps: int,
    seed: Optional[int],
    warmup: bool,
    p1_rng: np.random.Generator,
    p2_rng: np.random.Generator,
    starting_player_id: int = 1,
) -> dict[str, Any]:
    """Run one complete episode and return all transitions for both players.

    Weight arrays on the policy objects are accessed read-only so this function
    is safe to call from multiple threads simultaneously.  Each caller must
    supply its own ``p1_rng`` / ``p2_rng`` instances so there is no shared
    mutable RNG state between threads.
    """
    if _env_supports_fast_training(env):
        return _run_one_fast_episode(
            env, p1_policy, p2_policy, max_steps, seed, warmup, p1_rng, p2_rng,
            starting_player_id=starting_player_id,
        )

    p1_trans: list[dict[str, Any]] = []
    p2_trans: list[dict[str, Any]] = []
    cur_p1_r = cur_p2_r = 0.0

    reset_out = env.reset(seed=seed)
    obs = _get(reset_out, "observation", reset_out)
    step_info = _get(reset_out, "info", {})
    terminated = truncated = False
    steps_taken = 0
    final_p1_hp = final_p2_hp = final_turn_no = None
    final_p1_deck = final_p2_deck = None
    step_order = 0

    for _ in range(max_steps):
        acting = env._acting_player_id
        policy = p1_policy if acting == 1 else p2_policy
        rng    = p1_rng    if acting == 1 else p2_rng

        obs_vec = _resolve_obs_vec(obs, step_info, policy)

        if warmup:
            env_action = str(env.sample_action())
            action = _env_action_to_index(obs, env_action, policy)
            value = 0.0
            lp_all_action = 0.0
        else:
            logits, value, probs = _policy_forward(policy, obs, obs_vec)
            lp_all = _log_softmax(logits)[0]
            action = int(rng.choice(policy.n_actions, p=probs))
            env_action = _to_env_action(obs, action, policy._mask_actions)
            lp_all_action = float(lp_all[action])

        n_legal = _n_legal_of(obs)

        step_out    = env.step(env_action)
        env_reward  = float(_get(step_out, "reward", 0.0))
        terminated  = bool(_get(step_out, "terminated", False))
        truncated   = bool(_get(step_out, "truncated",  False))
        done        = terminated or truncated
        next_obs    = _get(step_out, "observation", obs)
        step_info   = _get(step_out, "info", {})
        steps_taken += 1
        # Track final game state for diagnostics
        try:
            final_p1_hp, final_p2_hp = absolute_p1_p2_hp_from_env(env)
            if final_p1_hp is not None:
                final_p1_hp = int(final_p1_hp)
            if final_p2_hp is not None:
                final_p2_hp = int(final_p2_hp)
            if isinstance(next_obs, str):
                _s = json.loads(next_obs)
            else:
                _s = next_obs if isinstance(next_obs, dict) else {}
            final_turn_no = int(_s.get("turnNo", 0) or 0)
            _raw = env._last_state
            final_p1_deck = _raw.get("playerDeckCount")
            final_p2_deck = _raw.get("opponentDeckCount")
        except Exception:
            pass
        next_obs_vec = _resolve_obs_vec(next_obs, step_info, policy)

        agent_reward = env_reward if acting == 1 else -env_reward
        trans = {
            "obs_vec":     obs_vec,
            "action":      action,
            "reward":      agent_reward,
            "value":       value,
            "log_prob":    lp_all_action,
            "done":        float(done),
            "n_legal":     n_legal if n_legal is not None else policy.n_actions,
            "next_obs_vec": next_obs_vec,
            "step_order": step_order,
        }
        step_order += 1
        if acting == 1:
            p1_trans.append(trans)
            cur_p1_r += env_reward
        else:
            p2_trans.append(trans)
            cur_p2_r += -env_reward

        if done:
            break
        obs = next_obs

    if not terminated and not truncated and steps_taken >= max_steps:
        truncated = True

    # ── inject terminal loss for the loser ───────────────────────────────────
    # The terminal +1 reward only goes to the player who happened to be ACTING
    # on the final step.  The other player received no terminal signal at all,
    # leaving their episode reward positive (from intermediate damage) even when
    # they lost.  Inject a synthetic done=True transition with reward=-1 so:
    #   • the critic learns V(terminal) ≈ -1 for the loser's perspective
    #   • cur_p1_r / cur_p2_r correctly reflect the game outcome for reporting
    if terminated and final_p1_hp is not None and final_p2_hp is not None:
        if final_p1_hp > final_p2_hp:
            # P1 won → P2 never saw a terminal -1
            if p2_trans:
                last = p2_trans[-1]
                p2_trans.append({
                    "obs_vec":      last["next_obs_vec"],
                    "action":       last["action"],
                    "reward":       -1.0,
                    "value":        0.0,
                    "log_prob":     last["log_prob"],
                    "done":         1.0,
                    "n_legal":      last["n_legal"],
                    "next_obs_vec": np.zeros_like(last["next_obs_vec"]),
                })
                cur_p2_r -= 1.0
        elif final_p2_hp > final_p1_hp:
            # P2 won → P1 never saw a terminal -1
            if p1_trans:
                last = p1_trans[-1]
                p1_trans.append({
                    "obs_vec":      last["next_obs_vec"],
                    "action":       last["action"],
                    "reward":       -1.0,
                    "value":        0.0,
                    "log_prob":     last["log_prob"],
                    "done":         1.0,
                    "n_legal":      last["n_legal"],
                    "next_obs_vec": np.zeros_like(last["next_obs_vec"]),
                })
                cur_p1_r -= 1.0

    return {
        "p1_transitions": p1_trans,
        "p2_transitions": p2_trans,
        "p1_reward":      cur_p1_r,
        "p2_reward":      cur_p2_r,
        "terminated":     terminated,
        "truncated":      truncated,
        "warmup":         warmup,
        "steps":          steps_taken,
        "p1_hp":          final_p1_hp,
        "p2_hp":          final_p2_hp,
        "turn_no":        final_turn_no,
        "p1_deck":        final_p1_deck,
        "p2_deck":        final_p2_deck,
    }


def _transitions_to_buf(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    """Flatten a list of transition dicts into a PPO rollout buffer dict."""
    buf = _empty_buf()
    for t in transitions:
        buf["obs"].append(t["obs_vec"])
        buf["actions"].append(t["action"])
        buf["rewards"].append(t["reward"])
        buf["values"].append(t["value"])
        buf["log_probs"].append(t["log_prob"])
        buf["dones"].append(t["done"])
        buf["n_legal"].append(t["n_legal"])
    return buf


def warm_start_from_episode_cache(
    episode_cache: EpisodeCache,
    p1_deck: str,
    p2_deck: str,
    p1_tiers: list[PPOAgent],
    p2_tiers: list[PPOAgent],
    obs_dim: int,
    *,
    max_load: Optional[int] = None,
) -> int:
    """Replay cached episodes as behavioural-cloning warm-start for both agents.

    Uses cross-entropy (behavioural cloning) instead of PPO to update the
    actor, which avoids the stale log-prob problem: cached episodes were
    collected under a previous policy so their stored log_probs are stale,
    making PPO ratio ≈ exp(random - old_random) ≈ noise — the actor gradient
    is meaningless.  BC trains the actor directly to reproduce the demonstrated
    actions regardless of the old policy.

    The critic is updated via MSE against GAE returns recomputed from the
    *current* critic so value bootstraps are always fresh.

    Parameters
    ----------
    max_load:
        Cap on how many episodes to replay.  Defaults to loading all compatible
        episodes (up to the cache's stored maximum).
    """
    episodes = episode_cache.load_episodes(
        p1_deck, p2_deck, obs_dim=obs_dim, max_load=max_load
    )
    if not episodes:
        return 0

    if _uses_unified_policy(p1_tiers, p2_tiers):
        policy = p1_tiers[0]
        for ep in episodes:
            merged = _merge_episode_transitions(
                ep.get("p1_transitions", []),
                ep.get("p2_transitions", []),
            )
            if merged:
                buf = _transitions_to_buf(merged)
                _bc_update(policy, buf, merged[-1]["next_obs_vec"])
        return len(episodes)

    replayed = 0
    for ep in episodes:
        p1_trans = ep["p1_transitions"]
        p2_trans = ep["p2_transitions"]

        if p1_trans:
            p1_buf = _transitions_to_buf(p1_trans)
            next_p1 = p1_trans[-1]["next_obs_vec"]
            _bc_update_all_tiers(p1_tiers, p1_buf, next_p1)

        if p2_trans:
            p2_buf = _transitions_to_buf(p2_trans)
            next_p2 = p2_trans[-1]["next_obs_vec"]
            _bc_update_all_tiers(p2_tiers, p2_buf, next_p2)

        replayed += 1

    return replayed


def _save_episode_to_cache(
    episode_cache: EpisodeCache,
    result: dict[str, Any],
    p1_deck: str,
    p2_deck: str,
    obs_dim: int,
) -> None:
    """Save a terminated (non-truncated) episode result to the episode cache.

    Both warmup (heuristic policy) and PPO-policy completed episodes are
    cached — every game completion is valuable training data regardless of
    which policy produced it.  The ``warmup`` flag is stored so the cache can
    report the mix of episode sources.
    """
    if result.get("truncated") or not result.get("terminated"):
        return
    p1_trans = result.get("p1_transitions", [])
    p2_trans = result.get("p2_transitions", [])
    if not p1_trans and not p2_trans:
        return
    is_warmup = bool(result.get("warmup", False))
    try:
        episode_cache.add_episode(
            p1_deck,
            p2_deck,
            obs_dim=obs_dim,
            p1_transitions=p1_trans,
            p2_transitions=p2_trans,
            p1_reward=float(result.get("p1_reward", 0.0)),
            p2_reward=float(result.get("p2_reward", 0.0)),
            steps=int(result.get("steps", 0)),
            warmup=is_warmup,
        )
    except Exception:
        pass  # never let caching errors abort training


def _safe_run_one_episode(
    env: TalisharEngineEnvironment,
    p1_policy: PPOAgent,
    p2_policy: PPOAgent,
    max_steps: int,
    seed: Optional[int],
    warmup: bool,
    p1_rng: np.random.Generator,
    p2_rng: np.random.Generator,
    matchup: "Matchup",
    base_url: str,
    game_format: str,
    starting_player_id: int = 1,
) -> dict[str, Any]:
    """Run one episode, recycling the env on any exception."""
    try:
        return _run_one_episode(
            env, p1_policy, p2_policy, max_steps, seed, warmup, p1_rng, p2_rng,
            starting_player_id=starting_player_id,
        )
    except Exception as exc:
        # Try to recycle the environment rather than leaving it in a bad state.
        try:
            env.close()
        except Exception:
            pass
        try:
            new_env = make_env(matchup, base_url=base_url, game_format=game_format, max_turns=max_steps)
            # Reuse the same object slot so the caller's envs[] reference stays valid.
            env.__dict__.update(new_env.__dict__)
        except Exception:
            pass
        raise exc


def train_agents_from_both_perspectives_parallel(
    matchup: Matchup,
    base_url: str,
    game_format: str,
    p1_tiers: list[PPOAgent],
    p2_tiers: list[PPOAgent],
    n_episodes: int,
    max_steps: int,
    seed: Optional[int] = None,
    warmup_episodes: int = DEFAULT_WARMUP_EPISODES,
    n_workers: int = 2,
    live_state_image_path: Optional[Path] = None,
    episode_cache: Optional[EpisodeCache] = None,
    on_episodes_progress: Optional[
        Callable[..., None]
    ] = None,
    suppress_train_progress: bool = False,
) -> tuple[list[float], list[float], dict[str, Any]]:
    """Parallel rollout version of ``train_agents_from_both_perspectives``.

    Creates independent Talishar game sessions and runs episodes in parallel
    batches via ``ThreadPoolExecutor``.  After each batch the PPO update runs
    single-threaded on the merged buffer, then the next batch starts with
    updated weights.

    ``n_workers`` controls how many episodes run concurrently per batch.
    Worker envs are kept alive for the full ``n_episodes`` run so sessions are
    not respawned between batches.

    Each worker has its own ``np.random.default_rng`` so there is no shared
    mutable state between threads.  Agent weight arrays are accessed read-only
    during the forward passes which is safe for simultaneous reads in NumPy.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    batch_parallelism = max(1, n_workers)

    p1_policy = p1_tiers[0]
    p2_policy = p2_tiers[0]

    # ── bootstrap: infer dims and init nets on a throw-away env ──────────────
    probe_env = make_env(matchup, base_url=base_url, game_format=game_format, max_turns=max_steps)
    _announce_training_backend(
        probe_env,
        label="parallel training bootstrap",
        require_fast_training=bool(matchup.cpp_engine_dir),
    )
    if _env_supports_fast_training(probe_env):
        probe_state = probe_env.fast_reset(seed=seed)
        n_actions_p1 = n_actions_p2 = int(probe_env.fast_action_capacity())
        mask_p1 = mask_p2 = True
        obs_vec = np.asarray(probe_state["obs_vec"], dtype=np.float64)
    else:
        n_actions_p1, mask_p1 = _infer_action_capacity(probe_env, seed=seed)
        n_actions_p2, mask_p2 = _infer_action_capacity(probe_env, seed=seed)
        probe_reset = probe_env.reset(seed=seed)
        probe_obs   = _get(probe_reset, "observation", probe_reset)
        obs_vec     = p1_policy._obs_to_vec(probe_obs)
    probe_env.close()

    _sync_tier_agent_config(p1_tiers, n_actions_p1, mask_p1)
    _sync_tier_agent_config(p2_tiers, n_actions_p2, mask_p2)
    _init_tier_nets(p1_tiers, obs_vec.shape[0])
    _init_tier_nets(p2_tiers, obs_vec.shape[0])

    # ── episode-cache warm-start (before any live games) ─────────────────────
    obs_dim = obs_vec.shape[0]
    if episode_cache is not None:
        cached_n = episode_cache.count(matchup.p1_deck, matchup.p2_deck, obs_dim=obs_dim)
        if cached_n > 0:
            if episode_cache.should_skip_warmup(matchup.p1_deck, matchup.p2_deck, obs_dim=obs_dim):
                print(
                    f"  [cache] {cached_n} cached episodes ≥ threshold "
                    f"({episode_cache.warmup_skip_threshold}) — skipping default-policy warmup, "
                    f"warm-starting from cache"
                )
                replayed = warm_start_from_episode_cache(
                    episode_cache, matchup.p1_deck, matchup.p2_deck,
                    p1_tiers, p2_tiers, obs_dim,
                )
                print(f"  [cache] replayed {replayed} cached episode(s) as PPO warm-start")
                warmup_episodes = 0  # skip default-policy phase
            else:
                print(
                    f"  [cache] {cached_n} cached episodes (below skip threshold "
                    f"{episode_cache.warmup_skip_threshold}) — partial warm-start from cache, "
                    f"continuing with default-policy warmup"
                )
                replayed = warm_start_from_episode_cache(
                    episode_cache, matchup.p1_deck, matchup.p2_deck,
                    p1_tiers, p2_tiers, obs_dim,
                )
                print(f"  [cache] replayed {replayed} cached episode(s) as partial warm-start")

    # ── create worker envs ────────────────────────────────────────────────────
    print(
        f"  [parallel] spawning {batch_parallelism} worker game session(s) "
        f"({n_episodes} episodes, batch={batch_parallelism})…"
    )
    envs: list[TalisharEngineEnvironment] = []
    for w in range(batch_parallelism):
        envs.append(
            make_env(matchup, base_url=base_url, game_format=game_format, max_turns=max_steps)
        )
    print(f"  [parallel] {batch_parallelism} sessions ready", flush=True)
    use_batched_fast_rollout = _env_supports_fast_training(envs[0])
    if use_batched_fast_rollout:
        print(
            f"  [parallel] batched PPO inference enabled across "
            f"{batch_parallelism} rollout slot(s)",
            flush=True,
        )

    p1_ep_rewards:  list[float] = []
    p2_ep_rewards:  list[float] = []
    p1_outcomes:    list[str] = []
    timeout_episodes  = 0
    terminated_episodes = 0
    skipped_episodes    = 0
    completed          = 0
    progress_every     = max(1, n_episodes // 100)
    batch_progress     = max(1, batch_parallelism)
    progress_t0        = time.time()
    # Per-episode wall-clock timeout: max_steps * 5 s per step, floor at 120 s.
    # Per-episode wall-clock timeout: 3 s per step (down from 5 s) is sufficient
    # for a local Docker server.  Floor raised to 180 s to cover game setup time.
    episode_timeout_secs = max(180, max_steps * 3)
    shutdown_flag = False
    warmup_p1_accum: list[dict[str, Any]] = []
    warmup_p2_accum: list[dict[str, Any]] = []
    ppo_p1_accum: list[dict[str, Any]] = []
    ppo_p2_accum: list[dict[str, Any]] = []
    warmup_bc_applied = warmup_episodes <= 0

    try:
        with ThreadPoolExecutor(max_workers=batch_parallelism) as pool:
            while completed < n_episodes and not shutdown_flag:
                batch_size = min(batch_parallelism, n_episodes - completed)
                in_warmup  = completed < warmup_episodes
                if completed == 0:
                    print(
                        f"  [parallel] running first batch ({batch_size} episode(s), "
                        f"timeout={episode_timeout_secs}s/ep)…",
                        flush=True,
                    )

                # Submit one episode per worker in this batch.
                seed_base = (seed + completed) if seed is not None else None
                batch_results: list[dict[str, Any]] = []
                if use_batched_fast_rollout:
                    batch_results = _run_parallel_batched_fast_episodes(
                        envs[:batch_size],
                        p1_policy,
                        p2_policy,
                        max_steps=max_steps,
                        warmup=in_warmup,
                        episode_indices=[completed + w for w in range(batch_size)],
                        seed_base=seed_base,
                    )
                else:
                    futures = {}
                    for w in range(batch_size):
                        episode_index = completed + w
                        starting_player_id = 1 + (episode_index % 2)
                        ep_seed    = (seed_base + w) if seed_base is not None else None
                        p1_rng     = np.random.default_rng(
                            (ep_seed * 31 + 7)  if ep_seed is not None else None
                        )
                        p2_rng     = np.random.default_rng(
                            (ep_seed * 31 + 13) if ep_seed is not None else None
                        )
                        fut = pool.submit(
                            _safe_run_one_episode,
                            envs[w], p1_policy, p2_policy,
                            max_steps, ep_seed, in_warmup, p1_rng, p2_rng,
                            matchup, base_url, game_format,
                            starting_player_id,
                        )
                        futures[fut] = w

                # Collect results and merge buffers.
                batch_p1_trans: list[dict] = []
                batch_p2_trans: list[dict] = []
                from concurrent.futures import TimeoutError as FutureTimeoutError
                result_iter: list[dict[str, Any]]
                if use_batched_fast_rollout:
                    result_iter = batch_results
                else:
                    result_iter = []
                    for fut in as_completed(futures, timeout=episode_timeout_secs + 30):
                        try:
                            result_iter.append(fut.result(timeout=episode_timeout_secs))
                        except FutureTimeoutError:
                            print(f"  [parallel] episode timed out after {episode_timeout_secs}s — skipping")
                            skipped_episodes += 1
                            p1_ep_rewards.append(0.0)
                            p2_ep_rewards.append(0.0)
                            p1_outcomes.append(
                                classify_p1_episode_outcome(skipped=True)
                            )
                            completed += 1
                            if on_episodes_progress is not None:
                                try:
                                    on_episodes_progress(
                                        completed, p1_ep_rewards, p2_ep_rewards, p1_outcomes
                                    )
                                except TypeError:
                                    on_episodes_progress(
                                        completed, p1_ep_rewards, p2_ep_rewards
                                    )
                                except Exception as exc:
                                    print(f"  [parallel] progress callback failed ({exc!r})")
                            continue
                        except KeyboardInterrupt:
                            shutdown_flag = True
                            break
                        except Exception as exc:
                            print(f"  [parallel] episode failed ({exc!r}) — skipping")
                            skipped_episodes += 1
                            p1_ep_rewards.append(0.0)
                            p2_ep_rewards.append(0.0)
                            p1_outcomes.append(
                                classify_p1_episode_outcome(skipped=True)
                            )
                            completed += 1
                            if on_episodes_progress is not None:
                                try:
                                    on_episodes_progress(
                                        completed, p1_ep_rewards, p2_ep_rewards, p1_outcomes
                                    )
                                except TypeError:
                                    on_episodes_progress(
                                        completed, p1_ep_rewards, p2_ep_rewards
                                    )
                                except Exception as exc_cb:
                                    print(f"  [parallel] progress callback failed ({exc_cb!r})")
                            continue
                    if shutdown_flag:
                        break

                for result in result_iter:
                    batch_p1_trans.extend(result["p1_transitions"])
                    batch_p2_trans.extend(result["p2_transitions"])
                    p1_ep_rewards.append(result["p1_reward"])
                    p2_ep_rewards.append(result["p2_reward"])
                    p1_outcomes.append(
                        classify_p1_episode_outcome(
                            p1_hp=result.get("p1_hp"),
                            p2_hp=result.get("p2_hp"),
                            p1_deck=result.get("p1_deck"),
                            p2_deck=result.get("p2_deck"),
                            terminated=bool(result.get("terminated")),
                            truncated=bool(result.get("truncated")),
                        )
                    )
                    if result["truncated"]:
                        timeout_episodes += 1
                        p1_hp    = result.get("p1_hp")
                        p2_hp    = result.get("p2_hp")
                        turn_no  = result.get("turn_no")
                        steps    = result.get("steps")
                        p1_deck  = result.get("p1_deck")
                        p2_deck  = result.get("p2_deck")
                        deck_str = (
                            f"  deck={p1_deck}/{p2_deck}"
                            if p1_deck is not None else ""
                        )
                        hp_str   = (
                            f"P1={p1_hp}hp  P2={p2_hp}hp  turn={turn_no}  steps={steps}{deck_str}"
                            if p1_hp is not None else "hp=unknown"
                        )
                        print(f"  [unfinished ep #{completed+1}] {hp_str}")
                    if result["terminated"]:
                        terminated_episodes += 1
                        if episode_cache is not None:
                            _save_episode_to_cache(
                                episode_cache, result,
                                matchup.p1_deck, matchup.p2_deck,
                                obs_dim=obs_vec.shape[0],
                            )
                    completed += 1
                    if on_episodes_progress is not None and (
                        completed == n_episodes
                        or completed <= 10
                        or (warmup_episodes > 0 and completed == warmup_episodes)
                        or completed % 50 == 0
                    ):
                        try:
                            on_episodes_progress(
                                completed, p1_ep_rewards, p2_ep_rewards, p1_outcomes
                            )
                        except TypeError:
                            on_episodes_progress(
                                completed, p1_ep_rewards, p2_ep_rewards
                            )
                        except Exception as exc:
                            print(f"  [parallel] progress callback failed ({exc!r})")
                if shutdown_flag:
                    break

                if live_state_image_path is not None and batch_p1_trans:
                    last_obs_vec = batch_p1_trans[-1]["obs_vec"]
                    _write_state_image(
                        {"obs_vec_shape": str(last_obs_vec.shape)},
                        live_state_image_path,
                        header=(
                            f"episode={completed}/{n_episodes} "
                            f"parallel_batch={batch_parallelism}"
                        ),
                    )

                if in_warmup:
                    warmup_p1_accum.extend(batch_p1_trans)
                    warmup_p2_accum.extend(batch_p2_trans)
                    if completed >= warmup_episodes and not warmup_bc_applied:
                        print(
                            f"  [warmup] behavioural-cloning update from "
                            f"{len(warmup_p1_accum) + len(warmup_p2_accum)} transitions"
                        )
                        _flush_warmup_buffers_auto(
                            p1_tiers, p2_tiers, warmup_p1_accum, warmup_p2_accum,
                        )
                        warmup_p1_accum.clear()
                        warmup_p2_accum.clear()
                        warmup_bc_applied = True
                else:
                    ppo_p1_accum.extend(batch_p1_trans)
                    ppo_p2_accum.extend(batch_p2_trans)
                    rollout_ready = (
                        len(ppo_p1_accum) >= DEFAULT_PPO_ROLLOUT_BATCH
                        or len(ppo_p2_accum) >= DEFAULT_PPO_ROLLOUT_BATCH
                        or completed >= n_episodes
                    )
                    if rollout_ready:
                        _flush_ppo_buffers_auto(
                            p1_tiers, p2_tiers, ppo_p1_accum, ppo_p2_accum,
                        )
                        ppo_p1_accum.clear()
                        ppo_p2_accum.clear()

                # Progress logging.
                if (
                    not suppress_train_progress
                    and (
                    completed <= max(10, batch_parallelism)
                    or completed % batch_progress == 0
                    or completed % progress_every == 0
                    or completed == n_episodes
                    )
                ):
                    elapsed  = time.time() - progress_t0
                    pct      = (completed / max(1, n_episodes)) * 100.0
                    p1_avg   = float(np.mean(p1_ep_rewards)) if p1_ep_rewards else 0.0
                    p2_avg   = float(np.mean(p2_ep_rewards)) if p2_ep_rewards else 0.0
                    ep_rate  = completed / max(elapsed, 1e-9)
                    eta_secs = (n_episodes - completed) / ep_rate if ep_rate > 0 else float("inf")
                    t_rate   = timeout_episodes / max(1, completed)
                    print(
                        f"  [train-progress] episodes={completed}/{n_episodes} "
                        f"({pct:6.2f}%) elapsed={elapsed:.1f}s "
                        f"rate={ep_rate:.3f}ep/s eta={eta_secs / 60:.1f}m "
                        f"batch={batch_parallelism} "
                        f"warmup={'yes' if in_warmup else 'no '} "
                        f"timeouts={timeout_episodes} ({t_rate * 100:.1f}%) "
                        f"skipped={skipped_episodes} "
                        f"p1_avg={p1_avg:+.3f} p2_avg={p2_avg:+.3f}"
                    )
    finally:
        if not warmup_bc_applied and (warmup_p1_accum or warmup_p2_accum):
            _flush_warmup_buffers_auto(p1_tiers, p2_tiers, warmup_p1_accum, warmup_p2_accum)
        if ppo_p1_accum or ppo_p2_accum:
            _flush_ppo_buffers_auto(p1_tiers, p2_tiers, ppo_p1_accum, ppo_p2_accum)
        for env in envs:
            try:
                env.close()
            except Exception:
                pass

    total_eps = len(p1_ep_rewards)
    outcome_summary = summarize_p1_outcomes(p1_outcomes, episodes=total_eps)
    stats = {
        "episodes":   total_eps,
        "timeouts":   outcome_summary["timeouts"],
        "terminated": terminated_episodes,
        "skipped":    skipped_episodes,
        "timeout_rate": outcome_summary["timeout_rate"],
        "p1_outcomes": p1_outcomes,
        **outcome_summary,
    }
    return p1_ep_rewards, p2_ep_rewards, stats


def train_agents_from_both_perspectives(
    env: TalisharEngineEnvironment,
    p1_tiers: list[PPOAgent],
    p2_tiers: list[PPOAgent],
    n_episodes: int,
    max_steps: int,
    seed: Optional[int] = None,
    warmup_episodes: int = DEFAULT_WARMUP_EPISODES,
    live_state_image_path: Optional[Path] = None,
    episode_cache: Optional[EpisodeCache] = None,
    p1_deck: str = "",
    p2_deck: str = "",
    suppress_train_progress: bool = False,
    after_episode: Optional[Callable[[int], None]] = None,
) -> tuple[list[float], list[float], dict[str, Any]]:
    p1_policy = p1_tiers[0]
    p2_policy = p2_tiers[0]

    if _env_supports_fast_training(env):
        n_actions_p1 = n_actions_p2 = int(env.fast_action_capacity())
        mask_p1 = mask_p2 = True
    else:
        n_actions_p1, mask_p1 = _infer_action_capacity(env, seed=seed)
        n_actions_p2, mask_p2 = _infer_action_capacity(env, seed=seed)

    _sync_tier_agent_config(p1_tiers, n_actions_p1, mask_p1)
    _sync_tier_agent_config(p2_tiers, n_actions_p2, mask_p2)

    p1_ep_rewards: list[float] = []
    p2_ep_rewards: list[float] = []
    p1_outcomes: list[str] = []
    p1_buf = _empty_buf()
    p2_buf = _empty_buf()
    unified_buf = p1_buf

    ep_seed = seed
    if _env_supports_fast_training(env):
        reset_out = {"info": env.fast_reset(seed=ep_seed)}
        obs = np.asarray(reset_out["info"]["obs_vec"], dtype=np.float64)
        obs_vec = obs
    else:
        reset_out = env.reset(seed=ep_seed)
        obs = _get(reset_out, "observation", reset_out)
        obs_vec = p1_policy._obs_to_vec(obs)
    _init_tier_nets(p1_tiers, obs_vec.shape[0])
    _init_tier_nets(p2_tiers, obs_vec.shape[0])

    # ── episode-cache warm-start (before any live games) ─────────────────────
    _obs_dim = obs_vec.shape[0]
    if episode_cache is not None and p1_deck and p2_deck:
        cached_n = episode_cache.count(p1_deck, p2_deck, obs_dim=_obs_dim)
        if cached_n > 0:
            if episode_cache.should_skip_warmup(p1_deck, p2_deck, obs_dim=_obs_dim):
                print(
                    f"  [cache] {cached_n} cached episodes ≥ threshold "
                    f"({episode_cache.warmup_skip_threshold}) — skipping default-policy warmup, "
                    f"warm-starting from cache"
                )
                replayed = warm_start_from_episode_cache(
                    episode_cache, p1_deck, p2_deck, p1_tiers, p2_tiers, _obs_dim,
                )
                print(f"  [cache] replayed {replayed} cached episode(s) as PPO warm-start")
                warmup_episodes = 0  # skip default-policy phase in the loop
            else:
                print(
                    f"  [cache] {cached_n} cached episodes (below skip threshold "
                    f"{episode_cache.warmup_skip_threshold}) — partial warm-start from cache, "
                    f"continuing with default-policy warmup"
                )
                replayed = warm_start_from_episode_cache(
                    episode_cache, p1_deck, p2_deck, p1_tiers, p2_tiers, _obs_dim,
                )
                print(f"  [cache] replayed {replayed} cached episode(s) as partial warm-start")

    completed = 0
    cur_p1_r = cur_p2_r = 0.0
    timeout_episodes = 0
    terminated_episodes = 0
    total_steps = n_episodes * max_steps
    global_step = 0
    episode_step = 0
    progress_every = max(1, n_episodes // 100)  # ~1% cadence after warmup
    progress_t0 = time.time()
    # Per-episode transition accumulators for episode caching (only populated
    # when episode_cache is provided to avoid overhead when unused).
    _ep_p1_trans: list[dict[str, Any]] = []
    _ep_p2_trans: list[dict[str, Any]] = []
    warmup_p1_trans: list[dict[str, Any]] = []
    warmup_p2_trans: list[dict[str, Any]] = []
    warmup_bc_applied = warmup_episodes <= 0
    step_info = _get(reset_out, "info", {})
    step_order = 0

    while completed < n_episodes and global_step < total_steps:
        acting = env._acting_player_id
        policy = p1_policy if acting == 1 else p2_policy
        tier_agents = p1_tiers if acting == 1 else p2_tiers
        if _uses_unified_policy(p1_tiers, p2_tiers):
            buf = unified_buf
        else:
            buf = p1_buf if acting == 1 else p2_buf
        in_warmup = completed < warmup_episodes

        obs_vec = _resolve_obs_vec(obs, step_info, policy)

        if in_warmup:
            env_action = str(env.sample_action())
            action = _env_action_to_index(obs, env_action, policy)
            value = 0.0
            log_prob = 0.0
        else:
            logits, value, probs = _policy_forward(policy, obs, obs_vec)
            lp_all = _log_softmax(logits)[0]
            action = int(policy._rng_np.choice(policy.n_actions, p=probs))
            env_action = _to_env_action(obs, action, policy._mask_actions)
            log_prob = float(lp_all[action])
        n_legal = _n_legal_of(obs)

        step_out = env.step(env_action)
        env_reward = float(_get(step_out, "reward", 0.0))
        terminated = bool(_get(step_out, "terminated", False))
        truncated = bool(_get(step_out, "truncated", False))
        done = terminated or truncated
        episode_step += 1
        step_info = _get(step_out, "info", {})

        if live_state_image_path is not None:
            current_obs = _get(step_out, "observation", obs)
            header = (
                f"episode={completed + 1}/{n_episodes} "
                f"step={episode_step} acting={acting} reward={env_reward:+.3f} "
                f"terminated={terminated} truncated={truncated}"
            )
            _write_live_state_snapshot(env, current_obs, live_state_image_path, header=header)

        agent_reward = env_reward if acting == 1 else -env_reward
        if acting == 1:
            cur_p1_r += env_reward
        else:
            cur_p2_r += -env_reward

        next_obs_vec = _resolve_obs_vec(_get(step_out, "observation", obs), step_info, policy)
        trans = {
            "obs_vec": obs_vec,
            "action": action,
            "reward": agent_reward,
            "value": value,
            "log_prob": log_prob,
            "done": float(done),
            "n_legal": n_legal if n_legal is not None else policy.n_actions,
            "next_obs_vec": next_obs_vec,
            "step_order": step_order,
        }
        step_order += 1

        if in_warmup:
            if acting == 1:
                warmup_p1_trans.append(trans)
            else:
                warmup_p2_trans.append(trans)
        else:
            buf["obs"].append(obs_vec)
            buf["actions"].append(action)
            buf["rewards"].append(agent_reward)
            buf["values"].append(value)
            buf["log_probs"].append(log_prob)
            buf["dones"].append(float(done))
            buf["n_legal"].append(n_legal if n_legal is not None else policy.n_actions)

        if episode_cache is not None:
            if acting == 1:
                _ep_p1_trans.append(trans)
            else:
                _ep_p2_trans.append(trans)

        global_step += 1

        if not done and episode_step >= max_steps:
            truncated = True
            done = True

        if done:
            p1_hp = p2_hp = None
            p1_deck = p2_deck = None
            try:
                p1_hp, p2_hp = absolute_p1_p2_hp_from_env(env)
                p1_deck, p2_deck = absolute_p1_p2_deck_from_env(env)
                if p1_hp is not None:
                    p1_hp = int(p1_hp)
                if p2_hp is not None:
                    p2_hp = int(p2_hp)
            except Exception:
                pass
            p1_outcomes.append(
                classify_p1_episode_outcome(
                    p1_hp=p1_hp,
                    p2_hp=p2_hp,
                    p1_deck=p1_deck,
                    p2_deck=p2_deck,
                    terminated=terminated,
                    truncated=truncated,
                )
            )
            p1_ep_rewards.append(cur_p1_r)
            p2_ep_rewards.append(cur_p2_r)
            completed += 1
            if after_episode is not None:
                try:
                    after_episode(completed)
                except Exception as exc:
                    print(f"  [train] after_episode callback failed ({exc!r})")
            if completed == warmup_episodes and not warmup_bc_applied:
                print(
                    f"  [warmup] behavioural-cloning update from "
                    f"{len(warmup_p1_trans) + len(warmup_p2_trans)} transitions"
                )
                _flush_warmup_buffers_auto(p1_tiers, p2_tiers, warmup_p1_trans, warmup_p2_trans)
                warmup_p1_trans.clear()
                warmup_p2_trans.clear()
                warmup_bc_applied = True
            if truncated:
                timeout_episodes += 1
            if terminated:
                terminated_episodes += 1
                # Cache this completed episode for future warm-starts.
                if episode_cache is not None and p1_deck and p2_deck:
                    try:
                        episode_cache.add_episode(
                            p1_deck, p2_deck,
                            obs_dim=_obs_dim,
                            p1_transitions=list(_ep_p1_trans),
                            p2_transitions=list(_ep_p2_trans),
                            p1_reward=cur_p1_r,
                            p2_reward=cur_p2_r,
                            steps=episode_step,
                            warmup=in_warmup,
                        )
                    except Exception:
                        pass

            # Reset per-episode accumulators regardless of outcome.
            _ep_p1_trans = []
            _ep_p2_trans = []

            if (
                not suppress_train_progress
                and (
                completed <= 10  # dense startup visibility
                or completed == n_episodes
                or completed % progress_every == 0
                )
            ):
                elapsed = time.time() - progress_t0
                pct = (completed / max(1, n_episodes)) * 100.0
                p1_avg = float(np.mean(p1_ep_rewards)) if p1_ep_rewards else 0.0
                p2_avg = float(np.mean(p2_ep_rewards)) if p2_ep_rewards else 0.0
                ep_rate = completed / max(elapsed, 1e-9)
                eta_secs = (n_episodes - completed) / ep_rate if ep_rate > 0 else float("inf")
                timeout_rate = timeout_episodes / max(1, completed)
                print(
                    f"  [train-progress] episodes={completed}/{n_episodes} "
                    f"({pct:6.2f}%) elapsed={elapsed:.1f}s "
                    f"rate={ep_rate:.3f}ep/s eta={eta_secs/60:.1f}m "
                    f"warmup={'yes' if completed < warmup_episodes else 'no '} "
                    f"timeouts={timeout_episodes} ({timeout_rate * 100:.1f}%) "
                    f"p1_avg={p1_avg:+.3f} p2_avg={p2_avg:+.3f}"
                )

            cur_p1_r = cur_p2_r = 0.0
            episode_step = 0
            step_order = 0
            ep_seed = (seed + completed) if seed is not None else None
            reset_out = env.reset(seed=ep_seed)
            obs = _get(reset_out, "observation", reset_out)
            step_info = _get(reset_out, "info", {})
        else:
            obs = _get(step_out, "observation", obs)

        if not in_warmup:
            if _uses_unified_policy(p1_tiers, p2_tiers):
                if len(unified_buf["obs"]) >= DEFAULT_PPO_ROLLOUT_BATCH:
                    _ppo_update(p1_tiers[0], unified_buf, next_obs_vec)
                    unified_buf.clear()
                    unified_buf.update(_empty_buf())
            else:
                for tiers, buf_ref in [(p1_tiers, p1_buf), (p2_tiers, p2_buf)]:
                    if len(buf_ref["obs"]) >= DEFAULT_PPO_ROLLOUT_BATCH:
                        _ppo_update_all_tiers(tiers, buf_ref, next_obs_vec)
                        buf_ref.clear()
                        buf_ref.update(_empty_buf())

    if _uses_unified_policy(p1_tiers, p2_tiers):
        if len(unified_buf["obs"]) > 0:
            next_vec = p1_tiers[0]._obs_to_vec(obs)
            _ppo_update(p1_tiers[0], unified_buf, next_vec)
    else:
        for tiers, buf_ref in [(p1_tiers, p1_buf), (p2_tiers, p2_buf)]:
            if len(buf_ref["obs"]) > 0:
                next_vec = tiers[0]._obs_to_vec(obs)
                _ppo_update_all_tiers(tiers, buf_ref, next_vec)

    total_eps = len(p1_ep_rewards)
    outcome_summary = summarize_p1_outcomes(p1_outcomes, episodes=total_eps)
    stats = {
        "episodes": total_eps,
        "timeouts": outcome_summary["timeouts"],
        "terminated": terminated_episodes,
        "timeout_rate": outcome_summary["timeout_rate"],
        "p1_outcomes": p1_outcomes,
        **outcome_summary,
    }
    return p1_ep_rewards, p2_ep_rewards, stats


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
    training_stats: Optional[dict[str, Any]] = None,
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
                "training_stats": training_stats,
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
                "training_stats": training_stats,
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
        "training_stats": training_stats,
    }


class _CheckpointEvalTracker:
    """Periodic head-to-head eval snapshots during unified self-play training."""

    def __init__(
        self,
        *,
        matchup: Matchup,
        base_url: str,
        game_format: str,
        max_steps: int,
        n_episodes: int,
        checkpoint_interval: int,
        checkpoint_eval_episodes: int,
        p1_policy: PPOAgent,
        p2_policy: PPOAgent,
        seed: Optional[int],
        out_dir: Path,
    ) -> None:
        self.matchup = matchup
        self.base_url = base_url
        self.game_format = game_format
        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.checkpoint_interval = max(1, int(checkpoint_interval))
        self.checkpoint_eval_episodes = int(checkpoint_eval_episodes)
        self.p1_policy = p1_policy
        self.p2_policy = p2_policy
        self.seed = seed
        self.out_dir = out_dir
        self.log: list[dict[str, Any]] = []
        self.first_win_rate: Optional[float] = None
        self.final_win_rate: Optional[float] = None
        self._episodes_done = 0

    def on_episode(self, _local_completed: int = 0) -> None:
        self._episodes_done += 1
        self._maybe_eval(self._episodes_done)

    def on_parallel_progress(
        self,
        completed: int,
        *_args: Any,
    ) -> None:
        self._maybe_eval(int(completed))

    def _maybe_eval(self, completed: int) -> None:
        if self.checkpoint_eval_episodes <= 0:
            return
        if completed % self.checkpoint_interval != 0 and completed != self.n_episodes:
            return
        from agent_cache import clone_agent_weights  # noqa: PLC0415

        eval_p1 = PPOAgent()
        eval_p2 = PPOAgent()
        clone_agent_weights(self.p1_policy, eval_p1)
        clone_agent_weights(self.p2_policy, eval_p2)
        from train_play import _evaluate_p1_vs_fixed_opponent  # noqa: PLC0415

        metrics = _evaluate_p1_vs_fixed_opponent(
            self.matchup,
            eval_p1,
            p2_agent=eval_p2,
            base_url=self.base_url,
            game_format=self.game_format,
            max_steps=self.max_steps,
            episodes=self.checkpoint_eval_episodes,
            seed=(self.seed + completed) if self.seed is not None else None,
            backend="cpp" if self.matchup.cpp_engine_dir else "auto",
            eval_label="Checkpoint eval",
        )
        record = {
            "episodes_completed": completed,
            "target_episodes": self.n_episodes,
            "eval_episodes": self.checkpoint_eval_episodes,
            **metrics,
        }
        self.log.append(record)
        wr = float(metrics["p1_win_rate"])
        if self.first_win_rate is None:
            self.first_win_rate = wr
        self.final_win_rate = wr
        ckpt_dir = (
            self.out_dir
            / self.matchup.name
            / "checkpoint_eval"
            / f"episode_{completed:06d}"
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "checkpoint_eval.json").write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )
        history_path = self.out_dir / "checkpoint_eval_history.json"
        history_path.write_text(json.dumps(self.log, indent=2), encoding="utf-8")
        print(
            f"  Checkpoint eval @ ep {completed}: "
            f"P1 win%={wr:.1%} "
            f"({metrics['p1_wins']}W/{metrics['p2_wins']}L/{metrics['draws']}D "
            f"over {self.checkpoint_eval_episodes} games)"
        )


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


def _train_matchup_parallel_seeds(
    matchup: Matchup,
    *,
    parallel_seeds: int,
    base_url: str,
    n_episodes: int,
    max_steps: int,
    out_dir: Path,
    eval_env_ids: dict[str, str],
    cache_store: Optional["AgentCacheStore"],
    seed: Optional[int],
    game_format: str,
    warmup_episodes: int,
    warmup_baseline_eval_episodes: int,
    show_frontend: bool,
    frontend_url: Optional[str],
    n_workers: int,
    build_cpp_engine: bool = True,
    require_cpp_engine: bool = False,
) -> dict:
    workers_per_seed = workers_per_parallel_seed(n_workers, parallel_seeds)
    if workers_per_seed != n_workers:
        print(
            f"  Parallel seeds: {parallel_seeds} × {workers_per_seed} worker(s)/seed "
            f"(total rollout budget {n_workers})"
        )

    shared_kwargs = dict(
        matchup=matchup,
        base_url=base_url,
        n_episodes=n_episodes,
        max_steps=max_steps,
        eval_env_ids=eval_env_ids,
        cache_store=cache_store,
        game_format=game_format,
        warmup_episodes=warmup_episodes,
        warmup_baseline_eval_episodes=0,
        show_frontend=show_frontend,
        frontend_url=frontend_url,
        n_workers=workers_per_seed,
        parallel_seeds=1,
        _skip_cache_converge=True,
        _force_train=True,
        build_cpp_engine=build_cpp_engine,
        require_cpp_engine=require_cpp_engine,
    )

    def _run_one_seed(
        seed_index: int,
        seed_i: Optional[int],
        seed_out: Path,
    ) -> dict[str, Any]:
        capture: dict[str, Any] = {}
        meta = train_matchup(
            out_dir=seed_out,
            seed=seed_i,
            _seed_run_capture=capture,
            _force_train=True,
            **shared_kwargs,
        )
        return {
            **capture,
            "seed_index": seed_index,
            "seed": seed_i,
            "out_dir": str(seed_out),
            "meta": meta,
            "p1_win_rate": float(capture.get("p1_win_rate", 0.0)),
            "p2_win_rate": float(capture.get("p2_win_rate", 0.0)),
        }

    summary = run_parallel_seed_jobs(
        parallel_seeds,
        seed,
        out_dir,
        _run_one_seed,
        label=f"matchup {matchup.name}",
    )
    best_p1, best_p2, best_p1_idx, best_p2_idx = select_best_agents_by_win_rate(
        summary.seed_rows
    )

    warmup_baseline: Optional[dict[str, Any]] = None
    if warmup_baseline_eval_episodes > 0:
        baseline = _evaluate_policy_pair(
            matchup,
            base_url=base_url,
            game_format=game_format,
            max_steps=max_steps,
            p1_policy=best_p1,
            p2_policy=best_p2,
            episodes=warmup_baseline_eval_episodes,
            seed=(seed + 100_000) if seed is not None else None,
        )
        ckpt_dir = _save_warmup_handoff_checkpoint(
            out_dir=out_dir,
            matchup=matchup,
            p1_policy=best_p1,
            p2_policy=best_p2,
            baseline={
                **baseline,
                "parallel_seeds": parallel_seeds,
                "best_p1_seed_index": best_p1_idx,
                "best_p2_seed_index": best_p2_idx,
                "avg_train_p1_win_rate": summary.avg_p1_win_rate,
                "avg_train_p2_win_rate": summary.avg_p2_win_rate,
            },
        )
        warmup_baseline = {**baseline, "checkpoint_dir": str(ckpt_dir)}
        print(
            "  Combined eval (best seeds): "
            f"P1 win%={baseline['p1_win_rate'] * 100:.1f} "
            f"P2 win%={baseline['p2_win_rate'] * 100:.1f} "
            f"draw%={baseline['draw_rate'] * 100:.1f}"
        )

    best_p1_row = summary.seed_rows[best_p1_idx]
    best_meta = best_p1_row.get("meta") or {}
    p1_id = f"{matchup.name}-p1-{uuid.uuid4().hex[:8]}"
    p2_id = f"{matchup.name}-p2-{uuid.uuid4().hex[:8]}"
    elapsed = float(best_meta.get("p1", {}).get("elapsed_secs", 0.0) or 0.0)

    meta_p1 = save_agent(
        best_p1, p1_id, out_dir, matchup, [],
        n_episodes, elapsed, game_format, "p1", eval_env_ids,
        warmup_baseline=warmup_baseline,
        training_stats={
            "parallel_seeds": parallel_seeds,
            "avg_p1_win_rate": summary.avg_p1_win_rate,
            "best_p1_seed_index": best_p1_idx,
        },
    )
    meta_p2 = save_agent(
        best_p2, p2_id, out_dir, matchup, [],
        n_episodes, elapsed, game_format, "p2", eval_env_ids,
        warmup_baseline=warmup_baseline,
        training_stats={
            "parallel_seeds": parallel_seeds,
            "avg_p2_win_rate": summary.avg_p2_win_rate,
            "best_p2_seed_index": best_p2_idx,
        },
    )
    meta_p1["training_win_rate"] = summary.avg_p1_win_rate
    meta_p2["training_win_rate"] = summary.avg_p2_win_rate
    meta_p1["warmup_baseline"] = warmup_baseline
    meta_p2["warmup_baseline"] = warmup_baseline

    if cache_store is not None:
        from agent_cache import talishar_asset_deck_fingerprint  # noqa: PLC0415

        assets = talishar_assets_path()
        p1_deck_fp = talishar_asset_deck_fingerprint(str(assets), matchup.p1_deck)
        p2_deck_fp = talishar_asset_deck_fingerprint(str(assets), matchup.p2_deck)
        cache_store.persist(
            best_p1,
            training_summary={
                "matchup_name": matchup.name,
                "p1_fingerprint": p1_deck_fp,
                "p2_fingerprint": p2_deck_fp,
                "p1_hero": matchup.p1_hero,
                "p2_hero": matchup.p2_hero,
                "episodes_completed": n_episodes,
                "target_episodes": n_episodes,
                "p1_win_rate": summary.avg_p1_win_rate,
                "p2_win_rate": summary.avg_p2_win_rate,
            },
        )

    print(
        f"\n  Parallel-seed train win% (avg over {parallel_seeds}): "
        f"P1={summary.avg_p1_win_rate:.1%}  P2={summary.avg_p2_win_rate:.1%}"
    )
    return {"p1": meta_p1, "p2": meta_p2}


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
    show_frontend: bool = False,
    frontend_url: Optional[str] = None,
    n_workers: int = 1,
    parallel_seeds: int = 1,
    checkpoint_interval_pct: float = DEFAULT_CHECKPOINT_INTERVAL_PCT,
    checkpoint_interval: Optional[int] = None,
    checkpoint_eval_episodes: int = DEFAULT_CHECKPOINT_EVAL_EPISODES,
    build_cpp_engine: bool = True,
    require_cpp_engine: bool = False,
    _seed_run_capture: Optional[dict[str, Any]] = None,
    _skip_cache_converge: bool = False,
    _force_train: bool = False,
) -> dict:
    ensure_matchup_cpp_engine(
        matchup,
        base_url=base_url,
        build=build_cpp_engine,
        require=require_cpp_engine,
    )

    if parallel_seeds > 1 and _seed_run_capture is None:
        return _train_matchup_parallel_seeds(
            matchup,
            parallel_seeds=parallel_seeds,
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
            show_frontend=show_frontend,
            frontend_url=frontend_url,
            n_workers=n_workers,
            build_cpp_engine=build_cpp_engine,
            require_cpp_engine=require_cpp_engine,
        )
    from agent_cache import AgentCacheStore, talishar_asset_deck_fingerprint
    print(f"\n{'=' * 60}")
    print(f"  Matchup : {matchup.name}")
    print(f"  Decks   : {matchup.p1_deck} (P1) vs {matchup.p2_deck} (P2)")
    print(
        f"  Mode    : both-perspectives training | {n_episodes} episodes | "
        f"max {max_steps} steps | warmup {warmup_episodes} episodes"
        + (f" | workers={n_workers}" if n_workers > 1 else "")
    )
    if show_frontend:
        print("  Live state image rendering: enabled (no browser tabs)")
    print(f"{'=' * 60}")

    if matchup.cpp_engine_dir:
        print(f"  C++ engine : {matchup.cpp_engine_dir}")

    if cache_store is None:
        cache_store = AgentCacheStore(
            REPO_ROOT / "results" / "agent_cache",
            game_format,
            obs_schema_version=PLAYER_OBS_SCHEMA_VERSION,
        )

    assets = talishar_assets_path()
    p1_deck_fp = talishar_asset_deck_fingerprint(str(assets), matchup.p1_deck)
    p2_deck_fp = talishar_asset_deck_fingerprint(str(assets), matchup.p2_deck)

    cached_record = cache_store.should_skip_training(
        p1_fingerprint=p1_deck_fp,
        p2_fingerprint=p2_deck_fp,
        target_episodes=n_episodes,
    )
    if cached_record is not None and not _force_train:
        cached_policy = cache_store.load_if_exists()
        if cached_policy is not None:
            print(
                f"  Cache hit — converged deck-vs-deck matchup "
                f"({cached_record.episodes_completed}/{cached_record.target_episodes} ep) "
                f"— skipping training (unified agent v{PLAYER_OBS_SCHEMA_VERSION})"
            )
            return {
                "p1": {
                    "matchup": matchup.name,
                    "cached": True,
                    "avg_reward": 0.0,
                    "best_reward": 0.0,
                    "elapsed_secs": 0.0,
                    "agent_id": "cached_unified",
                    "package_dir": str(out_dir / matchup.name / "cached_unified"),
                },
                "p2": {
                    "matchup": matchup.name,
                    "cached": True,
                    "avg_reward": 0.0,
                    "best_reward": 0.0,
                    "elapsed_secs": 0.0,
                    "agent_id": "cached_unified",
                    "package_dir": str(out_dir / matchup.name / "cached_unified"),
                },
                "skipped_training": True,
            }

    # Create a persistent episode cache for this game format.  Completed
    # (non-truncated) episodes are stored per deck matchup and replayed as PPO
    # warm-start data on future runs.  The cache root sits alongside the agent
    # cache so both can be shared across training scripts.
    episode_cache = EpisodeCache(
        cache_root=cache_store.user_cache_root,
        game_format=game_format,
    )
    _ep_cache_info = episode_cache.info(matchup.p1_deck, matchup.p2_deck)
    print(
        f"  Episode cache: {_ep_cache_info['total_episodes']} stored episode(s) for this matchup "
        f"(skip threshold: {episode_cache.warmup_skip_threshold})"
    )

    # Probe environment once to bootstrap the unified shared policy.
    probe_env = None
    for attempt in range(2):
        probe_env = make_env(
            matchup,
            base_url=base_url,
            game_format=game_format,
            max_turns=max_steps,
            show_frontend=False,
            frontend_url=frontend_url,
        )
        try:
            if _env_supports_fast_training(probe_env):
                probe_env.fast_reset(seed=seed)
            else:
                probe_env.reset(seed=seed)
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

    unified_bundle = _bootstrap_unified_policy(cache_store, probe_env, seed)
    p1_bundle = p2_bundle = unified_bundle
    print("  Unified policy init:", ", ".join(unified_bundle.init_sources))
    print(
        f"  obs_schema={PLAYER_OBS_SCHEMA_VERSION} "
        f"obs_dim={unified_bundle.policy.obs_dim} "
        f"n_actions={unified_bundle.policy.n_actions}"
    )

    if n_workers <= 1 and probe_env is not None:
        env = probe_env
    elif probe_env is not None:
        try:
            probe_env.close()
        except Exception:
            pass
        env = None
    else:
        env = None

    t0 = time.time()
    warmup_baseline: Optional[dict[str, Any]] = None
    warmup_count = max(0, min(warmup_episodes, n_episodes))
    p1_rewards: list[float] = []
    p2_rewards: list[float] = []
    overall_stats: dict[str, Any] = {
        "episodes": 0,
        "timeouts": 0,
        "terminated": 0,
        "timeout_rate": 0.0,
    }
    p1_outcomes: list[str] = []
    live_state_image_path: Optional[Path] = None
    if show_frontend:
        live_state_image_path = out_dir / matchup.name / "training_live_state.png"
        live_state_image_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Live state image path → {live_state_image_path}")

    ckpt_tracker: Optional[_CheckpointEvalTracker] = None
    effective_ckpt_interval = resolve_checkpoint_interval(
        n_episodes,
        checkpoint_interval=checkpoint_interval,
        checkpoint_interval_pct=checkpoint_interval_pct,
    )
    if checkpoint_eval_episodes > 0:
        ckpt_tracker = _CheckpointEvalTracker(
            matchup=matchup,
            base_url=base_url,
            game_format=game_format,
            max_steps=max_steps,
            n_episodes=n_episodes,
            checkpoint_interval=effective_ckpt_interval,
            checkpoint_eval_episodes=checkpoint_eval_episodes,
            p1_policy=p1_bundle.policy,
            p2_policy=p2_bundle.policy,
            seed=seed,
            out_dir=out_dir,
        )
        print(
            f"  Checkpoint eval: every {effective_ckpt_interval} episode(s), "
            f"{checkpoint_eval_episodes} eval game(s) per checkpoint"
        )

    # ── parallel path ─────────────────────────────────────────────────────────
    if n_workers > 1:
        print(f"  [parallel] Using {n_workers} worker game sessions for training")
        rem_p1, rem_p2, rem_stats = train_agents_from_both_perspectives_parallel(
            matchup=matchup,
            base_url=base_url,
            game_format=game_format,
            p1_tiers=p1_bundle.agents,
            p2_tiers=p2_bundle.agents,
            n_episodes=n_episodes,
            max_steps=max_steps,
            seed=seed,
            warmup_episodes=warmup_count,
            n_workers=n_workers,
            live_state_image_path=live_state_image_path,
            episode_cache=episode_cache,
            on_episodes_progress=(
                ckpt_tracker.on_parallel_progress if ckpt_tracker is not None else None
            ),
        )
        p1_rewards.extend(rem_p1)
        p2_rewards.extend(rem_p2)
        overall_stats["episodes"]   += int(rem_stats.get("episodes", 0))
        overall_stats["timeouts"]   += int(rem_stats.get("timeouts", 0))
        overall_stats["terminated"] += int(rem_stats.get("terminated", 0))
        p1_outcomes.extend(rem_stats.get("p1_outcomes") or [])

        # Run warmup-baseline eval so we still get a checkpoint/baseline record.
        if warmup_count > 0 and warmup_baseline_eval_episodes > 0:
            print(
                f"  Warmup baseline eval: {warmup_baseline_eval_episodes} episode(s)"
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
            warmup_baseline = {**baseline, "checkpoint_dir": str(ckpt_dir)}

    else:
        # ── serial path (reuse probed env when available) ───────────────────
        if env is None:
            for attempt in range(2):
                env = make_env(
                    matchup,
                    base_url=base_url,
                    game_format=game_format,
                    max_turns=max_steps,
                    show_frontend=show_frontend,
                    frontend_url=frontend_url,
                )
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

        if warmup_count > 0:
            print(f"  Warmup  : training first {warmup_count} episode(s) with Talishar default policy")
            warm_p1, warm_p2, warm_stats = train_agents_from_both_perspectives(
                env,
                p1_bundle.agents,
                p2_bundle.agents,
                n_episodes=warmup_count,
                max_steps=max_steps,
                seed=seed,
                warmup_episodes=warmup_count,
                live_state_image_path=live_state_image_path,
                episode_cache=episode_cache,
                p1_deck=matchup.p1_deck,
                p2_deck=matchup.p2_deck,
                after_episode=ckpt_tracker.on_episode if ckpt_tracker is not None else None,
            )
            p1_rewards.extend(warm_p1)
            p2_rewards.extend(warm_p2)
            overall_stats["episodes"]   += int(warm_stats.get("episodes", 0))
            overall_stats["timeouts"]   += int(warm_stats.get("timeouts", 0))
            overall_stats["terminated"] += int(warm_stats.get("terminated", 0))
            p1_outcomes.extend(warm_stats.get("p1_outcomes") or [])

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
            warmup_baseline = {**baseline, "checkpoint_dir": str(ckpt_dir)}
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
            rem_p1, rem_p2, rem_stats = train_agents_from_both_perspectives(
                env,
                p1_bundle.agents,
                p2_bundle.agents,
                n_episodes=remaining_episodes,
                max_steps=max_steps,
                seed=rem_seed,
                warmup_episodes=0,
                live_state_image_path=live_state_image_path,
                episode_cache=episode_cache,
                p1_deck=matchup.p1_deck,
                p2_deck=matchup.p2_deck,
                after_episode=ckpt_tracker.on_episode if ckpt_tracker is not None else None,
            )
            p1_rewards.extend(rem_p1)
            p2_rewards.extend(rem_p2)
            overall_stats["episodes"]   += int(rem_stats.get("episodes", 0))
            overall_stats["timeouts"]   += int(rem_stats.get("timeouts", 0))
            overall_stats["terminated"] += int(rem_stats.get("terminated", 0))
            p1_outcomes.extend(rem_stats.get("p1_outcomes") or [])

        env.close()

    overall_stats["timeout_rate"] = overall_stats["timeouts"] / max(
        1, int(overall_stats["episodes"])
    )
    outcome_summary = summarize_p1_outcomes(
        p1_outcomes,
        episodes=len(p1_rewards) if p1_rewards else None,
    )
    overall_stats.update(outcome_summary)

    elapsed = time.time() - t0

    p1_wr = float(np.mean([1.0 if r > 0 else 0.0 for r in p1_rewards])) if p1_rewards else None
    p2_wr = float(np.mean([1.0 if r > 0 else 0.0 for r in p2_rewards])) if p2_rewards else None

    training_stats = dict(overall_stats)
    if ckpt_tracker is not None and ckpt_tracker.log:
        training_stats["checkpoint_eval_history"] = ckpt_tracker.log

    cache_store.persist(
        unified_bundle.policy,
        episodes_delta=len(p1_rewards),
        training_summary=(
            {
                "matchup_name": matchup.name,
                "p1_fingerprint": p1_deck_fp,
                "p2_fingerprint": p2_deck_fp,
                "p1_hero": matchup.p1_hero,
                "p2_hero": matchup.p2_hero,
                "episodes_completed": len(p1_rewards),
                "target_episodes": n_episodes,
                "p1_win_rate": p1_wr,
                "p2_win_rate": p2_wr,
                "first_checkpoint_win_rate": ckpt_tracker.first_win_rate
                if ckpt_tracker is not None
                else None,
                "final_checkpoint_win_rate": ckpt_tracker.final_win_rate
                if ckpt_tracker is not None
                else None,
                "checkpoint_eval_win_rate": ckpt_tracker.final_win_rate
                if ckpt_tracker is not None
                else None,
                "training_stats": training_stats,
            }
            if len(p1_rewards) >= n_episodes and not _skip_cache_converge
            else None
        ),
    )

    print(
        f"  Done in {elapsed:.1f}s — "
        f"P1 avg={np.mean(p1_rewards):+.3f}  "
        f"P2 avg={np.mean(p2_rewards):+.3f}  "
        f"timeouts={overall_stats['timeouts']}/{overall_stats['episodes']} "
        f"({overall_stats['timeout_rate'] * 100:.1f}%)"
    )
    if p1_wr is not None and p2_wr is not None and _seed_run_capture is None:
        print(f"  Train win%: P1={p1_wr * 100:.1f}  P2={p2_wr * 100:.1f}")

    if _seed_run_capture is not None:
        _seed_run_capture.update(
            {
                "p1_agent": p1_bundle.policy,
                "p2_agent": p2_bundle.policy,
                "p1_win_rate": float(p1_wr or 0.0),
                "p2_win_rate": float(p2_wr or 0.0),
            }
        )

    p1_id = f"{matchup.name}-p1-{uuid.uuid4().hex[:8]}"
    p2_id = f"{matchup.name}-p2-{uuid.uuid4().hex[:8]}"

    meta_p1 = save_agent(
        p1_bundle.policy, p1_id, out_dir, matchup, p1_rewards,
        n_episodes, elapsed, game_format, "p1", eval_env_ids,
        warmup_baseline=warmup_baseline,
        training_stats=overall_stats,
    )
    meta_p2 = save_agent(
        p2_bundle.policy, p2_id, out_dir, matchup, p2_rewards,
        n_episodes, elapsed, game_format, "p2", eval_env_ids,
        warmup_baseline=warmup_baseline,
        training_stats=overall_stats,
    )
    if p1_wr is not None:
        meta_p1["training_win_rate"] = p1_wr
    if p2_wr is not None:
        meta_p2["training_win_rate"] = p2_wr
    return {"p1": meta_p1, "p2": meta_p2}


def print_training_summary(summary: list[dict], failed: list[str], out_dir: Path) -> None:
    print(f"\n{'=' * 60}")
    print("  TRAINING SUMMARY")
    print(f"{'=' * 60}")
    for m in summary:
        baseline = m.get("p1", {}).get("warmup_baseline")
        for role in ("p1", "p2"):
            r = m[role]
            tstats = r.get("training_stats") or {}
            timeout_line = ""
            if tstats:
                timeout_line = (
                    f"\n  {'timeouts':<28} {tstats.get('timeouts', 0)}/"
                    f"{tstats.get('episodes', 0)} ({tstats.get('timeout_rate', 0.0) * 100:.1f}%)"
                )
            print(
                f"  {r['matchup']}/{role:<4}  avg={r['avg_reward']:+.3f}  "
                f"best={r['best_reward']:+.3f}  ({r['elapsed_secs']:.0f}s)\n"
                f"  {'agent_id':<28} {r['agent_id']}\n"
                f"  {'package_dir':<28} {r['package_dir']}"
                f"{timeout_line}"
            )
            train_wr = r.get("training_win_rate")
            if train_wr is not None:
                print(f"  {'train win%':<28} {float(train_wr) * 100:.1f}")
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
    show_frontend: bool = False,
    frontend_url: Optional[str] = None,
    n_workers: int = 1,
    parallel_seeds: int = 1,
    checkpoint_interval_pct: float = DEFAULT_CHECKPOINT_INTERVAL_PCT,
    checkpoint_interval: Optional[int] = None,
    checkpoint_eval_episodes: int = DEFAULT_CHECKPOINT_EVAL_EPISODES,
    skip_converged: bool = True,
    build_cpp_engine: bool = True,
    require_cpp_engine: bool = False,
) -> tuple[list[dict], list[str]]:
    from agent_cache import AgentCacheStore

    cache_root = cache_dir or (REPO_ROOT / "results" / "agent_cache")
    cache_store = AgentCacheStore(
        cache_root,
        game_format,
        obs_schema_version=PLAYER_OBS_SCHEMA_VERSION,
    )

    summary: list[dict] = []
    failed: list[str] = []
    for matchup in matchups:
        try:
            train_kwargs: dict[str, Any] = {}
            if not skip_converged:
                train_kwargs["_force_train"] = True
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
                show_frontend=show_frontend,
                frontend_url=frontend_url,
                n_workers=n_workers,
                parallel_seeds=parallel_seeds,
                checkpoint_interval_pct=checkpoint_interval_pct,
                checkpoint_interval=checkpoint_interval,
                checkpoint_eval_episodes=checkpoint_eval_episodes,
                build_cpp_engine=build_cpp_engine,
                require_cpp_engine=require_cpp_engine,
                **train_kwargs,
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


def ensure_matchup_cpp_engine(
    matchup: Matchup,
    *,
    base_url: str,
    build: bool = True,
    require: bool = False,
) -> Optional[str]:
    """Discover or build the C++ engine for *matchup* and attach it to the matchup."""
    if matchup.cpp_engine_dir:
        return matchup.cpp_engine_dir

    from cpp_engine_matchup import ensure_cpp_engine_for_matchup  # noqa: PLC0415

    cpp_dir = ensure_cpp_engine_for_matchup(
        matchup.p1_deck,
        matchup.p2_deck,
        assets_path=str(talishar_assets_path()),
        talishar_url=base_url,
        build=build,
    )
    if cpp_dir:
        matchup.cpp_engine_dir = cpp_dir
        return cpp_dir
    if require:
        raise RuntimeError(
            f"C++ engine required but unavailable for "
            f"{matchup.p1_deck} vs {matchup.p2_deck}"
        )
    return None


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
            lines = txt_file.read_text(encoding="utf-8").splitlines()
            if not lines:
                continue
            first_line = lines[0].strip()
            if not first_line:
                continue
            hero_id = first_line.split()[0]
            result[hero_id] = first_line
        except (OSError, IndexError):
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
            result.extend(assign_pitch_variants(candidates, count))
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


def build_fabrary_matchup(
    slug1: str,
    stem1: str,
    entry1: dict,
    slug2: str,
    stem2: str,
    entry2: dict,
    format_name: str,
) -> Matchup:
    return Matchup(
        name=f"{slug1}-vs-{slug2}",
        p1_deck=stem1,
        p2_deck=stem2,
        description=(
            f"{entry1.get('name', slug1)} (P1) vs "
            f"{entry2.get('name', slug2)} (P2) — unified self-play"
        ),
        tags=[slug1, slug2, format_name],
        p1_hero=hero_slug(str(entry1.get("hero_id", slug1))),
        p2_hero=hero_slug(str(entry2.get("hero_id", slug2))),
    )


def sample_random_fabrary_matchups(
    decks: list[tuple[str, str, dict]],
    count: int,
    rng: Any,
    format_name: str,
    *,
    unique_pairs: bool = True,
) -> list[Matchup]:
    """Sample random deck-vs-deck matchups from a fabrary deck pool."""
    if len(decks) < 2:
        raise ValueError("need at least two decks to sample matchups")
    if count <= 0:
        return []

    matchups: list[Matchup] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = max(count * 30, 30)
    indices = list(range(len(decks)))

    while len(matchups) < count and attempts < max_attempts:
        attempts += 1
        i, j = rng.sample(indices, 2)
        slug1, stem1, entry1 = decks[i]
        slug2, stem2, entry2 = decks[j]
        if stem1 == stem2:
            continue
        pair_key = (min(stem1, stem2), max(stem1, stem2))
        if unique_pairs and pair_key in seen:
            continue
        seen.add(pair_key)
        if rng.random() < 0.5:
            slug1, stem1, entry1, slug2, stem2, entry2 = (
                slug2,
                stem2,
                entry2,
                slug1,
                stem1,
                entry1,
            )
        matchups.append(
            build_fabrary_matchup(
                slug1,
                stem1,
                entry1,
                slug2,
                stem2,
                entry2,
                format_name,
            )
        )
    if len(matchups) < count:
        raise RuntimeError(
            f"could only sample {len(matchups)} unique matchup(s) from "
            f"{len(decks)} decks (requested {count})"
        )
    return matchups


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
