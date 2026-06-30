"""Shared dual-agent PPO self-play training utilities for Talishar scripts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
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
from fab_bridge.unified_dashboard import (  # noqa: E402
    LOGIC_VS_LOGIC_BASELINE_NAME,
    maybe_refresh_unified_dashboard,
    update_unified_training_live,
    write_unified_random_matchups_dashboard,
)
from fab_bridge.unified_results import is_unified_random_matchup_run  # noqa: E402
from play_outcome_stats import (  # noqa: E402
    OutcomeCounters,
    absolute_p1_p2_deck_from_env,
    absolute_p1_p2_hp_from_env,
    classify_p1_episode_outcome,
    summarize_hero_outcomes,
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
from flesh_and_blood_rlbridge.talishar_backend_pool import (  # noqa: E402
    TalisharBackendPool,
    is_shard_reset_error,
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
    DEFAULT_ROLLOUT_MODE,
    DEFAULT_ROLLOUT_PROCESSES,
    DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
    DEFAULT_WARMUP_EPISODES,
    DEFAULT_CHECKPOINT_INTERVAL_PCT,
    DEFAULT_CHECKPOINT_EVAL_EPISODES,
    DEFAULT_CHECKPOINT_EVAL_AGENT_VS_LOGIC,
    DEFAULT_CHECKPOINT_EVAL_LOGIC_VS_LOGIC,
    DEFAULT_TALISHAR_BACKEND,
    RUNTIME,
    engine_env_kwargs,
    game_env_kwargs,
    episode_timeout_seconds,
    envs_per_rollout_process,
    normalize_rollout_mode,
    resolve_rollout_processes,
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

# Keep nested training artifacts under Windows MAX_PATH (260).
_MAX_MATCHUP_DIR_LEN = 48
_MAX_WIN_PATH_LEN = 250


def shorten_matchup_dir_name(
    name: str,
    p1_deck: str,
    p2_deck: str,
    *,
    max_len: int = _MAX_MATCHUP_DIR_LEN,
) -> str:
    """Return a filesystem-safe matchup folder name (hash suffix when truncated)."""
    safe = re.sub(r"[^\w\-.]", "_", name)
    if len(safe) <= max_len:
        return safe
    digest = hashlib.sha1(f"{p1_deck}|{p2_deck}".encode()).hexdigest()[:10]
    keep = max(8, max_len - len(digest) - 1)
    return f"{safe[:keep]}_{digest}"


def new_agent_id(role: str) -> str:
    """Short unique agent id that stays within Windows path limits."""
    return f"{role}-{uuid.uuid4().hex[:8]}"


def matchup_out_dir(out_dir: Path, matchup: "Matchup") -> Path:
    return out_dir / _resolve_matchup_subdir(out_dir, matchup)


def _resolve_matchup_subdir(out_dir: Path, matchup: "Matchup") -> str:
    if matchup.dir_name:
        candidate = matchup.dir_name
    else:
        candidate = shorten_matchup_dir_name(
            matchup.name,
            matchup.p1_deck,
            matchup.p2_deck,
        )
    probe = (
        out_dir.resolve()
        / candidate
        / "ppo_p1-xxxxxxxx"
        / "weights"
        / "agent_weights.json"
    )
    if len(str(probe)) <= _MAX_WIN_PATH_LEN:
        return candidate
    excess = len(str(probe)) - _MAX_WIN_PATH_LEN
    tighter = max(16, len(candidate) - excess - 4)
    return shorten_matchup_dir_name(
        matchup.name,
        matchup.p1_deck,
        matchup.p2_deck,
        max_len=tighter,
    )


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
    dir_name: str = ""
    # Optional override deck names for C++ engine cache lookup.
    # Set these to the original hero/asset IDs when p1_deck/p2_deck are
    # UUID-based (Phase 3) but the compiled engine was built for the hero IDs.
    cpp_engine_deck1: Optional[str] = None
    cpp_engine_deck2: Optional[str] = None
    # Explicit engine directory — bypasses key/cache lookup entirely.
    cpp_engine_dir: Optional[str] = None
    # FaBrary deck entries for unified random matchup sideboarding.
    p1_fabrary_entry: Optional[dict[str, Any]] = None
    p2_fabrary_entry: Optional[dict[str, Any]] = None

    def output_subdir(self) -> str:
        if self.dir_name:
            return self.dir_name
        return shorten_matchup_dir_name(self.name, self.p1_deck, self.p2_deck)


def make_env(
    matchup: Matchup,
    base_url: str,
    game_format: str,
    max_turns: int,
    *,
    show_frontend: bool = False,
    frontend_url: Optional[str] = None,
    request_timeout: Optional[float] = None,
    use_cpp_engine: bool = False,
    talishar_backend: str = DEFAULT_TALISHAR_BACKEND,
    cpp_engine_cache_dir: Optional[str] = None,
    enable_combat_tracker: bool = False,
    require_fast_training: Optional[bool] = None,
    rl_training_mode: bool = True,
) -> TalisharEngineEnvironment:
    """Create a :class:`TalisharEngineEnvironment` for *matchup*.

    By default uses the Talishar **fast** backend (optimized HTTP + optional
    RLStep overlay) for full-rules training rollouts.  Pass ``use_cpp_engine=True``
    to prefer a compiled C++ stub engine when available.

    When ``require_fast_training`` is omitted, it defaults to ``True``.
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

    engine_kw = dict(engine_env_kwargs(RUNTIME.engine))
    game_kw = dict(game_env_kwargs(RUNTIME.game))
    if request_timeout is not None:
        engine_kw["request_timeout"] = request_timeout

    env = TalisharEngineEnvironment(
        base_url=base_url,
        frontend_url=resolved_frontend_url,
        local_deck_name=matchup.p1_deck,
        opponent_deck_name=matchup.p2_deck,
        game_format=game_format,
        self_play=True,
        max_turns=max_turns,
        render_mode=("rgb_array" if show_frontend else None),
        use_cpp_engine=use_cpp_engine,
        talishar_backend=talishar_backend,
        cpp_engine_cache_dir=effective_cache_dir,
        cpp_engine_deck1=matchup.cpp_engine_deck1,
        cpp_engine_deck2=matchup.cpp_engine_deck2,
        cpp_engine_dir=matchup.cpp_engine_dir,
        enable_combat_tracker=enable_combat_tracker,
        cpp_obs_alignment=False,
        rl_training_mode=rl_training_mode,
        rl_slim_response=rl_training_mode,
        **engine_kw,
        **game_kw,
    )
    if require_fast_training is None:
        require_fast_training = True
    if require_fast_training and not _env_supports_fast_training(env):
        reasons = "; ".join(_fast_training_unavailable_reasons(env)) or "unknown"
        raise RuntimeError(
            "Fast training is unavailable: "
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


def _sample_policy_action_index(
    policy: PPOAgent,
    obs_vec: np.ndarray,
    n_legal: int,
    rng: np.random.Generator,
) -> int:
    """Sample one legal action index using the same path as fast training rollouts."""
    if getattr(policy, "_shared", None) is None:
        raise RuntimeError("Policy has no initialized networks for eval sampling")
    nl = max(1, int(n_legal))
    logits, _values = policy.predict_batch(np.asarray(obs_vec, dtype=np.float64)[None, :])
    actions, _log_probs = _sample_actions_from_logits_batch(
        np.asarray(logits, dtype=np.float64),
        np.array([nl], dtype=np.int64),
        [rng],
        policy.n_actions,
    )
    return int(actions[0])


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
            "step_order": slot.step_order,
        })
        slot.step_order += 1
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
            "step_order": slot.step_order,
        })
        slot.step_order += 1
        slot.cur_p1_r -= 1.0


def _skipped_fast_episode_result(*, warmup: bool) -> dict[str, Any]:
    return {
        "p1_transitions": [],
        "p2_transitions": [],
        "p1_reward": 0.0,
        "p2_reward": 0.0,
        "terminated": False,
        "truncated": False,
        "warmup": warmup,
        "steps": 0,
        "p1_hp": None,
        "p2_hp": None,
        "turn_no": None,
        "p1_deck": None,
        "p2_deck": None,
    }


def _episode_result_is_skipped(result: dict[str, Any]) -> bool:
    return (
        not result.get("p1_transitions")
        and not result.get("p2_transitions")
        and not int(result.get("steps") or 0)
    )


def _rebind_worker_env(
    env: TalisharEngineEnvironment,
    swap_env: TalisharEngineEnvironment | None,
    *,
    matchup: Matchup,
    swap_matchup: Matchup,
    base_url: str,
    game_format: str,
    max_turns: int,
    worker_index: int | None = None,
    worker_base_urls: list[str] | None = None,
) -> str:
    """Point a worker env (and optional swapped deck env) at a new Talishar backend."""
    try:
        env.close()
    except Exception:
        pass
    if swap_env is not None:
        try:
            swap_env.close()
        except Exception:
            pass
    new_env = make_env(
        matchup,
        base_url=base_url,
        game_format=game_format,
        max_turns=max_turns,
    )
    env.__dict__.update(new_env.__dict__)
    if swap_env is not None:
        new_swap = make_env(
            swap_matchup,
            base_url=base_url,
            game_format=game_format,
            max_turns=max_turns,
        )
        swap_env.__dict__.update(new_swap.__dict__)
    if worker_base_urls is not None and worker_index is not None:
        worker_base_urls[worker_index] = base_url
    return base_url


def _retry_fast_episode_on_healthy_shards(
    env: TalisharEngineEnvironment,
    swap_env: TalisharEngineEnvironment | None,
    *,
    worker: int,
    p1_policy: PPOAgent,
    p2_policy: PPOAgent,
    max_steps: int,
    warmup: bool,
    episode_index: int,
    seed_base: Optional[int],
    rollout_mode: str,
    talishar_pool: TalisharBackendPool | None,
    matchup: Matchup | None,
    swap_matchup: Matchup | None,
    game_format: str,
    worker_base_urls: list[str] | None = None,
) -> dict[str, Any]:
    """Run one fast episode, rebind to another shard after connection failures."""
    max_attempts = max(1, len(talishar_pool.urls)) if talishar_pool else 1
    swap_slice = [swap_env] if swap_env is not None else None
    last_exc: BaseException | None = None

    for _attempt in range(max_attempts):
        try:
            result = _run_parallel_fast_episode_batch(
                [env],
                p1_policy,
                p2_policy,
                max_steps=max_steps,
                warmup=warmup,
                episode_indices=[episode_index],
                seed_base=seed_base,
                swap_envs=swap_slice,
                rollout_mode=rollout_mode,
                max_workers=1,
                matchup=matchup,
            )[0]
            if talishar_pool is not None:
                talishar_pool.note_shard_success(getattr(env, "_base_url", "") or "")
            return result
        except Exception as exc:
            last_exc = exc
            failed_url = getattr(env, "_base_url", "") or ""
            can_rebind = (
                talishar_pool is not None
                and matchup is not None
                and swap_matchup is not None
                and game_format
                and is_shard_reset_error(exc)
            )
            if can_rebind:
                talishar_pool.note_shard_failure(failed_url)
                replacement = talishar_pool.pick_replacement(
                    failed_url,
                    worker_index=worker,
                )
                if replacement and replacement != failed_url:
                    _rebind_worker_env(
                        env,
                        swap_env,
                        matchup=matchup,
                        swap_matchup=swap_matchup,
                        base_url=replacement,
                        game_format=game_format,
                        max_turns=max_steps,
                        worker_index=worker,
                        worker_base_urls=worker_base_urls,
                    )
                    continue
            break

    print(
        f"  [parallel] episode failed ({last_exc!r}) — will retry without counting",
        flush=True,
    )
    try:
        from fab_bridge.unified_training_debug import (  # noqa: PLC0415
            log_exception as unified_debug_exception,
            shard_label,
        )

        unified_debug_exception(
            "episode_failed",
            "Fast episode failed after shard retries",
            last_exc or RuntimeError("episode failed"),
            worker=worker,
            shard=shard_label(getattr(env, "_base_url", "") or ""),
            episode_index=episode_index,
        )
    except Exception:
        pass
    return _skipped_fast_episode_result(warmup=warmup)


def _safe_parallel_fast_episode_batch(
    envs: list[TalisharEngineEnvironment],
    p1_policy: PPOAgent,
    p2_policy: PPOAgent,
    *,
    max_steps: int,
    warmup: bool,
    episode_indices: list[int],
    seed_base: Optional[int],
    swap_envs: Optional[list[TalisharEngineEnvironment]] = None,
    rollout_mode: str = DEFAULT_ROLLOUT_MODE,
    max_workers: int = 1,
    talishar_pool: TalisharBackendPool | None = None,
    matchup: Matchup | None = None,
    swap_matchup: Matchup | None = None,
    game_format: str = "",
    worker_base_urls: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run a fast rollout batch; isolate failures to individual episodes."""
    batch_size = len(envs)
    try:
        results = _run_parallel_fast_episode_batch(
            envs,
            p1_policy,
            p2_policy,
            max_steps=max_steps,
            warmup=warmup,
            episode_indices=episode_indices,
            seed_base=seed_base,
            swap_envs=swap_envs,
            rollout_mode=rollout_mode,
            max_workers=max_workers,
            matchup=matchup,
        )
        if talishar_pool is not None:
            for env in envs[:batch_size]:
                talishar_pool.note_shard_success(getattr(env, "_base_url", "") or "")
        return results
    except Exception as exc:
        print(
            f"  [parallel] fast batch failed ({exc!r}) — "
            f"retrying {batch_size} episode(s) individually",
            flush=True,
        )
        try:
            from fab_bridge.unified_training_debug import (  # noqa: PLC0415
                log_exception as unified_debug_exception,
                shard_label,
            )

            shards = [
                shard_label(getattr(env, "_base_url", "") or "")
                for env in envs[:batch_size]
            ]
            unified_debug_exception(
                "episode_batch_failed",
                "Fast rollout batch failed",
                exc,
                batch_size=batch_size,
                shards=shards,
                episode_indices=episode_indices[:batch_size],
            )
        except Exception:
            pass

    results: list[dict[str, Any]] = []
    for worker in range(batch_size):
        swap_env = swap_envs[worker] if swap_envs is not None else None
        results.append(
            _retry_fast_episode_on_healthy_shards(
                envs[worker],
                swap_env,
                worker=worker,
                p1_policy=p1_policy,
                p2_policy=p2_policy,
                max_steps=max_steps,
                warmup=warmup,
                episode_index=episode_indices[worker],
                seed_base=(seed_base + worker) if seed_base is not None else None,
                rollout_mode=rollout_mode,
                talishar_pool=talishar_pool,
                matchup=matchup,
                swap_matchup=swap_matchup,
                game_format=game_format,
                worker_base_urls=worker_base_urls,
            )
        )
    return results


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
        "active_p1_hero": slot.active_p1_hero,
        "active_p2_hero": slot.active_p2_hero,
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
    step_order: int = 0
    active: bool = True
    terminated: bool = False
    truncated: bool = False
    final_p1_hp: int = 0
    final_p2_hp: int = 0
    final_p1_deck: int = 0
    final_p2_deck: int = 0
    final_turn_no: int = 0
    active_p1_hero: str = ""
    active_p2_hero: str = ""


class _EnvRolloutWorker:
    """Dedicated thread for one env slot so ``requests.Session`` stays thread-safe."""

    def __init__(self, slot: _FastRolloutSlot) -> None:
        self._slot = slot
        self._queue: queue.Queue[Any] = queue.Queue()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"env-rollout-{id(slot)}",
            daemon=True,
        )
        self._thread.start()

    def submit_step(
        self,
        *,
        action: int,
        value: float,
        log_prob: float,
        acting: int,
        max_steps: int,
    ) -> None:
        done = threading.Event()
        errors: list[BaseException] = []
        self._queue.put(
            (action, value, log_prob, acting, max_steps, done, errors)
        )
        done.wait()
        if errors:
            raise errors[0]

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            action, value, log_prob, acting, max_steps, done, errors = item
            try:
                _apply_fast_rollout_action(
                    self._slot,
                    action=action,
                    value=value,
                    log_prob=log_prob,
                    acting=acting,
                    max_steps=max_steps,
                )
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()

    def shutdown(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=60.0)


def _announce_rollout_config(
    *,
    rollout_mode: str,
    rollout_processes: int,
    n_workers: int,
    base_url: str = "",
    backend_pool: TalisharBackendPool | None = None,
) -> None:
    envs_per_proc = envs_per_rollout_process(n_workers, rollout_processes)
    talishar_label = (
        backend_pool.format_log_label()
        if backend_pool is not None
        else base_url
    )
    print(
        f"  [rollout] mode={rollout_mode}  "
        f"processes={rollout_processes} × envs={envs_per_proc} "
        f"(budget={n_workers})  Talishar={talishar_label}",
        flush=True,
    )


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
    slot.step_order = 0
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


def _batched_fast_rollout_step_concurrent(
    slots: list[_FastRolloutSlot],
    env_workers: list[_EnvRolloutWorker],
    p1_policy: PPOAgent,
    p2_policy: PPOAgent,
    *,
    warmup: bool,
    max_steps: int,
) -> None:
    """Batched policy inference, then concurrent RLStep calls (one thread per env slot)."""
    from concurrent.futures import ThreadPoolExecutor

    active_indices = [index for index, slot in enumerate(slots) if slot.active]
    if not active_indices:
        return

    if warmup:
        with ThreadPoolExecutor(max_workers=len(active_indices)) as pool:
            futures = []
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
                futures.append(
                    pool.submit(
                        env_workers[index].submit_step,
                        action=action,
                        value=0.0,
                        log_prob=0.0,
                        acting=acting,
                        max_steps=max_steps,
                    )
                )
            for fut in futures:
                fut.result()
        return

    p1_indices: list[int] = []
    p2_indices: list[int] = []
    for index in active_indices:
        acting = int(slots[index].state["acting_player_id"])
        if acting == 1:
            p1_indices.append(index)
        else:
            p2_indices.append(index)

    p1_actions = p1_log_probs = p1_values = None
    p2_actions = p2_log_probs = p2_values = None
    if p1_indices:
        p1_obs = np.stack([
            np.asarray(slots[index].state["obs_vec"], dtype=np.float64)
            for index in p1_indices
        ])
        p1_logits, p1_values = p1_policy.predict_batch(p1_obs)
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
        p2_obs = np.stack([
            np.asarray(slots[index].state["obs_vec"], dtype=np.float64)
            for index in p2_indices
        ])
        p2_logits, p2_values = p2_policy.predict_batch(p2_obs)
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
    step_jobs: list[tuple[int, int, float, float, int]] = []
    for index in active_indices:
        slot = slots[index]
        acting = int(slot.state["acting_player_id"])
        if acting == 1:
            action = int(p1_actions[p1_cursor])  # type: ignore[index]
            log_prob = float(p1_log_probs[p1_cursor])  # type: ignore[index]
            value = float(p1_values[p1_cursor])  # type: ignore[index]
            p1_cursor += 1
        else:
            action = int(p2_actions[p2_cursor])  # type: ignore[index]
            log_prob = float(p2_log_probs[p2_cursor])  # type: ignore[index]
            value = float(p2_values[p2_cursor])  # type: ignore[index]
            p2_cursor += 1
        step_jobs.append((index, action, value, log_prob, acting))

    with ThreadPoolExecutor(max_workers=len(step_jobs)) as pool:
        futures = [
            pool.submit(
                env_workers[index].submit_step,
                action=action,
                value=value,
                log_prob=log_prob,
                acting=acting,
                max_steps=max_steps,
            )
            for index, action, value, log_prob, acting in step_jobs
        ]
        for fut in futures:
            fut.result()


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
        "step_order": slot.step_order,
    }
    slot.step_order += 1
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
    swap_envs: Optional[list[TalisharEngineEnvironment]] = None,
    rollout_mode: str = DEFAULT_ROLLOUT_MODE,
    matchup: Optional["Matchup"] = None,
) -> list[dict[str, Any]]:
    """Run one episode per env slot with batched PPO inference across active slots."""
    mode = normalize_rollout_mode(rollout_mode)
    slots: list[_FastRolloutSlot] = []
    swap_matchup = swapped_matchup(matchup) if matchup is not None and swap_envs is not None else None
    for worker, env in enumerate(envs):
        ep_index = episode_indices[worker]
        ep_seed = (seed_base + worker) if seed_base is not None else None
        slot_env = env
        use_swap = swap_envs is not None and ep_index % 2 == 1
        if use_swap:
            slot_env = swap_envs[worker]
        active_p1_hero = active_p2_hero = ""
        if matchup is not None:
            active = swap_matchup if use_swap and swap_matchup is not None else matchup
            active_p1_hero = active.p1_hero
            active_p2_hero = active.p2_hero
        slot = _FastRolloutSlot(
            env=slot_env,
            state={},
            p1_rng=np.random.default_rng((ep_seed * 31 + 7) if ep_seed is not None else None),
            p2_rng=np.random.default_rng((ep_seed * 31 + 13) if ep_seed is not None else None),
            active_p1_hero=active_p1_hero,
            active_p2_hero=active_p2_hero,
        )
        _reset_fast_rollout_slot(
            slot,
            seed=ep_seed,
            starting_player_id=1 + (ep_index % 2),
        )
        slots.append(slot)

    env_workers: list[_EnvRolloutWorker] = []
    if mode == "batched_concurrent":
        env_workers = [_EnvRolloutWorker(slot) for slot in slots]

    try:
        for _ in range(max_steps):
            if not any(slot.active for slot in slots):
                break
            if mode == "batched_concurrent":
                _batched_fast_rollout_step_concurrent(
                    slots,
                    env_workers,
                    p1_policy,
                    p2_policy,
                    warmup=warmup,
                    max_steps=max_steps,
                )
            else:
                _batched_fast_rollout_step(
                    slots,
                    p1_policy,
                    p2_policy,
                    warmup=warmup,
                    max_steps=max_steps,
                )
    finally:
        for worker in env_workers:
            worker.shutdown()

    return [
        _fast_episode_result_from_slot(slot, warmup=warmup, max_steps=max_steps)
        for slot in slots
    ]


def _run_parallel_fast_episode_batch(
    envs: list[TalisharEngineEnvironment],
    p1_policy: PPOAgent,
    p2_policy: PPOAgent,
    *,
    max_steps: int,
    warmup: bool,
    episode_indices: list[int],
    seed_base: Optional[int],
    swap_envs: Optional[list[TalisharEngineEnvironment]] = None,
    rollout_mode: str = DEFAULT_ROLLOUT_MODE,
    max_workers: int = 1,
    matchup: Optional["Matchup"] = None,
) -> list[dict[str, Any]]:
    """Dispatch one rollout batch using the configured fast rollout mode."""
    mode = normalize_rollout_mode(rollout_mode)
    if mode == "threaded_episodes":
        return _run_parallel_fast_episodes_threaded(
            envs,
            p1_policy,
            p2_policy,
            max_steps=max_steps,
            warmup=warmup,
            episode_indices=episode_indices,
            seed_base=seed_base,
            swap_envs=swap_envs,
            max_workers=max(1, max_workers),
            matchup=matchup,
        )
    return _run_parallel_batched_fast_episodes(
        envs,
        p1_policy,
        p2_policy,
        max_steps=max_steps,
        warmup=warmup,
        episode_indices=episode_indices,
        seed_base=seed_base,
        swap_envs=swap_envs,
        rollout_mode=mode,
        matchup=matchup,
    )


def _run_parallel_fast_episodes_threaded(
    envs: list[TalisharEngineEnvironment],
    p1_policy: PPOAgent,
    p2_policy: PPOAgent,
    *,
    max_steps: int,
    warmup: bool,
    episode_indices: list[int],
    seed_base: Optional[int],
    swap_envs: Optional[list[TalisharEngineEnvironment]] = None,
    max_workers: int,
    matchup: Optional["Matchup"] = None,
) -> list[dict[str, Any]]:
    """Run one episode per env using threads and per-step policy inference."""
    from concurrent.futures import ThreadPoolExecutor

    swap_matchup = swapped_matchup(matchup) if matchup is not None and swap_envs else None

    def _run_worker(worker: int) -> dict[str, Any]:
        ep_index = episode_indices[worker]
        ep_seed = (seed_base + worker) if seed_base is not None else None
        p1_rng = np.random.default_rng((ep_seed * 31 + 7) if ep_seed is not None else None)
        p2_rng = np.random.default_rng((ep_seed * 31 + 13) if ep_seed is not None else None)
        use_swap = swap_envs is not None and ep_index % 2 == 1
        episode_env = swap_envs[worker] if use_swap and swap_envs is not None else envs[worker]
        result = _run_one_fast_episode(
            episode_env,
            p1_policy,
            p2_policy,
            max_steps,
            ep_seed,
            warmup,
            p1_rng,
            p2_rng,
            starting_player_id=1 + (ep_index % 2),
        )
        if matchup is not None:
            active = swap_matchup if use_swap and swap_matchup is not None else matchup
            result["active_p1_hero"] = active.p1_hero
            result["active_p2_hero"] = active.p2_hero
        return result

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
    """Collect blockers for the fast training path (Talishar fast or C++)."""
    if hasattr(env, "fast_training_unavailable_reasons"):
        return list(env.fast_training_unavailable_reasons())
    inner = getattr(env, "_cpp_env", None)
    if inner is not None and hasattr(inner, "fast_training_unavailable_reasons"):
        return list(inner.fast_training_unavailable_reasons())
    if getattr(env, "_using_cpp", False):
        return ["C++ fast training API not exposed"]
    if hasattr(env, "_using_fast_talishar") and not env._using_fast_talishar:
        return ["talishar_backend is not fast"]
    return ["fast training API not exposed on environment wrapper"]


def _fast_rollout_backend_label(env: Any) -> str:
    """Short label for the active fast rollout backend."""
    if bool(getattr(env, "_using_cpp", False)):
        return "C++ engine"
    backend = getattr(env, "talishar_backend", DEFAULT_TALISHAR_BACKEND)
    return f"Talishar {backend}"


def _announce_training_backend(
    env: Any,
    *,
    require_fast_training: bool = False,
    label: str = "training",
) -> None:
    """Log which rollout path is active and optionally require the fast path."""
    using_cpp = bool(getattr(env, "_using_cpp", False))
    using_fast_talishar = bool(getattr(env, "_using_fast_talishar", False))
    fast = _env_supports_fast_training(env)
    if fast and using_cpp:
        print(
            f"  [fast] {label}: C++ numeric path "
            "(fast_reset / fast_step_index)",
            flush=True,
        )
        return
    if fast and using_fast_talishar:
        backend = getattr(env, "talishar_backend", "fast")
        print(
            f"  [fast] {label}: Talishar {backend} path "
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
    elif require_fast_training:
        reasons = "; ".join(_fast_training_unavailable_reasons(env)) or "unknown"
        raise RuntimeError(f"Fast {label} is unavailable: {reasons}")
    print(
        f"  [slow] {label}: falling back to step() + JSON observations",
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
    _flush_unified_merged_transitions(policy, merged)


def _flush_unified_merged_transitions(
    policy: PPOAgent,
    merged_trans: list[dict[str, Any]],
) -> None:
    if merged_trans:
        buf = _transitions_to_buf(merged_trans)
        _ppo_update_unified(policy, buf, merged_trans[-1]["next_obs_vec"])


@dataclass
class _UnifiedWeightReporter:
    run_dir: Path
    update_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


_unified_weight_reporter: Optional[_UnifiedWeightReporter] = None


def _set_unified_weight_reporter(run_dir: Optional[Path]) -> None:
    global _unified_weight_reporter
    if run_dir is None:
        _unified_weight_reporter = None
        return
    _unified_weight_reporter = _UnifiedWeightReporter(run_dir=run_dir.expanduser().resolve())


def _maybe_record_unified_policy_weights(policy: PPOAgent) -> None:
    reporter = _unified_weight_reporter
    if reporter is None:
        return
    from fab_bridge.policy_weights import summarize_policy_weights  # noqa: PLC0415
    from fab_bridge.unified_dashboard import (  # noqa: PLC0415
        maybe_refresh_unified_dashboard,
        record_policy_weight_update,
    )

    with reporter._lock:
        reporter.update_count += 1
        summary = summarize_policy_weights(policy)
        summary["update_count"] = reporter.update_count
        record_policy_weight_update(reporter.run_dir, summary)
    maybe_refresh_unified_dashboard(reporter.run_dir)


def _ppo_update_unified(
    policy: PPOAgent,
    buf: dict[str, Any],
    next_obs_vec: np.ndarray,
) -> None:
    _ppo_update(policy, buf, next_obs_vec)
    _maybe_record_unified_policy_weights(policy)


def _record_initial_unified_policy_weights(policy: PPOAgent, run_dir: Path) -> None:
    from fab_bridge.policy_weights import summarize_policy_weights  # noqa: PLC0415
    from fab_bridge.unified_dashboard import record_policy_weight_update  # noqa: PLC0415

    reporter = _unified_weight_reporter
    if reporter is not None:
        with reporter._lock:
            if reporter.update_count > 0:
                return
    summary = summarize_policy_weights(policy)
    summary["update_count"] = 0
    summary["phase"] = "bootstrap"
    record_policy_weight_update(run_dir.expanduser().resolve(), summary)


def _flush_unified_warmup_buffers(
    policy: PPOAgent,
    p1_trans: list[dict[str, Any]],
    p2_trans: list[dict[str, Any]],
) -> None:
    merged = _merge_episode_transitions(p1_trans, p2_trans)
    _flush_unified_merged_warmup_transitions(policy, merged)


def _flush_unified_merged_warmup_transitions(
    policy: PPOAgent,
    merged_trans: list[dict[str, Any]],
) -> None:
    if merged_trans:
        buf = _transitions_to_buf(merged_trans)
        _bc_update(policy, buf, merged_trans[-1]["next_obs_vec"])


def _uses_unified_policy(p1_tiers: list[PPOAgent], p2_tiers: list[PPOAgent]) -> bool:
    return bool(
        p1_tiers
        and p2_tiers
        and (p1_tiers is p2_tiers or p1_tiers[0] is p2_tiers[0])
    )


def swapped_matchup(matchup: Matchup) -> Matchup:
    """Same decks with P1/P2 seats reversed (for alternating self-play)."""
    return Matchup(
        name=f"{matchup.name}__swapped",
        p1_deck=matchup.p2_deck,
        p2_deck=matchup.p1_deck,
        description=f"{matchup.description} (swapped seats)",
        tags=list(matchup.tags),
        p1_hero=matchup.p2_hero,
        p2_hero=matchup.p1_hero,
        dir_name=matchup.dir_name,
        cpp_engine_deck1=matchup.cpp_engine_deck2,
        cpp_engine_deck2=matchup.cpp_engine_deck1,
        cpp_engine_dir=matchup.cpp_engine_dir,
    )


def _training_win_rates_from_outcomes(
    outcome_summary: dict[str, Any],
) -> tuple[Optional[float], Optional[float]]:
    """Seat-based win rates from classified episode outcomes."""
    episodes = int(outcome_summary.get("episodes", 0) or 0)
    if episodes <= 0:
        return None, None
    wins = int(outcome_summary.get("wins", 0) or 0)
    losses = int(outcome_summary.get("losses", 0) or 0)
    p1_wr = wins / episodes
    p2_wr = losses / episodes
    return p1_wr, p2_wr


def _training_hero_win_rates_from_outcomes(
    outcome_summary: dict[str, Any],
) -> tuple[Optional[float], Optional[float]]:
    """Nominal hero win rates from classified episode outcomes."""
    heroes = outcome_summary.get("heroes")
    if isinstance(heroes, dict):
        h1 = heroes.get("hero1_win_rate")
        h2 = heroes.get("hero2_win_rate")
        if h1 is not None and h2 is not None:
            return float(h1), float(h2)
    if outcome_summary.get("hero1_win_rate") is not None:
        return (
            float(outcome_summary["hero1_win_rate"]),
            float(outcome_summary.get("hero2_win_rate", 0.0) or 0.0),
        )
    return _training_win_rates_from_outcomes(outcome_summary)


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
                "step_order": step_order,
            })
            step_order += 1
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
                "step_order": step_order,
            })
            step_order += 1
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
            recycle_url = getattr(env, "_base_url", None) or base_url
            new_env = make_env(
                matchup,
                base_url=recycle_url,
                game_format=game_format,
                max_turns=max_steps,
            )
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
    rollout_mode: Optional[str] = None,
    rollout_processes: Optional[int] = None,
    shared_buffer: Any = None,
    backend_pool: TalisharBackendPool | None = None,
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

    talishar_pool = backend_pool or TalisharBackendPool.from_runtime(fallback_url=base_url)
    batch_parallelism = max(1, n_workers)
    resolved_rollout_mode = normalize_rollout_mode(
        rollout_mode or DEFAULT_ROLLOUT_MODE
    )
    resolved_rollout_processes = resolve_rollout_processes(
        rollout_processes,
        default=DEFAULT_ROLLOUT_PROCESSES,
    )
    _announce_rollout_config(
        rollout_mode=resolved_rollout_mode,
        rollout_processes=resolved_rollout_processes,
        n_workers=batch_parallelism,
        backend_pool=talishar_pool,
    )
    use_process_rollouts = resolved_rollout_processes > 1
    if use_process_rollouts:
        from rollout_worker_pool import collect_rollout_batch  # noqa: PLC0415

        rollout_staging = Path(tempfile.mkdtemp(prefix="fab_rollout_"))
    else:
        rollout_staging = None

    p1_policy = p1_tiers[0]
    p2_policy = p2_tiers[0]

    # ── bootstrap: infer dims and init nets on a throw-away env ──────────────
    probe_env = make_env(
        matchup,
        base_url=talishar_pool.primary_url,
        game_format=game_format,
        max_turns=max_steps,
    )
    _announce_training_backend(
        probe_env,
        label="parallel training bootstrap",
        require_fast_training=True,
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

    # ── create worker envs (single-process rollouts only) ───────────────────
    print(
        f"  [parallel] spawning {batch_parallelism} worker game session(s) "
        f"({n_episodes} episodes, batch={batch_parallelism})…"
    )
    envs: list[TalisharEngineEnvironment] = []
    swap_envs: list[TalisharEngineEnvironment] = []
    worker_base_urls: list[str] = []
    swap_matchup = swapped_matchup(matchup)
    if not use_process_rollouts:
        for w in range(batch_parallelism):
            worker_url = talishar_pool.url_for_worker(w)
            worker_base_urls.append(worker_url)
            envs.append(
                make_env(
                    matchup,
                    base_url=worker_url,
                    game_format=game_format,
                    max_turns=max_steps,
                )
            )
            try:
                from fab_bridge.unified_training_debug import (  # noqa: PLC0415
                    log_event as unified_debug_event,
                    shard_label,
                )

                unified_debug_event(
                    "connection",
                    "Training worker bound to shard",
                    matchup=matchup.name,
                    worker=w,
                    shard=shard_label(worker_url),
                    base_url=worker_url,
                )
            except Exception:
                pass
            swap_envs.append(
                make_env(
                    swap_matchup,
                    base_url=worker_url,
                    game_format=game_format,
                    max_turns=max_steps,
                )
            )
        print(f"  [parallel] {batch_parallelism} sessions ready", flush=True)
    else:
        print(
            f"  [parallel] subprocess rollout pool "
            f"({resolved_rollout_processes} process(es)) — envs spawn in workers",
            flush=True,
        )
    print(
        "  [parallel] unified self-play: one policy for both seats; "
        "alternating deck sides each episode",
        flush=True,
    )
    use_fast_rollout = (
        not use_process_rollouts
        and bool(envs)
        and _env_supports_fast_training(envs[0])
    )
    if use_fast_rollout or use_process_rollouts:
        print(
            f"  [parallel] fast rollout mode={resolved_rollout_mode} across "
            f"{batch_parallelism} slot(s)",
            flush=True,
        )

    p1_ep_rewards:  list[float] = []
    p2_ep_rewards:  list[float] = []
    p1_outcomes:    list[str] = []
    outcome_counters = OutcomeCounters(
        nominal_hero1=matchup.p1_hero,
        nominal_hero2=matchup.p2_hero,
    )
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
    episode_timeout_secs = episode_timeout_seconds(max_steps, RUNTIME.engine)
    shutdown_flag = False
    warmup_p1_accum: list[dict[str, Any]] = []
    warmup_p2_accum: list[dict[str, Any]] = []
    warmup_unified_accum: list[dict[str, Any]] = []
    ppo_p1_accum: list[dict[str, Any]] = []
    ppo_p2_accum: list[dict[str, Any]] = []
    ppo_unified_accum: list[dict[str, Any]] = []
    warmup_bc_applied = warmup_episodes <= 0
    use_unified_policy = _uses_unified_policy(p1_tiers, p2_tiers)
    consecutive_all_skipped_batches = 0
    max_consecutive_skip_batches = max(10, len(talishar_pool.urls) * 3)

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
                if use_process_rollouts:
                    assert rollout_staging is not None
                    batch_results = collect_rollout_batch(
                        p1_policy=p1_policy,
                        p2_policy=p2_policy,
                        matchup=matchup,
                        n_episodes=batch_size,
                        n_workers=batch_size,
                        max_steps=max_steps,
                        base_url=talishar_pool.primary_url,
                        game_format=game_format,
                        rollout_mode=resolved_rollout_mode,
                        rollout_processes=resolved_rollout_processes,
                        seed_base=seed_base,
                        warmup=in_warmup,
                        staging_dir=rollout_staging,
                        backend_pool_urls=list(talishar_pool.urls),
                    )
                elif use_fast_rollout:
                    batch_results = _safe_parallel_fast_episode_batch(
                        envs[:batch_size],
                        p1_policy,
                        p2_policy,
                        max_steps=max_steps,
                        warmup=in_warmup,
                        episode_indices=[completed + w for w in range(batch_size)],
                        seed_base=seed_base,
                        swap_envs=swap_envs[:batch_size],
                        rollout_mode=resolved_rollout_mode,
                        max_workers=batch_size,
                        talishar_pool=talishar_pool,
                        matchup=matchup,
                        swap_matchup=swap_matchup,
                        game_format=game_format,
                        worker_base_urls=worker_base_urls,
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
                        episode_env = envs[w]
                        if episode_index % 2 == 1:
                            episode_env = swap_envs[w]
                        fut = pool.submit(
                            _safe_run_one_episode,
                            episode_env, p1_policy, p2_policy,
                            max_steps, ep_seed, in_warmup, p1_rng, p2_rng,
                            matchup, worker_base_urls[w], game_format,
                            starting_player_id,
                        )
                        futures[fut] = w

                # Collect results and merge buffers.
                batch_p1_trans: list[dict] = []
                batch_p2_trans: list[dict] = []
                batch_unified_trans: list[dict[str, Any]] = []
                from concurrent.futures import TimeoutError as FutureTimeoutError
                result_iter: list[dict[str, Any]]
                if use_process_rollouts or use_fast_rollout:
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
                            try:
                                from fab_bridge.unified_training_debug import (  # noqa: PLC0415
                                    log_exception as unified_debug_exception,
                                    shard_label,
                                )

                                unified_debug_exception(
                                    "episode_failed",
                                    "Threaded episode failed",
                                    exc,
                                    worker=w,
                                    shard=shard_label(worker_base_urls[w]),
                                    matchup=matchup.name,
                                    episode_index=completed + w,
                                )
                            except Exception:
                                pass
                            skipped_episodes += 1
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

                batch_had_success = False
                for result in result_iter:
                    if _episode_result_is_skipped(result):
                        skipped_episodes += 1
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
                                print(
                                    f"  [parallel] progress callback failed ({exc!r})"
                                )
                        continue
                    batch_had_success = True
                    if use_unified_policy:
                        batch_unified_trans.extend(
                            _merge_episode_transitions(
                                result["p1_transitions"],
                                result["p2_transitions"],
                            )
                        )
                    else:
                        batch_p1_trans.extend(result["p1_transitions"])
                        batch_p2_trans.extend(result["p2_transitions"])
                    p1_ep_rewards.append(result["p1_reward"])
                    p2_ep_rewards.append(result["p2_reward"])
                    ep_outcome = classify_p1_episode_outcome(
                        p1_hp=result.get("p1_hp"),
                        p2_hp=result.get("p2_hp"),
                        p1_deck=result.get("p1_deck"),
                        p2_deck=result.get("p2_deck"),
                        terminated=bool(result.get("terminated")),
                        truncated=bool(result.get("truncated")),
                    )
                    p1_outcomes.append(ep_outcome)
                    outcome_counters.record_seat_outcome(
                        ep_outcome,
                        active_p1_hero=str(
                            result.get("active_p1_hero") or matchup.p1_hero
                        ),
                        active_p2_hero=str(
                            result.get("active_p2_hero") or matchup.p2_hero
                        ),
                        nominal_hero1=matchup.p1_hero,
                        nominal_hero2=matchup.p2_hero,
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
                if not batch_had_success and batch_size > 0:
                    consecutive_all_skipped_batches += 1
                    if consecutive_all_skipped_batches >= max_consecutive_skip_batches:
                        raise RuntimeError(
                            "All Talishar training shards failed repeatedly; "
                            f"no successful episodes in the last "
                            f"{consecutive_all_skipped_batches} batch(es). "
                            "Check docker Talishar backends and retry."
                        )
                else:
                    consecutive_all_skipped_batches = 0
                if shutdown_flag:
                    break

                preview_trans = (
                    batch_unified_trans
                    if use_unified_policy
                    else batch_p1_trans
                )
                if live_state_image_path is not None and preview_trans:
                    last_obs_vec = preview_trans[-1]["obs_vec"]
                    _write_state_image(
                        {"obs_vec_shape": str(last_obs_vec.shape)},
                        live_state_image_path,
                        header=(
                            f"episode={completed}/{n_episodes} "
                            f"parallel_batch={batch_parallelism}"
                        ),
                    )

                if in_warmup:
                    if use_unified_policy:
                        warmup_unified_accum.extend(batch_unified_trans)
                    else:
                        warmup_p1_accum.extend(batch_p1_trans)
                        warmup_p2_accum.extend(batch_p2_trans)
                    if completed >= warmup_episodes and not warmup_bc_applied:
                        trans_count = (
                            len(warmup_unified_accum)
                            if use_unified_policy
                            else len(warmup_p1_accum) + len(warmup_p2_accum)
                        )
                        print(
                            f"  [warmup] behavioural-cloning update from "
                            f"{trans_count} transitions"
                        )
                        if use_unified_policy:
                            if shared_buffer is not None:
                                with shared_buffer._lock:
                                    _flush_unified_merged_warmup_transitions(
                                        p1_tiers[0], warmup_unified_accum,
                                    )
                            else:
                                _flush_unified_merged_warmup_transitions(
                                    p1_tiers[0], warmup_unified_accum,
                                )
                            warmup_unified_accum.clear()
                        else:
                            _flush_warmup_buffers_auto(
                                p1_tiers, p2_tiers, warmup_p1_accum, warmup_p2_accum,
                            )
                            warmup_p1_accum.clear()
                            warmup_p2_accum.clear()
                        warmup_bc_applied = True
                else:
                    if use_unified_policy:
                        if shared_buffer is not None:
                            shared_buffer.extend_ppo(batch_unified_trans)
                            shared_buffer.maybe_flush_ppo(p1_tiers[0])
                        else:
                            ppo_unified_accum.extend(batch_unified_trans)
                            rollout_ready = (
                                len(ppo_unified_accum) >= DEFAULT_PPO_ROLLOUT_BATCH
                                or completed >= n_episodes
                            )
                            if rollout_ready:
                                _flush_unified_merged_transitions(
                                    p1_tiers[0], ppo_unified_accum,
                                )
                                ppo_unified_accum.clear()
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
        if not warmup_bc_applied:
            if use_unified_policy and warmup_unified_accum:
                if shared_buffer is not None:
                    with shared_buffer._lock:
                        _flush_unified_merged_warmup_transitions(
                            p1_tiers[0], warmup_unified_accum,
                        )
                else:
                    _flush_unified_merged_warmup_transitions(
                        p1_tiers[0], warmup_unified_accum,
                    )
            elif warmup_p1_accum or warmup_p2_accum:
                _flush_warmup_buffers_auto(
                    p1_tiers, p2_tiers, warmup_p1_accum, warmup_p2_accum,
                )
        if use_unified_policy and shared_buffer is None and ppo_unified_accum:
            _flush_unified_merged_transitions(p1_tiers[0], ppo_unified_accum)
        elif ppo_p1_accum or ppo_p2_accum:
            _flush_ppo_buffers_auto(p1_tiers, p2_tiers, ppo_p1_accum, ppo_p2_accum)
        for env in envs + swap_envs:
            try:
                env.close()
            except Exception:
                pass

    total_eps = len(p1_ep_rewards)
    outcome_summary = summarize_p1_outcomes(p1_outcomes, episodes=total_eps)
    hero_summary = summarize_hero_outcomes(outcome_counters, episodes=total_eps)
    stats = {
        "episodes":   total_eps,
        "timeouts":   outcome_summary["timeouts"],
        "terminated": terminated_episodes,
        "skipped":    skipped_episodes,
        "timeout_rate": outcome_summary["timeout_rate"],
        "p1_outcomes": p1_outcomes,
        **outcome_summary,
        **hero_summary,
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
                    _ppo_update_unified(p1_tiers[0], unified_buf, next_obs_vec)
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
            _ppo_update_unified(p1_tiers[0], unified_buf, next_vec)
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

    package_dir = matchup_out_dir(out_dir, matchup) / f"ppo_{agent_id}"
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


def _talishar_deck_spec(
    deck_stem: str,
    *,
    role: str,
    hero_id: str,
    matchup_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Build phase-3-compatible ``deck_spec`` from a Talishar Assets deck stem."""
    from flesh_and_blood_rlbridge.deck_context import _read_asset_deck  # noqa: PLC0415
    from flesh_and_blood_rlbridge.talishar_deck_assets import (  # noqa: PLC0415
        load_guide_sideboard_record,
        resolve_matchup_equipment_header,
        resolve_talishar_deck_stem,
    )

    assets = talishar_assets_path()
    resolved = resolve_talishar_deck_stem(assets, deck_stem)
    _hero_id, counts = _read_asset_deck(assets, resolved)
    guide = load_guide_sideboard_record(matchup_dir) if matchup_dir else {}
    equipment_header = resolve_matchup_equipment_header(
        role=role,
        hero_id=hero_id or _hero_id,
        deck_stem=resolved,
        assets_dir=assets,
        fallback=_hero_id,
        guide_sideboard=guide,
    )
    return {"equipment_header": equipment_header, "cards": counts}


def _save_unified_selfplay_checkpoint(
    *,
    out_dir: Path,
    matchup: Matchup,
    game_format: str,
    policy: PPOAgent,
    episodes_completed: int,
    target_episodes: int,
) -> None:
    """Persist discoverable unified self-play checkpoints for Talishar eval."""
    matchup_dir = matchup_out_dir(out_dir, matchup)
    p1_spec = _talishar_deck_spec(
        matchup.p1_deck,
        role="p1",
        hero_id=matchup.p1_hero,
        matchup_dir=matchup_dir,
    )
    p2_spec = _talishar_deck_spec(
        matchup.p2_deck,
        role="p2",
        hero_id=matchup.p2_hero,
        matchup_dir=matchup_dir,
    )
    role_specs = {
        "p1": p1_spec,
        "p2": p2_spec,
    }
    for role, deck_spec in role_specs.items():
        ckpt_dir = (
            matchup_dir
            / "unified_selfplay"
            / role
            / f"episode_{episodes_completed:06d}"
        )
        weights_dir = ckpt_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        policy.save(weights_dir / "agent_weights.json")
        metadata = {
            "checkpoint_type": "unified_selfplay",
            "created_at": datetime.now().isoformat(),
            "matchup": matchup.name,
            "role": role,
            "game_format": game_format,
            "weights_file": "agent_weights.json",
            "episodes_completed": episodes_completed,
            "target_episodes": target_episodes,
            "p1_deck": matchup.p1_deck,
            "p2_deck": matchup.p2_deck,
            "p1_hero": matchup.p1_hero,
            "p2_hero": matchup.p2_hero,
            "cpp_engine_deck1": matchup.cpp_engine_deck1,
            "cpp_engine_deck2": matchup.cpp_engine_deck2,
            "cpp_engine_dir": matchup.cpp_engine_dir,
            "opponent_mode": "dual",
            "opponent_deck_name": matchup.p2_hero if role == "p1" else matchup.p1_hero,
            "deck_spec": deck_spec,
        }
        (ckpt_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )


def _write_unified_checkpoint_eval_scope(
    out_dir: Path,
    matchup: Matchup,
) -> None:
    from fab_bridge.unified_results import CHECKPOINT_EVAL_SCOPE  # noqa: PLC0415

    payload = {
        "matchup": matchup.name,
        "matchup_dir": _resolve_matchup_subdir(out_dir, matchup),
        "updated_at": datetime.now().isoformat(),
    }
    (out_dir / CHECKPOINT_EVAL_SCOPE).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _unified_run_progress_callback(
    out_dir: Path,
    matchup: Matchup,
    *,
    target_episodes: int,
    matchups_completed: int,
    matchups_total: int,
    matchup_live_key: Optional[str] = None,
) -> Optional[Callable[..., None]]:
    if not is_unified_random_matchup_run(out_dir):
        return None

    def _callback(
        completed: int,
        p1_rewards: Any = None,
        p2_rewards: Any = None,
        p1_outcomes: Any = None,
    ) -> None:
        p1_wr: Optional[float] = None
        p2_wr: Optional[float] = None
        hero1_wr: Optional[float] = None
        hero2_wr: Optional[float] = None
        if p1_outcomes:
            summary = summarize_p1_outcomes(p1_outcomes, episodes=completed)
            p1_wr, p2_wr = _training_win_rates_from_outcomes(summary)
            hero1_wr, hero2_wr = _training_hero_win_rates_from_outcomes(summary)
        if matchup_live_key:
            from fab_bridge.unified_dashboard import update_unified_matchup_live  # noqa: PLC0415

            update_unified_matchup_live(
                out_dir,
                matchup_live_key,
                name=matchup.name,
                episodes_completed=completed,
                p1_win_rate=p1_wr,
                p2_win_rate=p2_wr,
                hero1_win_rate=hero1_wr,
                hero2_win_rate=hero2_wr,
                status="training",
            )
            update_unified_training_live(
                out_dir,
                target_episodes=target_episodes,
                matchups_total=matchups_total,
                matchups_completed=matchups_completed,
                status="training",
            )
        else:
            update_unified_training_live(
                out_dir,
                current_matchup=matchup.name,
                current_matchup_dir=_resolve_matchup_subdir(out_dir, matchup),
                target_episodes=target_episodes,
                matchups_total=matchups_total,
                matchups_completed=matchups_completed,
                episodes_completed=completed,
                p1_win_rate=p1_wr,
                p2_win_rate=p2_wr,
                status="training",
            )
        maybe_refresh_unified_dashboard(out_dir)

    return _callback


def _combined_unified_training_progress(
    *,
    ckpt_tracker: Optional["_CheckpointEvalTracker"] = None,
    checkpoint_coordinator: Optional[Any] = None,
    matchup_live_key: Optional[str] = None,
    dash_cb: Optional[Callable[..., None]] = None,
) -> Optional[Callable[..., None]]:
    if (
        ckpt_tracker is None
        and checkpoint_coordinator is None
        and dash_cb is None
    ):
        return None

    def _callback(completed: int, *args: Any) -> None:
        if checkpoint_coordinator is not None and matchup_live_key:
            checkpoint_coordinator.report_progress(matchup_live_key, completed)
        elif ckpt_tracker is not None:
            ckpt_tracker.on_parallel_progress(completed, *args)
        if dash_cb is not None:
            try:
                dash_cb(completed, *args)
            except TypeError:
                dash_cb(completed)

    return _callback


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
        policy_snapshot_fn: Optional[Callable[[], tuple[PPOAgent, PPOAgent]]] = None,
        eval_agent_vs_logic: bool = DEFAULT_CHECKPOINT_EVAL_AGENT_VS_LOGIC,
        eval_logic_vs_logic: bool = DEFAULT_CHECKPOINT_EVAL_LOGIC_VS_LOGIC,
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
        self._logic_vs_logic_baseline = self._load_logic_vs_logic_baseline()
        self._policy_snapshot_fn = policy_snapshot_fn
        self._eval_agent_vs_logic = bool(eval_agent_vs_logic)
        self._eval_logic_vs_logic = bool(eval_logic_vs_logic)

    def _logic_vs_logic_baseline_path(self) -> Path:
        return (
            matchup_out_dir(self.out_dir, self.matchup)
            / LOGIC_VS_LOGIC_BASELINE_NAME
        )

    def _load_logic_vs_logic_baseline(self) -> Optional[dict[str, Any]]:
        path = self._logic_vs_logic_baseline_path()
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, TypeError):
            return None
        return raw if isinstance(raw, dict) else None

    def _save_logic_vs_logic_baseline(self, baseline: dict[str, Any]) -> None:
        path = self._logic_vs_logic_baseline_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        self._logic_vs_logic_baseline = baseline

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
        if self._policy_snapshot_fn is not None:
            eval_p1, eval_p2 = self._policy_snapshot_fn()
        else:
            clone_agent_weights(self.p1_policy, eval_p1)
            clone_agent_weights(self.p2_policy, eval_p2)
        if eval_p1._shared is None or eval_p2._shared is None:
            raise RuntimeError(
                "Checkpoint eval could not clone unified policy weights "
                "(both seats must use the trained agent, not pass-only fallback)"
            )
        from train_play import _evaluate_p1_vs_fixed_opponent  # noqa: PLC0415
        from fab_bridge.cpp_eval_live_dashboard import (  # noqa: PLC0415
            unified_checkpoint_eval_live_path,
        )

        live_progress_path = unified_checkpoint_eval_live_path(self.out_dir)

        metrics = _evaluate_p1_vs_fixed_opponent(
            self.matchup,
            eval_p1,
            p2_agent=eval_p2,
            base_url=self.base_url,
            game_format=self.game_format,
            max_steps=self.max_steps,
            episodes=self.checkpoint_eval_episodes,
            seed=(self.seed + completed) if self.seed is not None else None,
            backend=DEFAULT_TALISHAR_BACKEND,
            eval_label="Checkpoint eval (self-play)",
            live_progress_path=live_progress_path,
        )
        vs_logic: Optional[dict[str, Any]] = None
        logic_vs_logic: Optional[dict[str, Any]] = None
        if self.checkpoint_eval_episodes > 0:
            from train_play import (  # noqa: PLC0415
                evaluate_agent_vs_logic_both_seats,
                evaluate_logic_vs_logic,
            )

            if self._eval_agent_vs_logic:
                vs_logic = evaluate_agent_vs_logic_both_seats(
                    self.matchup,
                    eval_p1,
                    base_url=self.base_url,
                    game_format=self.game_format,
                    max_steps=self.max_steps,
                    episodes=self.checkpoint_eval_episodes,
                    seed=(self.seed + completed) if self.seed is not None else None,
                    backend=DEFAULT_TALISHAR_BACKEND,
                    eval_label_prefix="Checkpoint eval vs logic",
                    live_progress_path=live_progress_path,
                )
            if self._eval_logic_vs_logic and self._logic_vs_logic_baseline is None:
                logic_vs_logic = evaluate_logic_vs_logic(
                    self.matchup,
                    base_url=self.base_url,
                    game_format=self.game_format,
                    max_steps=self.max_steps,
                    episodes=self.checkpoint_eval_episodes,
                    seed=(self.seed + 100_000) if self.seed is not None else None,
                    backend=DEFAULT_TALISHAR_BACKEND,
                    eval_label="Checkpoint eval logic win% vs logic",
                    live_progress_path=live_progress_path,
                )
                self._save_logic_vs_logic_baseline(logic_vs_logic)
        record = {
            "matchup": self.matchup.name,
            "matchup_dir": _resolve_matchup_subdir(self.out_dir, self.matchup),
            "episodes_completed": completed,
            "target_episodes": self.n_episodes,
            "eval_episodes": self.checkpoint_eval_episodes,
            "eval_mode": "self_play",
            **metrics,
        }
        if vs_logic is not None:
            record["vs_logic"] = vs_logic
        if logic_vs_logic is not None:
            record["logic_vs_logic"] = logic_vs_logic
        elif self._eval_logic_vs_logic and self._logic_vs_logic_baseline is not None:
            record["logic_vs_logic"] = self._logic_vs_logic_baseline
        self.log.append(record)
        heroes = metrics.get("heroes") if isinstance(metrics.get("heroes"), dict) else {}
        wr = float(
            heroes.get("hero1_win_rate")
            or metrics.get("hero1_win_rate")
            or metrics["p1_win_rate"]
        )
        if self.first_win_rate is None:
            self.first_win_rate = wr
        self.final_win_rate = wr
        _save_unified_selfplay_checkpoint(
            out_dir=self.out_dir,
            matchup=self.matchup,
            game_format=self.game_format,
            policy=self.p1_policy,
            episodes_completed=completed,
            target_episodes=self.n_episodes,
        )
        _write_unified_checkpoint_eval_scope(self.out_dir, self.matchup)
        ckpt_dir = (
            matchup_out_dir(self.out_dir, self.matchup)
            / "checkpoint_eval"
            / f"episode_{completed:06d}"
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "checkpoint_eval.json").write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )
        matchup_history_path = (
            matchup_out_dir(self.out_dir, self.matchup) / "checkpoint_eval_history.json"
        )
        matchup_history_path.write_text(json.dumps(self.log, indent=2), encoding="utf-8")
        history_path = self.out_dir / "checkpoint_eval_history.json"
        history_path.write_text(json.dumps(self.log, indent=2), encoding="utf-8")
        print(
            f"  Checkpoint eval @ ep {completed}: "
            f"self-play hero1 win%={wr:.1%} "
            f"({metrics.get('heroes', {}).get('hero1_wins', metrics['p1_wins'])}H1/"
            f"{metrics.get('heroes', {}).get('hero2_wins', metrics['p2_wins'])}H2 "
            f"over {self.checkpoint_eval_episodes} games)"
        )
        if vs_logic is not None:
            p1_logic_wr = float(
                vs_logic["agent_p1_seat"].get("agent_win_rate", 0.0) or 0.0
            )
            p2_logic_wr = float(
                vs_logic["agent_p2_seat"].get("agent_win_rate", 0.0) or 0.0
            )
            print(
                f"  Checkpoint eval vs logic @ ep {completed}: "
                f"agent@P1={p1_logic_wr:.1%}  agent@P2={p2_logic_wr:.1%} "
                f"({self.checkpoint_eval_episodes} games per seat)"
            )
        if logic_vs_logic is not None:
            logic_heroes = (
                logic_vs_logic.get("heroes")
                if isinstance(logic_vs_logic.get("heroes"), dict)
                else {}
            )
            logic_p1_wr = float(
                logic_heroes.get("hero1_win_rate")
                or logic_vs_logic.get("hero1_win_rate")
                or logic_vs_logic.get("p1_win_rate", 0.0)
                or 0.0
            )
            print(
                f"  Checkpoint eval logic win% vs logic (matchup baseline): "
                f"hero1 win%={logic_p1_wr:.1%} "
                f"({self.checkpoint_eval_episodes} games)"
            )
        maybe_refresh_unified_dashboard(self.out_dir, min_interval_seconds=0.0)


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
    ckpt_dir = matchup_out_dir(out_dir, matchup) / "warmup_handoff_baseline"
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
    build_cpp_engine: bool = False,
    require_cpp_engine: bool = False,
    backend_pool: TalisharBackendPool | None = None,
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
        backend_pool=backend_pool,
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
    p1_id = new_agent_id("p1")
    p2_id = new_agent_id("p2")
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


def _write_matchup_dir_label(out_dir: Path, matchup: Matchup) -> Path:
    """Create the matchup output folder and record the full matchup label."""
    matchup_root = matchup_out_dir(out_dir, matchup)
    matchup_root.mkdir(parents=True, exist_ok=True)
    label_path = matchup_root / "matchup_label.json"
    if not label_path.exists():
        label_path.write_text(
            json.dumps(
                {
                    "name": matchup.name,
                    "output_subdir": matchup_root.name,
                    "p1_deck": matchup.p1_deck,
                    "p2_deck": matchup.p2_deck,
                    "p1_hero": matchup.p1_hero,
                    "p2_hero": matchup.p2_hero,
                    "description": matchup.description,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return matchup_root


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
    checkpoint_eval_logic_vs_logic: bool = DEFAULT_CHECKPOINT_EVAL_LOGIC_VS_LOGIC,
    checkpoint_eval_agent_vs_logic: bool = DEFAULT_CHECKPOINT_EVAL_AGENT_VS_LOGIC,
    build_cpp_engine: bool = False,
    require_cpp_engine: bool = False,
    rollout_mode: Optional[str] = None,
    rollout_processes: Optional[int] = None,
    _seed_run_capture: Optional[dict[str, Any]] = None,
    _skip_cache_converge: bool = False,
    _force_train: bool = False,
    _unified_matchups_completed: int = 0,
    _unified_matchups_total: Optional[int] = None,
    _shared_policy: Optional[PPOAgent] = None,
    _shared_buffer: Any = None,
    _skip_persist: bool = False,
    _matchup_live_key: Optional[str] = None,
    _checkpoint_coordinator: Optional[Any] = None,
    backend_pool: TalisharBackendPool | None = None,
) -> dict:
    if matchup.p1_fabrary_entry and matchup.p2_fabrary_entry:
        from runtime_defaults import RUNTIME  # noqa: PLC0415

        apply_unified_matchup_sideboards(
            matchup,
            game_format=game_format,
            base_url=base_url,
            out_dir=out_dir,
            max_sideboard_steps=RUNTIME.full_pipeline.max_sideboard_steps,
        )
        matchup.cpp_engine_dir = None

    if build_cpp_engine or require_cpp_engine:
        ensure_matchup_cpp_engine(
            matchup,
            base_url=base_url,
            build=build_cpp_engine,
            require=require_cpp_engine,
        )
    else:
        matchup.cpp_engine_dir = None

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
            backend_pool=backend_pool,
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

    _write_matchup_dir_label(out_dir, matchup)

    if _unified_matchups_total is not None:
        if _matchup_live_key:
            from fab_bridge.unified_dashboard import update_unified_matchup_live  # noqa: PLC0415

            update_unified_matchup_live(
                out_dir,
                _matchup_live_key,
                name=matchup.name,
                episodes_completed=0,
                p1_win_rate=None,
                p2_win_rate=None,
                status="training",
            )
            update_unified_training_live(
                out_dir,
                target_episodes=n_episodes,
                matchups_total=_unified_matchups_total,
                matchups_completed=_unified_matchups_completed,
                status="training",
            )
        else:
            update_unified_training_live(
                out_dir,
                current_matchup=matchup.name,
                current_matchup_dir=_resolve_matchup_subdir(out_dir, matchup),
                target_episodes=n_episodes,
                matchups_total=_unified_matchups_total,
                matchups_completed=_unified_matchups_completed,
                episodes_completed=0,
                p1_win_rate=None,
                p2_win_rate=None,
                status="training",
            )
        if _matchup_live_key is None:
            write_unified_random_matchups_dashboard(out_dir, auto_refresh_seconds=5.0)

    if build_cpp_engine and matchup.cpp_engine_dir:
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
                    "package_dir": str(matchup_out_dir(out_dir, matchup) / "cached_unified"),
                },
                "p2": {
                    "matchup": matchup.name,
                    "cached": True,
                    "avg_reward": 0.0,
                    "best_reward": 0.0,
                    "elapsed_secs": 0.0,
                    "agent_id": "cached_unified",
                    "package_dir": str(matchup_out_dir(out_dir, matchup) / "cached_unified"),
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
    talishar_pool = backend_pool or TalisharBackendPool.from_runtime(fallback_url=base_url)
    if backend_pool is None:
        talishar_pool = talishar_pool.filter_healthy()
        print(f"  Talishar backends: {talishar_pool.format_log_label()}")

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
                    ["docker", "compose", "restart", "web-server"],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    check=False,
                )
                time.sleep(5)
            else:
                raise

    _announce_training_backend(
        probe_env,
        label="matchup training",
        require_fast_training=True,
    )

    unified_bundle = (
        None
        if _shared_policy is not None
        else _bootstrap_unified_policy(cache_store, probe_env, seed)
    )
    if _shared_policy is not None:
        unified_policy = _shared_policy
        print("  Unified policy init: shared (parallel batch)")
    else:
        unified_policy = unified_bundle.policy
        print("  Unified policy init:", ", ".join(unified_bundle.init_sources))
    policy_tiers = [unified_policy]
    print(
        f"  obs_schema={PLAYER_OBS_SCHEMA_VERSION} "
        f"obs_dim={unified_policy.obs_dim} "
        f"n_actions={unified_policy.n_actions}"
    )
    print(
        "  Unified self-play: one shared policy controls both seats; "
        "PPO updates merge P1+P2 transitions"
    )
    if _unified_matchups_total is not None and is_unified_random_matchup_run(out_dir):
        _record_initial_unified_policy_weights(unified_policy, out_dir)

    use_fast_parallel = _env_supports_fast_training(probe_env)
    rollout_workers = max(1, int(n_workers))
    if use_fast_parallel:
        backend_label = _fast_rollout_backend_label(probe_env)
        worker_note = (
            f" ({rollout_workers} workers)"
            if rollout_workers > 1
            else " (single worker)"
        )
        print(f"  Using {backend_label} fast-path rollouts{worker_note}")

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
        live_state_image_path = matchup_out_dir(out_dir, matchup) / "training_live_state.png"
        live_state_image_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Live state image path → {live_state_image_path}")

    ckpt_tracker: Optional[_CheckpointEvalTracker] = None
    dash_cb = (
        _unified_run_progress_callback(
            out_dir,
            matchup,
            target_episodes=n_episodes,
            matchups_completed=_unified_matchups_completed,
            matchups_total=_unified_matchups_total,
            matchup_live_key=_matchup_live_key,
        )
        if _unified_matchups_total is not None
        else None
    )
    policy_snapshot_fn: Optional[Callable[[], tuple[PPOAgent, PPOAgent]]] = None
    if _shared_buffer is not None:
        def _clone_for_eval() -> tuple[PPOAgent, PPOAgent]:
            snap = _shared_buffer.clone_policy_snapshot()
            return snap, snap

        policy_snapshot_fn = _clone_for_eval
    effective_ckpt_interval = resolve_checkpoint_interval(
        n_episodes,
        checkpoint_interval=checkpoint_interval,
        checkpoint_interval_pct=checkpoint_interval_pct,
    )
    if checkpoint_eval_episodes > 0 and _checkpoint_coordinator is None:
        ckpt_tracker = _CheckpointEvalTracker(
            matchup=matchup,
            base_url=base_url,
            game_format=game_format,
            max_steps=max_steps,
            n_episodes=n_episodes,
            checkpoint_interval=effective_ckpt_interval,
            checkpoint_eval_episodes=checkpoint_eval_episodes,
            p1_policy=unified_policy,
            p2_policy=unified_policy,
            seed=seed,
            out_dir=out_dir,
            policy_snapshot_fn=policy_snapshot_fn,
            eval_agent_vs_logic=checkpoint_eval_agent_vs_logic,
            eval_logic_vs_logic=checkpoint_eval_logic_vs_logic,
        )
        print(
            f"  Checkpoint eval: every {effective_ckpt_interval} episode(s), "
            f"{checkpoint_eval_episodes} eval game(s) per checkpoint "
            f"(self-play + vs logic on P1/P2 seats; logic win% vs logic once per matchup)"
        )
        _write_unified_checkpoint_eval_scope(out_dir, matchup)

    # ── parallel / fast rollout path ────────────────────────────────────────
    if rollout_workers > 1 or use_fast_parallel:
        if rollout_workers > 1:
            print(f"  [parallel] Using {rollout_workers} worker game sessions for training")
        rem_p1, rem_p2, rem_stats = train_agents_from_both_perspectives_parallel(
            matchup=matchup,
            base_url=base_url,
            game_format=game_format,
            p1_tiers=policy_tiers,
            p2_tiers=policy_tiers,
            n_episodes=n_episodes,
            max_steps=max_steps,
            seed=seed,
            warmup_episodes=warmup_count,
            n_workers=rollout_workers,
            live_state_image_path=live_state_image_path,
            episode_cache=episode_cache,
            on_episodes_progress=_combined_unified_training_progress(
                ckpt_tracker=ckpt_tracker,
                checkpoint_coordinator=_checkpoint_coordinator,
                matchup_live_key=_matchup_live_key,
                dash_cb=dash_cb,
            ),
            rollout_mode=rollout_mode,
            rollout_processes=rollout_processes,
            shared_buffer=_shared_buffer,
            backend_pool=talishar_pool,
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
                p1_policy=unified_policy,
                p2_policy=unified_policy,
                episodes=warmup_baseline_eval_episodes,
                seed=(seed + 100_000) if seed is not None else None,
            )
            ckpt_dir = _save_warmup_handoff_checkpoint(
                out_dir=out_dir,
                matchup=matchup,
                p1_policy=unified_policy,
                p2_policy=unified_policy,
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
                            ["docker", "compose", "restart", "web-server"],
                            cwd=REPO_ROOT,
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
                policy_tiers,
                policy_tiers,
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
                p1_policy=unified_policy,
                p2_policy=unified_policy,
                episodes=warmup_baseline_eval_episodes,
                seed=(seed + 100_000) if seed is not None else None,
            )
            ckpt_dir = _save_warmup_handoff_checkpoint(
                out_dir=out_dir,
                matchup=matchup,
                p1_policy=unified_policy,
                p2_policy=unified_policy,
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
                policy_tiers,
                policy_tiers,
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

    p1_wr, p2_wr = _training_win_rates_from_outcomes(outcome_summary)

    training_stats = dict(overall_stats)
    if ckpt_tracker is not None and ckpt_tracker.log:
        training_stats["checkpoint_eval_history"] = ckpt_tracker.log
    elif (
        _checkpoint_coordinator is not None
        and _matchup_live_key
    ):
        matchup_ckpt_hist = _checkpoint_coordinator.get_matchup_checkpoint_history(
            _matchup_live_key
        )
        if matchup_ckpt_hist:
            training_stats["checkpoint_eval_history"] = matchup_ckpt_hist

    if (
        _checkpoint_coordinator is not None
        and not _skip_persist
        and _matchup_live_key
    ):
        from checkpoint_eval_async import wait_for_checkpoint_evals  # noqa: PLC0415

        wait_for_checkpoint_evals()

    persist_summary: Optional[dict[str, Any]] = None
    if len(p1_rewards) >= n_episodes and not _skip_cache_converge:
        coordinator_wr = (
            _checkpoint_coordinator.get_matchup_checkpoint_win_rate(_matchup_live_key)
            if _checkpoint_coordinator is not None and _matchup_live_key
            else None
        )
        persist_summary = {
            "matchup_name": matchup.name,
            "matchup_dir": _matchup_live_key,
            "p1_fingerprint": p1_deck_fp,
            "p2_fingerprint": p2_deck_fp,
            "p1_hero": matchup.p1_hero,
            "p2_hero": matchup.p2_hero,
            "episodes_completed": len(p1_rewards),
            "target_episodes": n_episodes,
            "p1_win_rate": p1_wr,
            "p2_win_rate": p2_wr,
            "first_checkpoint_win_rate": (
                _checkpoint_coordinator.first_win_rate
                if _checkpoint_coordinator is not None
                else ckpt_tracker.first_win_rate if ckpt_tracker is not None else None
            ),
            "final_checkpoint_win_rate": (
                coordinator_wr
                if coordinator_wr is not None
                else ckpt_tracker.final_win_rate if ckpt_tracker is not None else None
            ),
            "checkpoint_eval_win_rate": (
                coordinator_wr
                if coordinator_wr is not None
                else ckpt_tracker.final_win_rate if ckpt_tracker is not None else None
            ),
            "training_stats": training_stats,
        }

    if not _skip_persist:
        cache_store.persist(
            unified_policy,
            episodes_delta=len(p1_rewards),
            training_summary=persist_summary,
        )

    print(
        f"  Done in {elapsed:.1f}s — "
        f"P1 avg={np.mean(p1_rewards):+.3f}  "
        f"P2 avg={np.mean(p2_rewards):+.3f}  "
        f"timeouts={overall_stats['timeouts']}/{overall_stats['episodes']} "
        f"({overall_stats['timeout_rate'] * 100:.1f}%)"
    )
    if p1_wr is not None and p2_wr is not None and _seed_run_capture is None:
        print(
            f"  Train win% (P1 seat / P2 seat): "
            f"P1={p1_wr * 100:.1f}  P2={p2_wr * 100:.1f}  "
            f"(same unified policy on both sides)"
        )

    if _seed_run_capture is not None:
        _seed_run_capture.update(
            {
                "p1_agent": unified_policy,
                "p2_agent": unified_policy,
                "p1_win_rate": float(p1_wr or 0.0),
                "p2_win_rate": float(p2_wr or 0.0),
            }
        )

    p1_id = new_agent_id("p1")
    p2_id = new_agent_id("p2")

    meta_p1 = save_agent(
        unified_policy, p1_id, out_dir, matchup, p1_rewards,
        n_episodes, elapsed, game_format, "p1", eval_env_ids,
        warmup_baseline=warmup_baseline,
        training_stats=overall_stats,
    )
    meta_p2 = save_agent(
        unified_policy, p2_id, out_dir, matchup, p2_rewards,
        n_episodes, elapsed, game_format, "p2", eval_env_ids,
        warmup_baseline=warmup_baseline,
        training_stats=overall_stats,
    )
    if p1_wr is not None:
        meta_p1["training_win_rate"] = p1_wr
    if p2_wr is not None:
        meta_p2["training_win_rate"] = p2_wr

    if _unified_matchups_total is not None:
        if _matchup_live_key:
            from fab_bridge.unified_dashboard import update_unified_matchup_live  # noqa: PLC0415

            update_unified_matchup_live(
                out_dir,
                _matchup_live_key,
                name=matchup.name,
                episodes_completed=n_episodes,
                p1_win_rate=p1_wr,
                p2_win_rate=p2_wr,
                status="complete",
            )
        else:
            update_unified_training_live(
                out_dir,
                current_matchup=matchup.name,
                current_matchup_dir=_resolve_matchup_subdir(out_dir, matchup),
                target_episodes=n_episodes,
                matchups_total=_unified_matchups_total,
                matchups_completed=_unified_matchups_completed,
                episodes_completed=n_episodes,
                p1_win_rate=p1_wr,
                p2_win_rate=p2_wr,
                status="training",
            )
        if _matchup_live_key is None:
            write_unified_random_matchups_dashboard(out_dir, auto_refresh_seconds=5.0)

    result: dict[str, Any] = {"p1": meta_p1, "p2": meta_p2}
    if persist_summary is not None:
        result["_persist_summary"] = persist_summary
    return result


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
    parallel_matchups: int = 1,
    checkpoint_interval_pct: float = DEFAULT_CHECKPOINT_INTERVAL_PCT,
    checkpoint_interval: Optional[int] = None,
    checkpoint_eval_episodes: int = DEFAULT_CHECKPOINT_EVAL_EPISODES,
    checkpoint_eval_logic_vs_logic: bool = DEFAULT_CHECKPOINT_EVAL_LOGIC_VS_LOGIC,
    checkpoint_eval_agent_vs_logic: bool = DEFAULT_CHECKPOINT_EVAL_AGENT_VS_LOGIC,
    skip_converged: bool = True,
    build_cpp_engine: bool = False,
    require_cpp_engine: bool = False,
    rollout_mode: Optional[str] = None,
    rollout_processes: Optional[int] = None,
    backend_pool: TalisharBackendPool | None = None,
    safe_parallel: bool = False,
    debug_training: bool = False,
) -> tuple[list[dict], list[str]]:
    from agent_cache import AgentCacheStore
    from fab_bridge.unified_training_debug import (  # noqa: PLC0415
        audit_matchup_decks,
        configure as configure_unified_debug,
        is_enabled as unified_debug_enabled,
        log_event as unified_debug_event,
        log_exception as unified_debug_exception,
        log_shard_pool,
        shard_label,
    )
    from unified_parallel_training import (
        UnifiedSharedExperienceBuffer,
        concurrent_training_game_slots,
        persist_batch_to_cache,
        resolve_safe_parallel_limits,
        run_parallel_matchup_batch,
        workers_per_parallel_matchup,
    )

    cache_root = cache_dir or (REPO_ROOT / "results" / "agent_cache")
    cache_store = AgentCacheStore(
        cache_root,
        game_format,
        obs_schema_version=PLAYER_OBS_SCHEMA_VERSION,
    )

    summary: list[dict] = []
    failed: list[str] = []
    unified_run = is_unified_random_matchup_run(out_dir)
    if debug_training or unified_debug_enabled():
        configure_unified_debug(run_dir=out_dir, enabled=True)
    matchups_total = len(matchups)
    parallel_matchups = max(1, int(parallel_matchups))
    n_workers = max(1, int(n_workers))
    if unified_run:
        _set_unified_weight_reporter(out_dir)
        manifest_path = out_dir / "run_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                matchups_total = int(
                    manifest.get("matchups_requested")
                    or len(manifest.get("matchups_sampled") or [])
                    or len(matchups)
                )
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                pass
        write_unified_random_matchups_dashboard(out_dir, auto_refresh_seconds=5.0)

    shared_policy: Optional[PPOAgent] = None
    matchups_completed_count = 0
    talishar_pool = backend_pool or TalisharBackendPool.from_runtime(fallback_url=base_url)
    urls_before = list(talishar_pool.urls)
    failed_health = talishar_pool.health_check()
    talishar_pool = talishar_pool.filter_healthy()
    log_shard_pool(
        urls_before=urls_before,
        urls_after=list(talishar_pool.urls),
        failed_health=failed_health,
    )
    if safe_parallel:
        safe_workers, safe_parallel_matchups, safe_workers_per_matchup = (
            resolve_safe_parallel_limits(
                workers=n_workers,
                parallel_matchups=parallel_matchups,
                n_training_shards=len(talishar_pool.urls),
            )
        )
        if (
            safe_workers != n_workers
            or safe_parallel_matchups != parallel_matchups
        ):
            before_slots = concurrent_training_game_slots(
                n_workers, parallel_matchups
            )
            after_slots = concurrent_training_game_slots(
                safe_workers, safe_parallel_matchups
            )
            print(
                f"  Safe parallel: capped concurrent games "
                f"{before_slots} → {after_slots} "
                f"(≤ {len(talishar_pool.urls)} shard(s)); "
                f"parallel_matchups {parallel_matchups} → {safe_parallel_matchups}, "
                f"workers/matchup "
                f"{workers_per_parallel_matchup(n_workers, parallel_matchups)} "
                f"→ {safe_workers_per_matchup}",
                flush=True,
            )
            unified_debug_event(
                "connection",
                "Safe parallel capped worker budget",
                before_slots=before_slots,
                after_slots=after_slots,
                parallel_matchups_before=parallel_matchups,
                parallel_matchups_after=safe_parallel_matchups,
                workers_before=n_workers,
                workers_after=safe_workers,
                training_shards=len(talishar_pool.urls),
            )
        n_workers = safe_workers
        parallel_matchups = safe_parallel_matchups
    if len(talishar_pool.urls) > 1:
        print(f"  Talishar backends: {talishar_pool.format_log_label()}")

    for batch_start in range(0, len(matchups), parallel_matchups):
        batch = matchups[batch_start : batch_start + parallel_matchups]
        batch_index = batch_start // parallel_matchups + 1
        workers_per_matchup = workers_per_parallel_matchup(n_workers, len(batch))
        use_parallel_batch = parallel_matchups > 1
        shared_buffer: Optional[UnifiedSharedExperienceBuffer] = None
        batch_failed: list[str] = []

        if unified_run and use_parallel_batch:
            active_matchups = {
                _resolve_matchup_subdir(out_dir, matchup): {
                    "name": matchup.name,
                    "episodes_completed": 0,
                    "p1_win_rate": None,
                    "p2_win_rate": None,
                    "status": "training",
                }
                for matchup in batch
            }
            update_unified_training_live(
                out_dir,
                matchups_total=matchups_total,
                matchups_completed=matchups_completed_count,
                parallel_matchups=len(batch),
                batch_index=batch_index,
                target_episodes=n_episodes,
                active_matchups=active_matchups,
                status="training",
            )
            write_unified_random_matchups_dashboard(out_dir, auto_refresh_seconds=5.0)

        if use_parallel_batch:
            shared_buffer = UnifiedSharedExperienceBuffer()
            if shared_policy is None and batch:
                bootstrap_matchup = batch[0]
                probe_env = make_env(
                    bootstrap_matchup,
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
                    shared_policy = _bootstrap_unified_policy(
                        cache_store, probe_env, seed,
                    ).policy
                    if unified_run:
                        _record_initial_unified_policy_weights(shared_policy, out_dir)
                finally:
                    try:
                        probe_env.close()
                    except Exception:
                        pass
            if shared_policy is not None:
                shared_buffer.bind_policy(shared_policy)
            print(
                f"\n  Parallel batch {batch_index}: "
                f"{len(batch)} matchup(s), "
                f"{workers_per_matchup} worker(s)/matchup",
                flush=True,
            )
        elif unified_run and checkpoint_eval_episodes > 0 and shared_policy is None and batch:
            bootstrap_matchup = batch[0]
            probe_env = make_env(
                bootstrap_matchup,
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
                shared_policy = _bootstrap_unified_policy(
                    cache_store, probe_env, seed,
                ).policy
                if unified_run:
                    _record_initial_unified_policy_weights(shared_policy, out_dir)
            finally:
                try:
                    probe_env.close()
                except Exception:
                    pass

        persist_payloads: list[dict[str, Any]] = []
        persist_lock = threading.Lock()
        checkpoint_coordinator: Optional[Any] = None
        effective_ckpt_interval = resolve_checkpoint_interval(
            n_episodes,
            checkpoint_interval=checkpoint_interval,
            checkpoint_interval_pct=checkpoint_interval_pct,
        )
        if unified_run and checkpoint_eval_episodes > 0:
            from unified_checkpoint_eval import UnifiedCheckpointCoordinator  # noqa: PLC0415
            from flesh_and_blood_rlbridge.talishar_backend_pool import (  # noqa: PLC0415
                build_eval_backend_pool,
                resolve_eval_backend_url,
            )
            from unified_logic_baseline import (  # noqa: PLC0415
                submit_batch_logic_vs_logic_baselines,
            )
            from unified_checkpoint_eval import allocate_eval_episodes  # noqa: PLC0415

            batch_matchups = {
                _resolve_matchup_subdir(out_dir, matchup): matchup
                for matchup in batch
            }

            try:
                eval_backend_pool = build_eval_backend_pool(
                    fallback_url=base_url,
                    game_format=game_format,
                )
                eval_base_url = eval_backend_pool.primary_url
            except RuntimeError as exc:
                print(
                    f"  WARNING: eval backend probe failed ({exc!r}); "
                    "using configured eval URL",
                    flush=True,
                )
                eval_backend_pool = None
                eval_base_url = resolve_eval_backend_url(fallback_url=base_url)

            def _policy_snapshot_for_batch() -> tuple[PPOAgent, PPOAgent]:
                if shared_buffer is not None:
                    snap = shared_buffer.clone_policy_snapshot()
                    return snap, snap
                from agent_cache import clone_agent_weights  # noqa: PLC0415

                if shared_policy is None:
                    raise RuntimeError("Merged checkpoint eval requires a shared policy")
                snap = PPOAgent()
                clone_agent_weights(shared_policy, snap)
                return snap, snap

            if shared_policy is not None:
                checkpoint_coordinator = UnifiedCheckpointCoordinator(
                    out_dir=out_dir,
                    matchups=batch_matchups,
                    base_url=eval_base_url,
                    game_format=game_format,
                    max_steps=max_steps,
                    n_episodes=n_episodes,
                    warmup_episodes=warmup_episodes,
                    checkpoint_interval=effective_ckpt_interval,
                    checkpoint_eval_episodes=checkpoint_eval_episodes,
                    unified_policy=shared_policy,
                    policy_snapshot_fn=_policy_snapshot_for_batch,
                    seed=seed,
                    eval_backend_pool=eval_backend_pool,
                    eval_agent_vs_logic=checkpoint_eval_agent_vs_logic,
                    eval_logic_vs_logic=checkpoint_eval_logic_vs_logic,
                )
                print(
                    f"  Merged checkpoint eval: every {effective_ckpt_interval} episode(s) "
                    f"when all {len(batch)} matchup(s) reach bucket; "
                    f"{checkpoint_eval_episodes} self-play eval game(s) total across matchups "
                    f"(async, non-blocking; eval shard {eval_base_url})"
                )

            baseline_alloc = allocate_eval_episodes(
                checkpoint_eval_episodes,
                list(batch_matchups.keys()),
                seed=seed,
            )
            baseline_eps_example = next(
                (count for count in baseline_alloc.values() if count > 0),
                0,
            )
            baseline_queued = 0
            if checkpoint_eval_logic_vs_logic:
                baseline_queued = submit_batch_logic_vs_logic_baselines(
                    batch_matchups,
                    out_dir=out_dir,
                    base_url=eval_base_url,
                    game_format=game_format,
                    max_steps=max_steps,
                    total_episodes=checkpoint_eval_episodes,
                    seed=(seed + 50_000) if seed is not None else None,
                )
            if baseline_queued > 0:
                print(
                    f"  Logic win% vs logic baseline: {baseline_queued} matchup(s), "
                    f"~{baseline_eps_example} game(s) each "
                    f"({checkpoint_eval_episodes} total, async on eval shard)",
                    flush=True,
                )
            elif baseline_alloc:
                print(
                    "  Logic win% vs logic baseline: already on disk for this batch",
                    flush=True,
                )

        def _train_one(matchup: Matchup) -> dict[str, Any]:
            nonlocal shared_policy
            if unified_run:
                matchup_dir = _resolve_matchup_subdir(out_dir, matchup)
                audit_matchup_decks(
                    matchup,
                    game_format=game_format,
                    matchup_dir=matchup_dir,
                )
                unified_debug_event(
                    "matchup_load",
                    f"Training matchup {matchup.name}",
                    matchup=matchup.name,
                    matchup_dir=str(matchup_dir),
                    workers=workers_per_matchup if use_parallel_batch else n_workers,
                )
            train_kwargs: dict[str, Any] = {}
            if not skip_converged:
                train_kwargs["_force_train"] = True
            if unified_run:
                train_kwargs["_unified_matchups_completed"] = matchups_completed_count
                train_kwargs["_unified_matchups_total"] = matchups_total
            if use_parallel_batch and shared_buffer is not None:
                train_kwargs["_shared_policy"] = shared_policy
                train_kwargs["_shared_buffer"] = shared_buffer
                train_kwargs["_skip_persist"] = True
                train_kwargs["_matchup_live_key"] = _resolve_matchup_subdir(
                    out_dir, matchup
                )
            if checkpoint_coordinator is not None:
                train_kwargs["_checkpoint_coordinator"] = checkpoint_coordinator
                if "_matchup_live_key" not in train_kwargs:
                    train_kwargs["_matchup_live_key"] = _resolve_matchup_subdir(
                        out_dir, matchup
                    )
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
                n_workers=workers_per_matchup if use_parallel_batch else n_workers,
                parallel_seeds=parallel_seeds,
                checkpoint_interval_pct=checkpoint_interval_pct,
                checkpoint_interval=checkpoint_interval,
                checkpoint_eval_episodes=checkpoint_eval_episodes,
                checkpoint_eval_logic_vs_logic=checkpoint_eval_logic_vs_logic,
                checkpoint_eval_agent_vs_logic=checkpoint_eval_agent_vs_logic,
                build_cpp_engine=build_cpp_engine,
                require_cpp_engine=require_cpp_engine,
                rollout_mode=rollout_mode,
                rollout_processes=rollout_processes,
                backend_pool=talishar_pool,
                **train_kwargs,
            )
            payload = meta.get("_persist_summary")
            if isinstance(payload, dict):
                with persist_lock:
                    persist_payloads.append(payload)
            return meta

        if use_parallel_batch:
            batch_summary, batch_failed = run_parallel_matchup_batch(
                batch,
                train_fn=_train_one,
                label=f"batch {batch_index}",
            )
            summary.extend(batch_summary)
            failed.extend(batch_failed)
            if shared_buffer is not None and shared_policy is not None:
                shared_buffer.flush_remaining(shared_policy)
            if checkpoint_coordinator is not None:
                from checkpoint_eval_async import wait_for_checkpoint_evals  # noqa: PLC0415

                wait_for_checkpoint_evals()
                for payload in persist_payloads:
                    matchup_dir = str(payload.get("matchup_dir") or "")
                    if not matchup_dir:
                        continue
                    wr = checkpoint_coordinator.get_matchup_checkpoint_win_rate(
                        matchup_dir
                    )
                    if wr is not None:
                        payload["checkpoint_eval_win_rate"] = wr
                        payload["final_checkpoint_win_rate"] = wr
                    hist = checkpoint_coordinator.get_matchup_checkpoint_history(
                        matchup_dir
                    )
                    if hist:
                        stats = payload.get("training_stats")
                        if not isinstance(stats, dict):
                            stats = {}
                        stats["checkpoint_eval_history"] = hist
                        payload["training_stats"] = stats
            if shared_policy is not None and persist_payloads:
                persist_batch_to_cache(
                    cache_store,
                    shared_policy,
                    persist_payloads,
                )
        else:
            for matchup in batch:
                try:
                    meta = _train_one(matchup)
                    summary.append(meta)
                    if not use_parallel_batch and shared_policy is None:
                        loaded = cache_store.load_if_exists()
                        if loaded is not None:
                            shared_policy = loaded
                except Exception as exc:
                    print(f"\n  ERROR training {matchup.name}: {exc}")
                    failed.append(matchup.name)

        if use_parallel_batch:
            matchups_completed_count += len(batch_summary)
        else:
            matchups_completed_count += sum(
                1 for m in batch if m.name not in failed
            )

        if unified_run:
            update_unified_training_live(
                out_dir,
                matchups_completed=matchups_completed_count,
                matchups_total=matchups_total,
                parallel_matchups=len(batch),
                batch_index=batch_index,
                active_matchups={},
                status=(
                    "complete"
                    if matchups_completed_count >= matchups_total
                    else "between_matchups"
                ),
            )
            write_unified_random_matchups_dashboard(
                out_dir,
                auto_refresh_seconds=5.0,
            )

    from checkpoint_eval_async import shutdown_checkpoint_eval_executor  # noqa: PLC0415

    shutdown_checkpoint_eval_executor(wait=True)
    if unified_run:
        _set_unified_weight_reporter(None)

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
    from flesh_and_blood_rlbridge.talishar_deck_assets import (  # noqa: PLC0415
        build_assets_equipment_headers,
    )

    return build_assets_equipment_headers(assets_path)


def _get_hero_header(hero_id: str, assets_map: dict[str, str]) -> str:
    base = hero_id.removeprefix("hero_")
    return assets_map.get(base) or assets_map.get(hero_id) or base


def _hero_talishar_id(hero_slug: str) -> str:
    return hero_slug.replace("-", "_").strip()


def fabrary_sideboard_card_pool(
    deck_entry: dict,
    format_name: str,
    assets_path: Path,
) -> dict[str, int]:
    """Registered card pool for guide sideboarding (prefers full SAGE precon when available)."""
    from flesh_and_blood_rlbridge.deck_context import _read_asset_deck  # noqa: PLC0415
    from flesh_and_blood_rlbridge.talishar_deck_assets import (  # noqa: PLC0415
        SAGE_PRECON_BY_HERO,
        resolve_talishar_deck_stem,
    )

    hero_token = str(deck_entry.get("hero_id", "")).removeprefix("hero_")
    deck_id = str(deck_entry.get("id", ""))
    candidates = [
        SAGE_PRECON_BY_HERO.get(hero_token),
        resolve_talishar_deck_stem(assets_path, deck_id),
        deck_id,
    ]
    best: dict[str, int] = {}
    for stem in candidates:
        if not stem:
            continue
        try:
            _, counts = _read_asset_deck(assets_path, stem)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        normalized = {str(k): int(v) for k, v in counts.items() if int(v) > 0}
        if sum(normalized.values()) > sum(best.values()):
            best = normalized
    if best:
        return best

    card_ids = resolve_fabrary_deck_cards(deck_entry, format_name)
    pool: dict[str, int] = {}
    for cid in card_ids:
        pool[cid] = pool.get(cid, 0) + 1
    return pool


def resolve_fabrary_equipment_header(deck_entry: dict, assets_path: Path) -> str:
    from flesh_and_blood_rlbridge.talishar_deck_assets import (  # noqa: PLC0415
        resolve_matchup_equipment_header,
    )

    hero_id = str(deck_entry.get("hero_id", "")).removeprefix("hero_")
    explicit = str(deck_entry.get("equipment_header", "") or "").strip()
    deck_stem = str(deck_entry.get("id", "") or deck_entry.get("deck_id", "") or "").strip()
    return resolve_matchup_equipment_header(
        role="p1",
        hero_id=hero_id,
        deck_stem=deck_stem,
        assets_dir=assets_path,
        fallback=explicit or hero_id,
    )


def apply_unified_matchup_sideboards(
    matchup: Matchup,
    *,
    game_format: str,
    base_url: str,
    out_dir: Path,
    max_sideboard_steps: int = 100,
) -> None:
    """Run guide sideboard for both seats and write Talishar asset decks."""
    from flesh_and_blood_rlbridge.opponent_deck import hero_class_for_id  # noqa: PLC0415

    from train_pipeline_common import (  # noqa: PLC0415
        PhaseAgents,
        _write_deck_file,
        apply_guide_sideboard_for_matchup,
        greedy_game_deck_cut,
    )

    entry1 = matchup.p1_fabrary_entry
    entry2 = matchup.p2_fabrary_entry
    if not entry1 or not entry2:
        return

    assets_path = talishar_assets_path()
    p1_hero = _hero_talishar_id(matchup.p1_hero)
    p2_hero = _hero_talishar_id(matchup.p2_hero)
    p1_equipment = resolve_fabrary_equipment_header(entry1, assets_path)
    p2_equipment = resolve_fabrary_equipment_header(entry2, assets_path)
    from flesh_and_blood_rlbridge.talishar_deck_assets import (  # noqa: PLC0415
        resolve_matchup_equipment_header,
    )

    p1_equipment = resolve_matchup_equipment_header(
        role="p1",
        hero_id=p1_hero,
        deck_stem=matchup.p1_deck,
        assets_dir=assets_path,
        fallback=p1_equipment,
    )
    p2_equipment = resolve_matchup_equipment_header(
        role="p2",
        hero_id=p2_hero,
        deck_stem=matchup.p2_deck,
        assets_dir=assets_path,
        fallback=p2_equipment,
    )
    p1_pool = fabrary_sideboard_card_pool(entry1, game_format, assets_path)
    p2_pool = fabrary_sideboard_card_pool(entry2, game_format, assets_path)

    print(
        f"\n{'=' * 60}\n"
        f"  Guide sideboard (pre-match)  {matchup.name}\n"
        f"  P1 pool: {sum(p1_pool.values())} cards  |  "
        f"P2 pool: {sum(p2_pool.values())} cards\n"
        f"{'=' * 60}"
    )

    p1_agents = PhaseAgents(player="p1", card_pool=p1_pool)
    p2_agents = PhaseAgents(player="p2", card_pool=p2_pool)
    assets_str = str(assets_path)

    apply_guide_sideboard_for_matchup(
        p1_agents,
        [p2_hero],
        hero_id=p1_hero,
        hero_class=hero_class_for_id(p1_hero),
        equipment_header=p1_equipment,
        game_format=game_format,
        opponent_deck_name=matchup.p2_deck,
        max_sideboard_steps=max_sideboard_steps,
        assets_path=assets_str,
        base_url=base_url,
        cpp_engine_dir=matchup.cpp_engine_dir,
    )
    apply_guide_sideboard_for_matchup(
        p2_agents,
        [p1_hero],
        hero_id=p2_hero,
        hero_class=hero_class_for_id(p2_hero),
        equipment_header=p2_equipment,
        game_format=game_format,
        opponent_deck_name=matchup.p1_deck,
        max_sideboard_steps=max_sideboard_steps,
        assets_path=assets_str,
        base_url=base_url,
        cpp_engine_dir=matchup.cpp_engine_dir,
    )

    min_size = FORMAT_DECK_RULES.get(game_format, FORMAT_DECK_RULES["silver_age"])[
        "deck_size"
    ]
    p1_game = p1_agents.active_decks.get(p2_hero) or greedy_game_deck_cut(
        p1_pool, min_size
    )
    p2_game = p2_agents.active_decks.get(p1_hero) or greedy_game_deck_cut(
        p2_pool, min_size
    )

    _write_deck_file(p1_game, p1_equipment, matchup.p1_deck, assets_str)
    _write_deck_file(p2_game, p2_equipment, matchup.p2_deck, assets_str)

    sideboard_record = {
        "matchup": matchup.name,
        "p1_hero": p1_hero,
        "p2_hero": p2_hero,
        "p1_equipment_header": p1_equipment,
        "p2_equipment_header": p2_equipment,
        "p1_pool_size": sum(p1_pool.values()),
        "p2_pool_size": sum(p2_pool.values()),
        "p1_game_deck_size": sum(p1_game.values()),
        "p2_game_deck_size": sum(p2_game.values()),
    }
    record_path = matchup_out_dir(out_dir, matchup) / "guide_sideboard.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(sideboard_record, indent=2), encoding="utf-8")


def resolve_fabrary_deck_cards(deck_entry: dict, format_name: str) -> list[str]:
    rules = FORMAT_DECK_RULES[format_name]
    max_copies = rules["max_copies"]
    deck_size = rules["deck_size"]

    card_ids_field = deck_entry.get("card_ids")
    if card_ids_field:
        result: list[str] = []
        for card_entry in card_ids_field:
            card_id = str(card_entry.get("id", "")).strip()
            if not card_id:
                continue
            count = min(int(card_entry.get("count", 1)), max_copies)
            result.extend([card_id] * count)
        if len(result) > deck_size:
            result = result[:deck_size]
        return result

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


def _secondary_talishar_deck_paths(deck_id: str) -> list[Path]:
    """Extra on-disk deck sources checked before name-based fabrary resolution."""
    return [REPO_ROOT / "assets" / "talishar_decks" / f"{deck_id}.txt"]


def _resolve_playable_deck_asset(
    deck_id: str,
    assets_path: Path,
) -> tuple[str, list[str]] | None:
    """Return header/cards from the best existing playable deck asset for *deck_id*."""
    from flesh_and_blood_rlbridge.talishar_deck_assets import (  # noqa: PLC0415
        deck_asset_is_playable,
        read_talishar_deck_asset,
        resolve_canonical_sage_precon_stem,
    )

    candidates: list[Path] = []
    canonical = resolve_canonical_sage_precon_stem(deck_id)
    if canonical:
        candidates.append(assets_path / f"{canonical}.txt")
    candidates.append(assets_path / f"{deck_id}.txt")
    candidates.extend(_secondary_talishar_deck_paths(deck_id))

    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        header, cards = read_talishar_deck_asset(path)
        if deck_asset_is_playable(header, cards):
            return header, cards
    return None


def _write_talishar_deck_asset(
    out_path: Path,
    *,
    hero_id: str,
    deck_stem: str,
    assets_path: Path,
    equipment_header: str,
    card_ids: list[str],
) -> None:
    from flesh_and_blood_rlbridge.talishar_deck_assets import (  # noqa: PLC0415
        ensure_full_equipment_header,
    )

    hero = hero_id.removeprefix("hero_")
    header_text = (equipment_header or "").strip()
    if len(header_text.split()) < 8:
        header_text = ensure_full_equipment_header(
            hero,
            header_text,
            assets_path,
            deck_stem=deck_stem,
        )
    out_path.write_text(f"{header_text}\n{' '.join(card_ids)}\n", encoding="utf-8")


def write_fabrary_deck_file(
    deck_entry: dict,
    assets_path: Path,
    assets_map: dict[str, str],
    format_name: str,
) -> Optional[str]:
    deck_id = str(deck_entry["id"])
    hero_id = str(deck_entry.get("hero_id", ""))
    out_path = assets_path / f"{deck_id}.txt"

    playable = _resolve_playable_deck_asset(deck_id, assets_path)
    if playable is not None:
        header, card_ids = playable
        header = str(deck_entry.get("equipment_header") or header).strip() or header
        _write_talishar_deck_asset(
            out_path,
            hero_id=hero_id,
            deck_stem=deck_id,
            assets_path=assets_path,
            equipment_header=header,
            card_ids=card_ids,
        )
        return deck_id

    from flesh_and_blood_rlbridge.talishar_deck_assets import (  # noqa: PLC0415
        resolve_equipment_header_line,
    )

    hero_header = str(deck_entry.get("equipment_header") or "").strip()
    if not hero_header:
        hero_header = resolve_equipment_header_line(
            hero_id.removeprefix("hero_"),
            assets_path,
            fallback=_get_hero_header(hero_id, assets_map),
        )
    card_ids = resolve_fabrary_deck_cards(deck_entry, format_name)
    min_size = FORMAT_DECK_RULES.get(format_name, FORMAT_DECK_RULES["silver_age"])[
        "deck_size"
    ]
    if not card_ids:
        print(f"  Error: deck {deck_id} resolved to zero cards; skipping.")
        return None
    if len(card_ids) < min_size:
        print(
            f"  Error: deck {deck_id} resolved to {len(card_ids)} cards "
            f"(need {min_size}); skipping."
        )
        return None
    _write_talishar_deck_asset(
        out_path,
        hero_id=hero_id,
        deck_stem=deck_id,
        assets_path=assets_path,
        equipment_header=hero_header,
        card_ids=card_ids,
    )
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
                    p1_fabrary_entry=entry1,
                    p2_fabrary_entry=entry2,
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
    name = f"{slug1}-vs-{slug2}"
    return Matchup(
        name=name,
        p1_deck=stem1,
        p2_deck=stem2,
        dir_name=shorten_matchup_dir_name(name, stem1, stem2),
        description=(
            f"{entry1.get('name', slug1)} (P1) vs "
            f"{entry2.get('name', slug2)} (P2) — unified self-play"
        ),
        tags=[slug1, slug2, format_name],
        p1_hero=hero_slug(str(entry1.get("hero_id", slug1))),
        p2_hero=hero_slug(str(entry2.get("hero_id", slug2))),
        p1_fabrary_entry=entry1,
        p2_fabrary_entry=entry2,
    )


def sample_random_fabrary_matchups(
    decks: list[tuple[str, str, dict]],
    count: int,
    rng: Any,
    format_name: str,
    *,
    unique_pairs: bool = True,
    fabrary_weighted_heroes: bool = False,
) -> list[Matchup]:
    """Sample random deck-vs-deck matchups from a fabrary deck pool."""
    if len(decks) < 2:
        raise ValueError("need at least two decks to sample matchups")
    if count <= 0:
        return []

    deck_weights: list[float] | None = None
    if fabrary_weighted_heroes:
        from flesh_and_blood_rlbridge.card_db.fabrary_meta import (  # noqa: PLC0415
            DEFAULT_PERIOD,
            DEFAULT_SAMPLING_GAMES,
            deck_hero_play_weight,
            hero_play_counts,
        )

        hero_counts = hero_play_counts(
            format_name,
            games=DEFAULT_SAMPLING_GAMES,
            period=DEFAULT_PERIOD,
        )
        deck_weights = [
            float(deck_hero_play_weight(entry, hero_counts))
            for _slug, _stem, entry in decks
        ]

    def _pick_index(available: list[int]) -> int:
        if deck_weights is None:
            return rng.choice(available)
        weights = [deck_weights[i] for i in available]
        return rng.choices(available, weights=weights, k=1)[0]

    matchups: list[Matchup] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = max(count * 30, 30)
    indices = list(range(len(decks)))

    while len(matchups) < count and attempts < max_attempts:
        attempts += 1
        available = list(indices)
        i = _pick_index(available)
        available.remove(i)
        j = _pick_index(available)
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
