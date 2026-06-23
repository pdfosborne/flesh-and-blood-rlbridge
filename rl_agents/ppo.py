"""
Proximal Policy Optimisation (PPO) Agent
=========================================
A from-scratch NumPy implementation of PPO-Clip for discrete action spaces.

Network architecture
--------------------
Actor and critic share two hidden layers; separate output heads::

    obs_dim → hidden_size → hidden_size → n_actions   (actor head)
    obs_dim → hidden_size → hidden_size → 1            (critic head)

Rollout inference uses one shared feature pass via
:meth:`PPOAgent.predict_policy_value`.

Algorithm  (PPO-Clip, Schulman et al. 2017)
-------------------------------------------
For each iteration:

1. Collect ``n_steps`` transitions following the current policy π_θ_old.
   At each step:
   - Sample action from softmax(actor(obs)).
   - Store ``(obs, action, reward, done, log_prob_old, value_old)``.

2. Compute returns and advantages using Generalised Advantage Estimation
   (GAE-λ)::

       δ_t  = r_t + γ · V(s_{t+1}) · (1−done) − V(s_t)
       A_t  = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}
       G_t  = A_t + V(s_t)   (return / value target)

3. For ``ppo_epochs`` passes over the collected data in random mini-batches:
   a. Recompute log-probabilities log π_θ(a|s) and value estimates V_θ(s).
   b. Probability ratio: ``r = exp(log π_θ − log π_θ_old)``
   c. Clipped actor loss::

          L_clip = −E[min(r·A, clip(r, 1−ε, 1+ε)·A)]

   d. Critic loss: ``L_vf = MSE(V_θ(s), G)``
   e. Entropy bonus: ``−H[π_θ(·|s)]`` (encourages exploration).
   f. Total loss: ``L_clip + c_vf · L_vf − c_ent · H``

4. Update actor and critic with one SGD step per mini-batch.

References
----------
Schulman et al. (2017), "Proximal Policy Optimization Algorithms",
arXiv:1707.06347.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import copy

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    raise ImportError("PPO agent requires PyTorch. Install with: pip install torch")

from ._agent_base import (
    AgentBase,
    TrainResult,
    _discrete_n,
    _flat_obs,
    _get,
    _infer_action_capacity,
    _n_legal_of,
    _to_env_action,
)

# Use GPU when available; float32 for throughput during training rollouts.
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_TORCH_DTYPE = torch.float32


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PPOTrainResult(TrainResult):
    """Training summary for :class:`PPOAgent`."""

    obs_dim: int = 0
    mean_actor_loss: float = 0.0
    mean_critic_loss: float = 0.0
    n_iterations: int = 0

    def __str__(self) -> str:
        return (
            f"PPOTrainResult  episodes={self.n_episodes}"
            f"  obs_dim={self.obs_dim}"
            f"  mean_reward={self.mean_reward:.4f}"
            f"  best_reward={self.best_reward:.4f}"
            f"  mean_actor_loss={self.mean_actor_loss:.6f}"
            f"  mean_critic_loss={self.mean_critic_loss:.6f}"
        )


# ── Shared-trunk policy / value network (one feature pass at rollout time) ───

def _linear_to_legacy(
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    layer: str,
) -> dict[str, list[list[float]]]:
    w = weight.detach().cpu().numpy().T
    b = bias.detach().cpu().numpy()
    return {f"W{layer}": w.tolist(), f"b{layer}": b.tolist()}


def _legacy_to_linear(
    d: dict[str, Any],
    *,
    layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    w = np.array(d[f"W{layer}"], dtype=np.float64).T
    b = np.array(d[f"b{layer}"], dtype=np.float64)
    return w, b


class _SharedPolicyValue(nn.Module):
    """Two hidden layers shared by actor and critic output heads."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        n_actions: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.fc1 = nn.Linear(in_dim, hidden, dtype=_TORCH_DTYPE)
        self.fc2 = nn.Linear(hidden, hidden, dtype=_TORCH_DTYPE)
        self.actor_head = nn.Linear(hidden, n_actions, dtype=_TORCH_DTYPE)
        self.critic_head = nn.Linear(hidden, 1, dtype=_TORCH_DTYPE)
        for module in (self.fc1, self.fc2, self.actor_head, self.critic_head):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))

    def predict_policy_value(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Single feature pass; returns logits ``(B, A)`` and values ``(B, 1)``."""
        with torch.no_grad():
            t = torch.as_tensor(x, dtype=_TORCH_DTYPE, device=_DEVICE)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            hidden = self.features(t)
            logits = self.actor_head(hidden).cpu().numpy()
            values = self.critic_head(hidden).cpu().numpy()
        return logits, values

    def actor_dict(self) -> dict[str, Any]:
        out = {}
        out.update(_linear_to_legacy(self.fc1.weight, self.fc1.bias, layer="1"))
        out.update(_linear_to_legacy(self.fc2.weight, self.fc2.bias, layer="2"))
        out.update(_linear_to_legacy(self.actor_head.weight, self.actor_head.bias, layer="3"))
        return out

    def critic_dict(self) -> dict[str, Any]:
        out = {}
        out.update(_linear_to_legacy(self.fc1.weight, self.fc1.bias, layer="1"))
        out.update(_linear_to_legacy(self.fc2.weight, self.fc2.bias, layer="2"))
        out.update(_linear_to_legacy(self.critic_head.weight, self.critic_head.bias, layer="3"))
        return out

    def load_actor_dict(self, d: dict[str, Any]) -> None:
        for layer, attr in ((1, "fc1"), (2, "fc2")):
            w, b = _legacy_to_linear(d, layer=layer)
            linear = getattr(self, attr)
            linear.weight.data = torch.tensor(w, dtype=_TORCH_DTYPE, device=_DEVICE)
            linear.bias.data = torch.tensor(b, dtype=_TORCH_DTYPE, device=_DEVICE)
        w, b = _legacy_to_linear(d, layer=3)
        self.actor_head.weight.data = torch.tensor(w, dtype=_TORCH_DTYPE, device=_DEVICE)
        self.actor_head.bias.data = torch.tensor(b, dtype=_TORCH_DTYPE, device=_DEVICE)

    def load_critic_head_dict(self, d: dict[str, Any]) -> None:
        w, b = _legacy_to_linear(d, layer=3)
        self.critic_head.weight.data = torch.tensor(w, dtype=_TORCH_DTYPE, device=_DEVICE)
        self.critic_head.bias.data = torch.tensor(b, dtype=_TORCH_DTYPE, device=_DEVICE)


class _ActorForward(nn.Module):
    def __init__(self, shared: _SharedPolicyValue) -> None:
        super().__init__()
        self._shared = shared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._shared.actor_head(self._shared.features(x))


class _CriticForward(nn.Module):
    def __init__(self, shared: _SharedPolicyValue) -> None:
        super().__init__()
        self._shared = shared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._shared.critic_head(self._shared.features(x))


class _MLPShim:
    """Actor/critic view over :class:`_SharedPolicyValue` for optimisers and legacy APIs."""

    def __init__(self, shared: _SharedPolicyValue, role: str, lr: float) -> None:
        if role not in ("actor", "critic"):
            raise ValueError(f"unknown role: {role!r}")
        self._shared = shared
        self._role = role
        self.lr = lr
        if role == "actor":
            self._net = _ActorForward(shared).to(_DEVICE)
            params = list(shared.fc1.parameters()) + list(shared.fc2.parameters())
            params += list(shared.actor_head.parameters())
        else:
            self._net = _CriticForward(shared).to(_DEVICE)
            params = list(shared.fc1.parameters()) + list(shared.fc2.parameters())
            params += list(shared.critic_head.parameters())
        self._opt = torch.optim.SGD(params, lr=lr)
        self._out_t: Optional[torch.Tensor] = None
        self._out: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(x, dtype=_TORCH_DTYPE, device=_DEVICE)
        self._out_t = self._net(t)
        self._out = self._out_t.detach().cpu().numpy()
        return self._out

    def predict(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(x, dtype=_TORCH_DTYPE, device=_DEVICE)
            return self._net(t).cpu().numpy()

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(x, dtype=_TORCH_DTYPE, device=_DEVICE)
            return self._net(t).cpu().numpy()

    def backward(self, loss_grad: np.ndarray) -> None:
        grad_t = torch.as_tensor(loss_grad, dtype=_TORCH_DTYPE, device=_DEVICE)
        self._opt.zero_grad()
        assert self._out_t is not None
        self._out_t.backward(grad_t)
        self._opt.step()

    def to_dict(self) -> dict[str, Any]:
        if self._role == "actor":
            return self._shared.actor_dict()
        return self._shared.critic_dict()

    def from_dict(self, d: dict[str, Any]) -> None:
        if self._role == "actor":
            self._shared.load_actor_dict(d)
        else:
            self._shared.load_critic_head_dict(d)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.where(np.isfinite(x), x, 0.0)  # replace NaN/inf before softmax
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    s = e.sum(axis=-1, keepdims=True)
    # If sum is zero (all -inf inputs), return uniform
    s = np.where(s == 0, 1.0, s)
    return e / s


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.where(np.isfinite(x), x, 0.0)  # replace NaN/inf before log-softmax
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def _gae(
    rewards: np.ndarray,     # (T,)
    values: np.ndarray,      # (T,)
    next_values: np.ndarray, # (T,)
    dones: np.ndarray,       # (T,)
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generalised Advantage Estimation.

    Returns advantages and returns (value targets), both shape (T,).
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    last_gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * (1.0 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# ── Agent ─────────────────────────────────────────────────────────────────────

class PPOAgent(AgentBase):
    """
    Proximal Policy Optimisation agent (pure NumPy, no framework required).

    Parameters
    ----------
    n_actions:
        Discrete action-space size.  Pass 0 to auto-detect from the
        environment on the first ``train()`` call.
    obs_dim:
        Flat observation dimension.  Pass 0 to infer from the first step.
    hidden_size:
        Hidden-layer width for both actor and critic networks.
    lr_actor:
        SGD learning rate for the actor network.
    lr_critic:
        SGD learning rate for the critic network.
    gamma:
        Discount factor for returns.
    lam:
        GAE λ parameter (0 = TD(0), 1 = Monte-Carlo).
    clip_eps:
        PPO probability-ratio clipping coefficient (ε).
    c_vf:
        Weighting coefficient for the critic (value) loss.
    c_ent:
        Entropy bonus coefficient (encourages exploration).
    n_steps:
        Number of environment steps collected per PPO iteration.
    ppo_epochs:
        Number of optimisation passes over each collected batch.
    mini_batch_size:
        Mini-batch size for each gradient update within an epoch.
    seed:
        Random seed for weight initialisation and action sampling.
    """

    name = "ppo"

    def __init__(
        self,
        n_actions: int = 0,
        obs_dim: int = 0,
        *,
        hidden_size: int = 64,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        c_vf: float = 0.5,
        c_ent: float = 0.01,
        n_steps: int = 256,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        seed: Optional[int] = None,
    ) -> None:
        self.n_actions      = n_actions
        self.obs_dim        = obs_dim
        self.hidden_size    = hidden_size
        self.lr_actor       = lr_actor
        self.lr_critic      = lr_critic
        self.gamma          = gamma
        self.lam            = lam
        self.clip_eps       = clip_eps
        self.c_vf           = c_vf
        self.c_ent          = c_ent
        self.n_steps        = n_steps
        self.ppo_epochs     = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self._seed          = seed
        self._rng_py        = random.Random(seed)
        self._rng_np        = np.random.default_rng(seed)
        # Whether to restrict the policy to per-state legal actions (set for
        # variable / text action spaces; disabled for fixed Discrete spaces).
        self._mask_actions  = False

        self._actor:  Optional[_MLPShim] = None
        self._critic: Optional[_MLPShim] = None
        self._shared: Optional[_SharedPolicyValue] = None

    # ── Lazy init ─────────────────────────────────────────────────────────────

    def _init_nets(self, obs_dim: int) -> None:
        if self._shared is not None:
            return
        self.obs_dim = obs_dim
        self._shared = _SharedPolicyValue(
            obs_dim,
            self.hidden_size,
            self.n_actions,
            self._seed,
        ).to(_DEVICE)
        self._actor = _MLPShim(self._shared, "actor", self.lr_actor)
        self._critic = _MLPShim(self._shared, "critic", self.lr_critic)

    def _obs_to_vec(self, obs: Any) -> np.ndarray:
        """Flatten and coerce *obs* to a fixed-size vector.

        If obs dimensionality changes across steps (common for dict/list states),
        vectors are padded/truncated to the first observed dimension.
        """
        from flesh_and_blood_rlbridge.obs_encoding import observation_fingerprint

        vec = observation_fingerprint(obs, obs_dim=self.obs_dim if self.obs_dim > 0 else 0)
        if self.obs_dim <= 0:
            self.obs_dim = int(vec.shape[0])
        return vec

    # ── Public API ────────────────────────────────────────────────────────────

    def _masked_logits(self, logits: np.ndarray, obs: Any) -> np.ndarray:
        """Set logits of illegal actions to a large negative value.

        Restricts the policy to the ``[0, n_legal)`` indices that map onto the
        current state's legal actions. A no-op for fixed Discrete spaces or
        when the observation does not expose ``legal_actions``.
        """
        if not getattr(self, "_mask_actions", False):
            return logits
        n_legal = _n_legal_of(obs)
        if n_legal is None or n_legal <= 0 or n_legal >= self.n_actions:
            return logits
        masked = np.array(logits, dtype=np.float64, copy=True)
        masked[..., n_legal:] = -1e9
        return masked

    def predict_batch(self, obs_vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batched actor logits and critic values for rollout buffers."""
        if self._shared is None:
            n = len(obs_vecs)
            return np.zeros((n, self.n_actions)), np.zeros(n)
        batch = np.asarray(obs_vecs, dtype=np.float64)
        logits, values = self._shared.predict_policy_value(batch)
        return logits, values.reshape(-1)

    def predict_policy_value(
        self,
        obs_vec: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Fused actor logits and critic value for one observation vector."""
        if self._shared is None:
            return np.zeros(self.n_actions or 1, dtype=np.float64), 0.0
        x = np.asarray(obs_vec, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        logits, values = self._shared.predict_policy_value(x)
        return logits, float(values.reshape(-1)[0])

    def act(self, obs: Any) -> Any:
        """Sample an action from the current policy (stochastic).

        Returns the environment-ready action: for variable action spaces this
        is the selected ``legal_actions`` entry; otherwise an integer index.
        """
        if self._actor is None:
            return 0
        x = self._obs_to_vec(obs)
        logits = self._masked_logits(self._actor.predict(x), obs)
        probs = _softmax(logits)
        idx = int(self._rng_np.choice(self.n_actions, p=probs))
        return _to_env_action(obs, idx, getattr(self, "_mask_actions", False))

    def act_greedy(self, obs: Any) -> Any:
        """Return the most probable legal action (greedy / deterministic)."""
        if self._actor is None:
            return 0
        x = self._obs_to_vec(obs)
        logits = self._masked_logits(self._actor.predict(x), obs)
        idx = int(np.argmax(logits))
        return _to_env_action(obs, idx, getattr(self, "_mask_actions", False))

    def train(
        self,
        env: Any,
        n_episodes: int = 200,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ) -> PPOTrainResult:
        """
        Train PPO on *env*.

        Training is structured in iterations; each iteration collects
        ``n_steps`` environment transitions and then runs ``ppo_epochs``
        passes of mini-batch gradient updates.  Episodes are tracked for
        reporting.

        Parameters
        ----------
        env:
            Any rlbridge environment with a discrete action space.
        n_episodes:
            Approximate total number of episodes (used to set total steps
            as ``n_episodes × max_steps``; actual count may differ slightly
            since episodes are allowed to run to completion).
        max_steps:
            Hard per-episode step cap when collecting rollouts.
        seed:
            Base environment seed.

        Returns
        -------
        PPOTrainResult
        """
        if self.n_actions == 0:
            self.n_actions, self._mask_actions = _infer_action_capacity(env, seed=seed)
        else:
            self._mask_actions = _discrete_n(env) is None

        total_steps = n_episodes * max_steps
        episode_rewards: list[float] = []
        actor_losses:  list[float] = []
        critic_losses: list[float] = []
        completed_episodes = 0
        best_ep_reward  = float("-inf")
        best_ep_history: list[tuple] = []

        # Running episode state
        ep_seed = seed
        reset_out = env.reset(seed=ep_seed)
        obs = _get(reset_out, "observation", reset_out)
        obs_vec = np.array(_flat_obs(obs), dtype=np.float64)
        self._init_nets(obs_vec.shape[0])
        obs_vec = self._obs_to_vec(obs)
        current_ep_reward = 0.0
        current_ep_steps  = 0
        current_ep_history: list[tuple] = []
        global_step = 0

        while global_step < total_steps:
            # ── Rollout collection ─────────────────────────────────────────────
            rollout_obs:       list[np.ndarray] = []
            rollout_actions:   list[int]        = []
            rollout_rewards:   list[float]      = []
            rollout_dones:     list[float]      = []
            rollout_log_probs: list[float]      = []
            rollout_values:    list[float]      = []
            rollout_n_legal:   list[int]        = []

            for _ in range(self.n_steps):
                logits, value = self.predict_policy_value(obs_vec)
                logits = self._masked_logits(logits, obs)
                log_probs_all = _log_softmax(logits)[0]
                probs = _softmax(logits)[0]
                action = int(self._rng_np.choice(self.n_actions, p=probs))
                log_prob = float(log_probs_all[action])

                n_legal = _n_legal_of(obs)
                env_action = _to_env_action(obs, action, self._mask_actions)
                step_out = env.step(env_action)
                next_obs   = _get(step_out, "observation", obs)
                reward     = float(_get(step_out, "reward", 0.0))
                terminated = bool(_get(step_out, "terminated", False))
                truncated  = bool(_get(step_out, "truncated", False))
                done = terminated or truncated

                rollout_obs.append(obs_vec)
                rollout_actions.append(action)
                rollout_rewards.append(reward)
                rollout_dones.append(float(done))
                rollout_log_probs.append(log_prob)
                rollout_values.append(value)
                rollout_n_legal.append(n_legal if n_legal is not None else self.n_actions)
                current_ep_history.append((obs, env_action))

                current_ep_reward += reward
                current_ep_steps  += 1
                global_step += 1

                if done or current_ep_steps >= max_steps:
                    episode_rewards.append(current_ep_reward)
                    if current_ep_reward > best_ep_reward:
                        best_ep_reward  = current_ep_reward
                        best_ep_history = list(current_ep_history)
                    completed_episodes += 1
                    current_ep_reward   = 0.0
                    current_ep_steps    = 0
                    current_ep_history  = []
                    ep_seed = (seed + completed_episodes) if seed is not None else None
                    reset_out = env.reset(seed=ep_seed)
                    obs = _get(reset_out, "observation", reset_out)
                    obs_vec = self._obs_to_vec(obs)
                else:
                    obs = next_obs
                    obs_vec = self._obs_to_vec(next_obs)

                if global_step >= total_steps:
                    break

            T = len(rollout_obs)
            obs_arr     = np.array(rollout_obs,       dtype=np.float64)    # (T, D)
            act_arr     = np.array(rollout_actions,   dtype=np.int64)      # (T,)
            values_arr  = np.array(rollout_values,    dtype=np.float64)    # (T,)
            log_old_arr = np.array(rollout_log_probs, dtype=np.float64)    # (T,)
            dones_arr   = np.array(rollout_dones,     dtype=np.float64)    # (T,)
            nlegal_arr  = np.array(rollout_n_legal,   dtype=np.int64)      # (T,)

            # Bootstrap value of state after last rollout step
            next_val = float(self._critic.predict(obs_vec[None, :]).flatten()[0])  # type: ignore[union-attr]
            next_vals_arr = np.append(values_arr[1:], next_val)

            advantages, returns = _gae(
                np.array(rollout_rewards, dtype=np.float64),
                values_arr,
                next_vals_arr,
                dones_arr,
                self.gamma,
                self.lam,
            )
            # Normalise advantages
            if T > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ── PPO update epochs ──────────────────────────────────────────────
            indices = np.arange(T)
            for _ in range(self.ppo_epochs):
                self._rng_np.shuffle(indices)
                for start in range(0, T, self.mini_batch_size):
                    mb_idx = indices[start: start + self.mini_batch_size]
                    if len(mb_idx) == 0:
                        continue

                    mb_obs  = obs_arr[mb_idx]                   # (B, D)
                    mb_acts = act_arr[mb_idx]                   # (B,)
                    mb_adv  = advantages[mb_idx]                # (B,)
                    mb_ret  = returns[mb_idx]                   # (B,)
                    mb_lp_old = log_old_arr[mb_idx]             # (B,)
                    mb_nlegal = nlegal_arr[mb_idx]              # (B,)

                    # ── Actor forward ──────────────────────────────────────────
                    logits_new = self._actor.forward(mb_obs)    # type: ignore[union-attr]
                    B = mb_obs.shape[0]
                    legal_mask: Optional[np.ndarray] = None
                    if self._mask_actions:
                        # Same per-state legal masking used during rollout, so
                        # the probability ratio and entropy stay consistent.
                        legal_mask = np.arange(self.n_actions)[None, :] < mb_nlegal[:, None]  # (B, A)
                        logits_new = np.where(legal_mask, logits_new, -1e9)
                    log_probs_new = _log_softmax(logits_new)    # (B, A)
                    probs_new = _softmax(logits_new)            # (B, A)
                    lp_new = log_probs_new[np.arange(B), mb_acts]  # (B,)

                    # ── PPO-clip actor loss ────────────────────────────────────
                    ratio   = np.exp(lp_new - mb_lp_old)
                    clip_r  = np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    actor_loss = -np.minimum(ratio * mb_adv, clip_r * mb_adv).mean()

                    # Entropy bonus: H = -Σ p·log(p)
                    entropy = -(probs_new * log_probs_new).sum(axis=1).mean()
                    actor_losses.append(float(actor_loss))

                    # Actor gradient: ∂L/∂logits
                    # dL/d(log_p_a) = -min(r, clip_r) * A / B
                    # Using chain rule through log-softmax:
                    clipped = (clip_r < ratio) | (clip_r > ratio)  # True where ratio was clipped
                    used_ratio = np.where(ratio <= clip_r, ratio, clip_r)
                    grad_logp  = -(used_ratio * mb_adv) / B        # (B,)
                    # Entropy grad: ∂(-H)/∂p_i = log(p_i) + 1; via softmax chain rule
                    # Full grad wrt logits (through log-softmax and then -entropy)
                    grad_logits = np.zeros_like(logits_new)
                    grad_logits[np.arange(B), mb_acts] += grad_logp
                    # Entropy gradient wrt logits: -(∂H/∂logits) = probs*(log_probs+1) - mean
                    ent_grad = probs_new * (log_probs_new + 1.0) - \
                               (probs_new * (log_probs_new + 1.0)).sum(axis=1, keepdims=True)
                    grad_logits += self.c_ent * ent_grad

                    if legal_mask is not None:
                        # No gradient through masked (illegal) action logits.
                        grad_logits = np.where(legal_mask, grad_logits, 0.0)

                    self._actor.backward(grad_logits)             # type: ignore[union-attr]

                    # ── Critic update ──────────────────────────────────────────
                    val_pred = self._critic.forward(mb_obs).flatten()   # type: ignore[union-attr]
                    crit_loss = float(np.mean((val_pred - mb_ret) ** 2))
                    critic_losses.append(crit_loss)

                    grad_val = 2.0 * (val_pred - mb_ret) / B
                    self._critic.backward(self.c_vf * grad_val[:, None])   # type: ignore[union-attr]

        return PPOTrainResult(
            agent_name=self.name,
            n_episodes=completed_episodes,
            episode_rewards=episode_rewards,
            final_epsilon=0.0,   # PPO uses stochastic policy, no ε
            best_episode_history=best_ep_history,
            obs_dim=self.obs_dim,
            mean_actor_loss=float(np.mean(actor_losses)) if actor_losses else 0.0,
            mean_critic_loss=float(np.mean(critic_losses)) if critic_losses else 0.0,
            n_iterations=global_step // self.n_steps,
        )

    def save(self, path: str | Path) -> None:
        """Save actor/critic weights and hyper-parameters to a JSON file."""
        data: dict[str, Any] = {
            "agent": self.name,
            "n_actions":      self.n_actions,
            "mask_actions":   self._mask_actions,
            "obs_dim":        self.obs_dim,
            "hidden_size":    self.hidden_size,
            "lr_actor":       self.lr_actor,
            "lr_critic":      self.lr_critic,
            "gamma":          self.gamma,
            "lam":            self.lam,
            "clip_eps":       self.clip_eps,
            "c_vf":           self.c_vf,
            "c_ent":          self.c_ent,
            "n_steps":        self.n_steps,
            "ppo_epochs":     self.ppo_epochs,
            "mini_batch_size": self.mini_batch_size,
        }
        if self._actor is not None:
            data["actor_weights"]  = self._actor.to_dict()
            data["critic_weights"] = self._critic.to_dict()  # type: ignore[union-attr]
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        """Restore actor/critic weights and hyper-parameters from a JSON file."""
        data = json.loads(Path(path).read_text())
        self.n_actions       = data["n_actions"]
        self._mask_actions   = bool(data.get("mask_actions", False))
        self.obs_dim         = data["obs_dim"]
        self.hidden_size     = data["hidden_size"]
        self.lr_actor        = data["lr_actor"]
        self.lr_critic       = data["lr_critic"]
        self.gamma           = data["gamma"]
        self.lam             = data["lam"]
        self.clip_eps        = data["clip_eps"]
        self.c_vf            = data["c_vf"]
        self.c_ent           = data["c_ent"]
        self.n_steps         = data["n_steps"]
        self.ppo_epochs      = data["ppo_epochs"]
        self.mini_batch_size = data["mini_batch_size"]
        if "actor_weights" in data:
            self._init_nets(self.obs_dim)
            self._actor.from_dict(data["actor_weights"])    # type: ignore[union-attr]
            self._critic.from_dict(data["critic_weights"])  # type: ignore[union-attr]