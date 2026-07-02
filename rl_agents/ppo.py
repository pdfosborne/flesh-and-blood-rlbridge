"""
Proximal Policy Optimisation (PPO) Agent
=========================================
PPO-Clip for discrete action spaces with an attention-based policy trunk.

The flat observation vector is tokenized (heroes, deck, hand, zones, combat)
and encoded with hero-conditioned cross-attention plus a small transformer
before separate actor and critic heads.

Rollout inference uses one shared encode pass via
:meth:`PPOAgent.predict_policy_value`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    raise ImportError("PPO agent requires PyTorch. Install with: pip install torch")

from .attention_policy_v2 import _AttentionPolicyValueV2
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

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_TORCH_DTYPE = torch.float32

from flesh_and_blood_rlbridge.card_text import TEXT_EMBED_VERSION

ARCHITECTURE = _AttentionPolicyValueV2.ARCHITECTURE
UNIFIED_AGENT_WEIGHT_VERSION = 4
_LEGACY_ARCHITECTURES = frozenset({"mlp", "", "attention_v1"})


def _encoder_params(shared: _AttentionPolicyValueV2) -> list[nn.Parameter]:
    return [
        param
        for name, param in shared.named_parameters()
        if not name.startswith("actor_head") and not name.startswith("critic_head")
    ]


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


class _ActorForward(nn.Module):
    def __init__(self, shared: _AttentionPolicyValueV2) -> None:
        super().__init__()
        self._shared = shared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._shared.forward_logits(x)


class _CriticForward(nn.Module):
    def __init__(self, shared: _AttentionPolicyValueV2) -> None:
        super().__init__()
        self._shared = shared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._shared.forward_value(x)


class _PolicyShim:
    """Actor/critic view over :class:`_AttentionPolicyValue` for optimisers."""

    def __init__(self, shared: _AttentionPolicyValueV2, role: str, lr: float) -> None:
        if role not in ("actor", "critic"):
            raise ValueError(f"unknown role: {role!r}")
        self._shared = shared
        self._role = role
        self.lr = lr
        if role == "actor":
            self._net = _ActorForward(shared).to(_DEVICE)
            params = _encoder_params(shared) + list(shared.actor_head.parameters())
        else:
            self._net = _CriticForward(shared).to(_DEVICE)
            params = _encoder_params(shared) + list(shared.critic_head.parameters())
        self._opt = torch.optim.Adam(params, lr=lr)
        self._out_t: Optional[torch.Tensor] = None
        self._out: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(x, dtype=_TORCH_DTYPE, device=_DEVICE)
        self._out_t = self._net(t)
        self._out = self._out_t.detach().cpu().numpy()
        return self._out

    def predict(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            was_training = self._net.training
            self._net.eval()
            try:
                t = torch.as_tensor(x, dtype=_TORCH_DTYPE, device=_DEVICE)
                return self._net(t).cpu().numpy()
            finally:
                self._net.train(was_training)

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            was_training = self._net.training
            self._net.eval()
            try:
                t = torch.as_tensor(x, dtype=_TORCH_DTYPE, device=_DEVICE)
                return self._net(t).cpu().numpy()
            finally:
                self._net.train(was_training)

    def backward(self, loss_grad: np.ndarray) -> None:
        grad_t = torch.as_tensor(loss_grad, dtype=_TORCH_DTYPE, device=_DEVICE)
        self._opt.zero_grad()
        assert self._out_t is not None
        self._out_t.backward(grad_t)
        self._opt.step()


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.where(np.isfinite(x), x, 0.0)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    s = e.sum(axis=-1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return e / s


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.where(np.isfinite(x), x, 0.0)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def _gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    last_gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * (1.0 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def _reject_legacy_weights(data: dict[str, Any]) -> None:
    arch = str(data.get("architecture", "mlp"))
    if arch in _LEGACY_ARCHITECTURES or "actor_weights" in data:
        raise ValueError(
            "Legacy or incompatible unified agent weights; retrain with attention_v2_text "
            f"(found architecture={arch!r})"
        )
    if arch != ARCHITECTURE:
        raise ValueError(f"Unsupported agent architecture: {arch!r}")


class PPOAgent(AgentBase):
    """Proximal Policy Optimisation agent with an attention policy trunk."""

    name = "ppo"

    def __init__(
        self,
        n_actions: int = 0,
        obs_dim: int = 0,
        *,
        hidden_size: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
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
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.c_vf = c_vf
        self.c_ent = c_ent
        self.n_steps = n_steps
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self._seed = seed
        self._rng_py = random.Random(seed)
        self._rng_np = np.random.default_rng(seed)
        self._mask_actions = False

        self._actor: Optional[_PolicyShim] = None
        self._critic: Optional[_PolicyShim] = None
        self._shared: Optional[_AttentionPolicyValueV2] = None

    def _init_nets(self, obs_dim: int) -> None:
        if self._shared is not None:
            return
        self.obs_dim = obs_dim
        self._shared = _AttentionPolicyValueV2(
            self.n_actions,
            d_model=self.hidden_size,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            seed=self._seed,
        ).to(_DEVICE)
        self._actor = _PolicyShim(self._shared, "actor", self.lr_actor)
        self._critic = _PolicyShim(self._shared, "critic", self.lr_critic)

    def _obs_to_vec(self, obs: Any) -> np.ndarray:
        from flesh_and_blood_rlbridge.obs_encoding import observation_fingerprint

        vec = observation_fingerprint(obs, obs_dim=self.obs_dim if self.obs_dim > 0 else 0)
        if self.obs_dim <= 0:
            self.obs_dim = int(vec.shape[0])
        return vec

    def _masked_logits(self, logits: np.ndarray, obs: Any) -> np.ndarray:
        if not getattr(self, "_mask_actions", False):
            return logits
        n_legal = _n_legal_of(obs)
        if n_legal is None or n_legal <= 0 or n_legal >= self.n_actions:
            return logits
        masked = np.array(logits, dtype=np.float64, copy=True)
        masked[..., n_legal:] = -1e9
        return masked

    def predict_batch(self, obs_vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        if self._shared is None:
            return np.zeros(self.n_actions or 1, dtype=np.float64), 0.0
        x = np.asarray(obs_vec, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        logits, values = self._shared.predict_policy_value(x)
        return logits, float(values.reshape(-1)[0])

    def act(self, obs: Any) -> Any:
        if self._actor is None:
            return 0
        x = self._obs_to_vec(obs)
        logits = self._masked_logits(self._actor.predict(x), obs)
        probs = _softmax(logits)
        idx = int(self._rng_np.choice(self.n_actions, p=probs))
        return _to_env_action(obs, idx, getattr(self, "_mask_actions", False))

    def act_greedy(self, obs: Any) -> Any:
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
        if self.n_actions == 0:
            self.n_actions, self._mask_actions = _infer_action_capacity(env, seed=seed)
        else:
            self._mask_actions = _discrete_n(env) is None

        total_steps = n_episodes * max_steps
        episode_rewards: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        completed_episodes = 0
        best_ep_reward = float("-inf")
        best_ep_history: list[tuple] = []

        ep_seed = seed
        reset_out = env.reset(seed=ep_seed)
        obs = _get(reset_out, "observation", reset_out)
        obs_vec = np.array(_flat_obs(obs), dtype=np.float64)
        self._init_nets(obs_vec.shape[0])
        obs_vec = self._obs_to_vec(obs)
        current_ep_reward = 0.0
        current_ep_steps = 0
        current_ep_history: list[tuple] = []
        global_step = 0

        while global_step < total_steps:
            rollout_obs: list[np.ndarray] = []
            rollout_actions: list[int] = []
            rollout_rewards: list[float] = []
            rollout_dones: list[float] = []
            rollout_log_probs: list[float] = []
            rollout_values: list[float] = []
            rollout_n_legal: list[int] = []

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
                next_obs = _get(step_out, "observation", obs)
                reward = float(_get(step_out, "reward", 0.0))
                terminated = bool(_get(step_out, "terminated", False))
                truncated = bool(_get(step_out, "truncated", False))
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
                current_ep_steps += 1
                global_step += 1

                if done or current_ep_steps >= max_steps:
                    episode_rewards.append(current_ep_reward)
                    if current_ep_reward > best_ep_reward:
                        best_ep_reward = current_ep_reward
                        best_ep_history = list(current_ep_history)
                    completed_episodes += 1
                    current_ep_reward = 0.0
                    current_ep_steps = 0
                    current_ep_history = []
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
            obs_arr = np.array(rollout_obs, dtype=np.float64)
            act_arr = np.array(rollout_actions, dtype=np.int64)
            values_arr = np.array(rollout_values, dtype=np.float64)
            log_old_arr = np.array(rollout_log_probs, dtype=np.float64)
            dones_arr = np.array(rollout_dones, dtype=np.float64)
            nlegal_arr = np.array(rollout_n_legal, dtype=np.int64)

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
            if T > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            indices = np.arange(T)
            for _ in range(self.ppo_epochs):
                self._rng_np.shuffle(indices)
                for start in range(0, T, self.mini_batch_size):
                    mb_idx = indices[start : start + self.mini_batch_size]
                    if len(mb_idx) == 0:
                        continue

                    mb_obs = obs_arr[mb_idx]
                    mb_acts = act_arr[mb_idx]
                    mb_adv = advantages[mb_idx]
                    mb_ret = returns[mb_idx]
                    mb_lp_old = log_old_arr[mb_idx]
                    mb_nlegal = nlegal_arr[mb_idx]

                    logits_new = self._actor.forward(mb_obs)  # type: ignore[union-attr]
                    B = mb_obs.shape[0]
                    legal_mask: Optional[np.ndarray] = None
                    if self._mask_actions:
                        legal_mask = np.arange(self.n_actions)[None, :] < mb_nlegal[:, None]
                        logits_new = np.where(legal_mask, logits_new, -1e9)
                    log_probs_new = _log_softmax(logits_new)
                    probs_new = _softmax(logits_new)
                    lp_new = log_probs_new[np.arange(B), mb_acts]

                    ratio = np.exp(lp_new - mb_lp_old)
                    clip_r = np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    actor_loss = -np.minimum(ratio * mb_adv, clip_r * mb_adv).mean()
                    actor_losses.append(float(actor_loss))

                    used_ratio = np.where(ratio <= clip_r, ratio, clip_r)
                    grad_logp = -(used_ratio * mb_adv) / B
                    grad_logits = np.zeros_like(logits_new)
                    grad_logits[np.arange(B), mb_acts] += grad_logp
                    ent_grad = probs_new * (log_probs_new + 1.0) - (
                        probs_new * (log_probs_new + 1.0)
                    ).sum(axis=1, keepdims=True)
                    grad_logits += self.c_ent * ent_grad

                    if legal_mask is not None:
                        grad_logits = np.where(legal_mask, grad_logits, 0.0)

                    self._actor.backward(grad_logits)  # type: ignore[union-attr]

                    val_pred = self._critic.forward(mb_obs).flatten()  # type: ignore[union-attr]
                    crit_loss = float(np.mean((val_pred - mb_ret) ** 2))
                    critic_losses.append(crit_loss)

                    grad_val = 2.0 * (val_pred - mb_ret) / B
                    self._critic.backward(self.c_vf * grad_val[:, None])  # type: ignore[union-attr]

        return PPOTrainResult(
            agent_name=self.name,
            n_episodes=completed_episodes,
            episode_rewards=episode_rewards,
            final_epsilon=0.0,
            best_episode_history=best_ep_history,
            obs_dim=self.obs_dim,
            mean_actor_loss=float(np.mean(actor_losses)) if actor_losses else 0.0,
            mean_critic_loss=float(np.mean(critic_losses)) if critic_losses else 0.0,
            n_iterations=global_step // self.n_steps,
        )

    def save(self, path: str | Path) -> None:
        data: dict[str, Any] = {
            "agent": self.name,
            "architecture": ARCHITECTURE,
            "weight_version": UNIFIED_AGENT_WEIGHT_VERSION,
            "text_embed_version": TEXT_EMBED_VERSION,
            "n_actions": self.n_actions,
            "mask_actions": self._mask_actions,
            "obs_dim": self.obs_dim,
            "d_model": self.hidden_size,
            "hidden_size": self.hidden_size,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "lr_actor": self.lr_actor,
            "lr_critic": self.lr_critic,
            "gamma": self.gamma,
            "lam": self.lam,
            "clip_eps": self.clip_eps,
            "c_vf": self.c_vf,
            "c_ent": self.c_ent,
            "n_steps": self.n_steps,
            "ppo_epochs": self.ppo_epochs,
            "mini_batch_size": self.mini_batch_size,
        }
        if self._shared is not None:
            data["state_dict"] = self._shared.state_dict_json()
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        _reject_legacy_weights(data)
        self.n_actions = data["n_actions"]
        self._mask_actions = bool(data.get("mask_actions", False))
        self.obs_dim = data["obs_dim"]
        self.hidden_size = int(data.get("d_model", data.get("hidden_size", 64)))
        self.n_layers = int(data.get("n_layers", 2))
        self.n_heads = int(data.get("n_heads", 4))
        self.lr_actor = data["lr_actor"]
        self.lr_critic = data["lr_critic"]
        self.gamma = data["gamma"]
        self.lam = data["lam"]
        self.clip_eps = data["clip_eps"]
        self.c_vf = data["c_vf"]
        self.c_ent = data["c_ent"]
        self.n_steps = data["n_steps"]
        self.ppo_epochs = data["ppo_epochs"]
        self.mini_batch_size = data["mini_batch_size"]
        if "state_dict" in data:
            self._init_nets(self.obs_dim)
            assert self._shared is not None
            self._shared.load_state_dict_json(data["state_dict"], _DEVICE)
