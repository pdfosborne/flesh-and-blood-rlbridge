"""Fast observation fingerprinting for RL training (matches PPOAgent input)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def _flat_obs(obs: Any) -> list[float]:
    """Flatten any observation type to floats (mirrors rl_agents._agent_base._flat_obs)."""
    if isinstance(obs, str):
        try:
            parsed = json.loads(obs)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            obs_vec = parsed.get("observationVec")
            if isinstance(obs_vec, list) and obs_vec:
                return _flat_obs(obs_vec)
        h = 0
        for ch in obs.encode():
            h = (h * 31 + ch) & 0xFFFFFFFF
        return [h / 0xFFFFFFFF]
    if isinstance(obs, (int, float)):
        return [float(obs)]
    if isinstance(obs, (list, tuple)):
        out: list[float] = []
        for v in obs:
            out.extend(_flat_obs(v))
        return out
    if isinstance(obs, dict):
        out: list[float] = []
        for k in sorted(obs.keys(), key=lambda x: str(x)):
            out.extend(_flat_obs(str(k)))
            out.extend(_flat_obs(obs[k]))
        return out
    if obs is None:
        return [0.0]
    if isinstance(obs, bool):
        return [1.0 if obs else 0.0]
    try:
        if isinstance(obs, np.ndarray):
            return obs.flatten().tolist()
    except Exception:
        pass
    return [float(hash(obs) & 0xFFFF) / 0xFFFF]


def observation_fingerprint(obs: Any, *, obs_dim: int = 0) -> np.ndarray:
    """Return the same vector PPOAgent._obs_to_vec produces for *obs*."""
    vec = np.array(_flat_obs(obs), dtype=np.float64)
    vec = np.where(np.isfinite(vec), vec, 0.0)
    if obs_dim <= 0:
        return vec
    if vec.shape[0] == obs_dim:
        return vec
    if vec.shape[0] < obs_dim:
        pad = np.zeros(obs_dim - vec.shape[0], dtype=np.float64)
        return np.concatenate([vec, pad])
    return vec[:obs_dim]
