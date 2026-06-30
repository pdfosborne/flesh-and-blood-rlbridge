"""Compact summaries of unified PPO policy weights for live dashboards."""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rl_agents.ppo import PPOAgent


def summarize_policy_weights(agent: "PPOAgent") -> dict[str, Any]:
    """Return JSON-serializable weight stats for *agent*'s shared network."""
    shared = getattr(agent, "_shared", None)
    if shared is None:
        return {"initialized": False}

    try:
        import torch
    except ImportError:
        return {
            "initialized": True,
            "obs_dim": int(getattr(agent, "obs_dim", 0) or 0),
            "n_actions": int(getattr(agent, "n_actions", 0) or 0),
            "d_model": int(getattr(agent, "hidden_size", 0) or 0),
            "n_layers": int(getattr(agent, "n_layers", 0) or 0),
            "n_heads": int(getattr(agent, "n_heads", 0) or 0),
            "torch_available": False,
        }

    params = list(shared.parameters())
    if not params:
        return {"initialized": False}

    total_params = sum(int(p.numel()) for p in params)
    l2_sum = 0.0
    stats_parts: list[str] = []
    for name, param in shared.named_parameters():
        tensor = param.detach().float().cpu()
        l2_sum += float(torch.sum(tensor * tensor).item())
        stats_parts.append(
            f"{name}:{tensor.mean().item():.6f}:{tensor.std(unbiased=False).item():.6f}"
        )

    l2_norm = math.sqrt(l2_sum)
    fingerprint = hashlib.sha256("|".join(stats_parts).encode("utf-8")).hexdigest()[:12]

    return {
        "initialized": True,
        "obs_dim": int(getattr(agent, "obs_dim", 0) or 0),
        "n_actions": int(getattr(agent, "n_actions", 0) or 0),
        "d_model": int(getattr(agent, "hidden_size", 0) or 0),
        "n_layers": int(getattr(agent, "n_layers", 0) or 0),
        "n_heads": int(getattr(agent, "n_heads", 0) or 0),
        "param_count": total_params,
        "l2_norm": round(l2_norm, 6),
        "fingerprint": fingerprint,
        "torch_available": True,
    }
