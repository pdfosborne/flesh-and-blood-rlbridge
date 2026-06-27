"""Observation alignment between the C++ fast engine and Talishar HTTP.

Policies trained on the simplified C++ simulator must see the same effective
input layout when evaluated on Talishar.  The C++ engine does not yet mirror
combat-chain encoding; zone conditionals also diverge when the mirror payload
lacks full HTTP state.  This module applies a shared post-process so both
backends present a consistent vector to the agent.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .player_observation import (
    COMBAT_CHAIN_END,
    COMBAT_SCALAR_OFF,
    PLAYER_OBS_DIM,
    ZONE_END,
    ZONE_OFF,
    ZONE_SLOT_DIM,
)

# Slices the C++ fast engine does not populate faithfully yet.
_CPP_NEUTRAL_SLICES: tuple[tuple[int, int], ...] = (
    (COMBAT_SCALAR_OFF, COMBAT_CHAIN_END),
)


def cpp_obs_alignment_enabled() -> bool:
    return os.environ.get("FAB_CPP_OBS_ALIGNMENT", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def align_observation_for_cpp_training(vec: np.ndarray) -> np.ndarray:
    """Return a copy of *vec* with C++-unimplemented slices neutralized."""
    out = np.array(vec, dtype=np.float64, copy=True).reshape(-1)
    if out.shape[0] != PLAYER_OBS_DIM:
        raise ValueError(f"expected obs dim {PLAYER_OBS_DIM}, got {out.shape[0]}")
    for start, end in _CPP_NEUTRAL_SLICES:
        out[start:end] = 0.0
    # Zone conditional_active (last dim per slot) — C++ uses a stub evaluator.
    zone = out[ZONE_OFF:ZONE_END].reshape(-1, ZONE_SLOT_DIM)
    if zone.size:
        zone[:, 4] = 1.0
        out[ZONE_OFF:ZONE_END] = zone.reshape(-1)
    return out


def observation_vectors_aligned(
    tal_vec: np.ndarray,
    cpp_vec: np.ndarray,
    *,
    atol: float = 0.05,
) -> tuple[bool, str]:
    """Compare aligned vectors for parity / transfer checks."""
    tal = align_observation_for_cpp_training(tal_vec)
    cpp = align_observation_for_cpp_training(cpp_vec)
    delta = float(np.max(np.abs(tal - cpp)))
    if delta > atol:
        return False, f"max aligned delta={delta:.4f} (tol={atol})"
    return True, ""


def merge_talishar_raw_state(
    cpp_raw: dict[str, Any],
    talishar_raw: dict[str, Any] | None,
) -> dict[str, Any]:
    """Prefer Talishar HTTP zone/combat fields when mirroring for parity."""
    if not isinstance(talishar_raw, dict) or not talishar_raw:
        return cpp_raw
    merged = dict(cpp_raw)
    for key, value in talishar_raw.items():
        if not isinstance(key, str):
            continue
        if key.startswith(("player", "opponent", "combat", "chain")) or key in {
            "amIActivePlayer",
            "turnPlayer",
            "canPassPhase",
            "popup",
            "lastPlayed",
            "lastTurnPitched",
        }:
            merged[key] = value
    return merged
