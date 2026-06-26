"""Deprecated: use player_observation.py."""

from __future__ import annotations

from .player_observation import (
    ACTION_CAPACITY as FAST_ACTION_CAPACITY,
    HAND_SLOTS as FAST_HAND_SLOTS,
    PLAYER_OBS_DIM as FAST_OBS_DIM,
    observation_payload as fast_observation_payload,
    player_observation_payload,
    player_observation_vector as fast_observation_vector,
)

__all__ = [
    "FAST_ACTION_CAPACITY",
    "FAST_HAND_SLOTS",
    "FAST_OBS_DIM",
    "fast_observation_payload",
    "fast_observation_vector",
    "player_observation_payload",
]
