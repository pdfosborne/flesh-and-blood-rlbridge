"""Vectorized token extraction from the flat player observation vector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .card_vocab import hero_vocab_size, vocab_size
from .player_observation import (
    COMBAT_CHAIN_END,
    COMBAT_CHAIN_OFF,
    COMBAT_CHAIN_SLOTS,
    COMBAT_CHAIN_SLOT_DIM,
    COMBAT_SCALAR_OFF,
    COMBAT_SCALAR_COUNT,
    CONTEXT_DIM,
    DECK_OFF,
    EFFECT_END,
    EFFECT_OFF,
    EFFECT_SLOT_DIM,
    EFFECT_SLOTS,
    FIRST_PLAYER_OFF,
    FORMAT_OFF,
    HAND_END,
    HAND_OFF,
    HAND_SLOTS,
    HAND_SLOT_DIM,
    HERO_OPP_OFF,
    HERO_SELF_OFF,
    LAYER_END,
    LAYER_OFF,
    LAYER_SLOT_DIM,
    LAYER_SLOTS,
    PLAYER_OBS_DIM,
    PLAYED_OPP_END,
    PLAYED_OPP_OFF,
    PLAYED_SELF_END,
    PLAYED_SELF_OFF,
    PLAYED_SLOT_DIM,
    PLAYED_SLOTS,
    SCALAR_COUNT,
    SCALAR_OFF,
    ZONE_END,
    ZONE_OFF,
    ZONE_SLOT_DIM,
    ZONE_SLOTS_PER_PLAYER,
    ZONE_SPECS,
)


@dataclass(frozen=True)
class ObsTokenLayout:
    """Slice bounds for the canonical player observation vector."""

    obs_dim: int = PLAYER_OBS_DIM
    context_dim: int = CONTEXT_DIM
    scalar_count: int = SCALAR_COUNT
    hand_slots: int = HAND_SLOTS
    hand_slot_dim: int = HAND_SLOT_DIM
    zone_slots_total: int = 2 * ZONE_SLOTS_PER_PLAYER
    zone_slot_dim: int = ZONE_SLOT_DIM
    combat_chain_slots: int = COMBAT_CHAIN_SLOTS
    played_slots_total: int = 2 * PLAYED_SLOTS
    effect_slots: int = EFFECT_SLOTS
    layer_slots: int = LAYER_SLOTS

    @property
    def board_token_count(self) -> int:
        """Scalars + hand + zones + combat + play history + effects + layers."""
        return (
            1
            + self.hand_slots
            + self.zone_slots_total
            + 1
            + self.combat_chain_slots
            + self.played_slots_total
            + self.effect_slots
            + self.layer_slots
        )

    @property
    def context_token_count(self) -> int:
        """hero_self, hero_opp, meta, deck pool."""
        return 4


def denorm_card_index(norm: torch.Tensor) -> torch.Tensor:
    """Map normalized card indices in [0, 1] to integer vocab ids."""
    size = max(vocab_size(), 1)
    return torch.clamp((norm * float(size)).round().long(), min=0, max=size)


def denorm_hero_index(norm: torch.Tensor) -> torch.Tensor:
    """Map normalized hero indices in [0, 1] to integer hero vocab ids."""
    size = max(hero_vocab_size(), 1)
    return torch.clamp((norm * float(size)).round().long(), min=0, max=size - 1)


def _zone_type_side_ids(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (zone_type_ids, side_ids) each shape (zone_slots_total,)."""
    zone_types: list[int] = []
    sides: list[int] = []
    for side in (0, 1):
        for zone_idx, (_, max_slots) in enumerate(ZONE_SPECS):
            zone_types.extend([zone_idx] * max_slots)
            sides.extend([side] * max_slots)
    return (
        torch.tensor(zone_types, dtype=torch.long, device=device),
        torch.tensor(sides, dtype=torch.long, device=device),
    )


@dataclass
class TokenBatch:
    """Batched token features extracted from observation vectors."""

    hero_self_ids: torch.Tensor       # (B,)
    hero_opp_ids: torch.Tensor        # (B,)
    deck_card_ids: torch.Tensor       # (B, DECK_SLOTS)
    meta: torch.Tensor                # (B, 2) format + first_player
    scalars: torch.Tensor             # (B, SCALAR_COUNT)
    hand: torch.Tensor                # (B, HAND_SLOTS, HAND_SLOT_DIM)
    zones: torch.Tensor               # (B, zone_slots, ZONE_SLOT_DIM)
    zone_type_ids: torch.Tensor       # (zone_slots,)
    side_ids: torch.Tensor            # (zone_slots,)
    combat_scalars: torch.Tensor      # (B, COMBAT_SCALAR_COUNT)
    combat_chain: torch.Tensor        # (B, COMBAT_CHAIN_SLOTS, COMBAT_CHAIN_SLOT_DIM)
    played_history: torch.Tensor      # (B, played_slots_total, PLAYED_SLOT_DIM)
    turn_effects: torch.Tensor        # (B, EFFECT_SLOTS, EFFECT_SLOT_DIM)
    layers: torch.Tensor              # (B, LAYER_SLOTS, LAYER_SLOT_DIM)
    board_padding_mask: torch.Tensor  # (B, board_token_count) True = ignore


def build_token_features(
    obs: torch.Tensor,
    *,
    layout: Optional[ObsTokenLayout] = None,
) -> TokenBatch:
    """Extract token features from a batch of flat observation vectors."""
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    if obs.shape[-1] != PLAYER_OBS_DIM:
        raise ValueError(f"expected obs dim {PLAYER_OBS_DIM}, got {obs.shape[-1]}")

    layout = layout or ObsTokenLayout()
    device = obs.device
    b = obs.shape[0]

    hero_self_ids = denorm_hero_index(obs[:, HERO_SELF_OFF])
    hero_opp_ids = denorm_hero_index(obs[:, HERO_OPP_OFF])
    meta = obs[:, [FORMAT_OFF, FIRST_PLAYER_OFF]]
    deck_card_ids = denorm_card_index(obs[:, DECK_OFF:CONTEXT_DIM])
    scalars = obs[:, SCALAR_OFF : SCALAR_OFF + SCALAR_COUNT]
    hand = obs[:, HAND_OFF:HAND_END].reshape(b, HAND_SLOTS, HAND_SLOT_DIM)
    zones = obs[:, ZONE_OFF:ZONE_END].reshape(b, layout.zone_slots_total, ZONE_SLOT_DIM)
    combat_scalars = obs[:, COMBAT_SCALAR_OFF : COMBAT_SCALAR_OFF + COMBAT_SCALAR_COUNT]
    combat_chain = obs[:, COMBAT_CHAIN_OFF:COMBAT_CHAIN_END].reshape(
        b, COMBAT_CHAIN_SLOTS, COMBAT_CHAIN_SLOT_DIM
    )
    played_history = obs[:, PLAYED_SELF_OFF:PLAYED_OPP_END].reshape(
        b, layout.played_slots_total, PLAYED_SLOT_DIM
    )
    turn_effects = obs[:, EFFECT_OFF:EFFECT_END].reshape(b, EFFECT_SLOTS, EFFECT_SLOT_DIM)
    layers = obs[:, LAYER_OFF:LAYER_END].reshape(b, LAYER_SLOTS, LAYER_SLOT_DIM)

    zone_type_ids, side_ids = _zone_type_side_ids(device)

    hand_pad = hand[..., 0] <= 0
    zone_pad = zones[..., 0] <= 0
    chain_pad = combat_chain[..., 0] <= 0
    played_pad = played_history[..., 0] <= 0
    effect_pad = turn_effects[..., 0] <= 0
    layer_pad = layers[..., 1] <= 0

    board_padding_mask = torch.cat(
        [
            torch.zeros(b, 1, dtype=torch.bool, device=device),   # scalars
            hand_pad,
            zone_pad,
            torch.zeros(b, 1, dtype=torch.bool, device=device),   # combat scalars
            chain_pad,
            played_pad,
            effect_pad,
            layer_pad,
        ],
        dim=1,
    )
    if board_padding_mask.shape[1] != layout.board_token_count:
        raise RuntimeError(
            f"board padding mask width {board_padding_mask.shape[1]} "
            f"!= expected {layout.board_token_count}"
        )

    return TokenBatch(
        hero_self_ids=hero_self_ids,
        hero_opp_ids=hero_opp_ids,
        deck_card_ids=deck_card_ids,
        meta=meta,
        scalars=scalars,
        hand=hand,
        zones=zones,
        zone_type_ids=zone_type_ids,
        side_ids=side_ids,
        combat_scalars=combat_scalars,
        combat_chain=combat_chain,
        played_history=played_history,
        turn_effects=turn_effects,
        layers=layers,
        board_padding_mask=board_padding_mask,
    )
