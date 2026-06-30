"""Attention-based shared actor/critic trunk for PPO."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from flesh_and_blood_rlbridge.card_vocab import hero_vocab_size, vocab_size
from flesh_and_blood_rlbridge.obs_tokenizer import ObsTokenLayout, TokenBatch, build_token_features
from flesh_and_blood_rlbridge.player_observation import (
    COMBAT_SCALAR_COUNT,
    EFFECT_SLOT_DIM,
    HAND_SLOT_DIM,
    LAYER_SLOT_DIM,
    PLAYER_OBS_DIM,
    PLAYED_SLOT_DIM,
    SCALAR_COUNT,
    ZONE_SLOT_DIM,
    ZONE_SPECS,
)

_TORCH_DTYPE = torch.float32


def _state_dict_to_json(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {k: v.detach().cpu().tolist() for k, v in state.items()}


def _state_dict_from_json(data: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        k: torch.tensor(v, dtype=_TORCH_DTYPE, device=device)
        for k, v in data.items()
    }


def _masked_mean_pool(tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool tokens; *pad_mask* True marks positions to ignore."""
    valid = (~pad_mask).unsqueeze(-1).to(tokens.dtype)
    summed = (tokens * valid).sum(dim=1)
    counts = valid.sum(dim=1).clamp(min=1.0)
    return summed / counts


class _AttentionPolicyValue(nn.Module):
    """Tokenized observation encoder with hero-conditioned cross-attention."""

    ARCHITECTURE = "attention_v1"

    def __init__(
        self,
        n_actions: int,
        *,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.n_actions = n_actions
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.layout = ObsTokenLayout()

        card_vocab = max(vocab_size(), 1)
        hero_vocab = max(hero_vocab_size(), 1)
        n_zone_types = len(ZONE_SPECS)

        self.hero_embed = nn.Embedding(hero_vocab, d_model, dtype=_TORCH_DTYPE)
        self.card_embed = nn.Embedding(card_vocab + 1, d_model, dtype=_TORCH_DTYPE)
        self.zone_type_embed = nn.Embedding(n_zone_types, d_model, dtype=_TORCH_DTYPE)
        self.side_embed = nn.Embedding(2, d_model, dtype=_TORCH_DTYPE)

        self.meta_proj = nn.Linear(2, d_model, dtype=_TORCH_DTYPE)
        self.scalar_proj = nn.Linear(SCALAR_COUNT, d_model, dtype=_TORCH_DTYPE)
        self.hand_proj = nn.Linear(HAND_SLOT_DIM, d_model, dtype=_TORCH_DTYPE)
        self.zone_proj = nn.Linear(ZONE_SLOT_DIM, d_model, dtype=_TORCH_DTYPE)
        self.combat_scalar_proj = nn.Linear(COMBAT_SCALAR_COUNT, d_model, dtype=_TORCH_DTYPE)
        self.combat_chain_proj = nn.Linear(2, d_model, dtype=_TORCH_DTYPE)
        self.played_proj = nn.Linear(PLAYED_SLOT_DIM, d_model, dtype=_TORCH_DTYPE)
        self.effect_proj = nn.Linear(EFFECT_SLOT_DIM, d_model, dtype=_TORCH_DTYPE)
        self.layer_proj = nn.Linear(LAYER_SLOT_DIM, d_model, dtype=_TORCH_DTYPE)

        self.cross_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            batch_first=True,
            dtype=_TORCH_DTYPE,
        )
        self.cross_norm = nn.LayerNorm(d_model, dtype=_TORCH_DTYPE)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
            dtype=_TORCH_DTYPE,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.actor_head = nn.Linear(d_model, n_actions, dtype=_TORCH_DTYPE)
        self.critic_head = nn.Linear(d_model, 1, dtype=_TORCH_DTYPE)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _deck_token(self, deck_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.card_embed(deck_ids.clamp(min=0, max=self.card_embed.num_embeddings - 1))
        valid = (deck_ids > 0).unsqueeze(-1).to(embeds.dtype)
        summed = (embeds * valid).sum(dim=1)
        counts = valid.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def _context_tokens(self, batch: TokenBatch) -> torch.Tensor:
        hero_self = self.hero_embed(batch.hero_self_ids)
        hero_opp = self.hero_embed(batch.hero_opp_ids)
        meta = self.meta_proj(batch.meta)
        deck = self._deck_token(batch.deck_card_ids)
        return torch.stack([hero_self, hero_opp, meta, deck], dim=1)

    def _board_tokens(self, batch: TokenBatch) -> torch.Tensor:
        b = batch.scalars.shape[0]
        device = batch.scalars.device
        d = self.d_model

        scalar_tok = self.scalar_proj(batch.scalars).unsqueeze(1)
        hand_tok = self.hand_proj(batch.hand)

        zone_flat = batch.zones.reshape(b * self.layout.zone_slots_total, ZONE_SLOT_DIM)
        zone_tok = self.zone_proj(zone_flat).reshape(b, self.layout.zone_slots_total, d)
        zone_tok = zone_tok + self.zone_type_embed(batch.zone_type_ids).unsqueeze(0)
        zone_tok = zone_tok + self.side_embed(batch.side_ids).unsqueeze(0)

        card_ids = batch.zones[..., 0].round().long().clamp(
            min=0, max=self.card_embed.num_embeddings - 1
        )
        zone_tok = zone_tok + self.card_embed(card_ids)

        hand_ids = batch.hand[..., 0].round().long().clamp(
            min=0, max=self.card_embed.num_embeddings - 1
        )
        hand_tok = hand_tok + self.card_embed(hand_ids)

        combat_scalar_tok = self.combat_scalar_proj(batch.combat_scalars).unsqueeze(1)
        chain_tok = self.combat_chain_proj(batch.combat_chain)
        chain_ids = batch.combat_chain[..., 0].round().long().clamp(
            min=0, max=self.card_embed.num_embeddings - 1
        )
        chain_tok = chain_tok + self.card_embed(chain_ids)

        played_tok = self.played_proj(batch.played_history)
        played_ids = batch.played_history[..., 0].round().long().clamp(
            min=0, max=self.card_embed.num_embeddings - 1
        )
        played_tok = played_tok + self.card_embed(played_ids)

        effect_tok = self.effect_proj(batch.turn_effects)
        effect_ids = batch.turn_effects[..., 0].round().long().clamp(
            min=0, max=self.card_embed.num_embeddings - 1
        )
        effect_tok = effect_tok + self.card_embed(effect_ids)

        layer_tok = self.layer_proj(batch.layers)
        layer_ids = batch.layers[..., 1].round().long().clamp(
            min=0, max=self.card_embed.num_embeddings - 1
        )
        layer_tok = layer_tok + self.card_embed(layer_ids)

        return torch.cat(
            [
                scalar_tok,
                hand_tok,
                zone_tok,
                combat_scalar_tok,
                chain_tok,
                played_tok,
                effect_tok,
                layer_tok,
            ],
            dim=1,
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        batch = build_token_features(obs, layout=self.layout)
        context = self._context_tokens(batch)
        board = self._board_tokens(batch)

        attn_out, _ = self.cross_attn(board, context, context, need_weights=False)
        board = self.cross_norm(board + attn_out)

        encoded = self.encoder(board, src_key_padding_mask=batch.board_padding_mask)
        return _masked_mean_pool(encoded, batch.board_padding_mask)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor_head(self.encode(x))

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.encode(x))

    def predict_policy_value(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            was_training = self.training
            self.eval()
            try:
                device = next(self.parameters()).device
                t = torch.as_tensor(x, dtype=_TORCH_DTYPE, device=device)
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                hidden = self.encode(t)
                logits = self.actor_head(hidden).cpu().numpy()
                values = self.critic_head(hidden).cpu().numpy()
            finally:
                self.train(was_training)
        return logits, values

    def state_dict_json(self) -> dict[str, Any]:
        return _state_dict_to_json(self.state_dict())

    def load_state_dict_json(self, data: dict[str, Any], device: torch.device) -> None:
        self.load_state_dict(_state_dict_from_json(data, device))

    @classmethod
    def expected_obs_dim(cls) -> int:
        return PLAYER_OBS_DIM
