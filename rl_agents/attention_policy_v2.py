"""Attention policy v2 with frozen card text embeddings."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from flesh_and_blood_rlbridge.card_text import TEXT_EMBED_VERSION, load_text_embedding_table, text_embed_dim
from flesh_and_blood_rlbridge.card_vocab import hero_vocab_size
from flesh_and_blood_rlbridge.obs_tokenizer import ObsTokenLayout, TokenBatch, build_token_features, denorm_card_index
from flesh_and_blood_rlbridge.player_observation import (
    COMBAT_CHAIN_SLOT_DIM,
    COMBAT_SCALAR_COUNT,
    HAND_SLOT_DIM,
    PLAYER_OBS_DIM,
    SCALAR_COUNT,
    ZONE_SLOT_DIM,
    ZONE_SPECS,
)

_TORCH_DTYPE = torch.float32
_HAND_FEAT_DIM = HAND_SLOT_DIM - 1
_ZONE_FEAT_DIM = ZONE_SLOT_DIM - 1
_CHAIN_FEAT_DIM = COMBAT_CHAIN_SLOT_DIM - 1


def _state_dict_to_json(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        k: v.detach().cpu().tolist()
        for k, v in state.items()
        if not k.startswith("text_embed_table")
    }


def _state_dict_from_json(data: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        k: torch.tensor(v, dtype=_TORCH_DTYPE, device=device)
        for k, v in data.items()
    }


def _masked_mean_pool(tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    valid = (~pad_mask).unsqueeze(-1).to(tokens.dtype)
    summed = (tokens * valid).sum(dim=1)
    counts = valid.sum(dim=1).clamp(min=1.0)
    return summed / counts


class _AttentionPolicyValueV2(nn.Module):
    """Tokenized observation encoder with frozen text card embeddings."""

    ARCHITECTURE = "attention_v2_text"

    def __init__(
        self,
        n_actions: int,
        *,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        seed: Optional[int] = None,
        text_embed_version: str = TEXT_EMBED_VERSION,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.n_actions = n_actions
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.text_embed_version = text_embed_version
        self.layout = ObsTokenLayout()

        hero_vocab = max(hero_vocab_size(), 1)
        n_zone_types = len(ZONE_SPECS)
        embed_dim = text_embed_dim()

        table = load_text_embedding_table()
        self.register_buffer(
            "text_embed_table",
            torch.tensor(table, dtype=_TORCH_DTYPE),
            persistent=False,
        )

        self.hero_embed = nn.Embedding(hero_vocab, d_model, dtype=_TORCH_DTYPE)
        self.zone_type_embed = nn.Embedding(n_zone_types, d_model, dtype=_TORCH_DTYPE)
        self.side_embed = nn.Embedding(2, d_model, dtype=_TORCH_DTYPE)

        self.text_proj = nn.Linear(embed_dim, d_model, dtype=_TORCH_DTYPE)
        self.meta_proj = nn.Linear(2, d_model, dtype=_TORCH_DTYPE)
        self.scalar_proj = nn.Linear(SCALAR_COUNT, d_model, dtype=_TORCH_DTYPE)
        self.hand_proj = nn.Linear(_HAND_FEAT_DIM, d_model, dtype=_TORCH_DTYPE)
        self.zone_proj = nn.Linear(_ZONE_FEAT_DIM, d_model, dtype=_TORCH_DTYPE)
        self.combat_scalar_proj = nn.Linear(COMBAT_SCALAR_COUNT, d_model, dtype=_TORCH_DTYPE)
        self.combat_chain_proj = nn.Linear(_CHAIN_FEAT_DIM, d_model, dtype=_TORCH_DTYPE)

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

    def _cond_scale(self, cond: torch.Tensor) -> torch.Tensor:
        return 0.25 + 0.75 * cond.clamp(0.0, 1.0)

    def _text_tokens(self, card_ids: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        ids = card_ids.clamp(min=0, max=self.text_embed_table.shape[0] - 1)
        embeds = self.text_proj(self.text_embed_table[ids])
        return embeds * self._cond_scale(cond).unsqueeze(-1)

    def _deck_token(self, deck_ids: torch.Tensor) -> torch.Tensor:
        embeds = self._text_tokens(deck_ids, torch.ones_like(deck_ids, dtype=_TORCH_DTYPE))
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

        hand_ids = denorm_card_index(batch.hand[..., 0])
        hand_cond = batch.hand[..., -1]
        hand_tok = self.hand_proj(batch.hand[..., :-1]) + self._text_tokens(hand_ids, hand_cond)

        zone_flat = batch.zones.reshape(b * self.layout.zone_slots_total, ZONE_SLOT_DIM)
        zone_tok = self.zone_proj(zone_flat[..., :-1]).reshape(b, self.layout.zone_slots_total, d)
        zone_tok = zone_tok + self.zone_type_embed(batch.zone_type_ids).unsqueeze(0)
        zone_tok = zone_tok + self.side_embed(batch.side_ids).unsqueeze(0)
        zone_ids = denorm_card_index(batch.zones[..., 0])
        zone_cond = batch.zones[..., -1]
        zone_tok = zone_tok + self._text_tokens(zone_ids, zone_cond)

        combat_scalar_tok = self.combat_scalar_proj(batch.combat_scalars).unsqueeze(1)
        chain_ids = denorm_card_index(batch.combat_chain[..., 0])
        chain_cond = batch.combat_chain[..., -1]
        chain_tok = self.combat_chain_proj(batch.combat_chain[..., :-1]) + self._text_tokens(
            chain_ids, chain_cond
        )

        return torch.cat(
            [scalar_tok, hand_tok, zone_tok, combat_scalar_tok, chain_tok],
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
        self.load_state_dict(_state_dict_from_json(data, device), strict=False)

    @classmethod
    def expected_obs_dim(cls) -> int:
        return PLAYER_OBS_DIM
