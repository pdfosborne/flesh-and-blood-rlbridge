from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from flesh_and_blood_rlbridge.environment import FleshAndBloodEnvironment


class RandomOpponent:
    """Random opponent over currently playable attacks/blocks."""

    def select_attack_index(self, env: "FleshAndBloodEnvironment") -> Optional[int]:
        opp = env._players[1]  # noqa: SLF001
        playable_indices: list[int] = []
        for i, cid in enumerate(opp.hand):
            card = env._cards[cid]  # noqa: SLF001
            if "attack_action" in card.card_types and card.cost <= opp.resources:
                playable_indices.append(i)
        if not playable_indices:
            return None
        return int(env._rng.choice(playable_indices))  # noqa: SLF001

    def select_block_index(self, env: "FleshAndBloodEnvironment") -> Optional[int]:
        if env._pending_combat is None or env._pending_combat.defender != 1:  # noqa: SLF001
            return None

        defender = env._players[1]  # noqa: SLF001
        block_indices: list[int] = []
        for i, cid in enumerate(defender.hand):
            card = env._cards[cid]  # noqa: SLF001
            if card.defense > 0:
                block_indices.append(i)
        if not block_indices:
            return None
        return int(env._rng.choice(block_indices))  # noqa: SLF001
