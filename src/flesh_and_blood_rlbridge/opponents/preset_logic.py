from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from flesh_and_blood_rlbridge.environment import FleshAndBloodEnvironment


class PresetLogicOpponent:
    """Heuristic scripted opponent using net defense-vs-attack values."""

    def select_attack_index(self, env: "FleshAndBloodEnvironment") -> Optional[int]:
        opp = env._players[1]  # noqa: SLF001
        agent = env._players[0]  # noqa: SLF001

        # Defender can block with at most one card in this environment loop.
        best_agent_block = 0
        for cid in agent.hand:
            c = env._cards[cid]  # noqa: SLF001
            if c.defense > best_agent_block:
                best_agent_block = c.defense

        playable: list[tuple[int, object, float]] = []
        for i, cid in enumerate(opp.hand):
            card = env._cards[cid]  # noqa: SLF001
            if "attack_action" in card.card_types and card.cost <= opp.resources:
                net_attack = float(card.power - best_agent_block)
                # Prefer cards that keep stronger defense cards for later blocks.
                score = net_attack - 0.1 * float(card.defense)
                playable.append((i, card, score))

        if not playable:
            return None
        idx, _, _ = max(
            playable,
            key=lambda x: (
                x[2],          # max net attack after expected block
                x[1].power,    # tie-break with higher raw attack
                -x[1].cost,    # and cheaper plays
            ),
        )
        return int(idx)

    def select_block_index(self, env: "FleshAndBloodEnvironment") -> Optional[int]:
        if env._pending_combat is None or env._pending_combat.defender != 1:  # noqa: SLF001
            return None

        defender = env._players[1]  # noqa: SLF001
        incoming_attack = float(env._pending_combat.attack_power)  # noqa: SLF001
        best_idx: Optional[int] = None
        best_tuple: Optional[tuple[float, float, float]] = None
        for i, cid in enumerate(defender.hand):
            card = env._cards[cid]  # noqa: SLF001
            if card.defense <= 0:
                continue
            effective_block = min(float(card.defense), incoming_attack)
            overblock = max(0.0, float(card.defense) - incoming_attack)
            # Maximize blocked damage first, then reduce waste and preserve attack.
            rank = (
                effective_block,
                -overblock,
                -float(card.power),
            )
            if best_tuple is None or rank > best_tuple:
                best_tuple = rank
                best_idx = i
        return best_idx
