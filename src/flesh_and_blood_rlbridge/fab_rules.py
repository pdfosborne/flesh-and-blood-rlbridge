"""Flesh and Blood comprehensive-rules helpers (CR v2.x).

References: https://rules.fabtcg.com/en/cr/08-keywords/
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Action card types per CR — used by Overpower (8.3.22).
ACTION_CARD_TYPES = frozenset(
    {"attack_action", "utility_action", "defense_reaction", "attack_reaction", "instant"}
)


def is_action_card(card_types: tuple[str, ...] | list[str]) -> bool:
    return bool(ACTION_CARD_TYPES.intersection(card_types))


def parse_ward_value(text: str, keywords: tuple[str, ...] | list[str] = (), *, has_blue_pitched: bool = False) -> int:
    """Ward N — destroy this to prevent N damage (CR 8.3.20)."""
    body = str(text or "").lower()
    m = re.search(
        r"ward x,? where x is (\d+) if you(?:'ve| have) pitched a blue card this turn, otherwise (\d+)",
        body,
    )
    if m:
        return int(m.group(1)) if has_blue_pitched else int(m.group(2))
    if "ward" not in keywords and "ward" not in body:
        return 0
    m = re.search(r"ward (\d+)", body)
    return int(m.group(1)) if m else 0


def parse_piercing_value(text: str) -> int:
    """Piercing N — +N power when defended by equipment (CR 8.3.23)."""
    m = re.search(r"piercing (\d+)", str(text or "").lower())
    return int(m.group(1)) if m else 0


@dataclass(frozen=True)
class ClashResult:
    attacker_wins: bool
    defender_wins: bool
    tie: bool
    attacker_power: int
    defender_power: int


def resolve_clash(attacker_top_power: int | None, defender_top_power: int | None) -> ClashResult:
    """Clash — reveal deck tops, greatest power wins (CR 8.5.45)."""
    a = attacker_top_power
    d = defender_top_power
    if a is None and d is None:
        return ClashResult(False, False, True, 0, 0)
    if a is None:
        return ClashResult(False, True, False, 0, d or 0)
    if d is None:
        return ClashResult(True, False, False, a, 0)
    if a > d:
        return ClashResult(True, False, False, a, d)
    if d > a:
        return ClashResult(False, True, False, a, d)
    return ClashResult(False, False, True, a, d)
