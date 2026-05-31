"""Token and banish-zone helpers for the Flesh and Blood simulator."""

from __future__ import annotations

import re

# Known FAB tokens mapped to card IDs in cards.json.
TOKEN_CARD_IDS: dict[str, str] = {
    "seismic surge": "seismic_surge",
    "crouching tiger": "crouching_tiger",
    "fang strike": "fang_strike",
    "slither": "slither",
    "gold": "gold",
    "might": "might",
    "agility": "agility",
    "vigor": "vigor",
    "fealty": "fealty",
    "silver": "silver",
    "runechant": "runechant",
    "toughness": "toughness",
    "frostbite": "frostbite",
    "inertia": "inertia",
    "embodiment of earth": "embodiment_of_earth",
    "ash": "ash",
    "spectral shield": "spectral_shield",
}

# Tokens whose rules text has a start-of-action-phase upkeep we simulate.
ACTION_PHASE_UPKEEP_TOKENS: frozenset[str] = frozenset({"seismic surge"})


def normalize_token_name(name: str) -> str:
    return " ".join(str(name or "").strip().lower().split())


def resolve_token_card_id(name: str, cards_by_name: dict[str, str]) -> str | None:
    """Map a rules-text token name to a card ID."""
    key = normalize_token_name(name)
    if key in TOKEN_CARD_IDS:
        return TOKEN_CARD_IDS[key]
    return cards_by_name.get(key)


def resolve_banished_card_id(name: str, cards_by_name: dict[str, str]) -> str | None:
    """Map a created-in-banish card name (e.g. Crouching Tiger) to a card ID."""
    key = normalize_token_name(name)
    if key in TOKEN_CARD_IDS:
        return TOKEN_CARD_IDS[key]
    return cards_by_name.get(key)


def build_token_name_index(cards: dict) -> dict[str, str]:
    """Build a lowercase name -> card id lookup for token/banish creation."""
    index: dict[str, str] = {}
    for card in cards.values():
        index[normalize_token_name(card.name)] = card.id
        slug = normalize_token_name(card.name.split("---")[-1] if "---" in card.name else card.name)
        index.setdefault(slug, card.id)
    return index


def parse_combo_condition(clause: str) -> str:
    """Return combo condition key from an on-attack clause, or empty string."""
    low = clause.lower()
    m = re.search(
        r"if an? (red|yellow|blue) attack action card was the last attack this combat chain",
        low,
    )
    if m:
        return f"combo_{m.group(1)}"
    return ""
