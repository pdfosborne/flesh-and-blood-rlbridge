"""Heuristic default policy for Talishar action selection.

The goal is intentionally simple and stable:

* in defensive windows, block to minimize incoming damage;
* in offensive windows, pick the highest-attack legal action;
* otherwise, pick a sensible non-pass action and avoid stalling.
"""

from __future__ import annotations

import re
from typing import Any, Optional

_YES_LABELS = ("yes", "confirm", "continue", "ok", "accept", "done")
_PASS_LABELS = ("pass", "end turn", "no block", "skip")
_DEFENSE_HINTS = ("block", "defend", "reaction")
_ATTACK_HINTS = ("attack", "swing")


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize(text: Any) -> str:
    return str(text or "").strip().lower()


def _is_pass_action(action: dict[str, Any]) -> bool:
    label = _normalize(action.get("label", ""))
    if _to_int(action.get("action_code", 0)) == 99:
        return True
    return any(token in label for token in _PASS_LABELS)


def _match_action_card(action: dict[str, Any], state: dict[str, Any]) -> Optional[dict[str, Any]]:
    zone = _normalize(action.get("zone", ""))
    if zone not in {
        "hand",
        "equipment",
        "arsenal",
        "aura",
        "ally",
        "item",
        "permanent",
        "discard",
        "banish",
    }:
        return None

    zone_key_by_name: dict[str, str] = {
        "hand": "playerHand",
        "equipment": "playerEquipment",
        "arsenal": "playerArse",
        "aura": "playerAuras",
        "ally": "playerAllies",
        "item": "playerItems",
        "permanent": "playerPermanents",
        "discard": "playerDiscard",
        "banish": "playerBanish",
    }

    button_input = str(action.get("button_input", ""))
    action_code = _to_int(action.get("action_code", 0))
    for i, raw in enumerate(state.get(zone_key_by_name[zone], [])):
        if not isinstance(raw, dict):
            continue
        if _to_int(raw.get("action", 0)) != action_code:
            continue
        card_button_input = str(raw.get("actionDataOverride", str(i)))
        if card_button_input == button_input:
            return raw
    return None


def _estimate_attack(action: dict[str, Any], state: dict[str, Any]) -> int:
    card = _match_action_card(action, state)
    if isinstance(card, dict):
        power = _to_int(card.get("power", 0), 0)
        if power > 0:
            return power

    label = _normalize(action.get("label", ""))
    # Try loose parsing from labels (e.g., "Attack 4" / "4 atk").
    match = re.search(r"(?:attack|atk)\D*(\d+)|(\d+)\s*(?:attack|atk)", label)
    if match:
        return _to_int(match.group(1) or match.group(2), 0)
    return 0


def _estimate_defense(action: dict[str, Any], state: dict[str, Any]) -> int:
    card = _match_action_card(action, state)
    if isinstance(card, dict):
        defense = _to_int(card.get("defense", 0), 0)
        if defense > 0:
            return defense

    label = _normalize(action.get("label", ""))
    # Try loose parsing from labels (e.g., "Block 3" / "3 def").
    match = re.search(r"(?:block|def)\D*(\d+)|(\d+)\s*(?:block|def)", label)
    if match:
        return _to_int(match.group(1) or match.group(2), 0)
    return 0


def _is_defense_window(state: dict[str, Any], legal_actions: list[dict[str, Any]]) -> bool:
    phase = _normalize(state.get("turnPhase", ""))
    if any(token in phase for token in _DEFENSE_HINTS):
        return True

    prompt = state.get("playerPrompt", {})
    if isinstance(prompt, dict):
        for key in ("promptText", "text", "message"):
            txt = _normalize(prompt.get(key, ""))
            if any(token in txt for token in _DEFENSE_HINTS):
                return True

    for action in legal_actions:
        label = _normalize(action.get("label", ""))
        if "block" in label:
            return True
    return False


def _best_index(indices: list[int], scores: dict[int, tuple[float, ...]]) -> int:
    return max(indices, key=lambda i: scores[i])


def choose_talishar_action_index(
    legal_actions: list[dict[str, Any]],
    state: Optional[dict[str, Any]] = None,
) -> int:
    """Choose a legal action index using a lightweight tactical heuristic."""
    if not legal_actions:
        return 0

    state = state or {}
    non_pass = [i for i, action in enumerate(legal_actions) if not _is_pass_action(action)]
    if not non_pass:
        return 0

    pass_indices = [i for i, action in enumerate(legal_actions) if _is_pass_action(action)]
    pass_index = pass_indices[0] if pass_indices else 0

    # Priority on explicit confirm/yes button choices in popup flows.
    yes_like = []
    for i in non_pass:
        label = _normalize(legal_actions[i].get("label", ""))
        zone = _normalize(legal_actions[i].get("zone", ""))
        if zone in {"button", "popup"} and any(t in label for t in _YES_LABELS):
            yes_like.append(i)
    if yes_like:
        return yes_like[0]

    defensive = _is_defense_window(state, legal_actions)
    if defensive:
        defense_scores: dict[int, tuple[float, ...]] = {}
        defense_candidates: list[int] = []
        for i in non_pass:
            action = legal_actions[i]
            block_value = float(_estimate_defense(action, state))
            if block_value <= 0:
                continue
            # Prefer higher block, then lower potential attack (save attackers).
            defense_scores[i] = (
                block_value,
                -float(_estimate_attack(action, state)),
            )
            defense_candidates.append(i)
        if defense_candidates:
            return _best_index(defense_candidates, defense_scores)
        return pass_index

    # Offensive/default windows: maximize attack, then prefer lower defense cards.
    attack_scores: dict[int, tuple[float, ...]] = {}
    attack_candidates: list[int] = []
    for i in non_pass:
        action = legal_actions[i]
        label = _normalize(action.get("label", ""))
        attack_value = float(_estimate_attack(action, state))
        if attack_value <= 0 and not any(token in label for token in _ATTACK_HINTS):
            continue
        card = _match_action_card(action, state) or {}
        cost = float(_to_int(card.get("cost", 0), 0))
        defense_value = float(_estimate_defense(action, state))
        attack_scores[i] = (
            attack_value,
            -cost,
            -defense_value,
        )
        attack_candidates.append(i)
    if attack_candidates:
        return _best_index(attack_candidates, attack_scores)

    # Generic fallback: pick the first non-pass legal action.
    return non_pass[0]
