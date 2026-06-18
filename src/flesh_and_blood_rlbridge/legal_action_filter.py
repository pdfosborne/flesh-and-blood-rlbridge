"""Shared legal-action filtering for Talishar HTTP and C++ engine environments.

Both :class:`TalisharEngineEnvironment` and :class:`CppEngineEnvironment` expose
the same filtered legal-action list to agents.  All phase-specific loop-breaking
rules live here so the two backends cannot drift apart.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from .talishar_default_policy import (
    _apply_block_phase_filter,
    _BUTTON_INPUT_PHASES,
    _card_pitch_value,
    _CHOOSE_HAND_PHASES,
    _DEFENSE_PHASES,
    _BLOCK_PHASES,
    _get_phase,
    _is_affordable_hand_play,
    _is_pass_action,
    _is_revert_action,
    _POPUP_PHASES,
    _strip_revert_actions,
    _to_int,
)

_PAY_TO_AVOID_RE = re.compile(r"pay\s+(\d+)\s+to\s+avoid", re.IGNORECASE)
_PAY_UNDERSCORE_AVOID_RE = re.compile(
    r"(?:if_you_want_to_)?pay[_ ](\d+)[_ ].*?(?:avoid|to_avoid)",
    re.IGNORECASE,
)
_PAY_ADDITIONAL_COST_RE = re.compile(
    r"pay[_ ]the[_ ]additional[_ ]cost[_ ]of[_ ](\d+)",
    re.IGNORECASE,
)

_PASS_FALLBACK: dict[str, Any] = {
    "action_code": 99,
    "button_input": "",
    "card_id": "",
    "zone": "button",
    "label": "Pass",
}


def normalize_action_descriptor(action: Any) -> dict[str, Any]:
    """Return a Talishar-style action dict from a dict or attribute object."""
    if isinstance(action, dict):
        return {
            "action_code": _to_int(action.get("action_code", 0)),
            "button_input": str(action.get("button_input", "") or ""),
            "card_id": str(action.get("card_id", "") or ""),
            "zone": str(action.get("zone", "") or ""),
            "label": str(action.get("label", "") or ""),
        }
    return {
        "action_code": _to_int(getattr(action, "action_code", 0)),
        "button_input": str(getattr(action, "button_input", "") or ""),
        "card_id": str(getattr(action, "card_id", "") or ""),
        "zone": str(getattr(action, "zone", "") or ""),
        "label": str(getattr(action, "label", "") or ""),
    }


def descriptors_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return normalize_action_descriptor(left) == normalize_action_descriptor(right)


def _prompt_text_from_state(state: dict[str, Any]) -> str:
    """Collect user-facing prompt text that may describe a resource payment."""
    parts: list[str] = []
    prompt = state.get("playerPrompt", {})
    if isinstance(prompt, dict):
        parts.append(str(prompt.get("helpText", "") or ""))

    popup = state.get("playerInputPopUp", {})
    if isinstance(popup, dict):
        inner = popup.get("popup", {})
        if isinstance(inner, dict):
            parts.append(str(inner.get("title", "") or ""))
            parts.append(str(inner.get("additionalComments", "") or ""))

    layers = state.get("activeLayers", [])
    if isinstance(layers, list):
        for layer in layers:
            if isinstance(layer, dict):
                parts.append(str(layer.get("caption", "") or layer.get("title", "") or ""))

    turn_phase = state.get("turnPhase", {})
    if isinstance(turn_phase, dict):
        parts.append(str(turn_phase.get("helpText", "") or ""))

    return " ".join(part for part in parts if part)


def _required_pay_cost_from_state(state: dict[str, Any]) -> Optional[int]:
    """Return the resource cost from a pay-to-avoid prompt, if present."""
    text = _prompt_text_from_state(state).replace("-", "_")
    for pattern in (_PAY_TO_AVOID_RE, _PAY_UNDERSCORE_AVOID_RE, _PAY_ADDITIONAL_COST_RE):
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def _available_pitch_resources(state: dict[str, Any]) -> int:
    """Return pool resources plus total pitch value still available in hand."""
    available = _to_int(state.get("playerPitchCount", 0), 0)
    hand = state.get("playerHand", [])
    if not isinstance(hand, list):
        return available
    for card in hand:
        if isinstance(card, dict):
            available += _card_pitch_value(card)
    return available


def _is_yes_action(action: dict[str, Any]) -> bool:
    code = _to_int(action.get("action_code", 0))
    button = str(action.get("button_input", "") or "").strip().upper()
    if code == 20 and button == "YES":
        return True
    label = str(action.get("label", "") or "").strip().lower()
    return label == "yes"


def _is_no_action(action: dict[str, Any]) -> bool:
    code = _to_int(action.get("action_code", 0))
    button = str(action.get("button_input", "") or "").strip().upper()
    if code == 20 and button == "NO":
        return True
    label = str(action.get("label", "") or "").strip().lower()
    return label == "no"


def _strip_unaffordable_pay_yes_actions(
    state: dict[str, Any],
    filtered: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Force *No* when a pay-to-avoid prompt cannot be afforded from hand.

    Talishar reverts the game state when the player accepts a payment prompt
    but cannot complete the follow-up pitch window.  That revert recreates the
    same YES/NO prompt and traps agents in a loop.
    """
    required = _required_pay_cost_from_state(state)
    if required is None:
        return filtered
    if not any(_is_yes_action(action) for action in filtered):
        return filtered
    if _available_pitch_resources(state) >= required:
        return filtered

    without_yes = [action for action in filtered if not _is_yes_action(action)]
    if without_yes:
        return without_yes

    no_actions = [action for action in filtered if _is_no_action(action)]
    if no_actions:
        return [no_actions[0]]
    return filtered


def _strip_blacklisted_actions(
    filtered: list[dict[str, Any]],
    block_blacklist: frozenset[str] | set[str],
) -> list[dict[str, Any]]:
    """Drop actions whose label/card_id was blacklisted after an aborted play."""
    if not block_blacklist:
        return filtered
    kept = [
        action
        for action in filtered
        if str(action.get("label", "") or "") not in block_blacklist
        and str(action.get("card_id", "") or "") not in block_blacklist
    ]
    if kept:
        return kept
    return filtered


def is_pass_only(filtered: list[dict[str, Any]]) -> bool:
    """Return True when every remaining action is a pass / end-turn no-op."""
    if not filtered:
        return True
    return all(_is_pass_action(action) for action in filtered)


def align_filtered_actions(
    original: list[Any],
    filtered: list[dict[str, Any]],
    *,
    to_descriptor: Callable[[Any], dict[str, Any]] = normalize_action_descriptor,
) -> list[Any]:
    """Map filtered dict descriptors back onto the original typed action objects."""
    out: list[Any] = []
    cursor = 0
    for action in original:
        if cursor >= len(filtered):
            break
        if descriptors_equal(to_descriptor(action), filtered[cursor]):
            out.append(action)
            cursor += 1
    return out


def filter_legal_actions(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    *,
    block_blacklist: frozenset[str] | set[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Remove actions that cause the agent to loop or get stuck.

    Each rule falls back to the original filtered list if it would otherwise
    produce an empty set.

    **Rule 1 — main-phase affordability** (phase ``m``):
        Strip action-27 hand-card plays whose resource cost exceeds the total
        pitch value of all *other* hand cards.

    **Rule 2 — equip removal** (all phases):
        Remove mode-3 (Equip) actions entirely.  Equipment activation plus undo
        is a common agent stall vector.

    **Rule 3 — pitch-phase Pass removal** (phase ``p``):
        Remove Pass (mode=99) from pitch-phase choices.

    **Rule 4 — pitch-phase undo removal** (phase ``p``):
        Never offer Cancel/undo (10000).  When nothing can be pitched, only
        Pass remains so the agent aborts without calling ``RevertGamestate``.

    **Rule 5 — mandatory-choice phase Pass removal**
        (phases: ``choosehand``, ``choosehandcancel``, ``choosemultizone``,
        ``buttoninput``, ``buttoninputnopass``):
        Remove Pass so the agent is forced to make the required pick.

    **Rule 6 — block/defense phase pass forcing** (phases ``b``, ``d``):
        When no viable blockers remain, strip hand block plays so only pass
        remains.

    **Rule 7 — undo / cancel removal** (all phases):
        Handled by :func:`_strip_revert_actions`.  Agents never see undo,
        cancel, or revert actions.

    **Rule 9 — per-turn play blacklist**:
        Drop hand/arsenal plays blacklisted after an unaffordable pitch abort.

    **Rule 8 — unaffordable pay-to-avoid YES removal** (phase ``yesno`` and
        similar popups):
        When a prompt asks to pay *N* resources to avoid damage/effects but the
        acting player's current resources plus hand pitch total is below *N*,
        remove *Yes* so the agent is forced to decline instead of entering a
        Talishar revert loop.
    """
    phase = _get_phase(state)
    filtered = list(legal_actions)

    # ── Rules 1 & 2: main-phase ───────────────────────────────────────────
    if phase == "m":
        affordable: list[dict[str, Any]] = []
        for action in filtered:
            code = _to_int(action.get("action_code", 0))
            if code == 3:
                continue
            if not _is_affordable_hand_play(action, state):
                continue
            affordable.append(action)
        if affordable:
            filtered = affordable

    # ── Rules 3 & 4: pitch-phase ──────────────────────────────────────────
    elif phase == "p":
        pitch_cards = [
            a for a in filtered
            if a.get("zone") == "hand"
            and _to_int(a.get("action_code", 0)) == 27
        ]
        if pitch_cards:
            must_pitch = [
                a for a in filtered
                if not _is_pass_action(a) and not _is_revert_action(a)
            ]
            if must_pitch:
                filtered = must_pitch
        else:
            pass_only = [a for a in filtered if _is_pass_action(a)]
            if pass_only:
                filtered = [pass_only[0]]
            else:
                filtered = [dict(_PASS_FALLBACK)]

    # ── Rule 5: mandatory-choice phases (CanPassPhase=0) ─────────────────
    elif phase in (
        _CHOOSE_HAND_PHASES
        | _BUTTON_INPUT_PHASES
        | _POPUP_PHASES
    ):
        no_pass = [
            a for a in filtered
            if _to_int(a.get("action_code", 0)) != 99
        ]
        if no_pass:
            filtered = no_pass

    # ── Rules 6 & 7: block / defense phases ───────────────────────────────
    elif phase in _BLOCK_PHASES | _DEFENSE_PHASES:
        filtered = _apply_block_phase_filter(
            state,
            filtered,
            block_blacklist=frozenset(block_blacklist),
        )

    # ── Rule 2 (global): strip equipment activation in every phase ─────────
    no_equip = [
        a for a in filtered
        if _to_int(a.get("action_code", 0)) != 3
    ]
    if no_equip:
        filtered = no_equip

    filtered = _strip_revert_actions(phase, filtered)
    filtered = _strip_unaffordable_pay_yes_actions(state, filtered)
    filtered = _strip_blacklisted_actions(filtered, block_blacklist)

    actionable = [a for a in filtered if not _is_pass_action(a)]
    if actionable:
        return filtered

    pass_actions = [a for a in filtered if _is_pass_action(a)]
    if pass_actions:
        return [pass_actions[0]]

    return [dict(_PASS_FALLBACK)]
