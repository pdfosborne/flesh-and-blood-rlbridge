"""Shared legal-action filtering for Talishar HTTP and C++ engine environments.

Both :class:`TalisharEngineEnvironment` and :class:`CppEngineEnvironment` expose
the same filtered legal-action list to agents.  Stall loops are handled by
:class:`~flesh_and_blood_rlbridge.state_loop_guard.TurnLoopGuard` in the
environment step path; this module keeps only validity rules (affordability,
illegal UI actions, mandatory choices, priority).
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .talishar_default_policy import (
    _apply_block_phase_filter,
    _BUTTON_INPUT_PHASES,
    _CHOOSE_HAND_PHASES,
    _DEFENSE_PHASES,
    _BLOCK_PHASES,
    _get_phase,
    _is_affordable_arsenal_play,
    _is_affordable_equipment_play,
    _is_affordable_hand_play,
    _is_pass_action,
    _is_revert_action,
    _pitch_hand_actions,
    _POPUP_PHASES,
    _strip_revert_actions,
    _to_int,
    can_pass_phase,
)

_WAITING_FOR_OTHER_PLAYER_RE = re.compile(
    r"waiting\s+for\s+other\s+player",
    re.IGNORECASE,
)

_PASS_FALLBACK: dict[str, Any] = {
    "action_code": 99,
    "button_input": "",
    "card_id": "",
    "zone": "button",
    "label": "Pass",
}

# Phases where hand/arsenal/equipment plays use different semantics (pitch,
# block, or mandatory zone selection) and must not use main-phase affordability.
_SKIP_PLAY_AFFORDABILITY_PHASES: frozenset[str] = (
    frozenset({"p"})
    | _BLOCK_PHASES
    | _DEFENSE_PHASES
    | _CHOOSE_HAND_PHASES
    | _POPUP_PHASES
    | _BUTTON_INPUT_PHASES
)


def _apply_play_affordability_filter(
    state: dict[str, Any],
    filtered: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop hand/arsenal/equipment plays that cannot be paid from hand pitch + pool."""
    affordable: list[dict[str, Any]] = []
    for action in filtered:
        if not _is_affordable_hand_play(action, state):
            continue
        if not _is_affordable_arsenal_play(action, state):
            continue
        if not _is_affordable_equipment_play(action, state):
            continue
        affordable.append(action)
    return affordable


def _pitch_abort_actions(
    legal_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return Cancel when pitch phase has no cards and Pass would be ignored."""
    reverts = [
        normalize_action_descriptor(action)
        for action in legal_actions
        if _is_revert_action(normalize_action_descriptor(action))
    ]
    if reverts:
        return [reverts[0]]
    return [dict(_PASS_FALLBACK)]


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
    """Collect user-facing prompt text from the game state."""
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


def is_waiting_for_other_player(state: dict[str, Any]) -> bool:
    """True when Talishar's prompt says this seat must wait on the opponent."""
    return bool(_WAITING_FOR_OTHER_PLAYER_RE.search(_prompt_text_from_state(state)))


def player_choosing_in_snapshot(state: dict[str, Any]) -> bool:
    """True when Talishar's phase caption indicates this seat is choosing."""
    turn_phase = state.get("turnPhase", {})
    if not isinstance(turn_phase, dict):
        return False
    caption = str(turn_phase.get("caption", "")).strip().lower()
    return caption.startswith("choose ")


def player_must_wait(state: dict[str, Any]) -> bool:
    """True when this player's snapshot should not submit non-pass actions."""
    if state.get("havePriority") is False:
        return True
    return is_waiting_for_other_player(state)


def _pass_only_without_priority(
    state: dict[str, Any],
    filtered: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse to pass when the acting seat lacks priority."""
    if not player_must_wait(state):
        return filtered
    pass_actions = [a for a in filtered if _is_pass_action(a)]
    if pass_actions:
        return [pass_actions[0]]
    if can_pass_phase(state):
        return [dict(_PASS_FALLBACK)]
    return []


def is_pass_only(filtered: list[dict[str, Any]]) -> bool:
    """Return True when every remaining action is a pass / end-turn no-op."""
    if not filtered:
        return True
    return all(_is_pass_action(action) for action in filtered)


def _has_arsenal_from_hand_actions(actions: list[dict[str, Any]]) -> bool:
    """True when hand cards can be moved into arsenal (end-of-turn ARS step)."""
    return any(
        _to_int(a.get("action_code", 0)) == 4
        and str(a.get("zone", "") or "").strip().lower() == "hand"
        for a in actions
    )


def _strip_pass_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove pass / end-turn actions from *actions*."""
    return [a for a in actions if not _is_pass_action(a)]


def align_filtered_actions(
    original: list[Any],
    filtered: list[dict[str, Any]],
    *,
    to_descriptor: Callable[[Any], dict[str, Any]] = normalize_action_descriptor,
) -> list[Any]:
    """Map filtered dict descriptors back onto the original typed action objects."""
    return materialize_filtered_actions(
        original,
        filtered,
        to_descriptor=to_descriptor,
    )


def materialize_filtered_actions(
    original: list[Any],
    filtered: list[dict[str, Any]],
    *,
    to_descriptor: Callable[[Any], dict[str, Any]] = normalize_action_descriptor,
    make_action: Callable[[dict[str, Any]], Any] | None = None,
) -> list[Any]:
    """Map filtered descriptors onto original actions or synthesize missing entries.

    ``filter_legal_actions`` may inject Pass/Cancel that were not in the raw
    engine list; *make_action* builds typed stand-ins when no original matches.
    """
    out: list[Any] = []
    for fdesc in filtered:
        norm = normalize_action_descriptor(fdesc)
        matched: Any | None = None
        for action in original:
            if descriptors_equal(to_descriptor(action), norm):
                matched = action
                break
        if matched is None and make_action is not None:
            matched = make_action(norm)
        elif matched is None:
            matched = dict(norm)
        out.append(matched)
    return out


def filter_legal_actions(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove invalid or impossible actions before presenting them to agents.

    Each rule falls back to the original filtered list if it would otherwise
    produce an empty set.  Decision loops are handled by
    :class:`~flesh_and_blood_rlbridge.state_loop_guard.TurnLoopGuard`, not here.

    **Affordability** (all phases except pitch, block/defense, and mandatory
        zone-selection windows):
        Strip hand, arsenal, and equipment plays whose resource cost exceeds
        floated resources plus pitch value available in hand.

    **Undo / cancel removal** (all phases, applied first and last):
        Strip every undo, cancel, and revert action.

    **Pitch phase**:
        Remove Pass when cards must be pitched; offer Pass-only when the hand
        has nothing to pitch and pass is allowed.

    **Mandatory-choice Pass removal** (``CanPassPhase=0``):
        Remove Pass whenever Talishar would ignore mode=99.

    **End-of-turn arsenal (phase ``ARS``)**:
        When hand cards can be added to arsenal, strip Pass.

    **Block/defense phases**:
        When no viable blockers remain, strip hand block plays so only pass
        remains.

    **No-priority pass-only**:
        When ``havePriority`` is false or the prompt says "waiting for other
        player", strip every action except Pass.
    """
    phase = _get_phase(state)
    filtered = _strip_revert_actions(phase, list(legal_actions))

    if phase not in _SKIP_PLAY_AFFORDABILITY_PHASES:
        affordable = _apply_play_affordability_filter(state, filtered)
        if affordable:
            filtered = affordable

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
        elif can_pass_phase(state):
            pass_only = [a for a in filtered if _is_pass_action(a)]
            if pass_only:
                filtered = [pass_only[0]]
            else:
                filtered = [dict(_PASS_FALLBACK)]
        else:
            filtered = _pitch_abort_actions(legal_actions)

    if not can_pass_phase(state):
        no_pass = _strip_pass_actions(filtered)
        if no_pass:
            filtered = no_pass
    elif phase in (_CHOOSE_HAND_PHASES | _BUTTON_INPUT_PHASES | _POPUP_PHASES):
        no_pass = _strip_pass_actions(filtered)
        if no_pass:
            filtered = no_pass

    if phase == "ars" and _has_arsenal_from_hand_actions(filtered):
        no_pass = _strip_pass_actions(filtered)
        if no_pass:
            filtered = no_pass

    if phase in _BLOCK_PHASES | _DEFENSE_PHASES:
        filtered = _apply_block_phase_filter(state, filtered)

    filtered = _strip_revert_actions(phase, filtered)

    if player_must_wait(state):
        return _pass_only_without_priority(state, filtered)

    if (
        phase == "p"
        and not _pitch_hand_actions(filtered)
        and not can_pass_phase(state)
    ):
        return _pitch_abort_actions(legal_actions)

    actionable = _strip_pass_actions(filtered)
    if actionable:
        return filtered

    if can_pass_phase(state):
        pass_actions = [a for a in filtered if _is_pass_action(a)]
        if pass_actions:
            return [pass_actions[0]]
        return [dict(_PASS_FALLBACK)]

    return filtered

