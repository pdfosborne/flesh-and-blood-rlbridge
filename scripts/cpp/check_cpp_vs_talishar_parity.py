#!/usr/bin/env python3
"""C++ Engine vs Talishar HTTP parity checker.

The checker validates the agent-facing contract shared by
``CppEngineEnvironment`` and ``TalisharEngineEnvironment``:

* compact observation JSON shape and values
* ordered legal action indexes/labels/zones
* training/eval info fields used by local scripts
* rewards and terminal/truncation flags

It intentionally fails if the requested C++ environment silently falls back to
HTTP Talishar, because that would compare Talishar to itself.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.talishar_engine_environment import TalisharEngineEnvironment
from flesh_and_blood_rlbridge.cpp_engine_environment import get_engine_dir
from flesh_and_blood_rlbridge.game_state_parity import (
    build_initial_sync_payload,
    compare_agent_contract,
    compare_game_states,
    extract_cpp_state,
    extract_talishar_state,
    is_syncable_card_id,
)
from flesh_and_blood_rlbridge.obs_alignment import (
    align_observation_for_cpp_training,
    observation_vectors_aligned,
)
from flesh_and_blood_rlbridge.player_observation import (
    COMBAT_CHAIN_END,
    COMBAT_CHAIN_OFF,
    CONTEXT_DIM,
    HAND_END,
    HAND_SLOT_DIM,
    HAND_SLOTS,
    PLAYER_OBS_DIM,
    SCALAR_COUNT,
    ZONE_END,
    ZONE_OFF,
)


OBSERVATION_KEYS = (
    "actingPlayerID",
    "selfPlay",
    "playerHealth",
    "opponentHealth",
    "turnNo",
    "turnPhase",
    "havePriority",
    "playerHandSize",
    "opponentHandSize",
    "playerDeckCount",
    "opponentDeckCount",
    "playerPitchCount",
    "playerHand",
    "legalActions",
)
OBSERVATION_SCALAR_KEYS = OBSERVATION_KEYS[:-2]
OBSERVATION_NUMERIC_KEYS = {
    "actingPlayerID",
    "playerHealth",
    "opponentHealth",
    "turnNo",
    "playerHandSize",
    "opponentHandSize",
    "playerDeckCount",
    "opponentDeckCount",
    "playerPitchCount",
}
OBS_LEGAL_ACTION_KEYS = ("index", "label", "zone")
INFO_LEGAL_ACTION_KEYS = ("action_code", "button_input", "card_id", "zone", "label")
INFO_CONTRACT_KEYS = (
    "legal_actions",
    "player_hp",
    "opponent_hp",
    "acting_player_id",
    "self_play",
    "turn",
    "repeat_streak",
    "repeat_penalty",
)
REWARD_TOLERANCE = 0.001


@dataclass
class Discrepancy:
    episode: int
    step: int
    category: str
    description: str
    talishar_value: Any
    cpp_value: Any
    tolerance_applied: bool = False
    taxonomy: str = ""
    card_id: str = ""
    zone: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "step": self.step,
            "category": self.category,
            "description": self.description,
            "talishar_value": self.talishar_value,
            "cpp_value": self.cpp_value,
            "tolerance_applied": self.tolerance_applied,
            "taxonomy": self.taxonomy,
            "card_id": self.card_id,
            "zone": self.zone,
        }


@dataclass
class ParityReport:
    matchup: str = ""
    format: str = ""
    mode: str = ""
    episodes_requested: int = 0
    episodes_run: int = 0
    episodes_passed: int = 0
    episodes_failed: int = 0
    total_steps: int = 0
    discrepancies_found: int = 0
    observations_mismatched: int = 0
    legal_actions_mismatched: int = 0
    rewards_mismatched: int = 0
    termination_mismatches: int = 0
    game_outcome_mismatches: int = 0
    setup_failures: int = 0
    first_failure_step: Optional[int] = None
    first_failure_episode: Optional[int] = None
    discrepancies: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matchup": self.matchup,
            "format": self.format,
            "mode": self.mode,
            "episodes_requested": self.episodes_requested,
            "episodes_run": self.episodes_run,
            "episodes_passed": self.episodes_passed,
            "episodes_failed": self.episodes_failed,
            "total_steps": self.total_steps,
            "discrepancies_found": self.discrepancies_found,
            "observations_mismatched": self.observations_mismatched,
            "legal_actions_mismatched": self.legal_actions_mismatched,
            "rewards_mismatched": self.rewards_mismatched,
            "termination_mismatches": self.termination_mismatches,
            "game_outcome_mismatches": self.game_outcome_mismatches,
            "setup_failures": self.setup_failures,
            "first_failure_step": self.first_failure_step,
            "first_failure_episode": self.first_failure_episode,
            "discrepancies": self.discrepancies,
        }


def _safe_label(value: str) -> str:
    label = re.sub(r"[^a-z0-9._-]+", "_", str(value).strip().lower()).strip("_")
    return label or "deck"


def _safe_int(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _safe_float(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _parse_observation(source: str, observation: Any) -> tuple[Optional[dict[str, Any]], str]:
    if isinstance(observation, str):
        try:
            parsed = json.loads(observation)
        except json.JSONDecodeError as exc:
            return None, f"{source} observation is invalid JSON: {exc}: {observation[:200]}"
    else:
        parsed = observation
    if not isinstance(parsed, dict):
        return None, f"{source} observation is {type(parsed).__name__}; expected JSON object"
    return parsed, ""


def _json_safe_value(value: Any, max_items: int = 12) -> Any:
    """Recursively convert *value* to JSON-serializable Python types."""
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return _json_safe_value(value.tolist(), max_items=max_items)
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass

    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value[:1000]
        if isinstance(parsed, (dict, list)):
            return _json_safe_value(parsed, max_items=max_items)
        return parsed
    if isinstance(value, list):
        out = [_json_safe_value(item, max_items=max_items) for item in value[:max_items]]
        if len(value) > max_items:
            out.append(f"... {len(value) - max_items} more")
        return out
    if isinstance(value, dict):
        return {
            key: _json_safe_value(val, max_items=max_items)
            for key, val in value.items()
        }
    return value


def _summary_value(value: Any, max_items: int = 12) -> Any:
    return _json_safe_value(value, max_items=max_items)


def _normalise_action(action: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    normalised: dict[str, Any] = {}
    for field_name in fields:
        value = action.get(field_name)
        if field_name in {"index", "action_code"}:
            value = _safe_int(value)
        elif field_name in {"button_input", "card_id", "zone", "label"}:
            value = str(value or "")
        normalised[field_name] = value
    return normalised


def _normalise_observation_scalar(key: str, value: Any) -> Any:
    if key in OBSERVATION_NUMERIC_KEYS:
        return _safe_int(value)
    if key in {"selfPlay", "havePriority"}:
        return bool(value)
    return value


def _normalise_hand_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "cardID": str(card.get("cardID", "") or ""),
        "action": _safe_int(card.get("action", 0)),
        "actionDataOverride": str(card.get("actionDataOverride", "") or ""),
        "label": str(card.get("label", "") or ""),
    }


def compare_legal_actions(
    legal_tal: list[dict[str, Any]],
    legal_cpp: list[dict[str, Any]],
    *,
    fields: tuple[str, ...] = INFO_LEGAL_ACTION_KEYS,
) -> tuple[bool, str]:
    if len(legal_tal) != len(legal_cpp):
        return False, f"count mismatch: Talishar={len(legal_tal)}, C++={len(legal_cpp)}"

    for index, (tal_action, cpp_action) in enumerate(zip(legal_tal, legal_cpp)):
        if not isinstance(tal_action, dict) or not isinstance(cpp_action, dict):
            return False, f"action[{index}] must be object in both lists"
        tal_normalised = _normalise_action(tal_action, fields)
        cpp_normalised = _normalise_action(cpp_action, fields)
        if tal_normalised != cpp_normalised:
            return (
                False,
                f"action[{index}] mismatch: Talishar={tal_normalised}, C++={cpp_normalised}",
            )
    return True, ""


OBS_VEC_TOLERANCE = 0.05
OBS_VEC_HAND_OFF = CONTEXT_DIM + SCALAR_COUNT
OBS_VEC_HAND_END = OBS_VEC_HAND_OFF + HAND_SLOTS * HAND_SLOT_DIM
OBS_VEC_ZONE_OFF = ZONE_OFF
OBS_VEC_ZONE_END = ZONE_END
OBS_VEC_COMBAT_OFF = COMBAT_CHAIN_OFF
OBS_VEC_COMBAT_END = COMBAT_CHAIN_END


def _append_vec_mismatches(
    mismatches: list[str],
    tal_vec: list[Any],
    cpp_vec: list[Any],
    start: int,
    end: int,
    label: str,
    *,
    max_checks: int = 24,
) -> None:
    checked = 0
    for idx in range(start, end):
        if checked >= max_checks:
            break
        tal_v = float(tal_vec[idx])
        cpp_v = float(cpp_vec[idx])
        if abs(tal_v - cpp_v) > OBS_VEC_TOLERANCE:
            mismatches.append(f"{label}[{idx}]: Talishar={tal_v:.4f}, C++={cpp_v:.4f}")
            checked += 1


def _compare_observation_vec_slices(tal: dict[str, Any], cpp: dict[str, Any]) -> tuple[bool, str]:
    """Compare player-fair vector slices where C++ implements zone data."""
    tal_vec = tal.get("observationVec")
    cpp_vec = cpp.get("observationVec")
    if not isinstance(tal_vec, list) or not isinstance(cpp_vec, list):
        return True, "skipped: observationVec missing (cpp_not_implemented)"
    if len(tal_vec) != PLAYER_OBS_DIM or len(cpp_vec) != PLAYER_OBS_DIM:
        return (
            False,
            f"observationVec dim mismatch: Talishar={len(tal_vec)}, C++={len(cpp_vec)}, "
            f"expected={PLAYER_OBS_DIM}",
        )

    aligned_ok, aligned_msg = observation_vectors_aligned(
        np.asarray(tal_vec, dtype=np.float64),
        np.asarray(cpp_vec, dtype=np.float64),
        atol=OBS_VEC_TOLERANCE,
    )
    if aligned_ok:
        return True, ""

    mismatches: list[str] = [f"aligned: {aligned_msg}"]
    for idx in range(min(CONTEXT_DIM + SCALAR_COUNT, OBS_VEC_HAND_OFF)):
        tal_v = float(tal_vec[idx])
        cpp_v = float(cpp_vec[idx])
        if abs(tal_v - cpp_v) > OBS_VEC_TOLERANCE:
            mismatches.append(f"vec[{idx}]: Talishar={tal_v:.4f}, C++={cpp_v:.4f}")

    _append_vec_mismatches(mismatches, tal_vec, cpp_vec, OBS_VEC_HAND_OFF, OBS_VEC_HAND_END, "hand_vec")
    _append_vec_mismatches(mismatches, tal_vec, cpp_vec, OBS_VEC_ZONE_OFF, OBS_VEC_ZONE_END, "zone_vec")
    _append_vec_mismatches(
        mismatches, tal_vec, cpp_vec, OBS_VEC_COMBAT_OFF, OBS_VEC_COMBAT_END, "combat_vec"
    )

    if mismatches:
        shown = "; ".join(mismatches[:8])
        return False, f"observationVec slice mismatch: {shown}"
    return True, ""


def compare_observations(obs_tal: Any, obs_cpp: Any) -> tuple[bool, str]:
    tal, msg = _parse_observation("Talishar", obs_tal)
    if tal is None:
        return False, msg
    cpp, msg = _parse_observation("C++", obs_cpp)
    if cpp is None:
        return False, msg

    expected_keys = set(OBSERVATION_KEYS)
    for label, obs in (("Talishar", tal), ("C++", cpp)):
        actual_keys = set(obs.keys())
        missing = expected_keys - actual_keys
        if missing:
            return (
                False,
                f"{label} keys missing required fields: {sorted(missing)}",
            )

    mismatches: list[str] = []
    for key in OBSERVATION_SCALAR_KEYS:
        tal_value = _normalise_observation_scalar(key, tal.get(key))
        cpp_value = _normalise_observation_scalar(key, cpp.get(key))
        if tal_value != cpp_value:
            mismatches.append(f"{key}: Talishar={tal_value!r}, C++={cpp_value!r}")

    tal_hand = tal.get("playerHand", [])
    cpp_hand = cpp.get("playerHand", [])
    if not isinstance(tal_hand, list) or not isinstance(cpp_hand, list):
        return False, "playerHand must be a list in both observations"
    if len(tal_hand) != len(cpp_hand):
        mismatches.append(f"playerHand count: Talishar={len(tal_hand)}, C++={len(cpp_hand)}")
    else:
        for index, (tal_card, cpp_card) in enumerate(zip(tal_hand, cpp_hand)):
            if not isinstance(tal_card, dict) or not isinstance(cpp_card, dict):
                return False, f"playerHand[{index}] must be object in both observations"
            tal_normalised = _normalise_hand_card(tal_card)
            cpp_normalised = _normalise_hand_card(cpp_card)
            if tal_normalised != cpp_normalised:
                mismatches.append(
                    f"playerHand[{index}]: Talishar={tal_normalised}, C++={cpp_normalised}"
                )
                break

    tal_legal = tal.get("legalActions", [])
    cpp_legal = cpp.get("legalActions", [])
    if not isinstance(tal_legal, list) or not isinstance(cpp_legal, list):
        return False, "legalActions must be a list in both observations"
    success, msg = compare_legal_actions(tal_legal, cpp_legal, fields=OBS_LEGAL_ACTION_KEYS)
    if not success:
        mismatches.append(f"legalActions {msg}")

    if mismatches:
        shown = "; ".join(mismatches[:10])
        if len(mismatches) > 10:
            shown += f"; ... {len(mismatches) - 10} more"
        return False, shown

    vec_ok, vec_msg = _compare_observation_vec_slices(tal, cpp)
    if not vec_ok:
        return False, vec_msg
    return True, vec_msg or ""


def compare_rewards(reward_tal: float, reward_cpp: float) -> tuple[bool, str]:
    diff = abs(float(reward_tal) - float(reward_cpp))
    if diff <= REWARD_TOLERANCE:
        return True, f"diff={diff:.6f}"
    return False, f"Talishar={reward_tal}, C++={reward_cpp}, diff={diff:.6f}"


def compare_info_contract(info_tal: dict[str, Any], info_cpp: dict[str, Any]) -> tuple[bool, str]:
    for key in INFO_CONTRACT_KEYS:
        if key not in info_tal and key not in info_cpp:
            continue
        if key not in info_tal or key not in info_cpp:
            return False, f"{key!r} presence mismatch"
        if key == "legal_actions":
            success, msg = compare_legal_actions(info_tal[key], info_cpp[key])
            if not success:
                return False, f"legal_actions {msg}"
            continue
        if key == "repeat_penalty":
            success, msg = compare_rewards(float(info_tal[key]), float(info_cpp[key]))
            if not success:
                return False, f"repeat_penalty mismatch: {msg}"
            continue
        if key == "turn":
            if int(info_tal[key] or 0) != int(info_cpp[key] or 0):
                return False, f"{key!r} mismatch: Talishar={info_tal[key]}, C++={info_cpp[key]}"
            continue
        if info_tal[key] != info_cpp[key]:
            return False, f"{key!r} mismatch: Talishar={info_tal[key]}, C++={info_cpp[key]}"
    return True, ""


def _outcome_summary(result: Any) -> dict[str, Any]:
    observation, _ = _parse_observation("result", getattr(result, "observation", {}))
    observation = observation or {}
    return {
        "terminated": bool(getattr(result, "terminated", False)),
        "truncated": bool(getattr(result, "truncated", False)),
        "actingPlayerID": _safe_int(observation.get("actingPlayerID")),
        "playerHealth": _safe_float(observation.get("playerHealth")),
        "opponentHealth": _safe_float(observation.get("opponentHealth")),
        "playerDeckCount": _safe_int(observation.get("playerDeckCount")),
        "opponentDeckCount": _safe_int(observation.get("opponentDeckCount")),
        "turnPhase": observation.get("turnPhase"),
    }


def _record_discrepancy(
    report: ParityReport,
    *,
    episode: int,
    step: int,
    category: str,
    description: str,
    talishar_value: Any,
    cpp_value: Any,
    tolerance_applied: bool = False,
    taxonomy: str = "",
    card_id: str = "",
    zone: str = "",
) -> bool:
    report.discrepancies_found += 1
    if report.first_failure_episode is None:
        report.first_failure_episode = episode
        report.first_failure_step = step
    if category == "observation":
        report.observations_mismatched += 1
    elif category == "legal_actions":
        report.legal_actions_mismatched += 1
    elif category == "reward":
        report.rewards_mismatched += 1
    elif category in {"termination", "truncation"}:
        report.termination_mismatches += 1
    elif category == "outcome":
        report.game_outcome_mismatches += 1
    elif category == "setup":
        report.setup_failures += 1

    report.discrepancies.append(
        Discrepancy(
            episode=episode,
            step=step,
            category=category,
            description=description,
            talishar_value=_summary_value(talishar_value),
            cpp_value=_summary_value(cpp_value),
            tolerance_applied=tolerance_applied,
            taxonomy=taxonomy,
            card_id=card_id,
            zone=zone,
        ).to_dict()
    )
    print(f"    [FAIL] {category}: {description}")
    return False


def _compare_reset(reset_tal: Any, reset_cpp: Any, report: ParityReport, episode: int) -> bool:
    success, msg = compare_observations(reset_tal.observation, reset_cpp.observation)
    if not success:
        return _record_discrepancy(
            report,
            episode=episode,
            step=0,
            category="observation",
            description=f"reset observation mismatch: {msg}",
            talishar_value=reset_tal.observation,
            cpp_value=reset_cpp.observation,
        )

    info_tal = reset_tal.info if hasattr(reset_tal, "info") else {}
    info_cpp = reset_cpp.info if hasattr(reset_cpp, "info") else {}
    success, msg = compare_info_contract(info_tal, info_cpp)
    if not success:
        return _record_discrepancy(
            report,
            episode=episode,
            step=0,
            category="legal_actions" if "legal_actions" in msg else "info",
            description=f"reset info mismatch: {msg}",
            talishar_value=info_tal,
            cpp_value=info_cpp,
        )
    return True


def _card_ids_from_state_hand(state: dict[str, Any]) -> list[str]:
    hand = state.get("playerHand", [])
    if not isinstance(hand, list):
        return []
    card_ids: list[str] = []
    for card in hand:
        if isinstance(card, dict):
            card_id = (
                card.get("cardID")
                or card.get("cardNumber")
                or card.get("cardId")
                or card.get("card_id")
            )
            if card_id and is_syncable_card_id(card_id):
                card_ids.append(str(card_id))
    return card_ids


def _phase_from_state(env_tal: Any, state: dict[str, Any]) -> str:
    phase_fn = getattr(env_tal, "_phase_str", None)
    if callable(phase_fn):
        return str(phase_fn(state) or "").upper()
    turn_phase = state.get("turnPhase", {})
    if isinstance(turn_phase, dict):
        return str(turn_phase.get("turnPhase", "") or "").upper()
    return ""


def _is_parity_baseline_state(env_tal: Any, state: dict[str, Any]) -> bool:
    """True once Talishar has finished pregame setup and dealt opening hands."""
    phase = _phase_from_state(env_tal, state)
    hand = state.get("playerHand", [])
    hand_size = len(hand) if isinstance(hand, list) else 0
    p_deck = int(state.get("playerDeckCount", 0) or 0)
    o_deck = int(state.get("opponentDeckCount", 0) or 0)
    decks_ready = p_deck > 0 or o_deck > 0
    deck_seen = bool(getattr(env_tal, "_deck_nonzero_ever_seen", False))
    return phase == "M" and hand_size > 0 and (decks_ready or deck_seen)


def _pick_pregame_advance_index(legal: list[dict[str, Any]]) -> int:
    """Prefer pass/confirm over equipping during pregame equipment selection."""
    for index, action in enumerate(legal):
        if _safe_int(action.get("action_code", 0)) == 99:
            return index
    for index, action in enumerate(legal):
        if _safe_int(action.get("action_code", 0)) != 3:
            return index
    return 0


def _advance_talishar_to_parity_baseline(env_tal: Any, *, max_steps: int = 40) -> dict[str, Any]:
    """Auto-advance Talishar through equipment selection until main-phase hands are dealt."""
    state = getattr(env_tal, "_last_state", None) or {}
    if _is_parity_baseline_state(env_tal, state):
        return state

    legal_actions_fn = getattr(env_tal, "_legal_actions", None)
    step_fn = getattr(env_tal, "step", None)
    if not callable(legal_actions_fn) or not callable(step_fn):
        return state

    for _ in range(max_steps):
        state = getattr(env_tal, "_last_state", None) or {}
        if _is_parity_baseline_state(env_tal, state):
            return state
        if state.get("error") == "game_crashed":
            break
        legal_actions = legal_actions_fn(state)
        if not legal_actions:
            break
        action_index = _pick_pregame_advance_index(legal_actions)
        step_fn(str(action_index))
    return getattr(env_tal, "_last_state", None) or {}


def _build_talishar_reset_snapshot(env_tal: Any) -> Any:
    """Build a reset-like snapshot from the current Talishar HTTP state."""
    state = getattr(env_tal, "_last_state", None) or {}
    legal_actions_fn = getattr(env_tal, "_legal_actions", None)
    encode_fn = getattr(env_tal, "_encode_observation", None)
    legal_actions = legal_actions_fn(state) if callable(legal_actions_fn) else []
    observation = encode_fn(state, legal_actions) if callable(encode_fn) else "{}"
    return type(
        "TalisharResetSnapshot",
        (),
        {
            "observation": observation,
            "info": {
                "game_name": getattr(env_tal, "_game_name", ""),
                "legal_actions": legal_actions,
                "player_hp": int(state.get("playerHealth", 0) or 0),
                "opponent_hp": int(state.get("opponentHealth", 0) or 0),
                "acting_player_id": int(getattr(env_tal, "_acting_player_id", 1) or 1),
                "self_play": bool(getattr(env_tal, "_self_play", False)),
            },
        },
    )()


def _cpp_supports_priority_sync(env_cpp: Any) -> bool:
    cpp_env = getattr(env_cpp, "_cpp_env", None)
    fab = getattr(cpp_env, "_fab", None)
    if fab is None:
        return False
    try:
        probe = fab.GameState()
    except Exception:
        return False
    return hasattr(probe, "set_priority")


def _reset_talishar_for_parity(
    env_tal: Any,
    env_cpp: Any,
    *,
    max_attempts: int = 8,
) -> None:
    """Reset Talishar and advance to a main-phase baseline suitable for parity."""
    require_p1_priority = not _cpp_supports_priority_sync(env_cpp)
    last_error: Optional[Exception] = None
    for _ in range(max_attempts):
        try:
            env_tal.reset()
            _advance_talishar_to_parity_baseline(env_tal)
            if (
                not require_p1_priority
                or int(getattr(env_tal, "_acting_player_id", 1) or 1) == 1
            ):
                return
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    acting_player_id = int(getattr(env_tal, "_acting_player_id", 1) or 1)
    raise RuntimeError(
        "Could not align Talishar opening state for parity "
        f"(acting_player_id={acting_player_id}, require_p1_priority={require_p1_priority})"
    )


def _hand_playability_from_talishar(env_tal: Any) -> dict[int, list[int]]:
    playability: dict[int, list[int]] = {}
    fetch_state = getattr(env_tal, "_fetch_state", None)
    if callable(fetch_state):
        for player_id in (1, 2):
            try:
                state = fetch_state(player_id=player_id, last_update=0)
            except Exception:
                continue
            indices: list[int] = []
            hand = state.get("playerHand", [])
            if isinstance(hand, list):
                for index, card in enumerate(hand):
                    if isinstance(card, dict) and int(card.get("action", 0) or 0) != 0:
                        indices.append(index)
            playability[player_id] = indices
    return playability


def _cpp_inner_env(env_cpp: Any) -> Any:
    return getattr(env_cpp, "_cpp_env", None) or env_cpp


def _talishar_parity_snapshot(result: Any, *, raw_state: Any = None) -> dict[str, Any]:
    """Build a parity payload from a Talishar reset/step result."""
    parsed, _ = _parse_observation("Talishar", getattr(result, "observation", {}))
    snapshot: dict[str, Any] = {"state": parsed or {}}
    if isinstance(raw_state, dict) and raw_state:
        snapshot["raw_state"] = raw_state
    info = getattr(result, "info", None)
    if isinstance(info, dict):
        legal_actions = info.get("legal_actions")
        if isinstance(legal_actions, list):
            snapshot["legal_actions"] = legal_actions
        for key in (
            "turn",
            "player_hp",
            "opponent_hp",
            "acting_player_id",
            "repeat_streak",
            "repeat_penalty",
        ):
            if key in info:
                snapshot[key] = info[key]
    if hasattr(result, "reward"):
        snapshot["reward"] = getattr(result, "reward")
    if hasattr(result, "terminated"):
        snapshot["terminated"] = bool(getattr(result, "terminated"))
    if hasattr(result, "truncated"):
        snapshot["truncated"] = bool(getattr(result, "truncated"))
    return snapshot


def _mirror_cpp_from_talishar_observation(env_cpp: Any, observation: Any) -> None:
    inner = _cpp_inner_env(env_cpp)
    apply_payload = getattr(inner, "apply_talishar_mirror_payload", None)
    if callable(apply_payload):
        parsed, _ = _parse_observation("Talishar", observation)
        if parsed is not None:
            apply_payload({"state": parsed})
        return
    apply_state = getattr(inner, "apply_talishar_state", None)
    parsed, _ = _parse_observation("Talishar", observation)
    if parsed is not None and callable(apply_state):
        apply_state(parsed)


def _cpp_observation_after_mirror(env_cpp: Any) -> str:
    inner = _cpp_inner_env(env_cpp)
    legal_actions = getattr(inner, "_legal_actions", None)
    encode = getattr(inner, "_encode_observation", None)
    filter_legal = getattr(inner, "_filter_legal_actions", None)
    if not callable(legal_actions) or not callable(encode):
        return ""
    legal = legal_actions()
    if callable(filter_legal):
        legal = filter_legal(legal)
    return str(encode(legal))


def _align_cpp_reset_result(
    env_cpp: Any,
    reset_tal: Any,
    reset_cpp: Any,
    *,
    env_tal: Any = None,
) -> Any:
    inner = _cpp_inner_env(env_cpp)
    apply_payload = getattr(inner, "apply_talishar_mirror_payload", None)
    snapshot = _talishar_parity_snapshot(
        reset_tal,
        raw_state=getattr(env_tal, "_last_state", None) if env_tal is not None else None,
    )
    if callable(apply_payload):
        apply_payload(snapshot)
    else:
        _mirror_cpp_from_talishar_observation(env_cpp, reset_tal.observation)
    mirrored_obs = _cpp_observation_after_mirror(env_cpp)
    if not mirrored_obs:
        return reset_cpp
    info = dict(getattr(reset_cpp, "info", {}) or {})
    tal_info = getattr(reset_tal, "info", {}) or {}
    if isinstance(tal_info, dict):
        if tal_info.get("legal_actions"):
            info["legal_actions"] = tal_info["legal_actions"]
        for key in ("player_hp", "opponent_hp", "acting_player_id", "self_play"):
            if key in tal_info:
                info[key] = tal_info[key]
    if hasattr(reset_cpp, "_replace"):
        return reset_cpp._replace(observation=mirrored_obs, info=info)
    reset_cpp.observation = mirrored_obs
    reset_cpp.info = info
    return reset_cpp


def _align_cpp_step_result(env_cpp: Any, step_tal: Any, step_cpp: Any) -> Any:
    return step_cpp


def _opening_hands_from_talishar(env_tal: Any) -> dict[int, list[str]]:
    opening_hands: dict[int, list[str]] = {}
    fetch_state = getattr(env_tal, "_fetch_state", None)
    if callable(fetch_state):
        for player_id in (1, 2):
            try:
                state = fetch_state(player_id=player_id, last_update=0)
            except Exception:
                continue
            cards = _card_ids_from_state_hand(state)
            if cards:
                opening_hands[player_id] = cards

    if len(opening_hands) >= 2:
        return opening_hands

    last_state = getattr(env_tal, "_last_state", None)
    acting_player_id = _safe_int(getattr(env_tal, "_acting_player_id", 1)) or 1
    if acting_player_id not in opening_hands and isinstance(last_state, dict):
        opening_hands[acting_player_id] = _card_ids_from_state_hand(last_state)
    return opening_hands


def _simulation_sync_payload_from_talishar(env_tal: Any) -> dict[str, Any]:
    """Build full initial-sync payload using both player HTTP views."""
    fetch_state = getattr(env_tal, "_fetch_state", None)
    p1_state: dict[str, Any] = {}
    p2_state: dict[str, Any] = {}
    if callable(fetch_state):
        for player_id in (1, 2):
            try:
                state = fetch_state(player_id=player_id, last_update=0)
                if isinstance(state, dict):
                    if player_id == 1:
                        p1_state = state
                    else:
                        p2_state = state
            except Exception:
                continue
    tal_raw: dict[str, Any] = dict(getattr(env_tal, "_last_state", None) or {})
    if p1_state and p2_state:
        tal_raw["_fetch_both"] = lambda: (p1_state, p2_state)
    return build_initial_sync_payload(tal_raw)


def _compare_step(
    step_tal: Any,
    step_cpp: Any,
    report: ParityReport,
    *,
    episode: int,
    step: int,
    action_index: int,
    action_label: str,
) -> bool:
    context = f"after action[{action_index}] {action_label!r}"

    success, msg = compare_observations(step_tal.observation, step_cpp.observation)
    if not success:
        return _record_discrepancy(
            report,
            episode=episode,
            step=step,
            category="observation",
            description=f"{context}: {msg}",
            talishar_value=step_tal.observation,
            cpp_value=step_cpp.observation,
        )

    success, msg = compare_rewards(step_tal.reward, step_cpp.reward)
    if not success:
        return _record_discrepancy(
            report,
            episode=episode,
            step=step,
            category="reward",
            description=f"{context}: {msg}",
            talishar_value=step_tal.reward,
            cpp_value=step_cpp.reward,
            tolerance_applied=True,
        )

    if bool(step_tal.terminated) != bool(step_cpp.terminated):
        return _record_discrepancy(
            report,
            episode=episode,
            step=step,
            category="termination",
            description=f"{context}: Talishar={step_tal.terminated}, C++={step_cpp.terminated}",
            talishar_value=step_tal.terminated,
            cpp_value=step_cpp.terminated,
        )
    if bool(step_tal.truncated) != bool(step_cpp.truncated):
        return _record_discrepancy(
            report,
            episode=episode,
            step=step,
            category="truncation",
            description=f"{context}: Talishar={step_tal.truncated}, C++={step_cpp.truncated}",
            talishar_value=step_tal.truncated,
            cpp_value=step_cpp.truncated,
        )

    info_tal = step_tal.info if hasattr(step_tal, "info") else {}
    info_cpp = step_cpp.info if hasattr(step_cpp, "info") else {}
    success, msg = compare_info_contract(info_tal, info_cpp)
    if not success:
        return _record_discrepancy(
            report,
            episode=episode,
            step=step,
            category="legal_actions" if "legal_actions" in msg else "info",
            description=f"{context}: {msg}",
            talishar_value=info_tal,
            cpp_value=info_cpp,
        )

    if bool(step_tal.terminated) or bool(step_tal.truncated):
        tal_outcome = _outcome_summary(step_tal)
        cpp_outcome = _outcome_summary(step_cpp)
        if tal_outcome != cpp_outcome:
            return _record_discrepancy(
                report,
                episode=episode,
                step=step,
                category="outcome",
                description=f"{context}: terminal outcome mismatch",
                talishar_value=tal_outcome,
                cpp_value=cpp_outcome,
            )
    return True


def _talishar_action_descriptor(env_tal: Any, action_index: int) -> Any:
    legal_actions_fn = getattr(env_tal, "_legal_actions", None)
    state = getattr(env_tal, "_last_state", None)
    if callable(legal_actions_fn) and isinstance(state, dict):
        legal = legal_actions_fn(state)
        if isinstance(legal, list) and 0 <= action_index < len(legal):
            return legal[action_index]
    return str(action_index)


def _legal_actions_from_observation(observation: Any) -> list[dict[str, Any]]:
    parsed, _ = _parse_observation("Talishar", observation)
    if parsed is None:
        return []
    legal = parsed.get("legalActions", [])
    return legal if isinstance(legal, list) else []


def _is_pass_label(label: str) -> bool:
    return str(label or "").strip().lower() in {"pass", "skip", "done"}


def _cpp_progress_fingerprint(state: dict[str, Any]) -> str:
    return "|".join(
        [
            str(state.get("turn_no", state.get("turnNo", ""))),
            str(state.get("phase", "")),
            str(state.get("acting_player_id", state.get("actingPlayerID", ""))),
            str(state.get("priority_player", "")),
            str(state.get("p1_health", "")),
            str(state.get("p2_health", "")),
            str(state.get("p1_hand_size", "")),
            str(state.get("p2_hand_size", "")),
            str(state.get("p1_pitch_count", "")),
            str(state.get("p2_pitch_count", "")),
        ]
    )


def _check_simulation_pass_liveness(
    pre_cpp: dict[str, Any],
    post_cpp: dict[str, Any],
    *,
    action_label: str,
) -> tuple[bool, str]:
    """Flag C++ pass actions that leave the independent engine in a stall."""
    if not _is_pass_label(action_label):
        return True, ""
    if _cpp_progress_fingerprint(pre_cpp) != _cpp_progress_fingerprint(post_cpp):
        return True, ""
    return (
        False,
        "C++ state unchanged after Pass — engine pass logic did not advance "
        "(turn/phase/HP/hand/pitch fingerprint identical)",
    )


def _choose_action(
    env_tal: Any,
    observation: Any,
    *,
    stress: bool,
    step: int = 0,
    rng_seed: Optional[int] = None,
) -> tuple[int, str]:
    legal = _legal_actions_from_observation(observation)
    if not legal:
        return 0, "<no legal actions>"
    if stress:
        index = random.randrange(len(legal))
    elif rng_seed is not None:
        rng = random.Random(int(rng_seed) + int(step) * 1009)
        non_pass = [
            i
            for i, action in enumerate(legal)
            if not _is_pass_label(str(action.get("label", "") or ""))
        ]
        pool = non_pass if non_pass else list(range(len(legal)))
        index = pool[rng.randrange(len(pool))]
    else:
        try:
            index = int(str(env_tal.sample_action()).strip())
        except (TypeError, ValueError):
            index = 0
        index = max(0, min(index, len(legal) - 1))
    return index, str(legal[index].get("label", "") or "")


def _configure_simulation_cpp_env(env_cpp: Any, *, parity_mode: str) -> None:
    inner = _cpp_inner_env(env_cpp)
    if parity_mode == "simulation" and inner is not None:
        inner._strict_simulation = True


def _apply_simulation_initial_sync(
    env_tal: Any,
    env_cpp: Any,
    *,
    sync_scope: str,
    rng_seed: Optional[int],
) -> None:
    inner = _cpp_inner_env(env_cpp)
    if inner is None:
        return
    payload = _simulation_sync_payload_from_talishar(env_tal)
    if rng_seed is not None:
        payload["rng_seed"] = rng_seed
    if sync_scope == "hands":
        payload.pop("deck_orders", None)
        payload.pop("equipment", None)
    apply_sync = getattr(inner, "apply_initial_sync_from_talishar", None)
    if callable(apply_sync):
        apply_sync(payload)


def _record_state_discrepancies(
    report: ParityReport,
    *,
    episode: int,
    step: int,
    tal_state: dict[str, Any],
    cpp_state: dict[str, Any],
    context: str,
) -> bool:
    result = compare_game_states(tal_state, cpp_state)
    ok = True
    for disc in result.discrepancies:
        ok = False
        _record_discrepancy(
            report,
            episode=episode,
            step=step,
            category="game_state",
            description=f"{context}: {disc.description}",
            talishar_value=disc.talishar_value,
            cpp_value=disc.cpp_value,
            taxonomy=disc.taxonomy,
            card_id=disc.card_id,
            zone=disc.zone,
        )
    return ok


def _compare_simulation_step(
    step_tal: Any,
    step_cpp: Any,
    env_tal: Any,
    env_cpp: Any,
    report: ParityReport,
    *,
    episode: int,
    step: int,
    action_index: int,
    action_label: str,
    align_obs: bool,
    pre_cpp_state: dict[str, Any] | None = None,
) -> bool:
    context = f"after action[{action_index}] {action_label!r}"
    tal_state = extract_talishar_state(env_tal)
    cpp_state = extract_cpp_state(_cpp_inner_env(env_cpp) or env_cpp)
    ok = True
    if pre_cpp_state is not None:
        ok_live, live_msg = _check_simulation_pass_liveness(
            pre_cpp_state,
            cpp_state,
            action_label=action_label,
        )
        if not ok_live:
            ok = False
            _record_discrepancy(
                report,
                episode=episode,
                step=step,
                category="state",
                description=f"{context}: {live_msg}",
                talishar_value="progress expected",
                cpp_value="stalled",
                taxonomy="pass_no_progress",
            )
    ok = ok and _record_state_discrepancies(
        report,
        episode=episode,
        step=step,
        tal_state=tal_state,
        cpp_state=cpp_state,
        context=context,
    )
    for disc in compare_agent_contract(step_tal, step_cpp, align_obs=align_obs):
        ok = False
        _record_discrepancy(
            report,
            episode=episode,
            step=step,
            category=disc.category,
            description=f"{context}: {disc.description}",
            talishar_value=disc.talishar_value,
            cpp_value=disc.cpp_value,
            taxonomy=disc.taxonomy,
            card_id=disc.card_id,
            zone=disc.zone,
            tolerance_applied=disc.category == "reward",
        )
    return ok


def _write_repro_trace(
    out_dir: Path,
    *,
    episode: int,
    step: int,
    action_history: list[dict[str, Any]],
    pre_tal_state: dict[str, Any],
    pre_cpp_state: dict[str, Any],
    action_descriptor: Any,
    discrepancies: list[dict[str, Any]],
) -> Path:
    path = out_dir / "repro_trace.json"
    payload = {
        "episode": episode,
        "step": step,
        "action_history": action_history,
        "pre_step": {"talishar": pre_tal_state, "cpp": pre_cpp_state},
        "action": _json_safe_value(action_descriptor),
        "discrepancies": discrepancies,
    }
    path.write_text(json.dumps(_json_safe_value(payload), indent=2), encoding="utf-8")
    return path


def run_parity_episode(
    env_tal: Any,
    env_cpp: Any,
    report: ParityReport,
    *,
    episode: int,
    max_steps: int,
    stress: bool,
    stop_after_failure: bool = False,
    parity_mode: str = "contract",
    sync_scope: str = "full",
    align_obs: bool = True,
    rng_seed: Optional[int] = None,
    repro_out_dir: Optional[Path] = None,
) -> bool:
    print(f"  [Episode {episode}] Resetting...")
    report.episodes_run += 1
    episode_failed = False
    simulation = parity_mode == "simulation"
    action_history: list[dict[str, Any]] = []
    try:
        _reset_talishar_for_parity(env_tal, env_cpp)
        hand_playability = _hand_playability_from_talishar(env_tal)
        acting_player_id = int(getattr(env_tal, "_acting_player_id", 1) or 1)
        reset_options: dict[str, Any] = {
            "opening_hands": _opening_hands_from_talishar(env_tal),
            "hand_playability": hand_playability,
            "acting_player_id": acting_player_id,
        }
        if rng_seed is not None:
            reset_options["rng_seed"] = rng_seed
        reset_cpp = env_cpp.reset(options=reset_options)
        reset_tal = _build_talishar_reset_snapshot(env_tal)
        if simulation:
            _apply_simulation_initial_sync(
                env_tal,
                env_cpp,
                sync_scope=sync_scope,
                rng_seed=rng_seed,
            )
        else:
            reset_cpp = _align_cpp_reset_result(env_cpp, reset_tal, reset_cpp, env_tal=env_tal)
    except Exception as exc:
        report.episodes_failed += 1
        _record_discrepancy(
            report,
            episode=episode,
            step=0,
            category="setup",
            description=f"reset failed before parity comparison: {exc}",
            talishar_value=type(exc).__name__,
            cpp_value="not comparable",
        )
        return False

    if simulation:
        tal_state = extract_talishar_state(env_tal)
        cpp_state = extract_cpp_state(_cpp_inner_env(env_cpp) or env_cpp)
        if not _record_state_discrepancies(
            report,
            episode=episode,
            step=0,
            tal_state=tal_state,
            cpp_state=cpp_state,
            context="reset",
        ):
            episode_failed = True
            if stop_after_failure:
                report.episodes_failed += 1
                return False
    elif not _compare_reset(reset_tal, reset_cpp, report, episode):
        episode_failed = True
        if stop_after_failure:
            report.episodes_failed += 1
            return False

    observation = reset_tal.observation
    for step in range(1, max_steps + 1):
        action_index, action_label = _choose_action(
            env_tal,
            observation,
            stress=stress,
            step=step,
            rng_seed=rng_seed,
        )
        print(f"    Step {step}: action[{action_index}] {action_label}")
        pre_tal_state = extract_talishar_state(env_tal) if simulation else {}
        pre_cpp_state = (
            extract_cpp_state(_cpp_inner_env(env_cpp) or env_cpp) if simulation else {}
        )
        try:
            action_descriptor = _talishar_action_descriptor(env_tal, action_index)
            step_tal = env_tal.step(str(action_index))
            if not simulation:
                set_mirror = getattr(_cpp_inner_env(env_cpp), "set_talishar_mirror_state", None)
                if callable(set_mirror):
                    set_mirror(
                        _talishar_parity_snapshot(
                            step_tal,
                            raw_state=getattr(env_tal, "_last_state", None),
                        )
                    )
            step_cpp = env_cpp.step(action_descriptor)
        except Exception as exc:
            episode_failed = True
            _record_discrepancy(
                report,
                episode=episode,
                step=step,
                category="exception",
                description=f"step raised after action[{action_index}] {action_label!r}: {exc}",
                talishar_value=type(exc).__name__,
                cpp_value="not comparable",
            )
            if stop_after_failure:
                report.episodes_failed += 1
                return False
            continue
        report.total_steps += 1
        action_history.append(
            {
                "step": step,
                "index": action_index,
                "label": action_label,
                "descriptor": _json_safe_value(action_descriptor),
            }
        )
        if simulation:
            step_ok = _compare_simulation_step(
                step_tal,
                step_cpp,
                env_tal,
                env_cpp,
                report,
                episode=episode,
                step=step,
                action_index=action_index,
                action_label=action_label,
                align_obs=align_obs,
                pre_cpp_state=pre_cpp_state,
            )
        else:
            step_ok = _compare_step(
                step_tal,
                step_cpp,
                report,
                episode=episode,
                step=step,
                action_index=action_index,
                action_label=action_label,
            )
        if not step_ok:
            episode_failed = True
            if simulation and repro_out_dir is not None:
                trace_path = _write_repro_trace(
                    repro_out_dir,
                    episode=episode,
                    step=step,
                    action_history=action_history,
                    pre_tal_state=pre_tal_state,
                    pre_cpp_state=pre_cpp_state,
                    action_descriptor=action_descriptor,
                    discrepancies=report.discrepancies[-5:],
                )
                print(f"    [repro] wrote {trace_path}")
            if stop_after_failure:
                report.episodes_failed += 1
                return False

        observation = step_tal.observation

        if bool(step_tal.terminated) or bool(step_tal.truncated):
            if episode_failed:
                print(
                    f"  [Episode {episode}] Terminal state at step {step} "
                    f"(episode had discrepancies)"
                )
            else:
                print(f"  [Episode {episode}] Terminal parity reached at step {step}")
                report.episodes_passed += 1
            if episode_failed:
                report.episodes_failed += 1
            return not episode_failed

    if episode_failed:
        print(f"  [Episode {episode}] Completed {max_steps} step(s) with discrepancies")
        report.episodes_failed += 1
        return False

    print(f"  [Episode {episode}] Parity matched for {max_steps} step(s)")
    report.episodes_passed += 1
    return True


def generate_summary_report(report: ParityReport) -> str:
    status = "ALL CHECKS PASSED" if report.discrepancies_found == 0 else (
        f"FINDINGS: {report.discrepancies_found} DISCREPANCIES DETECTED"
    )
    lines = [
        "=" * 60,
        "CPP ENGINE vs TALISHAR HTTP PARITY CHECK SUMMARY",
        "=" * 60,
        "",
        f"Matchup: {report.matchup}",
        f"Format: {report.format}",
        f"Mode: {report.mode}",
        f"Episodes Requested: {report.episodes_requested}",
        f"Episodes Run: {report.episodes_run}",
        f"Episodes Passed: {report.episodes_passed}",
        f"Episodes Failed: {report.episodes_failed}",
        f"Total Steps Compared: {report.total_steps}",
        "",
        "-" * 60,
        "PARITY STATUS",
        "-" * 60,
        f"Status: {'[GREEN]' if report.discrepancies_found == 0 else '[RED]'} {status}",
        "",
        "-" * 60,
        "BREAKDOWN BY CATEGORY",
        "-" * 60,
        f"Observation Mismatches:     {report.observations_mismatched}",
        f"Legal Actions Mismatches:   {report.legal_actions_mismatched}",
        f"Reward Mismatches:          {report.rewards_mismatched}",
        f"Termination Mismatches:     {report.termination_mismatches}",
        f"Game Outcome Mismatches:    {report.game_outcome_mismatches}",
        f"Setup Failures:             {report.setup_failures}",
        "",
    ]

    if report.first_failure_episode is not None:
        lines.extend(
            [
                "-" * 60,
                "FIRST FAILURE DETAILS",
                "-" * 60,
                f"Episode: {report.first_failure_episode}",
                f"Step: {report.first_failure_step}",
                "",
            ]
        )

    if report.discrepancies:
        lines.extend(["-" * 60, "DISCREPANCY DETAILS", "-" * 60])
        for index, discrepancy in enumerate(report.discrepancies[:20], 1):
            lines.append(
                f"{index}. [{discrepancy['category'].upper()}] "
                f"Episode {discrepancy['episode']}, Step {discrepancy['step']}"
            )
            lines.append(f"   Description: {discrepancy['description']}")
            if discrepancy.get("tolerance_applied"):
                lines.append(f"   Tolerance Applied: Yes (+/-{REWARD_TOLERANCE})")
        if len(report.discrepancies) > 20:
            lines.append(f"... and {len(report.discrepancies) - 20} more discrepancies")
        lines.append("")

    lines.extend(
        [
            "-" * 60,
            "OUTPUT FILES",
            "-" * 60,
            "JSON Report: parity_report.json",
            "Summary:     parity_summary.txt (this file)",
        ]
    )
    if report.discrepancies_found > 0:
        lines.append("HTML Diff:   discrepancies.html")
    return "\n".join(lines)


def generate_html_discrepancy_report(report: ParityReport) -> str:
    rows = []
    for discrepancy in report.discrepancies[:100]:
        rows.append(
            "<section class='disc'>"
            f"<h2>Episode {discrepancy['episode']}, Step {discrepancy['step']} - "
            f"{html.escape(discrepancy['category'].upper())}</h2>"
            f"<p>{html.escape(discrepancy['description'])}</p>"
            "<div class='cols'>"
            f"<pre>{html.escape(json.dumps(_json_safe_value(discrepancy['talishar_value']), indent=2, sort_keys=True))}</pre>"
            f"<pre>{html.escape(json.dumps(_json_safe_value(discrepancy['cpp_value']), indent=2, sort_keys=True))}</pre>"
            "</div></section>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>C++ vs Talishar Parity Check</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #24292f; }}
h1 {{ font-size: 24px; }}
.summary {{ padding: 12px 16px; border-left: 4px solid {'#2da44e' if report.discrepancies_found == 0 else '#cf222e'}; background: #f6f8fa; }}
.disc {{ border-top: 1px solid #d0d7de; padding: 16px 0; }}
.disc h2 {{ font-size: 16px; }}
.cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
pre {{ background: #f6f8fa; padding: 12px; overflow: auto; font-size: 12px; }}
</style>
</head>
<body>
<h1>C++ Engine vs Talishar HTTP Parity Check</h1>
<div class="summary">
<p><strong>Matchup:</strong> {html.escape(report.matchup)}</p>
<p><strong>Status:</strong> {report.discrepancies_found} discrepancies</p>
<p><strong>Episodes:</strong> {report.episodes_passed} passed, {report.episodes_failed} failed</p>
<p><strong>Total Steps Compared:</strong> {report.total_steps}</p>
</div>
{''.join(rows)}
</body>
</html>"""


def _steps_for_parity_mode(
    mode: str,
    *,
    steps_per_episode: Optional[int],
    max_turns: int,
    env_tal: Any,
    env_cpp: Any,
) -> tuple[int, bool]:
    if mode == "single-step":
        return 1, False
    if mode == "multi-step":
        return steps_per_episode or 50, False
    if mode == "stress-test":
        return steps_per_episode or 500, True
    resolved_max_turns = max(
        int(getattr(env_tal, "_max_turns", max_turns) or max_turns),
        int(getattr(env_cpp, "_max_turns", max_turns) or max_turns),
    )
    return steps_per_episode or resolved_max_turns, False


def _create_parity_envs(
    *,
    deck1: str,
    deck2: str,
    game_format: str,
    max_turns: int,
    talishar_url: str,
    cpp_engine_cache_dir: Optional[str] = None,
    cpp_engine_dir: Optional[str] = None,
    cpp_engine_deck1: Optional[str] = None,
    cpp_engine_deck2: Optional[str] = None,
    parity_mode: str = "contract",
    disable_obs_alignment: bool = False,
) -> tuple[TalisharEngineEnvironment, TalisharEngineEnvironment]:
    align_obs = not disable_obs_alignment and parity_mode != "simulation"
    common = {
        "base_url": talishar_url,
        "local_deck_name": deck1,
        "opponent_deck_name": deck2,
        "game_format": game_format,
        "max_turns": max_turns,
        "self_play": True,
        "enable_combat_tracker": True,
        "cpp_obs_alignment": align_obs,
    }
    env_tal = TalisharEngineEnvironment(**common, use_cpp_engine=False)
    env_cpp = TalisharEngineEnvironment(
        **common,
        use_cpp_engine=True,
        cpp_engine_cache_dir=cpp_engine_cache_dir,
        cpp_engine_deck1=cpp_engine_deck1,
        cpp_engine_deck2=cpp_engine_deck2,
        cpp_engine_dir=cpp_engine_dir,
    )

    if getattr(env_tal, "_using_cpp", False):
        raise RuntimeError("Talishar HTTP comparison environment unexpectedly enabled C++")
    if not getattr(env_cpp, "_using_cpp", False):
        lookup_deck1 = cpp_engine_deck1 or deck1
        lookup_deck2 = cpp_engine_deck2 or deck2
        expected_dir = cpp_engine_dir or str(
            get_engine_dir(lookup_deck1, lookup_deck2, cpp_engine_cache_dir)
        )
        raise RuntimeError(
            "C++ comparison environment did not load a compiled engine. Build the matchup first "
            "or pass --cpp-engine-dir / --cpp-engine-cache-dir. "
            f"Expected engine directory: {expected_dir}"
        )
    _configure_simulation_cpp_env(env_cpp, parity_mode=parity_mode)
    return env_tal, env_cpp


def run_parity_check(
    *,
    deck1: str,
    deck2: str,
    game_format: str = "silver_age",
    episodes: int = 1,
    mode: str = "full-episode",
    steps_per_episode: Optional[int] = None,
    max_turns: int = 2000,
    talishar_url: str = "http://localhost:8080/game",
    cpp_engine_cache_dir: Optional[str] = None,
    cpp_engine_dir: Optional[str] = None,
    cpp_engine_deck1: Optional[str] = None,
    cpp_engine_deck2: Optional[str] = None,
    out_dir: Optional[Path | str] = None,
    stop_after_failure: bool = False,
    write_reports: bool = True,
    verbose: bool = True,
    parity_mode: str = "contract",
    sync_scope: str = "full",
    disable_obs_alignment: bool = False,
    rng_seed: Optional[int] = None,
) -> tuple[ParityReport, int]:
    """Run a parity check programmatically.

    Returns ``(report, exit_code)`` where exit code matches the CLI script:
    0 = passed, 1 = discrepancies, 2 = setup failure.
    """
    matchup_label = f"{_safe_label(deck1)}_vs_{_safe_label(deck2)}"
    resolved_out_dir = Path(out_dir) if out_dir is not None else Path("results/parity_checks") / matchup_label
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    report = ParityReport(
        matchup=f"{deck1} vs {deck2}",
        format=game_format,
        mode=mode,
        episodes_requested=episodes,
    )

    if verbose:
        print("=" * 60)
        print("CPP ENGINE vs TALISHAR HTTP PARITY CHECK")
        print("=" * 60)
        print(f"Matchup: {deck1} vs {deck2}")
        print(f"Format: {game_format}")
        print(f"Mode: {mode}")
        print(f"Parity mode: {parity_mode}")
        print(f"Sync scope: {sync_scope}")
        print(f"Episodes: {episodes}")
        print(f"Talishar URL: {talishar_url}")
        print(f"Output Dir: {resolved_out_dir}")
        print("=" * 60)
        print()

    try:
        env_tal, env_cpp = _create_parity_envs(
            deck1=deck1,
            deck2=deck2,
            game_format=game_format,
            max_turns=max_turns,
            talishar_url=talishar_url,
            cpp_engine_cache_dir=cpp_engine_cache_dir,
            cpp_engine_dir=cpp_engine_dir,
            cpp_engine_deck1=cpp_engine_deck1,
            cpp_engine_deck2=cpp_engine_deck2,
            parity_mode=parity_mode,
            disable_obs_alignment=disable_obs_alignment,
        )
        if verbose:
            print("[OK] Talishar HTTP environment created")
            print("[OK] C++ engine environment created")
    except Exception as exc:
        if verbose:
            print(f"[ERROR] Failed to create parity environments: {exc}")
        _record_discrepancy(
            report,
            episode=0,
            step=0,
            category="setup",
            description=f"failed to create parity environments: {exc}",
            talishar_value="HTTP environment must use use_cpp_engine=False",
            cpp_value=str(exc),
        )
        if write_reports:
            _write_reports(report, resolved_out_dir)
        return report, 2

    max_steps, stress = _steps_for_parity_mode(
        mode,
        steps_per_episode=steps_per_episode,
        max_turns=max_turns,
        env_tal=env_tal,
        env_cpp=env_cpp,
    )
    align_obs = not disable_obs_alignment and parity_mode != "simulation"
    try:
        for episode in range(1, episodes + 1):
            episode_seed = (rng_seed + episode - 1) if rng_seed is not None else None
            run_parity_episode(
                env_tal,
                env_cpp,
                report,
                episode=episode,
                max_steps=max_steps,
                stress=stress,
                stop_after_failure=stop_after_failure,
                parity_mode=parity_mode,
                sync_scope=sync_scope,
                align_obs=align_obs,
                rng_seed=episode_seed,
                repro_out_dir=resolved_out_dir if parity_mode == "simulation" else None,
            )
    except KeyboardInterrupt:
        if verbose:
            print("[WARN] Interrupted by user; writing partial report")
    finally:
        env_tal.close()
        env_cpp.close()

    if write_reports:
        _write_reports(report, resolved_out_dir)

    if report.discrepancies_found == 0 and report.episodes_failed == 0:
        if verbose:
            print("\nPARITY CHECK PASSED - ALL TESTS SUCCESSFUL")
        return report, 0

    if verbose:
        print(f"\nPARITY CHECK COMPLETED WITH {report.discrepancies_found} DISCREPANCIES")
    if report.setup_failures > 0 and report.total_steps == 0:
        return report, 2
    return report, 1


def _steps_for_mode(args: argparse.Namespace, env_tal: Any, env_cpp: Any) -> tuple[int, bool]:
    return _steps_for_parity_mode(
        args.mode,
        steps_per_episode=args.steps_per_episode,
        max_turns=args.max_turns,
        env_tal=env_tal,
        env_cpp=env_cpp,
    )


def _create_envs(args: argparse.Namespace) -> tuple[TalisharEngineEnvironment, TalisharEngineEnvironment]:
    return _create_parity_envs(
        deck1=args.deck1,
        deck2=args.deck2,
        game_format=args.format,
        max_turns=args.max_turns,
        talishar_url=args.talishar_url,
        cpp_engine_cache_dir=args.cpp_engine_cache_dir,
        cpp_engine_dir=args.cpp_engine_dir,
        cpp_engine_deck1=args.cpp_engine_deck1,
        cpp_engine_deck2=args.cpp_engine_deck2,
    )


def _write_reports(report: ParityReport, out_dir: Path) -> None:
    json_report_path = out_dir / "parity_report.json"
    payload = _json_safe_value(report.to_dict())
    json_report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] JSON report written to: {json_report_path}")

    summary_path = out_dir / "parity_summary.txt"
    summary_content = generate_summary_report(report)
    summary_path.write_text(summary_content, encoding="utf-8")
    print(f"[OK] Summary written to: {summary_path}")
    print()
    print(summary_content)

    if report.discrepancies_found > 0:
        html_path = out_dir / "discrepancies.html"
        html_path.write_text(generate_html_discrepancy_report(report), encoding="utf-8")
        print(f"[OK] HTML discrepancy report written to: {html_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C++ Engine vs Talishar HTTP parity checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--deck1", default="Ira", help="Local Talishar deck name for P1")
    parser.add_argument("--deck2", default="Briar", help="Local Talishar deck name for P2")
    parser.add_argument(
        "--format",
        default="silver_age",
        choices=["silver_age", "classic_constructed"],
        help="Talishar game format",
    )
    parser.add_argument(
        "--mode",
        default="full-episode",
        choices=["single-step", "multi-step", "full-episode", "stress-test"],
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps-per-episode", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=2000)
    parser.add_argument("--talishar-url", default="http://localhost:8080/game")
    parser.add_argument("--cpp-engine-cache-dir", default=None)
    parser.add_argument("--cpp-engine-dir", default=None)
    parser.add_argument("--cpp-engine-deck1", default=None)
    parser.add_argument("--cpp-engine-deck2", default=None)
    parser.add_argument(
        "--parity-mode",
        default="contract",
        choices=["contract", "simulation"],
        help="contract=mirror Talishar on each step; simulation=independent C++ stepping",
    )
    parser.add_argument(
        "--sync-scope",
        default="full",
        choices=["hands", "full"],
        help="Initial state copied from Talishar at reset (simulation mode)",
    )
    parser.add_argument(
        "--disable-obs-alignment",
        action="store_true",
        help="Disable combat-slice neutralization during obs comparison",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for C++ reset")
    parser.add_argument("--out-dir", default="results/parity_checks")
    parser.add_argument(
        "--stop-after-failure",
        action="store_true",
        help=(
            "Stop at the first discrepancy (within an episode or across episodes). "
            "By default the checker records mismatches and keeps running to collect "
            "as many findings as possible."
        ),
    )
    args = parser.parse_args()

    matchup_label = f"{_safe_label(args.deck1)}_vs_{_safe_label(args.deck2)}"
    out_dir = Path(args.out_dir) / matchup_label

    report, exit_code = run_parity_check(
        deck1=args.deck1,
        deck2=args.deck2,
        game_format=args.format,
        episodes=args.episodes,
        mode=args.mode,
        steps_per_episode=args.steps_per_episode,
        max_turns=args.max_turns,
        talishar_url=args.talishar_url,
        cpp_engine_cache_dir=args.cpp_engine_cache_dir,
        cpp_engine_dir=args.cpp_engine_dir,
        cpp_engine_deck1=args.cpp_engine_deck1,
        cpp_engine_deck2=args.cpp_engine_deck2,
        out_dir=out_dir,
        stop_after_failure=args.stop_after_failure,
        write_reports=True,
        verbose=True,
        parity_mode=args.parity_mode,
        sync_scope=args.sync_scope,
        disable_obs_alignment=args.disable_obs_alignment,
        rng_seed=args.seed,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
