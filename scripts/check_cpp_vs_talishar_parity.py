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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.talishar_engine_environment import TalisharEngineEnvironment
from flesh_and_blood_rlbridge.cpp_engine_environment import get_engine_dir


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "step": self.step,
            "category": self.category,
            "description": self.description,
            "talishar_value": self.talishar_value,
            "cpp_value": self.cpp_value,
            "tolerance_applied": self.tolerance_applied,
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


def _summary_value(value: Any, max_items: int = 12) -> Any:
    if isinstance(value, str):
        parsed, _ = _parse_observation("value", value)
        return parsed if parsed is not None else value[:1000]
    if isinstance(value, list):
        out = [_summary_value(item, max_items=max_items) for item in value[:max_items]]
        if len(value) > max_items:
            out.append(f"... {len(value) - max_items} more")
        return out
    if isinstance(value, dict):
        return {key: _summary_value(val, max_items=max_items) for key, val in value.items()}
    return value


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
        if actual_keys != expected_keys:
            return (
                False,
                f"{label} keys mismatch: missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}",
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
    return True, ""


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
            if card_id:
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
            if indices:
                playability[player_id] = indices
    return playability


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

    last_state = getattr(env_tal, "_last_state", None)
    acting_player_id = _safe_int(getattr(env_tal, "_acting_player_id", 1)) or 1
    if acting_player_id not in opening_hands and isinstance(last_state, dict):
        opening_hands[acting_player_id] = _card_ids_from_state_hand(last_state)
    return opening_hands


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


def _legal_actions_from_observation(observation: Any) -> list[dict[str, Any]]:
    parsed, _ = _parse_observation("Talishar", observation)
    if parsed is None:
        return []
    legal = parsed.get("legalActions", [])
    return legal if isinstance(legal, list) else []


def _choose_action(env_tal: Any, observation: Any, *, stress: bool) -> tuple[int, str]:
    legal = _legal_actions_from_observation(observation)
    if not legal:
        return 0, "<no legal actions>"
    if stress:
        index = random.randrange(len(legal))
    else:
        try:
            index = int(str(env_tal.sample_action()).strip())
        except (TypeError, ValueError):
            index = 0
        index = max(0, min(index, len(legal) - 1))
    return index, str(legal[index].get("label", "") or "")


def run_parity_episode(
    env_tal: Any,
    env_cpp: Any,
    report: ParityReport,
    *,
    episode: int,
    max_steps: int,
    stress: bool,
    stop_after_failure: bool = False,
) -> bool:
    print(f"  [Episode {episode}] Resetting...")
    report.episodes_run += 1
    episode_failed = False
    try:
        _reset_talishar_for_parity(env_tal, env_cpp)
        opening_hands = _opening_hands_from_talishar(env_tal)
        hand_playability = _hand_playability_from_talishar(env_tal)
        acting_player_id = int(getattr(env_tal, "_acting_player_id", 1) or 1)
        reset_cpp = env_cpp.reset(
            options={
                "opening_hands": opening_hands,
                "hand_playability": hand_playability,
                "acting_player_id": acting_player_id,
            }
        )
        reset_tal = _build_talishar_reset_snapshot(env_tal)
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
    if not _compare_reset(reset_tal, reset_cpp, report, episode):
        episode_failed = True
        if stop_after_failure:
            report.episodes_failed += 1
            return False

    observation = reset_tal.observation
    for step in range(1, max_steps + 1):
        action_index, action_label = _choose_action(env_tal, observation, stress=stress)
        print(f"    Step {step}: action[{action_index}] {action_label}")
        try:
            step_tal = env_tal.step(str(action_index))
            step_cpp = env_cpp.step(str(action_index))
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
        if not _compare_step(
            step_tal,
            step_cpp,
            report,
            episode=episode,
            step=step,
            action_index=action_index,
            action_label=action_label,
        ):
            episode_failed = True
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
            f"<pre>{html.escape(json.dumps(discrepancy['talishar_value'], indent=2, sort_keys=True))}</pre>"
            f"<pre>{html.escape(json.dumps(discrepancy['cpp_value'], indent=2, sort_keys=True))}</pre>"
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


def _steps_for_mode(args: argparse.Namespace, env_tal: Any, env_cpp: Any) -> tuple[int, bool]:
    if args.mode == "single-step":
        return 1, False
    if args.mode == "multi-step":
        return args.steps_per_episode or 50, False
    if args.mode == "stress-test":
        return args.steps_per_episode or 500, True
    max_turns = max(
        int(getattr(env_tal, "_max_turns", 2000) or 2000),
        int(getattr(env_cpp, "_max_turns", 2000) or 2000),
    )
    return args.steps_per_episode or max_turns, False


def _create_envs(args: argparse.Namespace) -> tuple[TalisharEngineEnvironment, TalisharEngineEnvironment]:
    common = {
        "base_url": args.talishar_url,
        "local_deck_name": args.deck1,
        "opponent_deck_name": args.deck2,
        "game_format": args.format,
        "max_turns": args.max_turns,
        "self_play": True,
        "enable_combat_tracker": True,
    }
    env_tal = TalisharEngineEnvironment(**common, use_cpp_engine=False)
    env_cpp = TalisharEngineEnvironment(
        **common,
        use_cpp_engine=True,
        cpp_engine_cache_dir=args.cpp_engine_cache_dir,
        cpp_engine_deck1=args.cpp_engine_deck1,
        cpp_engine_deck2=args.cpp_engine_deck2,
        cpp_engine_dir=args.cpp_engine_dir,
    )

    if getattr(env_tal, "_using_cpp", False):
        raise RuntimeError("Talishar HTTP comparison environment unexpectedly enabled C++")
    if not getattr(env_cpp, "_using_cpp", False):
        lookup_deck1 = args.cpp_engine_deck1 or args.deck1
        lookup_deck2 = args.cpp_engine_deck2 or args.deck2
        expected_dir = args.cpp_engine_dir or str(
            get_engine_dir(lookup_deck1, lookup_deck2, args.cpp_engine_cache_dir)
        )
        raise RuntimeError(
            "C++ comparison environment did not load a compiled engine. Build the matchup first "
            "or pass --cpp-engine-dir / --cpp-engine-cache-dir. "
            f"Expected engine directory: {expected_dir}"
        )
    return env_tal, env_cpp


def _write_reports(report: ParityReport, out_dir: Path) -> None:
    json_report_path = out_dir / "parity_report.json"
    json_report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
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
    parser.add_argument("--talishar-url", default="http://localhost")
    parser.add_argument("--cpp-engine-cache-dir", default=None)
    parser.add_argument("--cpp-engine-dir", default=None)
    parser.add_argument("--cpp-engine-deck1", default=None)
    parser.add_argument("--cpp-engine-deck2", default=None)
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
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CPP ENGINE vs TALISHAR HTTP PARITY CHECK")
    print("=" * 60)
    print(f"Matchup: {args.deck1} vs {args.deck2}")
    print(f"Format: {args.format}")
    print(f"Mode: {args.mode}")
    print(f"Episodes: {args.episodes}")
    print(f"Talishar URL: {args.talishar_url}")
    print(f"Output Dir: {out_dir}")
    print("=" * 60)
    print()

    report = ParityReport(
        matchup=f"{args.deck1} vs {args.deck2}",
        format=args.format,
        mode=args.mode,
        episodes_requested=args.episodes,
    )

    try:
        env_tal, env_cpp = _create_envs(args)
        print("[OK] Talishar HTTP environment created")
        print("[OK] C++ engine environment created")
    except Exception as exc:
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
        _write_reports(report, out_dir)
        sys.exit(2)

    max_steps, stress = _steps_for_mode(args, env_tal, env_cpp)
    try:
        for episode in range(1, args.episodes + 1):
            run_parity_episode(
                env_tal,
                env_cpp,
                report,
                episode=episode,
                max_steps=max_steps,
                stress=stress,
                stop_after_failure=args.stop_after_failure,
            )
    except KeyboardInterrupt:
        print("[WARN] Interrupted by user; writing partial report")
    finally:
        env_tal.close()
        env_cpp.close()

    _write_reports(report, out_dir)

    if report.discrepancies_found == 0 and report.episodes_failed == 0:
        print("\nPARITY CHECK PASSED - ALL TESTS SUCCESSFUL")
        sys.exit(0)

    print(f"\nPARITY CHECK COMPLETED WITH {report.discrepancies_found} DISCREPANCIES")
    if report.setup_failures > 0 and report.total_steps == 0:
        sys.exit(2)
    sys.exit(1)


if __name__ == "__main__":
    main()
