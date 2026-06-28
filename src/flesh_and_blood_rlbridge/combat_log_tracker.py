"""Combat/turn tracker utilities shared by Talishar and C++ environments."""

from __future__ import annotations

import copy
import hashlib
import html
import json
import re
from collections import Counter
from typing import Any

_PASS_ACTION_CODES = {99, 101, 105}
_LINE_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def _is_pass_action(action: dict[str, Any]) -> bool:
    code = _to_int(action.get("action_code", 0))
    if code in _PASS_ACTION_CODES:
        return True
    label = _normalize_text(action.get("label", "")).lower()
    return any(tok in label for tok in ("pass", "end turn", "no block", "skip"))


def _normalize_action(action: dict[str, Any] | None) -> dict[str, Any]:
    src = action or {}
    return {
        "action_code": _to_int(src.get("action_code", 0)),
        "button_input": str(src.get("button_input", "") or ""),
        "card_id": str(src.get("card_id", "") or ""),
        "zone": _normalize_text(src.get("zone", "")).lower(),
        "label": _normalize_text(src.get("label", "")),
    }


def _action_key(action: dict[str, Any]) -> str:
    return "|".join(
        [
            str(_to_int(action.get("action_code", 0))),
            str(action.get("zone", "") or ""),
            str(action.get("button_input", "") or ""),
            str(action.get("card_id", "") or ""),
            _normalize_text(action.get("label", "")).lower(),
        ]
    )


def _classify_action(phase: str, action: dict[str, Any]) -> str:
    if _is_pass_action(action):
        return "pass"

    phase_l = _normalize_text(phase).lower()
    zone = _normalize_text(action.get("zone", "")).lower()
    label = _normalize_text(action.get("label", "")).lower()
    code = _to_int(action.get("action_code", 0))

    if phase_l in {"b", "d", "block", "defense", "defend"}:
        return "defend"
    if "block" in label or "defend" in label:
        return "defend"

    if phase_l in {"m", "a", "attack", "ars", "arsenal"}:
        if zone in {"hand", "arsenal", "equipment", "ally", "item", "permanent", "weapon"}:
            return "attack"
    if "attack" in label or "swing" in label or "strike" in label:
        return "attack"
    if zone == "hand" and code == 27 and phase_l != "p":
        return "attack"

    return "other"


def _normalize_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    src = snapshot or {}
    return {
        "acting_player_id": _to_int(src.get("acting_player_id", 1), 1),
        "turn_no": _to_int(src.get("turn_no", 0), 0),
        "phase": _normalize_text(src.get("phase", "")),
        "player_health": _to_int(src.get("player_health", 0), 0),
        "opponent_health": _to_int(src.get("opponent_health", 0), 0),
        "player_hand_size": _to_int(src.get("player_hand_size", 0), 0),
        "opponent_hand_size": _to_int(src.get("opponent_hand_size", 0), 0),
        "player_deck_count": _to_int(src.get("player_deck_count", 0), 0),
        "opponent_deck_count": _to_int(src.get("opponent_deck_count", 0), 0),
        "player_pitch_count": _to_int(src.get("player_pitch_count", 0), 0),
        "legal_count": _to_int(src.get("legal_count", 0), 0),
    }


def _board_state_key(snapshot: dict[str, Any]) -> str:
    key_data = {
        "acting_player_id": snapshot.get("acting_player_id", 1),
        "phase": snapshot.get("phase", ""),
        "player_health": snapshot.get("player_health", 0),
        "opponent_health": snapshot.get("opponent_health", 0),
        "player_hand_size": snapshot.get("player_hand_size", 0),
        "opponent_hand_size": snapshot.get("opponent_hand_size", 0),
        "player_deck_count": snapshot.get("player_deck_count", 0),
        "opponent_deck_count": snapshot.get("opponent_deck_count", 0),
        "player_pitch_count": snapshot.get("player_pitch_count", 0),
        "legal_count": snapshot.get("legal_count", 0),
    }
    return json.dumps(key_data, sort_keys=True, separators=(",", ":"))


_GAMESTATE_REVERT_NEEDLE = "reverting gamestate prior to"


def talishar_gamestate_revert_detected(state: dict[str, Any]) -> bool:
    """Return True when Talishar reverted the gamestate after an invalid action."""
    chat_log = state.get("chatLog", "")
    for line in extract_talishar_chat_log_lines(chat_log):
        if _GAMESTATE_REVERT_NEEDLE in line.lower():
            return True
    return False


def extract_talishar_chat_log_lines(chat_log: Any) -> list[str]:
    """Convert Talishar chatLog HTML into normalized plain-text lines."""
    if chat_log is None:
        return []
    raw = str(chat_log)
    if not raw.strip():
        return []

    parts = _LINE_BREAK_RE.split(raw)
    out: list[str] = []
    for part in parts:
        line = _TAG_RE.sub("", part)
        line = _normalize_text(line)
        if line:
            out.append(line)
    return out


def compare_trace_hashes(
    reference_events: list[dict[str, Any]],
    candidate_events: list[dict[str, Any]],
    *,
    max_diffs: int = 5,
) -> dict[str, Any]:
    """Compare two traces using per-step trace hashes."""
    ref_hashes = [str(e.get("trace_hash", "")) for e in reference_events]
    cand_hashes = [str(e.get("trace_hash", "")) for e in candidate_events]

    mismatches: list[dict[str, Any]] = []
    limit = min(len(ref_hashes), len(cand_hashes))
    for i in range(limit):
        if ref_hashes[i] != cand_hashes[i]:
            mismatches.append(
                {
                    "step": i + 1,
                    "reference": ref_hashes[i],
                    "candidate": cand_hashes[i],
                }
            )
            if len(mismatches) >= max_diffs:
                break

    return {
        "matches": len(mismatches) == 0 and len(ref_hashes) == len(cand_hashes),
        "reference_steps": len(ref_hashes),
        "candidate_steps": len(cand_hashes),
        "first_differences": mismatches,
    }


class CombatTurnTracker:
    """Collects combat logs and per-turn action statistics."""

    def __init__(self, *, engine_name: str, enabled: bool = True) -> None:
        self._engine_name = engine_name
        self._enabled = bool(enabled)
        self.clear()

    def clear(self) -> None:
        self._step_index = 0
        self._events: list[dict[str, Any]] = []
        self._metadata: dict[str, Any] = {}
        self._initial_snapshot: dict[str, Any] = {}
        self._initial_legal_actions: list[dict[str, Any]] = []
        self._combat_log_lines: list[str] = []

        self._board_stats: dict[str, dict[str, Any]] = {}
        self._pending_defend_by_player: dict[int, dict[str, Any]] = {}
        self._defend_to_attack_counts: Counter[str] = Counter()
        self._defend_to_attack_meta: dict[str, dict[str, Any]] = {}

        self._trace_digest_cache: str | None = None

    def reset(
        self,
        *,
        initial_snapshot: dict[str, Any],
        initial_legal_actions: list[dict[str, Any]],
        combat_log_lines: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.clear()
        if not self._enabled:
            return

        self._metadata = dict(metadata or {})
        self._initial_snapshot = _normalize_snapshot(initial_snapshot)
        self._initial_legal_actions = [_normalize_action(a) for a in initial_legal_actions]
        self._combat_log_lines = [_normalize_text(x) for x in (combat_log_lines or []) if _normalize_text(x)]

    @property
    def trace_digest(self) -> str:
        if self._trace_digest_cache is None:
            joined = "|".join(str(e.get("trace_hash", "")) for e in self._events)
            self._trace_digest_cache = hashlib.sha256(joined.encode("utf-8")).hexdigest()
        return self._trace_digest_cache

    @property
    def steps_recorded(self) -> int:
        return len(self._events)

    def trace(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._events)

    def record_step(
        self,
        *,
        before_snapshot: dict[str, Any],
        after_snapshot: dict[str, Any],
        action: dict[str, Any],
        legal_before: list[dict[str, Any]],
        legal_after: list[dict[str, Any]],
        reward: float,
        terminated: bool,
        truncated: bool,
        combat_log_lines: list[str] | None = None,
    ) -> dict[str, Any]:
        if not self._enabled:
            return {
                "enabled": False,
                "step": self._step_index,
            }

        before = _normalize_snapshot(before_snapshot)
        after = _normalize_snapshot(after_snapshot)
        chosen_action = _normalize_action(action)
        legal_before_n = [_normalize_action(a) for a in legal_before]
        legal_after_n = [_normalize_action(a) for a in legal_after]

        action_class = _classify_action(before.get("phase", ""), chosen_action)
        board_key = _board_state_key(before)
        action_key = _action_key(chosen_action)

        self._step_index += 1

        board_stats = self._board_stats.get(board_key)
        if board_stats is None:
            board_stats = {
                "board": before,
                "visits": 0,
                "attack_actions": Counter(),
                "defend_actions": Counter(),
                "pass_actions": Counter(),
                "other_actions": Counter(),
            }
            self._board_stats[board_key] = board_stats

        board_stats["visits"] += 1
        counter_key = f"{action_class}_actions"
        if counter_key in board_stats:
            board_stats[counter_key][action_key] += 1

        acting_player = _to_int(before.get("acting_player_id", 1), 1)
        if action_class == "defend":
            self._pending_defend_by_player[acting_player] = {
                "action_key": action_key,
                "action": chosen_action,
                "board": before,
                "turn_no": _to_int(before.get("turn_no", 0), 0),
            }
        elif action_class == "attack":
            pending = self._pending_defend_by_player.pop(acting_player, None)
            if pending is not None:
                transition_key = f"{pending['action_key']} => {action_key}"
                self._defend_to_attack_counts[transition_key] += 1
                if transition_key not in self._defend_to_attack_meta:
                    self._defend_to_attack_meta[transition_key] = {
                        "defend_action": pending["action"],
                        "attack_action": chosen_action,
                        "defend_board": pending["board"],
                    }

        log_delta = self._compute_log_delta(combat_log_lines)

        hash_payload = {
            "step": self._step_index,
            "before": {
                "acting_player_id": before.get("acting_player_id", 1),
                "turn_no": before.get("turn_no", 0),
                "phase": before.get("phase", ""),
                "player_health": before.get("player_health", 0),
                "opponent_health": before.get("opponent_health", 0),
            },
            "after": {
                "acting_player_id": after.get("acting_player_id", 1),
                "turn_no": after.get("turn_no", 0),
                "phase": after.get("phase", ""),
                "player_health": after.get("player_health", 0),
                "opponent_health": after.get("opponent_health", 0),
            },
            "action": chosen_action,
            "reward": round(float(reward), 6),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        trace_hash = hashlib.sha1(
            json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]

        event = {
            "step": self._step_index,
            "trace_hash": trace_hash,
            "before": before,
            "after": after,
            "action": chosen_action,
            "action_class": action_class,
            "board_state_key": board_key,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "legal_before_count": len(legal_before_n),
            "legal_after_count": len(legal_after_n),
            "combat_log_delta": log_delta,
        }

        self._events.append(event)
        self._trace_digest_cache = None
        return copy.deepcopy(event)

    def _compute_log_delta(self, combat_log_lines: list[str] | None) -> list[str]:
        current = [_normalize_text(x) for x in (combat_log_lines or []) if _normalize_text(x)]
        if not current:
            return []

        previous = self._combat_log_lines
        if not previous:
            self._combat_log_lines = current
            return current[-40:]

        prefix = 0
        max_prefix = min(len(previous), len(current))
        while prefix < max_prefix and previous[prefix] == current[prefix]:
            prefix += 1

        if prefix == len(previous):
            delta = current[prefix:]
        else:
            overlap = 0
            max_overlap = min(len(previous), len(current), 80)
            for k in range(max_overlap, 0, -1):
                if previous[-k:] == current[:k]:
                    overlap = k
                    break
            delta = current[overlap:] if overlap > 0 else current

        self._combat_log_lines = current[-4000:]
        return delta[-40:]

    def snapshot(
        self,
        *,
        top_k: int = 10,
        tail_events: int = 20,
        tail_log_lines: int = 40,
    ) -> dict[str, Any]:
        if not self._enabled:
            return {
                "engine": self._engine_name,
                "enabled": False,
                "steps_recorded": 0,
                "trace_digest": "",
            }

        board_rows: list[dict[str, Any]] = []
        for _, stats in sorted(
            self._board_stats.items(),
            key=lambda item: int(item[1].get("visits", 0)),
            reverse=True,
        )[:top_k]:
            board_rows.append(
                {
                    "board": copy.deepcopy(stats.get("board", {})),
                    "visits": int(stats.get("visits", 0)),
                    "attack_actions": [
                        {"action_key": k, "count": int(v)}
                        for k, v in stats.get("attack_actions", Counter()).most_common(top_k)
                    ],
                    "defend_actions": [
                        {"action_key": k, "count": int(v)}
                        for k, v in stats.get("defend_actions", Counter()).most_common(top_k)
                    ],
                    "pass_actions": [
                        {"action_key": k, "count": int(v)}
                        for k, v in stats.get("pass_actions", Counter()).most_common(top_k)
                    ],
                }
            )

        transitions: list[dict[str, Any]] = []
        for key, count in self._defend_to_attack_counts.most_common(top_k):
            meta = self._defend_to_attack_meta.get(key, {})
            transitions.append(
                {
                    "count": int(count),
                    "defend_action": copy.deepcopy(meta.get("defend_action", {})),
                    "attack_action": copy.deepcopy(meta.get("attack_action", {})),
                    "defend_board": copy.deepcopy(meta.get("defend_board", {})),
                }
            )

        return {
            "engine": self._engine_name,
            "enabled": True,
            "metadata": copy.deepcopy(self._metadata),
            "steps_recorded": len(self._events),
            "trace_digest": self.trace_digest,
            "initial_snapshot": copy.deepcopy(self._initial_snapshot),
            "initial_legal_count": len(self._initial_legal_actions),
            "recent_events": copy.deepcopy(self._events[-tail_events:]) if tail_events > 0 else [],
            "recent_combat_log_lines": copy.deepcopy(self._combat_log_lines[-tail_log_lines:]),
            "board_state_stats": board_rows,
            "defend_to_attack_transitions": transitions,
        }
