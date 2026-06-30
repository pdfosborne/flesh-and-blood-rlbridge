"""Per-turn step limits and decision-point loop detection for RL environments.

When an agent revisits the same decision point repeatedly, returns to a prior
board configuration within the same turn (undo / action-chain loops), or exceeds
a per-turn step budget, environments force Pass instead of accumulating
bug-specific filter rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .legal_action_filter import first_non_pass_action
from .talishar_default_policy import _get_phase, _is_pass_action, _to_int

DEFAULT_MAX_STEPS_PER_TURN = 100
DEFAULT_LOOP_REPEAT_THRESHOLD = 4

# Talishar JSON keys included in the per-turn board fingerprint (order matters).
_BOARD_ZONE_KEYS: tuple[str, ...] = (
    "playerHand",
    "opponentHand",
    "playerEquipment",
    "opponentEquipment",
    "playerArse",
    "opponentArse",
    "playerPitch",
    "opponentPitch",
    "playerDiscard",
    "opponentDiscard",
    "playerBanish",
    "opponentBanish",
    "playerAuras",
    "opponentAuras",
    "playerAllies",
    "opponentAllies",
    "playerItems",
    "opponentItems",
    "playerPermanents",
    "opponentPermanents",
)

_COMBAT_CHAIN_KEYS: tuple[str, ...] = (
    "combatChain",
    "activeChainLink",
    "chainLinks",
)


def legal_actions_fingerprint(legal_actions: list[dict[str, Any]]) -> str:
    """Stable fingerprint of the filtered legal-action set."""
    return "|".join(
        f"{_to_int(a.get('action_code', 0))}:"
        f"{str(a.get('button_input', '') or '')}:"
        f"{str(a.get('label', '') or '')}"
        for a in legal_actions
    )


def _card_key(card: dict[str, Any]) -> str:
    return str(
        card.get("cardNumber")
        or card.get("cardID")
        or card.get("card_id")
        or card.get("id")
        or ""
    )


def _card_fingerprint(card: Any) -> str:
    if not isinstance(card, dict):
        return ""
    parts = [_card_key(card)]
    for field in ("counters", "counter", "tapped", "facing", "action", "mod"):
        value = card.get(field)
        if value in (None, "", 0, False):
            continue
        parts.append(f"{field}={value}")
    counters_map = card.get("countersMap")
    if isinstance(counters_map, dict) and counters_map:
        items = ",".join(
            f"{key}:{_to_int(val)}"
            for key, val in sorted(counters_map.items(), key=lambda item: str(item[0]))
        )
        parts.append(f"countersMap={items}")
    return ":".join(parts)


def _zone_fingerprint(cards: Any) -> str:
    if not isinstance(cards, list):
        return ""
    return ",".join(_card_fingerprint(card) for card in cards)


def _chain_fingerprint(value: Any) -> str:
    if isinstance(value, list):
        return _zone_fingerprint(value)
    if isinstance(value, dict):
        if value.get("cardID") or value.get("cardNumber"):
            return _card_fingerprint(value)
        nested = value.get("links") or value.get("chainLinks")
        if isinstance(nested, list):
            return _zone_fingerprint(nested)
        return "|".join(f"{key}={value[key]}" for key in sorted(value))
    return str(value or "")


def board_state_fingerprint(state: dict[str, Any]) -> str:
    """Fingerprint public zones and scalars — ignores legal actions.

    Revisiting the same fingerprint within one player-turn window means actions
    reverted the game state (undo chains, equipment toggles, etc.).
    """
    phase = _get_phase(state)
    parts = [
        phase,
        str(_to_int(state.get("playerHealth", 0), 0)),
        str(_to_int(state.get("opponentHealth", 0), 0)),
        str(_to_int(state.get("playerPitchCount", 0), 0)),
        str(_to_int(state.get("opponentPitchCount", 0), 0)),
        str(_to_int(state.get("playerDeckCount", 0), 0)),
        str(_to_int(state.get("opponentDeckCount", 0), 0)),
        str(_to_int(state.get("playerHandSize", 0), 0)),
        str(_to_int(state.get("opponentHandSize", 0), 0)),
        str(_to_int(state.get("pendingAttackPower", 0), 0)),
        str(_to_int(state.get("pendingBlockValue", 0), 0)),
    ]
    for key in _BOARD_ZONE_KEYS:
        if key in state:
            parts.append(f"{key}={_zone_fingerprint(state[key])}")
    if "opponentHand" not in state and state.get("opponentHandSize") is not None:
        parts.append(f"opponentHandSize={_to_int(state.get('opponentHandSize', 0), 0)}")
    for key in _COMBAT_CHAIN_KEYS:
        if key in state:
            parts.append(f"{key}={_chain_fingerprint(state[key])}")
    return "|".join(parts)


def decision_point_fingerprint(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
) -> str:
    """Hash phase, turn, coarse resources, and legal actions for loop detection."""
    turn_no = _to_int(state.get("turnNo", state.get("turn_no", 0)), 0)
    pitch = _to_int(state.get("playerPitchCount", 0), 0)
    hand = state.get("playerHand", [])
    hand_size = len(hand) if isinstance(hand, list) else 0
    legal_fp = legal_actions_fingerprint(legal_actions)
    phase = _get_phase(state)
    return f"{turn_no}|{phase}|{pitch}|{hand_size}|{legal_fp}"


def first_pass_action(
    legal_actions: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Return the first pass action in *legal_actions*, if any."""
    for action in legal_actions:
        if _is_pass_action(action):
            return action
    return None


def _action_submission(action: dict[str, Any]) -> tuple[int, str]:
    return (
        _to_int(action.get("action_code", 0)),
        str(action.get("button_input", "") or ""),
    )


def _resolve_forced_action(
    legal_actions: list[dict[str, Any]],
    *,
    reason: str,
) -> Optional[dict[str, Any]]:
    """Pick the action environments should submit when breaking a loop."""
    if reason == "decision_loop":
        non_pass = first_non_pass_action(legal_actions)
        if non_pass is not None:
            return non_pass
    return first_pass_action(legal_actions) or (
        legal_actions[0] if legal_actions else None
    )


def resolve_forced_submission(
    legal_actions: list[dict[str, Any]],
    result: "LoopGuardResult",
) -> tuple[int, str]:
    """Map a loop-guard result to ``(mode, button_input)`` for env submission."""
    if not result.force_pass:
        return 99, ""
    forced = result.forced_action
    if forced is None:
        forced = _resolve_forced_action(legal_actions, reason=result.reason)
    if forced is None:
        return 99, ""
    return _action_submission(forced)


@dataclass(frozen=True)
class LoopGuardResult:
    force_pass: bool
    turn_steps: int
    loop_streak: int
    reason: str = ""
    forced_action: Optional[dict[str, Any]] = None


class TurnLoopGuard:
    """Track per-turn steps, repeated decision points, and board-state reverts."""

    def __init__(
        self,
        *,
        max_steps_per_turn: int = DEFAULT_MAX_STEPS_PER_TURN,
        loop_repeat_threshold: int = DEFAULT_LOOP_REPEAT_THRESHOLD,
    ) -> None:
        self._max_steps_per_turn = max(1, int(max_steps_per_turn))
        self._loop_repeat_threshold = max(2, int(loop_repeat_threshold))
        self.reset()

    def reset(self) -> None:
        self._turn_no: int = -1
        self._acting_player: int = -1
        self._steps_this_turn: int = 0
        self._last_fingerprint: Optional[str] = None
        self._loop_streak: int = 0
        self._board_history: list[str] = []

    def _reset_turn_window(self, *, turn_no: int, acting_player_id: int) -> None:
        self._turn_no = turn_no
        self._acting_player = acting_player_id
        self._steps_this_turn = 0
        self._last_fingerprint = None
        self._loop_streak = 0
        self._board_history = []

    def check(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        *,
        turn_no: int,
        acting_player_id: int,
    ) -> LoopGuardResult:
        """Record this decision point and return whether Pass should be forced."""
        if (
            turn_no != self._turn_no
            or acting_player_id != self._acting_player
        ):
            self._reset_turn_window(
                turn_no=turn_no,
                acting_player_id=acting_player_id,
            )

        self._steps_this_turn += 1

        board_fp = board_state_fingerprint(state)
        if self._board_history and board_fp in self._board_history[:-1]:
            reason = "board_revert"
            return LoopGuardResult(
                force_pass=True,
                turn_steps=self._steps_this_turn,
                loop_streak=self._loop_streak,
                reason=reason,
                forced_action=_resolve_forced_action(legal_actions, reason=reason),
            )
        if not self._board_history or board_fp != self._board_history[-1]:
            self._board_history.append(board_fp)

        fingerprint = decision_point_fingerprint(state, legal_actions)
        if fingerprint == self._last_fingerprint:
            self._loop_streak += 1
        else:
            self._last_fingerprint = fingerprint
            self._loop_streak = 1

        if self._steps_this_turn > self._max_steps_per_turn:
            reason = "turn_step_cap"
            return LoopGuardResult(
                force_pass=True,
                turn_steps=self._steps_this_turn,
                loop_streak=self._loop_streak,
                reason=reason,
                forced_action=_resolve_forced_action(legal_actions, reason=reason),
            )
        if self._loop_streak >= self._loop_repeat_threshold:
            reason = "decision_loop"
            return LoopGuardResult(
                force_pass=True,
                turn_steps=self._steps_this_turn,
                loop_streak=self._loop_streak,
                reason=reason,
                forced_action=_resolve_forced_action(legal_actions, reason=reason),
            )
        return LoopGuardResult(
            force_pass=False,
            turn_steps=self._steps_this_turn,
            loop_streak=self._loop_streak,
            forced_action=None,
        )
