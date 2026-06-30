"""Cross-turn stall detection for RL environments.

Detects macro stalls that per-turn :class:`~state_loop_guard.TurnLoopGuard`
and eval-loop heuristics miss: many turns without damage, or repeated main-phase
decisions where only Pass is legal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .legal_action_filter import is_pass_only
from .talishar_default_policy import _get_phase, _to_int


@dataclass(frozen=True)
class MacroStallConfig:
    """Thresholds for cross-turn stall truncation."""

    enabled: bool = True
    stall_no_damage_turns: int = 6
    stall_pass_only_turns: int = 6
    stall_no_damage_requires_low_hand: bool = False
    stall_low_hand_turns: int = 3
    stall_max_single_low_hand_turns: int = 5
    stall_min_attack_hand: int = 2


@dataclass(frozen=True)
class MacroStallResult:
    should_truncate: bool = False
    reason: str = ""
    turns_without_damage: int = 0
    pass_only_main_streak: int = 0


def _total_hp(
    state: dict[str, Any],
    *,
    p1_hp: int | None = None,
    p2_hp: int | None = None,
) -> float:
    if p1_hp is not None and p2_hp is not None:
        return float(p1_hp) + float(p2_hp)
    p1 = float(state.get("playerHealth", 0) or 0)
    p2 = float(state.get("opponentHealth", 0) or 0)
    return p1 + p2


def _acting_player(state: dict[str, Any]) -> int:
    return int(state.get("actingPlayerID", 1) or 1)


def _hand_size_for_acting(state: dict[str, Any]) -> int:
    return int(state.get("playerHandSize", 0) or 0)


@dataclass
class MacroStallGuard:
    """Track turn-level stall signals and decide when to truncate."""

    config: MacroStallConfig = field(default_factory=MacroStallConfig)
    _last_seen_turn_no: int = field(default=0, init=False)
    _turn_start_total_hp: float = field(default=0.0, init=False)
    _turns_without_damage: int = field(default=0, init=False)
    _pass_only_main_streak: int = field(default=0, init=False)
    _seen_main_markers: set[tuple[int, int]] = field(default_factory=set, init=False)
    _low_hand_turn_streak: dict[int, int] = field(default_factory=dict, init=False)

    def reset(self) -> None:
        self._last_seen_turn_no = 0
        self._turn_start_total_hp = 0.0
        self._turns_without_damage = 0
        self._pass_only_main_streak = 0
        self._seen_main_markers.clear()
        self._low_hand_turn_streak.clear()

    def observe(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        *,
        p1_hp: int | None = None,
        p2_hp: int | None = None,
    ) -> MacroStallResult:
        """Update stall counters from post-action *state* and return truncation decision."""
        if not self.config.enabled:
            return MacroStallResult()

        turn_no = _to_int(state.get("turnNo", 0))
        total_hp = _total_hp(state, p1_hp=p1_hp, p2_hp=p2_hp)
        acting = _acting_player(state)
        phase = _get_phase(state)

        if turn_no > self._last_seen_turn_no:
            if self._last_seen_turn_no > 0:
                if total_hp >= self._turn_start_total_hp:
                    self._turns_without_damage += 1
                else:
                    self._turns_without_damage = 0
            self._turn_start_total_hp = total_hp
            self._last_seen_turn_no = turn_no

        if phase == "m":
            marker = (acting, turn_no)
            if marker not in self._seen_main_markers:
                self._seen_main_markers.add(marker)
                hand_size = _hand_size_for_acting(state)
                if hand_size < self.config.stall_min_attack_hand:
                    self._low_hand_turn_streak[acting] = (
                        self._low_hand_turn_streak.get(acting, 0) + 1
                    )
                else:
                    self._low_hand_turn_streak[acting] = 0

                if is_pass_only(legal_actions):
                    self._pass_only_main_streak += 1
                else:
                    self._pass_only_main_streak = 0

        reason = ""
        should_truncate = False

        if self._turns_without_damage >= self.config.stall_no_damage_turns:
            if self.config.stall_no_damage_requires_low_hand:
                both_low = (
                    self._low_hand_turn_streak.get(1, 0)
                    >= self.config.stall_low_hand_turns
                    and self._low_hand_turn_streak.get(2, 0)
                    >= self.config.stall_low_hand_turns
                )
                one_sided_low = (
                    max(
                        self._low_hand_turn_streak.get(1, 0),
                        self._low_hand_turn_streak.get(2, 0),
                    )
                    >= self.config.stall_max_single_low_hand_turns
                )
                if both_low or one_sided_low:
                    should_truncate = True
                    reason = "no_damage_turns"
            else:
                should_truncate = True
                reason = "no_damage_turns"

        if (
            not should_truncate
            and self._pass_only_main_streak >= self.config.stall_pass_only_turns
        ):
            should_truncate = True
            reason = "pass_only_main"

        return MacroStallResult(
            should_truncate=should_truncate,
            reason=reason,
            turns_without_damage=self._turns_without_damage,
            pass_only_main_streak=self._pass_only_main_streak,
        )
