"""C++ FAB engine environment.

Drop-in replacement for :class:`TalisharEngineEnvironment` that loads a
pre-generated, compiled pybind11 ``fab_engine`` module for a specific
deck matchup.  No HTTP calls, no Docker — each step is a direct C++ function
call (~100× faster than the HTTP-backed environment).

Prerequisites
-------------
1. Generate the C++ source for a matchup::

       python scripts/generate_cpp_engine.py \\
           --talishar-src Talishar \\
           --deck1 Ira --deck2 Ira \\
           --out results/cpp_engines/Ira_vs_Ira

2. Implement the card stubs in ``results/cpp_engines/Ira_vs_Ira/cards.h``
   (PHP logic is included as comments above each stub).

3. Build::

       cd results/cpp_engines/Ira_vs_Ira
       pip install pybind11
       cmake -B build .
       cmake --build build --config Release

The compiled ``fab_engine*.pyd`` / ``fab_engine*.so`` is automatically
copied to the engine directory by the CMake post-build step.

Usage
-----
::

    from flesh_and_blood_rlbridge.cpp_engine_environment import (
        CppEngineEnvironment,
        is_cpp_engine_available,
    )

    engine_dir = "results/cpp_engines/Ira_vs_Ira"
    if is_cpp_engine_available(engine_dir):
        env = CppEngineEnvironment(engine_dir=engine_dir, max_turns=2000)
        obs, info = env.reset()
        result = env.step(0)

The observation JSON format mirrors
:meth:`TalisharEngineEnvironment._encode_observation` closely enough that
agents trained on one environment can be transferred to the other with
minimal adaptation.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from rlbridge.environments.base import rlbridgeEnvironment
from rlbridge.protocol.messages import RenderResult, ResetResult, StepResult, TextSpace

from .combat_log_tracker import CombatTurnTracker
from .talishar_default_policy import (
    _CARD_RESOURCE_STATS,
    _CARD_STATS,
    _MIN_BLOCK_VALUE,
    _strip_revert_actions,
)

# Reward constants mirror talishar_engine_environment.py
_TRUNCATION_PENALTY = 0
_REPEAT_ACTION_THRESHOLD = 3
_REPEAT_ACTION_PENALTY = -0.1
_STEP_PENALTY = -0.005


def _matchup_key(deck1: str, deck2: str) -> str:
    """Normalised cache-directory key for a deck pair."""
    return f"{deck1}_vs_{deck2}"


def _find_engine_module(engine_dir: str | Path) -> Optional[Path]:
    """Return the path to a compiled fab_engine module in *engine_dir*, or None."""
    ed = Path(engine_dir)
    for pattern in ("fab_engine*.pyd", "fab_engine*.so", "fab_engine.pyd", "fab_engine.so"):
        for p in ed.glob(pattern):
            return p
    # Also check inside a build sub-directory
    for pattern in ("fab_engine*.pyd", "fab_engine*.so"):
        for p in (ed / "build").rglob(pattern):
            return p
    return None


def is_cpp_engine_available(engine_dir: str | Path) -> bool:
    """Return True if a compiled fab_engine module exists in *engine_dir*."""
    return _find_engine_module(engine_dir) is not None


def load_fab_engine(engine_dir: str | Path) -> Any:
    """Import and return the ``fab_engine`` module from *engine_dir*.

    Adds *engine_dir* (and its build/ subdir) to ``sys.path`` so that the
    compiled extension can be located by the normal import machinery.

    Raises
    ------
    ImportError
        If no compiled module is found in *engine_dir*.
    """
    ed = Path(engine_dir).resolve()
    mod_path = _find_engine_module(ed)
    if mod_path is None:
        raise ImportError(
            f"No compiled fab_engine module found in {ed}.\n"
            "Build it first:\n"
            f"  cd {ed}\n"
            "  pip install pybind11\n"
            "  cmake -B build . && cmake --build build --config Release"
        )

    # Inject the parent directory into sys.path so the normal import works
    mod_dir = str(mod_path.parent)
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)

    # Use importlib to load from explicit path (avoids name collisions)
    spec = importlib.util.spec_from_file_location("fab_engine", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {mod_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class CppEngineEnvironment(rlbridgeEnvironment):
    """RL environment backed by a compiled C++ FAB game engine.

    Observation / action format is Talishar-compatible (same top-level
    contract and legal-action indexing), enabling direct policy transfer.

    Parameters
    ----------
    engine_dir:
        Path to the directory containing the compiled ``fab_engine`` module
        (i.e. ``results/cpp_engines/<matchup_key>/``).
    max_turns:
        Maximum steps before episode truncation.
    deck1, deck2:
        Deck names recorded in info dicts (cosmetic only).
    enable_combat_tracker:
        When ``True`` (default), capture per-step combat traces and board-state
        action statistics for parity checks against Talishar HTTP outcomes.
    """

    def __init__(
        self,
        *,
        engine_dir: str | Path,
        max_turns: int = 2000,
        deck1: str = "",
        deck2: str = "",
        enable_combat_tracker: bool = True,
    ) -> None:
        self._engine_dir = Path(engine_dir).resolve()
        self._max_turns = max_turns
        self._deck1 = deck1
        self._deck2 = deck2
        self._enable_combat_tracker = bool(enable_combat_tracker)
        self._combat_tracker = CombatTurnTracker(
            engine_name="cpp",
            enabled=self._enable_combat_tracker,
        )
        self._synthetic_combat_log: list[str] = []

        # Load the compiled module once at construction time
        self._fab = load_fab_engine(self._engine_dir)

        # Per-episode state
        self._gs: Any = None  # fab_engine.GameState
        self._steps: int = 0
        self._p1_hp: int = 20
        self._p2_hp: int = 20
        self._acting_player: int = 1  # 1-indexed to match Talishar convention
        self._repeat_streak: int = 0
        self._last_action_key: Optional[tuple[int, str]] = None
        self._last_turn_no: int = 0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _new_gamestate(self) -> Any:
        gs = self._fab.GameState()
        gs.register_all_cards()
        if hasattr(gs, "init_standard_decks"):
            gs.init_standard_decks()
        return gs

    def _legal_actions(self) -> list[Any]:
        """Return legal actions from the C++ engine as a list of LegalAction objects."""
        return self._gs.get_legal_actions()

    def _is_pass_like(self, action: Any) -> bool:
        code = int(getattr(action, "action_code", 0) or 0)
        if code in (99, 101, 105):
            return True
        label = str(getattr(action, "label", "") or "").strip().lower()
        return any(tok in label for tok in ("pass", "end turn", "no block", "skip"))

    def _card_cost(self, card: Any) -> int:
        try:
            cost = int(getattr(card, "cost", 0) or 0)
        except (TypeError, ValueError):
            cost = 0
        if cost > 0:
            return cost
        cid = str(getattr(card, "card_id", "") or "").strip()
        if cid in _CARD_RESOURCE_STATS:
            return _CARD_RESOURCE_STATS[cid][1]
        return 0

    def _card_pitch(self, card: Any) -> int:
        try:
            pitch = int(getattr(card, "pitch", 0) or 0)
        except (TypeError, ValueError):
            pitch = 0
        if pitch > 0:
            return pitch
        cid = str(getattr(card, "card_id", "") or "").strip()
        if cid in _CARD_RESOURCE_STATS:
            return _CARD_RESOURCE_STATS[cid][0]
        return 0

    def _is_affordable_hand_play(self, action: Any, hand_cards: list[Any]) -> bool:
        """Return True when a hand play action can be paid with other hand cards."""
        if str(getattr(action, "zone", "") or "").strip().lower() != "hand":
            return True
        if int(getattr(action, "action_code", 0) or 0) != 27:
            return True

        try:
            idx = int(str(getattr(action, "button_input", "") or ""))
        except (TypeError, ValueError):
            return False
        if idx < 0 or idx >= len(hand_cards):
            return False

        card = hand_cards[idx]
        cost = self._card_cost(card)
        if cost <= 0:
            return True

        available = 0
        for j, c in enumerate(hand_cards):
            if j == idx:
                continue
            available += self._card_pitch(c)
        return cost <= available

    def _card_defense(self, card: Any) -> int:
        try:
            defense = int(getattr(card, "defense", 0) or 0)
        except (TypeError, ValueError):
            defense = 0
        if defense > 0:
            return defense
        cid = str(getattr(card, "card_id", "") or "").strip()
        if cid in _CARD_STATS:
            return _CARD_STATS[cid][1]
        return 0

    def _is_hand_block_action(self, action: Any) -> bool:
        return (
            str(getattr(action, "zone", "") or "").strip().lower() == "hand"
            and int(getattr(action, "action_code", 0) or 0) == 27
        )

    def _is_viable_block_play(self, action: Any, hand_cards: list[Any]) -> bool:
        """Return True when a hand play can be used as a dedicated block."""
        if not self._is_hand_block_action(action):
            return True

        try:
            idx = int(str(getattr(action, "button_input", "") or ""))
        except (TypeError, ValueError):
            return False
        if idx < 0 or idx >= len(hand_cards):
            return False

        card = hand_cards[idx]
        if not self._is_affordable_hand_play(action, hand_cards):
            return False
        return self._card_defense(card) >= _MIN_BLOCK_VALUE

    def _apply_block_phase_filter(self, legal: list[Any]) -> list[Any]:
        """During block phase, only offer viable blocks or pass."""
        pass_actions = [a for a in legal if self._is_pass_like(a)]
        hand_cards = self._hand_cards()
        viable_blocks = [
            a
            for a in legal
            if self._is_hand_block_action(a) and self._is_viable_block_play(a, hand_cards)
        ]
        if viable_blocks:
            return viable_blocks + pass_actions

        if pass_actions:
            return [pass_actions[0]]
        return [
            type(
                "_Pass",
                (),
                {
                    "action_code": 99,
                    "button_input": "",
                    "card_id": "",
                    "zone": "button",
                    "label": "Pass",
                },
            )()
        ]

    def _filter_legal_actions(self, legal: list[Any]) -> list[Any]:
        """Filter legal actions to avoid impossible plays and dead loops.

        Note: Filtering is currently disabled to ensure parity with Talishar.
        """
        # If the C++ engine provides more actions than Talishar, we need to be careful.
        # However, the user wants exact parity.
        # If we find a mismatch in count, it's a problem with the engine state.
        return legal

    def _phase_code(self) -> str:
        """Return a Talishar-like phase token from the C++ engine phase value."""
        phase = getattr(self._gs, "phase", None)
        phase_value: Optional[int]
        if phase is None:
            phase_value = None
        else:
            try:
                phase_value = int(phase)
            except (TypeError, ValueError):
                phase_value = None

        # Matches the generated C++ enum order in scripts/generate_cpp_engine.py
        # START=0, MAIN=1, PITCH=2, ATTACK=3, BLOCK=4, DAMAGE=5, END=6, OVER=7
        mapping = {
            0: "startturn",
            1: "m",
            2: "p",
            3: "a",
            4: "d",
            5: "damage",
            6: "endphase",
            7: "OVER",
        }
        return mapping.get(phase_value, "m")

    def _acting_idx(self) -> int:
        return 0 if self._acting_player == 1 else 1

    def _hand_cards(self) -> list[Any]:
        attr = "p1_hand" if self._acting_idx() == 0 else "p2_hand"
        cards = getattr(self._gs, attr, None)
        return list(cards) if isinstance(cards, list) else []

    def _pitch_count(self) -> int:
        attr = "p1_pitch_size" if self._acting_idx() == 0 else "p2_pitch_size"
        try:
            return int(getattr(self._gs, attr, 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _encode_player_hand(self, legal: list[Any]) -> list[dict[str, Any]]:
        """Return Talishar-compatible playerHand entries for all hand cards."""
        cards = self._hand_cards()
        if not cards:
            # Back-compat fallback for older compiled modules lacking hand bindings.
            return [
                {
                    "cardID": str(getattr(a, "card_id", "") or ""),
                    "action": int(getattr(a, "action_code", 0) or 0),
                    "actionDataOverride": str(getattr(a, "button_input", "") or ""),
                    "label": str(getattr(a, "label", "") or ""),
                }
                for a in legal
                if str(getattr(a, "zone", "")).strip().lower() == "hand"
            ]

        hand_legal: dict[str, Any] = {}
        for a in legal:
            zone = str(getattr(a, "zone", "") or "").strip().lower()
            if zone == "hand" and int(getattr(a, "action_code", 0) or 0) == 27:
                hand_legal[str(getattr(a, "button_input", "") or "")] = a

        out: list[dict[str, Any]] = []
        for i, c in enumerate(cards):
            key = str(i)
            a = hand_legal.get(key)
            card_id = str(getattr(c, "card_id", "") or "")
            fallback_label = str(getattr(c, "name", "") or card_id)
            if a is not None:
                out.append(
                    {
                        "cardID": card_id,
                        "action": 27,
                        "actionDataOverride": key,
                        "label": str(getattr(a, "label", "") or fallback_label),
                    }
                )
            else:
                out.append(
                    {
                        "cardID": card_id,
                        "action": 0,
                        "actionDataOverride": "",
                        "label": fallback_label,
                    }
                )
        return out

    def _encode_observation(self, legal: list[Any]) -> str:
        """Encode the current C++ game state as a JSON string.

        Format mirrors :meth:`TalisharEngineEnvironment._encode_observation`
        (same keys and legal action indexing).
        """
        gs = self._gs
        legal = self._filter_legal_actions(legal)
        player_hand = self._encode_player_hand(legal)
        acting_idx = self._acting_idx()
        player_hand_size = (
            len(player_hand)
            if player_hand
            else (gs.p1_hand_size if acting_idx == 0 else gs.p2_hand_size)
        )
        obs: dict[str, Any] = {
            "actingPlayerID": self._acting_player,
            "selfPlay": True,
            "playerHealth": gs.p1_health if self._acting_player == 1 else gs.p2_health,
            "opponentHealth": gs.p2_health if self._acting_player == 1 else gs.p1_health,
            "turnNo": gs.turn_no,
            "turnPhase": self._phase_code(),
            "havePriority": not self._is_game_over(),
            "playerHandSize": player_hand_size,
            "opponentHandSize": (gs.p2_hand_size if self._acting_player == 1 else gs.p1_hand_size),
            "playerDeckCount": (gs.p1_deck_size if self._acting_player == 1 else gs.p2_deck_size),
            "opponentDeckCount": (gs.p2_deck_size if self._acting_player == 1 else gs.p1_deck_size),
            "playerPitchCount": self._pitch_count(),
            "playerHand": player_hand,
            "legalActions": [
                {"index": i, "label": a.label, "zone": a.zone} for i, a in enumerate(legal)
            ],
        }
        return json.dumps(obs, separators=(",", ":"))

    def _parse_action(self, action: Any, legal: list[Any]) -> tuple[int, Any]:
        """Return (list_index, LegalAction) for an integer or 'pass' action."""
        action_str = str(action).strip().lower()
        if action_str == "pass":
            # Find the pass action
            for i, a in enumerate(legal):
                if a.action_code == 99:
                    return i, a
            return len(legal) - 1, legal[-1]
        try:
            idx = int(action_str)
            idx = max(0, min(idx, len(legal) - 1))
            return idx, legal[idx]
        except (ValueError, TypeError):
            return len(legal) - 1, legal[-1]

    def _repeat_penalty(self, action_code: int, button_input: str) -> float:
        key = (action_code, button_input)
        turn = self._gs.turn_no
        player = self._gs.priority

        if turn != self._last_turn_no or player != (self._acting_player - 1):
            self._last_turn_no = turn
            self._last_action_key = key
            self._repeat_streak = 1
            return 0.0

        if key == self._last_action_key:
            self._repeat_streak += 1
        else:
            self._last_action_key = key
            self._repeat_streak = 1

        if self._repeat_streak >= _REPEAT_ACTION_THRESHOLD:
            return _REPEAT_ACTION_PENALTY
        return 0.0

    def _is_game_over(self) -> bool:
        return self._gs.game_over or self._gs.p1_health <= 0 or self._gs.p2_health <= 0

    def _compute_reward(
        self,
        prev_p1: int,
        prev_p2: int,
        terminated: bool,
        truncated: bool,
        repeat_penalty: float,
    ) -> float:
        gs = self._gs
        if terminated:
            # P1-centric: +1 if P1 won, -1 if P2 won, 0 for draw.
            # The training loop always negates this for P2's buffer
            # (agent_reward = env_reward if acting==1 else -env_reward).
            if gs.winner == 0:  # P1 won (0-indexed)
                reward = 1.0
            elif gs.winner == 1:  # P2 won
                reward = -1.0
            else:
                reward = 0.0  # draw / unresolved
        elif truncated:
            reward = float(_TRUNCATION_PENALTY)
        else:
            # P1-centric intermediate shaping: positive when P2 takes damage,
            # negative when P1 takes damage, regardless of who is acting.
            p1_now, p2_now = gs.p1_health, gs.p2_health
            dmg_dealt = max(0, prev_p2 - p2_now)  # P2 HP lost  → good for P1
            dmg_taken = max(0, prev_p1 - p1_now)  # P1 HP lost  → bad for P1
            reward = dmg_dealt * 0.01 - dmg_taken * 0.01 + _STEP_PENALTY
        return reward + repeat_penalty

    def _action_to_dict(self, action: Any) -> dict[str, Any]:
        return {
            "action_code": int(getattr(action, "action_code", 0) or 0),
            "button_input": str(getattr(action, "button_input", "") or ""),
            "card_id": str(getattr(action, "card_id", "") or ""),
            "zone": str(getattr(action, "zone", "") or ""),
            "label": str(getattr(action, "label", "") or ""),
        }

    def _legal_to_dicts(self, legal: list[Any]) -> list[dict[str, Any]]:
        return [self._action_to_dict(a) for a in legal]

    def _tracker_state_snapshot(self, legal: list[Any]) -> dict[str, Any]:
        gs = self._gs
        acting = int(self._acting_player)
        return {
            "acting_player_id": acting,
            "turn_no": int(getattr(gs, "turn_no", 0) or 0),
            "phase": self._phase_code(),
            "player_health": int(gs.p1_health if acting == 1 else gs.p2_health),
            "opponent_health": int(gs.p2_health if acting == 1 else gs.p1_health),
            "player_hand_size": int(gs.p1_hand_size if acting == 1 else gs.p2_hand_size),
            "opponent_hand_size": int(gs.p2_hand_size if acting == 1 else gs.p1_hand_size),
            "player_deck_count": int(gs.p1_deck_size if acting == 1 else gs.p2_deck_size),
            "opponent_deck_count": int(gs.p2_deck_size if acting == 1 else gs.p1_deck_size),
            "player_pitch_count": int(self._pitch_count()),
            "legal_count": len(legal),
        }

    def _append_synthetic_log(
        self,
        before: dict[str, Any],
        action: dict[str, Any],
        after: dict[str, Any],
        prev_p1: int,
        prev_p2: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        action_label = str(action.get("label", "") or "").strip()
        if not action_label:
            action_label = f"mode={int(action.get('action_code', 0))}"
        zone = str(action.get("zone", "") or "")
        self._synthetic_combat_log.append(
            f"P{before.get('acting_player_id', 1)} {action_label} ({zone})"
        )

        p1_now = int(self._gs.p1_health)
        p2_now = int(self._gs.p2_health)
        if prev_p1 != p1_now or prev_p2 != p2_now:
            self._synthetic_combat_log.append(f"HP P1 {prev_p1}->{p1_now} | P2 {prev_p2}->{p2_now}")

        if str(before.get("phase", "")) != str(after.get("phase", "")):
            self._synthetic_combat_log.append(
                f"Phase {before.get('phase', '')} -> {after.get('phase', '')}"
            )

        if terminated:
            winner = int(getattr(self._gs, "winner", -1))
            winner_label = "draw" if winner < 0 else f"P{winner + 1}"
            self._synthetic_combat_log.append(f"Game over winner={winner_label}")
        elif truncated:
            self._synthetic_combat_log.append("Episode truncated")

        if len(self._synthetic_combat_log) > 4000:
            self._synthetic_combat_log = self._synthetic_combat_log[-4000:]

    def _tracker_stub(self, latest_event: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if not self._enable_combat_tracker:
            return {"enabled": False}
        snap = self._combat_tracker.snapshot(top_k=5, tail_events=0, tail_log_lines=25)
        out: dict[str, Any] = {
            "enabled": True,
            "engine": str(snap.get("engine", "cpp")),
            "steps_recorded": int(snap.get("steps_recorded", 0) or 0),
            "trace_digest": str(snap.get("trace_digest", "") or ""),
        }
        if latest_event is not None:
            out["latest_event"] = latest_event
        return out

    def get_combat_tracker_snapshot(
        self,
        *,
        top_k: int = 10,
        tail_events: int = 20,
        tail_log_lines: int = 40,
    ) -> dict[str, Any]:
        """Return a detailed snapshot of tracked combat/turn statistics."""
        return self._combat_tracker.snapshot(
            top_k=top_k,
            tail_events=tail_events,
            tail_log_lines=tail_log_lines,
        )

    def get_combat_trace(self) -> list[dict[str, Any]]:
        """Return the full per-step trace captured by the combat tracker."""
        return self._combat_tracker.trace()

    def clear_combat_tracker(self) -> None:
        """Clear all currently tracked combat/turn events and counters."""
        self._combat_tracker.clear()
        self._synthetic_combat_log = []

    # ── rlbridge interface ────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        self._gs = self._new_gamestate()
        self._steps = 0
        self._p1_hp = self._gs.p1_health
        self._p2_hp = self._gs.p2_health
        self._acting_player = self._gs.priority + 1  # convert 0-indexed → 1-indexed
        self._repeat_streak = 0
        self._last_action_key = None
        self._last_turn_no = 0

        legal = self._filter_legal_actions(self._legal_actions())
        obs = self._encode_observation(legal)

        if self._enable_combat_tracker:
            self._synthetic_combat_log = [f"matchup {self._deck1 or 'P1'} vs {self._deck2 or 'P2'}"]
            self._combat_tracker.reset(
                initial_snapshot=self._tracker_state_snapshot(legal),
                initial_legal_actions=self._legal_to_dicts(legal),
                combat_log_lines=self._synthetic_combat_log,
                metadata={
                    "engine": "cpp",
                    "engine_dir": str(self._engine_dir),
                    "deck1": self._deck1,
                    "deck2": self._deck2,
                },
            )
        else:
            self._combat_tracker.clear()
            self._synthetic_combat_log = []

        return ResetResult(
            observation=obs,
            info={
                "engine": "cpp",
                "engine_dir": str(self._engine_dir),
                "legal_actions": [
                    {
                        "action_code": a.action_code,
                        "button_input": a.button_input,
                        "card_id": a.card_id,
                        "zone": a.zone,
                        "label": a.label,
                    }
                    for a in legal
                ],
                "player_hp": self._p1_hp,
                "opponent_hp": self._p2_hp,
                "acting_player_id": self._acting_player,
                "self_play": True,
                "combat_tracker": self._tracker_stub(),
            },
        )

    def step(self, action: Any) -> StepResult:
        assert self._gs is not None, "call reset() first"

        legal = self._filter_legal_actions(self._legal_actions())
        before_snapshot = self._tracker_state_snapshot(legal)
        legal_before = self._legal_to_dicts(legal)

        idx, chosen = self._parse_action(action, legal)
        chosen_dict = self._action_to_dict(chosen)

        prev_p1 = self._gs.p1_health
        prev_p2 = self._gs.p2_health

        # Apply to C++ engine
        self._gs.apply_action(chosen)
        self._steps += 1

        # Update acting player (C++ priority is 0-indexed)
        self._acting_player = self._gs.priority + 1

        terminated = self._is_game_over()
        truncated = not terminated and self._steps >= self._max_turns

        repeat_pen = self._repeat_penalty(chosen.action_code, chosen.button_input)
        reward = self._compute_reward(prev_p1, prev_p2, terminated, truncated, repeat_pen)

        new_legal = self._filter_legal_actions(self._legal_actions())
        obs = self._encode_observation(new_legal)
        after_snapshot = self._tracker_state_snapshot(new_legal)

        tracker_event: Optional[dict[str, Any]] = None
        if self._enable_combat_tracker:
            self._append_synthetic_log(
                before_snapshot,
                chosen_dict,
                after_snapshot,
                prev_p1,
                prev_p2,
                terminated,
                truncated,
            )
            tracker_event = self._combat_tracker.record_step(
                before_snapshot=before_snapshot,
                after_snapshot=after_snapshot,
                action=chosen_dict,
                legal_before=legal_before,
                legal_after=self._legal_to_dicts(new_legal),
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                combat_log_lines=self._synthetic_combat_log,
            )

        return StepResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "engine": "cpp",
                "legal_actions": [
                    {
                        "action_code": a.action_code,
                        "button_input": a.button_input,
                        "card_id": a.card_id,
                        "zone": a.zone,
                        "label": a.label,
                    }
                    for a in new_legal
                ],
                "turn": self._gs.turn_no,
                "player_hp": self._gs.p1_health,
                "opponent_hp": self._gs.p2_health,
                "acting_player_id": self._acting_player,
                "self_play": True,
                "repeat_streak": self._repeat_streak,
                "repeat_penalty": repeat_pen,
                "combat_tracker": self._tracker_stub(tracker_event),
            },
        )

    def sample_action(self) -> str:
        """Return a random legal action index as a string."""
        import random

        legal = self._filter_legal_actions(self._legal_actions())
        if not legal:
            return "0"
        return str(random.randrange(len(legal)))

    def render(self) -> RenderResult:
        if self._gs is None:
            return RenderResult(mode="ansi", text="No game state.")
        gs = self._gs
        lines = [
            "=== Flesh and Blood (C++ Engine) ===",
            f"Turn: {gs.turn_no}  Priority: P{self._acting_player}",
            f"P1: {gs.p1_health} HP  |  P2: {gs.p2_health} HP",
            f"P1 hand: {gs.p1_hand_size}  deck: {gs.p1_deck_size}",
            f"P2 hand: {gs.p2_hand_size}  deck: {gs.p2_deck_size}",
        ]
        legal = self._filter_legal_actions(self._legal_actions())
        if legal:
            lines.append("Legal actions:")
            for i, a in enumerate(legal[:10]):
                lines.append(f"  [{i}] {a.label} ({a.zone})")
        return RenderResult(mode="ansi", text="\n".join(lines))

    @property
    def observation_space(self) -> TextSpace:
        return TextSpace(min_length=0, max_length=8000)

    @property
    def action_space(self) -> TextSpace:
        return TextSpace(min_length=1, max_length=16)

    def close(self) -> None:
        self._gs = None
        self._combat_tracker.clear()
        self._synthetic_combat_log = []


# ── Cache management ──────────────────────────────────────────────────────────

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "cpp_engines"


def get_engine_dir(
    deck1: str,
    deck2: str,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return the canonical cache directory for a deck matchup.

    Prefers an exact ``{deck1}_vs_{deck2}`` directory.  If that directory
    has no compiled module, falls back to the most-recently-modified
    hashed variant ``{deck1}_vs_{deck2}-<hash>`` (produced by
    build_cpp_engine_for_matchup.ps1 when content-hashing is enabled).
    """
    base = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    key = _matchup_key(deck1, deck2)
    exact = base / key
    if is_cpp_engine_available(exact):
        return exact
    # Search for hashed variants: aurora_vs_briar-<16hex>
    candidates = sorted(
        base.glob(f"{key}-*"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    for candidate in candidates:
        if candidate.is_dir() and is_cpp_engine_available(candidate):
            return candidate
    # Return exact dir as default even if empty (lets callers report the error)
    return exact


def get_or_none(
    deck1: str,
    deck2: str,
    cache_dir: str | Path | None = None,
    max_turns: int = 2000,
    enable_combat_tracker: bool = True,
) -> Optional["CppEngineEnvironment"]:
    """Return a :class:`CppEngineEnvironment` if one is compiled for this matchup.

    Returns ``None`` without raising if the compiled module does not exist
    (caller should fall back to :class:`TalisharEngineEnvironment`).
    """
    engine_dir = get_engine_dir(deck1, deck2, cache_dir)
    if not is_cpp_engine_available(engine_dir):
        return None
    try:
        return CppEngineEnvironment(
            engine_dir=engine_dir,
            max_turns=max_turns,
            deck1=deck1,
            deck2=deck2,
            enable_combat_tracker=enable_combat_tracker,
        )
    except Exception:
        return None
