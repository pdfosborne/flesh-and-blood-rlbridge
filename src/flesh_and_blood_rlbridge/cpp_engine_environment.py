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
        env = CppEngineEnvironment(engine_dir=engine_dir, max_turns=200)
        obs, info = env.reset()
        result = env.step(0)

The observation JSON format is identical to
:meth:`TalisharEngineEnvironment._encode_observation` so agents trained on
one environment transfer seamlessly to the other.
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

    Observation / action format is identical to
    :class:`TalisharEngineEnvironment` so agents are interchangeable.

    Parameters
    ----------
    engine_dir:
        Path to the directory containing the compiled ``fab_engine`` module
        (i.e. ``results/cpp_engines/<matchup_key>/``).
    max_turns:
        Maximum steps before episode truncation.
    deck1, deck2:
        Deck names recorded in info dicts (cosmetic only).
    """

    def __init__(
        self,
        *,
        engine_dir: str | Path,
        max_turns: int = 200,
        deck1: str = "",
        deck2: str = "",
    ) -> None:
        self._engine_dir = Path(engine_dir).resolve()
        self._max_turns = max_turns
        self._deck1 = deck1
        self._deck2 = deck2

        # Load the compiled module once at construction time
        self._fab = load_fab_engine(self._engine_dir)

        # Per-episode state
        self._gs: Any = None          # fab_engine.GameState
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
        return gs

    def _legal_actions(self) -> list[Any]:
        """Return legal actions from the C++ engine as a list of LegalAction objects."""
        return self._gs.get_legal_actions()

    def _encode_observation(self, legal: list[Any]) -> str:
        """Encode the current C++ game state as a JSON string.

        Format mirrors :meth:`TalisharEngineEnvironment._encode_observation`
        so agents trained on Talishar transfer directly.
        """
        gs = self._gs
        obs: dict[str, Any] = {
            "actingPlayerID": self._acting_player,
            "selfPlay": True,
            "playerHealth": gs.p1_health if self._acting_player == 1 else gs.p2_health,
            "opponentHealth": gs.p2_health if self._acting_player == 1 else gs.p1_health,
            "turnNo": gs.turn_no,
            "turnPhase": "m",           # simplified — full phase tracking is TODO
            "havePriority": True,
            "playerHandSize": (
                gs.p1_hand_size if self._acting_player == 1 else gs.p2_hand_size
            ),
            "opponentHandSize": (
                gs.p2_hand_size if self._acting_player == 1 else gs.p1_hand_size
            ),
            "playerDeckCount": (
                gs.p1_deck_size if self._acting_player == 1 else gs.p2_deck_size
            ),
            "opponentDeckCount": (
                gs.p2_deck_size if self._acting_player == 1 else gs.p1_deck_size
            ),
            "playerPitchCount": 0,
            "playerHand": [],           # hand detail not yet exposed by bindings
            "legalActions": [
                {"index": i, "label": a.label, "zone": a.zone}
                for i, a in enumerate(legal)
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
            # Winner is 0-indexed (0=P1, 1=P2); acting_player is 1-indexed
            acting_idx = self._acting_player - 1
            if gs.winner == acting_idx:
                reward = 1.0
            elif gs.winner == 1 - acting_idx:
                reward = -1.0
            else:
                reward = 0.0  # draw / unresolved
        elif truncated:
            reward = float(_TRUNCATION_PENALTY)
        else:
            # Small intermediate reward for damage dealt/taken
            p1_now, p2_now = gs.p1_health, gs.p2_health
            if self._acting_player == 1:
                dmg_dealt = max(0, prev_p2 - p2_now)
                dmg_taken = max(0, prev_p1 - p1_now)
            else:
                dmg_dealt = max(0, prev_p1 - p1_now)
                dmg_taken = max(0, prev_p2 - p2_now)
            reward = dmg_dealt * 0.01 - dmg_taken * 0.01 + _STEP_PENALTY
        return reward + repeat_penalty

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

        legal = self._legal_actions()
        obs = self._encode_observation(legal)
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
            },
        )

    def step(self, action: Any) -> StepResult:
        assert self._gs is not None, "call reset() first"

        legal = self._legal_actions()
        if not legal:
            # No legal actions — pass and end
            legal = [type("_Pass", (), {
                "action_code": 99, "button_input": "", "card_id": "",
                "zone": "button", "label": "Pass"
            })()]

        idx, chosen = self._parse_action(action, legal)

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

        new_legal = self._legal_actions()
        obs = self._encode_observation(new_legal)

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
            },
        )

    def sample_action(self) -> str:
        """Return a random legal action index as a string."""
        import random
        legal = self._legal_actions()
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
        legal = self._legal_actions()
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


# ── Cache management ──────────────────────────────────────────────────────────

_DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "results" / "cpp_engines"
)


def get_engine_dir(
    deck1: str,
    deck2: str,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return the canonical cache directory for a deck matchup."""
    base = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    return base / _matchup_key(deck1, deck2)


def get_or_none(
    deck1: str,
    deck2: str,
    cache_dir: str | Path | None = None,
    max_turns: int = 200,
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
        )
    except Exception:
        return None
