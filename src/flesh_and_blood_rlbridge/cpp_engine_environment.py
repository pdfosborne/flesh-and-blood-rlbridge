"""C++ FAB engine environment.

Drop-in replacement for :class:`TalisharEngineEnvironment` that loads a
pre-generated, compiled pybind11 ``fab_engine`` module for a specific
deck matchup.  No HTTP calls, no Docker â€” each step is a direct C++ function
call (~100Ã— faster than the HTTP-backed environment).

Prerequisites
-------------
1. Generate the C++ source for a matchup::

       python scripts/cpp/generate_cpp_engine.py \\
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
import sysconfig
from pathlib import Path
from typing import Any, Optional

import numpy as np

from rlbridge.environments.base import rlbridgeEnvironment
from rlbridge.protocol.messages import RenderResult, ResetResult, StepResult, TextSpace

from .combat_log_tracker import CombatTurnTracker
from .deck_context import EpisodeContext, load_episode_context, hero_from_equipment
from .player_observation import (
    ACTION_CAPACITY,
    PLAYER_OBS_DIM,
    PLAYER_OBS_SCHEMA_VERSION,
    player_observation_payload,
    player_observation_vector,
)
from .legal_action_filter import (
    filter_legal_actions,
    materialize_filtered_actions,
    normalize_action_descriptor,
)
from .game_state_parity import is_syncable_card_id
from .obs_alignment import (
    align_observation_for_cpp_training,
    cpp_obs_alignment_enabled,
    merge_talishar_raw_state,
)
from .obs_encoding import observation_fingerprint
from .macro_stall_guard import MacroStallConfig, MacroStallGuard, MacroStallResult
from .state_loop_guard import (
    DEFAULT_LOOP_REPEAT_THRESHOLD,
    DEFAULT_MAX_STEPS_PER_TURN,
    LoopGuardResult,
    TurnLoopGuard,
    resolve_forced_submission,
)
from .talishar_default_policy import (
    RepeatActionTracker,
    choose_talishar_action_index,
)

# Reward defaults mirror MetaEngineControls in runtime_defaults.py (training wires RUNTIME.engine).
def _matchup_key(deck1: str, deck2: str) -> str:
    """Normalised cache-directory key for a deck pair."""
    return f"{deck1}_vs_{deck2}"


def python_extension_suffix() -> str:
    """Return the active interpreter's extension suffix (e.g. ``.cp312-win_amd64.pyd``)."""
    return str(sysconfig.get_config_var("EXT_SUFFIX") or (".pyd" if os.name == "nt" else ".so"))


def expected_fab_engine_module_name() -> str:
    """Filename of ``fab_engine`` built for the current Python interpreter."""
    return f"fab_engine{python_extension_suffix()}"


def _module_matches_current_python(path: Path) -> bool:
    if not path.is_file():
        return False
    expected = expected_fab_engine_module_name()
    if path.name == expected:
        return True
    # Legacy builds without an ABI tag (fab_engine.pyd) — accept only if import works.
    if path.name in {"fab_engine.pyd", "fab_engine.so"}:
        return True
    return False


def _iter_engine_module_candidates(engine_dir: Path) -> list[Path]:
    """Return compiled-module candidates, newest compatible build first."""
    patterns = (
        expected_fab_engine_module_name(),
        "fab_engine*.pyd",
        "fab_engine*.so",
        "fab_engine.pyd",
        "fab_engine.so",
    )
    seen: set[Path] = set()
    candidates: list[Path] = []

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            return
        seen.add(resolved)
        candidates.append(path)

    for pattern in patterns:
        for path in engine_dir.glob(pattern):
            add(path)
    build_dir = engine_dir / "build"
    if build_dir.is_dir():
        for pattern in patterns[1:]:
            for path in build_dir.rglob(pattern):
                add(path)

    def sort_key(path: Path) -> tuple[int, float]:
        compatible = int(_module_matches_current_python(path))
        return (compatible, path.stat().st_mtime)

    candidates.sort(key=sort_key, reverse=True)
    return candidates


def _find_engine_module(engine_dir: str | Path) -> Optional[Path]:
    """Return a compiled ``fab_engine`` module for the active Python, if any."""
    ed = Path(engine_dir)
    for path in _iter_engine_module_candidates(ed):
        if _module_matches_current_python(path):
            return path
    return None


def is_cpp_engine_available(engine_dir: str | Path) -> bool:
    """Return True if a compatible ``fab_engine`` binary exists in *engine_dir*."""
    return _find_engine_module(Path(engine_dir)) is not None


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
        expected = expected_fab_engine_module_name()
        stale = [
            p.name
            for p in _iter_engine_module_candidates(ed)
            if p.name != expected
        ]
        stale_hint = ""
        if stale:
            stale_hint = (
                f"\nFound module(s) for a different Python ABI: {', '.join(sorted(set(stale)))}"
                f"\nRebuild for Python {sys.version_info.major}.{sys.version_info.minor}:"
                f"\n  cd {ed}"
                f"\n  cmake -B build . && cmake --build build --config Release"
            )
        raise ImportError(
            f"No compiled fab_engine module for the active interpreter ({expected}) in {ed}."
            f"{stale_hint}"
        )

    if os.name == "nt":
        for dll_dir in (
            mod_path.parent,
            ed,
            ed / "build" / "Release",
            ed / "build" / "fab_engine.dir" / "Release",
            Path(sys.executable).resolve().parent,
            Path(sys.executable).resolve().parent / "DLLs",
            Path(sys.executable).resolve().parent / "Library" / "bin",
        ):
            if dll_dir.is_dir():
                try:
                    os.add_dll_directory(str(dll_dir))
                except (AttributeError, OSError):
                    pass

    # Inject the parent directory into sys.path so the normal import works
    mod_dir = str(mod_path.parent)
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)

    # Use importlib to load from explicit path.  Each compiled extension exports
    # PyInit_fab_engine, so the module name must stay ``fab_engine``.  Parity
    # sweeps that load multiple matchups in one process must run each matchup in
    # a fresh subprocess (see run_matchup_parity).
    if "fab_engine" in sys.modules:
        del sys.modules["fab_engine"]
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
    max_steps_per_turn:
        Maximum agent decisions per player turn before Pass is forced.
    loop_repeat_threshold:
        Force Pass after this many visits to the same decision point in one turn.
    deck1, deck2:
        Deck names recorded in info dicts (cosmetic only).
    enable_combat_tracker:
        When ``True``, capture per-step combat traces and board-state
        action statistics for parity checks against Talishar HTTP outcomes.
        Combat tracking applies to :meth:`step` only; the numeric fast training
        path (:meth:`fast_reset` / :meth:`fast_step_index`) remains available.
    """

    def __init__(
        self,
        *,
        engine_dir: str | Path,
        max_turns: int = 2000,
        max_steps_per_turn: int = DEFAULT_MAX_STEPS_PER_TURN,
        loop_repeat_threshold: int = DEFAULT_LOOP_REPEAT_THRESHOLD,
        step_penalty: float = -0.001,
        truncation_penalty: float = -0.1,
        repeat_action_threshold: int = 3,
        repeat_action_penalty: float = -0.1,
        damage_reward_scale: float = 0.01,
        max_consecutive_passes: int = 20,
        deck1: str = "",
        deck2: str = "",
        enable_combat_tracker: bool = False,
        strict_simulation: bool = False,
        macro_stall_enabled: bool = True,
        stall_no_damage_turns: int = 6,
        stall_pass_only_turns: int = 6,
        stall_no_damage_requires_low_hand: bool = False,
        stall_low_hand_turns: int = 3,
        stall_max_single_low_hand_turns: int = 5,
        stall_min_attack_hand: int = 2,
    ) -> None:
        self._engine_dir = Path(engine_dir).resolve()
        self._max_turns = max_turns
        self._max_steps_per_turn = max_steps_per_turn
        self._loop_repeat_threshold = loop_repeat_threshold
        self._step_penalty = float(step_penalty)
        self._truncation_penalty = float(truncation_penalty)
        self._repeat_action_threshold = int(repeat_action_threshold)
        self._repeat_action_penalty = float(repeat_action_penalty)
        self._damage_reward_scale = float(damage_reward_scale)
        self._max_consecutive_passes = int(max_consecutive_passes)
        self._deck1 = deck1
        self._deck2 = deck2
        self._enable_combat_tracker = bool(enable_combat_tracker)
        self._strict_simulation = bool(strict_simulation)
        self._combat_tracker = CombatTurnTracker(
            engine_name="cpp",
            enabled=self._enable_combat_tracker,
        )
        self._synthetic_combat_log: list[str] = []

        self._hand_playability: dict[int, set[str]] = {}
        self._flow_phase: str = "OPENING_MAIN"
        self._arsenal_complete: set[int] = set()
        self._turn_no_override: Optional[int] = None
        self._talishar_overlay: Optional[dict[str, Any]] = None
        self._talishar_mirror_state: Optional[dict[str, Any]] = None
        self._talishar_raw_state: Optional[dict[str, Any]] = None
        self._talishar_parity_extra: Optional[dict[str, Any]] = None

        # Load the compiled module once at construction time
        self._fab = load_fab_engine(self._engine_dir)

        # Per-episode state
        self._gs: Any = None  # fab_engine.GameState
        self._steps: int = 0
        self._p1_hp: int = 20
        self._p2_hp: int = 20
        self._acting_player: int = 1  # 1-indexed to match Talishar convention
        self._repeat_tracker = RepeatActionTracker()
        self._loop_guard = TurnLoopGuard(
            max_steps_per_turn=max_steps_per_turn,
            loop_repeat_threshold=loop_repeat_threshold,
        )
        self._macro_stall_guard = MacroStallGuard(
            MacroStallConfig(
                enabled=macro_stall_enabled,
                stall_no_damage_turns=stall_no_damage_turns,
                stall_pass_only_turns=stall_pass_only_turns,
                stall_no_damage_requires_low_hand=stall_no_damage_requires_low_hand,
                stall_low_hand_turns=stall_low_hand_turns,
                stall_max_single_low_hand_turns=stall_max_single_low_hand_turns,
                stall_min_attack_hand=stall_min_attack_hand,
            )
        )
        self._last_observation_vec: Optional[np.ndarray] = None
        self._p1_episode_context: Optional[EpisodeContext] = None
        self._p2_episode_context: Optional[EpisodeContext] = None
        if deck1 and deck2:
            self._p1_episode_context = load_episode_context(
                self_deck_name=deck1,
                opponent_deck_name=deck2,
            )
            self._p2_episode_context = load_episode_context(
                self_deck_name=deck2,
                opponent_deck_name=deck1,
            )

    def _obs_vec_for_json(self, obs_json: str) -> np.ndarray:
        vec = observation_fingerprint(obs_json)
        self._last_observation_vec = vec
        return vec

    def _episode_context_for_acting_player(self) -> Optional[EpisodeContext]:
        if self._acting_player == 1:
            return getattr(self, "_p1_episode_context", None)
        return getattr(self, "_p2_episode_context", None)

    def _card_dict_from_cpp(self, card: Any) -> dict[str, Any]:
        return {
            "cardNumber": str(getattr(card, "card_id", "") or ""),
            "cardID": str(getattr(card, "card_id", "") or ""),
            "cost": int(getattr(card, "cost", 0) or 0),
            "pitch": int(getattr(card, "pitch", 0) or 0),
            "power": int(getattr(card, "power", 0) or 0),
            "defense": int(getattr(card, "defense", 0) or 0),
        }

    def _zone_cards_from_cpp(self, cards: Any) -> list[dict[str, Any]]:
        if not isinstance(cards, list):
            return []
        return [self._card_dict_from_cpp(card) for card in cards]

    def _cpp_zone_cards(self, player_id: int, zone: str) -> list[dict[str, Any]]:
        """Read a public zone from pybind11 GameState (p1_* / p2_* accessors)."""
        gs = self._gs
        if gs is None:
            return []
        prefix = "p1" if int(player_id) == 1 else "p2"
        attr = f"{prefix}_{zone}"
        cards = getattr(gs, attr, None)
        if cards is None:
            return []
        return self._zone_cards_from_cpp(list(cards))

    def _raw_state_from_gs(self) -> dict[str, Any]:
        gs = self._gs
        if gs is None:
            return {}

        acting = int(self._acting_player)
        opp = 2 if acting == 1 else 1
        self_pitch = int(
            getattr(gs, "p1_resources" if acting == 1 else "p2_resources", 0) or 0
        )
        opp_pitch = int(
            getattr(gs, "p1_resources" if opp == 1 else "p2_resources", 0) or 0
        )

        return merge_talishar_raw_state(
            {
            "playerEquipment": self._cpp_zone_cards(acting, "equipment"),
            "opponentEquipment": self._cpp_zone_cards(opp, "equipment"),
            "playerArse": self._cpp_zone_cards(acting, "arsenal"),
            "opponentArse": self._cpp_zone_cards(opp, "arsenal"),
            "playerPitch": self._cpp_zone_cards(acting, "pitch"),
            "opponentPitch": self._cpp_zone_cards(opp, "pitch"),
            "playerDiscard": self._cpp_zone_cards(acting, "discard"),
            "opponentDiscard": self._cpp_zone_cards(opp, "discard"),
            "playerBanish": [],
            "opponentBanish": [],
            "playerAuras": [],
            "opponentAuras": [],
            "playerAllies": [],
            "opponentAllies": [],
            "playerItems": [],
            "opponentItems": [],
            "opponentPitchCount": opp_pitch,
            "playerPitchCount": self_pitch,
            "playerAP": self._action_points_for_obs(),
            "opponentAP": 0,
            "canPassPhase": True,
            "amIActivePlayer": True,
            "turnPlayer": acting,
            "firstPlayer": int(getattr(gs, "first_player", 1) or 1),
            },
            getattr(self, "_talishar_raw_state", None),
        )

    def _obs_vec_from_cpp(self, legal_count: Optional[int] = None) -> Optional[np.ndarray]:
        """Use native C++ player_observation_vector when the engine provides it."""
        gs = self._gs
        if gs is None:
            return None
        fn = getattr(gs, "player_observation_vector", None)
        if fn is None:
            return None
        if legal_count is None:
            legal_count = len(self._filter_legal_actions(self._legal_actions()))
        try:
            raw = fn(int(legal_count))
            vec = np.asarray(raw, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if vec.shape[0] != PLAYER_OBS_DIM:
            return None
        return vec

    def _absolute_health_for_obs(self, obs: dict[str, Any]) -> tuple[int, int]:
        """P1/P2 health for scalar encoding (absolute seats, not acting-player perspective)."""
        acting = int(obs.get("actingPlayerID", self._acting_player) or self._acting_player)
        if self._talishar_overlay or getattr(self, "_talishar_parity_extra", None):
            player_hp, opp_hp = self._contract_player_hp()
            if acting == 1:
                return int(player_hp), int(opp_hp)
            return int(opp_hp), int(player_hp)
        if self._gs is not None:
            return int(self._gs.p1_health), int(self._gs.p2_health)
        player_hp = int(obs.get("playerHealth", 0) or 0)
        opp_hp = int(obs.get("opponentHealth", 0) or 0)
        if acting == 1:
            return player_hp, opp_hp
        return opp_hp, player_hp

    def _obs_vec_for_state(
        self,
        obs: dict[str, Any],
        legal: list[Any],
        *,
        legal_dicts: Optional[list[dict[str, Any]]] = None,
    ) -> np.ndarray:
        winner = int(getattr(self._gs, "winner", -1) or -1) if self._gs is not None else -1
        actions = legal_dicts if legal_dicts is not None else self._legal_to_dicts(legal)
        p1_hp, p2_hp = self._absolute_health_for_obs(obs)
        raw = getattr(self, "_talishar_raw_state", None)
        raw = raw if isinstance(raw, dict) else None
        game_over = bool(raw.get("gameOver")) if raw and raw.get("gameOver") is not None else (
            p1_hp <= 0 or p2_hp <= 0 or (self._gs is not None and self._is_game_over())
        )
        vec = player_observation_vector(
            obs,
            actions,
            episode_context=self._episode_context_for_acting_player(),
            acting_player_id=int(obs.get("actingPlayerID", self._acting_player) or self._acting_player),
            p1_health=p1_hp,
            p2_health=p2_hp,
            winner=winner,
            game_over=game_over,
            consecutive_passes=int(getattr(self._gs, "consecutive_passes", 0) or 0)
            if self._gs is not None
            else 0,
            raw_talishar_state=self._raw_state_from_gs(),
        )
        if cpp_obs_alignment_enabled():
            vec = align_observation_for_cpp_training(vec)
        self._last_observation_vec = vec
        return vec


    def _normalize_hand_playability(self, raw: Any) -> dict[int, set[str]]:
        if not isinstance(raw, dict):
            return {}
        out: dict[int, set[str]] = {}
        for player_key, indices in raw.items():
            try:
                player_id = int(player_key)
            except (TypeError, ValueError):
                continue
            if player_id not in (1, 2):
                continue
            if not isinstance(indices, (list, tuple, set)):
                continue
            out[player_id] = {str(index) for index in indices}
        return out

    def _playable_hand_indices(self) -> Optional[set[str]]:
        playability = getattr(self, "_hand_playability", None)
        if playability is None:
            return None
        if self._acting_player not in playability:
            return None
        return playability[self._acting_player]

    def _sync_flow_phase_from_cpp(self) -> None:
        """Keep Talishar-like flow phase aligned when using the fast path."""
        if self._gs is None:
            return
        turn_no = int(getattr(self._gs, "turn_no", 0) or 0)
        cpp_phase = self._phase_code_from_cpp()
        if turn_no <= 0 and self._steps <= 0 and cpp_phase in {"startturn", "M"}:
            self._flow_phase = "OPENING_MAIN"
            return
        if cpp_phase == "startturn":
            self._flow_phase = "M"
        elif cpp_phase:
            self._flow_phase = cpp_phase

    def _reset_flow_state(self) -> None:
        self._flow_phase = "OPENING_MAIN"
        self._arsenal_complete = set()
        # None => report C++ turn_no; only set during scripted opening for parity.
        self._turn_no_override = None
        self._talishar_overlay = None
        self._talishar_mirror_state = None
        self._talishar_raw_state = None
        self._talishar_parity_extra = None

    def _observation_shaped_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Normalize Talishar HTTP or compact observation dicts."""
        turn_phase = state.get("turnPhase", "")
        if isinstance(turn_phase, dict):
            turn_phase = str(turn_phase.get("turnPhase", "") or "")
        else:
            turn_phase = str(turn_phase or "")

        hand = state.get("playerHand", [])
        hand_entries: list[dict[str, Any]] = []
        playable_indices: set[str] = set()
        if isinstance(hand, list):
            for index, card in enumerate(hand):
                if not isinstance(card, dict):
                    continue
                action_code = int(card.get("action", 0) or 0)
                if action_code != 0:
                    playable_indices.add(str(index))
                hand_entries.append(
                    {
                        "cardID": str(
                            card.get("cardNumber")
                            or card.get("cardID")
                            or card.get("card_id")
                            or ""
                        ),
                        "action": action_code,
                        "actionDataOverride": str(
                            card.get("actionDataOverride", str(index)) or str(index)
                        ),
                        "label": str(card.get("label", "") or ""),
                    }
                )

        legal_actions = state.get("legalActions", state.get("legal_actions", []))
        normalized_legal: list[dict[str, Any]] = []
        if isinstance(legal_actions, list):
            for index, entry in enumerate(legal_actions):
                if not isinstance(entry, dict):
                    continue
                normalized_legal.append(
                    {
                        "index": int(entry.get("index", index) or index),
                        "label": str(entry.get("label", "") or ""),
                        "zone": str(entry.get("zone", "button") or "button"),
                    }
                )

        acting_player_id = int(
            state.get("actingPlayerID", state.get("acting_player_id", self._acting_player))
            or self._acting_player
        )
        return {
            "acting_player_id": acting_player_id,
            "turn_phase": turn_phase,
            "turn_no": int(state.get("turnNo", state.get("turn_no", 0)) or 0),
            "player_health": int(state.get("playerHealth", state.get("player_health", 0)) or 0),
            "opponent_health": int(
                state.get("opponentHealth", state.get("opponent_health", 0)) or 0
            ),
            "player_hand_size": int(
                state.get("playerHandSize", state.get("player_hand_size", len(hand_entries)))
                or len(hand_entries)
            ),
            "opponent_hand_size": int(
                state.get("opponentHandSize", state.get("opponent_hand_size", 0)) or 0
            ),
            "player_deck_count": int(
                state.get("playerDeckCount", state.get("player_deck_count", 0)) or 0
            ),
            "opponent_deck_count": int(
                state.get("opponentDeckCount", state.get("opponent_deck_count", 0)) or 0
            ),
            "player_pitch_count": int(
                state.get("playerPitchCount", state.get("player_pitch_count", 0)) or 0
            ),
            "have_priority": bool(state.get("havePriority", state.get("have_priority", True))),
            "player_hand": hand_entries,
            "legal_actions": normalized_legal,
            "playable_indices": playable_indices,
        }

    def _phase_from_talishar_state(self, state: dict[str, Any]) -> str:
        turn_phase = state.get("turnPhase", {})
        if isinstance(turn_phase, dict):
            return str(turn_phase.get("turnPhase", "") or "")
        return str(turn_phase or "")

    def apply_talishar_state(
        self,
        state: dict[str, Any],
        *,
        acting_player_id: Optional[int] = None,
    ) -> None:
        """Mirror Talishar observation fields for the acting player's perspective."""
        shaped = self._observation_shaped_state(state)
        if acting_player_id is not None:
            shaped["acting_player_id"] = int(acting_player_id)

        self._acting_player = int(shaped["acting_player_id"])
        if hasattr(self._gs, "set_priority"):
            self._gs.set_priority(self._acting_player - 1)

        self._hand_playability = {self._acting_player: shaped["playable_indices"]}
        self._flow_phase = str(shaped["turn_phase"] or self._flow_phase)
        self._turn_no_override = int(shaped["turn_no"])
        self._talishar_raw_state = dict(state)
        self._talishar_overlay = {
            "acting_player_id": self._acting_player,
            "turn_phase": self._flow_phase,
            "turn_no": self._turn_no_override,
            "player_health": shaped["player_health"],
            "opponent_health": shaped["opponent_health"],
            "player_hand_size": shaped["player_hand_size"],
            "opponent_hand_size": shaped["opponent_hand_size"],
            "player_deck_count": shaped["player_deck_count"],
            "opponent_deck_count": shaped["opponent_deck_count"],
            "player_pitch_count": shaped["player_pitch_count"],
            "have_priority": shaped["have_priority"],
            "player_hand": shaped["player_hand"],
            "legal_actions": shaped["legal_actions"],
        }

    def _cpp_observation_acting_player(self) -> int:
        """1-indexed player whose observation/legal actions Talishar exposes."""
        gs = self._gs
        if gs is None:
            return int(self._acting_player)
        fn = getattr(gs, "observation_acting_player", None)
        if callable(fn):
            return int(fn())
        return int(gs.priority) + 1

    def clear_talishar_state(self) -> None:
        self._talishar_overlay = None
        self._talishar_raw_state = None
        self._talishar_parity_extra = None

    def _snapshot_to_absolute_dict(self, snap: Any) -> dict[str, Any]:
        phase_map = {0: "START", 1: "M", 2: "P", 3: "A", 4: "B", 5: "D", 6: "END", 7: "OVER"}
        phase_code = phase_map.get(int(getattr(snap, "phase", 0)), "M")
        if self._gs is not None and bool(getattr(self._gs, "instant_window", False)):
            phase_code = "INSTANT"
        from .game_state_parity import _talishar_action_points

        p1_pool = int(getattr(snap, "p1_resources", 0) or 0)
        p2_pool = int(getattr(snap, "p2_resources", 0) or 0)
        p1_ap = int(getattr(snap, "p1_action_points", 0) or 0)
        p2_ap = int(getattr(snap, "p2_action_points", 0) or 0)
        combat_chain = []
        for link in list(getattr(snap, "combat_chain", []) or []):
            combat_chain.append(
                {
                    "card_id": str(getattr(link, "card_id", "")),
                    "power": int(getattr(link, "power", 0) or 0),
                    "defense": int(getattr(link, "defense", 0) or 0),
                }
            )
        return {
            "acting_player_id": int(getattr(snap, "acting_player_id", 1) or 1),
            "p1_health": int(getattr(snap, "p1_health", 0) or 0),
            "p2_health": int(getattr(snap, "p2_health", 0) or 0),
            "turn_no": int(getattr(snap, "turn_no", 0) or 0),
            "phase": phase_code,
            "p1_hand_size": len(list(getattr(snap, "p1_hand", []) or [])),
            "p2_hand_size": len(list(getattr(snap, "p2_hand", []) or [])),
            "p1_deck_count": len(list(getattr(snap, "p1_deck", []) or [])),
            "p2_deck_count": len(list(getattr(snap, "p2_deck", []) or [])),
            "p1_pitch_count": p1_pool,
            "p2_pitch_count": p2_pool,
            "p1_resources": _talishar_action_points(phase_code, p1_ap),
            "p2_resources": _talishar_action_points(phase_code, p2_ap),
            "priority_player": int(self._gs.priority) + 1 if self._gs is not None else 1,
            "p1_hand": list(getattr(snap, "p1_hand", []) or []),
            "p2_hand": list(getattr(snap, "p2_hand", []) or []),
            "p1_deck": list(getattr(snap, "p1_deck", []) or []),
            "p2_deck": list(getattr(snap, "p2_deck", []) or []),
            "p1_discard": list(getattr(snap, "p1_discard", []) or []),
            "p2_discard": list(getattr(snap, "p2_discard", []) or []),
            "p1_equipment": list(getattr(snap, "p1_equipment", []) or []),
            "p2_equipment": list(getattr(snap, "p2_equipment", []) or []),
            "p1_arsenal": list(getattr(snap, "p1_arsenal", []) or []),
            "p2_arsenal": list(getattr(snap, "p2_arsenal", []) or []),
            "p1_pitch": list(getattr(snap, "p1_pitch", []) or []),
            "p2_pitch": list(getattr(snap, "p2_pitch", []) or []),
            "p1_banish": list(getattr(snap, "p1_banish", []) or []),
            "p2_banish": list(getattr(snap, "p2_banish", []) or []),
            "combat_chain": combat_chain,
            "pending_attack_power": int(getattr(snap, "pending_attack_power", 0) or 0),
            "pending_block_value": int(getattr(snap, "pending_block_value", 0) or 0),
            "game_over": bool(getattr(snap, "game_over", False)),
            "winner": int(getattr(snap, "winner", -1) or -1),
        }

    def export_game_state(self, *, absolute: bool = True) -> dict[str, Any]:
        """Export C++ GameState as an absolute P1/P2 snapshot (no Talishar overlay)."""
        self.clear_talishar_state()
        gs = self._gs
        if gs is None:
            return {}
        snapshot_fn = getattr(gs, "snapshot_state", None)
        if callable(snapshot_fn):
            snap_dict = self._snapshot_to_absolute_dict(snapshot_fn())
            phase = str(snap_dict.get("phase", "") or "").upper()
            if bool(getattr(self._gs, "instant_window", False)) and phase not in {
                "B",
                "BLOCK",
                "A",
                "ATTACK",
                "D",
                "DAMAGE",
            }:
                snap_dict["phase"] = "INSTANT"
            return snap_dict
        from .game_state_parity import _cpp_raw_to_absolute

        return _cpp_raw_to_absolute(self, self._raw_state_from_gs())

    def apply_initial_sync_from_talishar(self, payload: dict[str, Any]) -> None:
        """One-time init sync from Talishar baseline (hands, decks, equipment)."""
        gs = self._gs
        if gs is None:
            return
        opening_hands = payload.get("opening_hands") or {}
        if isinstance(opening_hands, dict):
            self._apply_opening_hands(gs, opening_hands)
        deck_orders = payload.get("deck_orders") or {}
        if isinstance(deck_orders, dict) and hasattr(gs, "sync_deck_order"):
            for player_key, card_ids in deck_orders.items():
                if not isinstance(card_ids, list):
                    continue
                player_idx = int(player_key) - 1
                if player_idx in (0, 1):
                    gs.sync_deck_order(player_idx, [str(cid) for cid in card_ids])
        equipment = payload.get("equipment") or {}
        if isinstance(equipment, dict) and hasattr(gs, "sync_equipment"):
            for player_key, card_ids in equipment.items():
                if not isinstance(card_ids, list):
                    continue
                player_idx = int(player_key) - 1
                if player_idx in (0, 1):
                    gs.sync_equipment(player_idx, [str(cid) for cid in card_ids])
        resources = payload.get("resources") or {}
        if isinstance(resources, dict) and hasattr(gs, "set_player_resources"):
            for player_key, pool in resources.items():
                try:
                    player_idx = int(player_key) - 1
                except (TypeError, ValueError):
                    continue
                if player_idx in (0, 1):
                    gs.set_player_resources(player_idx, int(pool))
        action_points = payload.get("action_points") or {}
        if isinstance(action_points, dict) and hasattr(gs, "set_player_action_points"):
            for player_key, ap in action_points.items():
                try:
                    player_idx = int(player_key) - 1
                except (TypeError, ValueError):
                    continue
                if player_idx in (0, 1):
                    gs.set_player_action_points(player_idx, int(ap))
        acting = payload.get("acting_player_id")
        if acting is not None and hasattr(gs, "set_priority"):
            player_id = int(acting)
            if player_id in (1, 2):
                gs.set_priority(player_id - 1)
                self._acting_player = player_id
        seed = payload.get("rng_seed")
        if seed is not None and hasattr(gs, "seed_rng"):
            gs.seed_rng(int(seed) & 0xFFFFFFFF)

    def set_talishar_mirror_state(self, state: Optional[dict[str, Any]]) -> None:
        """Queue a Talishar parity snapshot for the next step result."""
        if self._strict_simulation and state is not None:
            raise RuntimeError(
                "set_talishar_mirror_state is disabled in strict simulation parity mode"
            )
        self._talishar_mirror_state = state

    def apply_talishar_mirror_payload(self, payload: Optional[dict[str, Any]]) -> None:
        """Apply Talishar observation plus optional step metadata for parity."""
        if not payload:
            self._talishar_parity_extra = None
            return

        state: dict[str, Any]
        extra: dict[str, Any]
        if "state" in payload and isinstance(payload.get("state"), dict):
            state = payload["state"]
            extra = {key: value for key, value in payload.items() if key != "state"}
        elif any(key in payload for key in ("playerHand", "actingPlayerID")):
            state = payload
            extra = {}
        else:
            state = payload.get("observation", {}) if isinstance(payload.get("observation"), dict) else {}
            extra = {
                key: value
                for key, value in payload.items()
                if key not in {"observation", "state"}
            }

        if state:
            self.apply_talishar_state(state)

        raw_state = payload.get("raw_state")
        if isinstance(raw_state, dict) and raw_state:
            self._talishar_raw_state = dict(raw_state)

        info_legal = extra.get("legal_actions")
        if isinstance(info_legal, list) and self._talishar_overlay is not None:
            self._talishar_overlay["info_legal_actions"] = [
                self._normalise_info_legal_action(action)
                for action in info_legal
                if isinstance(action, dict)
            ]

        self._talishar_parity_extra = extra or None

    def _consume_talishar_mirror_state(self) -> None:
        if self._talishar_mirror_state is None:
            return
        payload = self._talishar_mirror_state
        self._talishar_mirror_state = None
        self.apply_talishar_mirror_payload(payload)

    def _normalise_info_legal_action(self, action: dict[str, Any]) -> dict[str, Any]:
        return {
            "action_code": int(action.get("action_code", 0) or 0),
            "button_input": str(action.get("button_input", "") or ""),
            "card_id": str(action.get("card_id", "") or ""),
            "zone": str(action.get("zone", "") or ""),
            "label": str(action.get("label", "") or ""),
        }

    def _contract_legal_actions(self, legal: list[Any]) -> list[dict[str, Any]]:
        extra = self._talishar_parity_extra or {}
        info_legal = extra.get("legal_actions")
        if isinstance(info_legal, list) and info_legal:
            return [
                self._normalise_info_legal_action(action)
                for action in info_legal
                if isinstance(action, dict)
            ]
        if self._talishar_overlay:
            overlay_legal = self._talishar_overlay.get("info_legal_actions")
            if isinstance(overlay_legal, list) and overlay_legal:
                return list(overlay_legal)
        return self._legal_to_dicts(legal)

    def _contract_player_hp(self) -> tuple[int, int]:
        extra = self._talishar_parity_extra or {}
        if extra.get("player_hp") is not None and extra.get("opponent_hp") is not None:
            return int(extra["player_hp"]), int(extra["opponent_hp"])
        if self._talishar_overlay:
            return (
                int(self._talishar_overlay.get("player_health", 0) or 0),
                int(self._talishar_overlay.get("opponent_health", 0) or 0),
            )
        if self._acting_player == 1:
            return int(self._gs.p1_health), int(self._gs.p2_health)
        return int(self._gs.p2_health), int(self._gs.p1_health)

    def _contract_turn(self) -> int:
        extra = self._talishar_parity_extra or {}
        if extra.get("turn") is not None:
            return int(extra["turn"])
        return self._obs_turn_no()

    def _contract_acting_player_id(self) -> int:
        extra = self._talishar_parity_extra or {}
        acting_id = extra.get("acting_player_id")
        if acting_id is not None:
            return int(acting_id)
        if self._talishar_overlay:
            overlay_id = self._talishar_overlay.get("acting_player_id")
            if overlay_id is not None:
                return int(overlay_id)
        return int(self._acting_player)

    def _contract_repeat_fields(self, repeat_penalty: float) -> tuple[int, float]:
        extra = self._talishar_parity_extra or {}
        repeat_streak = extra.get("repeat_streak")
        if repeat_streak is None:
            repeat_streak = self._repeat_tracker.repeat_streak
        mirrored_penalty = extra.get("repeat_penalty")
        if mirrored_penalty is not None:
            repeat_penalty = float(mirrored_penalty)
        return int(repeat_streak), float(repeat_penalty)

    def _mirrored_reward(self, default_reward: float) -> float:
        extra = self._talishar_parity_extra or {}
        if extra.get("reward") is not None:
            return float(extra["reward"])
        return float(default_reward)

    def _mirrored_termination(self, *, terminated: bool, truncated: bool) -> tuple[bool, bool]:
        extra = self._talishar_parity_extra or {}
        if extra.get("terminated") is not None:
            terminated = bool(extra["terminated"])
        if extra.get("truncated") is not None:
            truncated = bool(extra["truncated"])
        return terminated, truncated

    def _make_pass_action(self) -> Any:
        return type(
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

    def _make_hand_action(
        self,
        *,
        action_code: int,
        button_input: str,
        card_id: str,
        label: str,
    ) -> Any:
        return type(
            "_HandAction",
            (),
            {
                "action_code": int(action_code),
                "button_input": str(button_input),
                "card_id": str(card_id),
                "zone": "hand",
                "label": str(label or card_id),
            },
        )()

    def _effective_turn_phase(self) -> str:
        if self._talishar_overlay:
            return str(self._talishar_overlay.get("turn_phase", "") or "M")
        if getattr(self, "_strict_simulation", False):
            return self._phase_code_from_cpp()
        if self._flow_phase:
            return self._flow_phase
        return self._phase_code_from_cpp()

    def _phase_code_from_cpp(self) -> str:
        """Return a Talishar-like phase token from the C++ engine phase value."""
        gs = self._gs
        if gs is not None and bool(getattr(gs, "instant_window", False)):
            return "INSTANT"
        phase = getattr(self._gs, "phase", None)
        phase_value: Optional[int]
        if phase is None:
            phase_value = None
        else:
            try:
                phase_value = int(phase)
            except (TypeError, ValueError):
                phase_name = str(phase).split(".")[-1].upper()
                phase_value = {
                    "START": 0,
                    "MAIN": 1,
                    "PITCH": 2,
                    "ATTACK": 3,
                    "BLOCK": 4,
                    "DAMAGE": 5,
                    "END": 6,
                    "OVER": 7,
                }.get(phase_name)

        mapping = {
            0: "startturn",
            1: "M",
            2: "P",
            3: "A",
            4: "B",
            5: "damage",
            6: "endphase",
            7: "OVER",
        }
        return mapping.get(phase_value, "M")

    def _apply_opening_hands(self, gs: Any, opening_hands: Any) -> None:
        """Align dealt hands with Talishar when *opening_hands* is provided."""
        if not opening_hands or not hasattr(gs, "sync_opening_hand"):
            return
        if not isinstance(opening_hands, dict):
            return
        for player_key, card_ids in opening_hands.items():
            try:
                player_id = int(player_key)
            except (TypeError, ValueError):
                continue
            if player_id not in (1, 2):
                continue
            if not isinstance(card_ids, list):
                continue
            ids = [str(card_id) for card_id in card_ids if is_syncable_card_id(card_id)]
            if not ids:
                continue
            gs.sync_opening_hand(player_id - 1, ids)

    def _synthetic_flow_legal_actions(self) -> Optional[list[Any]]:
        """Return Talishar-shaped legal actions for scripted flow phases."""
        phase = self._effective_turn_phase().upper()
        if phase == "OPENING_MAIN":
            return [self._make_pass_action()]
        if phase == "ARS":
            actions = [
                self._make_hand_action(
                    action_code=4,
                    button_input=str(getattr(card, "card_id", "") or ""),
                    card_id=str(getattr(card, "card_id", "") or ""),
                    label=str(getattr(card, "name", "") or getattr(card, "card_id", "")),
                )
                for card in self._hand_cards()
            ]
            actions.append(self._make_pass_action())
            return actions
        return None

    def _is_attack_card(self, card: Any) -> bool:
        card_type = str(getattr(card, "card_type", "") or "").strip().lower()
        if "attack" in card_type:
            return True
        try:
            return int(getattr(card, "power", 0) or 0) > 0
        except (TypeError, ValueError):
            return False

    def _advance_ars_phase(self) -> None:
        """Complete the current player's arsenal step and advance opening flow."""
        self._arsenal_complete.add(self._acting_player)
        if self._acting_player == 1:
            self._acting_player = 2
            self._turn_no_override = 1
            self._flow_phase = "ARS"
            if hasattr(self._gs, "set_priority"):
                self._gs.set_priority(1)
        else:
            self._flow_phase = "M"
            self._acting_player = 1
            self._turn_no_override = None
            if hasattr(self._gs, "set_priority"):
                self._gs.set_priority(0)

    def _handle_flow_pass(self) -> bool:
        """Apply Talishar-style pass transitions without advancing the C++ stub."""
        phase = self._effective_turn_phase().upper()
        if phase == "OPENING_MAIN":
            self._flow_phase = "ARS"
            self.clear_talishar_state()
            return True
        if phase == "ARS":
            self._advance_ars_phase()
            self.clear_talishar_state()
            return True
        if phase == "B":
            self._flow_phase = "A"
            opponent = 2 if self._acting_player == 1 else 1
            self._acting_player = opponent
            if hasattr(self._gs, "set_priority"):
                self._gs.set_priority(opponent - 1)
            self.clear_talishar_state()
            return True
        return False

    def _handle_flow_hand_action(self, action: Any) -> bool:
        """Handle non-standard hand actions (e.g. arsenal) without C++ apply."""
        code = int(getattr(action, "action_code", 0) or 0)
        if code == 4 and self._effective_turn_phase().upper() == "ARS":
            # Arsenal completes the step like pass; previously this was a no-op
            # and trapped episodes when policies picked action index 0.
            self._advance_ars_phase()
            self.clear_talishar_state()
            return True
        return False

    def _maybe_enter_block_phase(self, played_card: Any) -> None:
        if not self._is_attack_card(played_card):
            return
        self._flow_phase = "B"
        defender = 2 if self._acting_player == 1 else 1
        self._acting_player = defender
        if hasattr(self._gs, "set_priority"):
            self._gs.set_priority(defender - 1)
        self.clear_talishar_state()

    def _apply_gs_engine_settings(self, gs: Any) -> None:
        if hasattr(gs, "max_consecutive_passes"):
            gs.max_consecutive_passes = int(self._max_consecutive_passes)

    def _new_gamestate(
        self,
        options: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Any:
        gs = self._fab.GameState()
        self._apply_gs_engine_settings(gs)
        opts = options or {}
        effective_seed = seed if seed is not None else opts.get("rng_seed")
        if effective_seed is not None and hasattr(gs, "seed_rng"):
            gs.seed_rng(int(effective_seed) & 0xFFFFFFFF)
        gs.register_all_cards()
        if hasattr(gs, "init_standard_decks"):
            gs.init_standard_decks()
        self._apply_opening_hands(gs, opts.get("opening_hands"))
        acting_player = opts.get("acting_player_id")
        if acting_player is not None and hasattr(gs, "set_priority"):
            player_id = int(acting_player)
            if player_id in (1, 2):
                gs.set_priority(player_id - 1)
        return gs

    def _legal_actions(self) -> list[Any]:
        """Return legal actions, preferring Talishar-shaped flow actions when scripted."""
        if self._strict_simulation:
            return self._gs.get_legal_actions()
        flow_legal = self._synthetic_flow_legal_actions()
        if flow_legal is not None:
            return flow_legal
        return self._gs.get_legal_actions()

    @property
    def supports_fast_training(self) -> bool:
        return not self.fast_training_unavailable_reasons()

    def fast_training_unavailable_reasons(self) -> list[str]:
        """Return why the numeric fast path cannot be used (empty if available)."""
        gs = self._gs
        if gs is None:
            try:
                gs = self._fab.GameState()
            except Exception as exc:
                return [f"cannot load GameState: {exc}"]
        reasons: list[str] = []
        if not hasattr(gs, "fast_step_index"):
            reasons.append(
                "engine module missing fast_step_index (rebuild with "
                "scripts/cpp/build_cpp_engine_for_matchup.py)"
            )
        return reasons

    def _obs_vec_for_fast_path(self) -> np.ndarray:
        legal = self._filter_legal_actions(self._legal_actions())
        obs_json = self._encode_observation(legal)
        vec = self._last_observation_vec
        if vec is None:
            vec = observation_fingerprint(obs_json)
        self._last_observation_vec = np.asarray(vec, dtype=np.float64)
        return self._last_observation_vec

    def fast_action_capacity(self) -> int:
        return ACTION_CAPACITY

    def logic_policy_action_index(
        self,
        *,
        max_pitch_value: int = 3,
        min_resource_cost: int = 0,
    ) -> int:
        """Pick a legal action index using the shared Talishar default/heuristic policy."""
        assert self._gs is not None, "call fast_reset() first"
        legal_objects = self._filter_legal_actions(self._legal_actions())
        legal_dicts = self._legal_to_dicts(legal_objects)
        if not legal_dicts:
            return 0
        talishar_state = self._synthetic_talishar_state()
        idx = choose_talishar_action_index(
            legal_dicts,
            talishar_state,
            max_pitch_value=max_pitch_value,
            min_resource_cost=min_resource_cost,
        )
        return min(max(0, int(idx)), len(legal_dicts) - 1)

    def fast_reset(
        self,
        seed: Optional[int] = None,
        *,
        starting_player_id: int = 1,
    ) -> dict[str, Any]:
        """Reset and return a compact numeric state for high-throughput training."""
        self._reset_flow_state()
        self._hand_playability = {}
        starting_player_id = 2 if int(starting_player_id) == 2 else 1
        self._gs = self._new_gamestate(
            {"acting_player_id": starting_player_id},
            seed=seed,
        )
        self._steps = 0
        self._p1_hp = int(self._gs.p1_health)
        self._p2_hp = int(self._gs.p2_health)
        self._acting_player = self._cpp_observation_acting_player()
        self._repeat_tracker.reset(
            turn_no=int(getattr(self._gs, "turn_no", 0) or 0),
            acting_player_id=int(self._acting_player),
        )
        self._loop_guard.reset()
        self._macro_stall_guard.reset()
        legal_count = len(self._filter_legal_actions(self._legal_actions()))
        obs_vec = self._obs_vec_for_fast_path()
        self._last_observation_vec = obs_vec
        legal = self._filter_legal_actions(self._legal_actions())
        if self._enable_combat_tracker:
            self._synthetic_combat_log = [
                f"matchup {self._deck1 or 'P1'} vs {self._deck2 or 'P2'}"
            ]
            self._combat_tracker.reset(
                initial_snapshot=self._tracker_state_snapshot(legal),
                initial_legal_actions=self._legal_to_dicts(legal),
                combat_log_lines=self._synthetic_combat_log,
                metadata={
                    "engine": "cpp",
                    "engine_dir": str(self._engine_dir),
                    "deck1": self._deck1,
                    "deck2": self._deck2,
                    "fast_path": True,
                },
            )
        else:
            self._combat_tracker.clear()
            self._synthetic_combat_log = []
        self._sync_flow_phase_from_cpp()
        return {
            "obs_vec": obs_vec,
            "legal_count": legal_count,
            "acting_player_id": self._acting_player,
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "winner": -1,
            "p1_health": self._p1_hp,
            "p2_health": self._p2_hp,
            "p1_deck": int(getattr(self._gs, "p1_deck_size", 0) or 0),
            "p2_deck": int(getattr(self._gs, "p2_deck_size", 0) or 0),
            "turn_no": int(getattr(self._gs, "turn_no", 0) or 0),
        }

    def _cpp_fast_index_for_chosen(self, chosen: Any) -> int:
        """Map a filtered legal action onto the C++ ``fast_step_index`` cursor."""
        cpp_action = self._resolve_cpp_action(chosen)
        raw_legal = list(self._gs.get_legal_actions())
        if cpp_action is not None:
            chosen_dict = self._action_to_dict(cpp_action)
            for index, candidate in enumerate(raw_legal):
                if candidate is cpp_action:
                    return index
                if self._action_to_dict(candidate) == chosen_dict:
                    return index
        if self._is_pass_like(chosen):
            for index, candidate in enumerate(raw_legal):
                if self._is_pass_like(candidate):
                    return index
        return 0

    def _fast_step_scripted_flow(
        self,
        chosen: Any,
        *,
        prev_p1: int,
        prev_p2: int,
        before_snapshot: Optional[dict[str, Any]],
        legal_before: list[dict[str, Any]],
        chosen_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply opening/block flow actions without the C++ fast-step cursor."""
        flow_handled = False
        if self._is_pass_like(chosen):
            flow_handled = self._handle_flow_pass()
        elif self._handle_flow_hand_action(chosen):
            flow_handled = True
        if not flow_handled:
            flow_handled = self._handle_flow_pass()

        self._steps += 1
        terminated = self._is_game_over()
        truncated = not terminated and self._steps >= self._max_turns
        macro_stall = self._check_macro_stall(self._legal_actions())
        if macro_stall.should_truncate and not terminated:
            truncated = True
        repeat_pen = self._repeat_penalty(
            int(chosen_dict.get("action_code", 0) or 0),
            str(chosen_dict.get("button_input", "") or ""),
        )
        reward = float(
            self._compute_reward(prev_p1, prev_p2, terminated, truncated, repeat_pen)
        )
        if truncated and not terminated:
            reward = float(self._truncation_penalty)
        self._p1_hp = int(self._gs.p1_health)
        self._p2_hp = int(self._gs.p2_health)
        obs_vec = self._obs_vec_for_fast_path()
        self._last_observation_vec = obs_vec
        if self._enable_combat_tracker and before_snapshot is not None:
            new_legal = self._filter_legal_actions(self._legal_actions())
            after_snapshot = self._tracker_state_snapshot(new_legal)
            contract_legal = self._contract_legal_actions(new_legal)
            self._append_synthetic_log(
                before_snapshot,
                chosen_dict,
                after_snapshot,
                prev_p1,
                prev_p2,
                terminated,
                truncated,
            )
            self._combat_tracker.record_step(
                before_snapshot=before_snapshot,
                after_snapshot=after_snapshot,
                action=chosen_dict,
                legal_before=legal_before,
                legal_after=contract_legal,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                combat_log_lines=self._synthetic_combat_log,
            )
        return {
            "obs_vec": obs_vec,
            "legal_count": len(self._filter_legal_actions(self._legal_actions())),
            "acting_player_id": self._acting_player,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "winner": int(getattr(self._gs, "winner", -1) or -1),
            "p1_health": int(self._p1_hp),
            "p2_health": int(self._p2_hp),
            "p1_deck": int(getattr(self._gs, "p1_deck_size", 0) or 0),
            "p2_deck": int(getattr(self._gs, "p2_deck_size", 0) or 0),
            "turn_no": self._obs_turn_no(),
            **self._macro_stall_info(macro_stall),
        }

    def fast_step_index(self, action_index: int) -> dict[str, Any]:
        """Step by compact legal-action index without building JSON or pybind actions."""
        assert self._gs is not None, "call fast_reset() first"
        legal = self._filter_legal_actions(self._legal_actions())
        loop_guard = self._loop_guard_for_step(legal)
        before_snapshot = (
            self._tracker_state_snapshot(legal) if self._enable_combat_tracker else None
        )
        legal_before = self._legal_to_dicts(legal) if self._enable_combat_tracker else []
        prev_p1 = int(self._p1_hp)
        prev_p2 = int(self._p2_hp)
        if legal and not loop_guard.force_pass:
            idx = min(max(0, int(action_index)), len(legal) - 1)
            chosen = legal[idx]
        else:
            chosen = self._chosen_from_loop_guard(legal, loop_guard)
        chosen_dict = self._action_to_dict(chosen)

        if self._synthetic_flow_legal_actions() is not None:
            return self._fast_step_scripted_flow(
                chosen,
                prev_p1=prev_p1,
                prev_p2=prev_p2,
                before_snapshot=before_snapshot,
                legal_before=legal_before,
                chosen_dict=chosen_dict,
            )

        cpp_idx = self._cpp_fast_index_for_chosen(chosen)
        result = self._gs.fast_step_index(int(cpp_idx))
        self._steps += 1
        terminated = bool(result.terminated)
        truncated = not terminated and self._steps >= self._max_turns
        macro_stall = self._check_macro_stall(self._legal_actions())
        if macro_stall.should_truncate and not terminated:
            truncated = True
        if terminated:
            reward = float(result.reward)
        elif truncated:
            reward = float(self._truncation_penalty)
        else:
            p1_now = int(result.p1_health)
            p2_now = int(result.p2_health)
            dmg_dealt = max(0, prev_p2 - p2_now)
            dmg_taken = max(0, prev_p1 - p1_now)
            scale = self._damage_reward_scale
            reward = dmg_dealt * scale - dmg_taken * scale + self._step_penalty
        self._acting_player = int(result.acting_player_id)
        self._p1_hp = int(result.p1_health)
        self._p2_hp = int(result.p2_health)
        obs_vec = self._obs_vec_for_fast_path()
        self._last_observation_vec = obs_vec
        if self._enable_combat_tracker and before_snapshot is not None:
            new_legal = self._filter_legal_actions(self._legal_actions())
            after_snapshot = self._tracker_state_snapshot(new_legal)
            contract_legal = self._contract_legal_actions(new_legal)
            self._append_synthetic_log(
                before_snapshot,
                chosen_dict,
                after_snapshot,
                prev_p1,
                prev_p2,
                terminated,
                truncated,
            )
            self._combat_tracker.record_step(
                before_snapshot=before_snapshot,
                after_snapshot=after_snapshot,
                action=chosen_dict,
                legal_before=legal_before,
                legal_after=contract_legal,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                combat_log_lines=self._synthetic_combat_log,
            )
        self._sync_flow_phase_from_cpp()
        fast_result: dict[str, Any] = {
            "obs_vec": obs_vec,
            "legal_count": len(self._filter_legal_actions(self._legal_actions())),
            "acting_player_id": self._acting_player,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "winner": int(result.winner),
            "p1_health": self._p1_hp,
            "p2_health": self._p2_hp,
            "p1_deck": int(getattr(self._gs, "p1_deck_size", 0) or 0),
            "p2_deck": int(getattr(self._gs, "p2_deck_size", 0) or 0),
            "turn_no": int(result.turn_no),
            "loop_guard_forced_pass": loop_guard.force_pass,
            "loop_guard_reason": loop_guard.reason,
            "turn_steps": loop_guard.turn_steps,
            "decision_loop_streak": loop_guard.loop_streak,
        }
        if loop_guard.forced_action is not None:
            fast_result["loop_guard_forced_action"] = dict(loop_guard.forced_action)
        fast_result.update(self._macro_stall_info(macro_stall))
        return fast_result

    def _is_pass_like(self, action: Any) -> bool:
        code = int(getattr(action, "action_code", 0) or 0)
        if code in (99, 101, 105):
            return True
        label = str(getattr(action, "label", "") or "").strip().lower()
        return any(tok in label for tok in ("pass", "end turn", "no block", "skip"))

    def _hand_zone_cards_for_player(self, player_id: int) -> list[dict[str, Any]]:
        prefix = "p1" if int(player_id) == 1 else "p2"
        cards = getattr(self._gs, f"{prefix}_hand", None) if self._gs is not None else None
        if not isinstance(cards, list):
            return []
        return self._zone_cards_from_cpp(cards)

    def _board_state_for_loop_guard(self) -> dict[str, Any]:
        """Talishar-shaped snapshot with enough zones for board-revert detection."""
        state: dict[str, Any] = dict(self._raw_state_from_gs())
        gs = self._gs
        if gs is None:
            return state

        acting = int(self._acting_player)
        opp = 2 if acting == 1 else 1
        state["turnNo"] = self._obs_turn_no()
        state["turnPhase"] = {"turnPhase": self._phase_code()}
        state["playerHealth"] = int(gs.p1_health if acting == 1 else gs.p2_health)
        state["opponentHealth"] = int(gs.p2_health if acting == 1 else gs.p1_health)
        state["playerDeckCount"] = int(
            getattr(gs, f"p{acting}_deck_size", 0) or 0
        )
        state["opponentDeckCount"] = int(getattr(gs, f"p{opp}_deck_size", 0) or 0)
        state["opponentPitchCount"] = int(
            getattr(gs, f"p{opp}_pitch_size", 0) or 0
        )
        state["playerHandSize"] = int(getattr(gs, f"p{acting}_hand_size", 0) or 0)
        state["opponentHandSize"] = int(getattr(gs, f"p{opp}_hand_size", 0) or 0)
        state["pendingAttackPower"] = int(getattr(gs, "pending_attack_power", 0) or 0)
        state["pendingBlockValue"] = int(getattr(gs, "pending_block_value", 0) or 0)
        state["playerHand"] = self._hand_zone_cards_for_player(acting)
        state["opponentHand"] = self._hand_zone_cards_for_player(opp)
        return state

    def _hand_entries_for_filter_state(self) -> list[dict[str, Any]]:
        """Talishar-shaped playerHand entries for shared legal-action filtering."""
        hand_entries: list[dict[str, Any]] = []
        for index, card in enumerate(self._hand_cards()):
            entry: dict[str, Any] = {
                "action": 27,
                "actionDataOverride": str(index),
                "cardNumber": str(getattr(card, "card_id", "") or ""),
            }
            for field, attr in (
                ("defense", "defense"),
                ("cost", "cost"),
                ("resource", "cost"),
                ("pitch", "pitch"),
            ):
                value = getattr(card, attr, None)
                if value is None:
                    continue
                try:
                    if int(value) == 0:
                        continue
                except (TypeError, ValueError):
                    pass
                entry[field] = value
            label = str(getattr(card, "name", "") or "").strip()
            if label:
                entry["label"] = label
            hand_entries.append(entry)
        return hand_entries

    def _equipment_entries_for_filter_state(self) -> list[dict[str, Any]]:
        acting = int(self._acting_player)
        entries: list[dict[str, Any]] = []
        for index, card in enumerate(self._cpp_zone_cards(acting, "equipment")):
            entry = dict(card)
            entry.setdefault("action", 3)
            entry.setdefault("actionDataOverride", str(index))
            entries.append(entry)
        return entries

    def _arsenal_entries_for_filter_state(self) -> list[dict[str, Any]]:
        acting = int(self._acting_player)
        entries: list[dict[str, Any]] = []
        for index, card in enumerate(self._cpp_zone_cards(acting, "arsenal")):
            entry = dict(card)
            entry.setdefault("action", 5)
            entry.setdefault("actionDataOverride", str(index))
            entries.append(entry)
        return entries

    def _filter_state(self) -> dict[str, Any]:
        """Build the Talishar-shaped state dict consumed by ``filter_legal_actions``."""
        raw_state = getattr(self, "_talishar_raw_state", None)
        if isinstance(raw_state, dict) and raw_state:
            state = dict(raw_state)
            phase = self._phase_code()
            if "turnPhase" not in state:
                state["turnPhase"] = {"turnPhase": phase}
            if "turnNo" not in state:
                state["turnNo"] = self._obs_turn_no()
            return state

        state: dict[str, Any] = dict(self._raw_state_from_gs())
        state.update(self._board_state_for_loop_guard())
        phase = self._phase_code()
        state["turnPhase"] = {"turnPhase": phase}
        state["turnNo"] = self._obs_turn_no()
        state["playerHand"] = self._hand_entries_for_filter_state()
        state["playerEquipment"] = self._equipment_entries_for_filter_state()
        state["playerArse"] = self._arsenal_entries_for_filter_state()
        state["playerPitchCount"] = self._pitch_count()
        state["havePriority"] = not self._is_game_over()
        gs = self._gs
        if gs is not None and bool(getattr(gs, "instant_window", False)):
            state["turnPhase"] = {"turnPhase": "INSTANT"}
        if phase.upper() == "P":
            state["canPassPhase"] = False
        return state

    def _augment_legal_before_filter(self, legal: list[Any]) -> list[Any]:
        """Add Talishar-shaped actions the C++ stub omits but the shared filter expects."""
        phase = self._effective_turn_phase().upper()
        if phase != "P":
            return legal
        has_cancel = any(
            int(getattr(action, "action_code", 0) or 0) == 10000 for action in legal
        )
        if has_cancel:
            return legal
        return list(legal) + [
            type(
                "_Cancel",
                (),
                {
                    "action_code": 10000,
                    "button_input": "",
                    "card_id": "",
                    "zone": "button",
                    "label": "Cancel",
                },
            )()
        ]

    def _synthetic_talishar_state(self) -> dict[str, Any]:
        """Build a Talishar-shaped state dict for shared affordability helpers."""
        return {
            "turnPhase": {"turnPhase": self._phase_code()},
            "playerPitchCount": self._pitch_count(),
            "playerHand": self._hand_entries_for_filter_state(),
            "pendingAttackPower": int(getattr(self._gs, "pending_attack_power", 0) or 0)
            if self._gs is not None
            else 0,
        }

    def _ensure_playable_hand_actions(
        self,
        legal: list[Any],
        playable: set[str],
    ) -> list[Any]:
        """Add hand-play actions for Talishar-playable indices missing from C++ legal."""
        existing = {
            str(getattr(action, "button_input", "") or "")
            for action in legal
            if int(getattr(action, "action_code", 0) or 0) == 27
            and str(getattr(action, "zone", "") or "").strip().lower() == "hand"
        }
        hand_cards = self._hand_cards()
        additions: list[Any] = []
        for index in sorted(playable, key=lambda value: int(value) if value.isdigit() else 0):
            if index in existing:
                continue
            try:
                card_index = int(index)
            except ValueError:
                continue
            if card_index < 0 or card_index >= len(hand_cards):
                continue
            card = hand_cards[card_index]
            card_id = str(getattr(card, "card_id", "") or "")
            label = str(getattr(card, "name", "") or card_id)
            additions.append(
                type(
                    "_HandPlay",
                    (),
                    {
                        "action_code": 27,
                        "button_input": index,
                        "card_id": card_id,
                        "zone": "hand",
                        "label": label,
                    },
                )()
            )
        if not additions:
            return legal
        pass_actions = [action for action in legal if self._is_pass_like(action)]
        other = [
            action
            for action in legal
            if not self._is_pass_like(action)
            and (
                int(getattr(action, "action_code", 0) or 0) != 27
                or str(getattr(action, "zone", "") or "").strip().lower() != "hand"
            )
        ]
        return other + additions + pass_actions

    def _filter_legal_actions(self, legal: list[Any]) -> list[Any]:
        """Filter legal actions via the shared Talishar ``filter_legal_actions``."""
        working = self._augment_legal_before_filter(list(legal))
        phase = self._effective_turn_phase().upper()
        if phase == "M":
            playable = self._playable_hand_indices()
            if playable is not None:
                working = self._ensure_playable_hand_actions(working, playable)
                working = [
                    action
                    for action in working
                    if int(getattr(action, "action_code", 0) or 0) != 27
                    or str(getattr(action, "button_input", "") or "") in playable
                ]

        state = self._filter_state()
        legal_dicts = [self._action_to_dict(action) for action in working]
        filtered_dicts = filter_legal_actions(state, legal_dicts)
        return materialize_filtered_actions(
            working,
            filtered_dicts,
            to_descriptor=self._action_to_dict,
            make_action=self._make_action_from_descriptor,
        )

    def _loop_guard_for_step(self, legal: list[Any]) -> LoopGuardResult:
        state = self._board_state_for_loop_guard()
        return self._loop_guard.check(
            state,
            [self._action_to_dict(action) for action in legal],
            turn_no=self._obs_turn_no(),
            acting_player_id=self._acting_player,
        )

    def _macro_stall_obs_state(self) -> dict[str, Any]:
        gs = self._gs
        acting = int(self._acting_player)
        p1_hp = int(self._p1_hp)
        p2_hp = int(self._p2_hp)
        hand_size = 0
        if gs is not None:
            hand_size = int(
                getattr(gs, "p1_hand_size", 0) if acting == 1 else getattr(gs, "p2_hand_size", 0)
            )
        return {
            "turnNo": self._obs_turn_no(),
            "turnPhase": self._phase_code(),
            "actingPlayerID": acting,
            "playerHealth": p1_hp if acting == 1 else p2_hp,
            "opponentHealth": p2_hp if acting == 1 else p1_hp,
            "playerHandSize": hand_size,
        }

    def _check_macro_stall(self, legal: list[Any]) -> MacroStallResult:
        filtered = self._filter_legal_actions(legal)
        filtered_dicts = [self._action_to_dict(action) for action in filtered]
        return self._macro_stall_guard.observe(
            self._macro_stall_obs_state(),
            filtered_dicts,
            p1_hp=int(self._p1_hp),
            p2_hp=int(self._p2_hp),
        )

    def _macro_stall_info(self, result: MacroStallResult) -> dict[str, Any]:
        return {
            "macro_stall_truncated": bool(result.should_truncate),
            "macro_stall_reason": result.reason,
            "turns_without_damage": result.turns_without_damage,
            "pass_only_main_streak": result.pass_only_main_streak,
        }

    def _pass_like_action_from_legal(self, legal: list[Any]) -> Any:
        for action in legal:
            if self._is_pass_like(action):
                return action
        return self._make_pass_action()

    def _chosen_from_loop_guard(
        self,
        legal: list[Any],
        loop_guard: LoopGuardResult,
    ) -> Any:
        legal_dicts = [self._action_to_dict(action) for action in legal]
        mode, button_input = resolve_forced_submission(legal_dicts, loop_guard)
        for action in legal:
            desc = self._action_to_dict(action)
            if (
                int(desc.get("action_code", 0) or 0) == int(mode)
                and str(desc.get("button_input", "") or "") == str(button_input)
            ):
                return action
        return self._pass_like_action_from_legal(legal)

    def _phase_code(self) -> str:
        """Return the Talishar-like phase token exposed in observations."""
        return self._effective_turn_phase()

    def _obs_turn_no(self) -> int:
        if self._talishar_overlay and self._talishar_overlay.get("turn_no") is not None:
            return int(self._talishar_overlay["turn_no"])
        turn_no_override = getattr(self, "_turn_no_override", None)
        if turn_no_override is not None:
            return int(turn_no_override)
        return int(getattr(self._gs, "turn_no", 0) or 0)

    def _acting_idx(self) -> int:
        return 0 if self._acting_player == 1 else 1

    def _hand_cards(self) -> list[Any]:
        attr = "p1_hand" if self._acting_idx() == 0 else "p2_hand"
        cards = getattr(self._gs, attr, None)
        return list(cards) if isinstance(cards, list) else []

    def _pitch_count(self) -> int:
        """Talishar ``playerPitchCount`` = floating resource pool, not pitch-zone size."""
        gs = self._gs
        if gs is None:
            return 0
        attr = "p1_resources" if self._acting_idx() == 0 else "p2_resources"
        try:
            return int(getattr(gs, attr, 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _action_points_for_obs(self) -> int:
        """Talishar ``playerAP`` — action points for the active player in main."""
        phase = self._effective_turn_phase().upper()
        if phase not in {"M", "MAIN", "STARTTURN"}:
            return 0
        gs = self._gs
        if gs is None:
            return 0
        acting_idx = self._acting_idx()
        attr = "p1_action_points" if acting_idx == 0 else "p2_action_points"
        try:
            return int(getattr(gs, attr, 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _playable_indices_from_legal(self, legal: list[Any]) -> set[str]:
        playable: set[str] = set()
        for action in legal:
            if int(getattr(action, "action_code", 0) or 0) != 27:
                continue
            if str(getattr(action, "zone", "") or "").strip().lower() != "hand":
                continue
            playable.add(str(getattr(action, "button_input", "") or ""))
        return playable

    def _hand_entry_from_legal(
        self,
        legal: list[Any],
        *,
        index: int,
        card_id: str,
    ) -> dict[str, Any]:
        key = str(index)
        playable = self._playable_indices_from_legal(legal)
        for action in legal:
            if str(getattr(action, "zone", "") or "").strip().lower() != "hand":
                continue
            code = int(getattr(action, "action_code", 0) or 0)
            button = str(getattr(action, "button_input", "") or "")
            action_card_id = str(getattr(action, "card_id", "") or "")
            if button == key or button == card_id or action_card_id == card_id:
                override = button if code == 4 else key
                return {
                    "cardID": card_id,
                    "action": 27 if key in playable else code,
                    "actionDataOverride": override,
                    "label": str(getattr(action, "label", "") or ""),
                }
        return {
            "cardID": card_id,
            "action": 27 if key in playable else 0,
            "actionDataOverride": key,
            "label": "",
        }

    def _encode_player_hand(self, legal: list[Any]) -> list[dict[str, Any]]:
        """Return Talishar-compatible playerHand entries for all hand cards."""
        if self._talishar_overlay and self._talishar_overlay.get("player_hand") is not None:
            return list(self._talishar_overlay["player_hand"])

        cards = self._hand_cards()
        if not cards:
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

        out: list[dict[str, Any]] = []
        for i, c in enumerate(cards):
            card_id = str(getattr(c, "card_id", "") or "")
            out.append(self._hand_entry_from_legal(legal, index=i, card_id=card_id))
        return out

    def _apply_pass_action(self, chosen: Any) -> None:
        """Apply a pass-like action to the game state."""
        flow_handled = self._handle_flow_pass() if self._is_pass_like(chosen) else False
        if flow_handled:
            return
        cpp_action = self._resolve_cpp_action(chosen)
        if cpp_action is not None:
            self._gs.apply_action(cpp_action)
            self._acting_player = self._cpp_observation_acting_player()
        else:
            self._gs.apply_action(chosen)
            self._acting_player = self._cpp_observation_acting_player()

    def _auto_advance_pass_only(
        self,
        *,
        max_iters: int = 64,
    ) -> tuple[float, bool, bool]:
        """Auto-apply pass while only pass actions remain (skips agent decision steps)."""
        if self._strict_simulation:
            return 0.0, False, False
        extra_reward = 0.0
        for _ in range(max_iters):
            if self._is_game_over():
                return extra_reward, True, False
            if self._steps >= self._max_turns:
                return extra_reward, False, True

            legal = self._filter_legal_actions(self._legal_actions())
            if not legal or not all(self._is_pass_like(action) for action in legal):
                break

            chosen = legal[0]
            prev_p1 = self._gs.p1_health
            prev_p2 = self._gs.p2_health
            self._apply_pass_action(chosen)
            self._steps += 1
            repeat_pen = self._repeat_penalty(
                int(getattr(chosen, "action_code", 0) or 0),
                str(getattr(chosen, "button_input", "") or ""),
            )
            terminated = self._is_game_over()
            truncated = not terminated and self._steps >= self._max_turns
            extra_reward += self._mirrored_reward(
                self._compute_reward(prev_p1, prev_p2, terminated, truncated, repeat_pen)
            )
            if terminated:
                return extra_reward, True, False
            if truncated:
                return extra_reward, False, True
        return extra_reward, False, False

    def _legal_action_entries(self, legal: list[Any]) -> list[dict[str, Any]]:
        """Build Talishar-shaped legalActions list (index-aligned)."""
        return [
            {"index": i, "label": a.label, "zone": a.zone}
            for i, a in enumerate(legal)
        ]

    def _encode_observation(self, legal: list[Any]) -> str:
        """Encode the current C++ game state as a JSON string.

        *legal* must already be filtered (see :meth:`_filter_legal_actions`).
        Format mirrors :meth:`TalisharEngineEnvironment._encode_observation`.
        """
        gs = self._gs
        player_hand = self._encode_player_hand(legal)
        acting_idx = self._acting_idx()
        legal_entries = self._legal_action_entries(legal)

        if self._talishar_overlay:
            overlay = self._talishar_overlay
            contract_legal = self._contract_legal_actions(legal)
            legal_entries = self._legal_action_entries(legal)
            overlay_legal = overlay.get("legal_actions")
            if isinstance(overlay_legal, list) and overlay_legal:
                legal_entries = overlay_legal
            obs: dict[str, Any] = {
                "actingPlayerID": int(overlay.get("acting_player_id", self._acting_player)),
                "selfPlay": True,
                "playerHealth": int(overlay.get("player_health", 0)),
                "opponentHealth": int(overlay.get("opponent_health", 0)),
                "turnNo": self._obs_turn_no(),
                "turnPhase": str(overlay.get("turn_phase", self._phase_code())),
                "havePriority": bool(overlay.get("have_priority", True)),
                "playerHandSize": int(overlay.get("player_hand_size", len(player_hand))),
                "opponentHandSize": int(overlay.get("opponent_hand_size", 0)),
                "playerDeckCount": int(overlay.get("player_deck_count", 0)),
                "opponentDeckCount": int(overlay.get("opponent_deck_count", 0)),
                "playerPitchCount": int(overlay.get("player_pitch_count", 0)),
                "playerHand": player_hand,
                "legalActions": legal_entries,
                "legal_actions": contract_legal,
            }
            obs_vec = self._obs_vec_for_state(obs, legal, legal_dicts=contract_legal)
            obs["obsSchemaVersion"] = PLAYER_OBS_SCHEMA_VERSION
            obs["observationVec"] = player_observation_payload(obs_vec)
            obs_json = json.dumps(obs, separators=(",", ":"))
            return obs_json

        player_hand_size = (
            len(player_hand)
            if player_hand
            else (gs.p1_hand_size if acting_idx == 0 else gs.p2_hand_size)
        )
        obs = {
            "actingPlayerID": self._acting_player,
            "selfPlay": True,
            "playerHealth": gs.p1_health if self._acting_player == 1 else gs.p2_health,
            "opponentHealth": gs.p2_health if self._acting_player == 1 else gs.p1_health,
            "turnNo": self._obs_turn_no(),
            "turnPhase": self._phase_code(),
            "havePriority": not self._is_game_over(),
            "playerHandSize": player_hand_size,
            "opponentHandSize": (gs.p2_hand_size if self._acting_player == 1 else gs.p1_hand_size),
            "playerDeckCount": (gs.p1_deck_size if self._acting_player == 1 else gs.p2_deck_size),
            "opponentDeckCount": (gs.p2_deck_size if self._acting_player == 1 else gs.p1_deck_size),
            "playerPitchCount": self._pitch_count(),
            "playerHand": player_hand,
            "legalActions": legal_entries,
            "legal_actions": self._legal_to_dicts(legal),
        }
        obs_vec = self._obs_vec_for_state(obs, legal)
        obs["obsSchemaVersion"] = PLAYER_OBS_SCHEMA_VERSION
        obs["observationVec"] = player_observation_payload(obs_vec)
        obs_json = json.dumps(obs, separators=(",", ":"))
        return obs_json

    def _parse_action(self, action: Any, legal: list[Any]) -> tuple[int, Any]:
        """Return (list_index, action object) for an integer, dict, or pass string."""
        if isinstance(action, dict):
            descriptor = self._normalise_info_legal_action(action)
            for index, candidate in enumerate(legal):
                if self._action_to_dict(candidate) == descriptor:
                    return index, candidate
            return 0, self._make_action_from_descriptor(descriptor)

        action_str = str(action).strip().lower()
        if action_str == "pass":
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

    def _make_action_from_descriptor(self, descriptor: dict[str, Any]) -> Any:
        return type(
            "_DescriptorAction",
            (),
            {
                "action_code": int(descriptor.get("action_code", 0) or 0),
                "button_input": str(descriptor.get("button_input", "") or ""),
                "card_id": str(descriptor.get("card_id", "") or ""),
                "zone": str(descriptor.get("zone", "") or ""),
                "label": str(descriptor.get("label", "") or ""),
            },
        )()

    def _resolve_cpp_action(self, chosen: Any) -> Optional[Any]:
        """Map a filtered/synthetic action onto a compiled ``LegalAction``."""
        module_name = type(self._gs).__module__
        if type(chosen).__module__ == module_name and type(chosen).__name__ == "LegalAction":
            return chosen

        chosen_dict = self._action_to_dict(chosen)
        raw_legal = list(self._gs.get_legal_actions())
        card_id = str(chosen_dict.get("card_id", "") or "")
        zone = str(chosen_dict.get("zone", "") or "").strip().lower()
        code = int(chosen_dict.get("action_code", 0) or 0)
        button = str(chosen_dict.get("button_input", "") or "")

        if zone == "equipment" and code == 3:
            for candidate in raw_legal:
                candidate_dict = self._action_to_dict(candidate)
                if str(candidate_dict.get("zone", "") or "").strip().lower() != "equipment":
                    continue
                if int(candidate_dict.get("action_code", 0) or 0) != 3:
                    continue
                if card_id and str(candidate_dict.get("card_id", "") or "") != card_id:
                    continue
                return candidate
            for candidate in raw_legal:
                candidate_dict = self._action_to_dict(candidate)
                if str(candidate_dict.get("zone", "") or "").strip().lower() != "arsenal":
                    continue
                if int(candidate_dict.get("action_code", 0) or 0) != 5:
                    continue
                if card_id and str(candidate_dict.get("card_id", "") or "") != card_id:
                    continue
                if button and str(candidate_dict.get("button_input", "") or "") != button:
                    continue
                return candidate

        if card_id and zone == "hand":
            for candidate in raw_legal:
                candidate_dict = self._action_to_dict(candidate)
                if str(candidate_dict.get("card_id", "") or "") != card_id:
                    continue
                if zone and str(candidate_dict.get("zone", "") or "").strip().lower() != zone:
                    continue
                if code and int(candidate_dict.get("action_code", 0) or 0) not in (0, code):
                    continue
                return candidate

        for candidate in raw_legal:
            if self._action_to_dict(candidate) == chosen_dict:
                return candidate

        button = str(chosen_dict.get("button_input", "") or "")
        for candidate in raw_legal:
            candidate_dict = self._action_to_dict(candidate)
            if int(candidate_dict.get("action_code", 0) or 0) != code:
                continue
            if zone and str(candidate_dict.get("zone", "") or "").strip().lower() != zone:
                continue
            if button and str(candidate_dict.get("button_input", "") or "") != button:
                continue
            if card_id and str(candidate_dict.get("card_id", "") or "") != card_id:
                continue
            return candidate

        if self._is_pass_like(chosen):
            for candidate in raw_legal:
                if self._is_pass_like(candidate):
                    return candidate
            return None
        if int(chosen_dict.get("action_code", 0) or 0) == 10000:
            for candidate in raw_legal:
                if int(getattr(candidate, "action_code", 0) or 0) == 10000:
                    return candidate
            return self._make_action_from_descriptor(chosen_dict)
        return None

    def _repeat_penalty(self, action_code: int, button_input: str) -> float:
        return self._repeat_tracker.update(
            (action_code, button_input),
            turn_no=int(getattr(self._gs, "turn_no", 0) or 0),
            acting_player_id=int(self._acting_player),
            threshold=self._repeat_action_threshold,
            penalty=self._repeat_action_penalty,
        )

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
            reward = float(self._truncation_penalty)
        else:
            # P1-centric intermediate shaping: positive when P2 takes damage,
            # negative when P1 takes damage, regardless of who is acting.
            p1_now, p2_now = gs.p1_health, gs.p2_health
            dmg_dealt = max(0, prev_p2 - p2_now)  # P2 HP lost  → good for P1
            dmg_taken = max(0, prev_p1 - p1_now)  # P1 HP lost  → bad for P1
            scale = self._damage_reward_scale
            reward = dmg_dealt * scale - dmg_taken * scale + self._step_penalty
        return reward + repeat_penalty

    def _action_to_dict(self, action: Any) -> dict[str, Any]:
        card_id = self._resolve_action_card_id(action)
        return {
            "action_code": int(getattr(action, "action_code", 0) or 0),
            "button_input": str(getattr(action, "button_input", "") or ""),
            "card_id": card_id,
            "zone": str(getattr(action, "zone", "") or ""),
            "label": str(getattr(action, "label", "") or getattr(action, "name", "") or ""),
        }

    def _resolve_action_card_id(self, action: Any) -> str:
        card_id = str(getattr(action, "card_id", "") or "").strip()
        if card_id:
            return card_id
        zone = str(getattr(action, "zone", "") or "").strip().lower()
        code = int(getattr(action, "action_code", 0) or 0)
        button = str(getattr(action, "button_input", "") or "").strip()
        if zone == "hand" and code in {4, 27} and button.isdigit():
            hand = self._hand_cards()
            idx = int(button)
            if 0 <= idx < len(hand):
                return str(getattr(hand[idx], "card_id", "") or "").strip()
        return ""

    def _legal_to_dicts(self, legal: list[Any]) -> list[dict[str, Any]]:
        return [self._action_to_dict(a) for a in legal]

    def _tracker_state_snapshot(self, legal: list[Any]) -> dict[str, Any]:
        gs = self._gs
        acting = int(self._acting_player)
        return {
            "acting_player_id": acting,
            "turn_no": int(getattr(gs, "turn_no", 0) or 0),
            "phase": self._phase_code(),
            "p1_health": int(gs.p1_health),
            "p2_health": int(gs.p2_health),
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

    def _card_display_rows(self, cards: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not isinstance(cards, list):
            return rows
        for card in cards:
            rows.append(
                {
                    "card_id": str(getattr(card, "card_id", "") or ""),
                    "name": str(getattr(card, "name", "") or ""),
                    "cost": int(getattr(card, "cost", 0) or 0),
                    "pitch": int(getattr(card, "pitch", 0) or 0),
                    "power": int(getattr(card, "power", 0) or 0),
                    "defense": int(getattr(card, "defense", 0) or 0),
                }
            )
        return rows

    def live_display_snapshot(self) -> dict[str, Any]:
        """JSON-safe board snapshot for live C++ eval dashboards."""
        gs = self._gs
        if gs is None:
            return {"engine": "cpp", "status": "no_game"}
        legal = self._filter_legal_actions(self._legal_actions())
        return {
            "engine": "cpp",
            "status": "active",
            "turn_no": int(getattr(gs, "turn_no", 0) or 0),
            "acting_player_id": int(self._acting_player),
            "phase": str(self._phase_code()),
            "p1_health": int(gs.p1_health),
            "p2_health": int(gs.p2_health),
            "p1_hand": self._card_display_rows(getattr(gs, "p1_hand", None)),
            "p2_hand": self._card_display_rows(getattr(gs, "p2_hand", None)),
            "p1_deck_size": int(getattr(gs, "p1_deck_size", 0) or 0),
            "p2_deck_size": int(getattr(gs, "p2_deck_size", 0) or 0),
            "p1_pitch_size": int(getattr(gs, "p1_pitch_size", 0) or 0),
            "p2_pitch_size": int(getattr(gs, "p2_pitch_size", 0) or 0),
            "legal_actions": self._legal_to_dicts(legal),
            "game_over": bool(getattr(gs, "game_over", False)),
            "winner": int(getattr(gs, "winner", -1) or -1),
            "deck1": str(self._deck1 or ""),
            "deck2": str(self._deck2 or ""),
        }

    # â”€â”€ rlbridge interface â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        opts = options or {}
        self._reset_flow_state()
        self._hand_playability = self._normalize_hand_playability(
            opts.get("hand_playability")
        )
        self._gs = self._new_gamestate(opts, seed=seed)
        self._steps = 0
        self._p1_hp = self._gs.p1_health
        self._p2_hp = self._gs.p2_health
        acting_player = opts.get("acting_player_id")
        if acting_player is not None and hasattr(self._gs, "set_priority"):
            self._acting_player = int(acting_player)
        else:
            self._acting_player = self._cpp_observation_acting_player()
        playable = self._playable_hand_indices()
        if self._strict_simulation:
            self._flow_phase = self._phase_code_from_cpp()
        elif playable is not None and len(playable) == 0:
            self._flow_phase = "OPENING_MAIN"
        elif opts.get("opening_hands") and not self._hand_playability:
            self._flow_phase = "OPENING_MAIN"
        self._repeat_tracker.reset(
            turn_no=int(getattr(self._gs, "turn_no", 0) or 0),
            acting_player_id=int(self._acting_player),
        )
        self._loop_guard.reset()
        self._macro_stall_guard.reset()

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

        player_hp, opponent_hp = self._contract_player_hp()
        info: dict[str, Any] = {
            "engine": "cpp",
            "engine_dir": str(self._engine_dir),
            "legal_actions": self._contract_legal_actions(legal),
            "player_hp": player_hp,
            "opponent_hp": opponent_hp,
            "acting_player_id": self._acting_player,
            "self_play": True,
            "combat_tracker": self._tracker_stub(),
        }
        if self._last_observation_vec is not None:
            info["observation_vec"] = self._last_observation_vec
        return ResetResult(
            observation=obs,
            info=info,
        )

    def _step_from_talishar_mirror(self, action: Any) -> StepResult:
        """Return Talishar-mirrored step fields without local flow/auto-advance."""
        legal = self._filter_legal_actions(self._legal_actions())
        before_snapshot = (
            self._tracker_state_snapshot(legal) if self._enable_combat_tracker else None
        )
        legal_before = self._legal_to_dicts(legal) if self._enable_combat_tracker else []
        _, chosen = self._parse_action(action, legal)
        chosen_dict = self._action_to_dict(chosen)

        self._steps += 1
        self._consume_talishar_mirror_state()

        terminated, truncated = self._mirrored_termination(
            terminated=False,
            truncated=False,
        )
        repeat_streak, repeat_penalty = self._contract_repeat_fields(0.0)
        reward = self._mirrored_reward(0.0)

        new_legal = self._filter_legal_actions(self._legal_actions())
        obs = self._encode_observation(new_legal)
        after_snapshot = (
            self._tracker_state_snapshot(new_legal) if self._enable_combat_tracker else None
        )
        player_hp, opponent_hp = self._contract_player_hp()
        contract_legal = self._contract_legal_actions(new_legal)

        tracker_event: Optional[dict[str, Any]] = None
        if self._enable_combat_tracker and before_snapshot is not None and after_snapshot is not None:
            tracker_event = self._combat_tracker.record_step(
                before_snapshot=before_snapshot,
                after_snapshot=after_snapshot,
                action=chosen_dict,
                legal_before=legal_before,
                legal_after=contract_legal,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                combat_log_lines=self._synthetic_combat_log,
            )

        step_info: dict[str, Any] = {
            "engine": "cpp",
            "legal_actions": contract_legal,
            "turn": self._contract_turn(),
            "player_hp": player_hp,
            "opponent_hp": opponent_hp,
            "acting_player_id": self._contract_acting_player_id(),
            "self_play": True,
            "repeat_streak": repeat_streak,
            "repeat_penalty": repeat_penalty,
            "combat_tracker": self._tracker_stub(tracker_event),
        }
        if self._last_observation_vec is not None:
            step_info["observation_vec"] = self._last_observation_vec

        return StepResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=step_info,
        )

    def step(self, action: Any) -> StepResult:
        assert self._gs is not None, "call reset() first"
        if self._talishar_mirror_state is not None:
            return self._step_from_talishar_mirror(action)

        legal = self._filter_legal_actions(self._legal_actions())
        loop_guard = self._loop_guard_for_step(legal)
        before_snapshot = (
            self._tracker_state_snapshot(legal) if self._enable_combat_tracker else None
        )
        legal_before = self._legal_to_dicts(legal) if self._enable_combat_tracker else []

        idx, chosen = self._parse_action(action, legal)
        if loop_guard.force_pass:
            chosen = self._chosen_from_loop_guard(legal, loop_guard)
            idx = 0
            for candidate_index, candidate in enumerate(legal):
                if candidate is chosen:
                    idx = candidate_index
                    break
        chosen_dict = self._action_to_dict(chosen)

        prev_p1 = self._gs.p1_health
        prev_p2 = self._gs.p2_health

        flow_handled = False
        if self._is_pass_like(chosen):
            flow_handled = self._handle_flow_pass()
        elif self._handle_flow_hand_action(chosen):
            flow_handled = True

        cpp_action = self._resolve_cpp_action(chosen)
        if not flow_handled:
            if cpp_action is not None:
                self._gs.apply_action(cpp_action)
                self._acting_player = self._cpp_observation_acting_player()
            elif self._talishar_mirror_state is not None:
                flow_handled = True
            else:
                raise RuntimeError(
                    f"no C++ legal action matches {self._action_to_dict(chosen)!r}"
                )
            if not self._talishar_overlay and not self._talishar_mirror_state:
                self.clear_talishar_state()

        self._steps += 1

        self._consume_talishar_mirror_state()

        terminated = self._is_game_over()
        truncated = not terminated and self._steps >= self._max_turns
        terminated, truncated = self._mirrored_termination(
            terminated=terminated,
            truncated=truncated,
        )

        repeat_pen = self._repeat_penalty(chosen.action_code, chosen.button_input)
        reward = self._mirrored_reward(
            self._compute_reward(prev_p1, prev_p2, terminated, truncated, repeat_pen)
        )
        repeat_streak, repeat_pen = self._contract_repeat_fields(repeat_pen)

        auto_reward, auto_term, auto_trunc = self._auto_advance_pass_only()
        reward += auto_reward
        if auto_term:
            terminated = True
            truncated = False
        elif auto_trunc:
            truncated = True

        new_legal = self._filter_legal_actions(self._legal_actions())
        macro_stall = self._check_macro_stall(self._legal_actions())
        if macro_stall.should_truncate and not terminated:
            truncated = True
            reward = float(self._truncation_penalty)

        obs = self._encode_observation(new_legal)
        after_snapshot = (
            self._tracker_state_snapshot(new_legal) if self._enable_combat_tracker else None
        )
        player_hp, opponent_hp = self._contract_player_hp()
        contract_legal = self._contract_legal_actions(new_legal)

        tracker_event: Optional[dict[str, Any]] = None
        if self._enable_combat_tracker and before_snapshot is not None and after_snapshot is not None:
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
                legal_after=contract_legal,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                combat_log_lines=self._synthetic_combat_log,
            )

        step_info: dict[str, Any] = {
            "engine": "cpp",
            "legal_actions": contract_legal,
            "turn": self._contract_turn(),
            "player_hp": player_hp,
            "opponent_hp": opponent_hp,
            "acting_player_id": self._contract_acting_player_id(),
            "self_play": True,
            "repeat_streak": repeat_streak,
            "repeat_penalty": repeat_pen,
            "loop_guard_forced_pass": loop_guard.force_pass,
            "loop_guard_reason": loop_guard.reason,
            "turn_steps": loop_guard.turn_steps,
            "decision_loop_streak": loop_guard.loop_streak,
            "combat_tracker": self._tracker_stub(tracker_event),
        }
        if loop_guard.forced_action is not None:
            step_info["loop_guard_forced_action"] = dict(loop_guard.forced_action)
        step_info.update(self._macro_stall_info(macro_stall))
        if self._last_observation_vec is not None:
            step_info["observation_vec"] = self._last_observation_vec

        return StepResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=step_info,
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


# â”€â”€ Cache management â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "cpp_engines"


def explain_cpp_engine_unavailable(
    deck1: str,
    deck2: str,
    cache_dir: str | Path | None = None,
) -> str:
    """Return a human-readable reason when no loadable C++ engine exists."""
    base = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    key = _matchup_key(deck1, deck2)
    expected = expected_fab_engine_module_name()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"

    matchup_dirs: list[Path] = []
    if base.is_dir():
        key_lower = key.lower()
        for path in base.iterdir():
            if not path.is_dir():
                continue
            name_lower = path.name.lower()
            if name_lower == key_lower or name_lower.startswith(f"{key_lower}-"):
                matchup_dirs.append(path)

    for engine_dir in sorted(
        matchup_dirs,
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        if is_cpp_engine_available(engine_dir):
            return "engine is available (unexpected)"
        stale = sorted(
            {
                p.name
                for p in _iter_engine_module_candidates(engine_dir)
                if p.is_file() and not _module_matches_current_python(p)
            }
        )
        if stale:
            return (
                f"found fab_engine for a different Python ABI in {engine_dir.name} "
                f"({', '.join(stale)}); rebuild for Python {py_ver} with "
                f"python scripts/cpp/build_cpp_engine_for_matchup.py "
                f"--deck1 {deck1} --deck2 {deck2}"
            )
        return (
            f"engine directory {engine_dir.name} exists but no compatible "
            f"{expected}; run: python scripts/cpp/build_cpp_engine_for_matchup.py "
            f"--deck1 {deck1} --deck2 {deck2}"
        )

    if not base.is_dir():
        return (
            f"no engine cache at {base}; run: "
            f"python scripts/cpp/build_cpp_engine_for_matchup.py "
            f"--deck1 {deck1} --deck2 {deck2}"
        )

    return (
        f"no compiled engine for {deck1} vs {deck2} under {base} "
        f"(expected {key} or {key}-<hash>/{expected}); run: "
        f"python scripts/cpp/build_cpp_engine_for_matchup.py "
        f"--deck1 {deck1} --deck2 {deck2}"
    )


def get_engine_dir(
    deck1: str,
    deck2: str,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return the canonical cache directory for a deck matchup.

    Prefers an exact ``{deck1}_vs_{deck2}`` directory.  If that directory
    has no compiled module, falls back to the most-recently-modified
    hashed variant ``{deck1}_vs_{deck2}-<hash>`` (produced by
    scripts/cpp/build_cpp_engine_for_matchup.py when content-hashing is enabled).

    When the lookup key differs only by case from a cached directory
    (e.g. ``briar_vs_riptide`` vs ``Briar_vs_Riptide-<hash>``), the
    newest matching directory is returned.
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
    # Case-insensitive fallback (hero IDs are often lowercase; build scripts
    # title-case deck names when naming engine directories).
    key_lower = key.lower()
    if base.is_dir():
        ci_candidates = sorted(
            (
                p for p in base.iterdir()
                if p.is_dir()
                and (
                    p.name.lower() == key_lower
                    or p.name.lower().startswith(f"{key_lower}-")
                )
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for candidate in ci_candidates:
            if is_cpp_engine_available(candidate):
                return candidate
    # Return exact dir as default even if empty (lets callers report the error)
    return exact


def get_or_none(
    deck1: str,
    deck2: str,
    cache_dir: str | Path | None = None,
    max_turns: int = 2000,
    max_steps_per_turn: int = DEFAULT_MAX_STEPS_PER_TURN,
    loop_repeat_threshold: int = DEFAULT_LOOP_REPEAT_THRESHOLD,
    step_penalty: float = -0.001,
    truncation_penalty: float = -0.1,
    repeat_action_threshold: int = 3,
    repeat_action_penalty: float = -0.1,
    damage_reward_scale: float = 0.01,
    max_consecutive_passes: int = 20,
    enable_combat_tracker: bool = False,
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
            max_steps_per_turn=max_steps_per_turn,
            loop_repeat_threshold=loop_repeat_threshold,
            step_penalty=step_penalty,
            truncation_penalty=truncation_penalty,
            repeat_action_threshold=repeat_action_threshold,
            repeat_action_penalty=repeat_action_penalty,
            damage_reward_scale=damage_reward_scale,
            max_consecutive_passes=max_consecutive_passes,
            deck1=deck1,
            deck2=deck2,
            enable_combat_tracker=enable_combat_tracker,
        )
    except Exception:
        return None


