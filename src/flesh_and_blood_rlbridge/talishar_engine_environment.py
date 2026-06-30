"""RL environment backed by a live Talishar server instance.

The environment wraps the Talishar HTTP API (https://github.com/Talishar/Talishar)
and exposes it as an rlbridge environment with text-based observations and actions.

Observations are JSON strings containing the current game state (life totals, hand,
legal actions, etc.).  Actions are integer strings indexing into the ``legalActions``
list in the latest observation, or the special string ``"pass"`` to pass priority.

Prerequisites
-------------
A running Talishar Docker instance is required.

**Linux** — run once to set up::

    # 1. Install Docker Compose V2 plugin (if not already present)
    DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
    mkdir -p $DOCKER_CONFIG/cli-plugins
    curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
      -o $DOCKER_CONFIG/cli-plugins/docker-compose
    chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose

    # 2. Add your user to the docker group (log out/in once after this)
    sudo usermod -aG docker $USER
    newgrp docker

    # 3. Clone Talishar and its front-end, then start the server
    git clone https://github.com/Talishar/Talishar
    git clone https://github.com/Talishar/Talishar-FE
    cd Talishar
    docker compose up -d

**Windows** — run once to set up (PowerShell)::

    # 1. Install Docker Desktop for Windows (includes Docker Compose V2):
    #    https://docs.docker.com/desktop/install/windows-install/
    #    Enable WSL 2 backend when prompted and restart if required.

    # 2. Clone Talishar and its front-end, then start the server
    git clone https://github.com/Talishar/Talishar
    git clone https://github.com/Talishar/Talishar-FE
    Set-Location Talishar
    docker compose up -d

    # The server will be accessible at http://localhost once the containers
    # are healthy (check with: docker compose ps).

Then set ``TALISHAR_URL=http://localhost`` or pass ``base_url`` to the constructor.
For visual rendering (``render_mode="human"``), run Talishar-FE locally
(``npm run dev`` in Talishar-FE, default ``http://localhost:5173``) or set
``TALISHAR_FE_URL``.

HTTP API summary
----------------
* ``POST /APIs/CreateGame.php``  — JSON body: ``{fabdb, format, deckTestMode, visibility}``
  Creates an AI practice game.  Response includes ``gameName`` and ``authKey``.
* ``GET  /Start.php``            — query: ``{gameName, playerID=1}``
  Initialises the gamestate file.  Returns ``{success, authKey}``.
* ``GET  /GetNextTurn.php``      — query: ``{gameName, playerID, authKey, lastUpdate}``
  Returns the full game-state JSON.
* ``GET  /ProcessInput.php``     — query: ``{gameName, playerID, authKey, mode, buttonInput}``
  Submits a player action.  In AI practice mode the server runs ``CombatDummyAI()``
  / ``EncounterAI()`` for player 2 after each call.  In self-play mode (see
  ``self_play``) both players are controlled via this API and P2 AI is disabled.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import webbrowser
from typing import Any, Optional

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from rlbridge.environments.base import rlbridgeEnvironment
from rlbridge.protocol.messages import RenderResult, ResetResult, StepResult, TextSpace

from .combat_log_tracker import (
    CombatTurnTracker,
    extract_talishar_chat_log_lines,
    talishar_gamestate_revert_detected,
)
from .frontend_action_overlay import (
    ActionCoachHint,
    overlay_hints_payload,
    playwright_update_overlay_script,
)
from .deck_context import load_episode_context
from .obs_alignment import (
    align_observation_for_cpp_training,
    cpp_obs_alignment_enabled,
)
from .player_observation import (
    ACTION_CAPACITY,
    PLAYER_OBS_SCHEMA_VERSION,
    player_observation_payload,
    player_observation_vector,
)
from .legal_action_filter import (
    filter_legal_actions,
    is_waiting_for_other_player,
    player_choosing_in_snapshot,
    player_must_wait,
)
from .macro_stall_guard import MacroStallConfig, MacroStallGuard, MacroStallResult
from .state_loop_guard import (
    DEFAULT_LOOP_REPEAT_THRESHOLD,
    DEFAULT_MAX_STEPS_PER_TURN,
    TurnLoopGuard,
    board_state_fingerprint,
    first_pass_action,
    resolve_forced_submission,
)
from .talishar_default_policy import (
    choose_talishar_action_index,
    RepeatActionTracker,
    _get_phase as _dp_get_phase,
    _card_pitch_value,
    _is_pass_action,
    _is_revert_action,
    _match_action_card,
    _to_int as _dp_to_int,
    _CONFIRM_PHASES as _dp_confirm_phases,
    _CHOOSE_HAND_PHASES as _dp_choose_hand_phases,
    _BUTTON_INPUT_PHASES as _dp_button_input_phases,
    _POPUP_PHASES as _dp_popup_phases,
    _BLOCK_PHASES as _dp_block_phases,
    _DEFENSE_PHASES as _dp_defense_phases,
)
from .talishar_fast_client import DEFAULT_TALISHAR_URL, TalisharFastClient
from .talishar_oracle import TalisharConnectionError

# ── Optional C++ engine integration ──────────────────────────────────────────
# Import is best-effort: if the module hasn't been built yet the C++ engine
# feature is simply unavailable and we fall back to HTTP Talishar silently.
try:
    from .cpp_engine_environment import (
        CppEngineEnvironment as _CppEngineEnvironment,
        get_engine_dir as _cpp_get_engine_dir,
        get_or_none as _cpp_get_or_none,
        is_cpp_engine_available as _cpp_is_available,
    )
    _CPP_ENGINE_SUPPORT = True
except Exception:  # pragma: no cover
    _CPP_ENGINE_SUPPORT = False
    _CppEngineEnvironment = None  # type: ignore[assignment, misc]
    _cpp_get_engine_dir = None    # type: ignore[assignment]
    _cpp_get_or_none = None       # type: ignore[assignment]
    _cpp_is_available = None      # type: ignore[assignment]

# Default practice deck: Ira Crimson Haze (young hero, silver_age-legal)
_DEFAULT_DECK_LINK = "https://fabrary.net/decks/01GJG7Z4WGWSZ95FY74KX4M557"
_DEFAULT_RENDER_WIDTH = 1920
_DEFAULT_RENDER_HEIGHT = 1080
_PRIORITY_POLL_INTERVAL = 0.15
_PRIORITY_MAX_POLLS = 120
_PRIORITY_DEADLOCK_POLLS = 12
_PRIORITY_STEP_SYNC_POLLS = 40
_FAST_PRIORITY_POLL_INTERVAL = 0.02
_FAST_PRIORITY_MAX_POLLS = 15
_FAST_PRIORITY_DEADLOCK_POLLS = 6
_FAST_PRIORITY_STEP_SYNC_POLLS = 8
_HTTP_REQUEST_RETRIES = 6
_HTTP_RETRY_BASE_SLEEP_S = 0.5
_DISABLE_CARD_HOVER_STORAGE_KEY = "talishar-disable-card-hover"

_PLAYWRIGHT_GDPR_INIT_SCRIPT = (
    "localStorage.setItem('gdpr-analytics-enabled','true');"
    "localStorage.setItem('gdpr-consent-accepted','true');"
    "localStorage.setItem('cookieConsent','accepted');"
)


def rewrite_frontend_api_url(request_url: str, backend_base_url: str) -> str | None:
    """Map Talishar-FE dev-server API paths to a specific PHP backend shard.

    Vite proxies ``/api/*`` and ``/APIs/*`` to ``VITE_BACKEND_PORT`` (usually
    8080).  During rgb_array capture the game lives on another shard (e.g. the
    dedicated render port), so Playwright must forward those requests directly.
    Returns ``None`` when *request_url* should not be proxied.
    """
    from urllib.parse import urlsplit, urlunsplit

    backend = str(backend_base_url or "").strip().rstrip("/")
    if not backend:
        return None
    parsed = urlsplit(str(request_url or "").strip())
    path = parsed.path or ""
    if path.startswith("/api/"):
        target_path = path[len("/api/") :]
    elif path.startswith("/APIs/") or path.startswith("/AccountFiles/"):
        target_path = path.lstrip("/")
    elif path.endswith(".php"):
        target_path = path.lstrip("/")
    else:
        return None
    if not target_path:
        return None
    target = f"{backend}/{target_path}"
    if parsed.query:
        target = f"{target}?{parsed.query}"
    return target

_PLAYWRIGHT_DISABLE_CARD_HOVER_INIT_SCRIPT = (
    f"localStorage.setItem('{_DISABLE_CARD_HOVER_STORAGE_KEY}','true');"
    "(function(){"
    "const styleId='rlbridge-disable-card-hover';"
    "if(document.getElementById(styleId)){return;}"
    "const style=document.createElement('style');"
    "style.id=styleId;"
    "style.textContent="
    "'[class*=\"popUpContainer\"]{display:none!important;visibility:hidden!important;opacity:0!important;}';"
    "document.documentElement.appendChild(style);"
    "const disableHover=()=>{if(document.body){document.body.style.pointerEvents='none';}};"
    "if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',disableHover);}"
    "else{disableHover();}"
    "})();"
)

_PLAYWRIGHT_PREPARE_CAPTURE_SCRIPT = """() => {
  document.body && (document.body.style.pointerEvents = 'none');
  document.querySelectorAll('[class*="popUpContainer"]').forEach((el) => {
    el.style.display = 'none';
    el.style.visibility = 'hidden';
    el.style.opacity = '0';
  });
}"""


def _normalize_game_format(fmt: str) -> str:
    token = str(fmt or "").strip().lower()
    if token in {"silver age", "silver_age", "sage"}:
        return "silver_age"
    return token or "silver_age"


class TalisharEngineEnvironment(rlbridgeEnvironment):
    """RL environment that uses a live Talishar server as its game engine.

    By default (``self_play=True``) a single policy controls whichever player
    currently has priority (requires Talishar ``CreateLocalGame.php`` with
    ``selfPlay`` support).  With ``self_play=False``, the agent plays as
    player 1 against the built-in Combat Dummy AI (player 2).

    Parameters
    ----------
    base_url:
        Base URL of the Talishar server (e.g. ``"http://localhost"``).
        Defaults to the ``TALISHAR_URL`` environment variable or
        ``"http://localhost"``.
    deck_link:
        Fabrary/FaBDB deck link for the agent's deck.
    game_format:
        Game format string accepted by Talishar (e.g. ``"silver_age"``, ``"cc"``).
    self_play:
        When ``True``, disable P2 AI and alternate control by priority.
    timeout:
        HTTP request timeout in seconds.
    max_turns:
        Maximum number of agent steps before the episode is truncated.
        Truncation (no winner) applies a negative reward to discourage idle loops.
        Repeating the same action within a turn (3+ times) incurs a small penalty.
    max_steps_per_turn:
        Maximum agent decisions per player turn before Pass is forced automatically.
    loop_repeat_threshold:
        Force Pass after this many visits to the same decision point in one turn.
    render_mode:
        Rendering mode (``"human"``, ``"ansi"``, or ``None``).
    frontend_url:
        Base URL of the Talishar-FE dev server used by ``"human"`` rendering.
        Defaults to the ``TALISHAR_FE_URL`` environment variable or
        ``"http://localhost:5173"``.
    render_width, render_height:
        Playwright viewport size for ``rgb_array`` screenshots (default 1920×1080).
        Override via ``TALISHAR_RENDER_WIDTH`` / ``TALISHAR_RENDER_HEIGHT``.
    enable_combat_tracker:
        When ``True``, capture per-step combat traces and board-state
        action statistics for debugging and parity checks.  Tracking applies
        to :meth:`step` only; C++ fast training remains available when enabled.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        frontend_url: Optional[str] = None,
        deck_link: str = _DEFAULT_DECK_LINK,
        game_format: str = "silver_age",
        timeout: float = 15.0,
        request_timeout: float = 30.0,
        max_turns: int = 2000,
        max_steps_per_turn: int = DEFAULT_MAX_STEPS_PER_TURN,
        loop_repeat_threshold: int = DEFAULT_LOOP_REPEAT_THRESHOLD,
        step_penalty: float = -0.001,
        truncation_penalty: float = -0.1,
        repeat_action_threshold: int = 3,
        repeat_action_penalty: float = -0.1,
        damage_reward_scale: float = 0.01,
        max_consecutive_passes: int = 20,
        render_mode: Optional[str] = None,
        local_deck_name: Optional[str] = "Ira",
        opponent_deck_name: Optional[str] = None,
        self_play: bool = True,
        render_width: Optional[int] = None,
        render_height: Optional[int] = None,
        block_max_pitch_value: int = 3,
        block_min_resource_cost: int = 0,
        # C++ engine options ─────────────────────────────────────────────────
        use_cpp_engine: bool = False,
        talishar_backend: str = "fast",
        cpp_engine_cache_dir: Optional[str] = None,
        # Override the deck names used *only* for C++ engine cache lookup.
        # Useful when the game files use UUID-based names (Phase 3) but the
        # compiled engine was built for the original hero/deck IDs.
        cpp_engine_deck1: Optional[str] = None,
        cpp_engine_deck2: Optional[str] = None,
        # Explicit engine directory — bypasses the key/cache lookup entirely.
        # Takes priority over cpp_engine_deck1/2 and cpp_engine_cache_dir.
        cpp_engine_dir: Optional[str] = None,
        # Print step-by-step progress during reset() — useful for debugging
        # connection / game-creation hangs.  Off by default to avoid polluting
        # training output.
        verbose: bool = False,
        # Record per-step combat/turn traces and board-state action stats.
        enable_combat_tracker: bool = False,
        # When True (default), apply the same obs neutralization used by the C++
        # fast engine so policies trained on C++ see matching inputs here.
        cpp_obs_alignment: Optional[bool] = None,
        # When ``render_mode="rgb_array"``, launch Playwright headless (default) or
        # visible for live viewing.
        playwright_headless: bool = True,
        # Pin Talishar-FE to a fixed player view (e.g. 1 for human-vs-agent).
        frontend_player_id: Optional[int] = None,
        # Keep card hover previews in the Talishar FE (human play).  Disabled by
        # default for rgb_array capture / spectator live view.
        enable_frontend_card_hover: bool = False,
        # RL training optimizations for RLStep overlay (skip backups, slim JSON).
        rl_training_mode: bool = False,
        rl_slim_response: bool = True,
        macro_stall_enabled: bool = True,
        stall_no_damage_turns: int = 6,
        stall_pass_only_turns: int = 6,
        stall_no_damage_requires_low_hand: bool = False,
        stall_low_hand_turns: int = 3,
        stall_max_single_low_hand_turns: int = 5,
        stall_min_attack_hand: int = 2,
    ) -> None:
        self._base_url = (
            base_url or os.environ.get("TALISHAR_URL", DEFAULT_TALISHAR_URL)
        ).rstrip("/")
        self._frontend_url = (
            frontend_url or os.environ.get("TALISHAR_FE_URL", "http://localhost:5173")
        ).rstrip("/")
        self._frontend_game_url_template = os.environ.get(
            "TALISHAR_FE_GAME_URL_TEMPLATE", ""
        ).strip()
        self._deck_link = deck_link
        self._local_deck_name = local_deck_name  # use CreateLocalGame.php when set
        self._opponent_deck_name = opponent_deck_name
        self._self_play = self_play
        self._format = _normalize_game_format(game_format)
        self._timeout = timeout
        self._request_timeout = request_timeout
        self._max_turns = max_turns
        self._max_steps_per_turn = max_steps_per_turn
        self._loop_repeat_threshold = loop_repeat_threshold
        self._step_penalty = float(step_penalty)
        self._truncation_penalty = float(truncation_penalty)
        self._repeat_action_threshold = int(repeat_action_threshold)
        self._repeat_action_penalty = float(repeat_action_penalty)
        self._damage_reward_scale = float(damage_reward_scale)
        self._max_consecutive_passes = int(max_consecutive_passes)
        self._block_max_pitch_value = block_max_pitch_value
        self._block_min_resource_cost = block_min_resource_cost
        self._render_mode = render_mode
        self._playwright_headless = bool(playwright_headless)
        self._frontend_player_id = (
            int(frontend_player_id) if frontend_player_id is not None else None
        )
        self._enable_frontend_card_hover = bool(enable_frontend_card_hover)
        self._render_width = int(
            render_width
            or os.environ.get("TALISHAR_RENDER_WIDTH", _DEFAULT_RENDER_WIDTH)
        )
        self._render_height = int(
            render_height
            or os.environ.get("TALISHAR_RENDER_HEIGHT", _DEFAULT_RENDER_HEIGHT)
        )
        self._opened_frontend_url: Optional[str] = None
        self._verbose = verbose
        self._enable_combat_tracker = bool(enable_combat_tracker)
        self._combat_tracker = CombatTurnTracker(
            engine_name="talishar_http",
            enabled=self._enable_combat_tracker,
        )
        self._talishar_backend_requested = str(talishar_backend or "auto").strip().lower()
        self._resolved_talishar_backend = "http"
        self._fast_client: Optional[TalisharFastClient] = None
        self._rlstep_available = False
        self._rl_training_mode = bool(rl_training_mode)
        self._rl_slim_response = bool(rl_slim_response)
        self._rl_use_min_gamestate = True
        self._rl_parity_checked = False
        alignment_default = cpp_obs_alignment_enabled() if cpp_obs_alignment is None else bool(cpp_obs_alignment)
        self._cpp_obs_alignment = alignment_default

        # HTTP session with connection pooling and automatic retry on transient
        # server errors.  Fast backend uses keep-alive via TalisharFastClient.
        self._session: requests.Session = self._make_session(keep_alive=False)

        # Per-episode state
        self._game_name: Optional[str] = None
        self._auth_key: str = ""
        self._p1_auth_key: str = ""
        self._p2_auth_key: str = ""
        self._acting_player_id: int = 1
        self._last_state: dict[str, Any] = {}
        self._last_update: int = 0
        self._steps: int = 0
        self._player_hp: int = 20
        self._opp_hp: int = 20
        self._initialized: bool = False
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
        # Whether playerDeckCount > 0 has ever been observed this episode.
        # Guards deck-exhaustion game-over checks so they do NOT fire during the
        # pre-game equipment-selection phase, when Talishar returns
        # playerDeckCount=0 / empty playerHand before decks are shuffled.
        self._deck_nonzero_ever_seen: bool = False
        # Multi-select popup tracking (mode=16 picks + mode=19 submit payload)
        self._multi_select_inputs: list[str] = []
        self._pending_chk_inputs: Optional[list[str]] = None
        self._last_observation_vec: Optional[np.ndarray] = None
        self._p1_episode_context: Optional[Any] = None
        self._p2_episode_context: Optional[Any] = None

        # Persistent Playwright worker thread for rgb_array rendering
        self._pw_page: Any = None
        self._pw_browser: Any = None
        self._pw_playwright: Any = None
        self._pw_cmd_queue: Any = None
        self._pw_worker_thread: Any = None
        self._last_action_overlay_key: Optional[str] = None

        # ── C++ engine fast-path ──────────────────────────────────────────────
        # If a compiled fab_engine module exists for this matchup in the cache,
        # delegate all environment calls to it instead of using HTTP Talishar.
        # Falls back to HTTP silently if the module hasn't been built yet.
        self._cpp_env: Optional[Any] = None
        if use_cpp_engine and _CPP_ENGINE_SUPPORT and local_deck_name:
            deck2 = opponent_deck_name or local_deck_name
            # Use override names for cache lookup if provided (e.g. when game
            # files have UUID-based names but the engine was compiled for the
            # original hero IDs).
            lookup_deck1 = cpp_engine_deck1 or local_deck_name
            lookup_deck2 = cpp_engine_deck2 or deck2
            if cpp_engine_dir is not None:
                # Explicit dir takes priority — load directly, skip key lookup.
                from .cpp_engine_environment import (  # noqa: PLC0415
                    CppEngineEnvironment as _CppEnv,
                    is_cpp_engine_available as _is_avail,
                )
                if _is_avail(cpp_engine_dir):
                    try:
                        self._cpp_env = _CppEnv(
                            engine_dir=cpp_engine_dir,
                            max_turns=max_turns,
                            max_steps_per_turn=max_steps_per_turn,
                            loop_repeat_threshold=loop_repeat_threshold,
                            step_penalty=step_penalty,
                            truncation_penalty=truncation_penalty,
                            repeat_action_threshold=repeat_action_threshold,
                            repeat_action_penalty=repeat_action_penalty,
                            damage_reward_scale=damage_reward_scale,
                            max_consecutive_passes=max_consecutive_passes,
                            deck1=lookup_deck1,
                            deck2=lookup_deck2,
                            enable_combat_tracker=self._enable_combat_tracker,
                            macro_stall_enabled=macro_stall_enabled,
                            stall_no_damage_turns=stall_no_damage_turns,
                            stall_pass_only_turns=stall_pass_only_turns,
                            stall_no_damage_requires_low_hand=stall_no_damage_requires_low_hand,
                            stall_low_hand_turns=stall_low_hand_turns,
                            stall_max_single_low_hand_turns=stall_max_single_low_hand_turns,
                            stall_min_attack_hand=stall_min_attack_hand,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"C++ engine failed to load from {cpp_engine_dir}: {exc!r}"
                        ) from exc
                else:
                    raise RuntimeError(
                        f"C++ engine required (--cpp-engine-dir={cpp_engine_dir}) but "
                        f"fab_engine is not importable for Python "
                        f"{sys.version_info.major}.{sys.version_info.minor}. "
                        "Rebuild with scripts/cpp/build_cpp_engine_for_matchup.py"
                    )
            else:
                self._cpp_env = _cpp_get_or_none(  # type: ignore[misc]
                    lookup_deck1,
                    lookup_deck2,
                    cache_dir=cpp_engine_cache_dir,
                    max_turns=max_turns,
                    max_steps_per_turn=max_steps_per_turn,
                    loop_repeat_threshold=loop_repeat_threshold,
                    step_penalty=step_penalty,
                    truncation_penalty=truncation_penalty,
                    repeat_action_threshold=repeat_action_threshold,
                    repeat_action_penalty=repeat_action_penalty,
                    damage_reward_scale=damage_reward_scale,
                    max_consecutive_passes=max_consecutive_passes,
                    enable_combat_tracker=self._enable_combat_tracker,
                )
            if self._cpp_env is not None:
                import warnings
                warnings.warn(
                    f"[TalisharEngineEnvironment] Using C++ engine for "
                    f"{lookup_deck1} vs {lookup_deck2} (no HTTP required). "
                    "Pass use_cpp_engine=False to force HTTP Talishar.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self._finalize_talishar_backend(use_cpp_engine=use_cpp_engine)

    def _finalize_talishar_backend(self, *, use_cpp_engine: bool) -> None:
        """Resolve fast/http backend and observation alignment after C++ probe."""
        if self._using_cpp:
            return
        requested = self._talishar_backend_requested
        if requested == "http":
            self._resolved_talishar_backend = "http"
        elif requested in ("fast", "auto", "cpp"):
            pool_size = None
            pool_env = os.environ.get("FAB_TALISHAR_HTTP_POOL_SIZE", "").strip()
            if pool_env:
                try:
                    pool_size = max(4, int(pool_env))
                except ValueError:
                    pool_size = None
            self._fast_client = TalisharFastClient(
                self._base_url,
                request_timeout=self._request_timeout,
                keep_alive=True,
                pool_size=pool_size,
            )
            self._session = self._fast_client.session
            self._resolved_talishar_backend = "fast"
            self._rlstep_available = self._fast_client.probe_rlstep()
            self._combat_tracker._engine_name = "talishar_fast"  # noqa: SLF001
        else:
            self._resolved_talishar_backend = "http"
        # Full Talishar backends use native obs — alignment is C++-only.
        self._cpp_obs_alignment = False

    @property
    def _using_fast_talishar(self) -> bool:
        return self._resolved_talishar_backend == "fast" and not self._using_cpp

    @property
    def talishar_backend(self) -> str:
        if self._using_cpp:
            return "cpp"
        return self._resolved_talishar_backend

    def _priority_poll_interval(self) -> float:
        return _FAST_PRIORITY_POLL_INTERVAL if self._using_fast_talishar else _PRIORITY_POLL_INTERVAL

    def _priority_max_polls(self) -> int:
        return _FAST_PRIORITY_MAX_POLLS if self._using_fast_talishar else _PRIORITY_MAX_POLLS

    def _priority_deadlock_polls(self) -> int:
        return _FAST_PRIORITY_DEADLOCK_POLLS if self._using_fast_talishar else _PRIORITY_DEADLOCK_POLLS

    def _priority_step_sync_polls(self) -> int:
        return (
            _FAST_PRIORITY_STEP_SYNC_POLLS
            if self._using_fast_talishar
            else _PRIORITY_STEP_SYNC_POLLS
        )

    def _absolute_p1_p2_health(self, state: dict[str, Any]) -> tuple[int, int]:
        acting_hp = int(state.get("playerHealth", self._player_hp) or 0)
        opp_hp = int(state.get("opponentHealth", self._opp_hp) or 0)
        if self._acting_player_id == 1:
            return acting_hp, opp_hp
        return opp_hp, acting_hp

    def _fast_training_unavailable_reasons(self) -> list[str]:
        if self._using_cpp:
            return ["delegating to C++ engine"]
        if not self._using_fast_talishar:
            return ["talishar_backend is not fast"]
        if self._fast_client is None:
            return ["TalisharFastClient not initialized"]
        return []

    def _apply_rlstep_states(self, resp: dict[str, Any]) -> dict[str, Any]:
        if resp.get("notYourTurn"):
            server_current = int(resp.get("currentPlayer", self._acting_player_id))
            self._acting_player_id = server_current
            self._auth_key = self._auth_key_for(server_current)
        states = resp.get("states") or {}
        for pid in (1, 2):
            raw = states.get(str(pid)) or states.get(pid)
            if not isinstance(raw, dict):
                continue
            if raw.get("havePriority", False) or self._is_game_over(raw):
                return self._adopt_player_state(raw, pid)
        return self._resolve_priority_from_states(states)

    def _resolve_priority_from_states(
        self,
        states: dict[Any, Any],
    ) -> dict[str, Any]:
        normalized: dict[int, dict[str, Any]] = {}
        for key, value in states.items():
            if not isinstance(value, dict):
                continue
            try:
                pid = int(key)
            except (TypeError, ValueError):
                continue
            normalized[pid] = value
        if not normalized:
            return self._last_state
        inferred = self._infer_priority_player(normalized)
        if inferred is not None:
            return self._adopt_player_state(normalized[inferred], inferred)
        return self._adopt_player_state(normalized.get(1, {}), 1)

    def _rlstep_payload_extras(self) -> dict[str, Any]:
        extras: dict[str, Any] = {}
        if os.environ.get("FAB_RLSTEP_PROFILE", "").strip().lower() in {"1", "true", "yes"}:
            extras["profileTimings"] = True
        return extras

    def _legal_action_fingerprint(self, actions: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
        out: set[tuple[Any, ...]] = set()
        for action in actions:
            out.add(
                (
                    int(action.get("action_code", 0) or 0),
                    str(action.get("button_input", "")),
                    str(action.get("zone", "")),
                    str(action.get("label", "")),
                )
            )
        return out

    def _maybe_check_rlstep_parity(self, resp: dict[str, Any]) -> None:
        if not self._rl_training_mode or self._rl_parity_checked:
            return
        if os.environ.get("FAB_RLSTEP_PARITY_CHECK", "").strip().lower() not in {
            "1",
            "true",
            "yes",
        }:
            return
        compare = resp.get("compareStates")
        if not isinstance(compare, dict):
            return
        rl_state = compare.get("rl")
        full_state = compare.get("full")
        if not isinstance(rl_state, dict) or not isinstance(full_state, dict):
            return
        rl_legal = self._filter_legal_actions(
            rl_state, self._extract_legal_actions(rl_state)
        )
        full_legal = self._filter_legal_actions(
            full_state, self._extract_legal_actions(full_state)
        )
        if self._legal_action_fingerprint(rl_legal) != self._legal_action_fingerprint(
            full_legal
        ):
            raise RuntimeError(
                f"RLStep parity mismatch: legal actions differ "
                f"(rl={len(rl_legal)} full={len(full_legal)})"
            )

        saved_vec = self._last_observation_vec
        self._encode_observation(rl_state, rl_legal)
        rl_vec = np.asarray(self._last_observation_vec, dtype=np.float64)
        self._encode_observation(full_state, full_legal)
        full_vec = np.asarray(self._last_observation_vec, dtype=np.float64)
        self._last_observation_vec = saved_vec
        if rl_vec.shape != full_vec.shape:
            raise RuntimeError(
                f"RLStep parity mismatch: obs shape {rl_vec.shape} vs {full_vec.shape}"
            )
        if float(np.max(np.abs(rl_vec - full_vec))) > 0.05:
            raise RuntimeError("RLStep parity mismatch: observation vector diverged")
        self._rl_parity_checked = True

    def _submit_action_and_sync(
        self,
        mode: int,
        button_input: str,
        *,
        player_id: Optional[int] = None,
    ) -> dict[str, Any]:
        pid = player_id if player_id is not None else self._acting_player_id
        if self._using_fast_talishar and self._rlstep_available and self._fast_client:
            payload: dict[str, Any] = {
                "gameName": self._game_name or "",
                "playerID": pid,
                "authKey": self._auth_key_for(pid),
                "mode": int(mode),
            }
            if button_input:
                payload["buttonInput"] = button_input
                payload["cardID"] = button_input
            if self._rl_training_mode:
                payload["trainingMode"] = True
                if self._rl_slim_response:
                    payload["slimResponse"] = True
                if not self._rl_use_min_gamestate:
                    payload["useRlGameState"] = False
                if (
                    not self._rl_parity_checked
                    and os.environ.get("FAB_RLSTEP_PARITY_CHECK", "").strip().lower()
                    in {"1", "true", "yes"}
                ):
                    payload["compareGameStateBuild"] = True
            payload.update(self._rlstep_payload_extras())
            resp = self._fast_client.post_rlstep(payload)
            if resp.get("success"):
                self._maybe_check_rlstep_parity(resp)
                state = self._apply_rlstep_states(resp)
                try:
                    self._last_update = int(resp.get("lastUpdate", self._last_update))
                except (TypeError, ValueError):
                    pass
                return state
        self._submit_action(mode, button_input, player_id=pid)
        return self._sync_after_action()

    def _sync_after_action(self) -> dict[str, Any]:
        if self._using_fast_talishar:
            if self._self_play:
                return self._wait_for_any_priority(
                    max_polls=self._priority_max_polls(),
                    interval=self._priority_poll_interval(),
                )
            return self._poll_until_priority(
                interval=self._priority_poll_interval(),
                max_polls=self._priority_max_polls(),
            )
        time.sleep(0.35)
        if self._self_play:
            return self._wait_for_any_priority()
        return self._poll_until_priority()

    def _build_fast_step_result(
        self,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        winner: int = -1,
        loop_guard: Optional[Any] = None,
        macro_stall: Optional[MacroStallResult] = None,
    ) -> dict[str, Any]:
        legal_raw = self._legal_actions(self._last_state)
        legal = self._filter_legal_actions(self._last_state, legal_raw)
        obs_vec = np.asarray(self._last_observation_vec, dtype=np.float64)
        p1_hp, p2_hp = self._absolute_p1_p2_health(self._last_state)
        result: dict[str, Any] = {
            "obs_vec": obs_vec,
            "legal_count": len(legal),
            "acting_player_id": self._acting_player_id,
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "winner": winner,
            "p1_health": p1_hp,
            "p2_health": p2_hp,
            "p1_deck": int(self._last_state.get("playerDeckCount", 0) or 0)
            if self._acting_player_id == 1
            else int(self._last_state.get("opponentDeckCount", 0) or 0),
            "p2_deck": int(self._last_state.get("opponentDeckCount", 0) or 0)
            if self._acting_player_id == 1
            else int(self._last_state.get("playerDeckCount", 0) or 0),
            "turn_no": int(self._last_state.get("turnNo", 0) or 0),
        }
        if loop_guard is not None:
            result["loop_guard_forced_pass"] = bool(getattr(loop_guard, "force_pass", False))
            result["loop_guard_reason"] = getattr(loop_guard, "reason", "")
            result["turn_steps"] = getattr(loop_guard, "turn_steps", 0)
            result["decision_loop_streak"] = getattr(loop_guard, "loop_streak", 0)
            forced = getattr(loop_guard, "forced_action", None)
            if forced is not None:
                result["loop_guard_forced_action"] = dict(forced)
        if macro_stall is not None:
            result.update(self._macro_stall_info(macro_stall))
        return result

    # ── C++ engine delegation helpers ────────────────────────────────────────

    @property
    def _using_cpp(self) -> bool:
        """True when all environment calls are delegated to the C++ engine."""
        return self._cpp_env is not None

    @property
    def supports_fast_training(self) -> bool:
        if self._using_cpp:
            return bool(getattr(self._cpp_env, "supports_fast_training", False))
        return self._using_fast_talishar

    def fast_action_capacity(self) -> int:
        if self._using_cpp and hasattr(self._cpp_env, "fast_action_capacity"):
            return int(self._cpp_env.fast_action_capacity())  # type: ignore[union-attr]
        return ACTION_CAPACITY

    def fast_reset(
        self,
        seed: Optional[int] = None,
        *,
        starting_player_id: int = 1,
    ) -> dict[str, Any]:
        if self._using_cpp and hasattr(self._cpp_env, "fast_reset"):
            result = self._cpp_env.fast_reset(  # type: ignore[union-attr]
                seed=seed,
                starting_player_id=starting_player_id,
            )
            self._acting_player_id = int(result["acting_player_id"])
            self._player_hp = int(result["p1_health"])
            self._opp_hp = int(result["p2_health"])
            self._steps = 0
            return result
        if not self._using_fast_talishar:
            raise RuntimeError(
                "fast_reset requires talishar_backend='fast' or use_cpp_engine=True"
            )

        _ = seed  # Talishar shuffles server-side; seed not wired yet.
        self._session.cookies.clear()
        self._last_update = 0
        self._rl_parity_checked = False
        self._game_name, self._p1_auth_key, self._p2_auth_key = self._create_game()
        self._acting_player_id = 2 if int(starting_player_id) == 2 else 1
        started_key = self._start_game(self._game_name)
        if started_key:
            self._p1_auth_key = started_key
        self._auth_key = self._auth_key_for(self._acting_player_id)

        if self._self_play:
            self._last_state = self._wait_for_any_priority(
                max_polls=self._priority_step_sync_polls(),
                interval=self._priority_poll_interval(),
            )
        else:
            self._last_state = self._poll_until_priority(
                max_polls=self._priority_max_polls(),
                interval=self._priority_poll_interval(),
            )
        self._player_hp = int(self._last_state.get("playerHealth", 20))
        self._opp_hp = int(self._last_state.get("opponentHealth", 20))
        self._steps = 0
        self._deck_nonzero_ever_seen = False
        self._loop_guard.reset()
        self._macro_stall_guard.reset()
        self._multi_select_inputs = []
        self._pending_chk_inputs = None
        self._reset_repeat_tracking(
            turn_no=int(self._last_state.get("turnNo", 0) or 0),
            acting_player_id=self._acting_player_id,
        )
        self._refresh_episode_contexts(
            first_player=_dp_to_int(self._last_state.get("firstPlayer"), 1),
        )
        self._initialized = True
        legal_actions = self._legal_actions(self._last_state)
        self._encode_observation(self._last_state, legal_actions)
        return self._build_fast_step_result(
            reward=0.0,
            terminated=self._is_game_over(self._last_state),
            truncated=False,
        )

    def fast_step_index(self, action_index: int) -> dict[str, Any]:
        if self._using_cpp and hasattr(self._cpp_env, "fast_step_index"):
            result = dict(self._cpp_env.fast_step_index(action_index))  # type: ignore[union-attr]
            self._acting_player_id = int(result["acting_player_id"])
            self._player_hp = int(result["p1_health"])
            self._opp_hp = int(result["p2_health"])
            self._steps = int(getattr(self._cpp_env, "_steps", self._steps + 1))  # type: ignore[union-attr]
            if "truncated" not in result:
                terminated = bool(result.get("terminated", False))
                result["truncated"] = (
                    not terminated and self._steps >= self._max_turns
                )
            return result
        if not self._using_fast_talishar:
            raise RuntimeError(
                "fast_step_index requires talishar_backend='fast' or use_cpp_engine=True"
            )

        state = self._last_state
        if not self._is_game_over(state):
            prior_acting = self._acting_player_id
            if not state.get("havePriority", False) or player_must_wait(state):
                state = self._ensure_acting_priority(
                    max_polls=self._priority_step_sync_polls(),
                )
                if (
                    (not state.get("havePriority", False) or player_must_wait(state))
                    and self._self_play
                ):
                    state = self._wait_for_any_priority(
                        max_polls=self._priority_step_sync_polls(),
                        interval=self._priority_poll_interval(),
                    )
            if player_must_wait(state):
                legal_actions = self._legal_actions(state)
                self._encode_observation(state, legal_actions)
                return self._build_fast_step_result(
                    reward=self._step_penalty,
                    terminated=self._is_game_over(state),
                    truncated=False,
                )
            self._last_state = state

        legal_actions = self._legal_actions(state)
        loop_guard = self._loop_guard_for_step(state, legal_actions)
        mode, button_input = self._parse_action(str(action_index), legal_actions)
        if loop_guard.force_pass:
            mode, button_input = self._force_loop_guard_submission(
                legal_actions,
                loop_guard,
            )
        mode, button_input = self._sanitize_revert_submission(
            mode,
            button_input,
            legal_actions,
            state,
        )

        prev_player_hp = self._player_hp
        prev_opp_hp = self._opp_hp
        try:
            new_state = self._submit_action_and_sync(mode, button_input)
        except RuntimeError as exc:
            msg = str(exc)
            if "ProcessInput.php" not in msg and "RLStep.php" not in msg:
                raise
            try:
                new_state = self._submit_action_and_sync(99, "")
            except Exception:
                new_state = self._sync_after_action()
        new_state = self._recover_from_gamestate_revert_if_needed(
            new_state,
            submitted_mode=mode,
            submitted_button=button_input,
        )

        self._steps += 1
        new_player_hp = int(new_state.get("playerHealth", self._player_hp))
        new_opp_hp = int(new_state.get("opponentHealth", self._opp_hp))
        terminated = self._is_game_over(new_state)
        truncated = not terminated and self._steps >= self._max_turns

        new_legal_raw = self._legal_actions(new_state)
        macro_stall = self._check_macro_stall(new_state, new_legal_raw)
        if macro_stall.should_truncate and not terminated:
            truncated = True

        action_key = (mode, button_input)
        turn_no = int(new_state.get("turnNo", 0) or 0)
        if terminated or truncated:
            self._reset_repeat_tracking(
                turn_no=turn_no,
                acting_player_id=self._acting_player_id,
            )
            repeat_penalty = 0.0
        else:
            repeat_penalty = self._compute_repeat_action_penalty(
                action_key,
                turn_no=turn_no,
                acting_player_id=self._acting_player_id,
            )

        if terminated:
            won = self._did_player_win(new_player_hp, new_opp_hp)
            draw = self._is_draw(new_state)
            exhausted_loss = self._is_resource_exhausted_loss(new_state)
            if draw:
                reward = 0.0
            elif exhausted_loss:
                reward = -1.0 if self._acting_player_id == 1 else 1.0
            else:
                reward = 1.0 if won else -1.0
            winner = self._absolute_p1_seat_winner(
                new_state,
                exhausted_loss=exhausted_loss,
                draw=draw,
            )
        elif truncated:
            reward = self._truncation_penalty
            winner = -1
        else:
            dmg_dealt = max(0, prev_opp_hp - new_opp_hp)
            dmg_taken = max(0, prev_player_hp - new_player_hp)
            scale = self._damage_reward_scale
            reward = dmg_dealt * scale - dmg_taken * scale + self._step_penalty
            winner = -1
        reward += repeat_penalty

        self._player_hp = new_player_hp
        self._opp_hp = new_opp_hp
        self._last_state = new_state
        new_legal_actions = self._legal_actions(new_state)
        self._encode_observation(new_state, new_legal_actions)
        return self._build_fast_step_result(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            winner=winner,
            loop_guard=loop_guard,
            macro_stall=macro_stall,
        )

    def fast_logic_policy_action_index(self) -> int:
        if self._using_cpp and hasattr(self._cpp_env, "logic_policy_action_index"):
            return int(
                self._cpp_env.logic_policy_action_index(  # type: ignore[union-attr]
                    max_pitch_value=self._block_max_pitch_value,
                    min_resource_cost=self._block_min_resource_cost,
                )
            )

        if not self._last_state:
            return 0
        legal = self._legal_actions(self._last_state)
        if not legal:
            return 0

        loop_guard = self._loop_guard_for_step(self._last_state, legal)
        if loop_guard.force_pass:
            return self._index_for_loop_guard_action(legal, loop_guard)

        idx = choose_talishar_action_index(
            legal,
            self._last_state,
            max_pitch_value=self._block_max_pitch_value,
            min_resource_cost=self._block_min_resource_cost,
        )
        return min(max(0, int(idx)), len(legal) - 1)

    def _tracker_state_snapshot(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        player_hand = state.get("playerHand", [])
        opp_hand = state.get("opponentHand", [])
        return {
            "acting_player_id": int(self._acting_player_id),
            "turn_no": int(state.get("turnNo", 0) or 0),
            "phase": self._phase_str(state),
            "player_health": int(state.get("playerHealth", 0) or 0),
            "opponent_health": int(state.get("opponentHealth", 0) or 0),
            "player_hand_size": len(player_hand) if isinstance(player_hand, list) else 0,
            "opponent_hand_size": len(opp_hand) if isinstance(opp_hand, list) else 0,
            "player_deck_count": int(state.get("playerDeckCount", 0) or 0),
            "opponent_deck_count": int(state.get("opponentDeckCount", 0) or 0),
            "player_pitch_count": int(state.get("playerPitchCount", 0) or 0),
            "legal_count": len(legal_actions),
        }

    def _tracker_action_dict(self, action: dict[str, Any]) -> dict[str, Any]:
        return {
            "action_code": int(_dp_to_int(action.get("action_code", 0))),
            "button_input": str(action.get("button_input", "") or ""),
            "card_id": str(action.get("card_id", "") or ""),
            "zone": str(action.get("zone", "") or ""),
            "label": str(action.get("label", "") or ""),
        }

    def _tracker_action_for_submission(
        self,
        legal_actions: list[dict[str, Any]],
        mode: int,
        button_input: str,
    ) -> dict[str, Any]:
        target_mode = int(mode)
        target_button = str(button_input or "")
        for action in legal_actions:
            if (
                int(_dp_to_int(action.get("action_code", 0))) == target_mode
                and str(action.get("button_input", "") or "") == target_button
            ):
                return self._tracker_action_dict(action)
        for action in legal_actions:
            if int(_dp_to_int(action.get("action_code", 0))) == target_mode:
                return self._tracker_action_dict(action)
        return {
            "action_code": target_mode,
            "button_input": target_button,
            "card_id": "",
            "zone": "button",
            "label": f"mode={target_mode}",
        }

    def _tracker_stub(self, latest_event: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if self._using_cpp:
            try:
                snap = self._cpp_env.get_combat_tracker_snapshot(  # type: ignore[union-attr]
                    top_k=5,
                    tail_events=0,
                    tail_log_lines=25,
                )
                return {
                    "enabled": bool(snap.get("enabled", True)),
                    "engine": snap.get("engine", "cpp"),
                    "steps_recorded": int(snap.get("steps_recorded", 0) or 0),
                    "trace_digest": str(snap.get("trace_digest", "") or ""),
                }
            except Exception:
                return {"enabled": False}

        if not self._enable_combat_tracker:
            return {"enabled": False}

        snap = self._combat_tracker.snapshot(top_k=5, tail_events=0, tail_log_lines=25)
        out: dict[str, Any] = {
            "enabled": True,
            "engine": str(snap.get("engine", "talishar_http")),
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
        if self._using_cpp:
            return self._cpp_env.get_combat_tracker_snapshot(  # type: ignore[union-attr]
                top_k=top_k,
                tail_events=tail_events,
                tail_log_lines=tail_log_lines,
            )
        return self._combat_tracker.snapshot(
            top_k=top_k,
            tail_events=tail_events,
            tail_log_lines=tail_log_lines,
        )

    def get_server_report(self, *, tail_log_lines: int = 40) -> dict[str, Any]:
        """Compact Talishar server state snapshot for stuck-game diagnostics."""
        state = self._last_state or {}
        legal_raw = self._legal_actions(state) if state else []
        legal = self._filter_legal_actions(state, legal_raw) if state else []
        snapshot = (
            self._tracker_state_snapshot(state, legal)
            if state
            else {
                "acting_player_id": int(self._acting_player_id),
                "turn_no": 0,
                "phase": "",
                "player_health": 0,
                "opponent_health": 0,
                "legal_count": 0,
            }
        )
        chat_lines = extract_talishar_chat_log_lines(state.get("chatLog", ""))
        game_name = str(
            state.get("gameName")
            or state.get("gameID")
            or state.get("game")
            or ""
        )
        return {
            **snapshot,
            "have_priority": bool(state.get("havePriority", False)),
            "game_over": self._is_game_over(state) if state else False,
            "game_name": game_name,
            "gamestate_revert": (
                talishar_gamestate_revert_detected(state) if state else False
            ),
            "legal_actions": [
                {
                    "action_code": int(_dp_to_int(a.get("action_code", 0))),
                    "label": str(a.get("label", "") or ""),
                    "zone": str(a.get("zone", "") or ""),
                    "button_input": str(a.get("button_input", "") or ""),
                }
                for a in legal[:40]
            ],
            "combat_log": chat_lines[-tail_log_lines:],
            "board_fingerprint": board_state_fingerprint(state) if state else "",
        }

    def get_combat_trace(self) -> list[dict[str, Any]]:
        """Return the full per-step trace captured by the combat tracker."""
        if self._using_cpp:
            return self._cpp_env.get_combat_trace()  # type: ignore[union-attr]
        return self._combat_tracker.trace()

    def clear_combat_tracker(self) -> None:
        """Clear all currently tracked combat/turn events and counters."""
        if self._using_cpp:
            self._cpp_env.clear_combat_tracker()  # type: ignore[union-attr]
            return
        self._combat_tracker.clear()

    def live_display_snapshot(self) -> dict[str, Any]:
        """JSON-safe board snapshot for live eval dashboards (C++ engine only)."""
        if self._using_cpp and self._cpp_env is not None:
            return self._cpp_env.live_display_snapshot()  # type: ignore[union-attr]
        return {"engine": "talishar_http", "status": "unsupported"}

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _make_session(self, *, keep_alive: bool = False) -> requests.Session:
        """Build a requests.Session with connection pooling, keep-alive, and retry."""
        session = requests.Session()
        retry = Retry(
            total=2,
            connect=2,
            read=2,
            backoff_factor=0.4,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods={"GET", "POST"},
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            pool_connections=2,
            pool_maxsize=4,
            max_retries=retry,
        )
        session.mount("http://",  adapter)
        session.mount("https://", adapter)
        session.headers.update({
            "User-Agent": "TalisharRLEnv/1.0",
            "Connection": "keep-alive" if keep_alive else "close",
        })
        return session

    def _reset_http_session(self) -> None:
        """Drop pooled connections after a transport failure."""
        try:
            self._session.close()
        except Exception:
            pass
        self._session = self._make_session()

    def _http_retry_sleep(self, attempt: int) -> None:
        delay = min(8.0, _HTTP_RETRY_BASE_SLEEP_S * (2 ** attempt))
        time.sleep(delay)

    def _http_get(
        self,
        path: str,
        params: dict[str, str],
        _retries: int = _HTTP_REQUEST_RETRIES,
        *,
        allow_empty_body: bool = False,
    ) -> dict[str, Any]:
        url = self._base_url + path
        last_exc: Exception = RuntimeError("no attempts")
        for attempt in range(_retries):
            try:
                if attempt > 0:
                    self._http_retry_sleep(attempt - 1)
                resp = self._session.get(
                    url, params=params, timeout=self._request_timeout
                )
                body_text = resp.text
                if allow_empty_body and body_text.strip() == "":
                    return {}
                # Strip any PHP warnings printed before the JSON object
                obj_start = body_text.find("{")
                arr_start = body_text.find("[")
                starts = [i for i in (obj_start, arr_start) if i >= 0]
                if allow_empty_body and not starts:
                    return {}
                if starts:
                    body_text = body_text[min(starts):]
                data = json.loads(body_text)
                return data if isinstance(data, dict) else {"_raw": data}
            except json.JSONDecodeError:
                raise RuntimeError(
                    f"GET {url} returned non-JSON (HTTP {resp.status_code}):\n{resp.text[:2000]}"
                ) from None
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < _retries - 1:
                    self._reset_http_session()
        raise TalisharConnectionError(f"GET {url} failed: {last_exc}") from last_exc

    def _http_post_json(
        self,
        path: str,
        payload: dict[str, Any],
        _retries: int = _HTTP_REQUEST_RETRIES,
    ) -> dict[str, Any]:
        url = self._base_url + path
        last_exc: Exception = RuntimeError("no attempts")
        for attempt in range(_retries):
            try:
                if attempt > 0:
                    self._http_retry_sleep(attempt - 1)
                resp = self._session.post(
                    url, json=payload, timeout=self._request_timeout
                )
                resp_text = resp.text
                # Strip any PHP warnings printed before the JSON object
                json_start = resp_text.find("{")
                if json_start > 0:
                    resp_text = resp_text[json_start:]
                data = json.loads(resp_text)
                return data if isinstance(data, dict) else {"_raw": data}
            except json.JSONDecodeError:
                raise RuntimeError(
                    f"POST {url} returned non-JSON (HTTP {resp.status_code}):\n{resp.text[:2000]}"
                ) from None
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < _retries - 1:
                    self._reset_http_session()
        raise TalisharConnectionError(f"POST {url} failed: {last_exc}") from last_exc

    # ── Game lifecycle ────────────────────────────────────────────────────────

    def _create_game(self) -> tuple[str, str, str]:
        """Create a Talishar game.

        Uses ``CreateLocalGame.php`` (no external deck API) when
        ``local_deck_name`` is set, otherwise falls back to ``CreateGame.php``
        with the configured ``deck_link``.  Self-play requires the local-game
        API with ``selfPlay`` enabled on the server.

        Returns
        -------
        (game_name, p1_auth_key, p2_auth_key)
        """
        if self._self_play and not self._local_deck_name:
            raise ValueError(
                "Talishar self-play requires local_deck_name (CreateLocalGame.php)"
            )
        if self._local_deck_name:
            payload: dict[str, Any] = {
                "deckName": self._local_deck_name,
                "format": self._format,
                "visibility": "private",
            }
            opponent = self._opponent_deck_name
            if opponent:
                payload["opponentDeckName"] = opponent
            elif self._self_play:
                payload["opponentDeckName"] = self._local_deck_name
            if self._self_play:
                payload["selfPlay"] = "1"
            endpoint = "/APIs/CreateLocalGame.php"
        else:
            if self._self_play:
                raise ValueError(
                    "Talishar self-play requires local_deck_name (CreateLocalGame.php)"
                )
            payload = {
                "fabdb": self._deck_link,
                "format": self._format,
                "deckTestMode": "1",
                "visibility": "private",
            }
            if self._opponent_deck_name:
                payload["deckTestDeck"] = self._opponent_deck_name
            endpoint = "/APIs/CreateGame.php"
        resp = self._http_post_json(endpoint, payload)
        if "error" in resp:
            raise RuntimeError(f"CreateGame failed: {resp['error']}")
        game_name = str(resp.get("gameName", ""))
        p1_auth_key = str(resp.get("authKey", ""))
        p2_auth_key = str(resp.get("p2AuthKey", ""))
        if not game_name:
            raise RuntimeError(
                f"CreateGame returned no gameName.  Full response: {resp}"
            )
        if self._self_play and not p2_auth_key:
            raise RuntimeError(
                "CreateLocalGame selfPlay did not return p2AuthKey.  "
                "Ensure Talishar/APIs/CreateLocalGame.php supports selfPlay."
            )
        return game_name, p1_auth_key, p2_auth_key

    def _start_game(self, game_name: str) -> str:
        """Call Start.php to write the initial gamestate file.

        Only P1 needs to call Start.php — it writes the complete gamestate for
        both players.  Calling it a second time for P2 would truncate and
        rewrite the file (fopen "w" mode), and any difference in PHP session
        context for P2 (e.g. $firstPlayer not set) produces a broken gamestate.

        Returns the (possibly updated) P1 auth key.
        """
        resp = self._http_get("/Start.php", {"gameName": game_name, "playerID": "1"})
        # Start.php returns {"success": true, "authKey": "..."}
        returned_key = resp.get("authKey", "")
        return str(returned_key) if returned_key else self._auth_key

    def _auth_key_for(self, player_id: int) -> str:
        if player_id == 2:
            return self._p2_auth_key
        return self._p1_auth_key

    def _fetch_state(
        self,
        player_id: Optional[int] = None,
        *,
        last_update: Optional[int] = None,
    ) -> dict[str, Any]:
        """Fetch the current game state from GetNextTurn.php."""
        pid = player_id if player_id is not None else self._acting_player_id
        lu = self._last_update if last_update is None else last_update
        params: dict[str, str] = {
            "gameName": self._game_name or "",
            "playerID": str(pid),
            "authKey": self._auth_key_for(pid),
            "lastUpdate": str(lu),
        }
        state = self._http_get("/GetNextTurn.php", params)
        if last_update is None:
            self._apply_last_update(state)
        return state

    def _submit_action(
        self,
        mode: int,
        button_input: str = "",
        *,
        player_id: Optional[int] = None,
    ) -> None:
        """Submit a player action via ProcessInput.php (GET endpoint).

        Most card-zone modes (27 = play from hand, 3 = equipment, 5 = arsenal,
        10/14/21/22 = other zones) read the card index or card-ID from the PHP
        ``$cardID`` variable (``$_GET["cardID"]``), NOT from ``$buttonInput``.
        Prompt/decision modes (17 = BUTTONINPUT, 20 = YESNO, 99 = pass, etc.)
        read from ``$buttonInput``.  Sending the value under both keys satisfies
        both families without any mode-specific branching.

        If the server returns ``{"notYourTurn": true}`` the acting player ID is
        corrected to the server's ``currentPlayer`` and the action is retried
        once, which handles the race between start-of-game priority assignment
        and the first Python call.
        """
        pid = player_id if player_id is not None else self._acting_player_id
        if _is_revert_action(
            {"action_code": mode, "button_input": button_input, "label": ""}
        ):
            mode, button_input = 99, ""
        chk_payload: list[str] = []
        if mode == 19:
            if self._pending_chk_inputs is not None:
                chk_payload = [str(v) for v in self._pending_chk_inputs if str(v) != ""]
            else:
                chk_payload = [str(v) for v in self._multi_select_inputs if str(v) != ""]
        for attempt in range(2):
            params: dict[str, str] = {
                "gameName": self._game_name or "",
                "playerID": str(pid),
                "authKey": self._auth_key_for(pid),
                "mode": str(mode),
            }
            if button_input:
                params["buttonInput"] = button_input
                params["cardID"] = button_input  # card-zone modes (27, 3, 5, …) index via $cardID
            if mode == 19 and chk_payload:
                params["chkCount"] = str(len(chk_payload))
                for i, value in enumerate(chk_payload):
                    params[f"chk{i}"] = value
            resp = self._http_get("/ProcessInput.php", params, allow_empty_body=True)
            # Log any PHP-level errors returned by ProcessInput so we can
            # diagnose WriteGamestateCache / SHMOP failures without guessing.
            if self._verbose and (resp.get("error") or resp.get("errorMessage")):
                err = resp.get("error") or resp.get("errorMessage")
                print(f"    [submit P{pid}] ProcessInput.php ERROR: {err!r}", flush=True)
            if resp.get("notYourTurn"):
                # Server says it's the other player's turn — correct and retry once
                server_current = int(resp.get("currentPlayer", 3 - pid))
                if self._verbose:
                    print(f"    [submit] notYourTurn — switching P{pid}→P{server_current}",
                          flush=True)
                pid = server_current
                self._acting_player_id = pid
                self._auth_key = self._auth_key_for(pid)
            else:
                self._pending_chk_inputs = None
                break

    def _apply_last_update(self, state: dict[str, Any]) -> None:
        try:
            self._last_update = int(state.get("lastUpdate", self._last_update))
        except (ValueError, TypeError):
            pass

    def _fetch_both_player_states(self) -> dict[int, dict[str, Any]]:
        """Return full snapshots for P1 and P2."""
        return {
            pid: self._fetch_state(player_id=pid, last_update=0)
            for pid in (1, 2)
        }

    def _infer_priority_player(
        self,
        states: dict[int, dict[str, Any]],
    ) -> Optional[int]:
        """Guess who should act when ``havePriority`` is missing on both seats."""
        for pid in (1, 2):
            state = states.get(pid, {})
            if state.get("havePriority", False):
                return pid
            if self._is_game_over(state):
                return pid

        waiting = [
            pid
            for pid in (1, 2)
            if is_waiting_for_other_player(states.get(pid, {}))
        ]
        if len(waiting) == 1:
            return 3 - waiting[0]

        choosing = [
            pid
            for pid in (1, 2)
            if player_choosing_in_snapshot(states.get(pid, {}))
        ]
        if len(choosing) == 1:
            return choosing[0]
        if len(choosing) == 2 and len(waiting) == 1:
            return next(pid for pid in choosing if pid != waiting[0])
        return None

    def _adopt_player_state(
        self,
        state: dict[str, Any],
        acting_player_id: int,
    ) -> dict[str, Any]:
        """Sync wrapper bookkeeping to *acting_player_id*'s snapshot."""
        self._acting_player_id = acting_player_id
        self._auth_key = self._auth_key_for(acting_player_id)
        self._apply_last_update(state)
        return state

    def _resolve_priority_holder(self) -> dict[str, Any]:
        """Find whichever player should act, using priority flags and prompts."""
        states = self._fetch_both_player_states()
        for pid in (1, 2):
            state = states[pid]
            if state.get("havePriority", False) or self._is_game_over(state):
                return self._adopt_player_state(state, pid)

        inferred = self._infer_priority_player(states)
        if inferred is not None:
            return self._adopt_player_state(states[inferred], inferred)

        return self._adopt_player_state(
            states.get(self._acting_player_id, states[1]),
            self._acting_player_id,
        )

    def _ensure_acting_priority(
        self,
        *,
        max_polls: int = _PRIORITY_STEP_SYNC_POLLS,
    ) -> dict[str, Any]:
        """Poll until some player has priority or we can infer who should act."""
        interval = _PRIORITY_POLL_INTERVAL
        for i in range(max_polls):
            states = self._fetch_both_player_states()
            error_count = 0
            for pid in (1, 2):
                state = states[pid]
                err_msg = state.get("error", "")
                is_transient_error = (
                    isinstance(err_msg, str) and "too short" in err_msg and i < 10
                )
                if state.get("havePriority", False) or self._is_game_over(state):
                    if not is_transient_error:
                        return self._adopt_player_state(state, pid)
                if err_msg and not is_transient_error:
                    error_count += 1
            if error_count == 2:
                return self._adopt_player_state({"error": "game_crashed"}, 1)
            if i >= _PRIORITY_DEADLOCK_POLLS and i % 3 == 0:
                inferred = self._infer_priority_player(states)
                if inferred is not None:
                    inferred_state = states[inferred]
                    if (
                        inferred_state.get("havePriority", False)
                        or player_choosing_in_snapshot(inferred_state)
                        or not is_waiting_for_other_player(inferred_state)
                    ):
                        return self._adopt_player_state(inferred_state, inferred)
            time.sleep(interval)
        return self._resolve_priority_holder()

    def _return_priority_resync(self, state: dict[str, Any]) -> StepResult:
        """Return an observation for the priority holder without submitting."""
        self._player_hp = int(state.get("playerHealth", self._player_hp))
        self._opp_hp = int(state.get("opponentHealth", self._opp_hp))
        self._last_state = state
        legal_actions = self._legal_actions(state)
        obs = self._encode_observation(state, legal_actions)
        step_info: dict[str, Any] = {
            "legal_actions": legal_actions,
            "turn": state.get("turnNo", 0),
            "player_hp": self._player_hp,
            "opponent_hp": self._opp_hp,
            "acting_player_id": self._acting_player_id,
            "self_play": self._self_play,
            "repeat_streak": self._repeat_tracker.repeat_streak,
            "repeat_penalty": 0.0,
            "priority_resync": True,
            "combat_tracker": self._tracker_stub(),
        }
        if self._last_observation_vec is not None:
            step_info["observation_vec"] = self._last_observation_vec
        return StepResult(
            observation=obs,
            reward=self._step_penalty,
            terminated=self._is_game_over(state),
            truncated=False,
            info=step_info,
        )

    def _sync_acting_player(self) -> dict[str, Any]:
        """Set ``_acting_player_id`` to whichever player currently has priority."""
        return self._resolve_priority_holder()

    def _adopt_server_state(self, state: dict[str, Any], acting_player_id: int) -> str:
        """Sync wrapper state from a GetNextTurn snapshot and return observation."""
        self._acting_player_id = acting_player_id
        self._auth_key = self._auth_key_for(acting_player_id)
        self._apply_last_update(state)
        self._player_hp = int(state.get("playerHealth", self._player_hp))
        self._opp_hp = int(state.get("opponentHealth", self._opp_hp))
        self._last_state = state
        legal_actions = self._legal_actions(state)
        return self._encode_observation(state, legal_actions)

    def wait_for_human_player(
        self,
        player_id: int = 1,
        *,
        poll_interval: float = 0.3,
        max_wait_s: float = 3600.0,
        on_waiting: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
    ) -> str:
        """Block until *player_id* yields priority via the Talishar frontend.

        The human submits actions through Talishar-FE (ProcessInput).  This
        method polls until another player gains priority or the game ends.

        *on_waiting* is called on each poll while *player_id* still has
        priority.  Use it to refresh frontend coaching overlays.

        If *cancel_event* is set and becomes set, raises ``LivePlayCancelled``.
        """
        from flesh_and_blood_rlbridge.live_play_cancel import LivePlayCancelled

        deadline = time.time() + max_wait_s
        last_overlay_poll = 0.0
        while time.time() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise LivePlayCancelled("Live play session cancelled")
            for pid in (1, 2):
                probe = self._fetch_state(player_id=pid, last_update=0)
                if self._is_game_over(probe):
                    return self._adopt_server_state(probe, pid)

            acting_pid: Optional[int] = None
            acting_state: Optional[dict[str, Any]] = None
            for pid in (1, 2):
                probe = self._fetch_state(player_id=pid, last_update=0)
                if probe.get("havePriority", False):
                    acting_pid = pid
                    acting_state = probe
                    break

            if acting_pid is None:
                time.sleep(poll_interval)
                continue
            if acting_pid != player_id:
                self.clear_frontend_action_overlay()
                return self._adopt_server_state(acting_state, acting_pid)

            now = time.time()
            if on_waiting is not None and (now - last_overlay_poll) >= poll_interval:
                try:
                    on_waiting(acting_state)
                except Exception:
                    pass
                last_overlay_poll = now

            time.sleep(poll_interval)

        raise TimeoutError(
            f"Timed out after {max_wait_s:.0f}s waiting for player {player_id} "
            "to act in the Talishar frontend."
        )

    def _poll_until_priority(
        self,
        max_polls: int = 600,
        interval: float = 0.1,
    ) -> dict[str, Any]:
        """Poll until the acting player has priority or the game ends."""
        target = self._acting_player_id
        for _ in range(max_polls):
            state = self._fetch_state(player_id=target)
            if state.get("havePriority", False) or self._is_game_over(state):
                return state
            time.sleep(interval)
        return self._fetch_state(player_id=target)

    def _wait_for_any_priority(
        self,
        max_polls: int = _PRIORITY_MAX_POLLS,
        interval: float = _PRIORITY_POLL_INTERVAL,
    ) -> dict[str, Any]:
        """Self-play step helper: check BOTH players each poll until either has priority.

        This replaces the ``_poll_until_priority`` + ``_sync_acting_player``
        pair used in the self-play branch of ``step()``.  The original pair
        had two problems:

        1. ``_poll_until_priority`` targets only the *last-acting* player.
           In self-play mode the server never auto-passes for the other side,
           so the just-acted player will not regain priority until the opponent
           explicitly submits an action.  Polling for one player while the
           other has priority wastes the full 60-second timeout on every step.

        2. ``_poll_until_priority`` uses incremental ``lastUpdate`` fetches.
           When no delta exists the response body may omit ``havePriority``
           entirely, causing the check to return False even when the player
           genuinely has priority.

        This method uses ``last_update=0`` (full snapshots) and alternates
        between P1 and P2 on every iteration so it finds the acting player
        within one round-trip after the server processes the submitted action.

        When neither seat reports priority for several polls, prompt/caption
        heuristics infer the correct acting player (e.g. instant windows where
        one seat shows "waiting for other player").

        Timeout: ``max_polls * interval`` seconds (default 18 s) before
        falling back to ``_resolve_priority_holder``.
        """
        for i in range(max_polls):
            states = self._fetch_both_player_states()
            error_count = 0
            for pid in (1, 2):
                state = states[pid]
                has_priority = state.get("havePriority", False)
                is_over = self._is_game_over(state)
                err_msg = state.get("error", "")
                is_transient_error = (
                    isinstance(err_msg, str) and "too short" in err_msg
                    and i < 10
                )
                if self._verbose and i == 0:
                    print(f"    [wait poll {i+1}] P{pid}  havePriority={has_priority}  "
                          f"phase={self._phase_str(state)!r}  "
                          f"hp={state.get('playerHealth','?')}/"
                          f"{state.get('opponentHealth','?')}  "
                          f"keys={list(state.keys())[:8]}"
                          + (f"  ERROR: {err_msg!r}" if err_msg else ""),
                          flush=True)
                if (has_priority or is_over) and not is_transient_error:
                    return self._adopt_player_state(state, pid)
                if err_msg and not is_transient_error:
                    error_count += 1
            if error_count == 2:
                if self._verbose:
                    print("    [wait] both players returned fatal error — "
                          "ending episode", flush=True)
                return {"error": "game_crashed"}
            if i >= _PRIORITY_DEADLOCK_POLLS and i % 3 == 0:
                inferred = self._infer_priority_player(states)
                if inferred is not None:
                    inferred_state = states[inferred]
                    if (
                        inferred_state.get("havePriority", False)
                        or player_choosing_in_snapshot(inferred_state)
                        or not is_waiting_for_other_player(inferred_state)
                    ):
                        if self._verbose:
                            print(
                                f"    [wait] inferred priority → P{inferred}",
                                flush=True,
                            )
                        return self._adopt_player_state(inferred_state, inferred)
            if self._verbose and i > 0 and i % 5 == 0:
                print(f"    [wait] poll {i+1}/{max_polls}: no priority yet...",
                      flush=True)
            time.sleep(interval)
        if self._verbose:
            print(f"    [wait] timed out after {max_polls} polls — resolving holder",
                  flush=True)
        return self._resolve_priority_holder()

    # ── State helpers ─────────────────────────────────────────────────────────

    def _phase_str(self, state: dict[str, Any]) -> str:
        turn_phase = state.get("turnPhase", {})
        if isinstance(turn_phase, dict):
            return str(turn_phase.get("turnPhase", ""))
        return ""

    def _is_game_over(self, state: dict[str, Any]) -> bool:
        # Fatal crash sentinel emitted by _wait_for_any_priority when both
        # players return persistent PHP errors.  Treat as episode termination
        # so the caller gets terminated=True and starts a fresh game next reset.
        if state.get("error") == "game_crashed":
            return True
        if self._phase_str(state) == "OVER":
            return True
        # Use None as sentinel so that a missing key (empty/pre-game state) does
        # NOT accidentally trigger game-over.  The original -1 default caused
        # false positives whenever Talishar returned a state without playerHealth.
        player_hp = state.get("playerHealth")
        opp_hp = state.get("opponentHealth")
        if player_hp is not None and isinstance(player_hp, (int, float)) and int(player_hp) <= 0:
            return True
        if opp_hp is not None and isinstance(opp_hp, (int, float)) and int(opp_hp) <= 0:
            return True
        # Track whether the deck has ever been non-zero this episode so that the
        # deck-exhaustion checks below do NOT fire during the pre-game equipment
        # phase.  During equipment selection Talishar returns playerDeckCount=0
        # with an empty hand because the deck hasn't been shuffled yet — those
        # are NOT end-of-game conditions.
        p_deck = state.get("playerDeckCount")
        o_deck = state.get("opponentDeckCount")
        if p_deck is not None and int(p_deck) > 0:
            self._deck_nonzero_ever_seen = True
        if o_deck is not None and int(o_deck) > 0:
            self._deck_nonzero_ever_seen = True
        # Draw / exhaustion checks only make sense once decks have been shuffled.
        if self._deck_nonzero_ever_seen:
            # Draw detection: both decks exhausted with neither player at 0 HP.
            if (
                p_deck is not None and o_deck is not None
                and int(p_deck) == 0 and int(o_deck) == 0
            ):
                return True
            # Resource exhaustion: acting player has empty deck AND empty hand —
            # they cannot play another card and will lose to fatigue.  End now.
            if p_deck is not None and int(p_deck) == 0:
                hand = state.get("playerHand", [])
                if isinstance(hand, list) and len(hand) == 0:
                    return True
        return False

    def _is_draw(self, state: dict[str, Any]) -> bool:
        """Return True when the game ended as a draw (both decks empty, no winner)."""
        if self._phase_str(state) == "OVER":
            return False
        p_hp = int(state.get("playerHealth", 1))
        o_hp = int(state.get("opponentHealth", 1))
        if p_hp <= 0 or o_hp <= 0:
            return False
        p_deck = state.get("playerDeckCount")
        o_deck = state.get("opponentDeckCount")
        return (
            p_deck is not None and o_deck is not None
            and int(p_deck) == 0 and int(o_deck) == 0
        )

    def _is_resource_exhausted_loss(self, state: dict[str, Any]) -> bool:
        """Return True when the acting player ran out of deck AND hand cards.

        The acting player is the one whose perspective the state is rendered from.
        An empty deck + empty hand means they can never play again → treat as a
        loss for the acting player (not a draw).
        """
        if self._is_draw(state):
            return False  # both empty → draw, handled separately
        p_deck = state.get("playerDeckCount")
        if p_deck is None or int(p_deck) != 0:
            return False
        hand = state.get("playerHand", [])
        return isinstance(hand, list) and len(hand) == 0

    def _did_player_win(self, player_hp: int, opp_hp: int) -> bool:
        """Return True if the acting player won (lethal on opponent, self alive)."""
        return opp_hp <= 0 and player_hp > 0

    def _absolute_p1_seat_winner(
        self,
        state: dict[str, Any],
        *,
        exhausted_loss: bool = False,
        draw: bool = False,
    ) -> int:
        """Return winner in fixed seat numbering (0=P1, 1=P2, -1=draw/undecided)."""
        if draw:
            return -1
        if exhausted_loss:
            return 1 if self._acting_player_id == 1 else 0
        p1_hp, p2_hp = self._absolute_p1_p2_health(state)
        if p1_hp <= 0 and p2_hp <= 0:
            return -1
        if p2_hp <= 0 and p1_hp > 0:
            return 0
        if p1_hp <= 0 and p2_hp > 0:
            return 1
        return -1

    def _reset_repeat_tracking(self, *, turn_no: int, acting_player_id: int) -> None:
        self._repeat_tracker.reset(turn_no=turn_no, acting_player_id=acting_player_id)

    def _compute_repeat_action_penalty(
        self,
        action_key: tuple[int, str],
        *,
        turn_no: int,
        acting_player_id: int,
    ) -> float:
        """Penalize exact repeats and play-undo oscillation within one turn."""
        return self._repeat_tracker.update(
            action_key,
            turn_no=turn_no,
            acting_player_id=acting_player_id,
            threshold=self._repeat_action_threshold,
            penalty=self._repeat_action_penalty,
        )

    def _extract_legal_actions(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract all legal actions from the current game state.

        Each action dict contains:

        * ``action_code`` — Talishar mode integer
        * ``button_input`` — value for the ``buttonInput`` GET parameter
        * ``card_id``      — card identifier string (may be empty)
        * ``zone``         — source zone (``"hand"``, ``"equipment"``, …)
        * ``label``        — human-readable description
        """
        actions: list[dict[str, Any]] = []
        phase = _dp_get_phase(state)
        in_multi_choose = (
            phase.startswith("multichoose")
            or phase.startswith("maymultichoose")
            or phase in {"choosemultizone", "maychoosemultizone"}
        )

        zone_keys: list[tuple[str, str]] = [
            ("playerHand", "hand"),
            ("playerEquipment", "equipment"),
            ("playerArse", "arsenal"),
            ("playerAuras", "aura"),
            ("playerAllies", "ally"),
            ("playerItems", "item"),
            ("playerPermanents", "permanent"),
            ("playerDiscard", "discard"),
            ("playerBanish", "banish"),
        ]
        for zone_key, zone_name in zone_keys:
            for i, card in enumerate(state.get(zone_key, [])):
                if not isinstance(card, dict):
                    continue
                action_code = card.get("action", 0)
                if not action_code:
                    continue
                label = card.get("label") or card.get("cardNumber") or f"{zone_name}[{i}]"
                actions.append(
                    {
                        "action_code": int(action_code),
                        "button_input": str(
                            card.get("actionDataOverride", str(i))
                        ),
                        "card_id": str(card.get("cardNumber", "")),
                        "zone": zone_name,
                        "label": str(label),
                    }
                )

        # Prompt buttons (Pass, Cancel, Undo Block, etc.)
        prompt = state.get("playerPrompt", {})
        if isinstance(prompt, dict):
            for btn in prompt.get("buttons", []):
                if not isinstance(btn, dict):
                    continue
                mode = btn.get("mode", 0)
                if not mode:
                    continue
                # Talishar uses "caption" on button objects, not "label".
                btn_label = btn.get("caption") or btn.get("label") or f"btn_{mode}"
                actions.append(
                    {
                        "action_code": int(mode),
                        "button_input": str(btn.get("buttonInput", "")),
                        "card_id": "",
                        "zone": "button",
                        "label": str(btn_label),
                    }
                )

        # Player input popup (multi-choice, YesNo, choose-card, etc.)
        popup = state.get("playerInputPopUp", {})
        if isinstance(popup, dict) and popup.get("active", False):
            for btn in popup.get("buttons", []):
                if not isinstance(btn, dict):
                    continue
                mode = btn.get("mode", 0)
                if not mode:
                    continue
                # Talishar uses "caption" on button objects, not "label".
                btn_label = btn.get("caption") or btn.get("label") or f"btn_{mode}"
                actions.append(
                    {
                        "action_code": int(mode),
                        "button_input": str(btn.get("buttonInput", "")),
                        "card_id": "",
                        "zone": "popup",
                        "label": str(btn_label),
                    }
                )

            # Also extract selectable CARDS inside the popup (e.g. choosemultizone
            # "Choose which card to reveal for Fusion" — cards array has action codes).
            inner = popup.get("popup", {})
            if isinstance(inner, dict):
                for i, card in enumerate(inner.get("cards", [])):
                    if not isinstance(card, dict):
                        continue
                    card_action = card.get("action", 0)
                    if not card_action and in_multi_choose:
                        # MULTICHOOSE* popup cards often omit explicit `action`
                        # but are selectable via mode 16 + actionDataOverride.
                        card_action = 16
                    if not card_action:
                        continue
                    card_label = card.get("cardNumber") or card.get("label") or f"popup_card_{i}"
                    actions.append(
                        {
                            "action_code": int(card_action),
                            "button_input": str(card.get("actionDataOverride", str(i))),
                            "card_id": str(card.get("cardNumber", "")),
                            "zone": "popup",
                            "label": str(card_label),
                        }
                    )

            # Multi-select submit actions are exposed as popup.formOptions
            # (typically mode=19, caption="Submit").
            form_options = popup.get("formOptions", {})
            if isinstance(form_options, dict):
                form_mode = form_options.get("mode", 0)
                if form_mode:
                    form_caption = form_options.get("caption") or "Submit"
                    actions.append(
                        {
                            "action_code": int(form_mode),
                            "button_input": "",
                            "card_id": "",
                            "zone": "popup",
                            "label": str(form_caption),
                        }
                    )

            # MULTICHOOSETEXT options are emitted as checkboxes.
            for i, option in enumerate(popup.get("multiChooseText", [])):
                if not isinstance(option, dict):
                    continue
                opt_input = option.get("input", option.get("value", i))
                opt_label = option.get("label") or f"option_{opt_input}"
                actions.append(
                    {
                        "action_code": 16,
                        "button_input": str(opt_input),
                        "card_id": "",
                        "zone": "popup",
                        "label": str(opt_label),
                    }
                )

        # Guarantee a pass action only when Talishar accepts mode=99 here.
        # Injecting Pass during CanPassPhase=0 phases causes silent no-ops and
        # infinite agent loops (e.g. CHOOSEARSENAL, CHOOSEHAND, pitch windows).
        from .talishar_default_policy import can_pass_phase

        allow_pass = can_pass_phase(state)
        has_pass = any(
            int(a.get("action_code", 0)) in (99, 101, 105)
            or any(
                tok in str(a.get("label", "")).strip().lower()
                for tok in ("pass", "end turn", "no block", "skip")
            )
            for a in actions
        )
        if allow_pass and (not has_pass or not actions):
            actions.append(
                {
                    "action_code": 99,
                    "button_input": "",
                    "card_id": "",
                    "zone": "button",
                    "label": "Pass",
                }
            )
        elif not allow_pass:
            actions = [
                a for a in actions
                if int(a.get("action_code", 0)) not in (99, 101, 105)
                and "pass" not in str(a.get("label", "")).strip().lower()
            ]

        return actions

    def _filter_legal_actions(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Delegate to :func:`legal_action_filter.filter_legal_actions`."""
        return filter_legal_actions(state, legal_actions)

    def _loop_guard_for_step(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> Any:
        turn_no = int(state.get("turnNo", 0) or 0)
        return self._loop_guard.check(
            state,
            legal_actions,
            turn_no=turn_no,
            acting_player_id=self._acting_player_id,
        )

    def _check_macro_stall(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> MacroStallResult:
        filtered = self._filter_legal_actions(state, legal_actions)
        p1_hp, p2_hp = self._absolute_p1_p2_health(state)
        return self._macro_stall_guard.observe(
            state,
            filtered,
            p1_hp=p1_hp,
            p2_hp=p2_hp,
        )

    def _macro_stall_info(self, result: MacroStallResult) -> dict[str, Any]:
        return {
            "macro_stall_truncated": bool(result.should_truncate),
            "macro_stall_reason": result.reason,
            "turns_without_damage": result.turns_without_damage,
            "pass_only_main_streak": result.pass_only_main_streak,
        }

    def _force_loop_guard_submission(
        self,
        legal_actions: list[dict[str, Any]],
        loop_guard: Any,
    ) -> tuple[int, str]:
        return resolve_forced_submission(legal_actions, loop_guard)

    def _force_pass_submission(
        self,
        legal_actions: list[dict[str, Any]],
    ) -> tuple[int, str]:
        pass_action = first_pass_action(legal_actions)
        if pass_action is not None:
            return (
                _dp_to_int(pass_action.get("action_code", 0)),
                str(pass_action.get("button_input", "") or ""),
            )
        return self._first_pass_action(legal_actions)

    def _index_for_loop_guard_action(
        self,
        legal_actions: list[dict[str, Any]],
        loop_guard: Any,
    ) -> int:
        mode, button_input = resolve_forced_submission(legal_actions, loop_guard)
        for index, action in enumerate(legal_actions):
            if (
                _dp_to_int(action.get("action_code", 0)) == mode
                and str(action.get("button_input", "") or "") == button_input
            ):
                return index
        pass_index = next(
            (i for i, a in enumerate(legal_actions) if _is_pass_action(a)),
            0,
        )
        return pass_index

    def _first_pass_action(
        self,
        legal_actions: list[dict[str, Any]],
    ) -> tuple[int, str]:
        for a in legal_actions:
            if _is_pass_action(a):
                return int(a.get("action_code", 99)), str(a.get("button_input", ""))
        if legal_actions:
            a0 = legal_actions[0]
            return int(a0.get("action_code", 99)), str(a0.get("button_input", ""))
        return 99, ""

    def _sanitize_revert_submission(
        self,
        mode: int,
        button_input: str,
        legal_actions: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> tuple[int, str]:
        """Never submit undo/cancel/revert for agents."""
        del state
        label = ""
        for action in legal_actions:
            if (
                _dp_to_int(action.get("action_code", 0)) == mode
                and str(action.get("button_input", "")) == button_input
            ):
                label = str(action.get("label", "") or "")
                break
        if not _is_revert_action(
            {
                "action_code": mode,
                "button_input": button_input,
                "label": label,
            }
        ):
            return mode, button_input
        return self._first_pass_action(legal_actions)

    def _is_pass_submission(self, mode: int, button_input: str) -> bool:
        if mode in {99, 101, 105}:
            return True
        return _is_pass_action(
            {
                "action_code": mode,
                "button_input": button_input,
                "label": "",
            }
        )

    def _recover_from_gamestate_revert_if_needed(
        self,
        state: dict[str, Any],
        *,
        submitted_mode: int,
        submitted_button: str,
    ) -> dict[str, Any]:
        """Force pass when Talishar reverts an invalid declaration to escape loops."""
        if self._using_cpp or not talishar_gamestate_revert_detected(state):
            return state
        if self._is_pass_submission(submitted_mode, submitted_button):
            return state
        if not state.get("havePriority", False) or player_must_wait(state):
            return state
        legal_actions = self._legal_actions(state)
        pass_mode, pass_button = self._force_pass_submission(legal_actions)
        if pass_mode == submitted_mode and pass_button == submitted_button:
            return state
        try:
            return self._submit_action_and_sync(pass_mode, pass_button)
        except RuntimeError:
            return state

    def _legal_actions(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Return filtered legal actions for *state* (single call-site helper)."""
        return self._filter_legal_actions(state, self._extract_legal_actions(state))

    def _fast_compatible_legal_actions(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return legal actions ordered like generated C++ ``fast_step_index``.

        Fast C++ training indexes playable hand cards by hand slot, then maps
        any remaining index to Pass.  Talishar exposes a richer, engine-native
        action list, so final eval must publish a compatible ordered subset for
        agents trained on the fast C++ action space.
        """
        by_button: dict[str, dict[str, Any]] = {
            str(action.get("button_input", "") or ""): action
            for action in legal_actions
            if str(action.get("zone", "") or "").lower() == "hand"
            and _dp_to_int(action.get("action_code", 0)) == 27
        }
        ordered: list[dict[str, Any]] = []
        for i, card in enumerate(state.get("playerHand", [])):
            if not isinstance(card, dict):
                continue
            button = str(card.get("actionDataOverride", str(i)))
            action = by_button.get(button)
            if action is not None:
                ordered.append(action)

        pass_actions = [action for action in legal_actions if _is_pass_action(action)]
        if pass_actions:
            ordered.append(pass_actions[0])

        if ordered:
            return ordered
        return legal_actions

    def _refresh_episode_contexts(self, *, first_player: int = 1) -> None:
        opp = self._opponent_deck_name or self._local_deck_name or "Ira"
        local = self._local_deck_name or "Ira"
        fp = 2 if int(first_player) == 2 else 1
        self._p1_episode_context = load_episode_context(
            self_deck_name=local,
            opponent_deck_name=opp,
            game_format=self._format,
            first_player=fp,
        )
        self._p2_episode_context = load_episode_context(
            self_deck_name=opp,
            opponent_deck_name=local,
            game_format=self._format,
            first_player=fp,
        )

    def _episode_context_for_acting_player(self, state: dict[str, Any]) -> Any:
        ctx = (
            self._p1_episode_context
            if self._acting_player_id == 1
            else self._p2_episode_context
        )
        if ctx is None:
            self._refresh_episode_contexts(
                first_player=_dp_to_int(state.get("firstPlayer"), 1),
            )
            ctx = (
                self._p1_episode_context
                if self._acting_player_id == 1
                else self._p2_episode_context
            )
        return ctx

    def _encode_observation(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> str:
        """Encode game state as a compact JSON observation string."""
        hand = [
            {
                "cardID": c.get("cardNumber", ""),
                "action": c.get("action", 0),
                "actionDataOverride": c.get("actionDataOverride", ""),
                "label": c.get("label", ""),
            }
            for c in state.get("playerHand", [])
            if isinstance(c, dict)
        ]
        obs: dict[str, Any] = {
            "actingPlayerID": self._acting_player_id,
            "selfPlay": self._self_play,
            "playerHealth": state.get("playerHealth", 0),
            "opponentHealth": state.get("opponentHealth", 0),
            "turnNo": state.get("turnNo", 0),
            "turnPhase": self._phase_str(state),
            "havePriority": state.get("havePriority", False),
            "playerHandSize": len(state.get("playerHand", [])),
            "opponentHandSize": len(state.get("opponentHand", [])),
            "playerDeckCount": state.get("playerDeckCount", 0),
            "opponentDeckCount": state.get("opponentDeckCount", 0),
            "playerPitchCount": state.get("playerPitchCount", 0),
            "playerHand": hand,
            "legalActions": [
                {"index": i, "label": a["label"], "zone": a["zone"]}
                for i, a in enumerate(legal_actions)
            ],
            "legal_actions": legal_actions,
            "obsSchemaVersion": PLAYER_OBS_SCHEMA_VERSION,
        }
        episode_ctx = self._episode_context_for_acting_player(state)
        self._last_observation_vec = player_observation_vector(
            obs,
            legal_actions,
            episode_context=episode_ctx,
            acting_player_id=self._acting_player_id,
            p1_health=(
                _dp_to_int(obs["playerHealth"])
                if self._acting_player_id == 1
                else _dp_to_int(obs["opponentHealth"])
            ),
            p2_health=(
                _dp_to_int(obs["opponentHealth"])
                if self._acting_player_id == 1
                else _dp_to_int(obs["playerHealth"])
            ),
            game_over=self._is_game_over(state),
            raw_talishar_state=state,
        )
        if self._cpp_obs_alignment:
            self._last_observation_vec = align_observation_for_cpp_training(
                self._last_observation_vec
            )
        obs["observationVec"] = player_observation_payload(self._last_observation_vec)
        return json.dumps(obs, separators=(",", ":"))

    def _render_player_id(self) -> int:
        """Player ID used for Talishar-FE rendering."""
        if self._frontend_player_id is not None:
            return self._frontend_player_id
        if self._self_play:
            return self._acting_player_id
        return 1

    def _frontend_game_url(self) -> Optional[str]:
        """Build a Talishar-FE URL that opens the live game board."""
        if not self._game_name:
            return None
        player_id = self._render_player_id()
        params: dict[str, str] = {
            "gameName": self._game_name,
            "playerID": str(player_id),
        }
        if self._render_mode == "rgb_array" and not self._enable_frontend_card_hover:
            params["disableCardHover"] = "1"
        auth_key = self._auth_key_for(player_id)
        if auth_key:
            params["authKey"] = auth_key
        query = urllib.parse.urlencode(params)
        if self._frontend_game_url_template:
            try:
                return self._frontend_game_url_template.format(
                    gameName=self._game_name,
                    playerID=player_id,
                    authKey=auth_key,
                )
            except Exception:
                # Fall back to built-in URL conventions.
                pass

        # Talishar-FE routes live games through /game/play (dev + production).
        return f"{self._frontend_url}/game/play?{query}"

    def _open_frontend(self) -> Optional[str]:
        """Open the Talishar-FE game board in the default browser once per URL."""
        url = self._frontend_game_url()
        if not url:
            return None
        if url != self._opened_frontend_url:
            webbrowser.open(url, new=0, autoraise=True)
            self._opened_frontend_url = url
        return url

    def _render_ansi(self) -> RenderResult:
        if not self._last_state:
            return RenderResult(mode="ansi", text="No game state available.")

        state = self._last_state
        phase = self._phase_str(state)
        role = f"P{self._acting_player_id}" if self._self_play else "P1"
        lines = [
            "=== Flesh and Blood (Talishar Engine) ===",
            f"Game: {self._game_name}  Turn: {state.get('turnNo', '?')}  Phase: {phase}",
            (
                f"You ({role}): {state.get('playerHealth', '?')} HP  |  "
                f"Opponent: {state.get('opponentHealth', '?')} HP"
            ),
        ]
        if self._self_play:
            lines.insert(2, f"Self-play — acting as player {self._acting_player_id}")
        lines.append(
            f"Hand: {len(state.get('playerHand', []))} cards  |  "
            f"Deck: {state.get('playerDeckCount', '?')} cards"
        )

        legal_actions = self._legal_actions(state)
        if legal_actions:
            lines.append("Legal actions:")
            for i, a in enumerate(legal_actions[:12]):
                lines.append(f"  [{i}] {a['label']} ({a['zone']})")
            if len(legal_actions) > 12:
                lines.append(f"  … and {len(legal_actions) - 12} more")

        fe_url = self._frontend_game_url()
        if fe_url:
            lines.append(f"Talishar FE: {fe_url}")

        return RenderResult(mode="ansi", text="\n".join(lines))

    def _parse_action(
        self,
        action: Any,
        legal_actions: list[dict[str, Any]],
    ) -> tuple[int, str]:
        """Parse an action value into ``(mode, button_input)``."""
        if isinstance(action, dict):
            mode = _dp_to_int(action.get("action_code", action.get("mode", 0)))
            button = str(action.get("button_input", action.get("buttonInput", "")) or "")
            for candidate in legal_actions:
                if (
                    _dp_to_int(candidate.get("action_code", 0)) == mode
                    and str(candidate.get("button_input", "") or "") == button
                ):
                    mode, button = self._maybe_prepare_multiselect_submit(
                        mode,
                        button,
                        legal_actions,
                    )
                    return self._coerce_progress_action(mode, button, legal_actions)
            if mode:
                return self._coerce_progress_action(mode, button, legal_actions)

        action_str = str(action).strip().lower()
        if action_str == "pass":
            return self._coerce_progress_action(99, "", legal_actions)
        try:
            idx = int(action_str)
            if 0 <= idx < len(legal_actions):
                a = legal_actions[idx]
                mode = int(a["action_code"])
                button = str(a.get("button_input", ""))
                mode, button = self._maybe_prepare_multiselect_submit(
                    mode,
                    button,
                    legal_actions,
                )
                return self._coerce_progress_action(mode, button, legal_actions)
        except (ValueError, TypeError):
            pass

        # Invalid action from agent — choose a progress-making fallback when
        # possible instead of blindly passing (which can no-op in chooser phases).
        if legal_actions:
            fb_mode, fb_btn = self._preferred_progress_action(legal_actions)
            fb_mode, fb_btn = self._maybe_prepare_multiselect_submit(
                fb_mode,
                fb_btn,
                legal_actions,
            )
            return self._coerce_progress_action(fb_mode, fb_btn, legal_actions)
        return 99, ""

    def _is_noop_pass_action(self, action: dict[str, Any]) -> bool:
        """Return True for pass/no-op actions that often stall chooser phases."""
        code = _dp_to_int(action.get("action_code", 0))
        if code != 99:
            return False
        btn = str(action.get("button_input", "") or "").strip()
        if btn:
            # mode=99 with non-empty button_input can be a real chooser decision.
            return False
        label = str(action.get("label", "") or "").strip().lower()
        return (
            label == ""
            or any(tok in label for tok in ("pass", "end turn", "skip", "decline", "no "))
        )

    def _preferred_progress_action(
        self,
        legal_actions: list[dict[str, Any]],
    ) -> tuple[int, str]:
        """Pick an action that is most likely to advance game state.

        In mandatory-choice phases, prefer non-no-op options over plain pass.
        """
        if not legal_actions:
            return 99, ""

        phase = _dp_get_phase(self._last_state)
        mandatory_choice_phase = phase in (
            _dp_choose_hand_phases | _dp_button_input_phases | _dp_popup_phases
        )

        if mandatory_choice_phase:
            # 1) Prefer any non-no-op action first.
            for a in legal_actions:
                if not self._is_noop_pass_action(a):
                    return int(a.get("action_code", 99)), str(a.get("button_input", ""))

            # 2) If all are mode=99, prefer one that carries input over empty-pass.
            for a in legal_actions:
                if _dp_to_int(a.get("action_code", 0)) == 99 and str(a.get("button_input", "") or "").strip():
                    return 99, str(a.get("button_input", ""))

        # Generic fallback: first non-revert legal action.
        for a in legal_actions:
            if not _is_revert_action(a):
                return int(a.get("action_code", 99)), str(a.get("button_input", ""))
        a0 = legal_actions[0]
        return int(a0.get("action_code", 99)), str(a0.get("button_input", ""))

    def _maybe_prepare_multiselect_submit(
        self,
        mode: int,
        button_input: str,
        legal_actions: list[dict[str, Any]],
    ) -> tuple[int, str]:
        """Prepare mode-19 submit payloads for MULTICHOOSE* phases.

        Talishar multi-select flows require mode=16 selections plus mode=19 with
        chkCount/chk{i} payload. We track selected inputs locally and auto-submit
        when possible to avoid pass/no-op loops.
        """
        phase = _dp_get_phase(self._last_state)
        in_multi_choose = (
            phase.startswith("multichoose")
            or phase.startswith("maymultichoose")
            or phase in {"choosemultizone", "maychoosemultizone"}
        )
        if not in_multi_choose:
            self._pending_chk_inputs = None
            return mode, button_input

        has_submit = any(_dp_to_int(a.get("action_code", 0)) == 19 for a in legal_actions)
        btn = str(button_input or "")

        if mode == 16 and btn:
            if btn in self._multi_select_inputs:
                self._multi_select_inputs = [x for x in self._multi_select_inputs if x != btn]
            else:
                self._multi_select_inputs.append(btn)
            if has_submit and self._multi_select_inputs:
                self._pending_chk_inputs = list(self._multi_select_inputs)
                return 19, ""

        if mode == 19:
            if not self._multi_select_inputs:
                for a in legal_actions:
                    if _dp_to_int(a.get("action_code", 0)) == 16:
                        seed = str(a.get("button_input", "") or "")
                        if seed:
                            self._multi_select_inputs.append(seed)
                            break
            self._pending_chk_inputs = list(self._multi_select_inputs)

        return mode, button_input

    def _coerce_progress_action(
        self,
        mode: int,
        button_input: str,
        legal_actions: list[dict[str, Any]],
    ) -> tuple[int, str]:
        """Replace no-op pass with a progress action when alternatives exist."""
        if not legal_actions:
            return mode, button_input

        phase = _dp_get_phase(self._last_state)
        mandatory_choice_phase = phase in (
            _dp_choose_hand_phases | _dp_button_input_phases | _dp_popup_phases
        )

        if not mandatory_choice_phase:
            return mode, button_input

        # If selected action is a plain pass (no input), but another option
        # exists, force a progress action to avoid endless chooser loops.
        if mode == 99 and not str(button_input or "").strip():
            fb_mode, fb_btn = self._preferred_progress_action(legal_actions)
            return fb_mode, fb_btn

        return mode, button_input

    # ── rlbridge interface ────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        # ── Fast-path: delegate entirely to C++ engine ────────────────────────
        if self._using_cpp:
            result = self._cpp_env.reset(seed=seed, options=options)  # type: ignore[union-attr]
            # Sync wrapper attributes so training code reading env._acting_player_id
            # (etc.) sees correct values even though the C++ env owns the state.
            self._acting_player_id = int(self._cpp_env._acting_player)  # type: ignore[union-attr]
            self._player_hp = int(self._cpp_env._gs.p1_health)          # type: ignore[union-attr]
            self._opp_hp    = int(self._cpp_env._gs.p2_health)          # type: ignore[union-attr]
            return result

        # Recycle the session's cookie store between episodes so connections
        # stay pooled (keep-alive) but stale cookies don't leak.
        self._session.cookies.clear()
        self._last_update = 0

        if self._verbose:
            print(f"    [talishar reset] POST CreateLocalGame.php  "
                  f"({self._local_deck_name} vs {self._opponent_deck_name})...",
                  flush=True)
        self._game_name, self._p1_auth_key, self._p2_auth_key = self._create_game()
        if self._verbose:
            print(f"    [talishar reset] game created: {self._game_name}", flush=True)

        self._acting_player_id = 1
        if self._verbose:
            print(f"    [talishar reset] GET Start.php...", flush=True)
        started_key = self._start_game(self._game_name)
        if self._verbose:
            print(f"    [talishar reset] Start.php done", flush=True)
        if started_key:
            self._p1_auth_key = started_key
        self._auth_key = self._p1_auth_key

        # Wait for initial state with priority (equipment selection or main phase)
        if self._verbose:
            print(f"    [talishar reset] GET GetNextTurn.php (syncing priority)...",
                  flush=True)
        if self._self_play:
            self._last_state = self._sync_acting_player()
        else:
            self._last_state = self._poll_until_priority()
        if self._verbose:
            print(f"    [talishar reset] game ready — "
                  f"P{self._acting_player_id} has priority  "
                  f"phase: {self._phase_str(self._last_state)!r}",
                  flush=True)
        self._player_hp = int(self._last_state.get("playerHealth", 20))
        self._opp_hp = int(self._last_state.get("opponentHealth", 20))
        self._steps = 0
        self._deck_nonzero_ever_seen = False
        self._loop_guard.reset()
        self._macro_stall_guard.reset()
        self._multi_select_inputs = []
        self._pending_chk_inputs = None
        self._reset_repeat_tracking(
            turn_no=int(self._last_state.get("turnNo", 0) or 0),
            acting_player_id=self._acting_player_id,
        )
        self._refresh_episode_contexts(
            first_player=_dp_to_int(self._last_state.get("firstPlayer"), 1),
        )
        self._initialized = True
        self._opened_frontend_url = None
        if self._render_mode == "human":
            self._open_frontend()
        elif self._render_mode == "rgb_array":
            self._open_playwright_page()

        legal_actions = self._legal_actions(self._last_state)
        obs = self._encode_observation(self._last_state, legal_actions)

        if self._enable_combat_tracker:
            self._combat_tracker.reset(
                initial_snapshot=self._tracker_state_snapshot(self._last_state, legal_actions),
                initial_legal_actions=[self._tracker_action_dict(a) for a in legal_actions],
                combat_log_lines=extract_talishar_chat_log_lines(self._last_state.get("chatLog", "")),
                metadata={
                    "engine": "talishar_http",
                    "game_name": self._game_name,
                    "self_play": self._self_play,
                },
            )
        else:
            self._combat_tracker.clear()

        reset_info: dict[str, Any] = {
                "game_name": self._game_name,
                "legal_actions": legal_actions,
                "player_hp": self._player_hp,
                "opponent_hp": self._opp_hp,
                "acting_player_id": self._acting_player_id,
                "self_play": self._self_play,
                "combat_tracker": self._tracker_stub(),
        }
        if self._last_observation_vec is not None:
            reset_info["observation_vec"] = self._last_observation_vec
        return ResetResult(
            observation=obs,
            info=reset_info,
        )

    def step(self, action: Any) -> StepResult:
        # ── Fast-path: delegate entirely to C++ engine ────────────────────────
        if self._using_cpp:
            result = self._cpp_env.step(action)  # type: ignore[union-attr]
            # Sync wrapper attributes after each step so training code reading
            # env._acting_player_id sees the correct (updated) value.
            self._acting_player_id = int(self._cpp_env._acting_player)  # type: ignore[union-attr]
            self._player_hp = int(self._cpp_env._gs.p1_health)          # type: ignore[union-attr]
            self._opp_hp    = int(self._cpp_env._gs.p2_health)          # type: ignore[union-attr]
            return result

        state = self._last_state
        if not self._is_game_over(state):
            prior_acting = self._acting_player_id
            if not state.get("havePriority", False) or player_must_wait(state):
                state = self._ensure_acting_priority()
                if (
                    (not state.get("havePriority", False) or player_must_wait(state))
                    and self._self_play
                ):
                    state = self._wait_for_any_priority(
                        max_polls=_PRIORITY_STEP_SYNC_POLLS,
                    )
            if player_must_wait(state):
                return self._return_priority_resync(state)
            if self._acting_player_id != prior_acting:
                return self._return_priority_resync(state)
            self._last_state = state

        legal_actions = self._legal_actions(state)
        before_snapshot = self._tracker_state_snapshot(state, legal_actions)

        loop_guard = self._loop_guard_for_step(state, legal_actions)

        mode, button_input = self._parse_action(action, legal_actions)
        if loop_guard.force_pass:
            mode, button_input = self._force_loop_guard_submission(
                legal_actions,
                loop_guard,
            )
        mode, button_input = self._sanitize_revert_submission(
            mode,
            button_input,
            legal_actions,
            state,
        )
        tracker_action = self._tracker_action_for_submission(
            legal_actions,
            mode,
            button_input,
        )

        if self._verbose:
            print(f"    [step {self._steps + 1}] P{self._acting_player_id} "
                  f"mode={mode} btn={button_input!r}  "
                  f"→ POST ProcessInput.php...", flush=True)
        try:
            new_state = self._submit_action_and_sync(mode, button_input)
        except RuntimeError as exc:
            msg = str(exc)
            if "ProcessInput.php" not in msg and "RLStep.php" not in msg and "non-JSON response" not in msg:
                raise
            try:
                new_state = self._submit_action_and_sync(99, "")
            except Exception:
                new_state = self._sync_after_action()
        new_state = self._recover_from_gamestate_revert_if_needed(
            new_state,
            submitted_mode=mode,
            submitted_button=button_input,
        )

        if self._verbose:
            print(f"    [step {self._steps + 1}] action submitted, "
                  f"waiting for priority...", flush=True)
        if self._verbose:
            print(f"    [step {self._steps + 1}] priority → P{self._acting_player_id}  "
                  f"phase={self._phase_str(new_state)!r}  "
                  f"hp={new_state.get('playerHealth','?')}/{new_state.get('opponentHealth','?')}  "
                  f"terminated={self._is_game_over(new_state)}", flush=True)
        new_player_hp = int(new_state.get("playerHealth", self._player_hp))
        new_opp_hp = int(new_state.get("opponentHealth", self._opp_hp))
        self._steps += 1

        terminated = self._is_game_over(new_state)
        truncated = not terminated and self._steps >= self._max_turns

        new_legal_raw = self._legal_actions(new_state)
        macro_stall = self._check_macro_stall(new_state, new_legal_raw)
        if macro_stall.should_truncate and not terminated:
            truncated = True

        action_key = (mode, button_input)
        turn_no = int(new_state.get("turnNo", 0) or 0)
        if terminated or truncated:
            self._reset_repeat_tracking(
                turn_no=turn_no,
                acting_player_id=self._acting_player_id,
            )
            repeat_penalty = 0.0
        else:
            repeat_penalty = self._compute_repeat_action_penalty(
                action_key,
                turn_no=turn_no,
                acting_player_id=self._acting_player_id,
            )

        if terminated:
            won = self._did_player_win(new_player_hp, new_opp_hp)
            draw = self._is_draw(new_state)
            exhausted_loss = self._is_resource_exhausted_loss(new_state)
            if draw:
                reward = 0.0
            elif exhausted_loss:
                # Acting player ran out of cards — they lose.
                # Reward is from P1's perspective: loss if P1 was acting, win if P2 was.
                reward = -1.0 if self._acting_player_id == 1 else 1.0
            else:
                reward = 1.0 if won else -1.0
        elif truncated:
            reward = self._truncation_penalty
        else:
            dmg_dealt = max(0, self._opp_hp - new_opp_hp)
            dmg_taken = max(0, self._player_hp - new_player_hp)
            scale = self._damage_reward_scale
            reward = dmg_dealt * scale - dmg_taken * scale + self._step_penalty
        reward += repeat_penalty

        self._player_hp = new_player_hp
        self._opp_hp = new_opp_hp
        self._last_state = new_state

        new_phase = _dp_get_phase(new_state)
        if not (
            new_phase.startswith("multichoose")
            or new_phase.startswith("maymultichoose")
            or new_phase in {"choosemultizone", "maychoosemultizone"}
        ):
            self._multi_select_inputs = []
            self._pending_chk_inputs = None

        new_legal_actions = self._legal_actions(new_state)
        obs = self._encode_observation(new_state, new_legal_actions)

        tracker_event: Optional[dict[str, Any]] = None
        if self._enable_combat_tracker:
            tracker_event = self._combat_tracker.record_step(
                before_snapshot=before_snapshot,
                after_snapshot=self._tracker_state_snapshot(new_state, new_legal_actions),
                action=tracker_action,
                legal_before=[self._tracker_action_dict(a) for a in legal_actions],
                legal_after=[self._tracker_action_dict(a) for a in new_legal_actions],
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                combat_log_lines=extract_talishar_chat_log_lines(new_state.get("chatLog", "")),
            )

        step_info: dict[str, Any] = {
                "legal_actions": new_legal_actions,
                "turn": new_state.get("turnNo", 0),
                "player_hp": new_player_hp,
                "opponent_hp": new_opp_hp,
                "acting_player_id": self._acting_player_id,
                "self_play": self._self_play,
                "repeat_streak": self._repeat_tracker.repeat_streak,
                "repeat_penalty": repeat_penalty,
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

    def _open_playwright_page(self) -> None:
        """
        Spawn a dedicated Playwright worker thread that owns the browser for its
        entire lifetime.  All browser calls (open + screenshots) are queued to
        that thread so Playwright's single-thread constraint is satisfied even
        when the caller runs inside asyncio.
        """
        url = self._frontend_game_url()
        if not url:
            return
        try:
            from playwright.sync_api import sync_playwright as _sync_playwright
        except ImportError:
            import warnings
            warnings.warn(
                "playwright is not installed – rgb_array rendering will fall back to text.\n"
                "Fix: pip install playwright && playwright install chromium",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        import queue as _queue
        import threading as _threading

        self._close_playwright_page()

        # cmd_queue: (fn, result_event, result_box)  fn=None → shutdown
        cmd_queue: _queue.Queue = _queue.Queue()
        self._pw_cmd_queue = cmd_queue

        ready = _threading.Event()
        error_box: list[Exception] = []

        def _worker() -> None:
            try:
                pw = _sync_playwright().start()
                browser = pw.chromium.launch(headless=self._playwright_headless)
                ctx = browser.new_context(
                    viewport={"width": self._render_width, "height": self._render_height}
                )
                page = ctx.new_page()
                backend_base = (self._base_url or "").rstrip("/")
                if backend_base:

                    def _proxy_backend_route(route: Any) -> None:
                        target = rewrite_frontend_api_url(route.request.url, backend_base)
                        if target is None:
                            route.continue_()
                            return
                        route.continue_(url=target)

                    page.route("**/*", _proxy_backend_route)

                init_script = _PLAYWRIGHT_GDPR_INIT_SCRIPT
                if self._render_mode == "rgb_array" and not self._enable_frontend_card_hover:
                    init_script += _PLAYWRIGHT_DISABLE_CARD_HOVER_INIT_SCRIPT
                elif self._enable_frontend_card_hover:
                    init_script += (
                        f"localStorage.removeItem('{_DISABLE_CARD_HOVER_STORAGE_KEY}');"
                    )
                page.add_init_script(init_script)
                page.goto(url, timeout=20000)
                page.wait_for_load_state("domcontentloaded", timeout=15000)
                page.wait_for_timeout(5000)
                for consent_label in (
                    "Accept All Cookies",
                    "Essential Only",
                    "Agree",
                ):
                    try:
                        btn = page.locator("button", has_text=consent_label).first
                        if btn.is_visible(timeout=500):
                            btn.click()
                            page.wait_for_timeout(1500)
                            break
                    except Exception:
                        pass
                try:
                    page.wait_for_url("**/game/play/**", timeout=15000)
                except Exception:
                    pass
                self._pw_page = page
                ready.set()
                # Process commands from other threads
                while True:
                    item = cmd_queue.get()
                    if item is None:
                        break
                    fn, done_event, result_box = item
                    try:
                        result_box.append(fn(page))
                    except Exception:
                        pass
                    finally:
                        done_event.set()
                browser.close()
                pw.stop()
            except Exception as exc:
                error_box.append(exc)
                ready.set()

        t = _threading.Thread(target=_worker, daemon=True)
        t.start()
        self._pw_worker_thread = t
        ready.wait(timeout=35)
        if error_box:
            import warnings
            warnings.warn(
                f"Playwright browser failed to launch – rgb_array rendering will fall back to text.\n"
                f"Error: {error_box[0]}\n"
                f"Make sure Chromium is installed: playwright install chromium",
                RuntimeWarning,
                stacklevel=2,
            )
            self._close_playwright_page()

    def _close_playwright_page(self) -> None:
        """Shut down the Playwright worker thread."""
        q = getattr(self, "_pw_cmd_queue", None)
        if q is not None:
            try:
                q.put_nowait(None)  # signal worker to exit
            except Exception:
                pass
        t = getattr(self, "_pw_worker_thread", None)
        if t is not None:
            t.join(timeout=5)
        self._pw_page = None
        self._pw_cmd_queue = None
        self._pw_worker_thread = None
        self._pw_browser = None
        self._pw_playwright = None
        self._last_action_overlay_key = None

    def _run_playwright(self, fn: Any, *, timeout: float = 12.0) -> Any:
        q = getattr(self, "_pw_cmd_queue", None)
        if q is None or self._pw_page is None:
            return None
        import threading

        result_box: list[Any] = []
        done = threading.Event()
        q.put((fn, done, result_box))
        done.wait(timeout=timeout)
        return result_box[0] if result_box else None

    def update_frontend_action_overlay(
        self,
        hints: list[ActionCoachHint],
        *,
        state_key: Optional[str] = None,
    ) -> None:
        """Paint action coaching hints on the live Talishar frontend."""
        if self._render_mode != "rgb_array":
            return
        if state_key is not None and state_key == self._last_action_overlay_key:
            return
        payload = overlay_hints_payload(hints)
        script = playwright_update_overlay_script()

        def _apply(page: Any, payload_json: str) -> None:
            page.evaluate(f"({script})", json.loads(payload_json))

        self._run_playwright(lambda page: _apply(page, payload), timeout=8.0)
        self._last_action_overlay_key = state_key

    def clear_frontend_action_overlay(self) -> None:
        """Remove coaching highlights from the Talishar frontend."""
        self._last_action_overlay_key = None
        script = playwright_update_overlay_script()
        self._run_playwright(lambda page: page.evaluate(f"({script})", []), timeout=5.0)

    def close(self) -> None:
        if self._using_cpp:
            self._cpp_env.close()  # type: ignore[union-attr]
            self._cpp_env = None
            self._combat_tracker.clear()
            return
        self._close_playwright_page()
        try:
            self._session.close()
        except Exception:
            pass
        self._game_name = None
        self._auth_key = ""
        self._p1_auth_key = ""
        self._p2_auth_key = ""
        self._acting_player_id = 1
        self._last_state = {}
        self._initialized = False
        self._opened_frontend_url = None
        self._last_action_overlay_key = None
        self._combat_tracker.clear()

    @property
    def observation_space(self) -> TextSpace:
        return TextSpace(min_length=0, max_length=32_000)

    @property
    def action_space(self) -> TextSpace:
        return TextSpace(min_length=1, max_length=64)

    def _render_rgb_array(self) -> RenderResult:
        """Queue a screenshot request to the Playwright worker thread."""
        q = getattr(self, "_pw_cmd_queue", None)
        if q is None or self._pw_page is None:
            return RenderResult(mode="rgb_array")
        import base64, threading
        result_box: list[bytes] = []
        done = threading.Event()

        def _shot(page: Any) -> bytes:
            # Give the frontend a bit more time to load card/equipment images
            # after state updates. 800ms was observed to be too short on some
            # machines / network conditions; increase to 1500ms.
            page.wait_for_timeout(1500)
            page.mouse.move(8, 8)
            try:
                page.evaluate(_PLAYWRIGHT_PREPARE_CAPTURE_SCRIPT)
            except Exception:
                pass
            return page.screenshot(full_page=False)

        q.put((_shot, done, result_box))
        done.wait(timeout=12)
        if not result_box:
            return RenderResult(mode="rgb_array")
        b64 = base64.b64encode(result_box[0]).decode()
        return RenderResult(
            mode="rgb_array",
            data=b64,
            width=self._render_width,
            height=self._render_height,
        )

    def render(self) -> RenderResult:
        if self._render_mode == "human":
            if not self._last_state:
                return RenderResult(mode="human", text="No game state available.")
            url = self._open_frontend()
            if not url:
                return RenderResult(mode="human", text="No active game.")
            return RenderResult(mode="human", text=url)

        if self._render_mode == "rgb_array":
            return self._render_rgb_array()

        if self._render_mode != "ansi":
            return RenderResult(mode=self._render_mode or "none", text="")

        return self._render_ansi()

    def sample_action(self) -> str:
        """Return a heuristic legal action index as a string."""
        # ── Fast-path: delegate to C++ engine ─────────────────────────────────
        if self._using_cpp:
            return self._cpp_env.sample_action()  # type: ignore[union-attr]

        if not self._last_state:
            return "pass"
        legal = self._legal_actions(self._last_state)
        if not legal:
            return "pass"

        loop_guard = self._loop_guard_for_step(self._last_state, legal)
        if loop_guard.force_pass:
            return str(self._index_for_loop_guard_action(legal, loop_guard))

        idx = choose_talishar_action_index(
            legal,
            self._last_state,
            max_pitch_value=self._block_max_pitch_value,
            min_resource_cost=self._block_min_resource_cost,
        )
        return str(idx)


def parse_acting_player_id(env: Any, obs: Any) -> int:
    """Return the player ID that should act next (1 or 2)."""
    if hasattr(env, "_acting_player_id"):
        return int(getattr(env, "_acting_player_id", 1) or 1)
    if isinstance(obs, str):
        try:
            obs = json.loads(obs)
        except json.JSONDecodeError:
            return 1
    if isinstance(obs, dict):
        return int(obs.get("actingPlayerID", 1) or 1)
    return 1


def _eval_agent_action(agent: Any, obs: Any) -> Any:
    if hasattr(agent, "act_greedy"):
        return agent.act_greedy(obs)
    return agent.act(obs)


def try_create_cpp_eval_environment(
    *,
    base_url: str,
    game_format: str,
    lookup_deck1: str,
    lookup_deck2: str,
    cpp_engine_dir: Optional[str] = None,
    cpp_engine_cache_dir: Optional[str] = None,
    max_turns: int = 60,
    use_cpp_engine: bool = True,
) -> Optional["TalisharEngineEnvironment"]:
    """Return a C++-backed :class:`TalisharEngineEnvironment` when one is available.

    When *cpp_engine_dir* is omitted, searches ``results/cpp_engines/`` for a
    compiled module matching *lookup_deck1* vs *lookup_deck2*.  Returns ``None``
    when no engine is importable (caller should fall back to HTTP Talishar).
    """
    if not use_cpp_engine or not _CPP_ENGINE_SUPPORT:
        return None

    kwargs: dict[str, Any] = {
        "base_url": base_url,
        "game_format": game_format,
        "local_deck_name": lookup_deck1,
        "opponent_deck_name": lookup_deck2,
        "max_turns": max_turns,
        "self_play": True,
        "use_cpp_engine": True,
        "cpp_engine_cache_dir": cpp_engine_cache_dir,
        "cpp_engine_deck1": lookup_deck1,
        "cpp_engine_deck2": lookup_deck2,
    }
    if cpp_engine_dir is not None:
        kwargs["cpp_engine_dir"] = cpp_engine_dir

    try:
        env = TalisharEngineEnvironment(**kwargs)
    except RuntimeError:
        return None

    if env._using_cpp:
        return env
    env.close()
    return None


def run_matchup_win_rate_eval(
    env: Any,
    num_games: int,
    *,
    eval_p1_agent: Optional[Any] = None,
    eval_p2_agent: Optional[Any] = None,
    max_steps: int = 60,
    deck_player_id: int = 1,
) -> float:
    """Play *num_games* on *env* and return deck-player win rate in ``[0.0, 1.0]``."""
    if num_games <= 0:
        return 0.5

    wins = 0
    p1_policy = eval_p1_agent
    p2_policy = eval_p2_agent
    for _ in range(num_games):
        try:
            if p1_policy is not None:
                out = run_talishar_eval_episode(
                    env,
                    p1_policy,
                    max_steps=max_steps,
                    p2_agent=p2_policy,
                    deck_player_id=deck_player_id,
                )
                if out.get("deck_player_won") is True:
                    wins += 1
            else:
                reset_result = env.reset()
                obs_data = json.loads(reset_result.observation)
                step_result: Optional[StepResult] = None
                done = False
                while not done:
                    step_result = env.step(env.sample_action())
                    done = step_result.terminated or step_result.truncated
                    if not done:
                        obs_data = json.loads(step_result.observation)
                if talishar_deck_player_won(
                    obs_data,
                    deck_player_id=deck_player_id,
                    terminated=bool(
                        step_result is not None and step_result.terminated
                    ),
                    truncated=bool(
                        step_result is not None and step_result.truncated
                    ),
                ):
                    wins += 1
        except Exception:  # noqa: BLE001
            continue
    return wins / num_games


def run_talishar_eval_episode(
    env: Any,
    p1_agent: Any,
    max_steps: int,
    seed: Optional[int] = None,
    *,
    p2_agent: Optional[Any] = None,
    deck_player_id: int = 1,
) -> dict[str, Any]:
    """Play one evaluation episode with trained agents on both sides (self-play).

    *p1_agent* controls player 1; *p2_agent* controls player 2 (defaults to
    *p1_agent* when omitted, i.e. one policy for both sides).  Requires a
    self-play Talishar environment.
    """
    policy_p2 = p2_agent if p2_agent is not None else p1_agent
    reset_out = env.reset(seed=seed)
    obs = (
        reset_out.observation
        if hasattr(reset_out, "observation")
        else reset_out.get("observation", reset_out)
    )
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    step_result: Any = None

    for step in range(1, max_steps + 1):
        acting = parse_acting_player_id(env, obs)
        policy = p1_agent if acting == 1 else policy_p2
        action = _eval_agent_action(policy, obs)
        step_result = env.step(action)
        obs = (
            step_result.observation
            if hasattr(step_result, "observation")
            else step_result.get("observation", obs)
        )
        reward = float(
            step_result.reward
            if hasattr(step_result, "reward")
            else step_result.get("reward", 0.0)
        )
        terminated = bool(
            step_result.terminated
            if hasattr(step_result, "terminated")
            else step_result.get("terminated", False)
        )
        truncated = bool(
            step_result.truncated
            if hasattr(step_result, "truncated")
            else step_result.get("truncated", False)
        )
        total_reward += reward
        steps = step
        if terminated or truncated:
            break

    return {
        "steps": steps,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "timed_out": truncated and not terminated,
        "final_observation": obs,
        "deck_player_id": deck_player_id,
        "deck_player_won": talishar_deck_player_won(
            obs,
            deck_player_id=deck_player_id,
            terminated=terminated,
            truncated=truncated,
        ),
    }


def talishar_deck_player_won(
    obs: Any,
    *,
    deck_player_id: int = 1,
    terminated: bool = False,
    truncated: bool = False,
) -> Optional[bool]:
    """Whether *deck_player_id* won, from a Talishar JSON observation.

    Returns:
        ``True``  — deck player won (opponent HP ≤ 0, own HP > 0)
        ``False`` — deck player lost (own HP ≤ 0, opponent HP > 0)
        ``None``  — draw (both at ≤ 0 HP) or timeout (no lethal HP)

    Lethal-HP rules match ``classify_p1_episode_outcome`` in play training.
    For deck-swap eval, map the seat outcome to nominal heroes via
    ``play_outcome_stats.OutcomeCounters`` instead of fixed ``deck_player_id=1``.
    """
    if truncated and not terminated:
        return None

    if isinstance(obs, str):
        try:
            obs = json.loads(obs)
        except json.JSONDecodeError:
            return None
    if not isinstance(obs, dict):
        return None

    player_hp = obs.get("playerHealth")
    opponent_hp = obs.get("opponentHealth")
    if player_hp is None or opponent_hp is None:
        return None

    p_player = float(player_hp or 0)
    p_opponent = float(opponent_hp or 0)
    acting = int(obs.get("actingPlayerID", deck_player_id) or deck_player_id)
    if acting != deck_player_id:
        p_player, p_opponent = p_opponent, p_player

    if p_player <= 0 and p_opponent <= 0:
        return None
    if p_opponent <= 0:
        return True
    if p_player <= 0:
        return False
    return None
