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

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from rlbridge.environments.base import rlbridgeEnvironment
from rlbridge.protocol.messages import RenderResult, ResetResult, StepResult, TextSpace

from .combat_log_tracker import CombatTurnTracker, extract_talishar_chat_log_lines
from .legal_action_filter import filter_legal_actions
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
_TRUNCATION_PENALTY = -0.1  # negative reward for hitting max_turns without a winner
_STEP_PENALTY = -0.001  # small per-step penalty to encourage faster game completion
_DISABLE_CARD_HOVER_STORAGE_KEY = "talishar-disable-card-hover"

_PLAYWRIGHT_GDPR_INIT_SCRIPT = (
    "localStorage.setItem('gdpr-analytics-enabled','true');"
    "localStorage.setItem('gdpr-consent-accepted','true');"
)

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
        When ``True`` (default), capture per-step combat traces and board-state
        action statistics for debugging and parity checks.
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
        render_mode: Optional[str] = None,
        local_deck_name: Optional[str] = "Ira",
        opponent_deck_name: Optional[str] = None,
        self_play: bool = True,
        render_width: Optional[int] = None,
        render_height: Optional[int] = None,
        block_max_pitch_value: int = 3,
        block_min_resource_cost: int = 0,
        # C++ engine options ─────────────────────────────────────────────────
        use_cpp_engine: bool = True,
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
    ) -> None:
        self._base_url = (
            base_url or os.environ.get("TALISHAR_URL", "http://localhost")
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
        self._block_max_pitch_value = block_max_pitch_value
        self._block_min_resource_cost = block_min_resource_cost
        self._render_mode = render_mode
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

        # HTTP session with connection pooling and automatic retry on transient
        # server errors.  One persistent TCP connection is reused for all steps
        # in an episode (keep-alive), eliminating per-step TCP handshake cost.
        self._session: requests.Session = self._make_session()

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
        # Cycle-breaker for sample_action: tracks (legal_action_fingerprint, last_idx)
        self._sample_action_last_fp: Optional[str] = None
        self._sample_action_repeat: int = 0
        # Whether playerDeckCount > 0 has ever been observed this episode.
        # Guards deck-exhaustion game-over checks so they do NOT fire during the
        # pre-game equipment-selection phase, when Talishar returns
        # playerDeckCount=0 / empty playerHand before decks are shuffled.
        self._deck_nonzero_ever_seen: bool = False
        # Pitch-window "unaffordable card" loop prevention
        self._last_m_label: Optional[str] = None
        self._last_block_label: Optional[str] = None
        self._last_played_label: Optional[str] = None
        self._unaffordable_labels: set[str] = set()
        self._last_turn_no_for_unaffordable: int = -1
        # Multi-select popup tracking (mode=16 picks + mode=19 submit payload)
        self._multi_select_inputs: list[str] = []
        self._pending_chk_inputs: Optional[list[str]] = None

        # Persistent Playwright worker thread for rgb_array rendering
        self._pw_page: Any = None
        self._pw_browser: Any = None
        self._pw_playwright: Any = None
        self._pw_cmd_queue: Any = None
        self._pw_worker_thread: Any = None

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
                            deck1=lookup_deck1,
                            deck2=lookup_deck2,
                            enable_combat_tracker=self._enable_combat_tracker,
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

    # ── C++ engine delegation helpers ────────────────────────────────────────

    @property
    def _using_cpp(self) -> bool:
        """True when all environment calls are delegated to the C++ engine."""
        return self._cpp_env is not None

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

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _make_session(self) -> requests.Session:
        """Build a requests.Session with connection pooling, keep-alive, and retry."""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods={"GET", "POST"},
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            pool_connections=2,
            pool_maxsize=8,
            max_retries=retry,
        )
        session.mount("http://",  adapter)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": "TalisharRLEnv/1.0"})
        return session

    def _http_get(
        self,
        path: str,
        params: dict[str, str],
        _retries: int = 3,
        *,
        allow_empty_body: bool = False,
    ) -> dict[str, Any]:
        url = self._base_url + path
        last_exc: Exception = RuntimeError("no attempts")
        for attempt in range(_retries):
            try:
                if attempt > 0:
                    time.sleep(0.3 * (2 ** attempt))
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
        raise TalisharConnectionError(f"GET {url} failed: {last_exc}") from last_exc

    def _http_post_json(self, path: str, payload: dict[str, Any], _retries: int = 3) -> dict[str, Any]:
        url = self._base_url + path
        last_exc: Exception = RuntimeError("no attempts")
        for attempt in range(_retries):
            try:
                if attempt > 0:
                    time.sleep(0.3 * (2 ** attempt))
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

    def _sync_acting_player(self) -> dict[str, Any]:
        """Set ``_acting_player_id`` to whichever player currently has priority."""
        # Full snapshots per player — sharing ``_last_update`` across player IDs
        # returns empty deltas and breaks priority detection.
        for pid in (1, 2):
            probe = self._fetch_state(player_id=pid, last_update=0)
            if probe.get("havePriority", False) or self._is_game_over(probe):
                self._acting_player_id = pid
                self._auth_key = self._auth_key_for(pid)
                self._apply_last_update(probe)
                return probe
        return self._fetch_state(player_id=self._acting_player_id, last_update=0)

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
        max_polls: int = 60,
        interval: float = 0.15,
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

        Timeout: ``max_polls * interval`` seconds (default 9 s) before
        falling back to ``_sync_acting_player``.
        """
        for i in range(max_polls):
            error_count = 0
            for pid in (1, 2):
                state = self._fetch_state(player_id=pid, last_update=0)
                has_priority = state.get("havePriority", False)
                is_over = self._is_game_over(state)
                err_msg = state.get("error", "")
                # "gamestate too short" is a transient file-write race — keep polling.
                # Only count it as a hard error when it persists past the first few polls.
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
                    self._acting_player_id = pid
                    self._auth_key = self._auth_key_for(pid)
                    self._apply_last_update(state)
                    return state
                if err_msg and not is_transient_error:
                    error_count += 1
            # If BOTH players returned non-transient error states the game has
            # crashed — return immediately rather than burning the full timeout.
            if error_count == 2:
                if self._verbose:
                    print("    [wait] both players returned fatal error — "
                          "ending episode", flush=True)
                return {"error": "game_crashed"}
            if self._verbose and i > 0 and i % 5 == 0:
                print(f"    [wait] poll {i+1}/{max_polls}: no priority yet...",
                      flush=True)
            time.sleep(interval)
        if self._verbose:
            print(f"    [wait] timed out after {max_polls} polls — falling back to sync",
                  flush=True)
        # Fallback — should rarely be reached
        return self._sync_acting_player()

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
        """Return True if the agent (player 1) won the game."""
        return opp_hp <= 0 and player_hp > 0

    def _reset_repeat_tracking(self, *, turn_no: int, acting_player_id: int) -> None:
        self._repeat_tracker.reset(turn_no=turn_no, acting_player_id=acting_player_id)

    def _repeat_action_penalty(
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
        return filter_legal_actions(
            state,
            legal_actions,
            block_blacklist=frozenset(getattr(self, "_unaffordable_labels", set())),
        )

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

    def _legal_actions(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Return filtered legal actions for *state* (single call-site helper)."""
        return self._filter_legal_actions(state, self._extract_legal_actions(state))

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
        }
        return json.dumps(obs, separators=(",", ":"))

    def _render_player_id(self) -> int:
        """Player ID used for Talishar-FE rendering."""
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
        if self._render_mode == "rgb_array":
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

        parsed = urllib.parse.urlsplit(self._frontend_url)
        is_vite_dev = parsed.port == 5173
        if is_vite_dev:
            # Vite FE dev server generally serves from root and expects router/query handling client-side.
            return f"{self._frontend_url}/?{query}"
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
        self._sample_action_last_fp = None
        self._sample_action_repeat = 0
        self._last_m_label = None
        self._last_block_label = None
        self._last_played_label = None
        self._unaffordable_labels = set()
        self._last_turn_no_for_unaffordable = -1
        self._multi_select_inputs = []
        self._pending_chk_inputs = None
        self._reset_repeat_tracking(
            turn_no=int(self._last_state.get("turnNo", 0) or 0),
            acting_player_id=self._acting_player_id,
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

        return ResetResult(
            observation=obs,
            info={
                "game_name": self._game_name,
                "legal_actions": legal_actions,
                "player_hp": self._player_hp,
                "opponent_hp": self._opp_hp,
                "acting_player_id": self._acting_player_id,
                "self_play": self._self_play,
                "combat_tracker": self._tracker_stub(),
            },
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
        legal_actions = self._legal_actions(state)
        phase_before = _dp_get_phase(state)
        before_snapshot = self._tracker_state_snapshot(state, legal_actions)

        mode, button_input = self._parse_action(action, legal_actions)
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

        if phase_before in _dp_block_phases and mode == 27:
            chosen = next(
                (a for a in legal_actions
                 if _dp_to_int(a.get("action_code", 0)) == mode
                 and str(a.get("button_input", "")) == button_input),
                None,
            )
            if chosen is not None and chosen.get("zone") == "hand":
                self._last_block_label = str(chosen.get("label", "") or "")

        if phase_before != "p" and mode in (5, 27, 36, 37, 38):
            chosen = next(
                (a for a in legal_actions
                 if _dp_to_int(a.get("action_code", 0)) == mode
                 and str(a.get("button_input", "")) == button_input),
                None,
            )
            if chosen is not None:
                label = str(
                    chosen.get("label", "")
                    or chosen.get("card_id", "")
                    or ""
                )
                if label:
                    self._last_played_label = label

        if self._verbose:
            print(f"    [step {self._steps + 1}] P{self._acting_player_id} "
                  f"mode={mode} btn={button_input!r}  "
                  f"→ POST ProcessInput.php...", flush=True)
        try:
            self._submit_action(mode, button_input)
        except RuntimeError as exc:
            msg = str(exc)
            if "ProcessInput.php" not in msg or "non-JSON response" not in msg:
                raise
            # Defensive fallback for Talishar PHP warning pages on specific card
            # interactions: pass priority and continue the episode.
            try:
                self._submit_action(99, "")
            except Exception:
                pass

        if self._verbose:
            print(f"    [step {self._steps + 1}] ProcessInput.php done, "
                  f"waiting for priority...", flush=True)
        # Brief pause so Talishar has time to finish writing the gamestate file
        # before we poll GetNextTurn.php.  Without this a race condition causes
        # "ParseGamestate: gamestate too short" on the very first poll.
        time.sleep(0.35)
        if self._self_play:
            new_state = self._wait_for_any_priority()
        else:
            new_state = self._poll_until_priority()
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

        action_key = (mode, button_input)
        turn_no = int(new_state.get("turnNo", 0) or 0)
        if terminated or truncated:
            self._reset_repeat_tracking(
                turn_no=turn_no,
                acting_player_id=self._acting_player_id,
            )
            repeat_penalty = 0.0
        else:
            repeat_penalty = self._repeat_action_penalty(
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
            reward = _TRUNCATION_PENALTY
        else:
            dmg_dealt = max(0, self._opp_hp - new_opp_hp)
            dmg_taken = max(0, self._player_hp - new_player_hp)
            reward = dmg_dealt * 0.01 - dmg_taken * 0.01 + _STEP_PENALTY
        reward += repeat_penalty

        self._player_hp = new_player_hp
        self._opp_hp = new_opp_hp
        self._last_state = new_state

        if phase_before == "p" and mode in (99, 10000):
            for label in (
                self._last_block_label,
                self._last_m_label,
                self._last_played_label,
            ):
                if label:
                    self._unaffordable_labels.add(label)
            self._last_block_label = None
            self._last_m_label = None
            self._last_played_label = None

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

        return StepResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "legal_actions": new_legal_actions,
                "turn": new_state.get("turnNo", 0),
                "player_hp": new_player_hp,
                "opponent_hp": new_opp_hp,
                "acting_player_id": self._acting_player_id,
                "self_play": self._self_play,
                "repeat_streak": self._repeat_tracker.repeat_streak,
                "repeat_penalty": repeat_penalty,
                "combat_tracker": self._tracker_stub(tracker_event),
            },
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
                browser = pw.chromium.launch(headless=True)
                ctx = browser.new_context(
                    viewport={"width": self._render_width, "height": self._render_height}
                )
                page = ctx.new_page()
                init_script = _PLAYWRIGHT_GDPR_INIT_SCRIPT
                if self._render_mode == "rgb_array":
                    init_script += _PLAYWRIGHT_DISABLE_CARD_HOVER_INIT_SCRIPT
                page.add_init_script(init_script)
                page.goto(url, timeout=20000)
                page.wait_for_load_state("domcontentloaded", timeout=15000)
                page.wait_for_timeout(5000)
                try:
                    btn = page.locator("button", has_text="Agree").first
                    if btn.is_visible(timeout=500):
                        btn.click()
                        page.wait_for_timeout(1500)
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
        """Return a heuristic legal action index as a string.

        Includes a safety cycle-breaker: if the identical set of legal actions
        is presented 5 times in a row (same codes + labels) the policy is
        clearly looping, so pick a uniformly random legal action to break out.
        """
        # ── Fast-path: delegate to C++ engine ─────────────────────────────────
        if self._using_cpp:
            return self._cpp_env.sample_action()  # type: ignore[union-attr]

        if not self._last_state:
            return "pass"
        legal = self._legal_actions(self._last_state)
        if not legal:
            return "pass"

        # Fingerprint the legal-action set (codes + labels, order-stable)
        fp = "|".join(
            f"{a.get('action_code',0)}:{a.get('label','')}" for a in legal
        )
        if fp == self._sample_action_last_fp:
            self._sample_action_repeat += 1
        else:
            self._sample_action_last_fp = fp
            self._sample_action_repeat = 1

        if self._sample_action_repeat >= 4:
            # Stuck: policy keeps choosing the same action on the same state.
            # Pick a uniformly random legal action to break the loop.
            self._sample_action_repeat = 0
            import random as _random
            return str(_random.randrange(len(legal)))

        # Detect turn change — reset unaffordable card blacklist each turn.
        _turn_no = int(self._last_state.get("turnNo", 0) or 0)
        if _turn_no != self._last_turn_no_for_unaffordable:
            self._last_turn_no_for_unaffordable = _turn_no
            self._unaffordable_labels = set()

        # Extract current phase inline (mirrors _get_phase in the policy module).
        _tp = self._last_state.get("turnPhase", {})
        _phase = str(
            (_tp.get("turnPhase", "") if isinstance(_tp, dict) else _tp) or ""
        ).strip().lower()

        # ── Empty pitch window: abort with Pass and blacklist the card ────────
        # Cancel (10000) is Talishar undo and must never be submitted.  Pass also
        # reverts the pending play, so blacklist hand/arsenal labels first.
        if _phase == "p":
            _pitch_cards = [
                a for a in legal
                if a.get("zone") == "hand" and int(a.get("action_code", 0)) == 27
            ]
            if not _pitch_cards:
                _abort_label = (
                    self._last_block_label
                    or self._last_m_label
                    or self._last_played_label
                )
                if _abort_label:
                    self._unaffordable_labels.add(_abort_label)
                    self._last_block_label = None
                    self._last_m_label = None
                    self._last_played_label = None
                _pi = next(
                    (i for i, a in enumerate(legal) if int(a.get("action_code", 0)) == 99),
                    0,
                )
                return str(_pi)

        idx = choose_talishar_action_index(
            legal, self._last_state,
            unaffordable=frozenset(self._unaffordable_labels),
            max_pitch_value=self._block_max_pitch_value,
            min_resource_cost=self._block_min_resource_cost,
        )

        # Track the last card played so we can blacklist it after an empty pitch.
        if _phase != "p":
            _chosen = legal[idx] if 0 <= idx < len(legal) else None
            if _chosen is not None and int(_chosen.get("action_code", 0)) in (5, 27, 36, 37, 38):
                _label = str(
                    _chosen.get("label", "")
                    or _chosen.get("card_id", "")
                    or ""
                )
                if _label:
                    if (
                        int(_chosen.get("action_code", 0)) == 27
                        and _chosen.get("zone") == "hand"
                        and _phase == "m"
                    ):
                        self._last_m_label = _label
                    self._last_played_label = _label
                else:
                    self._last_m_label = None
            else:
                self._last_m_label = None

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
        ``True``  — deck player won (opponent HP ≤ 0)
        ``False`` — deck player lost (own HP ≤ 0)
        ``None``  — draw (both decks empty, equal HP) **or** timeout (max
                    steps hit without a winner — *not* treated as a draw).

    Callers that need to distinguish draw from timeout should check the
    ``"timed_out"`` key in the episode dict returned by
    :func:`run_talishar_eval_episode`.
    """
    # A timeout is not a decided outcome — return None immediately without
    # inspecting HP so callers cannot accidentally treat it as a draw.
    if truncated and not terminated:
        return None

    if isinstance(obs, str):
        try:
            obs = json.loads(obs)
        except json.JSONDecodeError:
            return None
    if not isinstance(obs, dict):
        return None

    acting = int(obs.get("actingPlayerID", deck_player_id) or deck_player_id)
    player_hp = float(obs.get("playerHealth", 0.0) or 0.0)
    opp_hp = float(obs.get("opponentHealth", 0.0) or 0.0)
    if acting != deck_player_id:
        player_hp, opp_hp = opp_hp, player_hp

    if terminated:
        if opp_hp <= 0 < player_hp:
            return True
        if player_hp <= 0 < opp_hp:
            return False
        if player_hp == opp_hp:
            return None   # genuine draw
    return None
