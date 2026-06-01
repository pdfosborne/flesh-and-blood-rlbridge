"""RL environment backed by a live Talishar server instance.

The environment wraps the Talishar HTTP API (https://github.com/Talishar/Talishar)
and exposes it as an rlbridge environment with text-based observations and actions.

Observations are JSON strings containing the current game state (life totals, hand,
legal actions, etc.).  Actions are integer strings indexing into the ``legalActions``
list in the latest observation, or the special string ``"pass"`` to pass priority.

Prerequisites
-------------
A running Talishar Docker instance is required.  Run once to set up::

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

Then set ``TALISHAR_URL=http://localhost`` or pass ``base_url`` to the constructor.

HTTP API summary
----------------
* ``POST /APIs/CreateGame.php``  — JSON body: ``{fabdb, format, deckTestMode, visibility}``
  Creates an AI practice game.  Response includes ``gameName`` and ``authKey``.
* ``GET  /Start.php``            — query: ``{gameName, playerID=1}``
  Initialises the gamestate file.  Returns ``{success, authKey}``.
* ``GET  /GetNextTurn.php``      — query: ``{gameName, playerID, authKey, lastUpdate}``
  Returns the full game-state JSON.
* ``GET  /ProcessInput.php``     — query: ``{gameName, playerID, authKey, mode, buttonInput}``
  Submits a player action.  After this call the server runs ``CombatDummyAI()``
  automatically for player 2, so we just poll until ``havePriority`` is true.
"""

from __future__ import annotations

import http.cookiejar
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from rlbridge.environments.base import rlbridgeEnvironment
from rlbridge.protocol.messages import RenderResult, ResetResult, StepResult, TextSpace

from .talishar_oracle import TalisharConnectionError

# Default practice deck: Ira Crimson Haze (young hero, blitz-legal)
_DEFAULT_DECK_LINK = "https://fabrary.net/decks/01GJG7Z4WGWSZ95FY74KX4M557"


class TalisharEngineEnvironment(rlbridgeEnvironment):
    """RL environment that uses a live Talishar server as its game engine.

    The agent plays as player 1 against the built-in Combat Dummy AI (player 2).
    After each action submitted by the agent the server automatically runs the AI
    for player 2 via ``CombatDummyAI()``.  The environment polls
    ``GetNextTurn.php`` until ``havePriority`` is ``true`` (or the game ends)
    before returning control to the agent.

    Parameters
    ----------
    base_url:
        Base URL of the Talishar server (e.g. ``"http://localhost"``).
        Defaults to the ``TALISHAR_URL`` environment variable or
        ``"http://localhost"``.
    deck_link:
        Fabrary/FaBDB deck link for the agent's deck.
    game_format:
        Game format string accepted by Talishar (e.g. ``"blitz"``, ``"cc"``).
    timeout:
        HTTP request timeout in seconds.
    max_turns:
        Maximum number of agent steps before the episode is truncated.
    render_mode:
        Rendering mode (``"ansi"`` or ``None``).
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        deck_link: str = _DEFAULT_DECK_LINK,
        game_format: str = "blitz",
        timeout: float = 15.0,
        max_turns: int = 1000,
        render_mode: Optional[str] = None,
        local_deck_name: Optional[str] = "Ira",
        opponent_deck_name: Optional[str] = None,
    ) -> None:
        self._base_url = (
            base_url or os.environ.get("TALISHAR_URL", "http://localhost")
        ).rstrip("/")
        self._deck_link = deck_link
        self._local_deck_name = local_deck_name  # use CreateLocalGame.php when set
        self._opponent_deck_name = opponent_deck_name
        self._format = game_format
        self._timeout = timeout
        self._max_turns = max_turns
        self._render_mode = render_mode

        # HTTP session (rebuilt each episode so cookies/sessions don't leak)
        self._cookie_jar: http.cookiejar.CookieJar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self._cookie_jar)
        )

        # Per-episode state
        self._game_name: Optional[str] = None
        self._auth_key: str = ""
        self._last_state: dict[str, Any] = {}
        self._last_update: int = 0
        self._steps: int = 0
        self._player_hp: int = 20
        self._opp_hp: int = 20
        self._initialized: bool = False

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _http_get(self, path: str, params: dict[str, str], _retries: int = 3) -> dict[str, Any]:
        url = self._base_url + path + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "TalisharRLEnv/1.0"})
        last_exc: Exception = RuntimeError("no attempts")
        for attempt in range(_retries):
            try:
                if attempt > 0:
                    time.sleep(2.0 * attempt)
                    self._cookie_jar = http.cookiejar.CookieJar()
                    self._opener = urllib.request.build_opener(
                        urllib.request.HTTPCookieProcessor(self._cookie_jar)
                    )
                with self._opener.open(req, timeout=self._timeout) as resp:  # noqa: S310
                    body = resp.read()
                data = json.loads(body)
                return data if isinstance(data, dict) else {"_raw": data}
            except json.JSONDecodeError:
                return {}
            except (urllib.error.URLError, OSError) as exc:
                last_exc = exc
        raise TalisharConnectionError(f"GET {url} failed: {last_exc}") from last_exc

    def _http_post_json(self, path: str, payload: dict[str, Any], _retries: int = 3) -> dict[str, Any]:
        url = self._base_url + path
        body = json.dumps(payload).encode()
        last_exc: Exception = RuntimeError("no attempts")
        for attempt in range(_retries):
            try:
                if attempt > 0:
                    time.sleep(2.0 * attempt)
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json", "User-Agent": "TalisharRLEnv/1.0"},
                    method="POST",
                )
                with self._opener.open(req, timeout=self._timeout) as resp:  # noqa: S310
                    resp_body = resp.read()
                data = json.loads(resp_body)
                return data if isinstance(data, dict) else {"_raw": data}
            except json.JSONDecodeError:
                return {}
            except (urllib.error.URLError, OSError) as exc:
                last_exc = exc
        raise TalisharConnectionError(f"POST {url} failed: {last_exc}") from last_exc

    # ── Game lifecycle ────────────────────────────────────────────────────────

    def _create_game(self) -> tuple[str, str]:
        """Create an AI practice game.

        Uses ``CreateLocalGame.php`` (no external deck API) when
        ``local_deck_name`` is set, otherwise falls back to ``CreateGame.php``
        with the configured ``deck_link``.

        Returns
        -------
        (game_name, auth_key)
        """
        if self._local_deck_name:
            payload: dict[str, Any] = {
                "deckName": self._local_deck_name,
                "format": self._format,
                "visibility": "private",
            }
            if self._opponent_deck_name:
                payload["opponentDeckName"] = self._opponent_deck_name
            endpoint = "/APIs/CreateLocalGame.php"
        else:
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
        auth_key = str(resp.get("authKey", ""))
        if not game_name:
            raise RuntimeError(
                f"CreateGame returned no gameName.  Full response: {resp}"
            )
        return game_name, auth_key

    def _start_game(self, game_name: str) -> str:
        """Call Start.php to write the initial gamestate file.

        Returns the (possibly updated) auth key.
        """
        resp = self._http_get("/Start.php", {"gameName": game_name, "playerID": "1"})
        # Start.php returns {"success": true, "authKey": "..."}
        returned_key = resp.get("authKey", "")
        return str(returned_key) if returned_key else self._auth_key

    def _fetch_state(self) -> dict[str, Any]:
        """Fetch the current game state from GetNextTurn.php."""
        params: dict[str, str] = {
            "gameName": self._game_name or "",
            "playerID": "1",
            "authKey": self._auth_key,
            "lastUpdate": str(self._last_update),
        }
        state = self._http_get("/GetNextTurn.php", params)
        try:
            self._last_update = int(state.get("lastUpdate", self._last_update))
        except (ValueError, TypeError):
            pass
        return state

    def _submit_action(self, mode: int, button_input: str = "") -> None:
        """Submit a player action via ProcessInput.php (GET endpoint).

        Most card-zone modes (27 = play from hand, 3 = equipment, 5 = arsenal,
        10/14/21/22 = other zones) read the card index or card-ID from the PHP
        ``$cardID`` variable (``$_GET["cardID"]``), NOT from ``$buttonInput``.
        Prompt/decision modes (17 = BUTTONINPUT, 20 = YESNO, 99 = pass, etc.)
        read from ``$buttonInput``.  Sending the value under both keys satisfies
        both families without any mode-specific branching.
        """
        params: dict[str, str] = {
            "gameName": self._game_name or "",
            "playerID": "1",
            "authKey": self._auth_key,
            "mode": str(mode),
        }
        if button_input:
            params["buttonInput"] = button_input
            params["cardID"] = button_input  # card-zone modes (27, 3, 5, …) index via $cardID
        self._http_get("/ProcessInput.php", params)

    def _poll_until_priority(
        self,
        max_polls: int = 600,
        interval: float = 0.1,
    ) -> dict[str, Any]:
        """Poll GetNextTurn.php until it is the agent's turn or the game ends."""
        for _ in range(max_polls):
            state = self._fetch_state()
            if state.get("havePriority", False) or self._is_game_over(state):
                return state
            time.sleep(interval)
        # Timeout fallthrough — return whatever the last state is
        return self._fetch_state()

    # ── State helpers ─────────────────────────────────────────────────────────

    def _phase_str(self, state: dict[str, Any]) -> str:
        turn_phase = state.get("turnPhase", {})
        if isinstance(turn_phase, dict):
            return str(turn_phase.get("turnPhase", ""))
        return ""

    def _is_game_over(self, state: dict[str, Any]) -> bool:
        if self._phase_str(state) == "OVER":
            return True
        player_hp = state.get("playerHealth", -1)
        opp_hp = state.get("opponentHealth", -1)
        if isinstance(player_hp, (int, float)) and int(player_hp) <= 0:
            return True
        if isinstance(opp_hp, (int, float)) and int(opp_hp) <= 0:
            return True
        return False

    def _did_player_win(self, player_hp: int, opp_hp: int) -> bool:
        """Return True if the agent (player 1) won the game."""
        return opp_hp <= 0 and player_hp > 0

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
                actions.append(
                    {
                        "action_code": int(mode),
                        "button_input": str(btn.get("buttonInput", "")),
                        "card_id": "",
                        "zone": "button",
                        "label": str(btn.get("label", f"btn_{mode}")),
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
                actions.append(
                    {
                        "action_code": int(mode),
                        "button_input": str(btn.get("buttonInput", "")),
                        "card_id": "",
                        "zone": "popup",
                        "label": str(btn.get("label", f"popup_{mode}")),
                    }
                )

        # Always guarantee at least a pass action
        if not actions:
            actions.append(
                {
                    "action_code": 99,
                    "button_input": "",
                    "card_id": "",
                    "zone": "button",
                    "label": "Pass",
                }
            )

        return actions

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

    def _parse_action(
        self,
        action: Any,
        legal_actions: list[dict[str, Any]],
    ) -> tuple[int, str]:
        """Parse an action value into ``(mode, button_input)``."""
        action_str = str(action).strip().lower()
        if action_str == "pass":
            return 99, ""
        try:
            idx = int(action_str)
            if 0 <= idx < len(legal_actions):
                a = legal_actions[idx]
                return int(a["action_code"]), str(a.get("button_input", ""))
        except (ValueError, TypeError):
            pass
        # Fallback: pass priority
        return 99, ""

    # ── rlbridge interface ────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        # Fresh HTTP session for each episode
        self._cookie_jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self._cookie_jar)
        )
        self._last_update = 0

        self._game_name, self._auth_key = self._create_game()
        self._auth_key = self._start_game(self._game_name)

        # Wait for initial state with priority (equipment selection or main phase)
        self._last_state = self._poll_until_priority()
        self._player_hp = int(self._last_state.get("playerHealth", 20))
        self._opp_hp = int(self._last_state.get("opponentHealth", 20))
        self._steps = 0
        self._initialized = True

        legal_actions = self._extract_legal_actions(self._last_state)
        obs = self._encode_observation(self._last_state, legal_actions)

        return ResetResult(
            observation=obs,
            info={
                "game_name": self._game_name,
                "legal_actions": legal_actions,
                "player_hp": self._player_hp,
                "opponent_hp": self._opp_hp,
            },
        )

    def step(self, action: Any) -> StepResult:
        state = self._last_state
        legal_actions = self._extract_legal_actions(state)

        mode, button_input = self._parse_action(action, legal_actions)
        self._submit_action(mode, button_input)

        new_state = self._poll_until_priority()
        new_player_hp = int(new_state.get("playerHealth", self._player_hp))
        new_opp_hp = int(new_state.get("opponentHealth", self._opp_hp))
        self._steps += 1

        terminated = self._is_game_over(new_state)
        truncated = not terminated and self._steps >= self._max_turns

        if terminated:
            won = self._did_player_win(new_player_hp, new_opp_hp)
            reward = 1.0 if won else -1.0
        else:
            dmg_dealt = max(0, self._opp_hp - new_opp_hp)
            dmg_taken = max(0, self._player_hp - new_player_hp)
            reward = dmg_dealt * 0.01 - dmg_taken * 0.01 - 0.005

        self._player_hp = new_player_hp
        self._opp_hp = new_opp_hp
        self._last_state = new_state

        new_legal_actions = self._extract_legal_actions(new_state)
        obs = self._encode_observation(new_state, new_legal_actions)

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
            },
        )

    def close(self) -> None:
        self._game_name = None
        self._auth_key = ""
        self._last_state = {}
        self._initialized = False

    @property
    def observation_space(self) -> TextSpace:
        return TextSpace(min_length=0, max_length=32_000)

    @property
    def action_space(self) -> TextSpace:
        return TextSpace(min_length=1, max_length=64)

    def render(self) -> RenderResult:
        if not self._last_state:
            return RenderResult(mode="ansi", text="No game state available.")

        state = self._last_state
        phase = self._phase_str(state)
        lines = [
            "=== Flesh and Blood (Talishar Engine) ===",
            f"Game: {self._game_name}  Turn: {state.get('turnNo', '?')}  Phase: {phase}",
            (
                f"You (P1): {state.get('playerHealth', '?')} HP  |  "
                f"Opponent: {state.get('opponentHealth', '?')} HP"
            ),
            (
                f"Hand: {len(state.get('playerHand', []))} cards  |  "
                f"Deck: {state.get('playerDeckCount', '?')} cards"
            ),
        ]

        legal_actions = self._extract_legal_actions(state)
        if legal_actions:
            lines.append("Legal actions:")
            for i, a in enumerate(legal_actions[:12]):
                lines.append(f"  [{i}] {a['label']} ({a['zone']})")
            if len(legal_actions) > 12:
                lines.append(f"  … and {len(legal_actions) - 12} more")

        return RenderResult(mode="ansi", text="\n".join(lines))

    def sample_action(self) -> str:
        """Return a random legal action index as a string."""
        if not self._last_state:
            return "pass"
        legal = self._extract_legal_actions(self._last_state)
        if not legal:
            return "pass"
        return str(random.randrange(len(legal)))
