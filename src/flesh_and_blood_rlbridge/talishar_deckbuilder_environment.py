"""Talishar Deck Builder RL environment.

The agent constructs a Flesh and Blood deck card-by-card.  When the agent
finalizes a valid deck, it is evaluated by playing ``num_eval_games`` Talishar
self-play games using :class:`TalisharEngineEnvironment`.  Optional
``eval_p1_agent`` / ``eval_p2_agent`` policies control both sides; otherwise
actions are chosen by a smart heuristic default policy.  The episode reward reflects the built
deck's win rate as player 1.

Deck construction rules (from FaB Comprehensive Rules)
-------------------------------------------------------
* **Silver Age**: maximum 40 deck cards, maximum 2 copies of each non-token/non-hero
  card pitch variant.
* **Classic Constructed**: minimum 60 deck cards, maximum 3 copies.
* Equipment cards occupy dedicated slots outside the main deck and are fixed
  per hero for this environment.

Prerequisites
-------------
A running Talishar Docker instance is required for deck evaluation (same as
:class:`TalisharEngineEnvironment`).  The agent can still build decks without
a server; evaluation simply returns a neutral score of 0.5 when the server is
unreachable.

Set the ``TALISHAR_URL`` environment variable (or pass ``base_url``) to point
at the Talishar server.  Set ``TALISHAR_ASSETS_PATH`` (or pass
``talishar_assets_path``) to the ``Talishar/Assets/`` directory so the
environment can write temporary deck files during evaluation.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Optional

from rlbridge.environments.base import rlbridgeEnvironment
from rlbridge.protocol.messages import RenderResult, ResetResult, StepResult, TextSpace

from .talishar_engine_environment import (
    TalisharEngineEnvironment,
    run_matchup_win_rate_eval,
    try_create_cpp_eval_environment,
)
from .talishar_oracle import TalisharConnectionError
from .talishar_sideboard_environment import TalisharSideboardEnvironment

# ---------------------------------------------------------------------------
# Format rules
# ---------------------------------------------------------------------------

#: Per-format deck construction constraints (FaB CR §4).
# ``pool_size`` is the total registered card pool (main deck + sideboard inventory).
# The deckbuilder agent builds to ``pool_size``; the sideboard agent then selects
# ``min_deck_size`` or more of those cards to form the per-game deck.
# If ``pool_size`` == ``min_deck_size`` the format has no sideboard (e.g. Blitz).
_FORMAT_RULES: dict[str, dict[str, int]] = {
    "blitz": {"min_deck_size": 40, "max_copies": 2, "pool_size": 40},
    # Classic Constructed: 80-card pool (hero + equipment handled separately via
    # the equipment header), minimum 60-card game deck per FaB CR §4.
    "classic_constructed": {"min_deck_size": 60, "max_copies": 3, "pool_size": 80},
    "living_legend": {"min_deck_size": 60, "max_copies": 3, "pool_size": 80},
    # Silver Age: 40-card main deck + up to 15-card inventory (sideboard).
    # Rarity restriction is enforced via the ``legality.silver_age`` field in
    # the card database — cards banned in Silver Age are excluded by _load_card_pool.
    "silver_age": {"min_deck_size": 40, "max_copies": 2, "pool_size": 55},
    "upf": {"min_deck_size": 60, "max_copies": 3, "pool_size": 80},
}

_DEFAULT_FORMAT = "silver_age"


def _normalize_game_format(game_format: str) -> str:
    token = str(game_format or "").strip().lower()
    if token in {"silver age", "silver_age", "sage"}:
        return "silver_age"
    return token or _DEFAULT_FORMAT

# Ira Crimson Haze hero + silver age equipment (matches Talishar Assets/Ira.txt line 1)
_IRA_HERO_ID = "ira_crimson_haze"
_IRA_SILVER_AGE_HEADER = (
    "ira_crimson_haze harmonized_kodachi harmonized_kodachi "
    "blade_beckoner_helm blood_scent tearing_shuko pouncing_paws"
)

# Card types that must never appear in the main deck
_EXCLUDED_TYPES: frozenset[str] = frozenset({"hero", "equipment", "token", "resource"})

# Talishar uses underscore-format card IDs (e.g. ``flying_kick_red``)
_TALISHAR_ID_RE = re.compile(r"^[a-z][a-z0-9_]+$")

# ---------------------------------------------------------------------------
# Card pool helper
# ---------------------------------------------------------------------------


def _load_card_pool(
    hero_class: str = "Ninja",
    game_format: str = _DEFAULT_FORMAT,
    *,
    card_db_path: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return the candidate card pool for deck construction.

    Included cards must satisfy all of:

    * ``class`` is ``hero_class`` or ``"Generic"``
    * Not a hero, equipment, token, or resource card type
    * ID uses underscore format (Talishar-compatible)
    * Not banned in ``game_format`` (cards without an explicit format entry are
      allowed — the card_db only tracks formats that have been scraped)
    """
    if card_db_path is None:
        card_db_path = str(Path(__file__).parent / "card_db" / "cards.json")

    all_cards: list[dict[str, Any]] = json.loads(
        Path(card_db_path).read_text(encoding="utf-8")
    )

    normalized_format = _normalize_game_format(game_format)

    pool: list[dict[str, Any]] = []
    for card in all_cards:
        cid = card.get("id", "")
        if not _TALISHAR_ID_RE.match(cid):
            continue
        if card.get("class") not in (hero_class, "Generic"):
            continue
        card_types = set(card.get("card_types", []))
        if card_types & _EXCLUDED_TYPES:
            continue
        legality = card.get("legality", {})
        if not legality:
            continue
        if legality.get(normalized_format) == "banned":
            continue
        pool.append(card)
    return pool


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TalisharDeckBuilderEnvironment(rlbridgeEnvironment):
    """RL environment for building a Flesh and Blood deck.

    The agent adds and removes cards one at a time.  Once the deck is valid
    (meets the format minimum), the agent can finalize it.  Finalization
    triggers ``num_eval_games`` self-play evaluation games on Talishar.  The
    episode reward is proportional to the built deck's win rate as player 1.

    Observation (JSON string)
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    .. code-block:: json

        {
            "hero": "ira_crimson_haze",
            "format": "silver_age",
            "currentDeck": [{"id": "flying_kick_red", "name": "Flying Kick", "count": 2}],
            "deckSize": 2,
            "targetMinSize": 40,
            "maxCopies": 2,
            "isValid": false,
            "stepNo": 4,
            "availableActions": ["add:flying_kick_red", "remove:flying_kick_red", "finalize"]
        }

    Action
    ~~~~~~
    An integer string indexing into ``availableActions``, or the action string
    directly.  Legal values:

    * ``"add:<card_id>"``    — add one copy to the deck (only when < ``maxCopies``)
    * ``"remove:<card_id>"`` — remove one copy from the deck (only when > 0)
    * ``"finalize"``         — evaluate the deck (always available; penalised if
      deck is invalid)

    Rewards
    ~~~~~~~
    * ``-step_penalty`` per build step (default ``0.005``)
    * ``win_rate * 2 - 1`` on valid finalize — maps 0 % wins → −1.0,
      100 % wins → +1.0
    * ``-1.0`` on invalid finalize (terminates episode immediately)
    * Episode truncated after ``max_build_steps`` with reward ``-step_penalty``

    Parameters
    ----------
    hero_id:
        Talishar hero card ID (used in deck file header).
    hero_class:
        Class string used to filter the card pool (e.g. ``"Ninja"``).
    hero_equipment_header:
        Full first line of the deck file (hero + equipment IDs, space-separated).
        Defaults to reading from ``Assets/Ira.txt``, falling back to a hardcoded
        Ira-silver_age header.
    game_format:
        FaB format string — ``"silver_age"``, ``"classic_constructed"``, etc.
    num_eval_games:
        Number of Talishar self-play games to play when evaluating a finalized deck.
    opponent_deck_name:
        Talishar Assets deck name for player 2 (default ``"Ira"``).
    eval_p1_agent, eval_p2_agent:
        Optional trained policies for players 1 and 2 during evaluation.  When
        ``eval_p2_agent`` is omitted, ``eval_p1_agent`` controls both sides.
    base_url:
        Talishar server base URL.  Defaults to ``TALISHAR_URL`` env var or
        ``"http://localhost"``.
    talishar_assets_path:
        Path to the Talishar ``Assets/`` directory for writing temporary deck
        files.  Defaults to ``TALISHAR_ASSETS_PATH`` env var, then
        ``~/Documents/flesh-and-blood/Talishar/Assets/``.
    card_pool:
        Override the auto-loaded card pool.  Pass ``None`` to auto-load.
    max_build_steps:
        Maximum number of build steps before the episode is truncated.
    step_penalty:
        Per-step reward penalty to encourage efficiency.
    render_mode:
        ``"ansi"`` for text rendering, or ``None``.
    """

    def __init__(
        self,
        *,
        hero_id: str = _IRA_HERO_ID,
        hero_class: str = "Ninja",
        hero_equipment_header: Optional[str] = None,
        game_format: str = _DEFAULT_FORMAT,
        num_eval_games: int = 5,
        opponent_deck_name: str = "Ira",
        opponent_hero_id: str = "dorinthea_ironsong",
        eval_p1_agent: Optional[Any] = None,
        eval_p2_agent: Optional[Any] = None,
        sideboard_agent: Optional[Any] = None,
        num_sideboard_episodes: int = 10,
        base_url: Optional[str] = None,
        talishar_assets_path: Optional[str] = None,
        card_pool: Optional[list[dict[str, Any]]] = None,
        starting_deck: Optional[dict[str, int]] = None,
        max_build_steps: int = 200,
        step_penalty: float = 0.005,
        render_mode: Optional[str] = None,
        use_cpp_engine: bool = True,
        cpp_engine_dir: Optional[str] = None,
        cpp_engine_cache_dir: Optional[str] = None,
        cpp_engine_deck1: Optional[str] = None,
        cpp_engine_deck2: Optional[str] = None,
    ) -> None:
        self._hero_id = hero_id
        self._hero_class = hero_class
        self._game_format = _normalize_game_format(game_format)
        self._num_eval_games = num_eval_games
        self._opponent_deck_name = opponent_deck_name
        self._opponent_hero_id = opponent_hero_id
        self._eval_p1_agent = eval_p1_agent
        self._eval_p2_agent = eval_p2_agent
        self._sideboard_agent = sideboard_agent
        self._num_sideboard_episodes = num_sideboard_episodes
        self._base_url = base_url or os.environ.get("TALISHAR_URL", "http://localhost")
        self._step_penalty = step_penalty
        self._max_build_steps = max_build_steps
        self._render_mode = render_mode
        self._use_cpp_engine = use_cpp_engine
        self._cpp_engine_dir: Optional[str] = cpp_engine_dir
        self._cpp_engine_cache_dir = cpp_engine_cache_dir
        self._cpp_engine_deck1 = cpp_engine_deck1
        self._cpp_engine_deck2 = cpp_engine_deck2

        # Assets directory: used for reading the hero deck header and writing
        # temporary evaluation deck files.
        self._assets_path = self._resolve_assets_path(talishar_assets_path)

        # Equipment header (line 1 of the Talishar deck file)
        if hero_equipment_header is not None:
            self._equipment_header: str = hero_equipment_header
        else:
            ira_txt = Path(self._assets_path) / "Ira.txt"
            if ira_txt.exists():
                self._equipment_header = ira_txt.read_text(
                    encoding="utf-8"
                ).splitlines()[0].strip()
            else:
                self._equipment_header = _IRA_SILVER_AGE_HEADER

        # Format constraints
        rules = _FORMAT_RULES.get(self._game_format, _FORMAT_RULES["silver_age"])
        self._min_deck_size: int = rules["min_deck_size"]
        self._max_copies: int = rules["max_copies"]
        # Pool size: total registered card count (main deck + sideboard/inventory).
        # When pool_size > min_deck_size the format supports sideboarding.
        self._pool_size: int = rules["pool_size"]

        # Card pool
        if card_pool is not None:
            self._card_pool = card_pool
        else:
            self._card_pool = _load_card_pool(hero_class, self._game_format)
        self._pool_by_id: dict[str, dict[str, Any]] = {
            c["id"]: c for c in self._card_pool
        }

        # Optional starting deck: pre-populate each episode with these cards.
        # Cards outside the pool or exceeding max_copies are silently dropped.
        self._starting_deck: dict[str, int] = (
            self._sanitize_starting_deck(starting_deck) if starting_deck else {}
        )

        # Episode state (initialised properly in reset())
        self._deck: dict[str, int] = {}
        self._step_no: int = 0
        self._done: bool = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_assets_path(path: Optional[str]) -> str:
        if path:
            return str(path)
        env_path = os.environ.get("TALISHAR_ASSETS_PATH")
        if env_path:
            return env_path
        default = (
            Path.home() / "Documents" / "flesh-and-blood" / "Talishar" / "Assets"
        )
        if default.exists():
            return str(default)
        return "/tmp"  # noqa: S108 — fallback for environments without the Talishar checkout

    def _sanitize_starting_deck(self, deck: dict[str, int]) -> dict[str, int]:
        """Validate a starting deck against the loaded pool and format rules.

        Rules applied:
        * Cards not in ``_pool_by_id`` are dropped (not legal for this hero/format).
        * Copy counts are clamped to ``[1, max_copies]``.
        * The total is clamped to ``pool_size`` — a starting deck larger than the
          target pool would leave no room for the agent to make meaningful decisions.

        Returns the sanitized deck (may be empty if nothing survives).
        """
        sanitized: dict[str, int] = {}
        total = 0
        for card_id, count in deck.items():
            if card_id not in self._pool_by_id:
                continue
            clamped = max(1, min(int(count), self._max_copies))
            # Don't overfill — stop adding once pool_size is reached
            available_slots = self._pool_size - total
            if available_slots <= 0:
                break
            take = min(clamped, available_slots)
            sanitized[card_id] = take
            total += take
        return sanitized

    @property
    def _deck_size(self) -> int:
        return sum(self._deck.values())

    @property
    def _is_valid(self) -> bool:
        """Pool is valid when it reaches the target pool size."""
        return self._deck_size >= self._pool_size

    def _available_actions(self) -> list[str]:
        actions: list[str] = []
        # Only offer add actions while the pool hasn't reached its target size.
        # This enforces the pool_size cap (e.g. 55 cards for Silver Age) so the
        # agent cannot build an oversized pool that would be illegal in play.
        if self._deck_size < self._pool_size:
            for card_id, card in self._pool_by_id.items():
                _ = card  # pool iteration; card metadata used in observation only
                if self._deck.get(card_id, 0) < self._max_copies:
                    actions.append(f"add:{card_id}")
        for card_id, count in self._deck.items():
            if count > 0:
                actions.append(f"remove:{card_id}")
        actions.append("finalize")
        return actions

    def _encode_observation(self, available_actions: list[str]) -> str:
        current_deck = [
            {
                "id": cid,
                "name": self._pool_by_id.get(cid, {}).get("name", cid),
                "count": count,
            }
            for cid, count in sorted(self._deck.items())
            if count > 0
        ]
        obs: dict[str, Any] = {
            "hero": self._hero_id,
            "format": self._game_format,
            "currentDeck": current_deck,
            "deckSize": self._deck_size,
            "targetMinSize": self._min_deck_size,
            "targetPoolSize": self._pool_size,
            "hasSideboard": self._pool_size > self._min_deck_size,
            "maxCopies": self._max_copies,
            "isValid": self._is_valid,
            "stepNo": self._step_no,
            "availableActions": available_actions,
        }
        return json.dumps(obs, separators=(",", ":"))

    def _resolve_action(self, action: Any) -> str:
        """Resolve an action value to an action string.

        Accepts either an integer index into ``availableActions`` or a direct
        action string (e.g. ``"add:flying_kick_red"``).
        """
        action_str = str(action).strip()
        try:
            idx = int(action_str)
            avail = self._available_actions()
            if 0 <= idx < len(avail):
                return avail[idx]
        except ValueError:
            pass
        return action_str

    # ── rlbridge interface ────────────────────────────────────────────────────

    @property
    def observation_space(self) -> TextSpace:
        return TextSpace()

    @property
    def action_space(self) -> TextSpace:
        return TextSpace()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        """Reset the environment for a new deckbuilding episode.

        Parameters
        ----------
        options:
            Optional dict that may contain:

            ``"starting_deck"`` : dict[str, int]
                Override the starting deck for this episode only.  Useful for
                curriculum learning (e.g. start from last best pool).
                Validated against the pool and format rules before use.
        """
        # Determine starting deck for this episode
        if options and "starting_deck" in options:
            episode_start = self._sanitize_starting_deck(options["starting_deck"])
        else:
            episode_start = dict(self._starting_deck)

        self._deck = episode_start
        self._step_no = 0
        self._done = False
        actions = self._available_actions()
        obs = self._encode_observation(actions)
        return ResetResult(
            observation=obs,
            info={
                "hero": self._hero_id,
                "format": self._game_format,
                "pool_size": len(self._card_pool),
                "min_deck_size": self._min_deck_size,
                "target_pool_size": self._pool_size,
                "has_sideboard": self._pool_size > self._min_deck_size,
                "max_copies": self._max_copies,
                "starting_deck_size": sum(episode_start.values()),
            },
        )

    def step(self, action: Any) -> StepResult:
        if self._done:
            raise RuntimeError(
                "step() called on a finished episode; call reset() first"
            )

        action_str = self._resolve_action(action)
        self._step_no += 1
        reward: float = -self._step_penalty
        terminated = False
        truncated = False

        if action_str == "finalize":
            if not self._is_valid:
                reward = -1.0
                terminated = True
            else:
                win_rate = self._evaluate_deck()
                reward = win_rate * 2.0 - 1.0  # map [0.0, 1.0] → [-1.0, +1.0]
                terminated = True

        elif action_str.startswith("add:"):
            card_id = action_str[4:]
            if card_id in self._pool_by_id:
                current = self._deck.get(card_id, 0)
                if current < self._max_copies:
                    self._deck[card_id] = current + 1

        elif action_str.startswith("remove:"):
            card_id = action_str[7:]
            current = self._deck.get(card_id, 0)
            if current > 0:
                new_count = current - 1
                if new_count == 0:
                    del self._deck[card_id]
                else:
                    self._deck[card_id] = new_count

        if not terminated and self._step_no >= self._max_build_steps:
            truncated = True

        self._done = terminated or truncated
        actions = self._available_actions() if not self._done else []
        obs = self._encode_observation(actions)

        return StepResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

    def close(self) -> None:
        pass

    def render(self) -> RenderResult:
        if self._render_mode != "ansi":
            return RenderResult(mode=self._render_mode or "none", text="")

        lines = [
            f"=== Deck Builder [{self._hero_id} / {self._game_format}] ===",
            f"Step {self._step_no} | "
            f"Cards in pool: {self._deck_size} / {self._pool_size} target "
            f"(min game deck: {self._min_deck_size}) | "
            f"Valid: {self._is_valid}",
        ]
        if self._deck:
            lines.append("Current deck:")
            for cid, cnt in sorted(self._deck.items()):
                name = self._pool_by_id.get(cid, {}).get("name", cid)
                lines.append(f"  {cnt}x  {name}  ({cid})")
        else:
            lines.append("  (empty deck)")

        return RenderResult(mode="ansi", text="\n".join(lines))

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _write_game_deck_file(
        self, deck_name: str, game_deck: Optional[dict[str, int]] = None
    ) -> Path:
        """Write a game deck to ``{assets_path}/{deck_name}.txt``.

        ``game_deck`` defaults to the current ``self._deck`` (full pool).
        Returns the ``Path`` of the written file.
        """
        source = game_deck if game_deck is not None else self._deck
        card_ids: list[str] = []
        for card_id, count in sorted(source.items()):
            card_ids.extend([card_id] * count)

        content = f"{self._equipment_header}\n{' '.join(card_ids)}\n"
        out_path = Path(self._assets_path) / f"{deck_name}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        return out_path

    def _cpp_lookup_decks(self) -> tuple[str, str]:
        lookup1 = self._cpp_engine_deck1 or self._hero_id
        lookup2 = self._cpp_engine_deck2 or self._opponent_deck_name
        return lookup1, lookup2

    def _evaluate_deck(self) -> float:
        """Evaluate the current card pool via the sideboard → play pipeline.

        Steps:
        1. Run :class:`TalisharSideboardEnvironment` for ``num_sideboard_episodes``
           episodes to find the best game deck selection from the built pool.
           The sideboard agent (``self._sideboard_agent``) is used when available;
           otherwise actions are sampled randomly.
        2. Play ``num_eval_games`` games with the best-selected deck and return
           the win rate.  When a compiled C++ engine is available for this
           matchup it is used (fast, no HTTP); otherwise Talishar HTTP is used.

        Returns the win rate in ``[0.0, 1.0]``.  Returns ``0.5`` (neutral) if
        ``num_eval_games == 0``, the server is unreachable, or evaluation fails.
        """
        if self._num_eval_games == 0:
            return 0.5

        lookup1, lookup2 = self._cpp_lookup_decks()
        cpp_env = try_create_cpp_eval_environment(
            base_url=self._base_url,
            game_format=self._game_format,
            lookup_deck1=lookup1,
            lookup_deck2=lookup2,
            cpp_engine_dir=self._cpp_engine_dir,
            cpp_engine_cache_dir=self._cpp_engine_cache_dir,
            max_turns=60,
            use_cpp_engine=self._use_cpp_engine,
        )
        if cpp_env is not None:
            try:
                return run_matchup_win_rate_eval(
                    cpp_env,
                    self._num_eval_games,
                    eval_p1_agent=self._eval_p1_agent,
                    eval_p2_agent=self._eval_p2_agent,
                    max_steps=60,
                    deck_player_id=1,
                )
            except Exception:  # noqa: BLE001
                return 0.5
            finally:
                cpp_env.close()

        if self._cpp_engine_dir is not None:
            return 0.5

        game_deck = self._run_sideboard_phase()

        deck_name = f"rl_deck_{uuid.uuid4().hex[:12]}"
        deck_file: Optional[Path] = None

        try:
            deck_file = self._write_game_deck_file(deck_name, game_deck)

            env = TalisharEngineEnvironment(
                base_url=self._base_url,
                game_format=self._game_format,
                local_deck_name=deck_name,
                opponent_deck_name=self._opponent_deck_name,
                max_turns=60,
                self_play=True,
                use_cpp_engine=False,
            )
            try:
                return run_matchup_win_rate_eval(
                    env,
                    self._num_eval_games,
                    eval_p1_agent=self._eval_p1_agent,
                    eval_p2_agent=self._eval_p2_agent,
                    max_steps=60,
                    deck_player_id=1,
                )
            finally:
                env.close()

        except TalisharConnectionError:
            return 0.5
        except Exception:  # noqa: BLE001
            return 0.5
        finally:
            if deck_file is not None and deck_file.exists():
                deck_file.unlink(missing_ok=True)

    def _run_sideboard_phase(self) -> dict[str, int]:
        """Run the sideboard environment to find the best game deck from the pool.

        Runs ``num_sideboard_episodes`` episodes of :class:`TalisharSideboardEnvironment`
        against ``opponent_hero_id``.  The episode with the highest reward produces
        the game deck used for play evaluation.

        Falls back to a greedy top-N cut if all episodes fail to produce a valid deck.
        """
        try:
            sb_env = TalisharSideboardEnvironment(
                card_pool=self._deck,
                pool_by_id=self._pool_by_id,
                opponent_hero_id=self._opponent_hero_id,
                hero_id=self._hero_id,
                hero_equipment_header=self._equipment_header,
                game_format=self._game_format,
                # Disable recursive play evaluation inside the sideboard env —
                # the deckbuilder runs play evaluation after sideboard selection.
                num_eval_games=0,
                opponent_deck_name=self._opponent_deck_name,
                base_url=self._base_url,
                talishar_assets_path=self._assets_path,
                use_cpp_engine=self._use_cpp_engine,
                cpp_engine_dir=self._cpp_engine_dir,
                cpp_engine_cache_dir=self._cpp_engine_cache_dir,
                cpp_engine_deck1=self._cpp_engine_deck1,
                cpp_engine_deck2=self._cpp_engine_deck2,
            )
        except Exception:  # noqa: BLE001
            return self._greedy_game_deck()

        best_reward = float("-inf")
        best_deck: dict[str, int] = {}

        for _ in range(self._num_sideboard_episodes):
            result = sb_env.reset()
            ep_reward = 0.0
            done = False

            while not done:
                if self._sideboard_agent is not None and hasattr(
                    self._sideboard_agent, "act"
                ):
                    action = self._sideboard_agent.act(result.observation)
                else:
                    import random  # noqa: PLC0415
                    avail = json.loads(result.observation).get("availableActions", [])
                    action = random.choice(avail) if avail else "finalize"

                step = sb_env.step(action)
                ep_reward += step.reward
                done = step.terminated or step.truncated
                result = ResetResult(observation=step.observation, info={})

            candidate = sb_env.get_active_deck()
            if (
                sum(candidate.values()) >= sb_env._min_deck_size
                and ep_reward > best_reward
            ):
                best_reward = ep_reward
                best_deck = candidate

        return best_deck if best_deck else self._greedy_game_deck()

    def _greedy_game_deck(self) -> dict[str, int]:
        """Fallback: greedily take the highest-copy cards until ``min_deck_size``."""
        if self._pool_size == self._min_deck_size:
            return dict(self._deck)

        game_deck: dict[str, int] = {}
        remaining = self._min_deck_size
        for card_id, count in sorted(self._deck.items(), key=lambda kv: -kv[1]):
            if remaining <= 0:
                break
            take = min(count, remaining)
            game_deck[card_id] = take
            remaining -= take
        return game_deck

    def get_card_pool(self) -> dict[str, int]:
        """Return the currently built card pool (card_id → copy count).

        Call after a valid finalize to get the full pool for passing to
        :class:`TalisharSideboardEnvironment`.
        """
        return dict(self._deck)
