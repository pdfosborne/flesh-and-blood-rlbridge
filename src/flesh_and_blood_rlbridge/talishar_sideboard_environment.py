"""Talishar Sideboard RL environment.

Phase 2 of the three-phase FaB pipeline:

    1. **Deckbuilder** (:class:`~.TalisharDeckBuilderEnvironment`) — agent builds
       the full registered card pool (e.g. 80 cards for Classic Constructed,
       55 cards for Silver Age).

    2. **Sideboard** *(this module)* — given the built pool and the identity of
       the upcoming opponent, the agent selects which cards to include in the
       active game deck (≥ ``min_deck_size``), swapping cards between the
       *deck* and the *sideboard* (inventory).

    3. **Play** (:class:`~.TalisharEngineEnvironment`) — the selected deck is
       used to play evaluation matches; the win rate propagates back as reward.

Sideboard rules (FaB CR §4 / Silver Age supplement)
-----------------------------------------------------
* The agent starts with all pool cards placed in the *sideboard* (out of deck).
* On each step the agent either moves a card **into** the deck or **out of** the
  deck, subject to the constraint that each card appears ≤ ``max_copies`` times
  across both zones combined (the pool is already at most ``max_copies``-deep).
* ``finalize`` is available at any point; it is penalised if ``deck_size <
  min_deck_size``.
* Once the deck is valid (≥ ``min_deck_size``) the agent may finalize and
  trigger evaluation games.

Observation (JSON string)
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: json

    {
        "hero": "ira_crimson_haze",
        "format": "silver_age",
        "opponentHero": "dorinthea_ironsong",
        "deck": [{"id": "flying_kick_red", "name": "Flying Kick", "count": 2}],
        "sideboard": [{"id": "surging_strike_red", "name": "Surging Strike", "count": 1}],
        "deckSize": 38,
        "sideboardSize": 17,
        "minDeckSize": 40,
        "maxCopies": 2,
        "isValid": false,
        "stepNo": 4,
        "availableActions": [
            "move_to_deck:flying_kick_red",
            "move_to_sideboard:flying_kick_red",
            "finalize"
        ]
    }

Action
~~~~~~
* ``"move_to_deck:<card_id>"``      — move one copy from sideboard → deck
* ``"move_to_sideboard:<card_id>"`` — move one copy from deck → sideboard
* ``"finalize"``                     — lock selection and evaluate

Rewards
~~~~~~~
* ``-step_penalty`` per sideboard step (default ``0.002``)
* ``win_rate * 2 - 1`` on valid finalize → maps 0 %→−1.0, 100 %→+1.0
* ``-1.0`` on invalid finalize (terminates episode immediately)
* Episode truncated after ``max_sideboard_steps`` with reward ``-step_penalty``
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from rlbridge.environments.base import rlbridgeEnvironment
from rlbridge.protocol.messages import RenderResult, ResetResult, StepResult, TextSpace

from .talishar_engine_environment import (
    TalisharEngineEnvironment,
    run_talishar_eval_episode,
    talishar_deck_player_won,
)
from .talishar_oracle import TalisharConnectionError

# ---------------------------------------------------------------------------
# Per-format minimum deck size (mirrored from deckbuilder for convenience)
# ---------------------------------------------------------------------------

_FORMAT_MIN_DECK: dict[str, int] = {
    "blitz": 40,
    "classic_constructed": 60,
    "living_legend": 60,
    "silver_age": 40,
    "upf": 60,
}

_DEFAULT_FORMAT = "silver_age"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TalisharSideboardEnvironment(rlbridgeEnvironment):
    """RL environment for sideboarding a Flesh and Blood deck.

    The agent receives a pre-built card pool and an opponent hero identity.
    It then selects which cards to include in the active game deck by moving
    cards between the *deck* zone and the *sideboard* zone, subject to the
    format's minimum deck size.  Finalizing a valid deck triggers evaluation
    games on Talishar; the reward reflects the win rate vs. the given opponent.

    Typical usage in the three-phase pipeline
    ------------------------------------------
    .. code-block:: python

        from flesh_and_blood_rlbridge import (
            TalisharDeckBuilderEnvironment,
            TalisharSideboardEnvironment,
        )

        # --- Phase 1: build the card pool ---
        builder = TalisharDeckBuilderEnvironment(game_format="silver_age")
        result = builder.reset()
        done = False
        while not done:
            result = builder.step(builder.sample_action())
            done = result.terminated or result.truncated
        card_pool = builder.get_card_pool()
        pool_meta = builder._pool_by_id  # card metadata

        # --- Phase 2: sideboard for a specific opponent ---
        sideboarder = TalisharSideboardEnvironment(
            card_pool=card_pool,
            pool_by_id=pool_meta,
            opponent_hero_id="dorinthea_ironsong",
            game_format="silver_age",
        )
        result = sideboarder.reset()
        done = False
        while not done:
            result = sideboarder.step(sideboarder.sample_action())
            done = result.terminated or result.truncated
        # The active deck (ready for Phase 3 play) is available via:
        active_deck = sideboarder.get_active_deck()

    Parameters
    ----------
    card_pool:
        Dict of ``card_id → copy count`` produced by the deckbuilder.
    pool_by_id:
        Card metadata dict (``card_id → card dict``) for name lookups.
    opponent_hero_id:
        Talishar hero ID of the upcoming opponent (informs the observation).
    hero_id:
        Your hero's Talishar card ID.
    hero_equipment_header:
        Full first line of the deck file (hero + equipment IDs).
    game_format:
        FaB format string.
    num_eval_games:
        Number of Talishar games to run on finalize.
    opponent_deck_name:
        Talishar Assets deck name for the opponent.
    eval_p1_agent, eval_p2_agent:
        Optional trained policies for evaluation play.
    base_url:
        Talishar server base URL.
    talishar_assets_path:
        Path to the Talishar ``Assets/`` directory.
    max_sideboard_steps:
        Maximum sideboard steps before truncation.
    step_penalty:
        Per-step reward penalty.
    render_mode:
        ``"ansi"`` for text rendering, or ``None``.
    """

    def __init__(
        self,
        *,
        card_pool: dict[str, int],
        pool_by_id: dict[str, dict[str, Any]],
        opponent_hero_id: str = "dorinthea_ironsong",
        hero_id: str = "ira_crimson_haze",
        hero_equipment_header: str = (
            "ira_crimson_haze harmonized_kodachi harmonized_kodachi "
            "blade_beckoner_helm blood_scent tearing_shuko pouncing_paws"
        ),
        game_format: str = _DEFAULT_FORMAT,
        num_eval_games: int = 5,
        opponent_deck_name: str = "Ira",
        eval_p1_agent: Optional[Any] = None,
        eval_p2_agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        talishar_assets_path: Optional[str] = None,
        max_sideboard_steps: int = 100,
        step_penalty: float = 0.002,
        render_mode: Optional[str] = None,
    ) -> None:
        # Pool (fixed across the episode)
        self._card_pool: dict[str, int] = dict(card_pool)
        self._pool_by_id: dict[str, dict[str, Any]] = pool_by_id
        self._pool_total: int = sum(self._card_pool.values())

        self._opponent_hero_id = opponent_hero_id
        self._hero_id = hero_id
        self._equipment_header = hero_equipment_header
        self._game_format = game_format
        self._min_deck_size: int = _FORMAT_MIN_DECK.get(game_format, 40)
        self._num_eval_games = num_eval_games
        self._opponent_deck_name = opponent_deck_name
        self._eval_p1_agent = eval_p1_agent
        self._eval_p2_agent = eval_p2_agent
        self._base_url = base_url or os.environ.get("TALISHAR_URL", "http://localhost")
        self._step_penalty = step_penalty
        self._max_sideboard_steps = max_sideboard_steps
        self._render_mode = render_mode
        self._assets_path = self._resolve_assets_path(talishar_assets_path)

        # Episode state (set in reset())
        self._deck: dict[str, int] = {}
        self._sideboard: dict[str, int] = {}
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
        return "/tmp"  # noqa: S108

    @property
    def _deck_size(self) -> int:
        return sum(self._deck.values())

    @property
    def _sideboard_size(self) -> int:
        return sum(self._sideboard.values())

    @property
    def _is_valid(self) -> bool:
        return self._deck_size >= self._min_deck_size

    def _available_actions(self) -> list[str]:
        actions: list[str] = []
        # Move from sideboard into deck
        for card_id, count in self._sideboard.items():
            if count > 0:
                actions.append(f"move_to_deck:{card_id}")
        # Move from deck into sideboard
        for card_id, count in self._deck.items():
            if count > 0:
                actions.append(f"move_to_sideboard:{card_id}")
        actions.append("finalize")
        return actions

    def _encode_observation(self, available_actions: list[str]) -> str:
        def _fmt(zone: dict[str, int]) -> list[dict[str, Any]]:
            return [
                {
                    "id": cid,
                    "name": self._pool_by_id.get(cid, {}).get("name", cid),
                    "count": count,
                }
                for cid, count in sorted(zone.items())
                if count > 0
            ]

        obs: dict[str, Any] = {
            "hero": self._hero_id,
            "format": self._game_format,
            "opponentHero": self._opponent_hero_id,
            "deck": _fmt(self._deck),
            "sideboard": _fmt(self._sideboard),
            "deckSize": self._deck_size,
            "sideboardSize": self._sideboard_size,
            "minDeckSize": self._min_deck_size,
            "isValid": self._is_valid,
            "stepNo": self._step_no,
            "availableActions": available_actions,
        }
        return json.dumps(obs, separators=(",", ":"))

    def _resolve_action(self, action: Any) -> str:
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
        """Start a new sideboard episode.

        All pool cards begin in the *sideboard* zone; the deck is empty.
        The agent must move cards into the deck to build a valid game deck.

        ``options`` may contain:

        * ``"opponent_hero_id"`` — override the opponent hero for this episode.
        * ``"card_pool"`` — override the pool dict (useful for multi-opponent loops).
        """
        if options:
            if "opponent_hero_id" in options:
                self._opponent_hero_id = str(options["opponent_hero_id"])
            if "card_pool" in options:
                self._card_pool = dict(options["card_pool"])
                self._pool_total = sum(self._card_pool.values())

        # All pool cards start in the sideboard — agent moves them into the deck.
        self._sideboard = dict(self._card_pool)
        self._deck = {}
        self._step_no = 0
        self._done = False

        actions = self._available_actions()
        obs = self._encode_observation(actions)
        return ResetResult(
            observation=obs,
            info={
                "hero": self._hero_id,
                "opponent_hero": self._opponent_hero_id,
                "format": self._game_format,
                "pool_total": self._pool_total,
                "min_deck_size": self._min_deck_size,
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
            elif self._num_eval_games == 0:
                # Called from within TalisharDeckBuilderEnvironment._run_sideboard_phase:
                # play evaluation is handled by the deckbuilder — just terminate
                # with a neutral reward so the sideboard agent episode ends cleanly.
                reward = 0.0
                terminated = True
            else:
                win_rate = self._evaluate_deck()
                reward = win_rate * 2.0 - 1.0
                terminated = True

        elif action_str.startswith("move_to_deck:"):
            card_id = action_str[len("move_to_deck:"):]
            sb_count = self._sideboard.get(card_id, 0)
            if sb_count > 0:
                self._sideboard[card_id] = sb_count - 1
                if self._sideboard[card_id] == 0:
                    del self._sideboard[card_id]
                self._deck[card_id] = self._deck.get(card_id, 0) + 1

        elif action_str.startswith("move_to_sideboard:"):
            card_id = action_str[len("move_to_sideboard:"):]
            dk_count = self._deck.get(card_id, 0)
            if dk_count > 0:
                self._deck[card_id] = dk_count - 1
                if self._deck[card_id] == 0:
                    del self._deck[card_id]
                self._sideboard[card_id] = self._sideboard.get(card_id, 0) + 1

        if not terminated and self._step_no >= self._max_sideboard_steps:
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
            f"=== Sideboard [{self._hero_id} vs {self._opponent_hero_id} / {self._game_format}] ===",
            f"Step {self._step_no} | "
            f"Deck: {self._deck_size} / {self._min_deck_size}+ required | "
            f"Sideboard: {self._sideboard_size} | "
            f"Valid: {self._is_valid}",
        ]

        if self._deck:
            lines.append("Active deck:")
            for cid, cnt in sorted(self._deck.items()):
                name = self._pool_by_id.get(cid, {}).get("name", cid)
                lines.append(f"  {cnt}x  {name}  ({cid})")
        else:
            lines.append("  Active deck: (empty)")

        if self._sideboard:
            lines.append("Sideboard (not in game deck):")
            for cid, cnt in sorted(self._sideboard.items()):
                name = self._pool_by_id.get(cid, {}).get("name", cid)
                lines.append(f"  {cnt}x  {name}  ({cid})")

        return RenderResult(mode="ansi", text="\n".join(lines))

    # ── Public helpers ────────────────────────────────────────────────────────

    def get_active_deck(self) -> dict[str, int]:
        """Return the currently selected game deck (card_id → count).

        Call after a valid finalize to retrieve the deck for Phase 3 play.
        """
        return dict(self._deck)

    def get_sideboard_cards(self) -> dict[str, int]:
        """Return cards currently in the sideboard (not selected for the game deck)."""
        return dict(self._sideboard)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _write_deck_file(self, deck_name: str) -> Path:
        card_ids: list[str] = []
        for card_id, count in sorted(self._deck.items()):
            card_ids.extend([card_id] * count)

        content = f"{self._equipment_header}\n{' '.join(card_ids)}\n"
        out_path = Path(self._assets_path) / f"{deck_name}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        return out_path

    def _evaluate_deck(self) -> float:
        """Evaluate the active game deck against the configured opponent.

        Returns the win rate in ``[0.0, 1.0]``.  Returns ``0.5`` (neutral) on
        connection or evaluation failure.
        """
        deck_name = f"rl_sb_{uuid.uuid4().hex[:12]}"
        deck_file: Optional[Path] = None
        wins = 0

        try:
            deck_file = self._write_deck_file(deck_name)

            p1_policy = self._eval_p1_agent
            p2_policy = self._eval_p2_agent
            for _ in range(self._num_eval_games):
                env = TalisharEngineEnvironment(
                    base_url=self._base_url,
                    game_format=self._game_format,
                    local_deck_name=deck_name,
                    opponent_deck_name=self._opponent_deck_name,
                    max_turns=60,
                    self_play=True,
                )
                try:
                    if p1_policy is not None:
                        out = run_talishar_eval_episode(
                            env,
                            p1_policy,
                            max_steps=60,
                            p2_agent=p2_policy,
                            deck_player_id=1,
                        )
                        if out.get("deck_player_won") is True:
                            wins += 1
                    else:
                        from rlbridge.protocol.messages import StepResult as _SR  # noqa: PLC0415
                        reset_result = env.reset()
                        obs_data = json.loads(reset_result.observation)
                        step_result: Optional[_SR] = None
                        done = False
                        while not done:
                            step_result = env.step(env.sample_action())
                            done = step_result.terminated or step_result.truncated
                            if not done:
                                obs_data = json.loads(step_result.observation)
                        if talishar_deck_player_won(
                            obs_data,
                            deck_player_id=1,
                            terminated=bool(
                                step_result is not None and step_result.terminated
                            ),
                        ):
                            wins += 1
                finally:
                    env.close()

        except TalisharConnectionError:
            return 0.5
        except Exception:  # noqa: BLE001
            return 0.5
        finally:
            if deck_file is not None and deck_file.exists():
                deck_file.unlink(missing_ok=True)

        return wins / self._num_eval_games
