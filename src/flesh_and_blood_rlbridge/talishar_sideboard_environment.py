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
# Per-format deck size limits (mirrored from deckbuilder for convenience)
# ---------------------------------------------------------------------------

_FORMAT_MIN_DECK: dict[str, int] = {
    "blitz": 40,
    "classic_constructed": 60,
    "living_legend": 60,
    "silver_age": 40,
    "upf": 60,
}

# Maximum game-deck size (non-token cards only).
# For Silver Age the game deck is exactly 40 cards (pool=55, sideboard=15).
# For formats with no stated maximum use the pool_size as the ceiling.
_FORMAT_MAX_DECK: dict[str, int] = {
    "blitz": 40,                 # pool == min, no sideboard
    "classic_constructed": 80,   # pool_size; no formal maximum
    "living_legend": 80,
    "silver_age": 40,            # exactly 40 — remaining 15 stay in inventory
    "upf": 80,
}

# ---------------------------------------------------------------------------
# Equipment slot detection from FaB type lines
# ---------------------------------------------------------------------------
# FaB type lines encode slot explicitly, e.g.:
#   "Generic Equipment - Chest"
#   "Ninja Equipment - Arms"
#   "Mechanologist Equipment - Evo Head"   (suffix still ends with "Head")
#   "Lightning Runeblade Weapon - Sword (2H)"
#   "Ninja Hero - Young"
#
# The card DB has two ID formats:
#   underscore  (e.g. "blade_beckoner_helm") — used in equipment headers,
#               but type_line is often empty in the DB entry
#   hyphen      (e.g. "blade-beckoner-helm") — alternate entry that carries
#               the authoritative type_line
#
# Strategy: look up underscore ID first; if type_line is empty, retry with
# the hyphen form.  Fall back to ID-pattern heuristics only when both miss.
# ---------------------------------------------------------------------------

# Maps the suffix after " - " in an Equipment type_line to a canonical slot.
# All Mechanologist variants ("Base Head", "Evo Head" …) end with the same
# canonical token, so endswith() covers every variant automatically.
_SLOT_SUFFIX_MAP: tuple[tuple[str, str], ...] = (
    ("Head",     "head"),
    ("Chest",    "chest"),
    ("Arms",     "arms"),
    ("Legs",     "legs"),
    ("Off-Hand", "off_hand"),
    ("Quiver",   "off_hand"),   # Ranger quiver — valid off-hand
)

# ID-pattern fallbacks used only when the DB has no usable type_line.
_EQUIP_HEAD_PAT:   frozenset[str] = frozenset(["helm", "hood", "crown", "cap",
                                                "headband", "goggles", "mask", "hat",
                                                "visor", "tiara", "circlet"])
_EQUIP_CHEST_PAT:  frozenset[str] = frozenset(["coat", "robe", "vest", "chestplate",
                                                "chest", "jacket", "tunic", "cuirass",
                                                "cloak", "cape", "mantle", "doublet"])
_EQUIP_ARMS_PAT:   frozenset[str] = frozenset(["gauntlet", "glove", "bracer",
                                                "vambrace", "bangle", "shuko",
                                                "sleeve", "handwrap"])
_EQUIP_LEGS_PAT:   frozenset[str] = frozenset(["boots", "greaves", "pants", "leggings",
                                                "sabaton", "sabatons", "footwrap",
                                                "shin", "paws"])
_EQUIP_WEAPON_PAT: frozenset[str] = frozenset(["kodachi", "dawnblade", "rosetta",
                                                "galaxia", "pistol", "sword", "axe",
                                                "staff", "bow", "harpoon", "blade",
                                                "katana", "scimitar", "bauble"])

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
        cpp_engine_dir: Optional[str] = None,
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
        # Upper bound: can't exceed pool size; Silver Age = exactly 40.
        self._max_deck_size: int = _FORMAT_MAX_DECK.get(
            game_format, self._pool_total
        )
        self._num_eval_games = num_eval_games
        self._opponent_deck_name = opponent_deck_name
        self._eval_p1_agent = eval_p1_agent
        self._eval_p2_agent = eval_p2_agent
        self._base_url = base_url or os.environ.get("TALISHAR_URL", "http://localhost")
        self._step_penalty = step_penalty
        self._max_sideboard_steps = max_sideboard_steps
        self._render_mode = render_mode
        self._cpp_engine_dir: Optional[str] = cpp_engine_dir
        self._assets_path = self._resolve_assets_path(talishar_assets_path)

        # Episode state (set in reset())
        self._deck: dict[str, int] = {}
        self._sideboard: dict[str, int] = {}
        self._step_no: int = 0
        self._done: bool = False

        self._validate_equipment_header()

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

    def _is_token_card(self, card_id: str) -> bool:
        """Return True if *card_id* is a token card that must not count toward deck size.

        Tokens are never legal in a game deck — they are generated during play.
        We detect them via two paths (either is sufficient):

        1. The ``pool_by_id`` metadata contains ``"token"`` in ``card_types``.
        2. The card ID follows the conventional ``*_token`` or ``*_token_*`` pattern.
        """
        meta = self._pool_by_id.get(card_id, {})
        if "token" in meta.get("card_types", []):
            return True
        return card_id.endswith("_token") or "_token_" in card_id

    def _validate_equipment_header(self) -> None:
        """Warn if *hero_equipment_header* does not look like a complete loadout.

        A complete FaB loadout requires:
        * 1 hero / character card
        * At least 1 weapon (1×2H weapon, or up to 2×1H weapons)
        * 1 each of: head, chest, arms, legs equipment pieces

        Detection uses the card's ``type_line`` from the card DB (authoritative),
        e.g. ``"Generic Equipment - Chest"`` or ``"Lightning Runeblade Weapon - Sword (2H)"``.
        The DB stores two ID formats: underscore (used in equipment headers, but
        ``type_line`` is often empty) and hyphen (carries the authoritative
        ``type_line``).  Both are tried before falling back to ID-pattern heuristics.

        Issues a WARNING rather than raising, so it never blocks training.
        """
        items = self._equipment_header.split()
        if not items:
            print(
                f"  WARNING [{self._hero_id}] equipment_header is empty — "
                "hero + full equipment set required"
            )
            return

        # Load full card DB (equipment/weapon/hero cards are excluded from pool_by_id)
        db_path = Path(__file__).parent / "card_db" / "cards.json"
        full_db: dict[str, dict[str, Any]] = {}
        try:
            full_db = {
                c["id"]: c
                for c in json.loads(db_path.read_text(encoding="utf-8"))
                if "id" in c
            }
        except Exception:  # noqa: BLE001
            pass  # validation falls through to ID-pattern heuristics

        def _get_type_line(card_id: str) -> str:
            """Return type_line, trying underscore ID then hyphenated ID."""
            tl = full_db.get(card_id, {}).get("type_line", "")
            if not tl:
                tl = full_db.get(card_id.replace("_", "-"), {}).get("type_line", "")
            return tl.strip()

        slots_found: set[str] = set()
        weapon_hands: list[int] = []  # one entry (1 or 2) per weapon found

        for item in items:
            cid = item.lower()
            tl = _get_type_line(cid)
            tl_lower = tl.lower()

            if tl:
                # ── Authoritative: parse the type_line from the DB ──────────
                if "weapon" in tl_lower:
                    # Weapon - <Type> (1H) / (2H) — token weapons have no hand spec
                    if "(2h)" in tl_lower:
                        weapon_hands.append(2)
                    elif "(1h)" in tl_lower:
                        weapon_hands.append(1)
                    # else: token weapon (e.g. "Assassin Token Weapon - Dagger") — skip
                elif "equipment" in tl_lower:
                    # Slot lives after " - " in the type_line; all variants
                    # ("Base Head", "Evo Head", …) end with the canonical name.
                    suffix = tl.split(" - ")[-1] if " - " in tl else ""
                    for sfx_key, slot_name in _SLOT_SUFFIX_MAP:
                        if suffix.endswith(sfx_key):
                            slots_found.add(slot_name)
                            break
                    # else: unrecognised suffix (e.g. bare "Mechanologist Equipment")
                elif "hero" in tl_lower or "character" in tl_lower:
                    slots_found.add("hero")
            else:
                # ── Fallback: ID-pattern heuristics ────────────────────────
                if any(p in cid for p in _EQUIP_HEAD_PAT):
                    slots_found.add("head")
                elif any(p in cid for p in _EQUIP_CHEST_PAT):
                    slots_found.add("chest")
                elif any(p in cid for p in _EQUIP_ARMS_PAT):
                    slots_found.add("arms")
                elif any(p in cid for p in _EQUIP_LEGS_PAT):
                    slots_found.add("legs")
                elif any(p in cid for p in _EQUIP_WEAPON_PAT):
                    weapon_hands.append(1)  # assume 1H when type_line unavailable
                elif self._hero_id and self._hero_id.split("_")[0] in cid:
                    slots_found.add("hero")

        # ── Report problems ────────────────────────────────────────────────
        required = {"head", "chest", "arms", "legs"}
        missing = required - slots_found
        if missing:
            print(
                f"  WARNING [{self._hero_id}] equipment header may be missing "
                f"slot(s): {', '.join(sorted(missing))}  "
                f"(header='{self._equipment_header}')"
            )
        if not weapon_hands:
            print(
                f"  WARNING [{self._hero_id}] equipment header has no weapon  "
                f"(header='{self._equipment_header}')"
            )
        else:
            n2h = weapon_hands.count(2)
            n1h = weapon_hands.count(1)
            if n2h > 1:
                print(
                    f"  WARNING [{self._hero_id}] equipment has {n2h} two-handed "
                    "weapons (max 1)"
                )
            elif n2h == 1 and n1h > 0:
                print(
                    f"  WARNING [{self._hero_id}] equipment mixes a 2H weapon "
                    f"with {n1h} 1H weapon(s)"
                )
            elif n2h == 0 and n1h > 2:
                print(
                    f"  WARNING [{self._hero_id}] equipment has {n1h} one-handed "
                    "weapons (max 2 for dual-wield)"
                )

    @property
    def _deck_size(self) -> int:
        """Non-token cards currently in the game deck."""
        return sum(
            count for cid, count in self._deck.items()
            if not self._is_token_card(cid)
        )

    @property
    def _sideboard_size(self) -> int:
        """Non-token cards currently in the sideboard."""
        return sum(
            count for cid, count in self._sideboard.items()
            if not self._is_token_card(cid)
        )

    @property
    def _is_valid(self) -> bool:
        return self._min_deck_size <= self._deck_size <= self._max_deck_size

    def _available_actions(self) -> list[str]:
        actions: list[str] = []
        # Move from sideboard into deck — only when below max AND card is not a token
        if self._deck_size < self._max_deck_size:
            for card_id, count in self._sideboard.items():
                if count > 0 and not self._is_token_card(card_id):
                    actions.append(f"move_to_deck:{card_id}")
        # Move from deck into sideboard — only non-token cards (tokens can't be in deck)
        for card_id, count in self._deck.items():
            if count > 0 and not self._is_token_card(card_id):
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
            "maxDeckSize": self._max_deck_size,
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
                "max_deck_size": self._max_deck_size,
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
            f"Deck: {self._deck_size} / {self._min_deck_size}–{self._max_deck_size} required | "
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
            if self._is_token_card(card_id):
                continue  # tokens never belong in a Talishar deck file
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

        When ``cpp_engine_dir`` is set the C++ engine is used (fast, no HTTP);
        otherwise Talishar HTTP is used.
        """
        # ── C++ fast-path ──────────────────────────────────────────────────
        if self._cpp_engine_dir is not None:
            from .cpp_engine_environment import CppEngineEnvironment  # noqa: PLC0415
            wins = 0
            try:
                for _ in range(self._num_eval_games):
                    cpp_env = CppEngineEnvironment(
                        engine_dir=self._cpp_engine_dir, max_turns=200
                    )
                    try:
                        cpp_env.reset()
                        done = False
                        final_reward = 0.0
                        while not done:
                            sr = cpp_env.step(cpp_env.sample_action())
                            done = sr.terminated or sr.truncated
                            if done:
                                final_reward = sr.reward
                        if final_reward > 0.0:
                            wins += 1
                    finally:
                        cpp_env.close()
            except Exception:  # noqa: BLE001
                return 0.5
            return wins / self._num_eval_games

        # ── Talishar HTTP path ──────────────────────────────────────────────
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
