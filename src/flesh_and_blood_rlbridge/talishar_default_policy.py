"""Heuristic default policy for Talishar action selection.

Phase-based routing using actual Talishar phase codes.  Phase codes are
verified against Talishar/CoreLogic.php (CanPassPhase), Talishar/GameTerms.php
(TypeToPlay), Talishar/BuildGameState.php, Talishar/BuildPlayerInputPopup.php,
and Talishar/Libraries/NetworkingLibraries.php (ProcessInput case 99).

CRITICAL invariant from ProcessInput case 99:
  if (CanPassPhase($turn[0])) { PassInput(...); }
  // else: mode=99 is silently ignored — state unchanged, game stalls.

Therefore:
  • Phases with CanPassPhase=0 MUST receive the correct action code —
    sending 99 is a no-op that causes a spin-loop caught only by the
    cycle-breaker (4 repeats → random action).
  • Phases with CanPassPhase=1 can always be passed safely.

Cancel (10000) and Undo Block (10001) are intentionally excluded from
_PASS_MODE_CODES so they are never chosen as the confirming pass action.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

_YES_LABELS = ("yes", "confirm", "continue", "ok", "accept", "done")

# Mode codes treated as pass/skip.  10000 (Cancel) and 10001 (Undo Block)
# deliberately excluded — they undo committed plays and cause loops.
_PASS_MODE_CODES: frozenset[int] = frozenset({99, 101, 105})
# Talishar maps both "Cancel" (pitch) and general undo to 10000; 10001 is Undo Block.
# 10003 reverts to a prior turn; 100016–100019 are undo confirmation prompts.
_REVERT_MODE_CODES: frozenset[int] = frozenset({
    10000,
    10001,
    10003,
    100016,
    100017,
    100018,
    100019,
})

# Talishar phase code buckets (normalised to lowercase).
# ── CanPassPhase=1 (mode=99 advances the game) ───────────────────────────────
# Pass immediately — no useful action to take, or the choice is optional.
_CONFIRM_PHASES: frozenset[str] = frozenset({
    # Combat window responses — always pass (don't hold priority)
    "instant", "a", "chain",
    # Trigger / effect ordering — pass accepts default order
    "ordertriggers", "startturn", "endphase",
    # Optional "may choose" phases — pass = decline
    "maychoosemultizone", "maychoosehand", "maychoosediscard",
    "maychoosedeck", "maychoosearsenal", "maychoosecombatchain",
    "maymultichoosehand", "maymultichoosetext",
    "maychoosetheirdiscard", "maychoosemysoul", "maychoosepermanent",
    # End-of-turn pitch ordering — pass = shortcut (auto-order)
    "pdeck",
    # Informational OK popups
    "ok",
    # Rearrange opponent deck top — pass = no rearrange
    "coercive",
})
# ── CanPassPhase=0 (mode=99 is silently ignored → must pick) ─────────────────
_PITCH_PHASES:   frozenset[str] = frozenset({"p"})           # pitch hand card to pay cost
_POPUP_PHASES:   frozenset[str] = frozenset({"choosemultizone"})  # mandatory popup pick
# CHOOSEHAND: hand cards exposed as action=16 zone=hand (not in popup)
_CHOOSE_HAND_PHASES: frozenset[str] = frozenset({
    "choosehand", "choosehandcancel",
    # Mandatory multi-card hand selection windows (mode=99 is a no-op here)
    "multichoosehand", "multichoosetext",
})
# BUTTONINPUT: popup buttons with action=17; must press one (or 105)
_BUTTON_INPUT_PHASES: frozenset[str] = frozenset({
    "buttoninput", "buttoninputnopass",
})
# ── Other structured phases ───────────────────────────────────────────────────
_DEFENSE_PHASES: frozenset[str] = frozenset({"d"})
_BLOCK_PHASES:   frozenset[str] = frozenset({"b"})
# "m" is main phase — any unrecognised phase is also treated as main.

_ZONE_KEY_MAP: dict[str, str] = {
    "hand":      "playerHand",
    "equipment": "playerEquipment",
    "arsenal":   "playerArse",
    "aura":      "playerAuras",
    "ally":      "playerAllies",
    "item":      "playerItems",
    "permanent": "playerPermanents",
    "discard":   "playerDiscard",
    "banish":    "playerBanish",
}
_VALID_ZONES = frozenset(_ZONE_KEY_MAP)

# Block rules (aggressive to save runtime on episode length)
_MIN_BLOCK_VALUE: int = 3      # cards with defense < 3 are attack cards, not blockers
_MAX_BLOCK_CARDS: int = 1      # at most 1 card committed per block window
_MIN_HAND_FOR_ATTACK: int = 3  # if ≤ this many attackers remain in hand, skip blocking
_MAX_PITCH_VALUE: int = 2      # maximum pitch value for blocking
_MIN_RESOURCE_COST: int = 1      # minimum resource cost for blocking

def _load_card_stats() -> dict[str, tuple[int, int]]:
    """Load (power, defense) keyed by Talishar card-ID slug from cards.json."""
    db_path = Path(__file__).parent / "card_db" / "cards.json"
    try:
        records = json.loads(db_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    stats: dict[str, tuple[int, int]] = {}
    for rec in records:
        cid = rec.get("id", "")
        if "_" in cid:
            stats[cid] = (int(rec.get("power") or 0), int(rec.get("defense") or 0))
    return stats


def _load_card_resource_stats() -> dict[str, tuple[int, int]]:
    """Load (pitch, cost) keyed by Talishar card-ID slug from cards.json."""
    db_path = Path(__file__).parent / "card_db" / "cards.json"
    try:
        records = json.loads(db_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    stats: dict[str, tuple[int, int]] = {}
    for rec in records:
        cid = rec.get("id", "")
        if "_" in cid:
            stats[cid] = (int(rec.get("pitch") or 0), int(rec.get("cost") or 0))
    return stats


_CARD_STATS: dict[str, tuple[int, int]] = _load_card_stats()
_CARD_RESOURCE_STATS: dict[str, tuple[int, int]] = _load_card_resource_stats()


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize(text: Any) -> str:
    return str(text or "").strip().lower()


def _is_pass_action(action: dict[str, Any]) -> bool:
    code = _to_int(action.get("action_code", 0))
    if code in _PASS_MODE_CODES:
        return True
    label = _normalize(action.get("label", ""))
    return any(tok in label for tok in ("pass", "end turn", "no block", "skip"))


def _is_revert_action(action: dict[str, Any]) -> bool:
    """Return True for Cancel / Undo actions that revert committed game state."""
    code = _to_int(action.get("action_code", 0))
    if code in _REVERT_MODE_CODES:
        return True
    label = _normalize(action.get("label", ""))
    return "undo" in label or label == "cancel" or "revert" in label


_REPEAT_ACTION_THRESHOLD = 3
_REPEAT_ACTION_PENALTY = -0.1


class RepeatActionTracker:
    """Track repeated and oscillating action patterns within a turn.

    Penalises both exact repeats (A, A, A, …) and pairwise loops such as
    play-undo cycles (A, B, A, B, …) where *B* is commonly Cancel/Undo.
    """

    def __init__(self) -> None:
        self.turn_no: int = -1
        self.acting_player: int = -1
        self.last_action_key: Optional[tuple[int, str]] = None
        self.prev_action_key: Optional[tuple[int, str]] = None
        self.repeat_streak: int = 0

    def reset(self, *, turn_no: int, acting_player_id: int) -> None:
        self.turn_no = turn_no
        self.acting_player = acting_player_id
        self.last_action_key = None
        self.prev_action_key = None
        self.repeat_streak = 0

    def update(
        self,
        action_key: tuple[int, str],
        *,
        turn_no: int,
        acting_player_id: int,
        threshold: int = _REPEAT_ACTION_THRESHOLD,
        penalty: float = _REPEAT_ACTION_PENALTY,
    ) -> float:
        """Record *action_key* and return any repeat penalty for this step."""
        turn_changed = turn_no != self.turn_no
        player_changed = acting_player_id != self.acting_player
        if turn_changed or player_changed:
            self.turn_no = turn_no
            self.acting_player = acting_player_id
            self.prev_action_key = None
            self.last_action_key = action_key
            self.repeat_streak = 1
            return 0.0

        if action_key == self.last_action_key:
            self.repeat_streak += 1
        elif (
            self.prev_action_key is not None
            and action_key == self.prev_action_key
            and self.last_action_key != action_key
        ):
            # Play-undo and other A-B-A-B oscillation loops.
            self.repeat_streak += 1
        else:
            self.repeat_streak = 1

        self.prev_action_key = self.last_action_key
        self.last_action_key = action_key

        if self.repeat_streak >= threshold:
            return penalty
        return 0.0


def _pitch_hand_actions(filtered: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        a for a in filtered
        if a.get("zone") == "hand"
        and _to_int(a.get("action_code", 0)) == 27
    ]


def _strip_revert_actions(
    phase: str,
    filtered: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove all undo/cancel/revert actions.

    Talishar maps pitch-window *Cancel* (10000) to ``RevertGamestate()`` — the
    same code path as the UI undo button — so agents must never receive it.
    """
    del phase  # kept for call-site compatibility
    return [action for action in filtered if not _is_revert_action(action)]


def _get_phase(state: dict[str, Any]) -> str:
    """Return normalised Talishar phase code, e.g. 'm', 'd', 'instant'."""
    tp = state.get("turnPhase", {})
    if isinstance(tp, dict):
        return _normalize(tp.get("turnPhase", ""))
    return _normalize(tp)


# Phases where Talishar ``CanPassPhase()`` returns 0 (CoreLogic.php).
# mode=99 is silently ignored — agents must pick a real action.
_CANNOT_PASS_PHASES: frozenset[str] = frozenset({
    "p",
    "choosedeck",
    "choosetheirdeck",
    "handtopbottom",
    "choosecombatchain",
    "choosecharacter",
    "choosehand",
    "choosehandcancel",
    "multichoosediscard",
    "choosediscardcancel",
    "choosearsenal",
    "choosediscard",
    "multichoosehand",
    "choosemultizone",
    "choosebanish",
    "multichoosebanish",
    "buttoninputnopass",
    "choosefirstplayer",
    "multichoosedeck",
    "choosepermanent",
    "multichoosetext",
    "choosemysoul",
    "choosemyaura",
    "choosecard",
    "choosecardid",
    "over",
    "buttoninput",
    "multichoosetheirdiscard",
})


def can_pass_phase(state: dict[str, Any]) -> bool:
    """Return whether Talishar will accept mode=99 (Pass) in this state."""
    if "canPassPhase" in state:
        return bool(state.get("canPassPhase"))
    return _get_phase(state) not in _CANNOT_PASS_PHASES


def _match_action_card(
    action: dict[str, Any],
    state: dict[str, Any],
    zone_cache: dict[str, list] | None = None,
) -> Optional[dict[str, Any]]:
    zone = _normalize(action.get("zone", ""))
    if zone not in _VALID_ZONES:
        return None
    button_input = str(action.get("button_input", ""))
    action_code = _to_int(action.get("action_code", 0))
    cards = (zone_cache or {}).get(zone) if zone_cache is not None else None
    if cards is None:
        cards = state.get(_ZONE_KEY_MAP[zone], [])
    for i, raw in enumerate(cards):
        if not isinstance(raw, dict):
            continue
        if _to_int(raw.get("action", 0)) != action_code:
            continue
        if str(raw.get("actionDataOverride", str(i))) == button_input:
            return raw
    return None


def _card_id_from_action(action: dict[str, Any], card: Optional[dict[str, Any]]) -> str:
    cid = str(action.get("card_id", "")).strip()
    if not cid and isinstance(card, dict):
        cid = str(card.get("cardNumber", "")).strip()
    return cid


def _estimate_attack(
    action: dict[str, Any],
    state: dict[str, Any],
    card: Optional[dict[str, Any]] = None,
) -> int:
    if card is None:
        card = _match_action_card(action, state)
    cid = _card_id_from_action(action, card)
    if cid and cid in _CARD_STATS:
        power = _CARD_STATS[cid][0]
        if power > 0:
            return power
    if isinstance(card, dict):
        power = _to_int(card.get("power", 0), 0)
        if power > 0:
            return power
    label = _normalize(action.get("label", ""))
    m = re.search(r"(?:attack|atk)\D*(\d+)|(\d+)\s*(?:attack|atk)", label)
    if m:
        return _to_int(m.group(1) or m.group(2), 0)
    return 0


def _estimate_defense(
    action: dict[str, Any],
    state: dict[str, Any],
    card: Optional[dict[str, Any]] = None,
) -> int:
    if card is None:
        card = _match_action_card(action, state)
    cid = _card_id_from_action(action, card)
    if cid and cid in _CARD_STATS:
        defense = _CARD_STATS[cid][1]
        if defense > 0:
            return defense
    if isinstance(card, dict):
        defense = _to_int(card.get("defense", 0), 0)
        if defense > 0:
            return defense
    label = _normalize(action.get("label", ""))
    m = re.search(r"(?:block|def)\D*(\d+)|(\d+)\s*(?:block|def)", label)
    if m:
        return _to_int(m.group(1) or m.group(2), 0)
    return 0


def _best_index(indices: list[int], scores: dict[int, tuple[float, ...]]) -> int:
    return max(indices, key=lambda i: scores[i])


def _card_id_from(
    card: Optional[dict[str, Any]],
    action: Optional[dict[str, Any]] = None,
) -> str:
    if isinstance(card, dict):
        for key in ("cardNumber", "cardID", "card_id"):
            cid = str(card.get(key, "")).strip()
            if cid and cid.lower() not in {"cardback", "card_back"}:
                return cid
    if isinstance(action, dict):
        cid = str(action.get("card_id", "")).strip()
        if cid:
            return cid
    return ""


def _card_cost(
    card: Optional[dict[str, Any]],
    action: Optional[dict[str, Any]] = None,
) -> int:
    """Return the resource cost to play a card.

    Talishar hand JSON often omits ``cost``; fall back to the local card DB.
    """
    if isinstance(card, dict):
        for key in ("cost", "resourceCost"):
            val = card.get(key)
            if val is not None:
                return _to_int(val, 0)
    cid = _card_id_from(card, action)
    if cid in _CARD_RESOURCE_STATS:
        return _CARD_RESOURCE_STATS[cid][1]
    return 0


def _card_pitch_value(
    card: Optional[dict[str, Any]],
    action: Optional[dict[str, Any]] = None,
) -> int:
    """Return the pitch/resource value of a card (blue=3, yellow=2, red=1, unknown=0).

    Talishar exposes this as ``card["resource"]`` (the number of resources generated
    when the card is pitched).  Falls back to ``card["pitch"]`` and the card DB.
    """
    if not isinstance(card, dict):
        return 0
    for key in ("resource", "pitch", "pitchValue"):
        val = card.get(key)
        if val is not None:
            return _to_int(val, 0)
    cid = _card_id_from(card, action)
    if cid in _CARD_RESOURCE_STATS:
        return _CARD_RESOURCE_STATS[cid][0]
    return 0


def _card_defense(
    card: Optional[dict[str, Any]],
    action: Optional[dict[str, Any]] = None,
) -> int:
    """Return block/defense value, preferring live Talishar state over the card DB."""
    if isinstance(card, dict):
        for key in ("defense", "block", "blockValue"):
            val = card.get(key)
            if val is not None:
                return _to_int(val, 0)
    cid = _card_id_from(card, action)
    if cid in _CARD_STATS:
        return _CARD_STATS[cid][1]
    return 0


def _play_resource_budget(
    state: dict[str, Any],
    *,
    exclude_card_pitch: int = 0,
) -> int:
    """Resources available to pay for a play: floated pool plus hand pitch.

    When playing from hand, exclude the played card's pitch value because that
    card is being played, not pitched.
    """
    floating = _to_int(state.get("playerPitchCount", 0), 0)
    hand_cards = [c for c in state.get("playerHand", []) if isinstance(c, dict)]
    hand_pitch = sum(_card_pitch_value(c) for c in hand_cards)
    return floating + hand_pitch - exclude_card_pitch


def _is_affordable_hand_play(
    action: dict[str, Any],
    state: dict[str, Any],
) -> bool:
    """Return True when a hand play can be paid with pool + other hand cards."""
    zone = _normalize(action.get("zone", ""))
    code = _to_int(action.get("action_code", 0))
    if zone != "hand" or code != 27:
        return True

    card = _match_action_card(action, state)
    cost = _card_cost(card, action)
    if cost <= 0:
        return True

    card_pitch = _card_pitch_value(card, action)
    return cost <= _play_resource_budget(state, exclude_card_pitch=card_pitch)


def _is_affordable_arsenal_play(
    action: dict[str, Any],
    state: dict[str, Any],
) -> bool:
    """Return True when an arsenal play can be paid from hand pitch + pool.

    The arsenal card itself is not pitched; the full hand (and any floated
    resources) must cover the play cost.
    """
    zone = _normalize(action.get("zone", ""))
    code = _to_int(action.get("action_code", 0))
    if zone != "arsenal" or code != 5:
        return True

    card = _match_action_card(action, state)
    cost = _card_cost(card, action)
    if cost <= 0:
        return True

    return cost <= _play_resource_budget(state)


def _is_hand_block_action(action: dict[str, Any]) -> bool:
    return (
        _to_int(action.get("action_code", 0)) == 27
        and _normalize(action.get("zone", "")) == "hand"
    )


def _is_viable_block_action(
    action: dict[str, Any],
    state: dict[str, Any],
) -> bool:
    """Return True when a hand card can legally contribute to a block."""
    if not _is_hand_block_action(action):
        return True

    card = _match_action_card(action, state)
    if isinstance(card, dict) and _to_int(card.get("action", 0), 0) == 0:
        return False
    if not _is_affordable_hand_play(action, state):
        return False
    return _card_defense(card, action) >= _MIN_BLOCK_VALUE


def _apply_block_phase_filter(
    state: dict[str, Any],
    filtered: list[dict[str, Any]],
    *,
    block_blacklist: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """During block/defense phases, only offer viable blocks or pass."""
    pass_actions = [a for a in filtered if _is_pass_action(a)]
    auxiliary = [
        a for a in filtered
        if not _is_pass_action(a)
        and not _is_hand_block_action(a)
        and not _is_revert_action(a)
        and _to_int(a.get("action_code", 0)) != 3
    ]
    viable_blocks = [
        a for a in filtered
        if _is_hand_block_action(a)
        and _is_viable_block_action(a, state)
        and str(a.get("label", "") or "") not in block_blacklist
    ]
    if viable_blocks:
        return viable_blocks + pass_actions + auxiliary

    if pass_actions:
        return [pass_actions[0]]
    return [
        {
            "action_code": 99,
            "button_input": "",
            "card_id": "",
            "zone": "button",
            "label": "Pass",
        }
    ]


def choose_talishar_action_index(
    legal_actions: list[dict[str, Any]],
    state: Optional[dict[str, Any]] = None,
    *,
    unaffordable: frozenset[str] = frozenset(),
    max_pitch_value: int = _MAX_PITCH_VALUE,
    min_resource_cost: int = _MIN_RESOURCE_COST,
) -> int:
    """Choose a legal action index using phase-aware tactical heuristics.

    Parameters
    ----------
    legal_actions:
        List of legal action dicts from ``_extract_legal_actions``.
    state:
        Current Talishar game-state dict.
    unaffordable:
        Set of card labels known to cause an empty pitch window — skip them
        in the main phase to avoid an infinite loop.
    max_pitch_value:
        Block phase filter — only use cards whose pitch/resource value is
        **≤ this** as blockers.  Default 999 (no filter).
        Set to e.g. 1 to only block with red cards (pitch=1), preserving
        yellow (pitch=2) and blue (pitch=3) cards for pitching on your turn.
    min_resource_cost:
        Block phase filter — only use cards whose resource cost is **≥ this**
        as blockers.  Default 0 (no filter).
        Set to e.g. 1 to skip free cards, ensuring only real hand cards
        are spent on blocks.

    Decision tree (each bucket verified against Talishar PHP source):
      1. CONFIRM phases → pass immediately (CanPassPhase=1, safe).
      2. PITCH phase (P, CanPassPhase=0) → pitch cheapest hand card; if hand
         is empty, use Cancel(10000) to abort play (Pass would silently undo it).
      3. POPUP phases (choosemultizone, CanPassPhase=0) → pick first popup card.
      4. CHOOSEHAND phases (CanPassPhase=0) → pick first action-16 hand card;
         CHOOSEHANDCANCEL falls back to Cancel if no hand cards.
      5. BUTTONINPUT phases (CanPassPhase=0) → pick first mode-17 popup button.
      6. BLOCK phase (B) → play best dedicated-block card (defense ≥ 3), filtered
         by max_pitch_value and min_resource_cost; else pass.
      7. DEFENSE phase (D) → block with highest-defense card ≥ threshold, else pass.
      8. MAIN phase (M) and unrecognised:
         a. Yes/confirm popup buttons.
         b. Best attack (action-27 hand card, scored by power, minus unaffordable).
         c. Other non-pass button/popup actions.
         d. Pass.
         NOTE: Equip (action 3) is intentionally skipped — the pitch window it
         opens may only offer Cancel, causing an equip→Cancel→equip infinite loop.
    """
    if not legal_actions:
        return 0

    state = state or {}
    phase = _get_phase(state)

    non_pass: list[int] = []
    pass_indices: list[int] = []
    for i, action in enumerate(legal_actions):
        if _is_pass_action(action):
            pass_indices.append(i)
        else:
            non_pass.append(i)

    # Prefer explicit mode-99 pass as the pass_index.
    pass_index = 0
    for i in pass_indices:
        if _to_int(legal_actions[i].get("action_code", 0)) == 99:
            pass_index = i
            break
    else:
        if pass_indices:
            pass_index = pass_indices[0]

    if not non_pass:
        return pass_index

    # ── End-of-turn arsenal (ARS) — pick a hand card to add to arsenal ───────
    if phase == "ars":
        ars_candidates = [
            i for i in non_pass
            if _normalize(legal_actions[i].get("zone", "")) == "hand"
            and _to_int(legal_actions[i].get("action_code", 0)) == 4
        ]
        if ars_candidates:
            zone_cache_ars: dict[str, list] = {
                z: state.get(sk, []) for z, sk in _ZONE_KEY_MAP.items()
            }
            def _ars_pitch_score(idx: int) -> int:
                card = _match_action_card(legal_actions[idx], state, zone_cache_ars)
                return _card_pitch_value(legal_actions[idx], card)
            return max(ars_candidates, key=_ars_pitch_score)
        return pass_index

    # ── Mandatory popup pick from arsenal zone (CHOOSEARSENAL) ─────────────
    if phase == "choosearsenal":
        popup_ars = [
            i for i in non_pass
            if legal_actions[i].get("zone") in ("popup", "arsenal")
        ]
        if popup_ars:
            return popup_ars[0]
        hand_ars = [
            i for i in non_pass
            if _normalize(legal_actions[i].get("zone", "")) == "hand"
            and _to_int(legal_actions[i].get("action_code", 0)) == 16
        ]
        if hand_ars:
            return hand_ars[0]
        return non_pass[0]

    # ── 1. Confirm phases ────────────────────────────────────────────────────
    if phase in _CONFIRM_PHASES:
        return pass_index

    # ── 2. Pitch phase (P) — pitch hand cards to pay for the played card ─────
    # In Talishar's pitch window, action-27 hand cards are "pitch this card" actions.
    # Pass (99) confirms with whatever has been pitched so far.
    # CRITICAL: passing before pitching enough resources undoes the play and loops.
    # Strategy: pitch the cheapest (lowest-power) hand card available to pay
    # incrementally, then pass when nothing left to pitch.
    if phase in _PITCH_PHASES:
        # Collect pitchable hand cards (non-pass, non-Cancel, zone=hand)
        pitch_candidates = [
            i for i in non_pass
            if _normalize(legal_actions[i].get("zone", "")) == "hand"
            and _to_int(legal_actions[i].get("action_code", 0)) == 27
        ]
        if pitch_candidates:
            # Pitch the card with the lowest attack value (prefer blue/cheap cards)
            zone_cache_p: dict[str, list] = {
                z: state.get(sk, []) for z, sk in _ZONE_KEY_MAP.items()
            }
            pitch_scores: dict[int, float] = {}
            for i in pitch_candidates:
                card = _match_action_card(legal_actions[i], state, zone_cache_p)
                av = float(_estimate_attack(legal_actions[i], state, card))
                pitch_scores[i] = av
            # Pick lowest-attack card to pitch (preserve high-value cards if possible)
            return min(pitch_candidates, key=lambda i: pitch_scores[i])
        return pass_index

    # ── 3. Popup phases (mandatory card pick, e.g. choosemultizone) ──────────
    # CanPassPhase=0: must pick a card from playerInputPopUp.popup.cards (action 16).
    if phase in _POPUP_PHASES:
        popup_cards = [i for i in non_pass if legal_actions[i].get("zone") == "popup"]
        if popup_cards:
            return popup_cards[0]
        return pass_index

    # ── 4. CHOOSEHAND phases (CanPassPhase=0, hand cards have action=16) ─────
    # NOT exposed in popup — cards are in playerHand with action=16 zone=hand.
    # Passing silently does nothing; must select a hand card.
    if phase in _CHOOSE_HAND_PHASES:
        hand16 = [
            i for i in non_pass
            if _normalize(legal_actions[i].get("zone", "")) == "hand"
            and _to_int(legal_actions[i].get("action_code", 0)) == 16
        ]
        if hand16:
            return hand16[0]
        # CHOOSEHANDCANCEL: if no eligible hand cards, Cancel (10000) is valid.
        for i in non_pass:
            if _to_int(legal_actions[i].get("action_code", 0)) == 10000:
                return i
        return pass_index

    # ── 5. BUTTONINPUT phases (CanPassPhase=0, must press a mode-17 button) ──
    # Options shown as popup buttons with action_code=17.
    # 105 = "Skip All Runechants" shortcut is also valid for CHOOSEARCANE.
    if phase in _BUTTON_INPUT_PHASES:
        btn17 = [
            i for i in non_pass
            if _to_int(legal_actions[i].get("action_code", 0)) in (17, 105)
            and legal_actions[i].get("zone") in ("button", "popup")
        ]
        if btn17:
            return btn17[0]
        # Fallback: any non-pass popup/button action
        any_btn = [i for i in non_pass if legal_actions[i].get("zone") in ("button", "popup")]
        if any_btn:
            return any_btn[0]
        return pass_index

    # ── 6. Block phase (B) — selective blocking to preserve hand for attacks ──
    # Rules (to avoid over-blocking / deck exhaustion):
    #   a. Only use dedicated block cards: defense ≥ _MIN_BLOCK_VALUE (3).
    #      Cards with <3 defense are attack cards — keep them for the main phase.
    #   b. At most _MAX_BLOCK_CARDS (2) cards played per block window.
    #      Talishar calls B phase once per card played, so we play 1 here and
    #      rely on the phase repeating if we want a second; the counter in state
    #      tells us how many have already been committed this window.
    #   c. Be aggressive: if the hand has ≤ _MIN_HAND_FOR_ATTACK cards that are
    #      pure attackers (defense < 3), skip blocking entirely to preserve them.
    if phase in _BLOCK_PHASES:
        zone_cache_b: dict[str, list] = {
            z: state.get(sk, []) for z, sk in _ZONE_KEY_MAP.items()
        }
        hand_cards = zone_cache_b.get("hand") or []

        # Count how many dedicated blockers (def≥3) vs attackers (def<3) are in hand
        attacker_count = sum(
            1 for c in hand_cards
            if isinstance(c, dict) and _to_int(c.get("defense", 0), 0) < _MIN_BLOCK_VALUE
        )
        # If the hand is mostly attack cards and there are few of them, pass —
        # preserve them for the main phase.
        if attacker_count <= _MIN_HAND_FOR_ATTACK:
            return pass_index

        # How many blocks have already been committed this window?
        blocks_committed: int = 0
        try:
            acl = state.get("activeChainLink", {})
            if isinstance(acl, dict):
                blocks_committed = _to_int(acl.get("totalDefense", 0), 0) // _MIN_BLOCK_VALUE
        except Exception:
            pass
        if blocks_committed >= _MAX_BLOCK_CARDS:
            return pass_index

        # Only consider cards with defense ≥ _MIN_BLOCK_VALUE — real block cards.
        # Also apply caller-supplied pitch-value and cost filters.
        block_scores: dict[int, tuple[float, ...]] = {}
        block_candidates: list[int] = []
        for i in non_pass:
            action = legal_actions[i]
            if _to_int(action.get("action_code", 0)) != 27:
                continue
            if _normalize(action.get("zone", "")) != "hand":
                continue
            card = _match_action_card(action, state, zone_cache_b)
            bv = float(_estimate_defense(action, state, card))
            if bv < _MIN_BLOCK_VALUE:
                continue  # not a dedicated blocker — keep it for attack phase
            # Pitch-value filter: skip high-pitch cards (save them for pitching).
            pv = _card_pitch_value(card)
            if pv > max_pitch_value:
                continue
            # Cost filter: skip cheap cards below the minimum cost threshold.
            cv = _to_int((card or {}).get("cost", 0), 0)
            if cv < min_resource_cost:
                continue
            # Prefer highest defense; break ties by lowest attack (least wasted offence)
            block_scores[i] = (bv, -float(_estimate_attack(action, state, card)))
            block_candidates.append(i)
        if block_candidates:
            return _best_index(block_candidates, block_scores)
        return pass_index

    # ── 7. Defense phase ─────────────────────────────────────────────────────
    if phase in _DEFENSE_PHASES:
        zone_cache: dict[str, list] = {z: state.get(sk, []) for z, sk in _ZONE_KEY_MAP.items()
        }
        scores: dict[int, tuple[float, ...]] = {}
        candidates: list[int] = []
        for i in non_pass:
            action = legal_actions[i]
            card = _match_action_card(action, state, zone_cache)
            bv = float(_estimate_defense(action, state, card))
            if bv < _MIN_BLOCK_VALUE:
                continue
            scores[i] = (bv, -float(_estimate_attack(action, state, card)))
            candidates.append(i)
        if candidates:
            return _best_index(candidates, scores)
        return pass_index

    # ── 8. Main phase (M) and unrecognised ──────────────────────────────────
    zones  = [_normalize(a.get("zone",  "")) for a in legal_actions]
    labels = [_normalize(a.get("label", "")) for a in legal_actions]

    # 8a. Popup/button yes-confirm.
    yes_like = [
        i for i in non_pass
        if zones[i] in {"button", "popup"}
        and any(t in labels[i] for t in _YES_LABELS)
    ]
    if yes_like:
        return yes_like[0]

    zone_cache = {z: state.get(sk, []) for z, sk in _ZONE_KEY_MAP.items()}

    # 8b. Best action-27 hand card (attack / chain link).
    atk_scores: dict[int, tuple[float, ...]] = {}
    atk_candidates: list[int] = []
    for i in non_pass:
        action = legal_actions[i]
        if _to_int(action.get("action_code", 0)) != 27:
            continue
        # Skip cards known to be unaffordable (returned empty pitch window).
        if action.get("label", "") in unaffordable:
            continue
        card = _match_action_card(action, state, zone_cache)
        av = float(_estimate_attack(action, state, card))
        # Give every action-27 card a baseline score so nothing is skipped.
        if av <= 0:
            av = 0.1
        cost = float(_to_int((card or {}).get("cost", 0), 0))
        dv   = float(_estimate_defense(action, state, card))
        atk_scores[i] = (av, -cost, -dv)
        atk_candidates.append(i)
    if atk_candidates:
        return _best_index(atk_candidates, atk_scores)

    # 8c. Other button/popup non-pass actions (mandatory CHOOSE* popups, etc.).
    # This handles all remaining CanPassPhase=0 popup-based phases that fall
    # through to here (e.g. CHOOSEDECK, CHOOSEDISCARD, HANDTOPBOTTOM, etc.)
    # since their cards appear as zone="popup" action_code=16/11/12/13.
    btns = [i for i in non_pass if zones[i] in {"button", "popup"}]
    if btns:
        return btns[0]

    # 8d. Pass.
    # Deliberately skip equip (action 3): equipping opens a pitch window that may
    # only offer Cancel (no-cost equipment still shows an empty pitch window), and
    # repeatedly trying to equip then Cancel creates an infinite loop.
    return pass_index
