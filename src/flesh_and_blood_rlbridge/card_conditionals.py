"""Evaluate whether card conditional rules would be active at play time."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .fab_rules import derive_keywords_from_text

_PATTERNS_PATH = Path(__file__).parent / "card_db" / "conditional_patterns.json"

_COLOR_PITCH = {"red": 1, "yellow": 2, "blue": 3}
_PLAYED_COLOR_RE = re.compile(
    r"if you(?:'ve| have) played (?:a |an )?(red|yellow|blue) card this turn",
    re.I,
)
_AURA_COST_RE = re.compile(
    r"costs? .* less for each aura",
    re.I,
)
_WHEN_ATTACKS_RE = re.compile(r"\bwhen (?:this|it) attacks\b", re.I)
_WHEN_DEFENDS_RE = re.compile(r"\bwhen (?:this|it) defends\b", re.I)
_WHEN_HITS_RE = re.compile(r"\bwhen (?:this|it) hits\b", re.I)
_HAS_TURN_EFFECT_RE = re.compile(
    r"\b(?:until end of turn|this turn|next (?:attack|time)|gets? \+|gain[s]? \d)",
    re.I,
)
_NUM_CARDS_PLAYED_RE = re.compile(
    r"if you(?:'ve| have) played (?:\d+|a|an|one|two|three|\d+ or more)",
    re.I,
)

# Fixed clause vector indices (schema v3 hand slots).
CLAUSE_TAGS: tuple[str, ...] = (
    "played_red_this_turn",
    "played_yellow_this_turn",
    "played_blue_this_turn",
    "cost_reduction_per_aura",
    "when_attacks",
    "when_defends",
    "when_hits",
    "has_turn_effect",
)
HAND_CLAUSE_DIM = len(CLAUSE_TAGS)

_LAYER_TRIGGER_TYPES = frozenset({"TRIGGER", "PRETRIGGER", "MELD"})


@lru_cache(maxsize=1)
def _pattern_index() -> dict[str, list[str]]:
    if not _PATTERNS_PATH.is_file():
        return {}
    try:
        raw = json.loads(_PATTERNS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _card_key(card: dict[str, Any]) -> str:
    return str(
        card.get("cardNumber")
        or card.get("cardID")
        or card.get("card_id")
        or card.get("id")
        or ""
    )


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pitch_color(card: dict[str, Any] | str) -> int:
    if isinstance(card, str):
        cid = card.lower()
        pitch = 0
    else:
        pitch = _to_int(card.get("pitch"))
        cid = _card_key(card).lower()
    if pitch in _COLOR_PITCH.values():
        return pitch
    for color, val in _COLOR_PITCH.items():
        if cid.endswith(f"_{color}"):
            return val
    return 0


def _play_history_names(state: dict[str, Any], *, side: str = "self") -> list[str]:
    history = state.get("playHistory")
    if not isinstance(history, dict):
        return []
    key = "player" if side == "self" else "opponent"
    block = history.get(key, {})
    if not isinstance(block, dict):
        return []
    names = block.get("namesOfCardsPlayed", [])
    if not isinstance(names, list):
        return []
    return [str(n).strip().lower() for n in names if str(n).strip()]


def _played_color_this_turn(state: dict[str, Any], color: str, *, side: str = "self") -> bool:
    target = _COLOR_PITCH.get(color.lower(), 0)
    if target <= 0:
        return False

    for name in _play_history_names(state, side=side):
        if _pitch_color(name) == target:
            return True

    zone_keys = ("playerPitch", "cardsPlayedThisTurn", "thisTurnPitch")
    if side == "opp":
        zone_keys = ("opponentPitch", "opponentCardsPlayedThisTurn", "opponentThisTurnPitch")
    for zone_key in zone_keys:
        zone = state.get(zone_key, [])
        if not isinstance(zone, list):
            continue
        for card in zone:
            if isinstance(card, dict) and _pitch_color(card) == target:
                return True

    history = state.get("playHistory")
    if isinstance(history, dict):
        block = history.get("player" if side == "self" else "opponent", {})
        if isinstance(block, dict):
            color_key = f"num{color.capitalize()}Played"
            if _to_int(block.get(color_key)) > 0:
                return True

    return bool(state.get(f"played{color.capitalize()}ThisTurn"))


def _aura_count(state: dict[str, Any]) -> int:
    auras = state.get("playerAuras", [])
    if not isinstance(auras, list):
        return 0
    return sum(1 for c in auras if isinstance(c, dict) and _card_key(c))


def _phase_token(state: dict[str, Any], phase: str) -> str:
    token = phase or state.get("turnPhase", state.get("turn_phase", ""))
    if isinstance(token, dict):
        token = token.get("turnPhase", "")
    return str(token or "").strip().lower()


def _combat_chain_active(state: dict[str, Any]) -> bool:
    link = state.get("activeChainLink", {})
    if not isinstance(link, dict):
        return False
    attack = link.get("attackingCard")
    if isinstance(attack, dict) and _card_key(attack):
        return True
    reactions = link.get("reactions", [])
    return isinstance(reactions, list) and len(reactions) > 0


def _card_on_combat_chain(state: dict[str, Any], card_id: str) -> bool:
    if not card_id:
        return False
    needle = card_id.strip().lower()
    link = state.get("activeChainLink", {})
    if not isinstance(link, dict):
        return False
    attack = link.get("attackingCard")
    if isinstance(attack, dict) and _card_key(attack).lower() == needle:
        return True
    reactions = link.get("reactions", [])
    if isinstance(reactions, list):
        for card in reactions:
            if isinstance(card, dict) and _card_key(card).lower() == needle:
                return True
    return False


def _extract_effect_card_id(effect_id: str) -> str:
    raw = str(effect_id or "").strip()
    if not raw:
        return ""
    first = raw.split("-", 1)[0]
    return first.split(",", 1)[0].strip().lower()


def _has_turn_effect_for_card(state: dict[str, Any], card_id: str) -> bool:
    if not card_id:
        return False
    needle = card_id.strip().lower()
    effects = state.get("currentTurnEffects", [])
    if not isinstance(effects, list):
        return False
    for entry in effects:
        if not isinstance(entry, dict):
            continue
        effect_id = str(entry.get("effectId", entry.get("effect_id", "")) or "")
        if needle in effect_id.lower() or _extract_effect_card_id(effect_id) == needle:
            return True
    return False


def _clause_satisfied(tag: str, state: dict[str, Any], *, phase: str, side: str) -> bool:
    phase_l = _phase_token(state, phase)
    if tag == "played_red_this_turn":
        return _played_color_this_turn(state, "red", side=side)
    if tag == "played_yellow_this_turn":
        return _played_color_this_turn(state, "yellow", side=side)
    if tag == "played_blue_this_turn":
        return _played_color_this_turn(state, "blue", side=side)
    if tag == "cost_reduction_per_aura":
        return _aura_count(state) > 0
    if tag == "when_attacks":
        return phase_l in {"a", "attack", "m", "main"}
    if tag == "when_defends":
        return phase_l in {"b", "block", "d", "defense"}
    if tag == "when_hits":
        return phase_l in {"a", "attack", "b", "block", "d", "defense", "damage"} and _combat_chain_active(
            state
        )
    if tag == "has_turn_effect":
        return True
    return True


def _card_has_clause(tag: str, patterns: list[str]) -> bool:
    if tag in patterns:
        return True
    if tag == "has_turn_effect" and "has_turn_effect" in patterns:
        return True
    if tag.startswith("played_") and tag.endswith("_this_turn"):
        return tag in patterns
    return False


def classify_card_patterns(card_id: str, text: str, keywords: list[str] | None = None) -> list[str]:
    """Offline classifier used by build_conditional_patterns.py."""
    body = str(text or "")
    kws = {str(k).lower().replace(" ", "_") for k in (keywords or derive_keywords_from_text(body))}
    patterns: list[str] = []

    for match in _PLAYED_COLOR_RE.finditer(body):
        patterns.append(f"played_{match.group(1).lower()}_this_turn")
    if _AURA_COST_RE.search(body):
        patterns.append("cost_reduction_per_aura")
    if _WHEN_ATTACKS_RE.search(body):
        patterns.append("when_attacks")
    if _WHEN_DEFENDS_RE.search(body):
        patterns.append("when_defends")
    if _WHEN_HITS_RE.search(body):
        patterns.append("when_hits")
    if _HAS_TURN_EFFECT_RE.search(body) or _NUM_CARDS_PLAYED_RE.search(body):
        patterns.append("has_turn_effect")

    static_kw = {
        "go_again", "dominate", "intimidate", "overpower", "ward", "battleworn",
        "blade_break", "phalanx", "stealth", "boost", "reload", "transcend",
    }
    if kws & static_kw and not patterns:
        patterns.append("unconditional")

    if not patterns:
        patterns.append("unconditional")
    return patterns


def evaluate_clause_vector(
    card_id: str,
    state: dict[str, Any],
    *,
    zone: str = "hand",
    side: str = "self",
    phase: str = "",
    card_visible: bool = True,
) -> list[float]:
    """Return per-clause values: 0=N/A, 1=satisfied, 0.5=required but unsatisfied."""
    out = [0.0] * HAND_CLAUSE_DIM
    if not card_visible or not str(card_id or "").strip():
        return out

    patterns = _pattern_index().get(str(card_id).strip().lower(), ["unconditional"])
    if not patterns:
        patterns = ["unconditional"]

    cid = str(card_id).strip().lower()
    for idx, tag in enumerate(CLAUSE_TAGS):
        if tag == "has_turn_effect":
            if "has_turn_effect" not in patterns:
                continue
            if _has_turn_effect_for_card(state, cid):
                out[idx] = 1.0
            else:
                out[idx] = 0.5
            continue

        if not _card_has_clause(tag, patterns):
            continue

        if _clause_satisfied(tag, state, phase=phase, side=side):
            if tag == "when_hits" and not _card_on_combat_chain(state, cid):
                out[idx] = 0.5
            else:
                out[idx] = 1.0
        else:
            out[idx] = 0.5

    return out


def evaluate_conditional_active(
    card_id: str,
    state: dict[str, Any],
    *,
    zone: str = "hand",
    side: str = "self",
    phase: str = "",
    card_visible: bool = True,
) -> float:
    """Return 1.0 when conditional clauses would fire; 0.0 when hidden or known false."""
    if not card_visible or not str(card_id or "").strip():
        return 0.0

    patterns = _pattern_index().get(str(card_id).strip().lower(), ["unconditional"])
    if not patterns:
        patterns = ["unconditional"]

    phase_l = _phase_token(state, phase)

    for tag in patterns:
        if tag == "unconditional":
            continue
        if tag.startswith("played_") and tag.endswith("_this_turn"):
            color = tag[len("played_") : -len("_this_turn")]
            if not _played_color_this_turn(state, color, side=side):
                return 0.0
        elif tag == "cost_reduction_per_aura":
            if _aura_count(state) <= 0:
                return 0.0
        elif tag == "when_attacks":
            if phase_l not in {"a", "attack", "m", "main"}:
                return 0.0
        elif tag == "when_defends":
            if phase_l not in {"b", "block", "d", "defense"}:
                return 0.0
        elif tag == "when_hits":
            if not _combat_chain_active(state):
                return 0.0
        elif tag == "has_turn_effect":
            if not _has_turn_effect_for_card(state, str(card_id)):
                return 0.0

    return 1.0
