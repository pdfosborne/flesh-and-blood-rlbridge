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


def _pitch_color(card: dict[str, Any]) -> int:
    pitch = _to_int(card.get("pitch"))
    if pitch in _COLOR_PITCH.values():
        return pitch
    cid = _card_key(card).lower()
    for color, val in _COLOR_PITCH.items():
        if cid.endswith(f"_{color}"):
            return val
    return 0


def _played_color_this_turn(state: dict[str, Any], color: str) -> bool:
    target = _COLOR_PITCH.get(color.lower(), 0)
    if target <= 0:
        return False
    for zone_key in ("playerPitch", "cardsPlayedThisTurn", "thisTurnPitch"):
        zone = state.get(zone_key, [])
        if not isinstance(zone, list):
            continue
        for card in zone:
            if isinstance(card, dict) and _pitch_color(card) == target:
                return True
    return bool(state.get(f"played{color.capitalize()}ThisTurn"))


def _aura_count(state: dict[str, Any]) -> int:
    auras = state.get("playerAuras", [])
    if not isinstance(auras, list):
        return 0
    return sum(1 for c in auras if isinstance(c, dict) and _card_key(c))


def classify_card_patterns(card_id: str, text: str, keywords: list[str] | None = None) -> list[str]:
    """Offline classifier used by build_conditional_patterns.py."""
    body = str(text or "")
    kws = {str(k).lower().replace(" ", "_") for k in (keywords or derive_keywords_from_text(body))}
    patterns: list[str] = []

    if _PLAYED_COLOR_RE.search(body):
        match = _PLAYED_COLOR_RE.search(body)
        if match:
            patterns.append(f"played_{match.group(1).lower()}_this_turn")
    if _AURA_COST_RE.search(body):
        patterns.append("cost_reduction_per_aura")
    if _WHEN_ATTACKS_RE.search(body):
        patterns.append("when_attacks")
    if _WHEN_DEFENDS_RE.search(body):
        patterns.append("when_defends")

    static_kw = {
        "go_again", "dominate", "intimidate", "overpower", "ward", "battleworn",
        "blade_break", "phalanx", "stealth", "boost", "reload", "transcend",
    }
    if kws & static_kw and not patterns:
        patterns.append("unconditional")

    if not patterns:
        patterns.append("unconditional")
    return patterns


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

    phase_l = str(phase or state.get("turnPhase", "") or "").strip().lower()

    for tag in patterns:
        if tag == "unconditional":
            continue
        if tag.startswith("played_") and tag.endswith("_this_turn"):
            color = tag[len("played_") : -len("_this_turn")]
            if not _played_color_this_turn(state, color):
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

    return 1.0
