"""Flesh and Blood comprehensive-rules helpers (CR v2.x).

References: https://rules.fabtcg.com/en/cr/08-keywords/
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Action card types per CR — used by Overpower (8.3.22).
ACTION_CARD_TYPES = frozenset(
    {"attack_action", "utility_action", "defense_reaction", "attack_reaction", "instant"}
)

# Recognized keyword tokens in rules text (CR section 8).
KEYWORD_PATTERNS: dict[str, str] = {
    "go_again": r"\bgo again\b",
    "dominate": r"\bdominate\b",
    "overpower": r"\boverpower\b",
    "intimidate": r"\bintimidate\b",
    "ward": r"\bward\b",
    "fusion": r"\bfusion\b",
    "battleworn": r"\bbattleworn\b",
    "blade_break": r"\bblade break\b",
    "arcane_barrier": r"\barcane barrier\b",
    "temper": r"\btemper\b",
    "boost": r"\bboost\b",
    "reload": r"\breload\b",
    "blood_debt": r"\bblood debt\b",
    "phantasm": r"\bphantasm\b",
    "stealth": r"\bstealth\b",
    "crush": r"\bcrush\b",
    "combo": r"\bcombo\b",
    "ambush": r"\bambush\b",
}

# When a static keyword effect fires (trigger → effect in the engine).
KEYWORD_TRIGGER_WHEN: dict[str, str] = {
    "go_again": "on_play",
    "dominate": "on_attack",
    "overpower": "on_attack",
    "intimidate": "on_hit",
    "blood_debt": "on_play",
}


def is_action_card(card_types: tuple[str, ...] | list[str]) -> bool:
    return bool(ACTION_CARD_TYPES.intersection(card_types))


def _clean_text(text: str) -> str:
    return str(text or "").replace("{br}", ". ").replace("**", "")


def _sentences(text: str) -> list[str]:
    body = _clean_text(text).strip()
    if not body:
        return []
    parts = re.split(r"(?<=[.!?])\s+", body)
    return [p.strip(" .") for p in parts if p.strip(" .")]


def keyword_unconditional_in_sentence(sentence: str, keyword: str) -> bool:
    """True when *keyword* appears in *sentence* without an if/when/unless gate."""
    pattern = KEYWORD_PATTERNS.get(keyword)
    if not pattern:
        return False
    low = sentence.lower()
    for match in re.finditer(pattern, low):
        prefix = low[: match.start()]
        # "Fusion — reveal ..." is the fusion keyword line, not a conditional rider.
        if keyword == "fusion" and re.search(r"fusion\s*[-–—]", prefix + low[match.start() : match.end()]):
            return True
        if re.search(r"\b(if|when|unless|whenever|while)\b", prefix):
            continue
        return True
    return False


def keyword_active_in_text(text: str, keyword: str) -> bool:
    """True when *keyword* is printed as an unconditional rules keyword in *text*."""
    sentences = _sentences(text)
    if not sentences:
        return False
    return any(keyword_unconditional_in_sentence(s, keyword) for s in sentences)


def active_keywords(
    stored: tuple[str, ...] | list[str] | None,
    text: str,
) -> frozenset[str]:
    """Keywords that apply mechanically — excludes conditional mentions only."""
    stored_set = set(stored or ())
    active: set[str] = set()
    sentences = _sentences(text)
    for keyword in KEYWORD_PATTERNS:
        if sentences:
            if keyword_active_in_text(text, keyword):
                active.add(keyword)
        elif keyword in stored_set:
            active.add(keyword)
    return frozenset(active)


def has_active_keyword(
    stored: tuple[str, ...] | list[str] | None,
    text: str,
    keyword: str,
) -> bool:
    return keyword in active_keywords(stored, text)


def derive_keywords_from_text(text: str) -> tuple[str, ...]:
    """Extract unconditional keywords from rules text."""
    return tuple(sorted(active_keywords((), text)))


def resolve_keywords(
    stored: tuple[str, ...] | list[str] | None,
    text: str,
) -> tuple[str, ...]:
    """Merge stored + text keywords, keeping only unconditional ones."""
    merged = set(stored or ())
    merged.update(derive_keywords_from_text(text))
    return tuple(sorted(active_keywords(tuple(merged), text)))


def keyword_trigger_when(keyword: str) -> str | None:
    return KEYWORD_TRIGGER_WHEN.get(keyword)


def parse_ward_value(text: str, keywords: tuple[str, ...] | list[str] = (), *, has_blue_pitched: bool = False) -> int:
    """Ward N — destroy this to prevent N damage (CR 8.3.20)."""
    body = str(text or "").lower()
    m = re.search(
        r"ward x,? where x is (\d+) if you(?:'ve| have) pitched a blue card this turn, otherwise (\d+)",
        body,
    )
    if m:
        return int(m.group(1)) if has_blue_pitched else int(m.group(2))
    active = active_keywords(keywords, text)
    if "ward" not in active and "ward" not in body:
        return 0
    m = re.search(r"ward (\d+)", body)
    return int(m.group(1)) if m else 0


def parse_piercing_value(text: str) -> int:
    """Piercing N — +N power when defended by equipment (CR 8.3.23)."""
    m = re.search(r"piercing (\d+)", str(text or "").lower())
    return int(m.group(1)) if m else 0


@dataclass(frozen=True)
class ClashResult:
    attacker_wins: bool
    defender_wins: bool
    tie: bool
    attacker_power: int
    defender_power: int


def resolve_clash(attacker_top_power: int | None, defender_top_power: int | None) -> ClashResult:
    """Clash — reveal deck tops, greatest power wins (CR 8.5.45)."""
    a = attacker_top_power
    d = defender_top_power
    if a is None and d is None:
        return ClashResult(False, False, True, 0, 0)
    if a is None:
        return ClashResult(False, True, False, 0, d or 0)
    if d is None:
        return ClashResult(True, False, False, a, 0)
    if a > d:
        return ClashResult(True, False, False, a, d)
    if d > a:
        return ClashResult(False, True, False, a, d)
    return ClashResult(False, False, True, a, d)
