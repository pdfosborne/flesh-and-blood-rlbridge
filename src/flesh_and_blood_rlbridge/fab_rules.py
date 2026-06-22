"""FAB rules-text helpers shared across card DB and simulator tooling."""

from __future__ import annotations

import re
from typing import Iterable

# Stable output order; multi-word patterns before single-word overlaps.
_KEYWORD_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        ("go_again", r"\bgo again\b"),
        ("blood_debt", r"\bblood debt\b"),
        ("arcane_barrier", r"\barcane barrier\b"),
        ("blade_break", r"\bblade break\b"),
        ("battleworn", r"\bbattleworn\b"),
        ("overpower", r"\boverpower\b"),
        ("intimidate", r"\bintimidate\b"),
        ("dominate", r"\bdominate\b"),
        ("transcend", r"\btranscend\b"),
        ("contract", r"\bcontract\b"),
        ("phalanx", r"\bphalanx\b"),
        ("stealth", r"\bstealth\b"),
        ("reload", r"\breload\b"),
        ("boost", r"\bboost\b"),
        ("fusion", r"\b(?:[a-z]+ fusion|fused)\b"),
        ("ward", r"\*\*ward\b|\bward \d"),
        ("temper", r"\btemper\b"),
        ("crush", r"\bcrush\b"),
        ("surge", r"\bsurge\b"),
        ("clash", r"\bclash\b"),
        ("opt", r"\bopt \d"),
    )
)

_WARD_RE = re.compile(r"\*\*ward\s+(\d+)\*\*|\bward\s+(\d+)\b", re.IGNORECASE)


def _normalize_text(text: str) -> str:
    return str(text or "").replace("{br}", "\n")


def derive_keywords_from_text(text: str) -> list[str]:
    """Extract normalized keyword tags from card rules text."""
    body = _normalize_text(text)
    if not body.strip():
        return []
    found: list[str] = []
    seen: set[str] = set()
    for name, pattern in _KEYWORD_RULES:
        if name in seen:
            continue
        if pattern.search(body):
            seen.add(name)
            found.append(name)
    return found


def parse_ward_value(
    text: str,
    keywords: Iterable[str],
    *,
    has_blue_pitched: bool = False,
) -> int:
    """Return the Ward N value on equipment, or 0 if none."""
    del has_blue_pitched  # reserved for future equipment scaling rules
    body = _normalize_text(text)
    match = _WARD_RE.search(body)
    if match:
        return int(match.group(1) or match.group(2))
    normalized = {str(k).strip().lower().replace(" ", "_") for k in keywords}
    if "ward" in normalized:
        return 1
    return 0
