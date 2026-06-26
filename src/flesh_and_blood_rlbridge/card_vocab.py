"""Stable card / hero / format indices for player-fair observations."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .card_db.talishar_card_ids import TalisharCardIdResolver, load_talishar_card_ids

_CARD_DB = Path(__file__).parent / "card_db" / "cards.json"
_HERO_DB = Path(__file__).parent / "card_db" / "heroes.json"

_FORMATS = (
    "silver_age",
    "classic_constructed",
    "cc",
    "blitz",
    "commoner",
    "clash",
    "open",
    "sage",
)

_CARD_BACK_RE = re.compile(r"^card_back|^blank|^WTR000$", re.I)


@lru_cache(maxsize=1)
def _card_records() -> list[dict]:
    try:
        raw = json.loads(_CARD_DB.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [rec for rec in raw if isinstance(rec, dict)] if isinstance(raw, list) else []


@lru_cache(maxsize=1)
def _hero_records() -> list[dict]:
    try:
        raw = json.loads(_HERO_DB.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [rec for rec in raw if isinstance(rec, dict)] if isinstance(raw, list) else []


@lru_cache(maxsize=1)
def _build_card_index() -> dict[str, int]:
    index: dict[str, int] = {}
    next_id = 1
    resolver = TalisharCardIdResolver()

    def register(cid: str) -> None:
        nonlocal next_id
        token = str(cid or "").strip().lower()
        if not token or token in index:
            return
        index[token] = next_id
        next_id += 1

    for rec in _card_records():
        register(str(rec.get("id", "") or ""))

    for rec in _hero_records():
        register(str(rec.get("id", "") or ""))
        weapon = rec.get("weapon")
        if isinstance(weapon, dict):
            register(str(weapon.get("id", "") or ""))

    for talishar_id in load_talishar_card_ids():
        resolved = resolver.resolve(talishar_id) or talishar_id
        register(resolved)
        register(talishar_id)

    return index


@lru_cache(maxsize=1)
def _build_hero_index() -> dict[str, int]:
    index: dict[str, int] = {}
    next_id = 1
    for rec in _hero_records():
        hid = str(rec.get("id", "") or "").strip().lower()
        if not hid:
            continue
        index[hid] = next_id
        next_id += 1
        for variant in (hid.replace("hero_", ""), hid.split("_")[1] if "_" in hid else ""):
            if variant and variant not in index:
                index[variant] = index[hid]
    return index


def vocab_size() -> int:
    return len(_build_card_index()) + 1


def card_index(card_id: str, *, resolver: Optional[TalisharCardIdResolver] = None) -> int:
    """Return a stable 1..N index for *card_id*; 0 means unknown / hidden."""
    token = str(card_id or "").strip().lower()
    if not token or _CARD_BACK_RE.search(token):
        return 0
    if resolver is not None:
        resolved = resolver.resolve(token) or token
        token = str(resolved).strip().lower()
    idx = _build_card_index().get(token)
    if idx is not None:
        return idx
    collapsed = token.replace("-", "_")
    return _build_card_index().get(collapsed, 0)


def card_index_normalized(card_id: str, *, resolver: Optional[TalisharCardIdResolver] = None) -> float:
    return float(card_index(card_id, resolver=resolver)) / float(max(vocab_size(), 1))


def hero_index(hero_id: str) -> int:
    token = str(hero_id or "").strip().lower()
    if not token:
        return 0
    heroes = _build_hero_index()
    if token in heroes:
        return heroes[token]
    if not token.startswith("hero_"):
        return heroes.get(f"hero_{token}", 0)
    return 0


def hero_vocab_size() -> int:
    """Number of hero embedding slots (0 = unknown/padding)."""
    return len(_build_hero_index()) + 1


def hero_index_normalized(hero_id: str) -> float:
    size = max(hero_vocab_size(), 1)
    return float(hero_index(hero_id)) / float(size)


def format_index(game_format: str) -> int:
    token = str(game_format or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "silver age": "silver_age",
        "silverage": "silver_age",
        "classic constructed": "classic_constructed",
        "classicconstructed": "classic_constructed",
    }
    token = aliases.get(token, token)
    if token == "cc":
        token = "classic_constructed"
    if token == "sage":
        token = "silver_age"
    try:
        return _FORMATS.index(token) + 1
    except ValueError:
        return 0


def format_index_normalized(game_format: str) -> float:
    return float(format_index(game_format)) / float(len(_FORMATS) + 1)
