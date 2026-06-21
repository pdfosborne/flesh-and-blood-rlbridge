"""Fuzzy card search over the rlbridge ``cards.json`` database."""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
CARDS_DB_PATH = REPO_ROOT / "src" / "flesh_and_blood_rlbridge" / "card_db" / "cards.json"

_PITCH_SUFFIX = re.compile(r"_(red|blue|yellow|purple)$")
_FORMAT_LEGALITY_KEY = {
    "silver_age": "silver_age",
    "sage": "silver_age",
    "classic_constructed": "classic_constructed",
    "blitz": "blitz",
    "upf": "ultimate_pit_fight",
}


@dataclass(frozen=True)
class CardHit:
    card_id: str
    name: str
    pitch: Optional[int] = None
    cost: Optional[int] = None
    power: Optional[int] = None
    defense: Optional[int] = None
    card_class: str = ""
    type_line: str = ""


def _is_play_card_id(card_id: str) -> bool:
    cid = card_id.strip()
    if not cid or cid.endswith("_token") or "_token_" in cid:
        return False
    if _PITCH_SUFFIX.search(cid):
        return True
    if "-" in cid and "_" not in cid:
        return False
    return "_" in cid and not cid.startswith("fab-")


def _format_legal(rec: dict[str, Any], game_format: str) -> bool:
    key = _FORMAT_LEGALITY_KEY.get(game_format.lower(), game_format.lower())
    legality = rec.get("legality") or {}
    status = str(legality.get(key, "legal")).lower()
    return status not in {"banned", "not_legal", "illegal"}


def _score_query(query: str, hit: CardHit) -> float:
    q = query.strip().lower()
    if not q:
        return 0.0
    name = hit.name.lower()
    cid = hit.card_id.lower()
    if q == cid:
        return 200.0
    if q == name:
        return 190.0
    if name.startswith(q):
        return 150.0 + len(q)
    if q in name:
        return 120.0 + len(q) / max(len(name), 1) * 20.0
    if q in cid.replace("_", " "):
        return 100.0
    tokens = [t for t in re.split(r"\s+", q) if t]
    if tokens and all(t in name for t in tokens):
        return 90.0 + len(tokens) * 5.0
    return difflib.SequenceMatcher(None, q, name).ratio() * 80.0


@lru_cache(maxsize=4)
def _load_index(game_format: str) -> tuple[CardHit, ...]:
    try:
        records: list[dict[str, Any]] = json.loads(
            CARDS_DB_PATH.read_text(encoding="utf-8")
        )
    except OSError:
        return ()

    by_name_pitch: dict[tuple[str, int | None], CardHit] = {}
    for rec in records:
        cid = str(rec.get("id") or "").strip()
        if not cid or not _is_play_card_id(cid):
            continue
        if not _format_legal(rec, game_format):
            continue
        name = str(rec.get("name") or cid.replace("_", " ").title())
        pitch = rec.get("pitch")
        pitch_key = int(pitch) if pitch is not None else None
        key = (name.lower(), pitch_key)
        hit = CardHit(
            card_id=cid,
            name=name,
            pitch=pitch_key,
            cost=int(rec["cost"]) if rec.get("cost") is not None else None,
            power=int(rec["power"]) if rec.get("power") is not None else None,
            defense=int(rec["defense"]) if rec.get("defense") is not None else None,
            card_class=str(rec.get("class") or ""),
            type_line=str(rec.get("type_line") or ""),
        )
        existing = by_name_pitch.get(key)
        if existing is None or ("_" in cid and "-" not in cid):
            by_name_pitch[key] = hit

    return tuple(by_name_pitch.values())


class CardSearchIndex:
    """Search playable cards for a given format."""

    def __init__(self, game_format: str = "silver_age") -> None:
        self.game_format = game_format
        self._cards = _load_index(game_format)

    def display_name(self, card_id: str) -> str:
        token = card_id.lower()
        for hit in self._cards:
            if hit.card_id == card_id or hit.card_id.replace("_", "-") == token:
                return hit.name
        return card_id.replace("_", " ").title()

    def search(self, query: str, *, limit: int = 12) -> list[CardHit]:
        q = query.strip()
        if not q:
            return list(self._cards[:limit])
        scored = [
            ( _score_query(q, hit), hit)
            for hit in self._cards
        ]
        scored = [(score, hit) for score, hit in scored if score >= 35.0]
        scored.sort(key=lambda item: (-item[0], item[1].name, item[1].card_id))
        return [hit for _, hit in scored[:limit]]

    def lookup(self, card_id: str) -> Optional[CardHit]:
        token = card_id.strip().lower()
        for hit in self._cards:
            if hit.card_id == card_id or hit.card_id.lower() == token:
                return hit
        return None
