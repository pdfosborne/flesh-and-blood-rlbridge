"""Equipment loadout helpers for the sideboard TUI."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

from fab_tui.card_search import CARDS_DB_PATH, CardHit, _format_legal, _score_query
from flesh_and_blood_rlbridge.sideboard_guide_policy import (
    _card_meta,
    classify_opponent_archetype,
    score_card_for_archetype,
)

_SLOT_SUFFIX_MAP: tuple[tuple[str, str], ...] = (
    ("Head", "head"),
    ("Chest", "chest"),
    ("Arms", "arms"),
    ("Legs", "legs"),
    ("Off-Hand", "off_hand"),
    ("Quiver", "off_hand"),
)

_EQUIP_HEAD_PAT = frozenset(
    ["helm", "hood", "crown", "cap", "headband", "goggles", "mask", "hat", "visor", "tiara", "circlet"]
)
_EQUIP_CHEST_PAT = frozenset(
    ["coat", "robe", "vest", "chestplate", "chest", "jacket", "tunic", "cuirass", "cloak", "cape", "mantle", "doublet"]
)
_EQUIP_ARMS_PAT = frozenset(["gauntlet", "glove", "bracer", "vambrace", "bangle", "shuko", "sleeve", "handwrap"])
_EQUIP_LEGS_PAT = frozenset(
    ["boots", "greaves", "pants", "leggings", "sabaton", "sabatons", "footwrap", "shin", "paws"]
)
_EQUIP_WEAPON_PAT = frozenset(
    ["kodachi", "dawnblade", "rosetta", "galaxia", "pistol", "sword", "axe", "staff", "bow", "harpoon", "blade", "katana", "scimitar", "bauble"]
)

_SLOT_LABELS = {
    "hero": "Hero",
    "weapon": "Weapon",
    "head": "Head",
    "chest": "Chest",
    "arms": "Arms",
    "legs": "Legs",
    "off_hand": "Off-hand",
    "other": "Equipment",
}


@dataclass(frozen=True)
class EquipmentSlotEntry:
    index: int
    slot: str
    card_id: str
    label: str


@lru_cache(maxsize=1)
def _full_card_db() -> dict[str, dict[str, Any]]:
    try:
        records: list[dict[str, Any]] = json.loads(
            CARDS_DB_PATH.read_text(encoding="utf-8")
        )
    except OSError:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for rec in records:
        cid = str(rec.get("id") or "").strip()
        if not cid:
            continue
        out[cid] = rec
        hyphen = cid.replace("_", "-")
        if hyphen not in out:
            out[hyphen] = rec
    return out


def _type_line(card_id: str) -> str:
    db = _full_card_db()
    return str(
        db.get(card_id, {}).get("type_line")
        or db.get(card_id.replace("_", "-"), {}).get("type_line")
        or ""
    ).strip()


def equipment_slot(card_id: str, *, hero_id: str = "") -> str:
    """Classify an equipment / weapon card id into a loadout slot."""
    cid = card_id.lower()
    tl = _type_line(card_id)
    tl_lower = tl.lower()

    if tl:
        if "weapon" in tl_lower:
            return "weapon"
        if "equipment" in tl_lower:
            suffix = tl.split(" - ")[-1] if " - " in tl else ""
            for sfx_key, slot_name in _SLOT_SUFFIX_MAP:
                if suffix.endswith(sfx_key):
                    return slot_name
            return "other"
        if "hero" in tl_lower or "character" in tl_lower:
            return "hero"

    if hero_id and hero_id.split("_")[0].lower() in cid:
        return "hero"
    if any(p in cid for p in _EQUIP_HEAD_PAT):
        return "head"
    if any(p in cid for p in _EQUIP_CHEST_PAT):
        return "chest"
    if any(p in cid for p in _EQUIP_ARMS_PAT):
        return "arms"
    if any(p in cid for p in _EQUIP_LEGS_PAT):
        return "legs"
    if any(p in cid for p in _EQUIP_WEAPON_PAT):
        return "weapon"
    return "other"


def parse_equipment_header(
    equipment_header: str,
    *,
    hero_id: str,
    display_name,
) -> list[EquipmentSlotEntry]:
    """Split an equipment header into indexed slot entries."""
    parts = (equipment_header or "").strip().split()
    if not parts:
        return []
    hero_token = hero_id.replace("-", "_").lower()
    entries: list[EquipmentSlotEntry] = []
    for index, card_id in enumerate(parts, start=1):
        slot = equipment_slot(card_id, hero_id=hero_id)
        if slot == "hero" or card_id.replace("-", "_").lower() == hero_token:
            slot = "hero"
        entries.append(
            EquipmentSlotEntry(
                index=index,
                slot=slot,
                card_id=card_id,
                label=display_name(card_id),
            )
        )
    return entries


def rebuild_equipment_header(hero_id: str, pieces: list[str]) -> str:
    hero_token = hero_id.replace("-", "_").strip()
    if not hero_token:
        return " ".join(pieces).strip()
    if not pieces:
        return hero_token
    if pieces[0].replace("-", "_").lower() == hero_token.lower():
        return " ".join(pieces).strip()
    return f"{hero_token} {' '.join(pieces)}".strip()


def _equipment_pieces_from_header(equipment_header: str, *, hero_id: str) -> list[str]:
    parts = (equipment_header or "").strip().split()
    if not parts:
        return []
    hero_token = hero_id.replace("-", "_").lower()
    if parts[0].replace("-", "_").lower() == hero_token:
        return parts[1:]
    return parts


def suggest_guide_equipment_header(
    equipment_header: str,
    *,
    hero_id: str,
    opponent_hero_id: str,
    game_format: str,
    pool_by_id: Optional[dict[str, dict[str, Any]]] = None,
) -> str:
    """Recommend equipment swaps using guide-policy archetype scoring."""
    pieces = _equipment_pieces_from_header(equipment_header, hero_id=hero_id)
    if not pieces:
        return equipment_header

    archetype = classify_opponent_archetype(opponent_hero_id)
    pool_meta = pool_by_id or {}
    catalog = EquipmentSearchIndex(game_format, hero_id=hero_id).all_hits()
    by_slot: dict[str, list[str]] = {}
    for hit in catalog:
        slot = equipment_slot(hit.card_id, hero_id=hero_id)
        by_slot.setdefault(slot, []).append(hit.card_id)

    upgraded: list[str] = []
    for piece in pieces:
        slot = equipment_slot(piece, hero_id=hero_id)
        current_meta = _card_meta(piece, pool_meta)
        current_score = score_card_for_archetype(piece, current_meta, archetype)
        best_id = piece
        best_score = current_score
        for alt_id in by_slot.get(slot, []):
            alt_meta = _card_meta(alt_id, pool_meta)
            alt_score = score_card_for_archetype(alt_id, alt_meta, archetype)
            if alt_score > best_score:
                best_score = alt_score
                best_id = alt_id
        upgraded.append(best_id)

    hero_token = (equipment_header or "").strip().split()[0]
    if hero_token.replace("-", "_").lower() == hero_id.replace("-", "_").lower():
        return rebuild_equipment_header(hero_id, [hero_token, *upgraded])
    return rebuild_equipment_header(hero_id, upgraded)


class EquipmentSearchIndex:
    """Search equipment and weapon cards for a format / hero."""

    def __init__(self, game_format: str = "silver_age", *, hero_id: str = "") -> None:
        self.game_format = game_format
        self.hero_id = hero_id.replace("-", "_").lower()
        self._hits = self._load_hits()

    def _load_hits(self) -> tuple[CardHit, ...]:
        try:
            records: list[dict[str, Any]] = json.loads(
                CARDS_DB_PATH.read_text(encoding="utf-8")
            )
        except OSError:
            return ()

        hero_token = self.hero_id.split("_")[0] if self.hero_id else ""
        hits: list[CardHit] = []
        seen: set[str] = set()
        for rec in records:
            cid = str(rec.get("id") or "").strip()
            if not cid or cid in seen:
                continue
            type_line = str(rec.get("type_line") or "")
            tl_lower = type_line.lower()
            if not any(token in tl_lower for token in ("equipment", "weapon")):
                continue
            if not _format_legal(rec, self.game_format):
                continue
            card_class = str(rec.get("class") or "").lower()
            if card_class and card_class not in {"generic", ""}:
                if hero_token and hero_token not in card_class.replace(" ", "_"):
                    if self.hero_id and self.hero_id.split("_")[0] not in card_class:
                        continue
            seen.add(cid)
            hits.append(
                CardHit(
                    card_id=cid,
                    name=str(rec.get("name") or cid.replace("_", " ").title()),
                    type_line=type_line,
                )
            )
        hits.sort(key=lambda hit: (hit.name.lower(), hit.card_id))
        return tuple(hits)

    def all_hits(self) -> tuple[CardHit, ...]:
        return self._hits

    def display_name(self, card_id: str) -> str:
        token = card_id.strip().lower()
        for hit in self._hits:
            if hit.card_id == card_id or hit.card_id.lower() == token:
                return hit.name
        return card_id.replace("_", " ").title()

    def search(self, query: str, *, slot: str | None = None, limit: int = 12) -> list[CardHit]:
        q = query.strip()
        hits = list(self._hits)
        if slot:
            hits = [
                hit
                for hit in hits
                if equipment_slot(hit.card_id, hero_id=self.hero_id) == slot
            ]
        if not q:
            return hits[:limit]
        scored = [( _score_query(q, hit), hit) for hit in hits]
        scored = [(score, hit) for score, hit in scored if score >= 35.0]
        scored.sort(key=lambda item: (-item[0], item[1].name, item[1].card_id))
        return [hit for _, hit in scored[:limit]]

    def lookup(self, card_id: str) -> Optional[CardHit]:
        token = card_id.strip().lower()
        for hit in self._hits:
            if hit.card_id == card_id or hit.card_id.lower() == token:
                return hit
        meta = _full_card_db().get(card_id) or _full_card_db().get(card_id.replace("_", "-"))
        if not meta:
            return None
        return CardHit(
            card_id=card_id,
            name=str(meta.get("name") or card_id.replace("_", " ").title()),
            type_line=str(meta.get("type_line") or ""),
        )


def slot_display_name(slot: str) -> str:
    return _SLOT_LABELS.get(slot, slot.replace("_", " ").title())
