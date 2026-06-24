"""Equipment loadout helpers for the sideboard TUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

from fab_tui.card_search import CARDS_DB_PATH, CardHit, _format_legal, _score_query
from fab_tui.card_classification import classification_from_record
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

_KNOWN_EQUIPMENT_SLOTS: dict[str, str] = {
    "star_fall": "weapon",
    "nebula_blade": "weapon",
    "talishar_the_lost_prince": "weapon",
    "aether_ironweave": "chest",
    "garland_of_spring": "chest",
    "blossom_of_spring": "chest",
    "ironhide_plate": "chest",
    "spellbound_creepers": "legs",
    "snapdragon_scalers": "legs",
    "aether_crackers": "arms",
    "nullrune_gloves": "arms",
    "blade_beckoner_gauntlets": "arms",
    "blade_beckoner_helm": "head",
    "blade_beckoner_boots": "legs",
    "blade_beckoner_plating": "chest",
    "flail_of_agony": "weapon",
    "quiver_of_a_thousand_arrows": "weapon",
    "nullrune_robe": "chest",
    "nullrune_hood": "head",
}

_HEAD_TOKENS = frozenset(
    {"helm", "hood", "crown", "cap", "headband", "goggles", "mask", "hat", "visor", "tiara", "circlet"}
)
_CHEST_TOKENS = frozenset(
    {"plating", "robe", "vest", "tunic", "cloak", "cape", "mantle", "doublet", "jacket", "coat", "cuirass"}
)
_CHEST_SUBSTRINGS = ("chestplate",)
_ARMS_TOKENS = frozenset(
    {"gauntlet", "gauntlets", "glove", "gloves", "bracer", "bracers", "vambrace", "bangle", "shuko", "sleeve", "handwrap", "cuff"}
)
_LEGS_TOKENS = frozenset(
    {"boots", "greaves", "pants", "leggings", "sabaton", "sabatons", "footwrap", "paws"}
)
_LEGS_SUBSTRINGS = ("shin_guards",)
_WEAPON_FRAGMENTS = (
    "rosetta_thorn",
    "death_dealer",
    "driftwood_quiver",
    "cracked_bauble",
    "kodachi",
    "dawnblade",
    "galaxia",
    "pistol",
    "harpoon",
    "flail_of_agony",
    "star_fall",
    "nebula_blade",
    "quiver_of_a_thousand_arrows",
)
_WEAPON_TOKENS = frozenset(
    {"club", "thorn", "bow", "staff", "axe", "sword", "katana", "scimitar", "bauble", "quiver", "wand", "dagger", "mace", "hammer", "lance", "saber", "sabre"}
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


def _card_id_tokens(card_id: str) -> frozenset[str]:
    return frozenset(card_id.lower().split("_"))


def _slot_from_card_id(card_id: str) -> str | None:
    """Infer equipment slot from card id tokens when type_line is missing."""
    cid = card_id.lower()
    tokens = _card_id_tokens(card_id)

    known = _KNOWN_EQUIPMENT_SLOTS.get(cid)
    if known:
        return known

    if tokens & _HEAD_TOKENS:
        return "head"
    if tokens & _CHEST_TOKENS or any(part in cid for part in _CHEST_SUBSTRINGS):
        return "chest"
    if tokens & _ARMS_TOKENS:
        return "arms"
    if tokens & _LEGS_TOKENS or any(part in cid for part in _LEGS_SUBSTRINGS):
        return "legs"
    if any(frag in cid for frag in _WEAPON_FRAGMENTS) or tokens & _WEAPON_TOKENS:
        return "weapon"
    return None


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

    inferred = _slot_from_card_id(card_id)
    if inferred:
        return inferred
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


def hero_profile_for_id(hero_id: str) -> tuple[str, str]:
    """Return ``(hero_class, hero_talent)`` for a hero id."""
    rec = _hero_record(hero_id)
    if not rec:
        return "", ""
    hero_class = str(rec.get("class") or "").strip()
    hero_talent = str(rec.get("talent") or "").strip()
    return hero_class, hero_talent


def _equipment_matches_hero(
    rec: dict[str, Any],
    *,
    hero_class: str,
    hero_talent: str = "",
) -> bool:
    """True when equipment matches the hero's class and talent restrictions."""
    card_class = str(rec.get("class") or "").strip().lower()
    card_talent = str(rec.get("talent") or "").strip().lower()
    hero_cls = str(hero_class or "").strip().lower()
    hero_tal = str(hero_talent or "").strip().lower()

    if card_class and card_class != "generic":
        if not hero_cls or card_class != hero_cls:
            return False

    if card_talent:
        if not hero_tal or card_talent != hero_tal:
            return False

    return True


def _pool_equipment_by_slot(
    pool_by_id: dict[str, dict[str, Any]],
    *,
    hero_id: str,
    hero_class: str,
    hero_talent: str,
) -> dict[str, list[str]]:
    """Equipment cards from the sideboard pool, grouped by loadout slot."""
    by_slot: dict[str, list[str]] = {}
    for card_id, meta in pool_by_id.items():
        record = meta if _is_equipment_record(meta) else _full_card_db().get(card_id) or {}
        if not record or not _is_equipment_record(record):
            continue
        if not _equipment_matches_hero(record, hero_class=hero_class, hero_talent=hero_talent):
            continue
        slot = equipment_slot(card_id, hero_id=hero_id)
        if slot in {"hero", "other"}:
            continue
        by_slot.setdefault(slot, []).append(card_id)
    for slot in by_slot:
        by_slot[slot] = sorted(dict.fromkeys(by_slot[slot]))
    return by_slot


def suggest_guide_equipment_header(
    equipment_header: str,
    *,
    hero_id: str,
    opponent_hero_id: str,
    game_format: str,
    pool_by_id: Optional[dict[str, dict[str, Any]]] = None,
    hero_class: str = "",
    hero_talent: str = "",
) -> str:
    """Recommend equipment swaps using guide-policy archetype scoring."""
    pieces = _equipment_pieces_from_header(equipment_header, hero_id=hero_id)
    if not pieces:
        return equipment_header

    resolved_class, resolved_talent = hero_profile_for_id(hero_id)
    hero_cls = str(hero_class or resolved_class).strip()
    hero_tal = str(hero_talent or resolved_talent).strip()

    archetype = classify_opponent_archetype(opponent_hero_id)
    pool_meta = pool_by_id or {}
    by_slot = _pool_equipment_by_slot(
        pool_meta,
        hero_id=hero_id,
        hero_class=hero_cls,
        hero_talent=hero_tal,
    )

    upgraded: list[str] = []
    for piece in pieces:
        slot = equipment_slot(piece, hero_id=hero_id)
        candidates = [piece]
        for alt_id in by_slot.get(slot, []):
            if alt_id not in candidates:
                candidates.append(alt_id)
        current_meta = _card_meta(piece, pool_meta)
        current_score = score_card_for_archetype(piece, current_meta, archetype)
        best_id = piece
        best_score = current_score
        for alt_id in candidates:
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


def _is_equipment_record(rec: dict[str, Any]) -> bool:
    """True for weapons, armor, and other equippable items in ``cards.json``."""
    card_types = {str(t).lower() for t in (rec.get("card_types") or [])}
    if "hero" in card_types:
        return False
    if "utility_item" in card_types:
        return True
    tl_lower = str(rec.get("type_line") or "").lower()
    return "equipment" in tl_lower or "weapon" in tl_lower


def hero_class_for_id(hero_id: str) -> str:
    """Look up a hero card's class from ``cards.json``."""
    hero_class, _ = hero_profile_for_id(hero_id)
    return hero_class


def hero_talent_for_id(hero_id: str) -> str:
    """Look up a hero card's talent from ``cards.json``."""
    _, hero_talent = hero_profile_for_id(hero_id)
    return hero_talent


def _hero_record(hero_id: str) -> dict[str, Any] | None:
    token = hero_id.replace("-", "_").strip().lower()
    if not token:
        return None
    db = _full_card_db()
    for key in (hero_id, token, hero_id.replace("_", "-")):
        rec = db.get(key) or db.get(str(key).lower())
        if rec:
            return rec
    return None


class EquipmentSearchIndex:
    """Search equipment and weapon cards for a format / hero."""

    def __init__(
        self,
        game_format: str = "silver_age",
        *,
        hero_id: str = "",
        hero_class: str = "",
        hero_talent: str = "",
    ) -> None:
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

        hits: list[CardHit] = []
        seen: set[str] = set()
        for rec in records:
            cid = str(rec.get("id") or "").strip()
            if not cid or cid in seen:
                continue
            type_line = str(rec.get("type_line") or "")
            if not _is_equipment_record(rec):
                continue
            seen.add(cid)
            hits.append(
                CardHit(
                    card_id=cid,
                    name=str(rec.get("name") or cid.replace("_", " ").title()),
                    card_class=str(rec.get("class") or ""),
                    talent=str(rec.get("talent") or ""),
                    card_types=tuple(str(t) for t in (rec.get("card_types") or [])),
                    type_line=type_line,
                    classification=classification_from_record(rec),
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
            card_class=str(meta.get("class") or ""),
            talent=str(meta.get("talent") or ""),
            card_types=tuple(str(t) for t in (meta.get("card_types") or [])),
            type_line=str(meta.get("type_line") or ""),
            classification=classification_from_record(meta),
        )


def slot_display_name(slot: str) -> str:
    return _SLOT_LABELS.get(slot, slot.replace("_", " ").title())
