"""Equipment loadout helpers for the sideboard TUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

from fab_tui.card_search import CARDS_DB_PATH, CardHit, _format_legal, _score_query
from fab_tui.card_classification import classification_from_record
from flesh_and_blood_rlbridge.card_db.talishar_card_ids import load_talishar_card_subtypes
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

_TALISHAR_SUBTYPE_SLOTS: dict[str, str] = {
    "head": "head",
    "chest": "chest",
    "arms": "arms",
    "legs": "legs",
}

_TALISHAR_WEAPON_SUBTYPES = frozenset(
    {
        "sword",
        "staff",
        "bow",
        "dagger",
        "axe",
        "club",
        "hammer",
        "gun",
        "arrow",
        "quiver",
        "shield",
        "wand",
        "flail",
        "scythe",
        "knife",
        "rapier",
        "saber",
        "sabre",
        "mace",
        "spear",
        "lance",
        "adjudicator",
    }
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

# Talishar assigns the first equipment piece per slot to the active loadout;
# additional pieces of the same slot are equipment sideboard alternatives.
_LOADOUT_SLOTS = ("hero", "weapon", "head", "chest", "arms", "legs", "off_hand")
_EQUIPMENT_SLOTS = frozenset({"weapon", "head", "chest", "arms", "legs", "off_hand"})


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


def _slot_from_talishar_subtype(card_id: str) -> str | None:
    """Map Talishar ``GeneratedCardSubtype`` values to loadout slots."""
    subtypes = load_talishar_card_subtypes()
    token = card_id.replace("-", "_").lower()
    raw = subtypes.get(token)
    if not raw:
        return None
    for part in raw.split(","):
        slot = _TALISHAR_SUBTYPE_SLOTS.get(part.strip().lower())
        if slot:
            return slot
    first = raw.split(",")[0].strip().lower()
    if first in _TALISHAR_WEAPON_SUBTYPES:
        return "weapon"
    return None


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

    subtype_slot = _slot_from_talishar_subtype(card_id)
    if subtype_slot:
        return subtype_slot

    inferred = _slot_from_card_id(card_id)
    if inferred:
        return inferred
    return "other"


def split_equipment_header(
    equipment_header: str,
    *,
    hero_id: str,
) -> tuple[list[str], list[str]]:
    """Split a Talishar equipment line into active loadout and equipment sideboard."""
    parts = (equipment_header or "").strip().split()
    if not parts:
        return [], []

    active: list[str] = [parts[0]]
    sideboard: list[str] = []
    filled: set[str] = {"hero"}

    for card_id in parts[1:]:
        slot = equipment_slot(card_id, hero_id=hero_id)
        if slot == "hero":
            continue
        if slot in _EQUIPMENT_SLOTS and slot not in filled:
            active.append(card_id)
            filled.add(slot)
        elif slot in _EQUIPMENT_SLOTS or slot == "other":
            sideboard.append(card_id)

    return active, sideboard


def active_equipment_header(equipment_header: str, *, hero_id: str) -> str:
    """Return only the equipped pieces from a Talishar equipment header."""
    active, _ = split_equipment_header(equipment_header, hero_id=hero_id)
    return " ".join(active).strip()


def active_slot_map(active_parts: list[str], *, hero_id: str) -> dict[str, str]:
    """Map loadout slot → card id for an active equipment list (hero first)."""
    slot_map: dict[str, str] = {}
    for card_id in active_parts:
        slot = equipment_slot(card_id, hero_id=hero_id)
        if slot == "hero" or card_id.replace("-", "_").lower() == hero_id.replace("-", "_").lower():
            slot_map["hero"] = card_id
        elif slot in _EQUIPMENT_SLOTS and slot not in slot_map:
            slot_map[slot] = card_id
    if "hero" not in slot_map and active_parts:
        slot_map["hero"] = active_parts[0]
    return slot_map


def active_list_from_slot_map(hero_id: str, slot_map: dict[str, str]) -> list[str]:
    """Rebuild an active equipment list from a slot map."""
    hero = slot_map.get("hero") or hero_id
    ordered = [hero]
    for slot in _LOADOUT_SLOTS[1:]:
        card_id = str(slot_map.get(slot) or "").strip()
        if card_id:
            ordered.append(card_id)
    return ordered


def replace_equipment_in_slot(
    equipment_header: str,
    *,
    slot: str,
    replacement_card_id: str,
    hero_id: str,
) -> str:
    """Swap one loadout slot and preserve equipment sideboard alternatives."""
    active, sideboard = split_equipment_header(equipment_header, hero_id=hero_id)
    slot_map = active_slot_map(active, hero_id=hero_id)
    target = str(slot or "").strip().lower()
    if target not in _EQUIPMENT_SLOTS:
        return equipment_header.strip()

    old_id = str(slot_map.get(target) or "").strip()
    slot_map[target] = replacement_card_id
    new_active = active_list_from_slot_map(hero_id, slot_map)
    active_set = set(new_active)

    new_sideboard = [cid for cid in sideboard if cid not in active_set]
    if old_id and old_id not in active_set and equipment_slot(old_id, hero_id=hero_id) == target:
        new_sideboard.append(old_id)

    return " ".join([*new_active, *new_sideboard]).strip()


def parse_standard_loadout(
    equipment_header: str,
    *,
    hero_id: str,
    display_name,
    include_empty_slots: bool = True,
) -> list[EquipmentSlotEntry]:
    """Parse the active loadout, optionally including empty armor slots."""
    active, _ = split_equipment_header(equipment_header, hero_id=hero_id)
    if not active:
        return []

    slot_map = active_slot_map(active, hero_id=hero_id)
    slots = list(_LOADOUT_SLOTS[:6]) if include_empty_slots else list(_LOADOUT_SLOTS)
    entries: list[EquipmentSlotEntry] = []
    for index, slot in enumerate(slots, start=1):
        card_id = str(slot_map.get(slot) or "").strip()
        if not card_id and not include_empty_slots:
            continue
        label = display_name(card_id) if card_id else ""
        entries.append(
            EquipmentSlotEntry(
                index=index,
                slot=slot,
                card_id=card_id,
                label=label,
            )
        )
    return entries


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
    """Recommend one equipped piece per slot using guide-policy archetype scoring."""
    active, sideboard = split_equipment_header(equipment_header, hero_id=hero_id)
    if not active:
        return equipment_header

    resolved_class, resolved_talent = hero_profile_for_id(hero_id)
    hero_cls = str(hero_class or resolved_class).strip()
    hero_tal = str(hero_talent or resolved_talent).strip()
    archetype = classify_opponent_archetype(opponent_hero_id)
    pool_meta = pool_by_id or {}

    candidates_by_slot: dict[str, list[str]] = {}
    for piece in active[1:] + sideboard:
        slot = equipment_slot(piece, hero_id=hero_id)
        if slot in _EQUIPMENT_SLOTS:
            candidates_by_slot.setdefault(slot, [])
            if piece not in candidates_by_slot[slot]:
                candidates_by_slot[slot].append(piece)

    for slot, ids in _pool_equipment_by_slot(
        pool_meta,
        hero_id=hero_id,
        hero_class=hero_cls,
        hero_talent=hero_tal,
    ).items():
        for alt_id in ids:
            if alt_id not in candidates_by_slot.get(slot, []):
                candidates_by_slot.setdefault(slot, []).append(alt_id)

    slot_map = active_slot_map(active, hero_id=hero_id)
    for slot, candidates in candidates_by_slot.items():
        best_id = candidates[0]
        best_score = score_card_for_archetype(
            best_id,
            _card_meta(best_id, pool_meta),
            archetype,
        )
        for alt_id in candidates[1:]:
            alt_score = score_card_for_archetype(
                alt_id,
                _card_meta(alt_id, pool_meta),
                archetype,
            )
            if alt_score > best_score:
                best_score = alt_score
                best_id = alt_id
        slot_map[slot] = best_id

    new_active = active_list_from_slot_map(hero_id, slot_map)
    active_set = set(new_active)
    new_sideboard = [cid for cid in active[1:] + sideboard if cid not in active_set]
    return " ".join([*new_active, *new_sideboard]).strip()


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
