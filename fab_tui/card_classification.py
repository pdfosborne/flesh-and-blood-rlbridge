"""Build printed-style classification text from card DB records."""

from __future__ import annotations

import re
from typing import Any, Iterable


def _prefix_parts(*, talent: str | None, card_class: str) -> list[str]:
    parts: list[str] = []
    talent_text = str(talent or "").strip()
    if talent_text:
        parts.append(talent_text)
    class_text = str(card_class or "").strip()
    if class_text and class_text.lower() != "generic" and class_text not in parts:
        parts.append(class_text)
    return parts


def _join_type_line(prefixes: list[str], main: str, subtype: str | None = None) -> str:
    lead = " ".join(prefixes + [main]).strip()
    if subtype:
        return f"{lead} - {subtype}"
    return lead


def _infer_card_types(rec: dict[str, Any]) -> list[str]:
    """Guess normalized card types when ``cards.json`` leaves them empty."""
    existing = [str(t).lower() for t in (rec.get("card_types") or []) if str(t).strip()]
    if existing:
        return existing

    pitch = int(rec.get("pitch") or 0)
    power = int(rec.get("power") or 0)
    defense = int(rec.get("defense") or 0)
    text = str(rec.get("text") or "").lower()

    if pitch == 0:
        return []

    if "attack reaction" in text:
        return ["attack_reaction"]
    if "defense reaction" in text:
        return ["defense_reaction"]
    if power > 0 or "**attack**" in text or "when this attacks" in text or ": attack" in text:
        return ["attack_action"]
    if power == 0 and defense == 0:
        return ["instant"]
    return ["utility_action"]


def format_card_classification(
    *,
    type_line: str = "",
    talent: str | None = None,
    card_class: str = "",
    card_types: Iterable[str] | None = None,
    card_id: str = "",
) -> str:
    """Return typebox text such as ``Lightning Action - Attack``."""
    printed = str(type_line or "").strip()
    if printed:
        return printed

    types = {str(t).lower() for t in (card_types or []) if str(t).strip()}
    prefixes = _prefix_parts(talent=talent, card_class=card_class)

    if "hero" in types:
        return _join_type_line(prefixes, "Hero")

    if "attack_action" in types:
        return _join_type_line(prefixes, "Action", "Attack")
    if "utility_action" in types:
        return _join_type_line(prefixes, "Action")
    if "attack_reaction" in types:
        return _join_type_line(prefixes, "Attack Reaction")
    if "defense_reaction" in types:
        return _join_type_line(prefixes, "Defense Reaction")
    if "instant" in types:
        return _join_type_line(prefixes, "Instant")

    if "utility_item" in types:
        from fab_tui.equipment import equipment_slot, slot_display_name

        slot = equipment_slot(card_id)
        if slot == "weapon":
            return _join_type_line(prefixes, "Weapon")
        if slot in {"head", "chest", "arms", "legs", "off_hand"}:
            return _join_type_line(prefixes, "Equipment", slot_display_name(slot))

    return ""


def classification_from_record(rec: dict[str, Any]) -> str:
    card_id = str(rec.get("id") or rec.get("card_id") or "").strip()
    inferred_types = _infer_card_types(rec)
    return format_card_classification(
        type_line=str(rec.get("type_line") or ""),
        talent=rec.get("talent"),
        card_class=str(rec.get("class") or rec.get("card_class") or ""),
        card_types=inferred_types,
        card_id=card_id,
    )


def normalize_card_id(card_id: str) -> str:
    """Collapse repeated underscores so deck ids match DB ids."""
    return re.sub(r"_+", "_", str(card_id or "").strip().lower())
