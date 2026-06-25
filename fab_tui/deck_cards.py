"""Helpers for resolving deck card ids and pitch variants."""

from __future__ import annotations

from typing import Any


def assign_pitch_variants(candidates: list[dict[str, Any]], count: int) -> list[str]:
    """Map a name-only deck count onto Talishar pitch ids (max two per pitch).

    Used when materializing deck lists that specify card *name* and *count* only
    (e.g. ``fabrary_decks.json``). Pitch order follows ``cards.json`` (red,
    yellow, blue).
    """
    qty = max(0, int(count))
    if qty <= 0 or not candidates:
        return []

    ordered = sorted(candidates, key=lambda rec: int(rec.get("pitch") or 99))
    ids = [str(rec["id"]) for rec in ordered if rec.get("id")]
    if not ids:
        return []

    out: list[str] = []
    remaining = qty
    for card_id in ids:
        if remaining <= 0:
            break
        take = min(2, remaining)
        out.extend([card_id] * take)
        remaining -= take
    return out
