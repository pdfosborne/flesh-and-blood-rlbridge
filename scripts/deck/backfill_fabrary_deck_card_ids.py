#!/usr/bin/env python3
"""Backfill fabrary_decks.json card_ids from playable Talishar asset files."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAB_PATH = REPO / "src" / "flesh_and_blood_rlbridge" / "card_db" / "fabrary_decks.json"
ASSETS = REPO / "Talishar" / "Assets"
SECONDARY = REPO / "assets" / "talishar_decks"

sys.path.insert(0, str(REPO / "src"))
from flesh_and_blood_rlbridge.talishar_deck_assets import (  # noqa: E402
    deck_asset_is_playable,
    read_talishar_deck_asset,
    resolve_canonical_sage_precon_stem,
)


def _best_asset_path(deck_id: str) -> Path | None:
    canonical = resolve_canonical_sage_precon_stem(deck_id)
    for path in (
        ASSETS / f"{canonical}.txt" if canonical else None,
        ASSETS / f"{deck_id}.txt",
        SECONDARY / f"{deck_id}.txt",
    ):
        if path is None or not path.is_file():
            continue
        header, cards = read_talishar_deck_asset(path)
        if deck_asset_is_playable(header, cards):
            return path
    return None


def main() -> int:
    db = json.loads(FAB_PATH.read_text(encoding="utf-8"))
    updated = 0
    for entry in db.get("decks", []):
        if entry.get("format") != "silver_age":
            continue
        deck_id = str(entry.get("id", "")).strip()
        if not deck_id:
            continue
        path = _best_asset_path(deck_id)
        if path is None:
            continue
        _, cards = read_talishar_deck_asset(path)
        counts = Counter(cards)
        entry["card_ids"] = [
            {"id": card_id, "count": count}
            for card_id, count in sorted(counts.items())
        ]
        entry["cards"] = []
        updated += 1

    FAB_PATH.write_text(json.dumps(db, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated card_ids for {updated} silver_age deck(s) in {FAB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
