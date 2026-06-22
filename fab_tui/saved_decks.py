"""Persist user-refined sideboard lists for reuse in the TUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fab_tui.config import slugify
from fab_tui.decks import DECK_CACHE, hero_class_for

SAVED_DECKS_DIR = DECK_CACHE / "saved"
INDEX_PATH = SAVED_DECKS_DIR / "index.json"


@dataclass(frozen=True)
class SavedDeckEntry:
    deck_id: str
    label: str
    path: Path
    hero_id: str
    opponent_hero_id: str
    game_format: str
    saved_at: str


def _sideboard_from_pool(
    card_pool: dict[str, int],
    game_deck: dict[str, int],
) -> dict[str, int]:
    sideboard: dict[str, int] = {}
    for card_id in set(card_pool) | set(game_deck):
        remaining = int(card_pool.get(card_id, 0)) - int(game_deck.get(card_id, 0))
        if remaining > 0:
            sideboard[card_id] = remaining
    return sideboard


def _load_index() -> list[dict[str, Any]]:
    if not INDEX_PATH.is_file():
        return []
    try:
        raw = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    if isinstance(raw, dict):
        decks = raw.get("decks")
        if isinstance(decks, list):
            return [row for row in decks if isinstance(row, dict)]
    return []


def _write_index(rows: list[dict[str, Any]]) -> None:
    SAVED_DECKS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def list_saved_user_decks() -> list[SavedDeckEntry]:
    """Return saved sideboard lists, newest first."""
    entries: list[SavedDeckEntry] = []
    for row in _load_index():
        deck_id = str(row.get("deck_id") or "").strip()
        path_raw = str(row.get("path") or "").strip()
        if not deck_id or not path_raw:
            continue
        path = Path(path_raw)
        if not path.is_file():
            path = SAVED_DECKS_DIR / path.name
        if not path.is_file():
            continue
        entries.append(
            SavedDeckEntry(
                deck_id=deck_id,
                label=str(row.get("label") or deck_id),
                path=path.resolve(),
                hero_id=str(row.get("hero_id") or ""),
                opponent_hero_id=str(row.get("opponent_hero_id") or ""),
                game_format=str(row.get("game_format") or "silver_age"),
                saved_at=str(row.get("saved_at") or ""),
            )
        )
    entries.sort(key=lambda item: item.saved_at, reverse=True)
    return entries


def save_user_deck(
    *,
    baseline_deck: dict[str, int],
    card_pool: dict[str, int],
    equipment_header: str,
    hero_id: str,
    hero_class: str,
    game_format: str,
    label: str,
    opponent_hero_id: str = "",
    baseline_label: str = "",
) -> Path:
    """Write a refined list to disk and register it in the saved-decks index."""
    SAVED_DECKS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    hero_slug = slugify(hero_id or "hero")
    deck_id = f"{hero_slug}_{stamp}"
    path = SAVED_DECKS_DIR / f"{deck_id}.json"

    sideboard = _sideboard_from_pool(card_pool, baseline_deck)
    payload = {
        "name": label,
        "hero_id": hero_id,
        "hero_class": hero_class or hero_class_for(hero_id),
        "format": game_format,
        "equipment_header": equipment_header,
        "deck": {str(k): int(v) for k, v in baseline_deck.items() if int(v) > 0},
        "sideboard": {str(k): int(v) for k, v in sideboard.items() if int(v) > 0},
        "saved_meta": {
            "deck_id": deck_id,
            "label": label,
            "baseline_label": baseline_label,
            "opponent_hero_id": opponent_hero_id,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = _load_index()
    rows = [row for row in rows if str(row.get("deck_id")) != deck_id]
    rows.insert(
        0,
        {
            "deck_id": deck_id,
            "label": label,
            "path": str(path),
            "hero_id": hero_id,
            "opponent_hero_id": opponent_hero_id,
            "game_format": game_format,
            "saved_at": payload["saved_meta"]["saved_at"],
        },
    )
    _write_index(rows)
    return path


def default_saved_deck_label(
    *,
    hero_id: str,
    opponent_hero_id: str,
    baseline_label: str,
) -> str:
    hero = hero_id.replace("_", " ").title() if hero_id else "Deck"
    if opponent_hero_id:
        opp = opponent_hero_id.replace("_", " ").title()
        base = f"{hero} vs {opp}"
    else:
        base = hero
    if baseline_label and baseline_label not in base:
        return f"{base} — {baseline_label}"
    return base


def is_saved_user_deck(deck_path: Path) -> bool:
    """True when *deck_path* is a list saved by the sideboard wizard."""
    try:
        data = json.loads(deck_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data.get("saved_meta"), dict)
