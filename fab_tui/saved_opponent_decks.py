"""Persist user-refined opponent lists for reuse in the GUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fab_tui.config import slugify
from fab_tui.decks import DECK_CACHE, hero_class_for
from fab_tui.saved_decks import _sideboard_from_pool

SAVED_OPPONENTS_DIR = DECK_CACHE / "saved" / "opponents"
OPPONENT_INDEX_PATH = SAVED_OPPONENTS_DIR / "index.json"


@dataclass(frozen=True)
class SavedOpponentEntry:
    deck_id: str
    label: str
    path: Path
    hero_id: str
    player_hero_id: str
    game_format: str
    opponent_deck: str
    saved_at: str


def _load_index() -> list[dict[str, Any]]:
    if not OPPONENT_INDEX_PATH.is_file():
        return []
    try:
        raw = json.loads(OPPONENT_INDEX_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    return []


def _write_index(rows: list[dict[str, Any]]) -> None:
    SAVED_OPPONENTS_DIR.mkdir(parents=True, exist_ok=True)
    OPPONENT_INDEX_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def list_saved_opponent_decks() -> list[SavedOpponentEntry]:
    """Return saved opponent lists, newest first."""
    entries: list[SavedOpponentEntry] = []
    for row in _load_index():
        deck_id = str(row.get("deck_id") or "").strip()
        path_raw = str(row.get("path") or "").strip()
        if not deck_id or not path_raw:
            continue
        path = Path(path_raw)
        if not path.is_file():
            path = SAVED_OPPONENTS_DIR / path.name
        if not path.is_file():
            continue
        entries.append(
            SavedOpponentEntry(
                deck_id=deck_id,
                label=str(row.get("label") or deck_id),
                path=path.resolve(),
                hero_id=str(row.get("hero_id") or ""),
                player_hero_id=str(row.get("player_hero_id") or ""),
                game_format=str(row.get("game_format") or "silver_age"),
                opponent_deck=str(row.get("opponent_deck") or deck_id),
                saved_at=str(row.get("saved_at") or ""),
            )
        )
    entries.sort(key=lambda item: item.saved_at, reverse=True)
    return entries


def save_opponent_deck(
    *,
    game_deck: dict[str, int],
    card_pool: dict[str, int],
    equipment_header: str,
    hero_id: str,
    hero_class: str,
    game_format: str,
    label: str,
    opponent_deck: str = "",
    player_hero_id: str = "",
    baseline_label: str = "",
) -> tuple[Path, str]:
    """Write an opponent list JSON and return (path, Talishar asset stem)."""
    SAVED_OPPONENTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    hero_slug = slugify(hero_id or "opponent")
    deck_id = str(opponent_deck or "").strip() or f"gui_opp_{hero_slug}_{stamp}"
    path = SAVED_OPPONENTS_DIR / f"{deck_id}.json"

    sideboard = _sideboard_from_pool(card_pool, game_deck)
    payload = {
        "name": label,
        "hero_id": hero_id,
        "hero_class": hero_class or hero_class_for(hero_id),
        "format": game_format,
        "equipment_header": equipment_header,
        "deck": {str(k): int(v) for k, v in game_deck.items() if int(v) > 0},
        "sideboard": {str(k): int(v) for k, v in sideboard.items() if int(v) > 0},
        "saved_meta": {
            "deck_id": deck_id,
            "label": label,
            "baseline_label": baseline_label,
            "player_hero_id": player_hero_id,
            "opponent_deck": deck_id,
            "role": "opponent",
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
            "player_hero_id": player_hero_id,
            "game_format": game_format,
            "opponent_deck": deck_id,
            "saved_at": payload["saved_meta"]["saved_at"],
        },
    )
    _write_index(rows)
    return path, deck_id


def is_saved_opponent_deck(deck_path: Path) -> bool:
    try:
        data = json.loads(deck_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    meta = data.get("saved_meta")
    return isinstance(meta, dict) and meta.get("role") == "opponent"
