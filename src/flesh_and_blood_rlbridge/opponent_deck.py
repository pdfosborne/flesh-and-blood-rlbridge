"""Resolve Talishar Assets opponent decks for draft / eval matchups."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

# Aliases for Talishar ``Assets/<name>.txt`` deck stems.
_ASSET_ALIASES: dict[str, str] = {
    "combatdummy": "Dummy",
    "practice_dummy": "Dummy",
    "practicedummy": "Dummy",
    "practice dummy": "Dummy",
    "dummy": "Dummy",
}

# Hero token → class for Talishar asset decks (no JSON metadata).
_HERO_CLASSES: dict[str, str] = {
    "briar": "Elementalist",
    "dorinthea": "Warrior",
    "dorinthea_ironsong": "Warrior",
    "kayo": "Brute",
    "viserai": "Runeblade",
    "iyslander": "Elementalist",
    "dash": "Mechanologist",
    "fai": "Ninja",
    "azalea": "Ranger",
    "boltyn": "Light",
    "enigma": "Illusionist",
    "ira": "Ninja",
    "aurora": "Runeblade",
    "vynnset": "Runeblade",
    "arakni": "Assassin",
    "blaze": "Wizard",
    "blaze_firemind": "Wizard",
    "lyath": "Guardian",
    "lyath_goldmane": "Guardian",
    "gravy": "Necromancer",
    "gravy_bones": "Necromancer",
    "dummy": "Mechanologist",
}

# Curated preset opponents shown first in the TUI.
_CURATED_PRESETS: list[tuple[str, str]] = [
    ("Practice Dummy", "Dummy"),
    ("Ira (Ninja starter)", "Ira"),
]


@dataclass(frozen=True)
class TalisharAssetHeroInfo:
    asset_stem: str
    hero_id: str
    hero_class: str
    equipment_header: str


def hero_class_for_id(hero_id: str) -> str:
    token = hero_id.replace("-", "_").split("_")[0].lower()
    if hero_id in _HERO_CLASSES:
        return _HERO_CLASSES[hero_id]
    return _HERO_CLASSES.get(token, "Warrior")


def read_talishar_asset_hero_info(
    assets_path: str | Path,
    asset_stem: str,
) -> TalisharAssetHeroInfo | None:
    """Parse hero + equipment header from a Talishar ``Assets/<stem>.txt`` file."""
    assets = Path(assets_path)
    stem = normalize_talishar_asset_name(asset_stem, assets)
    path = assets / f"{stem}.txt"
    if not path.is_file():
        return None
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if not lines:
        return None
    equipment_header = lines[0]
    hero_id = equipment_header.split()[0].replace("-", "_").lower()
    if hero_id == "dummy":
        hero_id = "dummy"
    return TalisharAssetHeroInfo(
        asset_stem=stem,
        hero_id=hero_id,
        hero_class=hero_class_for_id(hero_id),
        equipment_header=equipment_header,
    )


def normalize_talishar_asset_name(name: str, assets_path: str | Path) -> str:
    """Return a deck stem that exists under ``Assets/``, or the best alias."""
    assets = Path(assets_path)
    raw = (name or "").strip()
    if not raw:
        raw = "Dummy"
    lowered = raw.lower().replace(" ", "_")
    alias = _ASSET_ALIASES.get(lowered, raw)
    for candidate in (raw, alias, _ASSET_ALIASES.get(lowered, "")):
        if candidate and (assets / f"{candidate}.txt").is_file():
            return candidate
    if (assets / f"{alias}.txt").is_file():
        return alias
    return alias


def list_preset_opponent_options(assets_path: str | Path) -> list[tuple[str, str]]:
    """Return ``(label, asset_stem)`` options for fixed Talishar opponents."""
    assets = Path(assets_path)
    seen: set[str] = set()
    options: list[tuple[str, str]] = []

    def add(label: str, stem: str) -> None:
        stem = normalize_talishar_asset_name(stem, assets)
        if stem in seen:
            return
        if not (assets / f"{stem}.txt").is_file():
            return
        seen.add(stem)
        options.append((label, stem))

    for label, stem in _CURATED_PRESETS:
        add(label, stem)

    for path in sorted(assets.glob("*SAGEPrecon.txt")):
        add(path.stem, path.stem)

    for path in sorted(assets.glob("*.txt")):
        stem = path.stem
        if stem.startswith("rl_"):
            continue
        add(stem, stem)

    return options


def greedy_game_deck_cut(
    pool: Mapping[str, int],
    min_size: int,
    *,
    max_copies: int | None = None,
) -> dict[str, int]:
    """Deterministic first-``min_size`` cards from a pool."""
    game_deck: dict[str, int] = {}
    remaining = min_size
    for card_id, count in pool.items():
        if remaining <= 0:
            break
        available = int(count)
        if max_copies is not None:
            available = min(available, max_copies)
        take = min(available, remaining)
        if take > 0:
            game_deck[str(card_id)] = take
            remaining -= take
    return game_deck


def resolve_opponent_deck_name(
    *,
    player_hero_id: str,
    opponent_mode: str,
    preset_opponent_deck: str,
    opponent_agents: Any | None,
    opponent_hero_id: str,
    assets_path: str,
    min_deck_size: int,
    write_deck_file: Callable[[dict[str, int], str, str, str], Path],
    opponent_equipment_header: str = "",
) -> str:
    """Resolve the Talishar ``Assets`` stem used as the P2 deck in eval games.

    - **dual** — write the other pipeline's active / pool deck to a temp asset.
    - **mirror** — empty string (caller uses the same local deck file).
    - **preset** — named Talishar asset (Practice Dummy, Ira, precon, …).
    """
    if opponent_mode == "mirror":
        return ""

    if opponent_mode == "dual" and opponent_agents is not None:
        active_decks = getattr(opponent_agents, "active_decks", None) or {}
        card_pool = getattr(opponent_agents, "card_pool", None) or {}
        opp_deck = active_decks.get(player_hero_id)
        if not opp_deck and card_pool:
            opp_deck = greedy_game_deck_cut(card_pool, min_deck_size)
        if opp_deck:
            equip = (
                opponent_equipment_header
                or getattr(opponent_agents, "equipment_header", "")
                or opponent_hero_id
                or player_hero_id
            )
            stem = f"rl_opp_{uuid.uuid4().hex[:10]}"
            write_deck_file(dict(opp_deck), equip, stem, assets_path)
            return stem

    normalized = normalize_talishar_asset_name(preset_opponent_deck, assets_path)
    assets = Path(assets_path)
    if not (assets / f"{normalized}.txt").is_file():
        print(
            f"  WARNING: Opponent deck Assets/{normalized}.txt not found — "
            "using Practice Dummy (Dummy.txt)"
        )
        normalized = normalize_talishar_asset_name("Dummy", assets_path)
    return normalized
