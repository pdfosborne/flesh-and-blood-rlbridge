"""Resolve Talishar ``Assets/<stem>.txt`` deck names from hero slugs or precon labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# SAGE precon decks (hero slug → Assets file stem)
SAGE_PRECON_BY_HERO: dict[str, str] = {
    "briar": "BriarSAGEPrecon",
    "dorinthea": "DorintheSAGEPrecon",
    "kayo": "KayoSAGEPrecon",
    "viserai": "ViseraiSAGEPrecon",
    "iyslander": "IyslanderSAGEPrecon",
    "dash": "DashSAGEPrecon",
    "fai": "FaiSAGEPrecon",
    "azalea": "AzaleaSAGEPrecon",
    "boltyn": "BoltynSAGEPrecon",
    "enigma": "EnigmaSAGEPrecon",
    "arakni_web_of_deceit": "ArakniWebOfDeceitSAGEPrecon",
    "gravy": "GravyBonesSAGEPrecon",
    "gravy_bones": "GravyBonesSAGEPrecon",
    "lyath": "LyathGoldmaneSAGEPrecon",
    "lyath_goldmane": "LyathGoldmaneSAGEPrecon",
    "blaze": "BlazeSAGEPrecon",
    "blaze_firemind": "BlazeSAGEPrecon",
    "ira": "Ira",
}


def build_assets_equipment_headers(assets_dir: str | Path) -> dict[str, str]:
    """Map hero id -> fullest equipment header line found in ``Assets/*.txt``."""
    assets = Path(assets_dir)
    result: dict[str, str] = {}
    if not assets.is_dir():
        return result
    for txt_file in sorted(assets.glob("*.txt")):
        try:
            lines = txt_file.read_text(encoding="utf-8").splitlines()
            if not lines:
                continue
            first_line = lines[0].strip()
            if not first_line:
                continue
            hero_id = first_line.split()[0]
            prev = result.get(hero_id, "")
            if len(first_line.split()) > len(prev.split()):
                result[hero_id] = first_line
        except (OSError, IndexError):
            continue
    return result


def resolve_equipment_header_line(
    hero_id: str,
    assets_dir: str | Path,
    *,
    fallback: str = "",
) -> str:
    """Return the best Talishar equipment header line for *hero_id*."""
    token = hero_id.removeprefix("hero_").replace("-", "_").strip()
    headers = build_assets_equipment_headers(assets_dir)
    best = (fallback or "").strip()
    for key in (token, hero_id):
        line = headers.get(key, "").strip()
        if len(line.split()) > len(best.split()):
            best = line
    if best:
        return best
    return token or hero_id


def _hero_tokens_match(left: str, right: str) -> bool:
    a = left.removeprefix("hero_").replace("-", "_").lower().strip()
    b = right.removeprefix("hero_").replace("-", "_").lower().strip()
    if not a or not b:
        return False
    return a == b or a.startswith(f"{b}_") or b.startswith(f"{a}_")


def equipment_header_from_deck_stem(
    deck_stem: str,
    assets_dir: str | Path,
    *,
    fallback: str = "",
) -> str:
    """Return the fullest equipment header line for a Talishar Assets deck stem.

    Reads the first line of ``Assets/<stem>.txt`` and merges it with the hero
    index from :func:`resolve_equipment_header_line` so eval/render paths recover
    full equipment even when checkpoint metadata only stores a hero slug.
    """
    assets = Path(assets_dir)
    resolved = resolve_talishar_deck_stem(assets, deck_stem)
    from_asset = ""
    asset_file = assets / f"{resolved}.txt"
    if asset_file.is_file():
        lines = [
            line.strip()
            for line in asset_file.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        if lines:
            from_asset = lines[0]
    hero_token = from_asset.split()[0] if from_asset else deck_stem
    from_index = resolve_equipment_header_line(
        hero_token,
        assets,
        fallback=from_asset or fallback,
    )
    best = (fallback or "").strip()
    for candidate in (from_asset, from_index):
        candidate = (candidate or "").strip()
        if len(candidate.split()) > len(best.split()):
            best = candidate
    return best


def ensure_full_equipment_header(
    hero_id: str,
    equipment_header: str,
    assets_dir: str | Path,
    *,
    deck_stem: str = "",
) -> str:
    """Return a Talishar line-1 header with hero plus filled equipment slots.

    Merges the supplied header with the richest matching ``Assets/*.txt`` line
    and promotes sideboard equipment into empty head/chest/arms/legs slots (and
    weapon when available).
    """
    from fab_tui.equipment import (  # noqa: PLC0415
        _equipment_pieces_from_header,
        active_list_from_slot_map,
        active_slot_map,
        equipment_slot,
        rebuild_equipment_header,
        split_equipment_header,
    )

    hero = hero_id.removeprefix("hero_").replace("-", "_").strip()
    assets = Path(assets_dir)
    candidates: list[str] = []

    def _add_candidate(raw: str) -> None:
        text = (raw or "").strip()
        if text and text not in candidates:
            candidates.append(text)

    _add_candidate(equipment_header)
    if deck_stem:
        _add_candidate(
            equipment_header_from_deck_stem(deck_stem, assets, fallback="")
        )
    _add_candidate(resolve_equipment_header_line(hero, assets, fallback=hero))

    if assets.is_dir():
        for txt_file in sorted(assets.glob("*.txt")):
            try:
                lines = [
                    line.strip()
                    for line in txt_file.read_text(encoding="utf-8", errors="replace").splitlines()
                    if line.strip()
                ]
            except OSError:
                continue
            if not lines:
                continue
            first_line = lines[0]
            first_token = first_line.split()[0]
            if _hero_tokens_match(first_token, hero):
                _add_candidate(first_line)

    if candidates:
        richest = max(candidates, key=lambda line: len(line.split()))
        hero = richest.split()[0]

    slot_map: dict[str, str] = {"hero": hero}
    sideboard_pool: list[str] = []

    for header in candidates:
        active, sideboard = split_equipment_header(header, hero_id=hero)
        for slot, card_id in active_slot_map(active, hero_id=hero).items():
            if card_id and slot != "hero" and slot not in slot_map:
                slot_map[slot] = card_id
        for card_id in sideboard:
            if card_id not in sideboard_pool:
                sideboard_pool.append(card_id)

    required_slots = ("head", "chest", "arms", "legs", "weapon")
    for slot in required_slots:
        if slot_map.get(slot):
            continue
        for card_id in sideboard_pool:
            if equipment_slot(card_id, hero_id=hero) == slot:
                slot_map[slot] = card_id
                break
        if slot_map.get(slot):
            continue
        for header in candidates:
            for card_id in _equipment_pieces_from_header(header, hero_id=hero):
                if equipment_slot(card_id, hero_id=hero) == slot:
                    slot_map[slot] = card_id
                    break
            if slot_map.get(slot):
                break

    active = active_list_from_slot_map(hero, slot_map)
    used = set(active)
    extras: list[str] = []
    for header in candidates:
        _, sideboard = split_equipment_header(header, hero_id=hero)
        for card_id in sideboard:
            if card_id not in used and card_id not in extras:
                extras.append(card_id)

    if active:
        return rebuild_equipment_header(hero, [*active[1:], *extras])
    return hero


def load_guide_sideboard_record(matchup_dir: Path) -> dict[str, Any]:
    """Load ``guide_sideboard.json`` when present under a matchup output directory."""
    path = Path(matchup_dir) / "guide_sideboard.json"
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def resolve_matchup_equipment_header(
    *,
    role: str,
    hero_id: str,
    deck_stem: str,
    assets_dir: str | Path,
    fallback: str = "",
    guide_sideboard: Optional[dict[str, Any]] = None,
) -> str:
    """Resolve a complete Talishar line-1 equipment header for eval/training.

    Prefers headers persisted by the guide sideboard step, then the on-disk
    ``Assets/<deck_stem>.txt`` line, then the hero equipment index.
    """
    hero = hero_id.removeprefix("hero_").replace("-", "_").strip()
    role_key = f"{role.strip().lower()}_equipment_header"
    from_sideboard = ""
    if guide_sideboard:
        from_sideboard = str(guide_sideboard.get(role_key, "") or "").strip()

    header = (from_sideboard or fallback or hero).strip()
    stem = (deck_stem or "").strip()
    if stem:
        header = equipment_header_from_deck_stem(
            stem,
            assets_dir,
            fallback=header,
        )
    header = resolve_equipment_header_line(hero, assets_dir, fallback=header)
    return ensure_full_equipment_header(
        hero,
        header,
        assets_dir,
        deck_stem=stem,
    )


def _asset_exists(assets_dir: Path, stem: str) -> bool:
    return bool(stem) and (assets_dir / f"{stem}.txt").is_file()


def resolve_talishar_deck_stem(assets_dir: str | Path, name: str) -> str:
    """Return an ``Assets`` file stem that exists for *name*.

    *name* may be a full asset stem (``BriarSAGEPrecon``,
    ``fab_precon_sage_ch1_kayo``), a hero slug (``briar``, ``dash``), or a
    title-cased hero token (``Briar``, ``Dash``).
    """
    assets = Path(assets_dir)
    raw = (name or "").strip()
    if not raw:
        return raw
    if _asset_exists(assets, raw):
        return raw

    token = raw.replace("-", "_")
    hero = token.split("_")[0].lower()
    title = hero[:1].upper() + hero[1:] if hero else raw

    candidates: list[str] = []

    def add(stem: str) -> None:
        if stem and stem not in candidates:
            candidates.append(stem)

    add(token)
    add(title)
    if token in SAGE_PRECON_BY_HERO:
        add(SAGE_PRECON_BY_HERO[token])
    if hero in SAGE_PRECON_BY_HERO:
        add(SAGE_PRECON_BY_HERO[hero])
    add(f"{title}SAGEPrecon")

    if assets.is_dir():
        for pattern in (
            f"fab_precon_*_{hero}.txt",
            f"fab_precon_*_{token}.txt",
            f"*{title}*SAGEPrecon.txt",
            f"*{hero}*SAGEPrecon.txt",
        ):
            for path in sorted(assets.glob(pattern)):
                add(path.stem)

    for stem in candidates:
        if _asset_exists(assets, stem):
            return stem

    return raw
