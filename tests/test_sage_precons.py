"""Validate official Silver Age precon Talishar asset files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_ASSETS = _REPO / "Talishar" / "Assets"
_PHP = _REPO / "Talishar" / "GeneratedCode" / "GeneratedCardDictionaries.php"
_CARDS = _REPO / "src" / "flesh_and_blood_rlbridge" / "card_db" / "cards.json"

SAGE_PRECON_STEMS = [
    "KayoSAGEPrecon",
    "ViseraiSAGEPrecon",
    "IyslanderSAGEPrecon",
    "DashSAGEPrecon",
    "FaiSAGEPrecon",
    "DorintheSAGEPrecon",
    "AzaleaSAGEPrecon",
    "EnigmaSAGEPrecon",
    "BoltynSAGEPrecon",
    "BriarSAGEPrecon",
    "GravyBonesSAGEPrecon",
    "LyathGoldmaneSAGEPrecon",
    "BlazeSAGEPrecon",
]

YOUNG_HEROES = {
    "kayo",
    "viserai",
    "iyslander",
    "dash",
    "fai",
    "dorinthea",
    "azalea",
    "enigma",
    "boltyn",
    "briar",
    "gravy_bones",
    "lyath_goldmane",
    "blaze_firemind",
}

ADULT_HERO_IDS = {
    "azalea_ace_in_the_hole",
    "dash_inventor_extraordinaire",
    "fai_rising_rebellion",
    "viserai_rune_blood",
    "kayo_berserker_runt",
    "iyslander_stormbind",
    "ser_boltyn_breaker_of_dawn",
    "dorinthea_ironsong",
}

_LEGAL_RARITIES = frozenset({"common", "rare", "basic", "token"})


def _load_cards_db() -> dict[str, dict]:
    raw = json.loads(_CARDS.read_text(encoding="utf-8"))
    return {str(rec["id"]): rec for rec in raw if isinstance(rec, dict) and rec.get("id")}


def _parse_asset(path: Path) -> tuple[list[str], list[str]]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if len(lines) < 2:
        raise AssertionError(f"{path.name}: expected header + deck lines")
    return lines[0].split(), " ".join(lines[1:]).split()


def _deck_only_ids(header: list[str], deck: list[str]) -> list[str]:
    hero = header[0] if header else ""
    header_rest = set(header[1:])
    out: list[str] = []
    for card_id in deck:
        if card_id == hero or card_id in header_rest:
            continue
        out.append(card_id)
    return out


@pytest.fixture(scope="module")
def talishar_ids() -> frozenset[str]:
    if not _PHP.is_file():
        pytest.skip("Talishar PHP not available")
    from flesh_and_blood_rlbridge.card_db.talishar_card_ids import (  # noqa: PLC0415
        load_talishar_card_ids,
    )

    return load_talishar_card_ids(str(_PHP))


@pytest.fixture(scope="module")
def talishar_subtypes() -> dict[str, str]:
    if not _PHP.is_file():
        pytest.skip("Talishar PHP not available")
    from flesh_and_blood_rlbridge.card_db.talishar_card_ids import (  # noqa: PLC0415
        load_talishar_card_subtypes,
    )

    return load_talishar_card_subtypes(str(_PHP))


@pytest.fixture(scope="module")
def cards_db() -> dict[str, dict]:
    if not _CARDS.is_file():
        pytest.skip("cards.json not available")
    return _load_cards_db()


@pytest.mark.parametrize("stem", SAGE_PRECON_STEMS)
def test_sage_precon_uses_young_hero(stem: str) -> None:
    path = _ASSETS / f"{stem}.txt"
    if not path.is_file():
        pytest.skip(f"missing {stem}")
    header, _deck = _parse_asset(path)
    hero = header[0]
    assert hero in YOUNG_HEROES, f"{stem}: hero {hero!r} is not a young hero id"
    assert hero not in ADULT_HERO_IDS


@pytest.mark.parametrize("stem", SAGE_PRECON_STEMS)
def test_sage_precon_copy_limits(stem: str) -> None:
    path = _ASSETS / f"{stem}.txt"
    if not path.is_file():
        pytest.skip(f"missing {stem}")
    _header, deck = _parse_asset(path)
    counts = Counter(deck)
    over = {cid: qty for cid, qty in counts.items() if qty > 2}
    assert not over, f"{stem}: more than 2 copies: {over}"


@pytest.mark.parametrize("stem", SAGE_PRECON_STEMS)
def test_sage_precon_cards_exist_in_talishar(stem: str, talishar_ids: frozenset[str]) -> None:
    path = _ASSETS / f"{stem}.txt"
    if not path.is_file():
        pytest.skip(f"missing {stem}")
    header, deck = _parse_asset(path)
    missing = [cid for cid in [*header, *deck] if cid not in talishar_ids]
    assert not missing, f"{stem}: unknown Talishar ids: {missing[:8]}"


@pytest.mark.parametrize("stem", SAGE_PRECON_STEMS)
def test_sage_precon_deck_rarity_legal(stem: str, cards_db: dict[str, dict]) -> None:
    path = _ASSETS / f"{stem}.txt"
    if not path.is_file():
        pytest.skip(f"missing {stem}")
    header, deck = _parse_asset(path)
    illegal: list[tuple[str, str]] = []
    for card_id in _deck_only_ids(header, deck):
        rec = cards_db.get(card_id)
        if rec is None:
            continue
        rarity = str(rec.get("rarity") or "").strip().lower()
        if rarity and rarity not in _LEGAL_RARITIES:
            illegal.append((card_id, rarity))
    assert not illegal, f"{stem}: illegal silver_age deck cards: {illegal[:8]}"


@pytest.mark.parametrize("stem", SAGE_PRECON_STEMS)
def test_sage_precon_equipment_slots_resolve(stem: str, talishar_subtypes: dict[str, str]) -> None:
    path = _ASSETS / f"{stem}.txt"
    if not path.is_file():
        pytest.skip(f"missing {stem}")
    header, _deck = _parse_asset(path)
    equipment = header[1:]
    unknown = [cid for cid in equipment if cid not in talishar_subtypes]
    assert not unknown, f"{stem}: equipment missing subtype map: {unknown}"


def test_no_aurora_sage_precon_asset() -> None:
    assert not (_ASSETS / "AuroraSAGEPrecon.txt").is_file()


def test_dorinthea_variants_match() -> None:
    a = _ASSETS / "DorintheSAGEPrecon.txt"
    b = _ASSETS / "DorintheaSAGEPrecon.txt"
    if not a.is_file() or not b.is_file():
        pytest.skip("dorinthea precons missing")
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")
