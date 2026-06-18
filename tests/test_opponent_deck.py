"""Tests for Talishar opponent deck resolution."""

from __future__ import annotations

from pathlib import Path

from flesh_and_blood_rlbridge.opponent_deck import (
    normalize_talishar_asset_name,
    read_talishar_asset_hero_info,
    resolve_opponent_deck_name,
)


def test_normalize_combat_dummy_alias(tmp_path: Path) -> None:
    (tmp_path / "Dummy.txt").write_text("DUMMY equip\ncard_red\n", encoding="utf-8")
    assert normalize_talishar_asset_name("CombatDummy", tmp_path) == "Dummy"
    assert normalize_talishar_asset_name("Practice Dummy", tmp_path) == "Dummy"


def test_resolve_dual_writes_opponent_deck(tmp_path: Path) -> None:
    written: list[str] = []

    def fake_write(
        deck: dict[str, int],
        equipment_header: str,
        deck_name: str,
        assets_path: str,
    ) -> Path:
        written.append(deck_name)
        path = Path(assets_path) / f"{deck_name}.txt"
        path.write_text(f"{equipment_header}\ncard_red\n", encoding="utf-8")
        return path

    class Opponent:
        active_decks = {"vynnset": {"beseech_the_demigon_red": 40}}
        card_pool = {}
        equipment_header = "briar star_fall"

    stem = resolve_opponent_deck_name(
        player_hero_id="vynnset",
        opponent_mode="dual",
        preset_opponent_deck="Dummy",
        opponent_agents=Opponent(),
        opponent_hero_id="briar",
        assets_path=str(tmp_path),
        min_deck_size=40,
        write_deck_file=fake_write,
    )
    assert stem.startswith("rl_opp_")
    assert (tmp_path / f"{stem}.txt").is_file()
    assert written == [stem]


def test_read_talishar_asset_kayo_precon() -> None:
    repo = Path(__file__).resolve().parents[1]
    assets = repo / "Talishar" / "Assets"
    path = assets / "fab_precon_sage_ch1_kayo.txt"
    if not path.is_file():
        return
    info = read_talishar_asset_hero_info(assets, "fab_precon_sage_ch1_kayo")
    assert info is not None
    assert info.hero_id == "kayo"
    assert info.hero_class == "Brute"
    assert info.equipment_header == "kayo"


def test_read_talishar_asset_ira() -> None:
    repo = Path(__file__).resolve().parents[1]
    assets = repo / "Talishar" / "Assets"
    if not (assets / "Ira.txt").is_file():
        return
    info = read_talishar_asset_hero_info(assets, "Ira")
    assert info is not None
    assert info.hero_id == "ira_crimson_haze"
    assert info.hero_class == "Ninja"
    assert "harmonized_kodachi" in info.equipment_header
