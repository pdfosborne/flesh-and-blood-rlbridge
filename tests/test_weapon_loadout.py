"""Tests for Talishar weapon hand selection in equipment loadouts."""

from __future__ import annotations

from pathlib import Path

from fab_tui.equipment import (
    select_weapon_loadout,
    weapon_hands_occupied,
)
from flesh_and_blood_rlbridge.talishar_deck_assets import (
    ensure_full_equipment_header,
    resolve_talishar_equipment_loadout,
)


def test_weapon_hands_occupied_treats_star_fall_as_two_handed() -> None:
    assert weapon_hands_occupied("star_fall", hero_id="briar") == 2
    assert weapon_hands_occupied("scorpio_comet_tail", hero_id="briar") == 2


def test_weapon_hands_occupied_treats_harmonized_kodachi_as_one_handed() -> None:
    assert weapon_hands_occupied("harmonized_kodachi", hero_id="katsu") == 1


def test_select_weapon_loadout_allows_dual_one_handed_weapons() -> None:
    active, off_hand, sideboard = select_weapon_loadout(
        ["harmonized_kodachi", "harmonized_kodachi_r"],
        hero_id="katsu",
    )
    assert active == ["harmonized_kodachi", "harmonized_kodachi_r"]
    assert off_hand == ""
    assert sideboard == []


def test_select_weapon_loadout_rejects_second_two_handed_weapon() -> None:
    active, off_hand, sideboard = select_weapon_loadout(
        ["scorpio_comet_tail", "star_fall"],
        hero_id="briar",
    )
    assert active == ["scorpio_comet_tail"]
    assert off_hand == ""
    assert sideboard == ["star_fall"]


def test_briar_header_keeps_one_weapon_active(tmp_path: Path) -> None:
    assets = tmp_path / "Assets"
    assets.mkdir()
    header = (
        "briar scorpio_comet_tail blade_beckoner_helm blossom_of_spring "
        "blade_beckoner_gauntlets blade_beckoner_boots crown_of_dichotomy "
        "quick_clicks star_fall swiftstrike_bracers"
    )
    (assets / "fab_briar_sage_aggro.txt").write_text(
        f"{header}\narcane_polarity_red\n",
        encoding="utf-8",
    )

    loadout = resolve_talishar_equipment_loadout(
        "briar",
        header,
        assets,
        deck_stem="fab_briar_sage_aggro",
    )
    line_parts = loadout.line1.split()
    weapon_count = sum(
        1
        for card_id in line_parts[1:]
        if card_id in {"scorpio_comet_tail", "star_fall"}
    )
    assert weapon_count == 1
    assert "scorpio_comet_tail" in line_parts
    assert "star_fall" not in line_parts
    assert "star_fall" in loadout.weapon_sb


def test_ensure_full_equipment_header_does_not_append_weapon_alternates(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "Assets"
    assets.mkdir()
    header = "briar scorpio_comet_tail star_fall blossom_of_spring"
    (assets / "fab_briar_sage_aggro.txt").write_text(
        f"{header}\narcane_polarity_red\n",
        encoding="utf-8",
    )

    line1 = ensure_full_equipment_header(
        "briar",
        header,
        assets,
        deck_stem="fab_briar_sage_aggro",
    )
    parts = line1.split()
    assert parts.count("scorpio_comet_tail") + parts.count("star_fall") == 1
