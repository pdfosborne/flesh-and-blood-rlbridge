"""Tests for equipment loadout helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "training"))

from fab_tui.equipment import (  # noqa: E402
    active_equipment_header,
    equipment_slot,
    parse_equipment_header,
    parse_standard_loadout,
    rebuild_equipment_header,
    split_equipment_header,
    suggest_guide_equipment_header,
)
from fab_tui.sideboard_picker import write_candidates_manifest  # noqa: E402
from train_pipeline_common import load_sideboard_candidates_from_json  # noqa: E402


def test_equipment_slot_detects_weapon_and_armor() -> None:
    assert equipment_slot("romping_club", hero_id="briar") == "weapon"
    assert equipment_slot("blossom_of_spring", hero_id="briar") == "chest"
    assert equipment_slot("snapdragon_scalers", hero_id="briar") == "legs"
    assert equipment_slot("blade_beckoner_plating", hero_id="briar") == "chest"
    assert equipment_slot("captains_coat", hero_id="briar") == "chest"


def test_dorinthea_precon_equipment_slots() -> None:
    assert equipment_slot("gallantry_gold", hero_id="dorinthea") == "arms"
    assert equipment_slot("valiant_dynamo", hero_id="dorinthea") == "legs"
    assert equipment_slot("refraction_bolters", hero_id="dorinthea") == "legs"
    assert equipment_slot("squires_bracers", hero_id="dorinthea") == "arms"
    assert equipment_slot("dawnblade", hero_id="dorinthea") == "weapon"


def test_split_equipment_header_keeps_one_piece_per_slot() -> None:
    header = (
        "dorinthea blossom_of_spring dawnblade gauntlets_of_unity helm_of_unity "
        "nullrune_gloves nullrune_hood nullrune_robe refraction_bolters"
    )
    active, sideboard = split_equipment_header(header, hero_id="dorinthea")
    assert active == [
        "dorinthea",
        "blossom_of_spring",
        "dawnblade",
        "gauntlets_of_unity",
        "helm_of_unity",
        "refraction_bolters",
    ]
    assert sideboard == [
        "nullrune_gloves",
        "nullrune_hood",
        "nullrune_robe",
    ]
    assert active_equipment_header(header, hero_id="dorinthea") == " ".join(active)


def test_parse_standard_loadout_shows_empty_chest_and_head() -> None:
    header = (
        "dorinthea blossom_of_spring dawnblade gauntlets_of_unity helm_of_unity "
        "nullrune_gloves nullrune_hood nullrune_robe refraction_bolters"
    )
    entries = parse_standard_loadout(
        header,
        hero_id="dorinthea",
        display_name=lambda cid: cid,
    )
    by_slot = {entry.slot: entry.card_id for entry in entries}
    assert by_slot["hero"] == "dorinthea"
    assert by_slot["weapon"] == "dawnblade"
    assert by_slot["head"] == "helm_of_unity"
    assert by_slot["chest"] == "blossom_of_spring"
    assert by_slot["arms"] == "gauntlets_of_unity"
    assert by_slot["legs"] == "refraction_bolters"


def test_parse_and_rebuild_equipment_header() -> None:
    header = "briar romping_club blossom_of_spring snapdragon_scalers"
    entries = parse_equipment_header(
        header,
        hero_id="briar",
        display_name=lambda cid: cid,
    )
    assert entries[0].slot == "hero"
    assert entries[1].slot == "weapon"
    rebuilt = rebuild_equipment_header(
        "briar",
        [entry.card_id for entry in entries],
    )
    assert rebuilt == header


def test_briar_weapon_alternatives_are_slot_filtered_only() -> None:
    from fab_tui.equipment import EquipmentSearchIndex, equipment_slot

    idx = EquipmentSearchIndex("silver_age", hero_id="briar")
    weapons = [h.card_id for h in idx.all_hits() if equipment_slot(h.card_id, hero_id="briar") == "weapon"]
    assert "talishar_the_lost_prince" in weapons
    assert "nebula_blade" in weapons
    assert "rosetta_thorn" in weapons
    assert not any(equipment_slot(cid, hero_id="briar") != "weapon" for cid in weapons)


def test_hero_class_for_id_reads_cards_db() -> None:
    from fab_tui.equipment import hero_class_for_id, hero_talent_for_id

    assert hero_class_for_id("briar") == "Runeblade"
    assert hero_talent_for_id("briar") == "Elemental"


def test_suggest_guide_equipment_header_rejects_second_two_handed_weapon() -> None:
    header = (
        "briar scorpio_comet_tail blade_beckoner_helm blossom_of_spring "
        "blade_beckoner_gauntlets blade_beckoner_boots star_fall"
    )
    suggested = suggest_guide_equipment_header(
        header,
        hero_id="briar",
        opponent_hero_id="dorinthea",
        game_format="silver_age",
        pool_by_id={},
    )
    active, sideboard = split_equipment_header(suggested, hero_id="briar")
    weapon_cards = [
        cid
        for cid in active[1:]
        if equipment_slot(cid, hero_id="briar") == "weapon"
    ]
    assert len(weapon_cards) == 1
    assert weapon_cards[0] in {"scorpio_comet_tail", "star_fall"}
    inactive = {"scorpio_comet_tail", "star_fall"} - {weapon_cards[0]}
    assert inactive.issubset(set(sideboard))


def test_suggest_guide_equipment_header_allows_dual_one_handed_weapons() -> None:
    header = (
        "katsu harmonized_kodachi harmonized_kodachi_r "
        "mask_of_momentum snapdragon_scalers"
    )
    suggested = suggest_guide_equipment_header(
        header,
        hero_id="katsu",
        opponent_hero_id="dorinthea",
        game_format="silver_age",
        pool_by_id={},
    )
    active, _ = split_equipment_header(suggested, hero_id="katsu")
    weapons = [
        cid
        for cid in active[1:]
        if equipment_slot(cid, hero_id="katsu") == "weapon"
    ]
    assert weapons == ["harmonized_kodachi", "harmonized_kodachi_r"]


def test_suggest_guide_equipment_header_keeps_briar_weapon() -> None:
    header = "briar rosetta_thorn blossom_of_spring mage_master_boots aether_crackers"
    suggested = suggest_guide_equipment_header(
        header,
        hero_id="briar",
        opponent_hero_id="azalea",
        game_format="silver_age",
        pool_by_id={},
    )
    assert "rosetta_thorn" in suggested.split()
    assert "driftwood_quiver" not in suggested.split()


def test_suggest_guide_equipment_header_keeps_shape() -> None:
    header = "briar romping_club blossom_of_spring snapdragon_scalers"
    suggested = suggest_guide_equipment_header(
        header,
        hero_id="briar",
        opponent_hero_id="iyslander",
        game_format="silver_age",
    )
    active, sideboard = split_equipment_header(suggested, hero_id="briar")
    assert active[0] == "briar"
    assert len(active) == 4
    assert equipment_slot(active[1], hero_id="briar") == "weapon"
    assert len({equipment_slot(cid, hero_id="briar") for cid in active[1:]}) == 3


def test_manifest_carries_equipment_header(tmp_path: Path) -> None:
    baseline = {f"c_{i}": 1 for i in range(40)}
    equipment = "briar romping_club blossom_of_spring snapdragon_scalers"
    manifest = write_candidates_manifest(
        tmp_path / "manifest.json",
        baseline_deck=baseline,
        card_pool=baseline,
        variants=[],
        baseline_label="Guide policy deck",
        equipment_header=equipment,
    )
    loaded, _ = load_sideboard_candidates_from_json(
        manifest,
        card_pool=baseline,
        min_deck_size=40,
    )
    assert len(loaded) == 1
    assert loaded[0].equipment_header == equipment
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["equipment_header"] == equipment
