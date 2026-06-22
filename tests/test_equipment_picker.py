"""Tests for equipment loadout helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "training"))

from fab_tui.equipment import (  # noqa: E402
    equipment_slot,
    parse_equipment_header,
    rebuild_equipment_header,
    suggest_guide_equipment_header,
)
from fab_tui.sideboard_picker import write_candidates_manifest  # noqa: E402
from train_pipeline_common import load_sideboard_candidates_from_json  # noqa: E402


def test_equipment_slot_detects_weapon_and_armor() -> None:
    assert equipment_slot("romping_club", hero_id="briar") == "weapon"
    assert equipment_slot("blossom_of_spring", hero_id="briar") == "chest"
    assert equipment_slot("snapdragon_scalers", hero_id="briar") == "legs"


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


def test_suggest_guide_equipment_header_keeps_shape() -> None:
    header = "briar romping_club blossom_of_spring snapdragon_scalers"
    suggested = suggest_guide_equipment_header(
        header,
        hero_id="briar",
        opponent_hero_id="iyslander",
        game_format="silver_age",
    )
    assert suggested.split()[0] == "briar"
    assert len(suggested.split()) == len(header.split())


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
