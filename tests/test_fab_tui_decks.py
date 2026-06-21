"""Tests for TUI deck hero auto-fill."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from fab_tui.config import ExperimentSpec, normalize_pipeline_format
from fab_tui.decks import apply_heroes_from_decks, export_precon_deck_json, read_deck_hero_info


def test_read_deck_hero_info_from_json(tmp_path: Path) -> None:
    deck = tmp_path / "deck.json"
    deck.write_text(
        json.dumps(
            {
                "name": "Silver Age Vynny",
                "hero_id": "vynnset",
                "hero_class": "Runeblade",
                "format": "silver_age",
                "equipment_header": "vynnset flail_of_agony ebon_fold",
                "deck": {"beseech_the_demigon_red": 2},
            }
        ),
        encoding="utf-8",
    )
    info = read_deck_hero_info(deck)
    assert info is not None
    assert info.hero_id == "vynnset"
    assert info.hero_class == "Runeblade"
    assert info.equipment_header.startswith("vynnset flail")


def test_apply_mirror_opponent_from_p1() -> None:
    from fab_tui.decks import apply_mirror_opponent_from_p1

    spec = ExperimentSpec(
        name="mirror_test",
        hero_id="vynnset",
        hero_class="Runeblade",
        equipment_header="vynnset flail_of_agony",
    )
    apply_mirror_opponent_from_p1(spec)
    assert spec.opponent_hero_id == "vynnset"
    assert spec.p2_hero_id == "vynnset"
    assert spec.p2_hero_class == "Runeblade"
    assert spec.p2_equipment_header == "vynnset flail_of_agony"


def test_apply_heroes_from_p1_warm_start(tmp_path: Path) -> None:
    deck = tmp_path / "p1.json"
    deck.write_text(
        json.dumps(
            {
                "hero_id": "vynnset",
                "hero_class": "Runeblade",
                "format": "silver_age",
                "equipment_header": "vynnset flail_of_agony",
                "deck": {},
            }
        ),
        encoding="utf-8",
    )
    spec = ExperimentSpec(name="test")
    spec.p1_starting_deck = str(deck)
    apply_heroes_from_decks(spec)
    assert spec.hero_id == "vynnset"
    assert spec.hero_class == "Runeblade"
    assert spec.game_format == "silver_age"
    assert spec.equipment_header == "vynnset flail_of_agony"


def test_normalize_pipeline_format_maps_sage() -> None:
    assert normalize_pipeline_format("sage") == "silver_age"
    assert normalize_pipeline_format("SAGE") == "silver_age"
    assert normalize_pipeline_format("blitz") == "blitz"


def test_export_precon_splits_sideboard_inventory(tmp_path: Path) -> None:
    assets = tmp_path / "Assets"
    assets.mkdir()
    cards = " ".join(f"card_{i}" for i in range(45))
    (assets / "TestSAGEPrecon.txt").write_text(
        f"test_hero equip\n{cards}\n",
        encoding="utf-8",
    )
    out = tmp_path / "test.json"
    export_precon_deck_json("TestSAGEPrecon", assets, out, game_format="sage")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert sum(data["deck"].values()) == 40
    assert sum(data["sideboard"].values()) == 5

def test_pipeline_argv_maps_sage_for_train_full_pipeline() -> None:
    spec = ExperimentSpec(name="sage-precon", game_format="sage")
    argv = spec.pipeline_argv()
    fmt_idx = argv.index("--format")
    assert argv[fmt_idx + 1] == "silver_age"

