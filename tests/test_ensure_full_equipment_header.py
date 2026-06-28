"""Tests for Talishar equipment header completion."""

from __future__ import annotations

from pathlib import Path

from flesh_and_blood_rlbridge.talishar_deck_assets import ensure_full_equipment_header


def test_ensure_full_equipment_header_fills_from_richer_asset(tmp_path: Path) -> None:
    assets = tmp_path / "Assets"
    assets.mkdir()
    (assets / "BravoSAGEPrecon.txt").write_text(
        "bravo_showstopper anothos ironhide_helm ironhide_plate "
        "ironhide_gauntlet ironhide_boots\n"
        "command_and_conquer_red\n",
        encoding="utf-8",
    )
    (assets / "fab_precon_sage_ch1_bravo.txt").write_text(
        "bravo\n"
        "command_and_conquer_red\n",
        encoding="utf-8",
    )

    header = ensure_full_equipment_header(
        "bravo",
        "bravo",
        assets,
        deck_stem="fab_precon_sage_ch1_bravo",
    )

    parts = header.split()
    assert parts[0].startswith("bravo")
    assert "ironhide_helm" in parts
    assert "ironhide_plate" in parts
    assert "ironhide_gauntlet" in parts
    assert "ironhide_boots" in parts
    assert "anothos" in parts


def test_write_deck_file_uses_full_equipment_header(tmp_path: Path) -> None:
    import sys

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "scripts" / "training"))
    from train_pipeline_common import _write_deck_file  # noqa: PLC0415

    assets = tmp_path / "Assets"
    assets.mkdir()
    (assets / "BravoSAGEPrecon.txt").write_text(
        "bravo_showstopper anothos ironhide_helm ironhide_plate "
        "ironhide_gauntlet ironhide_boots\n"
        "command_and_conquer_red\n",
        encoding="utf-8",
    )
    (assets / "AzaleaSAGEPrecon.txt").write_text(
        "azalea death_dealer bloom_of_spring blade_beckoner_helm "
        "blade_beckoner_plating blade_beckoner_gauntlets quick_glide\n"
        "bolt_n_shot_red\n",
        encoding="utf-8",
    )

    path = _write_deck_file(
        {"bolt_n_shot_red": 40},
        "azalea",
        "eval_azalea_test",
        str(assets),
    )
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert "blade_beckoner_helm" in first_line
    assert "death_dealer" in first_line

    bravo_path = _write_deck_file(
        {"command_and_conquer_red": 40},
        "bravo",
        "fab_precon_sage_ch1_bravo",
        str(assets),
    )
    bravo_line = bravo_path.read_text(encoding="utf-8").splitlines()[0]
    assert "ironhide_helm" in bravo_line
    assert "anothos" in bravo_line


def test_resolve_matchup_equipment_header_prefers_guide_sideboard(tmp_path: Path) -> None:
    from flesh_and_blood_rlbridge.talishar_deck_assets import (
        load_guide_sideboard_record,
        resolve_matchup_equipment_header,
    )

    assets = tmp_path / "Assets"
    assets.mkdir()
    (assets / "BravoSAGEPrecon.txt").write_text(
        "bravo_showstopper anothos ironhide_helm ironhide_plate "
        "ironhide_gauntlet ironhide_boots\n"
        "command_and_conquer_red\n",
        encoding="utf-8",
    )
    (assets / "fab_bravo.txt").write_text(
        "bravo\n"
        "command_and_conquer_red\n",
        encoding="utf-8",
    )
    matchup_dir = tmp_path / "match_a"
    matchup_dir.mkdir()
    (matchup_dir / "guide_sideboard.json").write_text(
        '{"p1_equipment_header": "bravo"}',
        encoding="utf-8",
    )
    guide = load_guide_sideboard_record(matchup_dir)
    header = resolve_matchup_equipment_header(
        role="p1",
        hero_id="bravo",
        deck_stem="fab_bravo",
        assets_dir=assets,
        fallback="bravo",
        guide_sideboard=guide,
    )
    assert "ironhide_helm" in header.split()
    assert "anothos" in header.split()
