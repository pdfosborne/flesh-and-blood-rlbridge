"""Tests for Talishar Assets deck stem resolution."""

from __future__ import annotations

from pathlib import Path

from flesh_and_blood_rlbridge.talishar_deck_assets import resolve_talishar_deck_stem


def test_resolve_sage_precon_from_hero_slug() -> None:
    repo = Path(__file__).resolve().parents[1]
    assets = repo / "Talishar" / "Assets"
    if not (assets / "BriarSAGEPrecon.txt").is_file():
        return
    assert resolve_talishar_deck_stem(assets, "briar") == "BriarSAGEPrecon"
    assert resolve_talishar_deck_stem(assets, "Briar") == "BriarSAGEPrecon"
    assert resolve_talishar_deck_stem(assets, "dash") == "DashSAGEPrecon"
    assert resolve_talishar_deck_stem(assets, "Dash") == "DashSAGEPrecon"


def test_resolve_fab_precon_asset_stem() -> None:
    repo = Path(__file__).resolve().parents[1]
    assets = repo / "Talishar" / "Assets"
    kayo = assets / "fab_precon_sage_ch1_kayo.txt"
    if not kayo.is_file():
        return
    assert resolve_talishar_deck_stem(assets, "fab_precon_sage_ch1_kayo") == kayo.stem
