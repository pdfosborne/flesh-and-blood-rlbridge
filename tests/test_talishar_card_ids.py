"""Tests for Talishar card ID resolution."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from flesh_and_blood_rlbridge.card_db.talishar_card_ids import (  # noqa: E402
    TalisharCardIdResolver,
    _apostrophe_slug_variants,
    _collapse_numeric_underscores,
    load_talishar_card_ids,
)


def test_collapse_numeric_underscores() -> None:
    assert _collapse_numeric_underscores("10_000_year_reunion_red") == "10000_year_reunion_red"


def test_apostrophe_slug_variants() -> None:
    variants = _apostrophe_slug_variants("autumn_s_touch_red")
    assert "autumns_touch_red" in variants


def test_resolver_maps_mismatched_ids_to_talishar() -> None:
    if not load_talishar_card_ids():
        return
    resolver = TalisharCardIdResolver()
    assert resolver.resolve("autumn_s_touch_red") == "autumns_touch_red"
    assert resolver.resolve("10_000_year_reunion_red") == "10000_year_reunion_red"
    assert resolver.resolve("arcane_seeds_life_red") == "arcane_seeds__life_red"
    assert resolver.resolve("burn_up_shock_red") == "burn_up__shock_red"


def test_sanitize_deck_remaps_invalid_ids() -> None:
    if not load_talishar_card_ids():
        return
    resolver = TalisharCardIdResolver()
    cleaned, warnings = resolver.sanitize_deck(
        {
            "autumn_s_touch_red": 2,
            "arcane_seeds_life_red": 2,
            "burn_up_shock_red": 2,
            "totally_fake_card_red": 1,
        }
    )
    assert cleaned.get("autumns_touch_red") == 2
    assert cleaned.get("arcane_seeds__life_red") == 2
    assert cleaned.get("burn_up__shock_red") == 2
    assert "totally_fake_card_red" not in cleaned
    assert any("Mapped autumn_s_touch_red" in row for row in warnings)
    assert any("Mapped arcane_seeds_life_red" in row for row in warnings)
    assert any("Mapped burn_up_shock_red" in row for row in warnings)
    assert any("Dropped unknown" in row for row in warnings)
