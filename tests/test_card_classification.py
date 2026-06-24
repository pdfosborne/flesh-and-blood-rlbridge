"""Tests for printed-style card classification text."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from fab_tui.card_classification import format_card_classification  # noqa: E402


def test_attack_action_with_talent() -> None:
    text = format_card_classification(
        talent="Lightning",
        card_class="Generic",
        card_types=["attack_action"],
        card_id="lightning_surge_red",
    )
    assert text == "Lightning Action - Attack"


def test_defense_reaction_with_class() -> None:
    text = format_card_classification(
        card_class="Wizard",
        card_types=["defense_reaction"],
        card_id="absorb_in_aether_red",
    )
    assert text == "Wizard Defense Reaction"


def test_equipment_chest_piece() -> None:
    text = format_card_classification(
        card_class="Generic",
        card_types=["utility_item"],
        card_id="blade_beckoner_plating",
    )
    assert text == "Equipment - Chest"


def test_instant_with_talent() -> None:
    text = format_card_classification(
        talent="Lightning",
        card_class="Generic",
        card_types=["instant"],
        card_id="cloud_cover_yellow",
    )
    assert text == "Lightning Instant"


def test_infer_instant_from_record() -> None:
    from fab_tui.card_classification import classification_from_record

    text = classification_from_record(
        {
            "id": "cloud_cover_yellow",
            "name": "Cloud Cover",
            "pitch": 2,
            "power": 0,
            "defense": 0,
            "class": "Generic",
            "talent": "Lightning",
            "card_types": [],
            "text": "The next time you would be dealt damage this turn, prevent 2 of that damage.",
        }
    )
    assert text == "Lightning Instant"


def test_prefers_printed_type_line() -> None:
    text = format_card_classification(
        type_line="Ninja Action - Attack",
        card_class="Ninja",
        card_types=["attack_action"],
        card_id="silver_talons_red",
    )
    assert text == "Ninja Action - Attack"
