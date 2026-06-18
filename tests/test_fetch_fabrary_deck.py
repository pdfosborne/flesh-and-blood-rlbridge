"""Tests for FaBrary deck parsing (arena vs inventory)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts" / "deck"))

from fetch_fabrary_deck import parse_fabrary_deck  # noqa: E402


def _card(
    identifier: str,
    *,
    total: int = 0,
    sideboard_total: int = 0,
    types: list[str] | None = None,
    subtypes: list[str] | None = None,
) -> dict:
    return {
        "identifier": identifier,
        "total": total,
        "sideboardTotal": sideboard_total,
        "types": types or [],
        "subtypes": subtypes or [],
    }


def test_vynnset_arena_equipment_not_inventory() -> None:
    """Arena-equipped gear uses quantity; inventory gear stays in sideboard."""
    raw = {
        "name": "Silver Age Vynny",
        "format": "silver_age",
        "heroIdentifier": "vynnset",
        "heroClass": "Runeblade",
        "cards": [
            _card(
                "flail-of-agony",
                total=1,
                types=["Weapon"],
                subtypes=["1H", "Flail"],
            ),
            _card(
                "ebon-fold",
                total=1,
                types=["Equipment"],
                subtypes=["Head"],
            ),
            _card(
                "blossom-of-spring",
                total=1,
                types=["Equipment"],
                subtypes=["Chest"],
            ),
            _card(
                "runehold-release",
                total=1,
                types=["Equipment"],
                subtypes=["Arms"],
            ),
            _card(
                "sutcliffes-suede-hides",
                total=1,
                types=["Equipment"],
                subtypes=["Legs"],
            ),
            _card(
                "bloodied-oval",
                total=1,
                types=["Equipment"],
                subtypes=["Off-Hand"],
            ),
            _card(
                "blade-beckoner-helm",
                sideboard_total=1,
                types=["Equipment"],
                subtypes=["Head"],
            ),
            _card(
                "runebleed-robe",
                sideboard_total=1,
                types=["Equipment"],
                subtypes=["Chest"],
            ),
            _card("beseech-the-demigon-red", total=2, types=["Action"], subtypes=["Non-Attack"]),
        ],
    }

    deck_info = parse_fabrary_deck(raw)

    assert deck_info["hero_id"] == "vynnset"
    assert deck_info["equipment_header"] == (
        "vynnset flail_of_agony ebon_fold blossom_of_spring "
        "runehold_release sutcliffes_suede_hides bloodied_oval"
    )
    assert "flail_of_agony" not in deck_info["deck"]
    assert deck_info["deck"]["beseech_the_demigon_red"] == 2
    assert deck_info["sideboard"]["blade_beckoner_helm"] == 1
    assert deck_info["sideboard"]["runebleed_robe"] == 1
    assert "ebon_fold" not in deck_info["sideboard"]
