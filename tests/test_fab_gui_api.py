"""Tests for the web GUI API layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fab_gui import api as gui_api
from fab_tui.config import EnvironmentSettings


def test_card_image_proxy_path() -> None:
    path = gui_api.card_image_proxy_path("lightning_press_red")
    assert path == "/api/card-image/lightning_press_red"


def test_search_cards_returns_image_urls() -> None:
    env = EnvironmentSettings()
    hits = gui_api.search_cards("lightning", game_format="silver_age", talishar_url=env.talishar_url, limit=3)
    assert hits
    assert all(hit["image_url"].startswith("/api/card-image/") for hit in hits)


def test_search_cards_include_classification() -> None:
    env = EnvironmentSettings()
    hits = gui_api.search_cards(
        "lightning surge",
        game_format="silver_age",
        talishar_url=env.talishar_url,
        limit=5,
    )
    surge = next((hit for hit in hits if hit["card_id"].startswith("lightning_surge")), None)
    assert surge is not None
    assert surge["classification"] == "Lightning Action - Attack"


def test_deck_entries_include_instant_classification() -> None:
    env = EnvironmentSettings()
    entries = gui_api.deck_counts_to_entries(
        {"cloud_cover_yellow": 2},
        game_format="silver_age",
        talishar_url=env.talishar_url,
    )
    assert len(entries) == 1
    assert entries[0]["classification"] == "Lightning Instant"


def test_deck_entries_resolve_underscored_ids() -> None:
    env = EnvironmentSettings()
    entries = gui_api.deck_counts_to_entries(
        {"burn_up_shock_red": 1},
        game_format="silver_age",
        talishar_url=env.talishar_url,
    )
    assert len(entries) == 1
    assert entries[0]["name"] == "Burn Up // Shock"
    assert entries[0]["classification"] == "Lightning Runeblade Action"


def test_equipment_alternatives_filter_by_slot_only() -> None:
    env = EnvironmentSettings()
    hits = gui_api.equipment_alternatives_for_slot(
        game_format="silver_age",
        hero_id="briar",
        talishar_url=env.talishar_url,
        slot="weapon",
    )
    ids = {hit["card_id"] for hit in hits}
    assert "rosetta_thorn" in ids
    assert "cloak_of_darkness" not in ids


def test_compute_guide_baseline_preserves_equipment_header() -> None:
    equipment = "briar rosetta_thorn blossom_of_spring snapdragon_scalers"
    pool = {f"c_{i}": 1 for i in range(55)}
    result = gui_api.compute_guide_baseline(
        card_pool=pool,
        opponent_hero_id="kayo",
        hero_id="briar",
        hero_class="Runeblade",
        game_format="silver_age",
        equipment_header=equipment,
    )
    assert result["equipment_header"] == equipment


def test_try_swap_valid() -> None:
    deck = {"a_red": 2, "b_blue": 1}
    pool = {"a_red": 2, "b_blue": 1, "c_yellow": 1}
    result = gui_api.try_swap(deck, pool, "a_red", "c_yellow")
    assert result is not None
    assert result["deck"]["a_red"] == 1
    assert result["deck"]["c_yellow"] == 1


def test_persist_session_deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gui_api, "TUI_DECK_CACHE", tmp_path)
    path = gui_api.persist_session_deck(
        {
            "session_id": "sess1",
            "hero_id": "aurora",
            "hero_class": "Runeblade",
            "game_format": "silver_age",
            "equipment_header": "aurora star_fall",
            "deck": {"a_red": 40},
            "card_pool": {"a_red": 40, "b_blue": 15},
        }
    )
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["hero_id"] == "aurora"
    assert data["sideboard"]["b_blue"] == 15
