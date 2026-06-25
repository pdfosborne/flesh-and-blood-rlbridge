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


def test_card_image_cdn_url() -> None:
    url = gui_api.card_image_cdn_url("lightning_press_red")
    assert url == (
        "https://images.talishar.net/public/cardimages/english/lightning_press_red.webp"
    )


def test_fetch_card_image_falls_back_to_cdn(monkeypatch: pytest.MonkeyPatch) -> None:
    env = EnvironmentSettings()
    calls: list[str] = []

    def fake_fetch(url: str) -> tuple[bytes, str] | None:
        calls.append(url)
        if "images.talishar.net" in url:
            return b"webp", "image/webp"
        return None

    monkeypatch.setattr(gui_api, "_fetch_webp_url", fake_fetch)
    payload = gui_api.fetch_card_image(env, "lightning_press_red")
    assert payload == (b"webp", "image/webp")
    assert any("images.talishar.net" in url for url in calls)


def test_card_image_fetch_urls_prioritize_cdn() -> None:
    env = EnvironmentSettings()
    urls = gui_api._card_image_fetch_urls(env, "a_red")
    assert urls[0].startswith(gui_api.TALISHAR_CARD_IMAGES_CDN)


def test_fetch_webp_url_sends_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    captured: dict[str, urllib.request.Request] = {}

    class FakeResp:
        def read(self) -> bytes:
            return b"RIFFwebp"

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(req, timeout=8):  # noqa: ARG001
        captured["req"] = req
        return FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    payload = gui_api._fetch_webp_url("https://images.talishar.net/public/cardimages/english/x.webp")
    assert payload == (b"RIFFwebp", "image/webp")
    assert captured["req"].get_header("User-agent")


def test_search_cards_returns_image_urls() -> None:
    env = EnvironmentSettings()
    hits = gui_api.search_cards("lightning", game_format="silver_age", talishar_url=env.talishar_url, limit=3)
    assert hits
    assert all(hit["image_url"].startswith(gui_api.TALISHAR_CARD_IMAGES_CDN) for hit in hits)


def test_card_image_display_url() -> None:
    url = gui_api.card_image_display_url("lightning_press_red")
    assert url == (
        "https://images.talishar.net/public/cardimages/english/lightning_press_red.webp"
    )


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


def test_deck_entries_pitch_follows_card_id_suffix() -> None:
    env = EnvironmentSettings()
    entries = gui_api.deck_counts_to_entries(
        {
            "snatch_red": 2,
            "snatch_yellow": 2,
        },
        game_format="silver_age",
        talishar_url=env.talishar_url,
    )
    by_id = {row["card_id"]: row for row in entries}
    assert by_id["snatch_red"]["pitch"] == 1
    assert by_id["snatch_yellow"]["pitch"] == 2
    assert all(
        row["image_url"].endswith(f"/{row['card_id']}.webp")
        for row in entries
    )


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


def test_card_pool_from_parts_merges_deck_and_sideboard() -> None:
    pool = gui_api.card_pool_from_parts(
        deck={"a_red": 2, "b_blue": 2},
        sideboard={"b_blue": 1, "c_yellow": 3},
    )
    assert pool == {"a_red": 2, "b_blue": 3, "c_yellow": 3}


def test_guide_baseline_with_opponent_sideboards_both_decks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = EnvironmentSettings()
    monkeypatch.setattr(gui_api, "TUI_DECK_CACHE", tmp_path)
    player_path = gui_api.import_precon("KayoSAGEPrecon", env)
    player = gui_api.load_deck_payload(player_path, env)
    opponent = gui_api.opponent_from_precon("DorintheSAGEPrecon", env)
    result = gui_api.guide_baseline_with_opponent(
        env,
        {
            "deck": player["deck"],
            "sideboard": player["sideboard"],
            "opponent_hero_id": opponent["opponent_hero_id"],
            "hero_id": player["hero_id"],
            "hero_class": player["hero_class"],
            "game_format": player["game_format"],
            "equipment_header": player["equipment_header"],
            "opponent": opponent,
        },
    )
    assert result["baseline_deck"] != player["deck"]
    assert result["deck_entries"]
    assert result["opponent_guide"]["deck_entries"]
    assert result["opponent_guide"]["deck_size"] > 0


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
