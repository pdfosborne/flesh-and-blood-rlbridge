"""Tests for saved opponent list persistence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from fab_tui.saved_opponent_decks import (  # noqa: E402
    SAVED_OPPONENTS_DIR,
    is_saved_opponent_deck,
    list_saved_opponent_decks,
    save_opponent_deck,
)
from fab_tui.sideboard_picker import load_deck_and_pool  # noqa: E402


def test_save_and_list_opponent_deck(tmp_path: Path, monkeypatch) -> None:
    saved_dir = tmp_path / "opponents"
    monkeypatch.setattr("fab_tui.saved_opponent_decks.SAVED_OPPONENTS_DIR", saved_dir)
    monkeypatch.setattr("fab_tui.saved_opponent_decks.OPPONENT_INDEX_PATH", saved_dir / "index.json")

    game_deck = {f"c_{i}": 1 for i in range(40)}
    card_pool = dict(game_deck)
    card_pool["sb_red"] = 2
    equipment = "dorinthea stormy_halcyon"

    path, asset_stem = save_opponent_deck(
        game_deck=game_deck,
        card_pool=card_pool,
        equipment_header=equipment,
        hero_id="dorinthea",
        hero_class="Warrior",
        game_format="silver_age",
        label="Dorinthea vs Kayo — test opponent",
        player_hero_id="kayo_berserker_runt",
        baseline_label="Guide policy sideboard",
    )
    assert path.is_file()
    assert asset_stem
    assert is_saved_opponent_deck(path)

    entries = list_saved_opponent_decks()
    assert len(entries) == 1
    assert entries[0].label == "Dorinthea vs Kayo — test opponent"
    assert entries[0].opponent_deck == asset_stem

    deck, pool = load_deck_and_pool(path)
    assert sum(deck.values()) == 40
    assert pool["sb_red"] == 2
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["saved_meta"]["role"] == "opponent"
    assert data["saved_meta"]["player_hero_id"] == "kayo_berserker_runt"
