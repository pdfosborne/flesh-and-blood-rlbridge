"""Tests for saved sideboard list persistence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from fab_tui.saved_decks import (  # noqa: E402
    SAVED_DECKS_DIR,
    is_saved_user_deck,
    list_saved_user_decks,
    save_user_deck,
)
from fab_tui.sideboard_picker import load_deck_and_pool  # noqa: E402


def test_save_and_list_user_deck(tmp_path: Path, monkeypatch) -> None:
    saved_dir = tmp_path / "saved"
    monkeypatch.setattr("fab_tui.saved_decks.SAVED_DECKS_DIR", saved_dir)
    monkeypatch.setattr("fab_tui.saved_decks.INDEX_PATH", saved_dir / "index.json")

    game_deck = {f"c_{i}": 1 for i in range(40)}
    card_pool = dict(game_deck)
    card_pool["sb_red"] = 2
    equipment = "briar romping_club blossom_of_spring snapdragon_scalers"

    path = save_user_deck(
        baseline_deck=game_deck,
        card_pool=card_pool,
        equipment_header=equipment,
        hero_id="briar",
        hero_class="Elementalist",
        game_format="silver_age",
        label="Briar vs Briar — test list",
        opponent_hero_id="briar",
        baseline_label="Guide policy deck",
    )
    assert path.is_file()
    assert is_saved_user_deck(path)

    entries = list_saved_user_decks()
    assert len(entries) == 1
    assert entries[0].label == "Briar vs Briar — test list"
    assert entries[0].path == path.resolve()

    deck, pool = load_deck_and_pool(path)
    assert sum(deck.values()) == 40
    assert pool["sb_red"] == 2
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["equipment_header"] == equipment
    assert data["saved_meta"]["opponent_hero_id"] == "briar"


def test_is_saved_user_deck_false_for_plain_json(tmp_path: Path) -> None:
    path = tmp_path / "plain.json"
    path.write_text(json.dumps({"deck": {"a_red": 40}}), encoding="utf-8")
    assert not is_saved_user_deck(path)
