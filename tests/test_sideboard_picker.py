"""Tests for manual sideboard swap helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "training"))

from fab_tui.sideboard_picker import (
    apply_manual_swap,
    compute_guide_policy_deck,
    load_deck_and_pool,
    variants_to_candidate_payload,
    write_candidates_manifest,
)
from train_pipeline_common import load_sideboard_candidates_from_json


def test_apply_manual_swap_from_inventory() -> None:
    pool = {"out_red": 2, "in_red": 2, "filler_red": 38}
    deck = {"out_red": 2, "filler_red": 38}
    result = apply_manual_swap(deck, pool, "out_red", "in_red")
    assert result is not None
    new_deck, new_pool = result
    assert new_deck["out_red"] == 1
    assert new_deck["in_red"] == 1
    assert sum(new_deck.values()) == 40
    assert new_pool["in_red"] == 2


def test_apply_manual_swap_adds_external_card_to_pool() -> None:
    pool = {f"c_{i}": 1 for i in range(40)}
    deck = dict(pool)
    result = apply_manual_swap(deck, pool, "c_0", "new_card_red")
    assert result is not None
    new_deck, new_pool = result
    assert "c_0" not in new_deck
    assert new_deck["new_card_red"] == 1
    assert new_pool["new_card_red"] == 1


def test_load_deck_and_pool_merges_sideboard() -> None:
    path = Path(__file__).parent / "_tmp_deck.json"
    path.write_text(
        json.dumps(
            {
                "deck": {"a_red": 2, "b_red": 2},
                "sideboard": {"c_red": 2},
            }
        ),
        encoding="utf-8",
    )
    try:
        deck, pool = load_deck_and_pool(path)
        assert deck == {"a_red": 2, "b_red": 2}
        assert pool == {"a_red": 2, "b_red": 2, "c_red": 2}
    finally:
        path.unlink(missing_ok=True)


def test_compute_guide_policy_deck_reaches_min_size() -> None:
    pool = {f"card_{i}": 1 for i in range(45)}
    deck = compute_guide_policy_deck(
        pool,
        opponent_hero_id="briar",
        hero_id="aurora",
        hero_class="Runeblade",
        game_format="silver_age",
    )
    assert sum(deck.values()) == 40


def test_variants_payload_uses_custom_baseline_label() -> None:
    baseline = {f"c_{i}": 1 for i in range(40)}
    payload = variants_to_candidate_payload(
        baseline,
        [],
        baseline_label="Guide policy deck",
    )
    assert payload["candidates"][0]["label"] == "Guide policy deck"


def test_candidates_manifest_round_trip(tmp_path: Path) -> None:
    baseline = {f"c_{i}": 1 for i in range(40)}
    pool = dict(baseline)
    pool["side_red"] = 2
    variants = []
    manifest = write_candidates_manifest(
        tmp_path / "manifest.json",
        baseline_deck=baseline,
        card_pool=pool,
        variants=variants,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(payload["candidates"]) == 1

    loaded, loaded_pool = load_sideboard_candidates_from_json(
        manifest,
        card_pool=baseline,
        min_deck_size=40,
    )
    assert len(loaded) == 1
    assert loaded[0].candidate_id == "baseline"
    assert loaded_pool["side_red"] == 2
