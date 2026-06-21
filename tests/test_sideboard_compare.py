"""Tests for sideboard candidate generation and guide swap helpers."""

from __future__ import annotations

import sys

import pytest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts" / "training"))

from flesh_and_blood_rlbridge.sideboard_guide_policy import (  # noqa: E402
    apply_sideboard_swap,
    enumerate_ranked_swaps,
    simulate_guide_sideboard_deck,
    sideboard_inventory,
)
from scripts.training.train_pipeline_common import (  # noqa: E402
    generate_sideboard_candidates,
    load_deck_and_pool_from_json,
)
from scripts.training.train_sideboard_compare import resolve_max_parallel  # noqa: E402


def test_resolve_max_parallel() -> None:
    assert resolve_max_parallel(0, 4) == 4
    assert resolve_max_parallel(-1, 3) == 3
    assert resolve_max_parallel(2, 4) == 2
    assert resolve_max_parallel(10, 4) == 4
    assert resolve_max_parallel(0, 0) == 1


def test_sideboard_inventory() -> None:
    pool = {"a": 2, "b": 2, "c": 1}
    deck = {"a": 2, "b": 1}
    inv = sideboard_inventory(pool, deck)
    assert inv == {"b": 1, "c": 1}


def test_apply_sideboard_swap() -> None:
    pool = {"in_red": 2, "out_red": 2, "neutral": 2}
    deck = {"out_red": 2, "neutral": 2}
    swapped = apply_sideboard_swap(deck, pool, "out_red", "in_red")
    assert swapped is not None
    assert swapped["out_red"] == 1
    assert swapped["in_red"] == 1
    assert swapped["neutral"] == 2


def test_apply_sideboard_swap_rejects_missing_inventory() -> None:
    pool = {"out_red": 2}
    deck = {"out_red": 2}
    assert apply_sideboard_swap(deck, pool, "out_red", "missing") is None


def test_enumerate_ranked_swaps_sorted_by_margin() -> None:
    pool = {
        "slow_red": 2,
        "fast_red": 2,
        "block_red": 2,
    }
    deck = {"slow_red": 2, "fast_red": 2}
    swaps = enumerate_ranked_swaps(
        pool,
        deck,
        "fai",
        pool_by_id={
            "slow_red": {"keywords": ["clash"], "defense": 2, "power": 2},
            "fast_red": {"keywords": ["go_again"], "defense": 2, "power": 4},
            "block_red": {"card_types": ["defense_reaction"], "defense": 5, "power": 0},
        },
        min_margin=0.0,
    )
    assert swaps
    margins = [swap.margin for swap in swaps]
    assert margins == sorted(margins, reverse=True)


def test_simulate_guide_sideboard_deck_reaches_min_size() -> None:
    pool = {f"card_{i}": 1 for i in range(45)}
    deck = simulate_guide_sideboard_deck(
        pool,
        "briar",
        hero_id="aurora",
        game_format="silver_age",
    )
    assert sum(deck.values()) == 40


def test_generate_sideboard_candidates_includes_baseline_and_swaps() -> None:
    pool = {f"card_{i}": 1 for i in range(45)}
    pool["slow_red"] = 2
    pool["block_red"] = 2
    deck = {f"card_{i}": 1 for i in range(38)}
    deck["slow_red"] = 2
    pool_by_id = {
        cid: {"keywords": [], "defense": 3, "power": 3}
        for cid in pool
    }
    pool_by_id["slow_red"] = {"keywords": ["clash"], "defense": 2, "power": 2}
    pool_by_id["block_red"] = {
        "card_types": ["defense_reaction"],
        "defense": 5,
        "power": 0,
    }

    candidates = generate_sideboard_candidates(
        pool,
        deck,
        "fai",
        pool_by_id,
        hero_id="aurora",
        game_format="silver_age",
        num_options=3,
        include_guide_full=False,
        min_swap_margin=0.0,
    )
    ids = [c.candidate_id for c in candidates]
    assert "baseline" in ids
    assert any(cid.startswith("swap_") for cid in ids)
    assert len(candidates) <= 3


def test_load_sideboard_candidates_from_json_merges_sideboard(tmp_path: Path) -> None:
    deck_json = tmp_path / "deck.json"
    deck_json.write_text(
        """
        {
          "deck": {"a_red": 2, "b_red": 2},
          "sideboard": {"c_red": 2}
        }
        """,
        encoding="utf-8",
    )
    game_deck, card_pool = load_deck_and_pool_from_json(str(deck_json))
    assert game_deck == {"a_red": 2, "b_red": 2}
    assert card_pool == {"a_red": 2, "b_red": 2, "c_red": 2}


def test_default_agent_cache_dir_is_top_level_results() -> None:
    from scripts.training.train_pipeline_common import (  # noqa: PLC0415
        DEFAULT_AGENT_CACHE_DIR,
        REPO_ROOT,
    )

    assert DEFAULT_AGENT_CACHE_DIR == REPO_ROOT / "results" / "agent_cache"


def test_resolve_play_checkpoint_interval_pct() -> None:
    from scripts.training.train_play import (  # noqa: PLC0415
        DEFAULT_CHECKPOINT_INTERVAL_PCT,
        resolve_play_checkpoint_interval,
    )

    assert resolve_play_checkpoint_interval(
        500,
        checkpoint_interval_pct=DEFAULT_CHECKPOINT_INTERVAL_PCT,
    ) == 25
    assert resolve_play_checkpoint_interval(100, checkpoint_interval_pct=10.0) == 10
    assert resolve_play_checkpoint_interval(37, checkpoint_interval_pct=10.0) == 4
    assert resolve_play_checkpoint_interval(500, checkpoint_interval=25) == 25


def test_final_eval_delta_and_ranking() -> None:
    from scripts.training.train_sideboard_compare import (  # noqa: PLC0415
        _attach_final_eval_deltas,
        _baseline_final_eval_win_rate,
        _rank_sideboard_results,
    )

    results = [
        {
            "candidate_id": "baseline",
            "label": "Default",
            "play_win_rate": 0.55,
            "final_eval_win_rate": 0.50,
        },
        {
            "candidate_id": "manual_01",
            "label": "Swap A",
            "play_win_rate": 0.52,
            "final_eval_win_rate": 0.58,
        },
        {
            "candidate_id": "manual_02",
            "label": "Swap B",
            "play_win_rate": 0.60,
            "final_eval_win_rate": 0.44,
        },
    ]
    _attach_final_eval_deltas(results)
    assert _baseline_final_eval_win_rate(results) == 0.50
    assert results[1]["final_eval_delta_vs_baseline"] == pytest.approx(0.08)
    assert results[2]["final_eval_delta_vs_baseline"] == pytest.approx(-0.06)

    ranked = _rank_sideboard_results(results)
    assert ranked[0]["candidate_id"] == "manual_01"

