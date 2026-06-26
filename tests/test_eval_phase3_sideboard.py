"""Tests for phase-3 checkpoint discovery on sideboard compare outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts" / "eval"))

from eval_phase3_checkpoint import (  # noqa: E402
    _latest_checkpoint,
    is_sideboard_compare_dir,
    list_sideboard_candidate_ids,
)
from fab_tui.results import _label_for_results_dir  # noqa: E402


def _write_checkpoint(
    candidate_dir: Path,
    *,
    matchup: str,
    episode: int,
    win_rate: float = 0.5,
) -> Path:
    ckpt_dir = candidate_dir / matchup / "p1" / f"episode_{episode:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "weights").mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "weights" / "agent_weights.json").write_text("{}", encoding="utf-8")
    (ckpt_dir / "metadata.json").write_text(
        json.dumps({
            "matchup": matchup,
            "episodes_completed": episode,
            "target_episodes": 100,
            "game_format": "silver_age",
            "p1_hero": "aurora",
            "p2_hero": "briar",
            "opponent_mode": "preset",
            "opponent_deck_name": "BriarSAGEPrecon",
            "deck_spec": {
                "equipment_header": "aurora",
                "cards": {"a_red": 40},
            },
            "win_rate": win_rate,
        }),
        encoding="utf-8",
    )
    return ckpt_dir


def test_sideboard_compare_dir_detection(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    (out_dir / "candidates").mkdir(parents=True)
    (out_dir / "candidates_manifest.json").write_text("{}", encoding="utf-8")
    assert is_sideboard_compare_dir(out_dir)
    assert not is_sideboard_compare_dir(tmp_path)


def test_latest_checkpoint_finds_sideboard_candidate(tmp_path: Path) -> None:
    out_dir = tmp_path / "sideboard_run"
    baseline_dir = out_dir / "candidates" / "baseline"
    swap_dir = out_dir / "candidates" / "swap_01"
    baseline_dir.mkdir(parents=True)
    swap_dir.mkdir(parents=True)
    (out_dir / "candidates_manifest.json").write_text(
        json.dumps({
            "hero_id": "aurora",
            "opponent_hero_id": "briar",
            "candidates": [
                {"candidate_id": "baseline", "label": "Baseline"},
                {"candidate_id": "swap_01", "label": "Swap"},
            ],
        }),
        encoding="utf-8",
    )

    _write_checkpoint(baseline_dir, matchup="p3_base-vs-opp", episode=50)
    _write_checkpoint(swap_dir, matchup="p3_swap-vs-opp", episode=80)

    baseline_bundle = _latest_checkpoint(out_dir, "p1", candidate_id="baseline")
    assert baseline_bundle is not None
    assert baseline_bundle.episodes_completed == 50

    swap_bundle = _latest_checkpoint(out_dir, "p1", candidate_id="swap_01")
    assert swap_bundle is not None
    assert swap_bundle.episodes_completed == 80

    latest_any = _latest_checkpoint(out_dir, "p1")
    assert latest_any is not None
    assert latest_any.episodes_completed == 80

    assert list_sideboard_candidate_ids(out_dir) == ["baseline", "swap_01"]


def test_latest_checkpoint_finds_parallel_seed_sideboard_candidate(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "sideboard_run"
    candidate_dir = out_dir / "candidates" / "manual_01"
    candidate_dir.mkdir(parents=True)
    (out_dir / "candidates_manifest.json").write_text(
        json.dumps({
            "hero_id": "briar",
            "opponent_hero_id": "briar",
            "candidates": [{"candidate_id": "manual_01", "label": "Manual"}],
        }),
        encoding="utf-8",
    )

    seed_dir = candidate_dir / "parallel_seeds" / "seed_0"
    ckpt_dir = seed_dir / "p3_abc-vs-GEPrecon" / "p1" / "episode_000050"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "weights").mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "weights" / "agent_weights.json").write_text("{}", encoding="utf-8")
    (ckpt_dir / "metadata.json").write_text(
        json.dumps({
            "matchup": "p3_abc-vs-GEPrecon",
            "episodes_completed": 50,
            "target_episodes": 100,
            "game_format": "silver_age",
            "p1_hero": "briar",
            "p2_hero": "briar",
            "opponent_mode": "preset",
            "opponent_deck_name": "GEPrecon",
            "deck_spec": {
                "equipment_header": "briar",
                "cards": {"a_red": 40},
            },
            "win_rate": 0.55,
        }),
        encoding="utf-8",
    )

    bundle = _latest_checkpoint(out_dir, "p1", candidate_id="manual_01")
    assert bundle is not None
    assert bundle.episodes_completed == 50
    assert "parallel_seeds" in str(bundle.checkpoint_dir)


def test_resolve_parity_deck_names_prefers_eval_decks() -> None:
    from eval_phase3_checkpoint import (  # noqa: PLC0415
        CheckpointBundle,
        _resolve_parity_deck_names,
    )

    bundle = CheckpointBundle(
        role="p1",
        checkpoint_dir=Path("/tmp/ep"),
        metadata={
            "p1_deck": "rl_p3_p1_f61998c2",
            "p2_deck": "GEPrecon",
        },
        weights_path=Path("/tmp/ep/weights/agent_weights.json"),
    )
    deck1, deck2, err = _resolve_parity_deck_names(
        bundle,
        None,
        deck1="eval_p1_1bd9b946",
        deck2="DorintheSAGEPrecon",
    )
    assert err == ""
    assert deck1 == "eval_p1_1bd9b946"
    assert deck2 == "DorintheSAGEPrecon"


def test_latest_checkpoint_prefers_highest_episode_over_mtime(tmp_path: Path) -> None:
    results_dir = tmp_path / "run"
    results_dir.mkdir()
    seeds_root = results_dir / "parallel_seeds"
    seed0 = seeds_root / "seed_0"
    seed1 = seeds_root / "seed_1"
    (results_dir / "parallel_seeds_summary.json").write_text(
        json.dumps({"best_p1_seed_index": 0, "best_p2_seed_index": 0}),
        encoding="utf-8",
    )

    low_ep = seed0 / "p3_a-vs-b" / "p1" / "episode_000095"
    high_mtime_low_ep = seed1 / "p3_c-vs-d" / "p1" / "episode_000005"

    for ckpt_dir, episode, mtime_offset in (
        (low_ep, 95, 0),
        (high_mtime_low_ep, 5, 10_000),
    ):
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "weights").mkdir(parents=True)
        (ckpt_dir / "weights" / "agent_weights.json").write_text("{}", encoding="utf-8")
        (ckpt_dir / "metadata.json").write_text(
            json.dumps({"matchup": ckpt_dir.parents[1].name, "episodes_completed": episode}),
            encoding="utf-8",
        )
        meta_path = ckpt_dir / "metadata.json"
        meta_path.touch()
        ts = meta_path.stat().st_mtime + mtime_offset
        import os

        os.utime(meta_path, (ts, ts))

    bundle = _latest_checkpoint(results_dir, "p1")
    assert bundle is not None
    assert bundle.episodes_completed == 95


def test_label_for_results_dir_uses_deck_state_active_decks(tmp_path: Path) -> None:
    run_dir = tmp_path / "briar_vs_briar"
    run_dir.mkdir()
    (run_dir / "deck_state.json").write_text(
        json.dumps(
            {
                "p1": {"active_decks": {"briar": {}}},
                "p2": {"active_decks": {"briar": {}}},
            }
        ),
        encoding="utf-8",
    )
    assert _label_for_results_dir(run_dir) == "briar vs briar"
