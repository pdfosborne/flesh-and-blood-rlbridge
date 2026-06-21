"""Tests for central runtime default configuration."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace

import runtime_defaults as rd  # noqa: E402
from runtime_defaults import (  # noqa: E402
    DEFAULT_CHECKPOINT_EVAL_EPISODES,
    DEFAULT_CHECKPOINT_INTERVAL_PCT,
    DEFAULT_PARALLEL_SEEDS,
    META,
    MetaGameControls,
    RUNTIME,
    apply_meta,
    build_runtime,
)


def test_flat_aliases_match_runtime_sections() -> None:
    assert DEFAULT_PARALLEL_SEEDS == RUNTIME.play.parallel_seeds
    assert DEFAULT_CHECKPOINT_INTERVAL_PCT == RUNTIME.play.checkpoint_interval_pct
    assert DEFAULT_CHECKPOINT_EVAL_EPISODES == RUNTIME.play.checkpoint_eval_episodes


def test_sideboard_compare_defaults_from_meta() -> None:
    sb = RUNTIME.sideboard_compare
    assert sb.max_parallel == META.sideboard_max_parallel
    assert sb.play_episodes == META.play_episodes
    assert sb.num_options == 4


def test_play_episodes_shared_across_workflows() -> None:
    runtime = build_runtime(replace(META, play_episodes=12_345))
    assert runtime.sideboard_compare.play_episodes == 12_345
    assert runtime.full_pipeline.play_episodes == 12_345
    assert runtime.matchup_sim.play_episodes == 12_345
    assert runtime.dual_matchup.episodes == 12_345
    assert runtime.tui.play_episodes == 12_345


def test_final_eval_shared_across_workflows() -> None:
    runtime = build_runtime(replace(META, final_eval_episodes=77, final_eval_max_steps=888))
    assert runtime.sideboard_compare.final_eval_episodes == 77
    assert runtime.sideboard_compare.final_eval_max_steps == 888
    assert runtime.full_pipeline.final_eval_episodes == 77
    assert runtime.full_pipeline.final_eval_max_steps == 888
    assert runtime.matchup_sim.final_eval_episodes == 77
    assert runtime.matchup_sim.final_eval_max_steps == 888
    assert runtime.tui.final_eval_episodes == 77
    assert runtime.tui.final_eval_max_steps == 888


def test_max_play_steps_shared_across_workflows() -> None:
    runtime = build_runtime(replace(META, max_play_steps=777))
    assert runtime.play.max_play_steps == 777
    assert runtime.matchup_sim.max_play_steps == 777
    assert runtime.dual_matchup.max_steps == 777


def test_build_runtime_propagates_meta_workers() -> None:
    runtime = build_runtime(replace(META, workers=8, parallel_seeds=2))
    assert runtime.play.workers == 8
    assert runtime.play.parallel_seeds == 2
    assert runtime.matchup_sim.workers == 8
    assert runtime.dual_matchup.workers == 8
    assert runtime.eval_dashboard.parallel_workers == META.eval_parallel_workers


def test_game_controls_from_meta() -> None:
    runtime = build_runtime(
        replace(META, game=MetaGameControls(stall_no_damage_turns=9, stall_min_attack_hand=1))
    )
    assert runtime.game.stall_no_damage_turns == 9
    assert runtime.game.stall_min_attack_hand == 1
    assert runtime.game.stall_low_hand_turns == META.game.stall_low_hand_turns


def test_apply_meta_updates_module_aliases() -> None:
    original = META.parallel_seeds
    try:
        apply_meta(parallel_seeds=7)
        assert rd.DEFAULT_PARALLEL_SEEDS == 7
        assert rd.RUNTIME.play.parallel_seeds == 7
    finally:
        apply_meta(parallel_seeds=original)
