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
    MetaEngineControls,
    MetaGameControls,
    MetaUnifiedRandomMatchups,
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


def test_engine_controls_from_meta() -> None:
    runtime = build_runtime(
        replace(
            META,
            engine=MetaEngineControls(
                max_steps_per_turn=50,
                step_penalty=-0.002,
                truncation_penalty=-0.2,
                talishar_request_timeout=45.0,
            ),
        )
    )
    assert runtime.engine.max_steps_per_turn == 50
    assert runtime.engine.step_penalty == -0.002
    assert runtime.engine.truncation_penalty == -0.2
    assert runtime.engine.talishar_request_timeout == 45.0
    assert runtime.engine.loop_repeat_threshold == META.engine.loop_repeat_threshold


def test_engine_env_kwargs_from_runtime() -> None:
    kw = rd.engine_env_kwargs(RUNTIME.engine)
    assert kw["max_steps_per_turn"] == RUNTIME.engine.max_steps_per_turn
    assert kw["step_penalty"] == RUNTIME.engine.step_penalty
    assert kw["request_timeout"] == RUNTIME.engine.talishar_request_timeout


def test_episode_timeout_seconds_from_meta() -> None:
    engine = MetaEngineControls(
        episode_timeout_seconds_per_step=2.0,
        episode_timeout_floor_seconds=100.0,
    )
    assert rd.episode_timeout_seconds(200, engine) == 400.0
    assert rd.episode_timeout_seconds(10, engine) == 100.0


def test_apply_meta_updates_engine_aliases() -> None:
    original = META.engine.step_penalty
    try:
        apply_meta(engine=MetaEngineControls(step_penalty=-0.005))
        assert rd.DEFAULT_STEP_PENALTY == -0.005
        assert rd.RUNTIME.engine.step_penalty == -0.005
    finally:
        apply_meta(engine=replace(META.engine, step_penalty=original))


def test_game_controls_from_meta() -> None:
    runtime = build_runtime(
        replace(META, game=MetaGameControls(stall_no_damage_turns=9, stall_min_attack_hand=1))
    )
    assert runtime.game.stall_no_damage_turns == 9
    assert runtime.game.stall_min_attack_hand == 1
    assert runtime.game.stall_low_hand_turns == META.game.stall_low_hand_turns
    kw = rd.game_env_kwargs(runtime.game)
    assert kw["stall_no_damage_turns"] == 9
    assert kw["macro_stall_enabled"] is True


def test_apply_meta_updates_module_aliases() -> None:
    original = META.parallel_seeds
    try:
        apply_meta(parallel_seeds=7)
        assert rd.DEFAULT_PARALLEL_SEEDS == 7
        assert rd.RUNTIME.play.parallel_seeds == 7
    finally:
        apply_meta(parallel_seeds=original)


def test_unified_random_matchups_defaults_from_meta() -> None:
    urm = RUNTIME.unified_random_matchups
    meta_urm = META.unified_random_matchups
    assert urm.matchups == meta_urm.matchups
    assert urm.episodes == meta_urm.episodes
    assert urm.max_steps == meta_urm.max_steps
    expected_workers = meta_urm.workers
    if expected_workers <= 0:
        expected_workers = 1 if META.workers is None else META.workers
    assert urm.workers == expected_workers
    assert urm.checkpoint_eval_episodes == meta_urm.checkpoint_eval_episodes
    assert urm.custom_deck_links == meta_urm.custom_deck_links


def test_unified_episodes_independent_of_play_episodes() -> None:
    runtime = build_runtime(
        replace(META, play_episodes=999, unified_random_matchups=replace(
            META.unified_random_matchups,
            episodes=42,
        ))
    )
    assert runtime.unified_random_matchups.episodes == 42
    assert runtime.matchup_sim.play_episodes == 999
    assert runtime.dual_matchup.episodes == 999
