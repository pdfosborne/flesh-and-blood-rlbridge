"""Tests for fab_tui experiment spec helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fab_tui.config import ExperimentSpec, SideboardCompareSpec


def test_experiment_spec_pins_out_dir_on_first_resolve() -> None:
    spec = ExperimentSpec(name="demo-run")
    first = spec.resolved_out_dir()
    second = spec.resolved_out_dir()
    assert first == second
    assert spec.out_dir == str(first)


def test_sideboard_compare_spec_pins_out_dir_on_first_resolve() -> None:
    spec = SideboardCompareSpec(
        starting_deck="deck.json",
        opponent_hero_id="dorinthea",
        opponent_deck="DorintheSAGEPrecon",
        hero_id="briar",
    )
    first = spec.resolved_out_dir()
    second = spec.resolved_out_dir()
    assert first == second
    assert spec.out_dir == str(first)
    assert first.name.startswith("briar_vs_dorinthea_")


def test_pipeline_argv_uses_single_out_dir() -> None:
    spec = ExperimentSpec(name="demo-run")
    argv = spec.pipeline_argv()
    out_dirs = [
        argv[index + 1]
        for index, token in enumerate(argv)
        if token == "--out-dir"
    ]
    results_json = [
        argv[index + 1]
        for index, token in enumerate(argv)
        if token == "--results-json"
    ]
    assert len(out_dirs) == 1
    assert len(results_json) == 1
    assert results_json[0] == str(Path(out_dirs[0]) / "results.json")
