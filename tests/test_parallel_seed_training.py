"""Tests for parallel RNG-seed training helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "training"))

from runtime_defaults import DEFAULT_PARALLEL_SEEDS  # noqa: E402
from parallel_seed_training import (  # noqa: E402
    SEED_STRIDE,
    average_win_rates,
    derive_training_seed,
    merge_parallel_seed_checkpoint_history,
    select_best_agents_by_win_rate,
    workers_per_parallel_seed,
)


def test_derive_training_seed_stride() -> None:
    assert derive_training_seed(None, 3) is None
    assert derive_training_seed(7, 0) == 7
    assert derive_training_seed(7, 2) == 7 + 2 * SEED_STRIDE


def test_average_win_rates() -> None:
    p1, p2 = average_win_rates([(0.6, 0.4), (0.8, 0.2)])
    assert abs(p1 - 0.7) < 1e-9
    assert abs(p2 - 0.3) < 1e-9
    assert average_win_rates([]) == (0.0, 0.0)


def test_select_best_agents_by_win_rate() -> None:
    rows = [
        {
            "seed_index": 0,
            "p1_win_rate": 0.5,
            "p2_win_rate": 0.6,
            "p1_agent": "a0",
            "p2_agent": "b0",
        },
        {
            "seed_index": 1,
            "p1_win_rate": 0.7,
            "p2_win_rate": 0.4,
            "p1_agent": "a1",
            "p2_agent": "b1",
        },
    ]
    p1, p2, i1, i2 = select_best_agents_by_win_rate(rows)
    assert p1 == "a1"
    assert p2 == "b0"
    assert i1 == 1
    assert i2 == 0


def test_merge_parallel_seed_checkpoint_history(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    for seed_idx, rates in [(0, [0.40, 0.55]), (1, [0.50, 0.48])]:
        seed_dir = candidate / "parallel_seeds" / f"seed_{seed_idx}"
        seed_dir.mkdir(parents=True)
        history = [
            {
                "episodes_completed": 25,
                "p1_win_rate": rates[0],
                "p1_wins": int(rates[0] * 100),
                "losses": 100 - int(rates[0] * 100),
                "draws": 0,
                "timeouts": 0,
                "eval_episodes": 100,
            },
            {
                "episodes_completed": 50,
                "p1_win_rate": rates[1],
                "p1_wins": int(rates[1] * 100),
                "losses": 100 - int(rates[1] * 100),
                "draws": 0,
                "timeouts": 0,
                "eval_episodes": 100,
            },
        ]
        (seed_dir / "checkpoint_eval_history.json").write_text(
            json.dumps(history),
            encoding="utf-8",
        )

    merged = merge_parallel_seed_checkpoint_history(candidate, write=True)
    assert len(merged) == 2
    assert merged[0]["p1_win_rate"] == pytest.approx(0.50)
    assert merged[0]["best_p1_seed_index"] == 1
    assert merged[1]["p1_win_rate"] == pytest.approx(0.55)
    assert merged[1]["best_p1_seed_index"] == 0
    assert (candidate / "checkpoint_eval_history.json").is_file()


def test_default_parallel_seeds() -> None:
    assert DEFAULT_PARALLEL_SEEDS == 3


def test_workers_per_parallel_seed() -> None:
    assert workers_per_parallel_seed(16, 5) == 3
    assert workers_per_parallel_seed(16, 16) == 1
    assert workers_per_parallel_seed(None, 4, cpu_count=8) == 2
    assert workers_per_parallel_seed(0, 3) == 1
