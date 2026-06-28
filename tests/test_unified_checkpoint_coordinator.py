"""Tests for merged async unified checkpoint eval coordinator."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

from unified_checkpoint_eval import (  # noqa: E402
    UnifiedCheckpointCoordinator,
    allocate_eval_episodes,
)


def test_allocate_eval_episodes_even_split() -> None:
    alloc = allocate_eval_episodes(100, ["a", "b", "c", "d", "e"], seed=42)
    assert sum(alloc.values()) == 100
    assert all(count == 20 for count in alloc.values())


def test_allocate_eval_episodes_remainder() -> None:
    alloc = allocate_eval_episodes(10, ["a", "b", "c"], seed=1)
    assert sum(alloc.values()) == 10
    assert sorted(alloc.values()) == [3, 3, 4]


def test_coordinator_waits_for_all_matchups_before_trigger(monkeypatch) -> None:
    submitted: list[int] = []

    def fake_submit(fn, *, label: str = "") -> MagicMock:
        submitted.append(1)
        fut = MagicMock()
        fut.done.return_value = True
        return fut

    monkeypatch.setattr(
        "unified_checkpoint_eval.submit_checkpoint_eval",
        fake_submit,
    )
    monkeypatch.setattr(
        "train_dual_agent_common._save_unified_selfplay_checkpoint",
        lambda **_: None,
    )

    from train_dual_agent_common import Matchup, PPOAgent  # noqa: E402

    policy = MagicMock(spec=PPOAgent)
    policy._shared = object()
    snap = MagicMock(spec=PPOAgent)
    snap._shared = object()

    match_a = Matchup(
        name="a-vs-b",
        p1_deck="deck_a",
        p2_deck="deck_b",
        description="",
        dir_name="match_a",
    )
    match_b = Matchup(
        name="c-vs-d",
        p1_deck="deck_c",
        p2_deck="deck_d",
        description="",
        dir_name="match_b",
    )
    coord = UnifiedCheckpointCoordinator(
        out_dir=Path("/tmp/unified-run"),
        matchups={"match_a": match_a, "match_b": match_b},
        base_url="http://localhost",
        game_format="sage",
        max_steps=100,
        n_episodes=1000,
        warmup_episodes=50,
        checkpoint_interval=100,
        checkpoint_eval_episodes=100,
        unified_policy=policy,
        policy_snapshot_fn=lambda: (snap, snap),
        seed=0,
    )

    coord.report_progress("match_a", 50)
    assert submitted == []

    coord.report_progress("match_b", 40)
    assert submitted == []

    coord.report_progress("match_b", 50)
    assert len(submitted) == 1


def test_coordinator_report_progress_returns_immediately(monkeypatch, tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    def slow_submit(fn, *, label: str = "") -> MagicMock:
        def run() -> None:
            started.set()
            release.wait(timeout=5.0)
        t = threading.Thread(target=run, daemon=True)
        t.start()
        fut = MagicMock()
        fut.done.return_value = False
        return fut

    monkeypatch.setattr(
        "unified_checkpoint_eval.submit_checkpoint_eval",
        slow_submit,
    )
    monkeypatch.setattr(
        "train_dual_agent_common._save_unified_selfplay_checkpoint",
        lambda **_: None,
    )

    from train_dual_agent_common import Matchup, PPOAgent  # noqa: E402

    policy = MagicMock(spec=PPOAgent)
    snap = MagicMock(spec=PPOAgent)
    snap._shared = object()
    matchup = Matchup(
        name="solo",
        p1_deck="d1",
        p2_deck="d2",
        description="",
        dir_name="solo_match",
    )
    coord = UnifiedCheckpointCoordinator(
        out_dir=tmp_path,
        matchups={"solo_match": matchup},
        base_url="http://localhost",
        game_format="sage",
        max_steps=100,
        n_episodes=100,
        warmup_episodes=10,
        checkpoint_interval=50,
        checkpoint_eval_episodes=20,
        unified_policy=policy,
        policy_snapshot_fn=lambda: (snap, snap),
        seed=0,
    )

    t0 = time.monotonic()
    coord.report_progress("solo_match", 10)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5
    assert started.wait(timeout=2.0)
    release.set()


def test_bucket_dedup(monkeypatch) -> None:
    call_eps: list[int] = []

    def fake_submit(fn, *, label: str = "") -> MagicMock:
        fn()
        return MagicMock(done=lambda: True)

    monkeypatch.setattr(
        "unified_checkpoint_eval.submit_checkpoint_eval",
        fake_submit,
    )
    monkeypatch.setattr(
        "train_dual_agent_common._save_unified_selfplay_checkpoint",
        lambda **_: None,
    )

    from train_dual_agent_common import Matchup, PPOAgent  # noqa: E402

    policy = MagicMock(spec=PPOAgent)
    snap = MagicMock(spec=PPOAgent)
    snap._shared = object()
    matchup = Matchup(
        name="solo",
        p1_deck="d1",
        p2_deck="d2",
        description="",
        dir_name="solo_match",
    )

    def fake_run_merged_eval(self, *, episodes_completed, bucket_kind, eval_p1, eval_p2):
        call_eps.append(episodes_completed)
        self.log.append(
            {
                "eval_mode": "merged",
                "episodes_completed": episodes_completed,
                "aggregate": {"self_play_win_rate_mean": 0.5},
                "per_matchup": {
                    "solo_match": {
                        "p1_win_rate": 0.5,
                        "eval_episodes": 20,
                    }
                },
            }
        )

    monkeypatch.setattr(
        UnifiedCheckpointCoordinator,
        "_run_merged_eval",
        fake_run_merged_eval,
    )

    coord = UnifiedCheckpointCoordinator(
        out_dir=Path("/tmp/x"),
        matchups={"solo_match": matchup},
        base_url="http://localhost",
        game_format="sage",
        max_steps=100,
        n_episodes=100,
        warmup_episodes=10,
        checkpoint_interval=50,
        checkpoint_eval_episodes=20,
        unified_policy=policy,
        policy_snapshot_fn=lambda: (snap, snap),
        seed=0,
    )
    coord.report_progress("solo_match", 10)
    coord.report_progress("solo_match", 10)
    assert call_eps == [10]
