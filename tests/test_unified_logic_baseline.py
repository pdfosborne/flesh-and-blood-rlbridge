"""Tests for async logic-vs-logic baselines in unified training."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

from fab_bridge.unified_dashboard import LOGIC_VS_LOGIC_BASELINE_NAME  # noqa: E402
from unified_logic_baseline import (  # noqa: E402
    run_batch_logic_vs_logic_baselines,
    run_matchup_logic_vs_logic_baseline,
    submit_batch_logic_vs_logic_baselines,
)


def _sample_metrics(*, episodes: int = 20) -> dict:
    return {
        "episodes": episodes,
        "p1_wins": 10,
        "p2_wins": 10,
        "draws": 0,
        "timeouts": 0,
        "errors": 0,
        "p1_win_rate": 0.5,
        "p2_win_rate": 0.5,
        "draw_rate": 0.0,
        "timeout_rate": 0.0,
        "p1_policy": "logic",
        "p2_policy": "logic",
    }


def test_run_matchup_logic_vs_logic_baseline_writes_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from train_dual_agent_common import Matchup  # noqa: E402

    calls: list[int] = []

    def fake_evaluate(*_args, episodes: int = 0, **_kwargs):
        calls.append(int(episodes))
        return _sample_metrics(episodes=episodes)

    monkeypatch.setattr(
        "train_play.evaluate_logic_vs_logic",
        fake_evaluate,
    )
    monkeypatch.setattr(
        "fab_bridge.unified_dashboard.maybe_refresh_unified_dashboard",
        lambda *_args, **_kwargs: None,
    )

    matchup = Matchup(
        name="a-vs-b",
        p1_deck="deck_a",
        p2_deck="deck_b",
        description="",
        dir_name="match_a",
    )
    metrics = run_matchup_logic_vs_logic_baseline(
        matchup,
        out_dir=tmp_path,
        base_url="http://localhost/game",
        game_format="silver_age",
        max_steps=100,
        episodes=20,
        seed=7,
    )
    assert calls == [20]
    assert metrics["p1_win_rate"] == 0.5
    path = tmp_path / "match_a" / LOGIC_VS_LOGIC_BASELINE_NAME
    assert path.is_file()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["episodes"] == 20


def test_run_batch_logic_vs_logic_baselines_allocates_total(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from train_dual_agent_common import Matchup  # noqa: E402

    episode_calls: list[int] = []

    def fake_run_matchup(matchup, *, episodes: int, **_kwargs):
        episode_calls.append(int(episodes))
        path = tmp_path / matchup.dir_name / LOGIC_VS_LOGIC_BASELINE_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_sample_metrics(episodes=episodes)),
            encoding="utf-8",
        )
        return _sample_metrics(episodes=episodes)

    monkeypatch.setattr(
        "unified_logic_baseline.run_matchup_logic_vs_logic_baseline",
        fake_run_matchup,
    )
    monkeypatch.setattr(
        "unified_logic_baseline.submit_checkpoint_eval",
        lambda fn, **_: fn(),
    )

    matchups = {
        f"match_{index}": Matchup(
            name=f"m{index}-vs-n",
            p1_deck=f"d{index}a",
            p2_deck=f"d{index}b",
            description="",
            dir_name=f"match_{index}",
        )
        for index in range(5)
    }
    run_batch_logic_vs_logic_baselines(
        matchups,
        out_dir=tmp_path,
        base_url="http://localhost/game",
        game_format="silver_age",
        max_steps=100,
        total_episodes=100,
        seed=0,
    )
    assert sum(episode_calls) == 100
    assert sorted(episode_calls) == [20, 20, 20, 20, 20]


def test_submit_batch_skips_existing_baselines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from train_dual_agent_common import Matchup  # noqa: E402

    submitted: list[str] = []

    def fake_submit(fn, *, label: str = "", run_dir=None) -> MagicMock:
        submitted.append(label)
        return MagicMock(done=lambda: True)

    monkeypatch.setattr(
        "unified_logic_baseline.submit_checkpoint_eval",
        fake_submit,
    )

    matchup = Matchup(
        name="a-vs-b",
        p1_deck="deck_a",
        p2_deck="deck_b",
        description="",
        dir_name="match_a",
    )
    baseline_path = tmp_path / "match_a" / LOGIC_VS_LOGIC_BASELINE_NAME
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text(json.dumps(_sample_metrics()), encoding="utf-8")

    queued = submit_batch_logic_vs_logic_baselines(
        {"match_a": matchup},
        out_dir=tmp_path,
        base_url="http://localhost/game",
        game_format="silver_age",
        max_steps=100,
        total_episodes=20,
        seed=0,
    )
    assert queued == 0
    assert submitted == []
