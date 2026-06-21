"""Tests for fab_tui eval runner command construction."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from fab_tui.config import EnvironmentSettings, EvalSpec
from fab_tui.runner import run_eval_dashboard


def test_run_eval_dashboard_passes_render_only_flag(monkeypatch) -> None:
    captured: list[list[str]] = []

    def _fake_run_streaming(cmd: list[str]) -> int:
        captured.append(cmd)
        return 0

    monkeypatch.setattr("fab_tui.runner.run_streaming", _fake_run_streaming)

    spec = EvalSpec(
        results_dir=str(_REPO / "results" / "demo"),
        render_only=True,
        max_steps=500,
    )
    rc = run_eval_dashboard(spec, EnvironmentSettings())

    assert rc == 0
    assert captured
    assert "--render-only" in captured[0]
    assert "--render-max-steps" in captured[0]
