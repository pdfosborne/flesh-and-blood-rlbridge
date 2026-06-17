"""Tests for eval win-rate chart helpers."""
import json
from pathlib import Path

from eval_phase3_checkpoint import _cumulative_win_rates


def test_cumulative_win_rates_running_totals() -> None:
    history = [
        {"wins": 0, "eval_episodes": 10, "timeouts": 7},
        {"wins": 3, "eval_episodes": 10, "timeouts": 7},
        {"wins": 4, "eval_episodes": 10, "timeouts": 6},
    ]
    y_all, y_dec = _cumulative_win_rates(history)
    assert y_all == [0.0, 15.0, 100.0 * 7 / 30]
    assert y_dec[0] == 0.0
    assert abs(y_dec[1] - 50.0) < 1e-9
    assert abs(y_dec[2] - 100.0 * 7 / 10) < 1e-9


def test_cumulative_chart_from_real_history_if_present() -> None:
    history_path = (
        Path(__file__).resolve().parents[1]
        / "results/matchup_sims/briar_vs_riptide/p3_edc3ad02-vs-3015819b/p1/eval_history.json"
    )
    if not history_path.is_file():
        return

    history = json.loads(history_path.read_text(encoding="utf-8"))
    y_all, y_dec = _cumulative_win_rates(history)
    assert len(y_all) == len(history)
    assert all(0.0 <= v <= 100.0 for v in y_all)
    assert all(0.0 <= v <= 100.0 for v in y_dec)
    assert y_all[-1] <= y_dec[-1] + 1e-6 or y_dec[-1] <= y_all[-1] + 1e-6
