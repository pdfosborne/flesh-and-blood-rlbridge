"""Tests for parity failure analysis script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "cpp"))

from analyze_parity_failures import analyze_sweep_dir  # noqa: E402


def test_analyze_sweep_dir_writes_backlog(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "advanced_test"
    sweep_dir.mkdir()
    report = {
        "discrepancies": [
            {
                "taxonomy": "zone_hand",
                "category": "game_state",
                "description": "hand mismatch",
                "card_id": "snatch_red",
            }
        ],
        "discrepancies_found": 1,
    }
    report_path = tmp_path / "parity_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    summary = [
        {
            "deck1": "Ira",
            "deck2": "Ira",
            "status": "discrepancy",
            "report": str(report_path),
        }
    ]
    (sweep_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    payload = analyze_sweep_dir(sweep_dir)
    assert payload["taxonomy_counts"]["zone_hand"] == 1
    assert (sweep_dir / "fix_backlog.md").is_file()
