"""Live smoke test for strict simulation parity (requires Talishar + C++ engine)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "cpp"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.cpp_engine_environment import (  # noqa: E402
    get_engine_dir,
    is_cpp_engine_available,
)

TALISHAR_URL = os.environ.get("TALISHAR_URL", "")
SKIP_LIVE = not TALISHAR_URL or not is_cpp_engine_available(get_engine_dir("Ira", "Ira"))


@pytest.mark.skipif(SKIP_LIVE, reason="Talishar URL or Ira_vs_Ira C++ engine unavailable")
def test_ira_simulation_parity_smoke() -> None:
    from check_cpp_vs_talishar_parity import run_parity_check

    report, exit_code = run_parity_check(
        deck1="Ira",
        deck2="Ira",
        episodes=1,
        mode="multi-step",
        steps_per_episode=5,
        talishar_url=TALISHAR_URL,
        parity_mode="simulation",
        sync_scope="hands",
        write_reports=False,
        verbose=False,
    )
    assert report.episodes_run == 1
    assert report.total_steps >= 0
    # Smoke test: run completes without setup failure (exit 2)
    assert exit_code != 2
