"""C++ generator must end turns on main-phase pass (Talishar BeginTurnPass semantics)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "cpp"))

GENERATOR = REPO_ROOT / "scripts" / "cpp" / "generate_cpp_engine.py"


def test_generator_pass_turn_ends_main_phase_in_one_pass() -> None:
    text = GENERATOR.read_text(encoding="utf-8")
    assert "void GameState::_pass_turn()" in text
    assert "void GameState::_finalize_turn_for_player(int player_idx)" in text
    assert "if (phase == TurnPhase::MAIN || phase == TurnPhase::END)" in text
    assert "_pass_turn();" in text


def test_generator_pitch_pass_does_not_end_turn_without_paid_cost() -> None:
    text = GENERATOR.read_text(encoding="utf-8")
    assert "players[active_player].resources >= pending_card_cost" in text
    assert "if (phase == TurnPhase::PITCH)" in text


def test_parity_has_pass_liveness_check() -> None:
    from check_cpp_vs_talishar_parity import (  # noqa: PLC0415
        _check_simulation_pass_liveness,
        _cpp_progress_fingerprint,
    )

    pre = {"turn_no": 1, "phase": "P", "p1_health": 20, "p2_health": 16}
    post = dict(pre)
    ok, msg = _check_simulation_pass_liveness(pre, post, action_label="Pass")
    assert not ok
    assert "unchanged" in msg.lower()
    assert _cpp_progress_fingerprint(pre) == _cpp_progress_fingerprint(post)
