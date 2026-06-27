"""Verify generated C++ engines include attack/block combat flow."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "scripts" / "cpp" / "generate_cpp_engine.py"


@pytest.mark.parametrize(
    "snippet",
    [
        "void GameState::_resolve_combat()",
        "void GameState::_apply_hand_play_main(size_t idx)",
        "void GameState::_apply_block_index(size_t idx)",
        "if (phase == TurnPhase::BLOCK)",
        "pending_attack_power",
        "pending_block_value",
        "active_player",
    ],
)
def test_generator_emits_combat_flow(snippet: str) -> None:
    text = GENERATOR.read_text(encoding="utf-8")
    assert snippet in text
