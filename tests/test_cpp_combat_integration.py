"""Integration test for C++ attack → block → damage flow (requires built engine)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flesh_and_blood_rlbridge.cpp_engine_environment import (  # noqa: E402
    CppEngineEnvironment,
    is_cpp_engine_available,
)

ENGINE_DIR = REPO_ROOT / "results" / "cpp_engines" / "Ira_vs_Ira-eb83df4f3135d489"
pytestmark = pytest.mark.skipif(
    not is_cpp_engine_available(ENGINE_DIR),
    reason="Ira_vs_Ira C++ engine not built",
)


def _find_attack_step(env: CppEngineEnvironment, *, max_seeds: int = 32) -> bool:
    for seed in range(max_seeds):
        env.fast_reset(seed=seed)
        gs = env._gs
        legal = env._filter_legal_actions(env._legal_actions())
        for idx, action in enumerate(legal):
            if int(action.action_code) != 27:
                continue
            hand_index = int(str(action.button_input))
            hand = list(gs.p1_hand if env._acting_player == 1 else gs.p2_hand)
            if hand_index >= len(hand):
                continue
            card = hand[hand_index]
            if "attack" not in str(card.card_type).lower():
                continue
            env.fast_step_index(idx)
            return int(gs.phase) == 4 and int(gs.pending_attack_power) > 0
    return False


def test_attack_enters_block_phase_with_pending_power() -> None:
    env = CppEngineEnvironment(engine_dir=ENGINE_DIR, deck1="Ira", deck2="Ira")
    assert _find_attack_step(env), "expected at least one affordable attack in sample seeds"


def test_block_pass_resolves_damage_and_returns_to_main() -> None:
    env = CppEngineEnvironment(engine_dir=ENGINE_DIR, deck1="Ira", deck2="Ira")
    if not _find_attack_step(env):
        pytest.skip("no affordable attack in sample seeds")

    gs = env._gs
    attacker = int(gs.active_player)
    defender_hp_before = int(gs.p2_health if attacker == 0 else gs.p1_health)
    pending = int(gs.pending_attack_power)

    legal = env._filter_legal_actions(env._legal_actions())
    pass_idx = next(
        index for index, action in enumerate(legal) if int(action.action_code) == 99
    )
    env.fast_step_index(pass_idx)

    assert int(gs.phase) == 1
    assert int(gs.priority) == attacker
    assert int(gs.pending_attack_power) == 0
    defender_hp_after = int(gs.p2_health if attacker == 0 else gs.p1_health)
    assert defender_hp_after == defender_hp_before - pending
