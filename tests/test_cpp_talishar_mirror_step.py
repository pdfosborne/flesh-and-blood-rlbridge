"""Regression tests for Talishar-mirrored C++ parity steps."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.cpp_engine_environment import CppEngineEnvironment
from flesh_and_blood_rlbridge.talishar_default_policy import RepeatActionTracker


def _pass_action() -> SimpleNamespace:
    return SimpleNamespace(
        action_code=99,
        button_input="",
        card_id="",
        zone="button",
        label="Pass",
    )


def _build_mirror_step_env(*, game_over: bool = True) -> CppEngineEnvironment:
    env = object.__new__(CppEngineEnvironment)
    pass_action = _pass_action()
    env._gs = SimpleNamespace(
        turn_no=3,
        priority=0,
        p1_health=20,
        p2_health=19,
        phase=4,
        p1_hand_size=1,
        p2_hand_size=4,
        p1_deck_size=30,
        p2_deck_size=30,
        p1_pitch_size=0,
        p2_pitch_size=0,
        game_over=game_over,
        p1_hand=[],
        p2_hand=[],
        get_legal_actions=lambda: [pass_action],
        set_priority=lambda _player: None,
        apply_action=lambda _action: None,
    )
    env._acting_player = 1
    env._flow_phase = "B"
    env._steps = 6
    env._max_turns = 200
    env._hand_playability = {}
    env._turn_no_override = None
    env._talishar_overlay = None
    env._talishar_parity_extra = None
    env._talishar_mirror_state = None
    env._enable_combat_tracker = False
    env._synthetic_combat_log = []
    env._last_observation_vec = None
    env._repeat_tracker = RepeatActionTracker()
    env._arsenal_complete = set()
    return env


def _mirror_payload(*, acting_player_id: int = 1) -> dict:
    return {
        "state": {
            "actingPlayerID": acting_player_id,
            "playerHealth": 20,
            "opponentHealth": 19,
            "turnNo": 3,
            "turnPhase": "M",
            "havePriority": True,
            "playerHandSize": 1,
            "opponentHandSize": 4,
            "playerDeckCount": 30,
            "opponentDeckCount": 30,
            "playerPitchCount": 0,
            "playerHand": [
                {
                    "cardNumber": "WTR001",
                    "action": 0,
                    "actionDataOverride": "0",
                }
            ],
            "legalActions": [{"index": 0, "label": "Pass", "zone": "button"}],
        },
        "reward": 0.009,
        "terminated": False,
        "truncated": False,
        "acting_player_id": acting_player_id,
        "player_hp": 20,
        "opponent_hp": 19,
        "turn": 3,
        "legal_actions": [
            {
                "action_code": 99,
                "button_input": "",
                "card_id": "",
                "zone": "button",
                "label": "Pass",
            }
        ],
    }


def test_mirror_step_skips_flow_pass_and_auto_advance() -> None:
    env = _build_mirror_step_env(game_over=True)
    env.set_talishar_mirror_state(_mirror_payload(acting_player_id=1))

    result = env.step(_pass_action())

    assert result.reward == 0.009
    assert result.terminated is False
    assert result.truncated is False
    assert result.info["acting_player_id"] == 1


def test_mirror_step_uses_talishar_acting_player_not_local_flow() -> None:
    env = _build_mirror_step_env(game_over=False)
    env._acting_player = 2
    env.set_talishar_mirror_state(_mirror_payload(acting_player_id=1))

    result = env.step(_pass_action())

    assert result.info["acting_player_id"] == 1
