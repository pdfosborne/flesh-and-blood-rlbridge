"""Regression tests for C++ fast-step action-index alignment."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.cpp_engine_environment import CppEngineEnvironment
from flesh_and_blood_rlbridge.deck_context import EpisodeContext
from flesh_and_blood_rlbridge.legal_action_filter import filter_legal_actions
from flesh_and_blood_rlbridge.player_observation import PLAYER_OBS_DIM, SCALAR_OFF
from flesh_and_blood_rlbridge.state_loop_guard import TurnLoopGuard
from flesh_and_blood_rlbridge.talishar_default_policy import RepeatActionTracker


class _FakeCard:
    def __init__(self, *, card_id: str, name: str, cost: int = 0, pitch: int = 1) -> None:
        self.card_id = card_id
        self.name = name
        self.cost = cost
        self.pitch = pitch
        self.power = 0
        self.defense = 0


class _FakeAction:
    def __init__(
        self,
        *,
        action_code: int,
        button_input: str,
        card_id: str,
        zone: str,
        label: str,
    ) -> None:
        self.action_code = action_code
        self.button_input = button_input
        self.card_id = card_id
        self.zone = zone
        self.label = label


class _RecordingGS:
    """Minimal GameState stub that records which fast-step index was used."""

    def __init__(self) -> None:
        self.priority = 0
        self.turn_no = 1
        self.p1_health = 20
        self.p2_health = 20
        self.p1_hand_size = 2
        self.p2_hand_size = 2
        self.p1_deck_size = 30
        self.p2_deck_size = 30
        self.p1_pitch_size = 0
        self.p2_pitch_size = 0
        self.game_over = False
        self.winner = -1
        self.first_player = 1
        self.consecutive_passes = 0
        self.phase = 1
        self.p1_hand = [
            _FakeCard(card_id="cheap_red", name="Cheap", cost=0, pitch=1),
            _FakeCard(card_id="dear_red", name="Dear", cost=3, pitch=1),
        ]
        self.p2_hand = []
        self.p1_equipment = []
        self.p2_equipment = []
        self.p1_arsenal = []
        self.p2_arsenal = []
        self.p1_pitch = []
        self.p2_pitch = []
        self.p1_discard = []
        self.p2_discard = []
        self.last_fast_index: int | None = None

    def get_legal_actions(self) -> list[_FakeAction]:
        return [
            _FakeAction(
                action_code=27,
                button_input="0",
                card_id="cheap_red",
                zone="hand",
                label="Cheap",
            ),
            _FakeAction(
                action_code=27,
                button_input="1",
                card_id="dear_red",
                zone="hand",
                label="Dear",
            ),
            _FakeAction(
                action_code=99,
                button_input="",
                card_id="",
                zone="button",
                label="Pass",
            ),
        ]

    def fast_step_index(self, action_index: int) -> SimpleNamespace:
        self.last_fast_index = int(action_index)
        return SimpleNamespace(
            legal_count=1,
            acting_player_id=1,
            reward=0.0,
            terminated=False,
            winner=-1,
            p1_health=self.p1_health,
            p2_health=self.p2_health,
            turn_no=self.turn_no,
        )


def _build_fast_env(*, acting_player: int = 1) -> CppEngineEnvironment:
    env = object.__new__(CppEngineEnvironment)
    env._gs = _RecordingGS()
    env._acting_player = acting_player
    env._flow_phase = "M"
    env._arsenal_complete = set()
    env._turn_no_override = None
    env._talishar_overlay = None
    env._talishar_mirror_state = None
    env._talishar_raw_state = None
    env._talishar_parity_extra = None
    env._hand_playability = {}
    env._steps = 0
    env._max_turns = 200
    env._p1_hp = 20
    env._p2_hp = 20
    env._enable_combat_tracker = False
    env._synthetic_combat_log = []
    env._last_observation_vec = None
    env._repeat_tracker = RepeatActionTracker()
    env._loop_guard = TurnLoopGuard()
    env._deck1 = "deck_a"
    env._deck2 = "deck_b"
    ctx_p1 = EpisodeContext(
        self_hero_id="hero_a",
        opp_hero_id="hero_b",
        format="silver_age",
        self_deck_counts={"cheap_red": 1},
        first_player=1,
    )
    ctx_p2 = EpisodeContext(
        self_hero_id="hero_b",
        opp_hero_id="hero_a",
        format="silver_age",
        self_deck_counts={"cheap_red": 1},
        first_player=1,
    )
    env._p1_episode_context = ctx_p1
    env._p2_episode_context = ctx_p2
    return env


def test_cpp_fast_step_maps_filtered_pass_to_cpp_pass_index() -> None:
    env = _build_fast_env()
    raw = env._gs.get_legal_actions()
    state = {
        "turnPhase": {"turnPhase": "M"},
        "playerHand": [
            {
                "action": 27,
                "actionDataOverride": "0",
                "cardNumber": "cheap_red",
                "cost": 0,
                "resource": 1,
            },
            {
                "action": 27,
                "actionDataOverride": "1",
                "cardNumber": "dear_red",
                "cost": 3,
                "resource": 1,
            },
        ],
        "playerPitchCount": 0,
    }
    filtered = filter_legal_actions(state, [env._action_to_dict(a) for a in raw])
    assert len(filtered) == 2  # cheap play + pass; dear stripped

    pass_idx = next(
        index
        for index, action in enumerate(filtered)
        if int(action["action_code"]) == 99
    )
    env.fast_step_index(pass_idx)
    assert env._gs.last_fast_index == 2


def test_opening_main_fast_step_advances_flow_without_cpp_play() -> None:
    env = _build_fast_env()
    env._flow_phase = "OPENING_MAIN"
    env._gs.turn_no = 0
    env._steps = 0

    result = env.fast_step_index(0)

    assert env._flow_phase == "ARS"
    assert env._gs.last_fast_index is None
    assert result["acting_player_id"] == 1
    assert len(result["obs_vec"]) == PLAYER_OBS_DIM


def test_observation_swaps_player_perspective_by_acting_seat() -> None:
    from flesh_and_blood_rlbridge.player_observation import player_observation_vector

    raw = {
        "playerEquipment": [{"cardNumber": "eq_a"}],
        "opponentEquipment": [{"cardNumber": "eq_b"}],
        "playerArse": [{"cardNumber": "ars_a"}],
        "opponentArse": [],
        "playerPitchCount": 1,
        "opponentPitchCount": 2,
        "canPassPhase": True,
        "amIActivePlayer": True,
        "turnPlayer": 1,
        "firstPlayer": 1,
    }
    ctx_p1 = EpisodeContext(
        self_hero_id="hero_a",
        opp_hero_id="hero_b",
        format="silver_age",
        self_deck_counts={"card_a": 1},
        first_player=1,
    )
    ctx_p2 = EpisodeContext(
        self_hero_id="hero_b",
        opp_hero_id="hero_a",
        format="silver_age",
        self_deck_counts={"card_b": 1},
        first_player=1,
    )
    obs_p1 = player_observation_vector(
        {
            "actingPlayerID": 1,
            "playerHealth": 18,
            "opponentHealth": 20,
            "playerHand": [],
            "playerHandSize": 0,
            "opponentHandSize": 4,
        },
        [],
        episode_context=ctx_p1,
        acting_player_id=1,
        p1_health=18,
        p2_health=20,
        raw_talishar_state=raw,
    )
    raw_p2 = {
        "playerEquipment": [{"cardNumber": "eq_b"}],
        "opponentEquipment": [{"cardNumber": "eq_a"}],
        "playerArse": [],
        "opponentArse": [{"cardNumber": "ars_a"}],
        "playerPitchCount": 2,
        "opponentPitchCount": 1,
        "canPassPhase": True,
        "amIActivePlayer": True,
        "turnPlayer": 2,
        "firstPlayer": 1,
    }
    obs_p2 = player_observation_vector(
        {
            "actingPlayerID": 2,
            "playerHealth": 20,
            "opponentHealth": 18,
            "playerHand": [],
            "playerHandSize": 4,
            "opponentHandSize": 0,
        },
        [],
        episode_context=ctx_p2,
        acting_player_id=2,
        p1_health=18,
        p2_health=20,
        raw_talishar_state=raw_p2,
    )
    assert obs_p1.shape == (PLAYER_OBS_DIM,)
    assert obs_p2.shape == (PLAYER_OBS_DIM,)
    assert not np.allclose(obs_p1, obs_p2)
    # Absolute P1/P2 health scalars are seat-invariant; acting scalar differs.
    assert obs_p1[SCALAR_OFF + 3] == obs_p2[SCALAR_OFF + 3]
    assert obs_p1[SCALAR_OFF + 4] == obs_p2[SCALAR_OFF + 4]
    assert obs_p1[SCALAR_OFF] != obs_p2[SCALAR_OFF]
