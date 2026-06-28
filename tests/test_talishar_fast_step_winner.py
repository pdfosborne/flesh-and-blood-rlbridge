"""Tests for fixed-seat winner reporting in Talishar fast_step_index."""

from __future__ import annotations

from flesh_and_blood_rlbridge.talishar_engine_environment import TalisharEngineEnvironment


def _bare_env(*, acting_player_id: int) -> TalisharEngineEnvironment:
    env = object.__new__(TalisharEngineEnvironment)
    env._acting_player_id = acting_player_id
    env._player_hp = 0
    env._opp_hp = 0
    return env


def test_absolute_p1_seat_winner_when_p2_acting_and_p2_wins() -> None:
    env = _bare_env(acting_player_id=2)
    state = {"playerHealth": 12, "opponentHealth": 0}
    assert env._absolute_p1_seat_winner(state) == 1


def test_absolute_p1_seat_winner_when_p2_acting_and_p1_wins() -> None:
    env = _bare_env(acting_player_id=2)
    state = {"playerHealth": 0, "opponentHealth": 15}
    assert env._absolute_p1_seat_winner(state) == 0


def test_absolute_p1_seat_winner_when_p1_acting_and_p1_wins() -> None:
    env = _bare_env(acting_player_id=1)
    state = {"playerHealth": 18, "opponentHealth": 0}
    assert env._absolute_p1_seat_winner(state) == 0


def test_absolute_p1_seat_winner_exhausted_loss_for_acting_p1() -> None:
    env = _bare_env(acting_player_id=1)
    state = {"playerHealth": 10, "opponentHealth": 10}
    assert env._absolute_p1_seat_winner(state, exhausted_loss=True) == 1
