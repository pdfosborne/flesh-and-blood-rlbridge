"""Tests for final-eval HP aggregation and outcome summary helpers."""

from train_pipeline_common import sideboard_from_pool
from train_play import (
    _aggregate_hp_by_turn,
    _build_eval_outcome_summary,
    _build_matchup_deck_sheet_html,
    _parse_obs_hp,
    _track_turn_hp,
)


def test_parse_obs_hp_from_dict() -> None:
    turn, p1, p2 = _parse_obs_hp(
        {"turnNo": 3, "playerHealth": 18, "opponentHealth": 12}
    )
    assert turn == 3
    assert p1 == 18
    assert p2 == 12


def test_track_turn_hp_keeps_latest_snapshot() -> None:
    traj: dict[int, tuple[int, int]] = {}
    _track_turn_hp(traj, 0, 20, 20)
    _track_turn_hp(traj, 1, 19, 18)
    _track_turn_hp(traj, 1, 17, 16)
    assert traj == {1: (17, 16)}


def test_aggregate_hp_by_turn_mean_and_std() -> None:
    trajectories = [
        {1: (20, 20), 2: (18, 17)},
        {1: (20, 19), 2: (16, 15)},
    ]
    rows = _aggregate_hp_by_turn(trajectories)
    assert [r["turn"] for r in rows] == [1, 2]
    assert rows[0]["p1_hp_mean"] == 20.0
    assert rows[0]["p2_hp_mean"] == 19.5
    assert rows[1]["p1_hp_mean"] == 17.0
    assert rows[1]["p2_hp_mean"] == 16.0
    assert rows[0]["n"] == 2
    assert rows[1]["p1_hp_std"] == 1.0


def test_build_eval_outcome_summary_percentages() -> None:
    summary = _build_eval_outcome_summary(
        episodes=10,
        wins=6,
        losses=3,
        draws=1,
        episode_log=[
            {"steps": 40, "p1_hp": 10, "p2_hp": 0},
            {"steps": 50, "p1_hp": 0, "p2_hp": 8},
        ],
    )
    assert summary["win_pct"] == 60.0
    assert summary["loss_pct"] == 30.0
    assert summary["draw_pct"] == 10.0
    assert summary["avg_steps"] == 45.0
    assert summary["avg_final_player_hp"] == 5.0
    assert summary["avg_final_opponent_hp"] == 4.0


def test_sideboard_from_pool() -> None:
    pool = {"a_red": 3, "b_blue": 2, "c_yellow": 1}
    game = {"a_red": 2, "b_blue": 2}
    sb = sideboard_from_pool(pool, game)
    assert sb == {"a_red": 1, "c_yellow": 1}


def test_matchup_deck_sheet_html_includes_zones() -> None:
    export = {
        "hero_id": "aurora",
        "opponent_hero_id": "briar",
        "matchup": "aurora vs briar",
        "format": "sage",
        "pool_size": 55,
        "game_deck": {
            "total_cards": 2,
            "cards": [
                {
                    "id": "a_red",
                    "name": "Alpha",
                    "count": 1,
                    "pitch": 1,
                    "image_url": "http://localhost/WebpImages/a_red.webp",
                }
            ],
        },
        "sideboard": {
            "total_cards": 1,
            "cards": [
                {
                    "id": "b_blue",
                    "name": "Beta",
                    "count": 1,
                    "pitch": 3,
                    "image_url": "http://localhost/WebpImages/b_blue.webp",
                }
            ],
        },
        "equipment": [],
    }
    page = _build_matchup_deck_sheet_html(export, eval_summary={"win_pct": 60.0, "loss_pct": 35.0, "draw_pct": 5.0})
    assert "Game deck (2)" in page
    assert "Sideboard (1)" in page
    assert "a_red.webp" in page
    assert "Win 60.0%" in page
