"""Tests for eval/render anti-stuck diagnostic logging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fab_bridge import anti_stuck_logging as stuck


@pytest.fixture(autouse=True)
def _reset_stuck_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAB_ANTI_STUCK_LOGGING", raising=False)
    stuck._ENABLED = False  # noqa: SLF001
    stuck._RUN_DIR = None  # noqa: SLF001
    stuck._LOG_PATH = None  # noqa: SLF001


class _MockEnv:
    def __init__(
        self,
        *,
        server_report: dict | None = None,
        game_over: bool = False,
        legal_count: int = 1,
    ) -> None:
        self._server_report = server_report or {
            "turn_no": 1,
            "phase": "M",
            "player_health": 20,
            "opponent_health": 20,
            "legal_count": legal_count,
            "combat_log": ["Player 1 passes."],
            "board_fingerprint": "turn1|m|40|hand",
            "game_over": game_over,
            "gamestate_revert": False,
            "legal_actions": [
                {"action_code": 99, "label": "Pass", "zone": "button"},
            ],
        }

    def get_server_report(self) -> dict:
        return dict(self._server_report)

    def get_combat_tracker_snapshot(self, **_kwargs: object) -> dict:
        return {"enabled": True, "steps_recorded": 1}


def _pass_obs() -> dict:
    return {
        "turnNo": 1,
        "turnPhase": "M",
        "actingPlayerID": 1,
        "playerHealth": 20,
        "opponentHealth": 20,
        "legal_actions": [
            {"action_code": 99, "label": "Pass", "zone": "button"},
        ],
    }


def test_configure_writes_jsonl(tmp_path: Path) -> None:
    log_path = stuck.configure(run_dir=tmp_path, enabled=True)
    assert log_path == tmp_path / "anti_stuck_reports.jsonl"
    assert log_path.is_file()
    stuck.log_event("anti_stuck", "hello", shard="localhost:8081")
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
    record = json.loads(lines[-1])
    assert record["category"] == "anti_stuck"
    assert record["details"]["shard"] == "localhost:8081"


def test_disabled_does_not_write(tmp_path: Path) -> None:
    assert stuck.configure(run_dir=tmp_path, enabled=False) is None
    stuck.log_event("anti_stuck", "ignored")
    assert not (tmp_path / "anti_stuck_reports.jsonl").exists()


def test_read_from_manifest(tmp_path: Path) -> None:
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"anti_stuck_logging": True}),
        encoding="utf-8",
    )
    assert stuck.read_from_manifest(tmp_path) is True


def test_pass_loop_logs_once_at_threshold(tmp_path: Path) -> None:
    stuck.configure(run_dir=tmp_path, enabled=True)
    config = stuck.AntiStuckConfig(pass_streak=3, no_progress_steps=99, repeat_streak=99)
    monitor = stuck.AntiStuckMonitor(config=config)
    monitor.begin_episode(episode=1, mode="eval", p1_deck="a", p2_deck="b", base_url="http://x")
    env = _MockEnv()

    for _ in range(3):
        monitor.observe_step(
            env,
            obs_data=_pass_obs(),
            step_info={},
            action=0,
        )

    lines = (tmp_path / "anti_stuck_reports.jsonl").read_text(encoding="utf-8").strip().splitlines()
    detections = [
        json.loads(line)
        for line in lines
        if json.loads(line).get("message") == "pass_loop detected"
    ]
    assert len(detections) == 1
    assert detections[0]["details"]["stuck_kind"] == "pass_loop"

    monitor.observe_step(
        env,
        obs_data=_pass_obs(),
        step_info={},
        action=0,
    )
    lines_after = (tmp_path / "anti_stuck_reports.jsonl").read_text(encoding="utf-8").strip().splitlines()
    detections_after = [
        json.loads(line)
        for line in lines_after
        if json.loads(line).get("message") == "pass_loop detected"
    ]
    assert len(detections_after) == 1


def test_no_legal_actions_detection(tmp_path: Path) -> None:
    stuck.configure(run_dir=tmp_path, enabled=True)
    monitor = stuck.AntiStuckMonitor(
        config=stuck.AntiStuckConfig(pass_streak=99, no_progress_steps=99, repeat_streak=99),
    )
    monitor.begin_episode(episode=2, mode="render", p1_deck="a", p2_deck="b", base_url="http://x")
    env = _MockEnv(legal_count=0)
    monitor.observe_step(
        env,
        obs_data={"turnNo": 1, "turnPhase": "M", "legal_actions": []},
        step_info={},
        action=0,
        terminated=False,
        truncated=False,
    )
    lines = (tmp_path / "anti_stuck_reports.jsonl").read_text(encoding="utf-8").strip().splitlines()
    kinds = [
        json.loads(line)["details"]["stuck_kind"]
        for line in lines
        if json.loads(line).get("message") == "no_legal_actions detected"
    ]
    assert kinds == ["no_legal_actions"]


def test_action_loop_on_repeat_streak(tmp_path: Path) -> None:
    stuck.configure(run_dir=tmp_path, enabled=True)
    monitor = stuck.AntiStuckMonitor(
        config=stuck.AntiStuckConfig(pass_streak=99, no_progress_steps=99, repeat_streak=2),
    )
    monitor.begin_episode(episode=3, mode="eval", p1_deck="a", p2_deck="b", base_url="http://x")
    env = _MockEnv()
    obs = {
        "turnNo": 1,
        "turnPhase": "M",
        "legal_actions": [
            {"action_code": 27, "label": "Pitch", "zone": "hand"},
        ],
    }
    monitor.observe_step(env, obs_data=obs, step_info={"repeat_streak": 2}, action=0)
    lines = (tmp_path / "anti_stuck_reports.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert any(
        json.loads(line).get("message") == "action_loop detected"
        for line in lines
    )


def test_finish_episode_summary_when_still_stuck(tmp_path: Path) -> None:
    stuck.configure(run_dir=tmp_path, enabled=True)
    monitor = stuck.AntiStuckMonitor(
        config=stuck.AntiStuckConfig(pass_streak=1, no_progress_steps=99, repeat_streak=99),
    )
    monitor.begin_episode(episode=4, mode="eval", p1_deck="a", p2_deck="b", base_url="http://x")
    env = _MockEnv()
    monitor.observe_step(env, obs_data=_pass_obs(), step_info={}, action=0)
    monitor.finish_episode("timeout")
    lines = (tmp_path / "anti_stuck_reports.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert any(
        json.loads(line).get("message") == "episode_stuck_summary"
        for line in lines
    )


def test_macro_stall_detection_from_step_info(tmp_path: Path) -> None:
    stuck.configure(run_dir=tmp_path, enabled=True)
    monitor = stuck.AntiStuckMonitor(
        config=stuck.AntiStuckConfig(pass_streak=99, no_progress_steps=99, repeat_streak=99),
    )
    monitor.begin_episode(episode=5, mode="render", p1_deck="a", p2_deck="b", base_url="http://x")
    env = _MockEnv()
    monitor.observe_step(
        env,
        obs_data=_pass_obs(),
        step_info={
            "macro_stall_truncated": True,
            "macro_stall_reason": "no_damage_turns",
            "turns_without_damage": 6,
            "pass_only_main_streak": 0,
        },
        action=0,
        truncated=True,
    )
    lines = (tmp_path / "anti_stuck_reports.jsonl").read_text(encoding="utf-8").strip().splitlines()
    macro_lines = [
        json.loads(line)
        for line in lines
        if json.loads(line).get("message") == "macro_stall detected"
    ]
    assert len(macro_lines) == 1
    assert macro_lines[0]["details"]["reason"] == "no_damage_turns"
    monitor.finish_episode("stall_timeout")
    all_lines = (tmp_path / "anti_stuck_reports.jsonl").read_text(encoding="utf-8").strip().splitlines()
    summaries = [
        json.loads(line)
        for line in all_lines
        if json.loads(line).get("message") == "episode_stuck_summary"
    ]
    assert summaries
    assert "macro_stall" in summaries[-1]["details"]["active_kinds"]


def _choosetop_obs_filtered() -> dict:
    """Observation after filter_legal_actions strips Pass in CHOOSETOP."""
    return {
        "turnNo": 2,
        "turnPhase": "CHOOSETOP",
        "actingPlayerID": 1,
        "playerHealth": 12,
        "opponentHealth": 14,
        "legal_actions": [
            {
                "action_code": 8,
                "label": "Top",
                "zone": "popup",
                "button_input": "widowmaker_yellow",
            },
        ],
    }


def test_choosetop_filtered_obs_does_not_trigger_pass_loop(tmp_path: Path) -> None:
    """When Pass is filtered out, repeated Top picks must not log pass_loop."""
    stuck.configure(run_dir=tmp_path, enabled=True)
    monitor = stuck.AntiStuckMonitor(
        config=stuck.AntiStuckConfig(pass_streak=3, no_progress_steps=99, repeat_streak=99),
    )
    monitor.begin_episode(
        episode=5,
        mode="render",
        p1_deck="azalea",
        p2_deck="dash",
        base_url="http://x",
    )
    env = _MockEnv(
        server_report={
            "turn_no": 2,
            "phase": "CHOOSETOP",
            "player_health": 12,
            "opponent_health": 14,
            "legal_count": 1,
            "combat_log": ["Player 1 put a card on top of the deck"],
            "board_fingerprint": "turn2|choosetop|26|hand",
            "game_over": False,
            "gamestate_revert": False,
            "legal_actions": [
                {"action_code": 8, "label": "Top", "zone": "popup"},
            ],
        },
        legal_count=1,
    )
    obs = _choosetop_obs_filtered()

    for _ in range(4):
        monitor.observe_step(
            env,
            obs_data=obs,
            step_info={"loop_guard_forced_pass": False},
            action=0,
        )

    lines = (tmp_path / "anti_stuck_reports.jsonl").read_text(encoding="utf-8").strip().splitlines()
    pass_loops = [
        json.loads(line)
        for line in lines
        if json.loads(line).get("message") == "pass_loop detected"
    ]
    assert pass_loops == []
