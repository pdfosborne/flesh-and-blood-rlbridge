"""Tests for C++ eval live dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from fab_bridge.cpp_eval_live_dashboard import (
    CPP_EVAL_LIVE_DASHBOARD,
    CPP_EVAL_LIVE_STATE,
    UNIFIED_CHECKPOINT_EVAL_LIVE,
    collect_cpp_eval_live_state,
    cpp_eval_live_paths,
    eval_engine_from_env,
    render_cpp_eval_live_html,
    unified_checkpoint_eval_live_path,
    update_cpp_eval_live_replay,
    write_cpp_eval_live_dashboard,
    write_cpp_eval_live_state,
)


class _FakeCard:
    def __init__(self, card_id: str, name: str) -> None:
        self.card_id = card_id
        self.name = name
        self.cost = 1
        self.pitch = 2
        self.power = 3
        self.defense = 1


class _FakeGameState:
    turn_no = 2
    p1_health = 18
    p2_health = 16
    p1_hand = [_FakeCard("card_a", "Attack A")]
    p2_hand = [_FakeCard("card_b", "Attack B")]
    p1_deck_size = 30
    p2_deck_size = 28
    p1_pitch_size = 1
    p2_pitch_size = 0
    game_over = False
    winner = -1


class _FakeCppEnv:
    def __init__(self) -> None:
        self._gs = _FakeGameState()
        self._acting_player = 1
        self._deck1 = "HeroA"
        self._deck2 = "HeroB"
        self._events = [
            {
                "step": 1,
                "before": {
                    "acting_player_id": 1,
                    "turn_no": 2,
                    "phase": "m",
                    "player_health": 18,
                    "opponent_health": 16,
                },
                "action": {"label": "Attack A", "action_code": 27, "zone": "hand"},
                "action_class": "attack",
            }
        ]

    def live_display_snapshot(self) -> dict:
        gs = self._gs
        return {
            "engine": "cpp",
            "status": "active",
            "turn_no": gs.turn_no,
            "acting_player_id": self._acting_player,
            "phase": "m",
            "p1_health": gs.p1_health,
            "p2_health": gs.p2_health,
            "p1_hand": [{"card_id": "card_a", "name": "Attack A", "cost": 1, "pitch": 2, "power": 3}],
            "p2_hand": [{"card_id": "card_b", "name": "Attack B", "cost": 1, "pitch": 2, "power": 3}],
            "p1_deck_size": gs.p1_deck_size,
            "p2_deck_size": gs.p2_deck_size,
            "p1_pitch_size": gs.p1_pitch_size,
            "p2_pitch_size": gs.p2_pitch_size,
            "legal_actions": [
                {"label": "Attack A", "action_code": 27, "zone": "hand"},
                {"label": "Pass", "action_code": 99, "zone": "button"},
            ],
            "game_over": False,
            "winner": -1,
            "deck1": self._deck1,
            "deck2": self._deck2,
        }

    def get_combat_tracker_snapshot(self, **kwargs) -> dict:
        return {
            "enabled": True,
            "engine": "cpp",
            "recent_events": self._events,
            "recent_combat_log_lines": ["P1 Attack A (hand)", "HP P1 20->18 | P2 20->16"],
        }


class _FakeEnv:
    def __init__(self) -> None:
        self._cpp_env = _FakeCppEnv()

    def live_display_snapshot(self) -> dict:
        return self._cpp_env.live_display_snapshot()

    def get_combat_tracker_snapshot(self, **kwargs) -> dict:
        return self._cpp_env.get_combat_tracker_snapshot(**kwargs)


class _FakeTalisharFastEnv:
    def __init__(self) -> None:
        self._using_fast_talishar = True
        self._using_cpp = False
        self.talishar_backend = "fast"
        self._inner = _FakeCppEnv()

    def live_display_snapshot(self) -> dict:
        return self._inner.live_display_snapshot()

    def get_combat_tracker_snapshot(self, **kwargs) -> dict:
        return self._inner.get_combat_tracker_snapshot(**kwargs)


def test_cpp_eval_live_paths(tmp_path: Path) -> None:
    progress = tmp_path / "eval_live.json"
    state_path, html_path = cpp_eval_live_paths(progress)
    assert state_path.name == CPP_EVAL_LIVE_STATE
    assert html_path.name == CPP_EVAL_LIVE_DASHBOARD
    assert state_path.parent == tmp_path


def test_unified_checkpoint_eval_live_path(tmp_path: Path) -> None:
    path = unified_checkpoint_eval_live_path(tmp_path)
    assert path.name == UNIFIED_CHECKPOINT_EVAL_LIVE
    assert path.parent == tmp_path.resolve()


def test_collect_and_render_dashboard() -> None:
    env = _FakeEnv()
    state = collect_cpp_eval_live_state(
        env,
        episode=2,
        episodes_total=10,
        step=3,
        action_index=0,
        eval_label="agent_vs_agent",
        aggregate={"wins": 1, "losses": 0, "draws": 0, "timeouts": 0, "episodes_completed": 1},
    )
    assert state["episode"] == 2
    assert state["board"]["p1_health"] == 18
    assert state["chosen_action"]["label"] == "Attack A"
    assert state["engine"] == "cpp"
    assert state["engine_display"] == "C++ engine"

    html = render_cpp_eval_live_html(state, auto_refresh_seconds=2.0)
    assert "C++ engine eval" in html
    assert "Attack A" in html
    assert 'http-equiv="refresh" content="2"' in html
    assert "agent_vs_agent" in html


def test_collect_and_render_talishar_fast_dashboard() -> None:
    env = _FakeTalisharFastEnv()
    assert eval_engine_from_env(env) == ("talishar_fast", "Talishar fast")
    state = collect_cpp_eval_live_state(
        env,
        episode=1,
        episodes_total=5,
        step=2,
        eval_label="checkpoint_eval",
        aggregate={"wins": 0, "losses": 0, "draws": 0, "timeouts": 0, "episodes_completed": 0},
    )
    assert state["engine"] == "talishar_fast"
    html = render_cpp_eval_live_html(state, auto_refresh_seconds=None)
    assert "Talishar fast eval" in html
    assert "C++ engine eval" not in html


def test_write_and_update_roundtrip(tmp_path: Path) -> None:
    progress = tmp_path / "eval_live.json"
    env = _FakeEnv()
    state_path, html_path = update_cpp_eval_live_replay(
        progress,
        env,
        episode=1,
        episodes_total=5,
        step=1,
        action_index=0,
        eval_label="test",
        aggregate={"wins": 0, "losses": 0, "draws": 0, "timeouts": 0, "episodes_completed": 0},
        announce=False,
    )
    assert state_path.is_file()
    loaded = json.loads(state_path.read_text(encoding="utf-8"))
    assert loaded["step"] == 1
    assert html_path is not None
    assert html_path.is_file()
    assert "Recent steps" in html_path.read_text(encoding="utf-8")


def test_complete_state_stops_refresh() -> None:
    state = {
        "complete": True,
        "eval_label": "done",
        "episode": 5,
        "episodes_total": 5,
        "step": 12,
        "board": {},
        "aggregate": {},
    }
    html = render_cpp_eval_live_html(state, auto_refresh_seconds=1.0)
    assert 'http-equiv="refresh"' not in html
    assert "Complete" in html


def test_write_cpp_eval_live_dashboard_from_file(tmp_path: Path) -> None:
    state_path = tmp_path / CPP_EVAL_LIVE_STATE
    html_path = tmp_path / CPP_EVAL_LIVE_DASHBOARD
    write_cpp_eval_live_state(
        state_path,
        {
            "eval_label": "x",
            "episode": 1,
            "episodes_total": 1,
            "step": 0,
            "board": {"p1_health": 20, "p2_health": 20, "legal_actions": []},
            "aggregate": {},
            "complete": True,
        },
    )
    out = write_cpp_eval_live_dashboard(state_path, html_path)
    assert out == html_path
    assert html_path.is_file()
