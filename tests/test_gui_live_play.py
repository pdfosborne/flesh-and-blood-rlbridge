"""Tests for GUI embedded live play session API."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from fab_gui.api import (
    LIVE_PLAY,
    LivePlayRegistry,
    LivePlaySession,
    live_play_status,
    open_live_play_chromium,
    start_live_play,
    stop_live_play,
)
from fab_tui.config import EnvironmentSettings, browser_talishar_fe_url
from rl_agents.ppo import UNIFIED_AGENT_WEIGHT_VERSION


def test_live_play_registry_rejects_concurrent_session() -> None:
    registry = LivePlayRegistry()
    first = LivePlaySession(session_id="abc123")
    registry.add(first)
    second = LivePlaySession(session_id="def456")
    with pytest.raises(RuntimeError, match="already running"):
        registry.add(second)


def test_stop_live_play_sets_cancel_event() -> None:
    LIVE_PLAY._sessions.clear()
    LIVE_PLAY._active_id = None
    session = LivePlaySession(session_id="sess1", status="playing")
    LIVE_PLAY._sessions["sess1"] = session
    LIVE_PLAY._active_id = "sess1"

    result = stop_live_play("sess1")
    assert result["stopped"] is True
    assert session.cancel_event.is_set()


def test_live_play_status_inactive_when_none() -> None:
    assert live_play_status("missing") == {"active": False}


def test_browser_talishar_fe_url_maps_docker_host() -> None:
    assert browser_talishar_fe_url("http://talishar-fe:5173") == "http://127.0.0.1:5173"
    assert browser_talishar_fe_url("http://talishar-fe:5173", page_host="localhost") == (
        "http://localhost:5173"
    )
    assert browser_talishar_fe_url("http://localhost:5173", page_host="127.0.0.1") == (
        "http://localhost:5173"
    )


def test_open_live_play_chromium_requires_active_session() -> None:
    LIVE_PLAY._sessions.clear()
    LIVE_PLAY._active_id = None
    with pytest.raises(RuntimeError, match="No active"):
        open_live_play_chromium()


def test_open_live_play_chromium_calls_playwright(monkeypatch) -> None:
    LIVE_PLAY._sessions.clear()
    LIVE_PLAY._active_id = None
    session = LivePlaySession(
        session_id="sess1",
        status="playing",
        frontend_url="http://localhost:5173/game/play?gameName=1",
    )
    LIVE_PLAY._sessions["sess1"] = session
    LIVE_PLAY._active_id = "sess1"

    opened: list[str] = []

    def _fake_open(url: str) -> dict:
        opened.append(url)
        return {"opened": True, "url": url}

    monkeypatch.setattr("talishar_live_play.open_frontend_in_chromium", _fake_open)
    result = open_live_play_chromium()
    assert result["opened"] is True
    assert opened == [session.frontend_url]
    assert session.chromium_opened is True


def test_start_live_play_uses_internal_fe_url_for_probe(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    fmt_dir = cache_dir / "silver_age"
    fmt_dir.mkdir(parents=True)
    weights = fmt_dir / f"unified_agent_v{UNIFIED_AGENT_WEIGHT_VERSION}.json"
    weights.write_text("{}", encoding="utf-8")

    env = EnvironmentSettings(
        talishar_url="http://web-server/game",
        talishar_fe_url="http://talishar-fe:5173",
        assets_path=str(tmp_path / "Assets"),
    )
    (tmp_path / "Assets").mkdir()

    probed: list[str] = []
    passed_fe: list[str] = []

    monkeypatch.setattr(
        "talishar_live_play._verify_talishar_reachable",
        lambda _url: None,
    )
    monkeypatch.setattr(
        "talishar_live_play._verify_frontend_reachable",
        lambda url: probed.append(url),
    )
    monkeypatch.setattr(
        "fab_gui.api.sync_opponent_deck_api",
        lambda *_args, **_kwargs: None,
    )

    def _fake_run(**kwargs) -> dict:
        passed_fe.append(str(kwargs.get("fe_url", "")))
        return {
            "record": {"wins": 0, "losses": 0, "draws": 0, "timeouts": 0},
            "cancelled": True,
        }

    monkeypatch.setattr(
        "talishar_live_play.run_embedded_unified_live_play_session",
        _fake_run,
    )

    LIVE_PLAY._sessions.clear()
    LIVE_PLAY._active_id = None

    start_live_play(
        env,
        {
            "deck": {"a_red": 40},
            "game_format": "silver_age",
            "equipment_header": "aurora",
            "opponent_deck": "OppPrecon",
            "opponent_game_deck": {"b_red": 40},
            "cache_dir": str(cache_dir),
            "human_deck": "opponent",
            "prefer_chromium": False,
        },
        page_host="localhost",
    )

    assert probed == ["http://talishar-fe:5173"]
    assert passed_fe == ["http://talishar-fe:5173"]


def test_start_live_play_missing_unified_agent(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    env = EnvironmentSettings(
        talishar_url="http://localhost:8080/game",
        talishar_fe_url="http://localhost:5173",
        assets_path=str(tmp_path / "Assets"),
    )
    (tmp_path / "Assets").mkdir()

    monkeypatch.setattr(
        "talishar_live_play._verify_talishar_reachable",
        lambda _url: None,
    )
    monkeypatch.setattr(
        "talishar_live_play._verify_frontend_reachable",
        lambda _url: None,
    )

    body = {
        "deck": {"a_red": 40},
        "game_format": "silver_age",
        "equipment_header": "aurora",
        "opponent_deck": "OppPrecon",
        "opponent_game_deck": {"b_red": 40},
        "cache_dir": str(cache_dir),
    }
    with pytest.raises(FileNotFoundError, match="unified agent"):
        start_live_play(env, body)


def test_start_live_play_spawns_thread(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    fmt_dir = cache_dir / "silver_age"
    fmt_dir.mkdir(parents=True)
    weights = fmt_dir / f"unified_agent_v{UNIFIED_AGENT_WEIGHT_VERSION}.json"
    weights.write_text("{}", encoding="utf-8")

    env = EnvironmentSettings(
        talishar_url="http://localhost:8080/game",
        talishar_fe_url="http://localhost:5173",
        assets_path=str(tmp_path / "Assets"),
    )
    (tmp_path / "Assets").mkdir()

    monkeypatch.setattr(
        "talishar_live_play._verify_talishar_reachable",
        lambda _url: None,
    )
    monkeypatch.setattr(
        "talishar_live_play._verify_frontend_reachable",
        lambda _url: None,
    )
    monkeypatch.setattr(
        "fab_gui.api.sync_opponent_deck_api",
        lambda *_args, **_kwargs: None,
    )

    started = threading.Event()

    def _fake_run(**_kwargs) -> dict:
        started.set()
        return {
            "record": {"wins": 1, "losses": 0, "draws": 0, "timeouts": 0},
            "cancelled": False,
        }

    monkeypatch.setattr(
        "talishar_live_play.run_embedded_unified_live_play_session",
        _fake_run,
    )

    # Reset global registry between tests
    LIVE_PLAY._sessions.clear()
    LIVE_PLAY._active_id = None

    payload = start_live_play(
        env,
        {
            "deck": {"a_red": 40},
            "game_format": "silver_age",
            "equipment_header": "aurora",
            "opponent_deck": "OppPrecon",
            "opponent_game_deck": {"b_red": 40},
            "cache_dir": str(cache_dir),
            "human_deck": "opponent",
        },
    )
    assert payload["session_id"]
    assert started.wait(timeout=2.0)

    session = LIVE_PLAY.get(payload["session_id"])
    assert session is not None
    for _ in range(50):
        if session.status in {"finished", "failed", "cancelled"}:
            break
        threading.Event().wait(0.05)
    assert session.status == "finished"
