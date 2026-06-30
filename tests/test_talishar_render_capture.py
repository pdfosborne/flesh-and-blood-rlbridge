"""Tests for Talishar FE capture URL/render helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.talishar_engine_environment import (
    TalisharEngineEnvironment,
    rewrite_frontend_api_url,
)


def test_frontend_game_url_includes_disable_card_hover_for_rgb_array() -> None:
    env = object.__new__(TalisharEngineEnvironment)
    env._game_name = "ABC123"
    env._self_play = True
    env._acting_player_id = 1
    env._auth_key = "secret"
    env._p1_auth_key = "secret"
    env._p2_auth_key = ""
    env._frontend_url = "http://localhost:5173"
    env._frontend_game_url_template = None
    env._frontend_player_id = None
    env._enable_frontend_card_hover = False
    env._render_mode = "rgb_array"

    url = env._frontend_game_url()
    assert url is not None
    assert "disableCardHover=1" in url
    assert "gameName=ABC123" in url


def test_frontend_game_url_omits_disable_card_hover_when_hover_enabled() -> None:
    env = object.__new__(TalisharEngineEnvironment)
    env._game_name = "ABC123"
    env._self_play = True
    env._acting_player_id = 1
    env._auth_key = "secret"
    env._p1_auth_key = "secret"
    env._p2_auth_key = ""
    env._frontend_url = "http://localhost:5173"
    env._frontend_game_url_template = None
    env._frontend_player_id = 1
    env._enable_frontend_card_hover = True
    env._render_mode = "rgb_array"

    url = env._frontend_game_url()
    assert url is not None
    assert "disableCardHover" not in url


def test_frontend_game_url_honors_fixed_frontend_player_id() -> None:
    env = object.__new__(TalisharEngineEnvironment)
    env._game_name = "ABC123"
    env._self_play = True
    env._acting_player_id = 2
    env._auth_key = "secret"
    env._p1_auth_key = "secret"
    env._p2_auth_key = "other"
    env._frontend_url = "http://localhost:5173"
    env._frontend_game_url_template = None
    env._frontend_player_id = 1
    env._enable_frontend_card_hover = False
    env._render_mode = "rgb_array"

    url = env._frontend_game_url()
    assert url is not None
    assert "playerID=1" in url
    assert "authKey=secret" in url


def test_frontend_game_url_uses_game_play_path_for_vite() -> None:
    env = object.__new__(TalisharEngineEnvironment)
    env._game_name = "ABC123"
    env._self_play = True
    env._acting_player_id = 1
    env._auth_key = "secret"
    env._p1_auth_key = "secret"
    env._p2_auth_key = ""
    env._frontend_url = "http://localhost:5173"
    env._frontend_game_url_template = None
    env._frontend_player_id = None
    env._enable_frontend_card_hover = False
    env._render_mode = "human"

    url = env._frontend_game_url()
    assert url is not None
    assert url.startswith("http://localhost:5173/game/play?")
    assert "gameName=ABC123" in url


def test_frontend_game_url_omits_disable_card_hover_without_rgb_array() -> None:
    env = object.__new__(TalisharEngineEnvironment)
    env._game_name = "ABC123"
    env._self_play = False
    env._acting_player_id = 1
    env._auth_key = "secret"
    env._p1_auth_key = "secret"
    env._p2_auth_key = ""
    env._frontend_url = "http://localhost:5173"
    env._frontend_game_url_template = None
    env._frontend_player_id = None
    env._enable_frontend_card_hover = False
    env._render_mode = None

    url = env._frontend_game_url()
    assert url is not None
    assert "disableCardHover" not in url


def test_rewrite_frontend_api_url_maps_vite_api_prefix() -> None:
    backend = "http://localhost:8091/game"
    assert (
        rewrite_frontend_api_url(
            "http://localhost:5173/api/GetNextTurn.php?gameName=ABC",
            backend,
        )
        == "http://localhost:8091/game/GetNextTurn.php?gameName=ABC"
    )


def test_rewrite_frontend_api_url_maps_apis_and_account_files() -> None:
    backend = "http://localhost:8091/game"
    assert (
        rewrite_frontend_api_url(
            "http://localhost:5173/APIs/CreateLocalGame.php",
            backend,
        )
        == "http://localhost:8091/game/APIs/CreateLocalGame.php"
    )
    assert (
        rewrite_frontend_api_url(
            "http://localhost:5173/AccountFiles/foo.json",
            backend,
        )
        == "http://localhost:8091/game/AccountFiles/foo.json"
    )


def test_rewrite_frontend_api_url_ignores_static_assets() -> None:
    assert (
        rewrite_frontend_api_url(
            "http://localhost:5173/src/main.tsx",
            "http://localhost:8091/game",
        )
        is None
    )
