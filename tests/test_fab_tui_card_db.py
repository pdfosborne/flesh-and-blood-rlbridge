"""Tests for TUI card database rescan helpers."""

from __future__ import annotations

from fab_tui import card_search as card_search_mod
from fab_tui.runner import run_card_db_rescan


def test_run_card_db_rescan_invokes_updater(monkeypatch, tmp_path) -> None:
    cards = tmp_path / "cards.json"
    decks = tmp_path / "fabrary_decks.json"
    cards.write_text("[]", encoding="utf-8")
    decks.write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_streaming(cmd, *, cwd=None, extra_env=None):
        captured["cmd"] = cmd
        captured["extra_env"] = extra_env
        return 0

    monkeypatch.setattr("fab_tui.runner.run_streaming", _fake_streaming)
    monkeypatch.setattr("fab_tui.runner.CARDS_DB_PATH", cards)
    monkeypatch.setattr("fab_tui.runner.FABRARY_DECKS_PATH", decks)
    monkeypatch.setattr(
        "fab_tui.runner.CARDS_DB_UPDATE_SCRIPT",
        tmp_path / "update_cards_db_from_fabtcg.py",
    )
    (tmp_path / "update_cards_db_from_fabtcg.py").write_text("", encoding="utf-8")

    rc = run_card_db_rescan(legality_scope="all")
    assert rc == 0
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "flesh_and_blood_rlbridge.card_db.update_cards_db_from_fabtcg" in cmd
    assert "--legality-scope" in cmd
    assert cmd[cmd.index("--legality-scope") + 1] == "all"
    extra_env = captured["extra_env"]
    assert isinstance(extra_env, dict)
    assert "PYTHONPATH" in extra_env


def test_clear_card_db_caches_clears_search_index() -> None:
    card_search_mod._load_index("silver_age")
    assert card_search_mod._load_index.cache_info().currsize >= 1
    card_search_mod.clear_card_db_caches()
    assert card_search_mod._load_index.cache_info().currsize == 0
