"""Tests for Talishar clone/sync bootstrap."""

from __future__ import annotations

from pathlib import Path

from fab_bridge.talishar_setup import BUNDLED_DECKS_DIR, sync_bundled_decks


def test_sync_bundled_decks_copies_txt_files(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "bundled"
    src.mkdir()
    (src / "fab_test_deck.txt").write_text("hero\n", encoding="utf-8")
    dest_assets = tmp_path / "Talishar" / "Assets"
    dest_assets.mkdir(parents=True)

    monkeypatch.setattr("fab_bridge.talishar_setup.BUNDLED_DECKS_DIR", src)
    monkeypatch.setattr(
        "fab_bridge.talishar_setup.talishar_assets_dir",
        lambda: dest_assets,
    )

    copied = sync_bundled_decks(quiet=True)
    assert copied == 1
    assert (dest_assets / "fab_test_deck.txt").read_text(encoding="utf-8") == "hero\n"


def test_bundled_decks_dir_has_fab_decks() -> None:
    assert BUNDLED_DECKS_DIR.is_dir()
    assert any(BUNDLED_DECKS_DIR.glob("fab_*.txt"))
