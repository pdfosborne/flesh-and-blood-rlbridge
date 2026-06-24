"""Tests for fab_bridge path resolution."""

from __future__ import annotations

from pathlib import Path

from fab_bridge.paths import repo_root, talishar_assets_dir, talishar_dir


def test_repo_root_has_expected_layout() -> None:
    root = repo_root()
    assert (root / "scripts").is_dir()
    assert (root / "runscripts").is_dir()
    assert (root / "fab_bridge").is_dir()


def test_talishar_paths_under_repo() -> None:
    root = repo_root()
    assert talishar_dir() == root / "Talishar"
    assert talishar_assets_dir().name == "Assets"
