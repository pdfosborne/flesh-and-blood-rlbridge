"""Tests for SAGE precon listing and main.py sage CLI helpers."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from fab_tui.sage_picker import list_sage_precon_options


def test_list_sage_precon_options_from_assets(tmp_path: Path) -> None:
    assets = tmp_path / "Assets"
    assets.mkdir()
    (assets / "KayoSAGEPrecon.txt").write_text(
        "kayo equip\nbare_fangs_red\n",
        encoding="utf-8",
    )
    (assets / "BriarSAGEPrecon.txt").write_text(
        "briar equip\nemboss_red\n",
        encoding="utf-8",
    )

    options = list_sage_precon_options(assets)
    heroes = {opt.hero_slug for opt in options}
    assert "kayo" in heroes
    assert "briar" in heroes
    assert all((assets / f"{opt.deck_name}.txt").is_file() for opt in options)


def test_list_sage_precon_skips_missing_assets(tmp_path: Path) -> None:
    assets = tmp_path / "Assets"
    assets.mkdir()
    options = list_sage_precon_options(assets)
    assert options == []
