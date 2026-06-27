"""Tests for add_custom_decks_to_pool script."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "deck"))

from add_custom_decks_to_pool import main  # noqa: E402


def test_add_custom_decks_dry_run_with_cli_link() -> None:
    rc = main(["01KR40W4Z2ZS9EQPT6VT6CDSPE", "--dry-run"])
    assert rc == 0
