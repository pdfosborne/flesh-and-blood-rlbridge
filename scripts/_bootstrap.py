"""Configure ``sys.path`` for scripts living in categorized subdirectories."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"

_SUBDIRS = ("training", "eval", "cpp", "deck")


def configure_paths() -> Path:
    """Add repo root, script subdirectories, and ``src`` to ``sys.path``. Returns repo root."""
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    for sub in _SUBDIRS:
        path = SCRIPTS_ROOT / sub
        if path.is_dir():
            entry = str(path)
            if entry not in sys.path:
                sys.path.insert(0, entry)
    src = str(SRC_ROOT)
    if SRC_ROOT.is_dir() and src not in sys.path:
        sys.path.insert(0, src)
    return REPO_ROOT


def script_path(*parts: str) -> Path:
    """Resolve a path under ``scripts/``."""
    return SCRIPTS_ROOT.joinpath(*parts)
