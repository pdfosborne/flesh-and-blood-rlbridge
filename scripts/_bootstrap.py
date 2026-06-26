"""Configure ``sys.path`` for scripts living in categorized subdirectories."""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from fab_bridge.paths import configure_import_paths as _configure_fab_paths
    from fab_bridge.paths import repo_root as _fab_repo_root
except ImportError:
    _configure_fab_paths = None
    _fab_repo_root = None

SCRIPTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"

_SUBDIRS = ("training", "eval", "cpp", "deck")


def _add_script_subdirs(scripts_root: Path) -> None:
    for sub in _SUBDIRS:
        path = scripts_root / sub
        if path.is_dir():
            entry = str(path)
            if entry not in sys.path:
                sys.path.insert(0, entry)


def configure_paths() -> Path:
    """Add repo root, script subdirectories, and ``src`` to ``sys.path``. Returns repo root."""
    if _configure_fab_paths is not None and _fab_repo_root is not None:
        repo = _fab_repo_root()
        _configure_fab_paths()
        scripts_root = repo / "scripts"
        if not scripts_root.is_dir():
            scripts_root = SCRIPTS_ROOT
    else:
        override = os.environ.get("FAB_BRIDGE_HOME", "").strip()
        if override:
            repo = Path(override).expanduser().resolve()
        else:
            repo = REPO_ROOT

        repo_str = str(repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        src = str(repo / "src")
        if (repo / "src").is_dir() and src not in sys.path:
            sys.path.insert(0, src)
        scripts_root = SCRIPTS_ROOT

    _add_script_subdirs(scripts_root)
    return repo


def script_path(*parts: str) -> Path:
    """Resolve a path under ``scripts/``."""
    return SCRIPTS_ROOT.joinpath(*parts)
