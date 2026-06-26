"""Resolve the FAB RL Bridge application root at runtime."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_ROOT_MARKERS = ("scripts", "runscripts", "Talishar")


def _looks_like_repo_root(path: Path) -> bool:
    return all((path / marker).exists() for marker in _ROOT_MARKERS)


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return the repository / install root containing scripts and Talishar."""
    override = os.environ.get("FAB_BRIDGE_HOME", "").strip()
    if override:
        root = Path(override).expanduser().resolve()
        if not _looks_like_repo_root(root):
            msg = (
                f"FAB_BRIDGE_HOME={root} is missing expected directories "
                f"({', '.join(_ROOT_MARKERS)})."
            )
            raise RuntimeError(msg)
        return root

    # Editable install / dev checkout: fab_bridge lives at <root>/fab_bridge/
    from_pkg = Path(__file__).resolve().parent.parent
    if _looks_like_repo_root(from_pkg):
        return from_pkg

    # Wheel install: bundled assets may live beside the package in site-packages.
    from_site = from_pkg.parent
    if _looks_like_repo_root(from_site):
        return from_site

    # Legacy layout before fab_bridge existed (fab_tui at repo root).
    legacy = Path(__file__).resolve().parents[2]
    if _looks_like_repo_root(legacy):
        return legacy

    cwd = Path.cwd().resolve()
    if _looks_like_repo_root(cwd):
        return cwd

    # Best-effort fallback for partial checkouts (e.g. Docker image with Assets only).
    return from_pkg


def talishar_dir() -> Path:
    return repo_root() / "Talishar"


def talishar_assets_dir() -> Path:
    custom = os.environ.get("TALISHAR_ASSETS_PATH", "").strip()
    if custom:
        return Path(custom).expanduser().resolve()
    return talishar_dir() / "Assets"


def src_dir() -> Path:
    return repo_root() / "src"


def prepend_sys_path(entry: str | Path) -> None:
    """Move *entry* to the front of ``sys.path`` (idempotent)."""
    import sys

    entry_str = str(entry)
    while entry_str in sys.path:
        sys.path.remove(entry_str)
    sys.path.insert(0, entry_str)


def configure_import_paths() -> Path:
    """Ensure repo root and ``src`` are prepended to ``sys.path`` (idempotent)."""
    root = str(repo_root())
    src = str(src_dir())
    # Insert src first so repo root ends up at index 0 (runtime_defaults, fab_tui, …).
    for entry in (src, root):
        prepend_sys_path(entry)
    return repo_root()
