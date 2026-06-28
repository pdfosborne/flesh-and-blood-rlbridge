"""Clone upstream Talishar and sync rl-bridge deck overlays into Talishar/Assets."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from fab_bridge.paths import repo_root, talishar_assets_dir, talishar_dir

TALISHAR_UPSTREAM = "https://github.com/Talishar/Talishar.git"
BUNDLED_DECKS_DIR = repo_root() / "assets" / "talishar_decks"


def _talishar_present(path: Path) -> bool:
    return path.is_dir() and (path / "GameLogic.php").is_file()


def ensure_talishar(*, clone: bool = True, quiet: bool = False) -> Path:
    """Ensure Talishar/ exists (clone upstream when missing) and deck overlays are synced."""
    root = repo_root()
    td = talishar_dir()
    if not _talishar_present(td):
        if not clone:
            raise FileNotFoundError(f"Talishar not found: {td}")
        if not quiet:
            print(f"[setup] Cloning Talishar into {td}…")
        td.parent.mkdir(parents=True, exist_ok=True)
        if td.exists() and not _talishar_present(td):
            raise FileNotFoundError(
                f"{td} exists but is not a Talishar checkout (missing GameLogic.php)."
            )
        subprocess.run(  # noqa: S603
            ["git", "clone", "--depth", "1", TALISHAR_UPSTREAM, str(td)],
            cwd=root,
            check=True,
        )
    sync_bundled_decks(quiet=quiet)
    return td


def sync_bundled_decks(*, quiet: bool = False) -> int:
    """Copy assets/talishar_decks/*.txt into Talishar/Assets/. Returns files copied."""
    src_dir = BUNDLED_DECKS_DIR
    dest_dir = talishar_assets_dir()
    if not src_dir.is_dir():
        return 0
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(src_dir.glob("*.txt")):
        dest = dest_dir / src.name
        if not dest.exists() or src.read_bytes() != dest.read_bytes():
            shutil.copy2(src, dest)
            copied += 1
    if copied and not quiet:
        print(f"[setup] Synced {copied} deck file(s) to {dest_dir}")
    return copied


def ensure_talishar_or_exit(*, clone: bool = True) -> int:
    try:
        ensure_talishar(clone=clone)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: Talishar setup failed: {exc}", file=sys.stderr)
        print(
            "  Run: python -c \"from fab_bridge.talishar_setup import ensure_talishar; ensure_talishar()\"",
            file=sys.stderr,
        )
        return 1
    return 0
