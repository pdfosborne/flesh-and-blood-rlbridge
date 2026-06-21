"""Discover, build, and resolve C++ engine directories for deck matchups."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SCRIPTS_ROOT.parent
_SCRIPTS_CPP = _SCRIPTS_ROOT / "cpp"
_SCRIPTS_TRAINING = _SCRIPTS_ROOT / "training"
_DEFAULT_CACHE = _REPO_ROOT / "results" / "cpp_engines"


def _ensure_src_path() -> None:
    src = _REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def resolve_cpp_lookup_decks(
    assets_path: str,
    deck1: str,
    deck2: str,
) -> tuple[str, str]:
    """Return Talishar Assets stems used for C++ engine cache lookup."""
    _ensure_src_path()
    from flesh_and_blood_rlbridge.talishar_deck_assets import resolve_talishar_deck_stem

    return (
        resolve_talishar_deck_stem(assets_path, deck1),
        resolve_talishar_deck_stem(assets_path, deck2),
    )


def discover_cpp_engine_dir(
    deck1: str,
    deck2: str,
    *,
    assets_path: str,
    deck1_json: Path | None = None,
    deck2_json: Path | None = None,
    cache_dir: Path | None = None,
) -> Path | None:
    """Return a compiled engine directory for this matchup, if one exists."""
    _ensure_src_path()
    from flesh_and_blood_rlbridge.cpp_engine_environment import (
        get_engine_dir,
        is_cpp_engine_available,
    )

    lookup1, lookup2 = resolve_cpp_lookup_decks(assets_path, deck1, deck2)
    base = cache_dir or _DEFAULT_CACHE

    engine_dir = get_engine_dir(lookup1, lookup2, cache_dir=base)
    if is_cpp_engine_available(engine_dir):
        return engine_dir

    if deck1_json or deck2_json:
        prefix = f"{lookup1}_vs_{lookup2}-"
        if base.is_dir():
            for candidate in sorted(
                base.glob(f"{prefix}*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                if candidate.is_dir() and is_cpp_engine_available(candidate):
                    return candidate
    return None


def build_cpp_engine(
    deck1: str,
    deck2: str,
    *,
    assets_path: str,
    talishar_url: str,
    deck1_json: Path | None = None,
    deck2_json: Path | None = None,
    cache_dir: Path | None = None,
) -> int:
    """Run ``build_cpp_engine_for_matchup.py`` for *deck1* vs *deck2*."""
    lookup1, lookup2 = resolve_cpp_lookup_decks(assets_path, deck1, deck2)
    cmd = [
        sys.executable,
        str(_SCRIPTS_CPP / "build_cpp_engine_for_matchup.py"),
        "--deck1",
        lookup1,
        "--deck2",
        lookup2,
        "--talishar-src",
        str(_REPO_ROOT / "Talishar"),
        "--talishar-url",
        talishar_url,
        "--no-server",
    ]
    if cache_dir is not None:
        cmd.extend(["--cache-dir", str(cache_dir)])
    if deck1_json and deck1_json.is_file():
        cmd.extend(["--deck1-json", str(deck1_json)])
    if deck2_json and deck2_json.is_file():
        cmd.extend(["--deck2-json", str(deck2_json)])
    return subprocess.call(cmd, cwd=str(_REPO_ROOT))


def ensure_cpp_engine_for_matchup(
    deck1: str,
    deck2: str,
    *,
    assets_path: str,
    talishar_url: str,
    deck1_json: Path | None = None,
    deck2_json: Path | None = None,
    build: bool = True,
    cache_dir: Path | None = None,
) -> Optional[str]:
    """Return a usable C++ engine directory, building one first when *build* is True."""
    existing = discover_cpp_engine_dir(
        deck1,
        deck2,
        assets_path=assets_path,
        deck1_json=deck1_json,
        deck2_json=deck2_json,
        cache_dir=cache_dir,
    )
    if existing is not None:
        return str(existing)

    if not build:
        return None

    lookup1, lookup2 = resolve_cpp_lookup_decks(assets_path, deck1, deck2)
    print(
        f"\n  [cpp] No compiled engine for {lookup1} vs {lookup2} — building now..."
    )
    rc = build_cpp_engine(
        deck1,
        deck2,
        assets_path=assets_path,
        talishar_url=talishar_url,
        deck1_json=deck1_json,
        deck2_json=deck2_json,
        cache_dir=cache_dir,
    )
    if rc != 0:
        print(f"  [cpp] WARNING: engine build failed (exit {rc}) — HTTP Talishar fallback")
        return None

    built = discover_cpp_engine_dir(
        deck1,
        deck2,
        assets_path=assets_path,
        deck1_json=deck1_json,
        deck2_json=deck2_json,
        cache_dir=cache_dir,
    )
    if built is None:
        print("  [cpp] WARNING: build succeeded but engine directory not found")
        return None

    print(f"  [cpp] Engine ready: {built}")
    return str(built)
