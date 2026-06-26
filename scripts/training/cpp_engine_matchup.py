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


def resolve_cpp_eval_deck_stems(
    assets_path: str,
    hero_id: str,
    opponent_deck: str,
) -> tuple[str, str]:
    """Map hero / opponent labels to Talishar precon stems for C++ engine lookup."""
    return resolve_cpp_lookup_decks(assets_path, hero_id, opponent_deck)


def _deck_input_hash(
    deck1: str,
    deck2: str,
    deck1_json: Path | None,
    deck2_json: Path | None,
) -> str:
    import importlib.util

    builder = _SCRIPTS_CPP / "build_cpp_engine_for_matchup.py"
    spec = importlib.util.spec_from_file_location("build_cpp_engine_for_matchup", builder)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load C++ engine builder from {builder}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.deck_input_hash(deck1, deck2, deck1_json, deck2_json)


def _discover_by_input_hash(
    lookup1: str,
    lookup2: str,
    *,
    deck1_json: Path | None,
    deck2_json: Path | None,
    base: Path,
) -> Path | None:
    _ensure_src_path()
    from flesh_and_blood_rlbridge.cpp_engine_environment import is_cpp_engine_available

    target = _deck_input_hash(lookup1, lookup2, deck1_json, deck2_json)
    exact = base / f"{lookup1}_vs_{lookup2}-{target}"
    if exact.is_dir() and is_cpp_engine_available(exact):
        return exact
    prefix = f"{lookup1}_vs_{lookup2}-"
    for candidate in sorted(
        base.glob(f"{prefix}*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        hash_file = candidate / "engine_input_hash.txt"
        if (
            candidate.is_dir()
            and hash_file.is_file()
            and hash_file.read_text(encoding="utf-8").strip() == target
            and is_cpp_engine_available(candidate)
        ):
            return candidate
    return None


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

    if deck1_json or deck2_json:
        hashed = _discover_by_input_hash(
            lookup1,
            lookup2,
            deck1_json=deck1_json,
            deck2_json=deck2_json,
            base=base,
        )
        if hashed is not None:
            return hashed
        return None

    engine_dir = get_engine_dir(lookup1, lookup2, cache_dir=base)
    if is_cpp_engine_available(engine_dir):
        return engine_dir

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
    lookup1, lookup2 = resolve_cpp_lookup_decks(assets_path, deck1, deck2)

    existing = discover_cpp_engine_dir(
        lookup1,
        lookup2,
        assets_path=assets_path,
        cache_dir=cache_dir,
    )
    if existing is not None:
        print(f"  [cpp] Reusing compiled engine: {existing}")
        return str(existing)

    if deck1_json or deck2_json:
        existing = discover_cpp_engine_dir(
            lookup1,
            lookup2,
            assets_path=assets_path,
            deck1_json=deck1_json,
            deck2_json=deck2_json,
            cache_dir=cache_dir,
        )
        if existing is not None:
            print(f"  [cpp] Reusing compiled engine (deck JSON hash): {existing}")
            return str(existing)

    if not build:
        return None

    print(
        f"\n  [cpp] No compiled engine for {lookup1} vs {lookup2} — building now..."
    )
    rc = build_cpp_engine(
        lookup1,
        lookup2,
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
        lookup1,
        lookup2,
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
