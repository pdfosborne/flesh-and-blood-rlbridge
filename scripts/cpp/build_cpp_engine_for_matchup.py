#!/usr/bin/env python3
"""Generate and build a C++ FAB engine for a specific deck matchup.

Given two Talishar Assets deck names this script:

  1. Generates C++ source via generate_cpp_engine.py
  2. Checks/installs pybind11
  3. Configures and builds the CMake project
  4. Verifies the compiled fab_engine module
  5. Prints a usage summary

The compiled engine is cached in:
    results/cpp_engines/<Deck1>_vs_<Deck2>-<hash>/

On the next training run, TalisharEngineEnvironment auto-detects and uses the
cached engine (no code changes required).

Examples:
    python scripts/cpp/build_cpp_engine_for_matchup.py --deck1 Ira --deck2 Ira
    python scripts/cpp/build_cpp_engine_for_matchup.py --deck1 BriarSAGEPrecon --deck2 DorintheSAGEPrecon
    python scripts/cpp/build_cpp_engine_for_matchup.py --pipeline-json results/full_pipeline/results.json --no-server
    python scripts/cpp/build_cpp_engine_for_matchup.py --deck1 Ira --deck2 Ira --no-build
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATE_SCRIPT = Path(__file__).resolve().parent / "generate_cpp_engine.py"
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "cpp_engines"
DEFAULT_TALISHAR_SRC = REPO_ROOT / "Talishar"
MODULE_EXTENSIONS = {".pyd", ".so"}

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
from flesh_and_blood_rlbridge.cpp_engine_environment import (  # noqa: E402
    _find_engine_module,
    expected_fab_engine_module_name,
    is_cpp_engine_available,
    load_fab_engine,
)


def _header(text: str) -> None:
    print()
    print("=" * 62)
    print(f"  {text}")
    print("=" * 62)


def _step(number: str, text: str) -> None:
    print()
    print(f"Step {number} : {text}")


def _ok(text: str) -> None:
    print(f"  OK   {text}")


def _warn(text: str) -> None:
    print(f"  WARN {text}")


def _fail(text: str) -> None:
    print(f"  FAIL {text}")


def _run(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    if not capture:
        print(f"  $ {' '.join(cmd)}")
    completed = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
        capture_output=capture,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")
    return completed


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deck_input_hash(
    deck1: str,
    deck2: str,
    deck1_json: Path | None,
    deck2_json: Path | None,
) -> str:
    json_hash1 = _file_sha256(deck1_json) if deck1_json and deck1_json.is_file() else "none"
    json_hash2 = _file_sha256(deck2_json) if deck2_json and deck2_json.is_file() else "none"
    generator_hash = _file_sha256(GENERATE_SCRIPT) if GENERATE_SCRIPT.is_file() else "none"
    combined = f"{deck1.lower()}|{deck2.lower()}|{json_hash1}|{json_hash2}|gen:{generator_hash}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def deck_names_from_pipeline(path: Path) -> tuple[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    deck1 = ""
    deck2 = ""
    p1 = data.get("p1") if isinstance(data.get("p1"), dict) else {}
    p2 = data.get("p2") if isinstance(data.get("p2"), dict) else {}
    if p1.get("deck_asset_name"):
        deck1 = str(p1["deck_asset_name"])
    if p2.get("deck_asset_name"):
        deck2 = str(p2["deck_asset_name"])
    if not deck1 and data.get("hero_id"):
        deck1 = str(data["hero_id"])
    if not deck2 and data.get("opponent_deck_name"):
        deck2 = str(data["opponent_deck_name"])
    return deck1, deck2


def find_compiled_module(engine_dir: Path) -> Path | None:
    return _find_engine_module(engine_dir)


def cached_engine_up_to_date(engine_dir: Path, input_hash: str) -> Path | None:
    module = find_compiled_module(engine_dir)
    hash_file = engine_dir / "engine_input_hash.txt"
    if module is None or not hash_file.is_file():
        stale = sorted(
            {
                p.name
                for p in engine_dir.glob("fab_engine*")
                if p.is_file() and p.suffix in MODULE_EXTENSIONS
            }
        )
        if stale:
            print()
            print(
                f"  No fab_engine module for Python {sys.version_info.major}.{sys.version_info.minor} "
                f"(expected {expected_fab_engine_module_name()})."
            )
            print(f"  Stale module(s) present: {', '.join(stale)} — rebuilding.")
        return None
    stored_hash = hash_file.read_text(encoding="utf-8").strip()
    if stored_hash != input_hash:
        print()
        print(f"  Deck inputs changed (old={stored_hash}  new={input_hash}) -- rebuilding.")
        return None
    if not is_cpp_engine_available(engine_dir):
        print()
        print("  Cached fab_engine module cannot be imported for this Python — rebuilding.")
        return None
    return module


def resolve_base_url(talishar_url: str | None) -> str:
    if talishar_url:
        return talishar_url
    return os.environ.get("TALISHAR_URL", "http://localhost")


def find_cmake() -> str | None:
    cmake = shutil.which("cmake")
    if cmake:
        return cmake
    candidates = [
        Path(r"C:\Program Files\CMake\bin\cmake.exe"),
        Path(r"C:\Program Files (x86)\CMake\bin\cmake.exe"),
        Path(os.environ.get("ProgramFiles", "")) / "CMake" / "bin" / "cmake.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            os.environ["PATH"] = f"{candidate.parent}{os.pathsep}{os.environ.get('PATH', '')}"
            found = shutil.which("cmake")
            if found:
                _ok(f"Found cmake at: {candidate}")
                return found
    return None


def python_module_output(*args: str) -> tuple[int, str]:
    completed = subprocess.run(
        [sys.executable, *args],
        check=False,
        text=True,
        capture_output=True,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode, output.strip()


def ensure_pybind11() -> None:
    rc, version = python_module_output("-m", "pybind11", "--version")
    if rc == 0:
        _ok(f"pybind11 already installed: {version}")
        return
    _warn("pybind11 not found - installing")
    _run([sys.executable, "-m", "pip", "install", "pybind11", "--quiet"])
    _ok("pybind11 installed")


def pybind11_cmake_dir() -> str:
    rc, output = python_module_output("-m", "pybind11", "--cmakedir")
    if rc == 0 and output:
        return output.splitlines()[-1].strip()
    _warn("pybind11 cmake dir lookup failed - installing")
    _run([sys.executable, "-m", "pip", "install", "pybind11", "--quiet"])
    rc, output = python_module_output("-m", "pybind11", "--cmakedir")
    if rc != 0 or not output:
        raise RuntimeError(f"pybind11 --cmakedir failed: {output}")
    return output.splitlines()[-1].strip()


def _which_in_env(name: str, env: dict[str, str]) -> str | None:
    path = env.get("PATH", "")
    if path:
        return shutil.which(name, path=path)
    return shutil.which(name)


def _parse_cmd_set_output(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key:
            parsed[key] = value
    return parsed


def _windows_msvc_build_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Load vcvars64 so CMake can compile with MSVC outside a Developer shell."""
    env = dict(base or os.environ)
    if sys.platform != "win32":
        return env

    vswhere_candidates = [
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe",
    ]
    for vswhere in vswhere_candidates:
        if not vswhere.is_file():
            continue
        completed = subprocess.run(
            [str(vswhere), "-latest", "-property", "installationPath"],
            check=False,
            text=True,
            capture_output=True,
        )
        install_path = (completed.stdout or "").strip()
        if not install_path:
            continue
        vcvars = Path(install_path) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if not vcvars.is_file():
            continue
        boot = subprocess.run(
            ["cmd", "/c", f'call "{vcvars}" >nul 2>&1 && set'],
            check=False,
            text=True,
            capture_output=True,
            env=env,
        )
        if boot.returncode == 0 and boot.stdout:
            env.update(_parse_cmd_set_output(boot.stdout))
            if _which_in_env("cl", env):
                _ok("MSVC environment loaded (vcvars64)")
                return env
    return env


def cmake_generator_args(
    engine_dir: Path,
    pybind11_dir: str,
    *,
    env: dict[str, str] | None = None,
) -> list[str]:
    build_env = env or os.environ
    if _which_in_env("ninja", build_env) and _which_in_env("cl", build_env):
        _ok("Generator: Ninja (MSVC)")
        return ["-G", "Ninja"]
    if shutil.which("ninja"):
        _ok("Generator: Ninja")
        return ["-G", "Ninja"]
    if sys.platform == "win32":
        if _which_in_env("cl", build_env):
            _ok("Generator: auto (cl.exe found)")
            return []
        vswhere_candidates = [
            Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe",
            Path(os.environ.get("ProgramFiles", "")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe",
        ]
        for vswhere in vswhere_candidates:
            if not vswhere.is_file():
                continue
            completed = subprocess.run(
                [str(vswhere), "-latest", "-property", "installationVersion"],
                check=False,
                text=True,
                capture_output=True,
            )
            version = (completed.stdout or "").strip().split(".")[0]
            generator = {
                "17": "Visual Studio 17 2022",
                "16": "Visual Studio 16 2019",
                "15": "Visual Studio 15 2017",
            }.get(version, "Visual Studio 17 2022")
            _ok(f"Generator: {generator} (via vswhere)")
            return ["-G", generator, "-A", "x64"]
        if shutil.which("g++"):
            _ok("Generator: MinGW Makefiles")
            return ["-G", "MinGW Makefiles"]
    elif shutil.which("g++") or shutil.which("gcc"):
        _ok("Generator: Unix Makefiles")
        return ["-G", "Unix Makefiles"]

    _warn("No compiler found in PATH.")
    print()
    print("  Options:")
    if sys.platform == "win32":
        print("  1. Run this script from a VS Developer PowerShell:")
        print("     Start > Visual Studio > Developer PowerShell for VS")
        print("  2. Install Ninja + MSVC Build Tools, then re-run")
        print("  3. Install MinGW (g++) and re-run")
    else:
        print("  1. Install build-essential (g++) and cmake, then re-run")
        print("  2. Install ninja-build for faster builds, then re-run")
    print()
    print("  Build manually once a compiler is available:")
    print(f'    cd "{engine_dir}"')
    print(f'    cmake -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="{pybind11_dir}" .')
    print("    cmake --build build --config Release")
    raise SystemExit(1)


def print_card_manifest_summary(engine_dir: Path) -> None:
    manifest_path = engine_dir / "card_manifest.json"
    if not manifest_path.is_file():
        _warn("card_manifest.json not found - skipping card count summary")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cards = manifest.get("cards") or {}
    total = len(cards)
    with_php = sum(1 for card in cards.values() if card.get("php_found"))
    no_php = total - with_php
    print()
    print(f"  Card stubs : {total} total")
    print(f"  PHP logic  : {with_php} (ready to translate)")
    if no_php > 0:
        print(f"  No PHP     : {no_php} (manual implementation needed)")
    else:
        print("  No PHP     : 0")
    print()
    print(f"  cards.h : {engine_dir / 'cards.h'}")
    print("  Each stub has PHP logic as comments. Remove throw once implemented.")


def generate_cpp_source(
    *,
    deck1: str,
    deck2: str,
    talishar_src: Path,
    engine_dir: Path,
    base_url: str,
    no_server: bool,
    deck1_json: Path | None,
    deck2_json: Path | None,
) -> None:
    _step("1", "Generating C++ source")
    cmd = [
        sys.executable,
        str(GENERATE_SCRIPT),
        "--deck1",
        deck1,
        "--deck2",
        deck2,
        "--talishar-src",
        str(talishar_src),
        "--out",
        str(engine_dir),
    ]
    if no_server:
        cmd.append("--no-server")
    if base_url != "http://localhost":
        cmd.extend(["--base-url", base_url])
    if deck1_json and deck1_json.is_file():
        cmd.extend(["--deck1-json", str(deck1_json)])
    if deck2_json and deck2_json.is_file():
        cmd.extend(["--deck2-json", str(deck2_json)])
    _run(cmd)
    _ok(f"C++ source written to {engine_dir}")


def build_engine(engine_dir: Path, input_hash: str, pybind11_dir: str) -> None:
    _step("4", "Building C++ engine")
    cmake = find_cmake()
    if cmake is None:
        _fail("cmake not found. Install from https://cmake.org/download/")
        raise SystemExit(1)
    version = subprocess.run(
        [cmake, "--version"],
        check=False,
        text=True,
        capture_output=True,
    )
    _ok((version.stdout or version.stderr or "").splitlines()[0])

    print("  Locating pybind11 cmake dir...")
    _ok(f"pybind11 cmake dir: {pybind11_dir}")

    build_env = _windows_msvc_build_env()
    generator_args = cmake_generator_args(engine_dir, pybind11_dir, env=build_env)
    build_dir = engine_dir / "build"
    if (build_dir / "CMakeCache.txt").is_file():
        print("  Clearing stale CMake cache...")
        shutil.rmtree(build_dir, ignore_errors=True)

    for path in engine_dir.glob("fab_engine*"):
        if path.is_file() and path.suffix in MODULE_EXTENSIONS:
            path.unlink(missing_ok=True)

    configure_cmd = [
        cmake,
        "-B",
        "build",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-Dpybind11_DIR={pybind11_dir}",
        *generator_args,
        ".",
    ]
    print("  Configuring...")
    configure = subprocess.run(
        configure_cmd,
        cwd=str(engine_dir),
        check=False,
        text=True,
        capture_output=True,
        env=build_env,
    )
    for line in (configure.stdout or configure.stderr or "").splitlines():
        print(f"    {line}")
    if configure.returncode != 0:
        _fail("cmake configure failed")
        raise SystemExit(1)

    print("  Compiling...")
    build = subprocess.run(
        [cmake, "--build", "build", "--config", "Release", "--parallel"],
        cwd=str(engine_dir),
        check=False,
        text=True,
        capture_output=True,
        env=build_env,
    )
    for line in (build.stdout or build.stderr or "").splitlines():
        print(f"    {line}")
    if build.returncode != 0:
        _fail("cmake build failed - check errors above")
        raise SystemExit(1)

    hash_file = engine_dir / "engine_input_hash.txt"
    hash_file.write_text(input_hash, encoding="utf-8")
    _ok(f"Cached engine hash: {input_hash} -> {hash_file}")


def verify_module(engine_dir: Path) -> None:
    _step("5", "Verifying compiled module")
    module = find_compiled_module(engine_dir)
    if module is None:
        _fail(f"Compiled module not found in {engine_dir}")
        raise SystemExit(1)

    _ok(f"Module: {module}")
    engine_fwd = str(engine_dir).replace("\\", "/")
    py_code = (
        "import sys\n"
        f"sys.path.insert(0, '{engine_fwd}')\n"
        "import fab_engine\n"
        "gs = fab_engine.GameState()\n"
        "gs.seed_rng(123)\n"
        "gs.register_all_cards()\n"
        "gs.init_standard_decks()\n"
        "legal = gs.get_legal_actions()\n"
        "print(str(len(legal)) + ' legal actions on fresh state')\n"
        "fast = gs.fast_step_index(0)\n"
        "print('fast obs=' + str(len(fast.obs_vec)) + ' legal=' + str(fast.legal_count))\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(py_code)
        tmp_py = Path(handle.name)
    try:
        completed = subprocess.run(
            [sys.executable, str(tmp_py)],
            check=False,
            text=True,
            capture_output=True,
        )
        output = (completed.stdout or completed.stderr or "").strip()
        if completed.returncode == 0:
            _ok(output)
        else:
            _warn("Import test failed (likely unimplemented card stubs):")
            print(f"    {output}")
            print("  This is expected until cards.h stubs are fully implemented.")
    finally:
        tmp_py.unlink(missing_ok=True)


def print_summary(engine_dir: Path, matchup_key: str, deck1: str, deck2: str) -> None:
    _header("Done")
    print()
    print(f"  Engine dir : {engine_dir}")
    print(f"  Matchup    : {matchup_key}")
    print()
    print("  TalisharEngineEnvironment will auto-detect and use this engine")
    print("  on the next training run (no code changes needed).")
    print()
    print("  To regenerate after editing cards.h:")
    print(f"    python scripts/cpp/build_cpp_engine_for_matchup.py --deck1 {deck1} --deck2 {deck2} --no-build")
    print()
    print("  To bypass the C++ engine and use HTTP Talishar:")
    print("    TalisharEngineEnvironment(..., use_cpp_engine=False)")
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--deck1", default="", help="Talishar Assets deck name for player 1")
    parser.add_argument("--deck2", default="", help="Talishar Assets deck name for player 2")
    parser.add_argument(
        "--talishar-src",
        default=str(DEFAULT_TALISHAR_SRC),
        help="Path to the Talishar PHP source root",
    )
    parser.add_argument(
        "--talishar-url",
        default="",
        help="URL of the running Talishar server (default: TALISHAR_URL or http://localhost)",
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Skip live Talishar game; use PHP source scan only",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Generate C++ source but skip cmake build",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Override the default cache root",
    )
    parser.add_argument(
        "--pipeline-json",
        default="",
        help="Path to train_full_pipeline.py results JSON (deck names read automatically)",
    )
    parser.add_argument(
        "--deck1-json",
        default="",
        help="Optional FaBrary/FABdb deck JSON for P1",
    )
    parser.add_argument(
        "--deck2-json",
        default="",
        help="Optional FaBrary/FABdb deck JSON for P2",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    deck1 = args.deck1
    deck2 = args.deck2

    if args.pipeline_json:
        _step("0", "Reading deck names from pipeline JSON")
        pipeline_path = Path(args.pipeline_json)
        if not pipeline_path.is_file():
            _fail(f"Pipeline JSON not found: {args.pipeline_json}")
            return 1
        p1_name, p2_name = deck_names_from_pipeline(pipeline_path.resolve())
        if not deck1 and p1_name:
            deck1 = p1_name
        if not deck2 and p2_name:
            deck2 = p2_name
        _ok(f"Deck1={deck1}  Deck2={deck2}")

    if not deck1 or not deck2:
        _fail("Deck1 and Deck2 must be specified (or supply --pipeline-json).")
        print()
        print("Usage examples:")
        print("  python scripts/cpp/build_cpp_engine_for_matchup.py --deck1 Ira --deck2 Ira")
        print(
            "  python scripts/cpp/build_cpp_engine_for_matchup.py "
            "--deck1 BriarSAGEPrecon --deck2 DorintheSAGEPrecon"
        )
        return 1

    talishar_src = Path(args.talishar_src)
    assets_dir = talishar_src / "Assets"
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    from flesh_and_blood_rlbridge.talishar_deck_assets import resolve_talishar_deck_stem

    resolved1 = resolve_talishar_deck_stem(assets_dir, deck1)
    resolved2 = resolve_talishar_deck_stem(assets_dir, deck2)
    if resolved1 != deck1:
        _ok(f"Resolved deck1: {deck1} -> {resolved1}")
        deck1 = resolved1
    if resolved2 != deck2:
        _ok(f"Resolved deck2: {deck2} -> {resolved2}")
        deck2 = resolved2

    deck1_json = Path(args.deck1_json) if args.deck1_json else None
    deck2_json = Path(args.deck2_json) if args.deck2_json else None
    input_hash = deck_input_hash(deck1, deck2, deck1_json, deck2_json)
    matchup_key = f"{deck1}_vs_{deck2}-{input_hash}"
    engine_dir = Path(args.cache_dir) / matchup_key
    engine_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_build:
        cached_module = cached_engine_up_to_date(engine_dir, input_hash)
        if cached_module is not None:
            print()
            print(f"  Engine already up to date (hash {input_hash}).")
            print(f"  Skipping rebuild.  Module: {cached_module}")
            print(f"  (Delete '{engine_dir / 'engine_input_hash.txt'}' or '{cached_module}' to force a rebuild.)")
            print()
            return 0

    base_url = resolve_base_url(args.talishar_url or None)
    talishar_src = Path(args.talishar_src)

    _header("FAB C++ Engine Builder")
    print(f"  Matchup    : {deck1}  vs  {deck2}")
    print(f"  Engine dir : {engine_dir}")
    print(f"  PHP source : {talishar_src}")
    print(f"  Server URL : {base_url}")
    if args.no_server:
        print("  Mode       : PHP scan only (--no-server)")

    os.chdir(REPO_ROOT)
    generate_cpp_source(
        deck1=deck1,
        deck2=deck2,
        talishar_src=talishar_src,
        engine_dir=engine_dir,
        base_url=base_url,
        no_server=args.no_server,
        deck1_json=deck1_json,
        deck2_json=deck2_json,
    )

    _step("2", "Reviewing generated card stubs")
    print_card_manifest_summary(engine_dir)

    _step("3", "Checking pybind11")
    ensure_pybind11()

    if args.no_build:
        print()
        print("  Skipping build (--no-build flag set).")
        print("  Build manually:")
        print(f'    cd "{engine_dir}"')
        print("    cmake -B build -DCMAKE_BUILD_TYPE=Release .")
        print("    cmake --build build --config Release")
    else:
        pybind11_dir = pybind11_cmake_dir()
        build_engine(engine_dir, input_hash, pybind11_dir)
        verify_module(engine_dir)

    print_summary(engine_dir, matchup_key, deck1, deck2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
