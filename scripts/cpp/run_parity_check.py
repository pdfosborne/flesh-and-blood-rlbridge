#!/usr/bin/env python3
"""Run C++ Engine vs Talishar HTTP parity check.

Orchestrates parity verification between CppEngineEnvironment and
TalisharEngineEnvironment (HTTP-backed) for a deck matchup.

Set TALISHAR_URL in the environment if Talishar is not on the default URL.
The C++ engine must be pre-built for the matchup unless a cache dir is provided.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_ROOT))
import _bootstrap  # noqa: E402

_bootstrap.configure_paths()

from check_cpp_vs_talishar_parity import (  # noqa: E402
    _safe_label,
    run_parity_check,
)

VALID_MODES = ("single-step", "multi-step", "full-episode", "stress-test")


def _default_talishar_url() -> str:
    return os.environ.get("TALISHAR_URL", "http://localhost:8080/game")


def _matchup_dir(deck1: str, deck2: str) -> str:
    return f"{_safe_label(deck1)}_vs_{_safe_label(deck2)}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--deck1",
        "--deck1-source",
        dest="deck1",
        default="Ira",
        help="FaBrary URL, deck slug, local JSON path, or Talishar deck name (P1)",
    )
    parser.add_argument(
        "--deck2",
        "--deck2-source",
        dest="deck2",
        default="Briar",
        help="FaBrary URL, deck slug, local JSON path, or Talishar deck name (P2)",
    )
    parser.add_argument(
        "--format",
        default="silver_age",
        choices=["silver_age", "classic_constructed"],
        help="Game format",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to test")
    parser.add_argument(
        "--mode",
        default="full-episode",
        choices=list(VALID_MODES),
        help="Parity test mode",
    )
    parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=0,
        help="Steps per episode (multi-step mode only; 0 = use checker default)",
    )
    parser.add_argument(
        "--talishar-url",
        default="",
        help="Talishar server URL (default: TALISHAR_URL or http://localhost:8080/game)",
    )
    parser.add_argument(
        "--cpp-engine-dir",
        default="",
        help="Explicit directory containing a compiled fab_engine module",
    )
    parser.add_argument(
        "--cpp-engine-cache-dir",
        default="",
        help="Cache directory containing compiled C++ engine matchup subdirectories",
    )
    parser.add_argument(
        "--cpp-engine-deck1",
        default="",
        help="Override deck 1 name used only for C++ engine cache lookup",
    )
    parser.add_argument(
        "--cpp-engine-deck2",
        default="",
        help="Override deck 2 name used only for C++ engine cache lookup",
    )
    parser.add_argument(
        "--stop-after-failure",
        action="store_true",
        help="Stop at the first discrepancy instead of collecting all findings",
    )
    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
        help="Deprecated alias; continuing after failures is already the default",
    )
    return parser


def execute_parity_check(
    *,
    deck1: str = "Ira",
    deck2: str = "Briar",
    game_format: str = "silver_age",
    episodes: int = 10,
    mode: str = "full-episode",
    steps_per_episode: int = 0,
    talishar_url: str = "",
    cpp_engine_dir: str = "",
    cpp_engine_cache_dir: str = "",
    cpp_engine_deck1: str = "",
    cpp_engine_deck2: str = "",
    stop_after_failure: bool = False,
    verbose_header: bool = True,
) -> int:
    if mode not in VALID_MODES:
        print(f"ERROR: Invalid mode '{mode}'. Must be one of: {', '.join(VALID_MODES)}")
        return 2

    url = talishar_url or _default_talishar_url()
    if verbose_header:
        print()
        print("=" * 64)
        print("  C++ vs Talishar Parity Check")
        print("=" * 64)
        print(f"  Deck 1   : {deck1}")
        print(f"  Deck 2   : {deck2}")
        print(f"  Format   : {game_format}")
        print(f"  Mode     : {mode}")
        print(f"  Episodes : {episodes}")
        if steps_per_episode > 0:
            print(f"  Steps/Ep : {steps_per_episode}")
        print(f"  Talishar : {url}")
        if cpp_engine_dir:
            print(f"  C++ Dir  : {cpp_engine_dir}")
        if cpp_engine_cache_dir:
            print(f"  C++ Cache: {cpp_engine_cache_dir}")
        if stop_after_failure:
            print("  Stop     : after first discrepancy")
        print("=" * 64)
        print()
        print("  Starting parity check...")
        print()

    out_dir = REPO_ROOT / "results" / "parity_checks" / _matchup_dir(deck1, deck2)
    _, exit_code = run_parity_check(
        deck1=deck1,
        deck2=deck2,
        game_format=game_format,
        episodes=episodes,
        mode=mode,
        steps_per_episode=steps_per_episode if steps_per_episode > 0 else None,
        talishar_url=url,
        cpp_engine_dir=cpp_engine_dir or None,
        cpp_engine_cache_dir=cpp_engine_cache_dir or None,
        cpp_engine_deck1=cpp_engine_deck1 or None,
        cpp_engine_deck2=cpp_engine_deck2 or None,
        out_dir=out_dir,
        stop_after_failure=stop_after_failure,
        write_reports=True,
        verbose=verbose_header,
    )

    if verbose_header:
        _print_summary(exit_code, deck1, deck2)
    return exit_code


def _print_summary(exit_code: int, deck1: str, deck2: str) -> None:
    matchup_dir = _matchup_dir(deck1, deck2)
    base = REPO_ROOT / "results" / "parity_checks" / matchup_dir

    print()
    print("=" * 64)
    print("  Parity Check Complete")
    print("=" * 64)
    if exit_code == 0:
        print("  Status   : ALL CHECKS PASSED")
        print("  Verdict  : C++ and Talishar environments are PARITY-MATCHED")
    elif exit_code == 1:
        print("  Status   : DISCREPANCIES DETECTED")
        print("  Verdict  : See parity_summary.txt for details")
    elif exit_code == 2:
        print("  Status   : SETUP FAILED")
        print("  Verdict  : C++ engine or Talishar setup is not ready; see parity_summary.txt")
    else:
        print(f"  Status   : ERROR ({exit_code})")

    print()
    print("  Output files:")
    print(f"    JSON Report: {base / 'parity_report.json'}")
    print(f"    Summary     : {base / 'parity_summary.txt'}")
    if exit_code != 0:
        print(f"    HTML Diff   : {base / 'discrepancies.html'}")
    print()
    print("=" * 64)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(REPO_ROOT)
    return execute_parity_check(
        deck1=args.deck1,
        deck2=args.deck2,
        game_format=args.format,
        episodes=args.episodes,
        mode=args.mode,
        steps_per_episode=args.steps_per_episode,
        talishar_url=args.talishar_url,
        cpp_engine_dir=args.cpp_engine_dir,
        cpp_engine_cache_dir=args.cpp_engine_cache_dir,
        cpp_engine_deck1=args.cpp_engine_deck1,
        cpp_engine_deck2=args.cpp_engine_deck2,
        stop_after_failure=args.stop_after_failure,
        verbose_header=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
