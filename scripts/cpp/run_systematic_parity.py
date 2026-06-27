#!/usr/bin/env python3
"""Systematic Talishar vs C++ parity workflow.

Runs simulation-mode parity sweeps, analyzes failures, rebuilds engines when the
generator changes, and re-runs failing matchups until they pass or max rounds.

Typical usage::

    # One matchup, iterate fixes until pass or 3 rounds
    python scripts/cpp/run_systematic_parity.py --deck1 Ira --deck2 Ira

    # Sweep all SAGE precons (cap matchups for dev)
    python scripts/cpp/run_systematic_parity.py --sweep --max-matchups 5 --max-rounds 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_ROOT))
import _bootstrap  # noqa: E402

_bootstrap.configure_paths()

from check_cpp_vs_talishar_parity import _safe_label, run_parity_check  # noqa: E402
from run_advanced_parity_sweep import DEFAULT_DECK_PATTERNS, main as advanced_sweep_main  # noqa: E402
from analyze_parity_failures import analyze_sweep_dir  # noqa: E402

ENGINE_BUILDER = REPO_ROOT / "scripts" / "cpp" / "build_cpp_engine_for_matchup.py"
GENERATOR = REPO_ROOT / "scripts" / "cpp" / "generate_cpp_engine.py"


@dataclass
class RoundResult:
    round_no: int
    deck1: str
    deck2: str
    exit_code: int
    discrepancies: int
    first_taxonomy: str
    report_path: str


def _default_talishar_url() -> str:
    return os.environ.get("TALISHAR_URL", "http://localhost:8080/game")


def _report_path(deck1: str, deck2: str) -> Path:
    return REPO_ROOT / "results" / "parity_checks" / f"{_safe_label(deck1)}_vs_{_safe_label(deck2)}" / "parity_report.json"


def _first_taxonomy(report_path: Path) -> str:
    if not report_path.is_file():
        return ""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    discs = report.get("discrepancies") or []
    if not discs:
        return ""
    return str(discs[0].get("taxonomy") or discs[0].get("category") or "")


def rebuild_engine(deck1: str, deck2: str, *, talishar_url: str, no_server: bool) -> int:
    cmd = [
        sys.executable,
        str(ENGINE_BUILDER),
        "--deck1",
        deck1,
        "--deck2",
        deck2,
        "--talishar-src",
        str(REPO_ROOT / "Talishar"),
        "--talishar-url",
        talishar_url,
    ]
    if no_server:
        cmd.append("--no-server")
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return int(completed.returncode)


def run_matchup_round(
    *,
    deck1: str,
    deck2: str,
    round_no: int,
    talishar_url: str,
    episodes: int,
    steps: int,
    seed: int,
    sync_scope: str,
) -> RoundResult:
    report, exit_code = run_parity_check(
        deck1=deck1,
        deck2=deck2,
        episodes=episodes,
        mode="multi-step",
        steps_per_episode=steps,
        talishar_url=talishar_url,
        parity_mode="simulation",
        sync_scope=sync_scope,
        rng_seed=seed + round_no,
        stop_after_failure=True,
        write_reports=True,
        verbose=True,
    )
    report_path = _report_path(deck1, deck2)
    return RoundResult(
        round_no=round_no,
        deck1=deck1,
        deck2=deck2,
        exit_code=exit_code,
        discrepancies=int(report.discrepancies_found),
        first_taxonomy=_first_taxonomy(report_path),
        report_path=str(report_path),
    )


def systematic_matchup(
    *,
    deck1: str,
    deck2: str,
    max_rounds: int,
    rebuild_each_round: bool,
    talishar_url: str,
    episodes: int,
    steps: int,
    seed: int,
    sync_scope: str,
    no_server: bool,
) -> list[RoundResult]:
    results: list[RoundResult] = []
    for round_no in range(1, max_rounds + 1):
        print()
        print("=" * 72)
        print(f"  SYSTEMATIC PARITY  Round {round_no}/{max_rounds}  {deck1} vs {deck2}")
        print("=" * 72)

        if rebuild_each_round or round_no == 1:
            print("  Rebuilding C++ engine from generator...")
            rc = rebuild_engine(deck1, deck2, talishar_url=talishar_url, no_server=no_server)
            if rc != 0:
                print(f"  Engine build failed (exit {rc})")
                break

        result = run_matchup_round(
            deck1=deck1,
            deck2=deck2,
            round_no=round_no,
            talishar_url=talishar_url,
            episodes=episodes,
            steps=steps,
            seed=seed,
            sync_scope=sync_scope,
        )
        results.append(result)
        print(f"  Round {round_no} exit={result.exit_code} discrepancies={result.discrepancies}")
        if result.first_taxonomy:
            print(f"  First taxonomy: {result.first_taxonomy}")

        if result.exit_code == 0:
            print("  Matchup PASSED simulation parity.")
            break

        repro = Path(result.report_path).parent / "repro_trace.json"
        if repro.is_file():
            print(f"  Repro trace: {repro}")

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck1", default="Ira")
    parser.add_argument("--deck2", default="Ira")
    parser.add_argument("--sweep", action="store_true", help="Run advanced sweep instead of single matchup")
    parser.add_argument("--max-matchups", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sync-scope", default="full", choices=["hands", "full"])
    parser.add_argument("--talishar-url", default="")
    parser.add_argument("--no-server", action="store_true", help="Build engines without live Talishar card discovery")
    parser.add_argument("--rebuild-each-round", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    talishar_url = args.talishar_url or _default_talishar_url()
    os.chdir(REPO_ROOT)

    if args.sweep:
        sweep_argv = [
            "--parity-mode",
            "simulation",
            "--sync-scope",
            args.sync_scope,
            "--episodes-per-matchup",
            str(args.episodes),
            "--steps-per-episode",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--talishar-url",
            talishar_url,
        ]
        if args.max_matchups > 0:
            sweep_argv.extend(["--max-matchups", str(args.max_matchups)])
        if args.no_server:
            sweep_argv.append("--build-missing-engines")
        rc = advanced_sweep_main(sweep_argv)
        # Analyze latest sweep
        from run_advanced_parity_sweep import DEFAULT_OUTPUT_DIR
        from datetime import datetime

        sweeps = sorted(Path(DEFAULT_OUTPUT_DIR).glob("advanced_*"), reverse=True)
        if sweeps:
            analyze_sweep_dir(sweeps[0])
            print(f"Fix backlog: {sweeps[0] / 'fix_backlog.md'}")
        return rc

    results = systematic_matchup(
        deck1=args.deck1,
        deck2=args.deck2,
        max_rounds=args.max_rounds,
        rebuild_each_round=args.rebuild_each_round,
        talishar_url=talishar_url,
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        sync_scope=args.sync_scope,
        no_server=args.no_server,
    )
    if not results:
        return 2
    return 0 if results[-1].exit_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
