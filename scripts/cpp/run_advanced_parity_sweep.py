#!/usr/bin/env python3
"""Advanced strict-simulation parity sweep across local Talishar deck matchups.

Defaults to ``--parity-mode simulation`` (independent C++ stepping, full game-state
comparison). Discovers SAGE precon and fab_precon assets, builds missing engines
optionally, and aggregates failure taxonomy under results/parity_sweeps/advanced_*.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_ROOT))
import _bootstrap  # noqa: E402

_bootstrap.configure_paths()

from run_random_parity_sweep import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    SweepRow,
    build_cpp_engine,
    build_matchups,
    cpp_engine_available,
    discover_decks,
    parse_report,
    run_matchup_parity,
    save_sweep_summary,
    status_from_exit_code,
    _default_talishar_url,
    _report_path,
    _section,
)

DEFAULT_DECK_PATTERNS = ["*SAGEPrecon.txt", "fab_precon_*.txt", "Ira.txt"]


def _first_taxonomy(report_path: Path) -> str:
    if not report_path.is_file():
        return ""
    try:
        report = parse_report(report_path)
        discrepancies = report.get("discrepancies") or []
        if not discrepancies:
            return ""
        return str(discrepancies[0].get("taxonomy", "") or "")
    except Exception:
        return ""


def build_failure_index(rows: list[SweepRow], sweep_dir: Path) -> dict[str, int]:
    index: dict[str, int] = {}
    for row in rows:
        if row.status != "discrepancy" or not row.report:
            taxonomy = _first_taxonomy(Path(row.report))
            if taxonomy:
                index[taxonomy] = index.get(taxonomy, 0) + 1
    (sweep_dir / "failure_index.json").write_text(
        json.dumps(index, indent=2),
        encoding="utf-8",
    )
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--talishar-url", default="")
    parser.add_argument(
        "--deck-name-pattern",
        action="append",
        default=None,
        help="Asset glob under Talishar/Assets (repeatable)",
    )
    parser.add_argument(
        "--format",
        default="silver_age",
        choices=["silver_age", "classic_constructed"],
    )
    parser.add_argument(
        "--mode",
        default="multi-step",
        choices=["single-step", "multi-step", "full-episode", "stress-test"],
    )
    parser.add_argument("--episodes-per-matchup", type=int, default=5)
    parser.add_argument("--steps-per-episode", type=int, default=50)
    parser.add_argument("--max-matchups", type=int, default=0)
    parser.add_argument("--seed", type=int, default=8675309)
    parser.add_argument("--unordered-only", action="store_true", default=True)
    parser.add_argument("--exclude-self-matchups", action="store_true", default=True)
    parser.add_argument("--build-missing-engines", action="store_true")
    parser.add_argument("--skip-missing-engines", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-after-failure", action="store_true")
    parser.add_argument("--cpp-engine-cache-dir", default="")
    parser.add_argument(
        "--parity-mode",
        default="simulation",
        choices=["contract", "simulation"],
    )
    parser.add_argument(
        "--sync-scope",
        default="full",
        choices=["hands", "full"],
    )
    parser.add_argument("--disable-obs-alignment", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    talishar_url = args.talishar_url or _default_talishar_url()
    patterns = args.deck_name_pattern or DEFAULT_DECK_PATTERNS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.output_dir) / f"advanced_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(REPO_ROOT)

    _section("Advanced Parity Sweep")
    decks = discover_decks(patterns)
    print(f"  Decks discovered: {len(decks)}")
    print(f"  Parity mode     : {args.parity_mode}")
    print(f"  Sync scope      : {args.sync_scope}")

    matchups = build_matchups(
        decks,
        unordered_only=args.unordered_only,
        exclude_self_matchups=args.exclude_self_matchups,
    )
    if args.max_matchups > 0:
        matchups = matchups[: args.max_matchups]

    rows: list[SweepRow] = []
    total = len(matchups)

    for index, (deck1, deck2) in enumerate(matchups, start=1):
        label = f"{deck1} vs {deck2}"
        _section(f"[{index} / {total}] {label}")

        if args.skip_missing_engines and not args.build_missing_engines:
            if not cpp_engine_available(deck1, deck2, args.cpp_engine_cache_dir):
                rows.append(
                    SweepRow(
                        index=index,
                        deck1=deck1,
                        deck2=deck2,
                        status="skipped_no_engine",
                        exit_code=None,
                        discrepancies_found=None,
                        setup_failures=None,
                        total_steps=0,
                        first_failure="No compiled fab_engine",
                        report=None,
                    )
                )
                continue

        if args.build_missing_engines:
            build_rc = build_cpp_engine(deck1, deck2, talishar_url)
            if build_rc != 0:
                rows.append(
                    SweepRow(
                        index=index,
                        deck1=deck1,
                        deck2=deck2,
                        status="build_failed",
                        exit_code=build_rc,
                        discrepancies_found=None,
                        setup_failures=None,
                        total_steps=0,
                        first_failure="C++ engine build failed",
                        report=None,
                    )
                )
                continue

        from run_parity_check import execute_parity_check

        exit_code = execute_parity_check(
            deck1=deck1,
            deck2=deck2,
            game_format=args.format,
            episodes=args.episodes_per_matchup,
            mode=args.mode,
            steps_per_episode=args.steps_per_episode,
            talishar_url=talishar_url,
            cpp_engine_cache_dir=args.cpp_engine_cache_dir,
            stop_after_failure=args.stop_after_failure,
            verbose_header=False,
            parity_mode=args.parity_mode,
            sync_scope=args.sync_scope,
            disable_obs_alignment=args.disable_obs_alignment,
            rng_seed=args.seed,
        )

        report_path = _report_path(deck1, deck2)
        first_failure = ""
        discrepancies = None
        setup_failures = None
        total_steps = None
        if report_path.is_file():
            try:
                report = parse_report(report_path)
                discrepancies = report.get("discrepancies_found")
                setup_failures = report.get("setup_failures")
                total_steps = report.get("total_steps")
                discrepancies_list = report.get("discrepancies") or []
                if discrepancies_list:
                    first = discrepancies_list[0]
                    taxonomy = first.get("taxonomy", "")
                    first_failure = str(first.get("description", ""))
                    if taxonomy:
                        first_failure = f"[{taxonomy}] {first_failure}"
            except Exception as exc:
                first_failure = f"Could not parse report: {exc}"

        repro_src = report_path.parent / "repro_trace.json"
        if repro_src.is_file() and exit_code != 0:
            repro_dst = sweep_dir / f"repro_{deck1}_vs_{deck2}.json"
            shutil.copy2(repro_src, repro_dst)

        rows.append(
            SweepRow(
                index=index,
                deck1=deck1,
                deck2=deck2,
                status=status_from_exit_code(exit_code),
                exit_code=exit_code,
                discrepancies_found=discrepancies,
                setup_failures=setup_failures,
                total_steps=total_steps,
                first_failure=first_failure,
                report=str(report_path),
            )
        )
        save_sweep_summary(
            rows,
            sweep_dir,
            timestamp=timestamp,
            mode=args.mode,
            game_format=args.format,
            episodes_per_matchup=args.episodes_per_matchup,
            seed=args.seed,
            talishar_url=talishar_url,
        )

    build_failure_index(rows, sweep_dir)
    _section("Advanced Sweep Complete")
    print(f"  Sweep dir: {sweep_dir}")
    passed = sum(1 for row in rows if row.status == "passed")
    failed = sum(1 for row in rows if row.status == "discrepancy")
    if failed:
        return 1
    return 0 if passed or not rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
