#!/usr/bin/env python3
"""Run randomized parity sweeps across local Talishar deck matchups.

Discovers Talishar local deck asset files, builds the complete deck-vs-deck
matchup range, shuffles it, and invokes run_parity_check.py for each pair.

Each matchup writes per-matchup artifacts under
results/parity_checks/<deck1>_vs_<deck2>/.
This script also aggregates a sweep-level summary under
results/parity_sweeps/sweep_<timestamp>/.

By default only matchups with a compiled C++ engine in results/cpp_engines
are run. Use --build-missing-engines to generate/build engines first, or
--include-missing-engines to attempt parity anyway (likely setup failures).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_ROOT = REPO_ROOT / "scripts"
SCRIPTS_CPP = _SCRIPTS_ROOT / "cpp"
ASSETS_DIR = REPO_ROOT / "Talishar" / "Assets"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "parity_sweeps"
ENGINE_BUILDER = SCRIPTS_CPP / "build_cpp_engine_for_matchup.py"
PARITY_RUNNER = SCRIPTS_CPP / "run_parity_check.py"

sys.path.insert(0, str(_SCRIPTS_ROOT))
import _bootstrap  # noqa: E402

_bootstrap.configure_paths()

from check_cpp_vs_talishar_parity import _safe_label  # noqa: E402
from flesh_and_blood_rlbridge.cpp_engine_environment import (  # noqa: E402
    get_engine_dir,
    is_cpp_engine_available,
)


@dataclass
class SweepRow:
    index: int
    deck1: str
    deck2: str
    status: str
    exit_code: int | None
    discrepancies_found: int | None
    setup_failures: int | None
    total_steps: int | None
    first_failure: str
    report: str | None


def _default_talishar_url() -> str:
    return os.environ.get("TALISHAR_URL", "http://localhost:8080/game")


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _matchup_dir(deck1: str, deck2: str) -> str:
    return f"{_safe_label(deck1)}_vs_{_safe_label(deck2)}"


def _report_path(deck1: str, deck2: str) -> Path:
    return REPO_ROOT / "results" / "parity_checks" / _matchup_dir(deck1, deck2) / "parity_report.json"


def discover_decks(patterns: list[str]) -> list[str]:
    if not ASSETS_DIR.is_dir():
        raise FileNotFoundError(f"Talishar Assets directory not found: {ASSETS_DIR}")

    deck_files: dict[str, Path] = {}
    for pattern in patterns:
        for path in ASSETS_DIR.glob(pattern):
            if not path.is_file():
                continue
            name = path.name
            base = path.stem
            if base == "Dummy":
                continue
            if name.startswith(("eval_", "rl_")) or name.startswith("MetafyDictionary"):
                continue
            deck_files[base] = path

    decks = sorted(deck_files)
    if len(decks) < 2:
        raise RuntimeError(
            f"Need at least two deck assets. Patterns: {', '.join(patterns)}"
        )
    return decks


def build_matchups(
    decks: list[str],
    *,
    unordered_only: bool,
    exclude_self_matchups: bool,
) -> list[tuple[str, str]]:
    matchups: list[tuple[str, str]] = []
    for deck1 in decks:
        for deck2 in decks:
            if exclude_self_matchups and deck1 == deck2:
                continue
            if unordered_only and deck1.casefold() >= deck2.casefold():
                continue
            matchups.append((deck1, deck2))
    return matchups


def cpp_engine_available(deck1: str, deck2: str, cache_dir: str = "") -> bool:
    engine_dir = get_engine_dir(deck1, deck2, cache_dir=cache_dir or None)
    return is_cpp_engine_available(engine_dir)


def build_cpp_engine(deck1: str, deck2: str, talishar_url: str) -> int:
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
    print(f"  $ {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return int(completed.returncode)


def run_matchup_parity(
    *,
    deck1: str,
    deck2: str,
    game_format: str,
    mode: str,
    episodes: int,
    steps_per_episode: int,
    talishar_url: str,
    cpp_engine_cache_dir: str,
    stop_after_failure: bool,
    parity_mode: str = "contract",
    sync_scope: str = "full",
    disable_obs_alignment: bool = False,
    rng_seed: int | None = None,
) -> int:
    """Run one matchup parity check in a fresh process (fab_engine is not reloadable)."""
    cmd = [
        sys.executable,
        str(PARITY_RUNNER),
        "--deck1",
        deck1,
        "--deck2",
        deck2,
        "--format",
        game_format,
        "--mode",
        mode,
        "--episodes",
        str(episodes),
        "--parity-mode",
        parity_mode,
        "--sync-scope",
        sync_scope,
        "--talishar-url",
        talishar_url,
    ]
    if steps_per_episode > 0:
        cmd.extend(["--steps-per-episode", str(steps_per_episode)])
    if cpp_engine_cache_dir:
        cmd.extend(["--cpp-engine-cache-dir", cpp_engine_cache_dir])
    if stop_after_failure:
        cmd.append("--stop-after-failure")
    if disable_obs_alignment:
        cmd.append("--disable-obs-alignment")
    if rng_seed is not None:
        cmd.extend(["--seed", str(rng_seed)])
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return int(completed.returncode)


def parse_report(report_path: Path) -> dict[str, Any]:
    return json.loads(report_path.read_text(encoding="utf-8"))


def status_from_exit_code(exit_code: int) -> str:
    return {
        0: "passed",
        1: "discrepancy",
        2: "setup_failed",
    }.get(exit_code, "error")


def save_sweep_summary(
    rows: list[SweepRow],
    sweep_dir: Path,
    *,
    timestamp: str,
    mode: str,
    game_format: str,
    episodes_per_matchup: int,
    seed: int,
    talishar_url: str,
) -> None:
    sweep_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(row) for row in rows]

    (sweep_dir / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    if payload:
        with (sweep_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=payload[0].keys())
            writer.writeheader()
            writer.writerows(payload)

    counts = {key: 0 for key in ("passed", "discrepancy", "setup_failed", "build_failed", "skipped_no_engine", "error")}
    for row in rows:
        if row.status in counts:
            counts[row.status] += 1

    lines = [
        "RANDOM PARITY SWEEP SUMMARY",
        "===========================",
        "",
        f"Timestamp       : {timestamp}",
        f"Mode            : {mode}",
        f"Format          : {game_format}",
        f"Episodes/matchup: {episodes_per_matchup}",
        f"Seed            : {seed}",
        f"Talishar URL    : {talishar_url}",
        "",
        f"Matchups run    : {len(rows)}",
        f"Passed          : {counts['passed']}",
        f"Discrepancies   : {counts['discrepancy']}",
        f"Setup failed    : {counts['setup_failed']}",
        f"Build failed    : {counts['build_failed']}",
        f"Skipped (no C++): {counts['skipped_no_engine']}",
        f"Errors          : {counts['error']}",
        "",
        "DETAILS",
        "-------",
    ]
    for row in rows:
        detail = f"[{row.status}] {row.deck1} vs {row.deck2}"
        if row.discrepancies_found is not None:
            detail += f" | discrepancies={row.discrepancies_found} steps={row.total_steps}"
        if row.first_failure:
            detail += f" | {row.first_failure}"
        lines.append(detail)

    (sweep_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--talishar-url", default="", help="Talishar HTTP base URL")
    parser.add_argument(
        "--deck-name-pattern",
        action="append",
        default=None,
        help="Asset filename glob under Talishar/Assets (repeatable). Default: Ira.txt and *SAGEPrecon.txt",
    )
    parser.add_argument(
        "--format",
        default="silver_age",
        choices=["silver_age", "classic_constructed"],
    )
    parser.add_argument(
        "--mode",
        default="full-episode",
        choices=["single-step", "multi-step", "full-episode", "stress-test"],
    )
    parser.add_argument("--episodes-per-matchup", type=int, default=100)
    parser.add_argument("--steps-per-episode", type=int, default=500)
    parser.add_argument(
        "--max-matchups",
        type=int,
        default=0,
        help="Optional cap after shuffling the complete matchup range (0 = all)",
    )
    parser.add_argument("--seed", type=int, default=8675309)
    parser.add_argument("--unordered-only", action="store_true")
    parser.add_argument("--exclude-self-matchups", action="store_true")
    parser.add_argument("--build-missing-engines", action="store_true")
    parser.add_argument(
        "--skip-missing-engines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip matchups without a compiled C++ engine (default: true)",
    )
    parser.add_argument(
        "--include-missing-engines",
        action="store_true",
        help="Attempt parity even when no compiled engine exists",
    )
    parser.add_argument("--stop-after-failure", action="store_true")
    parser.add_argument("--cpp-engine-cache-dir", default="")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Root directory for sweep summaries",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not PARITY_RUNNER.is_file():
        raise FileNotFoundError(f"Parity runner not found: {PARITY_RUNNER}")
    if args.build_missing_engines and not ENGINE_BUILDER.is_file():
        raise FileNotFoundError(f"Engine builder not found: {ENGINE_BUILDER}")

    talishar_url = args.talishar_url or _default_talishar_url()
    skip_missing_engines = args.skip_missing_engines
    if args.include_missing_engines:
        skip_missing_engines = False

    patterns = args.deck_name_pattern or ["Ira.txt", "*SAGEPrecon.txt"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.output_dir) / f"sweep_{timestamp}"

    os.chdir(REPO_ROOT)

    _section("Discovering Decks")
    decks = discover_decks(patterns)
    print(f"  Decks discovered: {len(decks)}")
    print(f"  Patterns        : {', '.join(patterns)}")
    print(f"  Deck list       : {', '.join(decks)}")

    _section("Preparing Randomized Matchups")
    matchups = build_matchups(
        decks,
        unordered_only=args.unordered_only,
        exclude_self_matchups=args.exclude_self_matchups,
    )
    rng = random.Random(args.seed)
    rng.shuffle(matchups)
    if args.max_matchups > 0:
        matchups = matchups[: args.max_matchups]

    print(f"  Seed            : {args.seed}")
    print(f"  Matchups queued : {len(matchups)}")
    print(f"  Mode            : {args.mode}")
    print(f"  Episodes/match  : {args.episodes_per_matchup}")
    if args.steps_per_episode > 0:
        print(f"  Steps/episode   : {args.steps_per_episode}")
    print(f"  Unordered only  : {args.unordered_only}")
    print(f"  Exclude self    : {args.exclude_self_matchups}")
    print(f"  Skip no engine  : {skip_missing_engines}")
    print(f"  Build missing   : {args.build_missing_engines}")
    print(f"  Stop on failure : {args.stop_after_failure}")
    print(f"  Talishar URL    : {talishar_url}")
    print(f"  Sweep dir       : {sweep_dir}")

    rows: list[SweepRow] = []
    total = len(matchups)

    for index, (deck1, deck2) in enumerate(matchups, start=1):
        label = f"{deck1} vs {deck2}"
        _section(f"[{index} / {total}] {label}")

        if skip_missing_engines and not args.build_missing_engines:
            if not cpp_engine_available(deck1, deck2, args.cpp_engine_cache_dir):
                print("  No compiled C++ engine found; skipping.")
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
                        first_failure="No compiled fab_engine in results/cpp_engines",
                        report=None,
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
                continue

        if args.build_missing_engines:
            print("  Building/checking C++ engine cache...")
            build_rc = build_cpp_engine(deck1, deck2, talishar_url)
            if build_rc != 0:
                print(f"  Build failed for {label} (exit {build_rc}).")
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
                continue

        exit_code = run_matchup_parity(
            deck1=deck1,
            deck2=deck2,
            game_format=args.format,
            mode=args.mode,
            episodes=args.episodes_per_matchup,
            steps_per_episode=args.steps_per_episode,
            talishar_url=talishar_url,
            cpp_engine_cache_dir=args.cpp_engine_cache_dir,
            stop_after_failure=args.stop_after_failure,
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
                    first_failure = str(discrepancies_list[0].get("description", ""))
            except Exception as exc:
                first_failure = f"Could not parse report: {exc}"
        else:
            first_failure = f"Report not found: {report_path}"

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

    _section("Sweep Complete")
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

    passed = sum(1 for row in rows if row.status == "passed")
    discrepancy = sum(1 for row in rows if row.status == "discrepancy")
    setup_failed = sum(1 for row in rows if row.status == "setup_failed")
    build_failed = sum(1 for row in rows if row.status == "build_failed")
    skipped = sum(1 for row in rows if row.status == "skipped_no_engine")
    errors = sum(1 for row in rows if row.status == "error")

    print(f"  Passed          : {passed}")
    print(f"  Discrepancies   : {discrepancy}")
    print(f"  Setup failed    : {setup_failed}")
    print(f"  Build failed    : {build_failed}")
    print(f"  Skipped (no C++): {skipped}")
    print(f"  Errors          : {errors}")
    print()
    print(f"  Sweep summary   : {sweep_dir / 'summary.txt'}")
    print(f"  Sweep JSON      : {sweep_dir / 'summary.json'}")
    print(f"  Sweep CSV       : {sweep_dir / 'summary.csv'}")

    if discrepancy + setup_failed + build_failed + errors > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
