#!/usr/bin/env python3
"""Multi-deck agent-transfer readiness sweep (C++ train → Talishar HTTP eval).

For each self-match deck, runs contract-mode parity (mirrored stepping — what
policy-agreement tests use) and records reset/step obs+legal contract status.
Each matchup runs in a subprocess so fab_engine loads the correct binary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_ROOT))
import _bootstrap  # noqa: E402

_bootstrap.configure_paths()

from run_random_parity_sweep import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    _default_talishar_url,
    _report_path,
    _section,
    build_cpp_engine,
    cpp_engine_available,
    discover_decks,
    parse_report,
    status_from_exit_code,
)


@dataclass
class TransferRow:
    deck: str
    contract_status: str
    contract_exit: int | None
    discrepancies: int | None
    total_steps: int | None
    first_failure: str
    report: str | None


def _run_contract_parity(
    deck: str,
    *,
    talishar_url: str,
    steps: int,
    seed: int | None,
    build: bool,
) -> tuple[int, Path | None]:
    if build and not cpp_engine_available(deck, deck):
        rc = build_cpp_engine(deck, deck, talishar_url)
        if rc != 0:
            return rc, None

    cmd = [
        sys.executable,
        str(_SCRIPTS_ROOT / "cpp" / "run_parity_check.py"),
        "--deck1",
        deck,
        "--deck2",
        deck,
        "--parity-mode",
        "contract",
        "--mode",
        "multi-step",
        "--episodes",
        "1",
        "--steps-per-episode",
        str(steps),
        "--talishar-url",
        talishar_url,
        "--stop-after-failure",
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return int(completed.returncode), _report_path(deck, deck)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deck-name-pattern",
        action="append",
        default=None,
        help="Asset glob (default: Ira + *SAGEPrecon.txt)",
    )
    parser.add_argument("--max-decks", type=int, default=0)
    parser.add_argument("--steps-per-episode", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--build-missing-engines", action="store_true")
    parser.add_argument("--talishar-url", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    talishar_url = args.talishar_url or _default_talishar_url()
    patterns = args.deck_name_pattern or ["Ira.txt", "*SAGEPrecon.txt", "fab_precon_*.txt"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.output_dir) / f"transfer_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO_ROOT)

    decks = discover_decks(patterns)
    if args.max_decks > 0:
        decks = decks[: args.max_decks]

    _section("Agent transfer sweep (contract parity)")
    print(f"  Decks: {len(decks)}")
    print(f"  Steps: {args.steps_per_episode}")
    print(f"  Output: {sweep_dir}")

    rows: list[TransferRow] = []
    for index, deck in enumerate(decks, start=1):
        _section(f"[{index}/{len(decks)}] {deck} vs {deck}")
        exit_code, report_path = _run_contract_parity(
            deck,
            talishar_url=talishar_url,
            steps=args.steps_per_episode,
            seed=args.seed,
            build=args.build_missing_engines,
        )
        status = status_from_exit_code(exit_code) if exit_code in (0, 1, 2) else "error"
        discrepancies = None
        total_steps = None
        first_failure = ""
        report_str = str(report_path) if report_path else None
        if report_path and report_path.is_file():
            try:
                report = parse_report(report_path)
                discrepancies = int(report.get("discrepancies_found", 0) or 0)
                total_steps = int(report.get("total_steps", 0) or 0)
                discs = report.get("discrepancies") or []
                if discs:
                    first_failure = str(discs[0].get("description", "") or "")
            except Exception as exc:  # noqa: BLE001
                first_failure = str(exc)
        rows.append(
            TransferRow(
                deck=deck,
                contract_status=status,
                contract_exit=exit_code,
                discrepancies=discrepancies,
                total_steps=total_steps,
                first_failure=first_failure,
                report=report_str,
            )
        )

    passed = sum(1 for r in rows if r.contract_status == "passed")
    failed = sum(1 for r in rows if r.contract_status == "discrepancy")
    (sweep_dir / "summary.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2),
        encoding="utf-8",
    )
    lines = [
        f"Agent transfer contract sweep — {timestamp}",
        f"Passed: {passed}/{len(rows)}",
        f"Failed: {failed}/{len(rows)}",
        "",
    ]
    for row in rows:
        mark = "OK" if row.contract_status == "passed" else "FAIL"
        steps = row.total_steps if row.total_steps is not None else "?"
        lines.append(
            f"  [{mark}] {row.deck}: steps={steps} disc={row.discrepancies} "
            f"{row.first_failure[:80]}"
        )
    (sweep_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _section("Transfer sweep complete")
    print(f"  Passed: {passed}/{len(rows)}")
    print(f"  Summary: {sweep_dir / 'summary.txt'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
