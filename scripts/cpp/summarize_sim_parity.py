#!/usr/bin/env python3
"""Run simulation parity for listed matchups and print a compact summary."""
from __future__ import annotations

import io
import contextlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import _bootstrap  # noqa: E402

_bootstrap.configure_paths()

from check_cpp_vs_talishar_parity import _safe_label, run_parity_check  # noqa: E402


def main() -> int:
    matchups = [
        ("Ira", "Ira"),
        ("FaiSAGEPrecon", "FaiSAGEPrecon"),
        ("DorintheaSAGEPrecon", "DorintheaSAGEPrecon"),
        ("BriarSAGEPrecon", "BriarSAGEPrecon"),
        ("KayoSAGEPrecon", "KayoSAGEPrecon"),
        ("BlazeSAGEPrecon", "BlazeSAGEPrecon"),
        ("DorintheaSAGEPrecon", "FaiSAGEPrecon"),
        ("Ira", "FaiSAGEPrecon"),
    ]
    print(f"{'Matchup':<45} {'Disc':>6} {'Step':>5} {'Taxonomy':<20} First failure")
    print("-" * 120)
    failed = 0
    for d1, d2 in matchups:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            report, code = run_parity_check(
                deck1=d1,
                deck2=d2,
                episodes=2,
                mode="multi-step",
                steps_per_episode=20,
                parity_mode="simulation",
                sync_scope="full",
                rng_seed=42,
                stop_after_failure=False,
                write_reports=True,
                verbose=False,
            )
        disc_count = len(report.discrepancies)
        first_step = ""
        first_tax = ""
        first_desc = "PASS"
        if report.discrepancies:
            failed += 1
            d0 = report.discrepancies[0]
            if isinstance(d0, dict):
                first_step = str(d0.get("step", ""))
                first_tax = str(d0.get("taxonomy") or d0.get("category") or "")
                first_desc = str(d0.get("description") or "")[:70]
            else:
                first_step = str(getattr(d0, "step", ""))
                first_tax = str(getattr(d0, "taxonomy", None) or getattr(d0, "category", ""))
                first_desc = str(getattr(d0, "description", "") or "")[:70]
        print(
            f"{d1} vs {d2:<25} {disc_count:>6} {first_step:>5} {first_tax:<20} {first_desc}"
        )
    print("-" * 120)
    print(f"Simulation: {len(matchups) - failed}/{len(matchups)} passed (0 discrepancies)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
