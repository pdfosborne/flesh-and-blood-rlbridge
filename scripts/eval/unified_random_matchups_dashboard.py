#!/usr/bin/env python3
"""Live HTML dashboard for unified random matchup training.

Typical usage alongside training::

    python scripts/eval/unified_random_matchups_dashboard.py \\
        --out-dir results/unified_random_matchups/silver_age/20260626_215125 \\
        --watch --poll-seconds 5 --open-browser
"""

from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_ROOT))
import _bootstrap  # noqa: E402

_bootstrap.configure_paths()

from fab_bridge.unified_dashboard import (  # noqa: E402
    UNIFIED_DASHBOARD_NAME,
    collect_unified_run_state,
    write_unified_random_matchups_dashboard,
)
from fab_bridge.unified_results import is_unified_random_matchup_run, resolve_unified_run_root  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live HTML dashboard for unified random matchup training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Unified run directory (contains run_manifest.json)",
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument(
        "--no-auto-refresh",
        action="store_true",
        help="Disable the HTML meta refresh tag",
    )
    args = parser.parse_args()

    out_dir = resolve_unified_run_root(Path(args.out_dir))
    if not is_unified_random_matchup_run(out_dir):
        raise SystemExit(f"Not a unified random matchups run: {out_dir}")

    refresh = None if args.no_auto_refresh else args.poll_seconds
    opened = False

    print("=" * 62)
    print("  Unified Random Matchups Dashboard")
    print("=" * 62)
    print(f"  Watching : {out_dir}")
    print(f"  Output   : {out_dir / UNIFIED_DASHBOARD_NAME}")
    print("=" * 62)

    while True:
        html_path = write_unified_random_matchups_dashboard(
            out_dir,
            auto_refresh_seconds=refresh,
        )
        state = collect_unified_run_state(out_dir)
        if args.open_browser and not opened and html_path is not None:
            webbrowser.open(html_path.resolve().as_uri())
            opened = True
        if not args.watch or state.get("complete"):
            break
        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    main()
