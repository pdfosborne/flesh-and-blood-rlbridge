"""First-time setup: verify layout, create dirs, start Talishar via Docker."""

from __future__ import annotations

import sys
from pathlib import Path

from fab_bridge.doctor import print_report, run_doctor
from fab_bridge.paths import configure_import_paths, repo_root, talishar_assets_dir, talishar_dir

configure_import_paths()


def run_init(*, start_talishar: bool = True, backend_only: bool = True, sync_agents: bool = True) -> int:
    root = repo_root()
    print(f"FAB RL Bridge root: {root}")

    for sub in ("results", "results/agent_cache"):
        path = root / sub
        try:
            path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print(
                f"  WARNING: cannot create {path} (permission denied).",
                file=sys.stderr,
            )
            print(
                f"  Run: sudo chown -R \"$USER:$USER\" {root / 'results'}",
                file=sys.stderr,
            )
            break
        print(f"  ensured {path}")

    if sync_agents:
        print()
        print("Ensuring unified agent weights (sync + bootstrap fallback)…")
        from fab_bridge.agents import agent_cache_dir, default_manifest_url, ensure_agents_available  # noqa: PLC0415

        try:
            results = ensure_agents_available(
                manifest_url=default_manifest_url(),
                cache_dir=agent_cache_dir(),
            )
            for row in results:
                print(f"  [{row.action}] {row.format}: {row.detail}")
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: agent ensure skipped ({exc})")
            print("  Run later: fab-bridge agents ensure")

    assets = talishar_assets_dir()
    if not assets.is_dir() or not any(assets.glob("*.txt")):
        print()
        print("Talishar Assets are missing or empty.")
        print("Clone this repository with submodules / subtrees, or set TALISHAR_ASSETS_PATH.")
        print(f"  expected: {assets}")
        print()
        print("  git clone https://github.com/pdfosborne/flesh-and-blood-rlbridge.git")
        print("  # then populate Talishar/ (see README)")
        return 1

    if not talishar_dir().is_dir():
        print(f"ERROR: Talishar directory not found: {talishar_dir()}", file=sys.stderr)
        return 1

    report = run_doctor(require_docker=start_talishar)
    if not report.ok and start_talishar:
        print_report(report)
        return 1

    if start_talishar:
        print()
        print("Starting Talishar backend (Docker Compose)…")
        from start_talishar import run as start_talishar_run  # noqa: PLC0415

        rc = start_talishar_run(backend_only=backend_only, fe_only=False, down=False)
        if rc != 0:
            return rc

    print()
    print("Setup complete. Launch the GUI with:")
    print("  fab-gui")
    print("or the terminal UI with:")
    print("  fab-tui")
    return 0
