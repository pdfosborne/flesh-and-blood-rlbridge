#!/usr/bin/env python3
"""Start the Talishar backend (Docker Compose) and Talishar-FE Vite dev server.

Usage:
    python start_talishar.py              # start everything
    python start_talishar.py --backend-only
    python start_talishar.py --fe-only
    python start_talishar.py --down
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from fab_bridge.paths import configure_import_paths, repo_root

configure_import_paths()
REPO_ROOT = repo_root()
TALISHAR_DIR = REPO_ROOT / "Talishar"
FE_DIR = REPO_ROOT / "Talishar-FE"
BACKEND_URL = "http://localhost:8080"
FE_URL = "http://localhost:5173"
PHPMYADMIN_URL = "http://localhost:5001"


def _header(msg: str) -> None:
    print(f"\n=== {msg} ===")


def _reachable(url: str, timeout: float = 2) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return int(resp.status) < 500
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _wait_for(url: str, seconds: int, label: str) -> bool:
    print(f"  Waiting for {label}...", end="", flush=True)
    for _ in range(seconds):
        if _reachable(url):
            print()
            return True
        time.sleep(1)
        print(".", end="", flush=True)
    print()
    return False


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)  # noqa: S603


def _start_vite_detached(fe_dir: Path) -> None:
    if sys.platform == "win32":
        subprocess.Popen(  # noqa: S603
            ["npm", "run", "dev"],
            cwd=fe_dir,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
    else:
        subprocess.Popen(  # noqa: S603
            ["npm", "run", "dev"],
            cwd=fe_dir,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def run(
    *,
    backend_only: bool = False,
    fe_only: bool = False,
    down: bool = False,
) -> int:
    if down:
        _header("Stopping Talishar backend containers")
        _run(["docker", "compose", "down"], cwd=TALISHAR_DIR)
        print("Backend stopped.")
        return 0

    if not fe_only:
        _header("Starting Talishar backend (Docker Compose)")
        if not TALISHAR_DIR.is_dir():
            print(f"ERROR: Talishar directory not found: {TALISHAR_DIR}", file=sys.stderr)
            return 1

        try:
            _run(["docker", "compose", "up", "-d", "--build"], cwd=TALISHAR_DIR)
        except subprocess.CalledProcessError as exc:
            print(f"ERROR: docker compose up failed (exit {exc.returncode})", file=sys.stderr)
            return exc.returncode or 1

        print("Backend containers started.")
        print(f"  API / game engine : {BACKEND_URL}")
        print(f"  phpMyAdmin        : {PHPMYADMIN_URL}")

        if _wait_for(BACKEND_URL + "/", 30, "backend to become reachable"):
            print("  Backend is up.")
        else:
            print("  Backend did not respond within 30s - containers may still be initialising.")

    if not backend_only:
        _header("Starting Talishar-FE (Vite dev server)")
        if not FE_DIR.is_dir():
            print(f"ERROR: Talishar-FE directory not found: {FE_DIR}", file=sys.stderr)
            return 1

        if not (FE_DIR / "node_modules").is_dir():
            print("  node_modules not found - running npm install...")
            _run(["npm", "install"], cwd=FE_DIR)

        if _reachable(FE_URL):
            print(f"  Vite dev server already running at {FE_URL}")
        else:
            _start_vite_detached(FE_DIR)
            if sys.platform == "win32":
                print("  Vite dev server launched in a new window.")
            else:
                print("  Vite dev server launched in the background.")
            print(f"  Frontend URL : {FE_URL}")

            if _wait_for(FE_URL, 20, "Vite to start"):
                print("  Frontend is up.")
            else:
                print("  Frontend did not respond yet - it may need a few more seconds.")

    _header("Talishar stack ready")
    if not fe_only:
        print(f"  Backend  : {BACKEND_URL}")
    if not backend_only:
        print(f"  Frontend : {FE_URL}")

    print()
    print("Set these env vars before training:")
    print('  export TALISHAR_URL="http://localhost:8080"')
    print('  export TALISHAR_FE_URL="http://localhost:5173"')
    print()
    print("To stop the backend containers:  python start_talishar.py --down")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start or stop the Talishar stack.")
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="skip the FE dev server",
    )
    parser.add_argument(
        "--fe-only",
        action="store_true",
        help="skip Docker (backend already up)",
    )
    parser.add_argument(
        "--down",
        action="store_true",
        help="stop backend containers",
    )
    args = parser.parse_args(argv)

    if sum([args.backend_only, args.fe_only, args.down]) > 1:
        parser.error("use at most one of --backend-only, --fe-only, or --down")

    return run(backend_only=args.backend_only, fe_only=args.fe_only, down=args.down)


if __name__ == "__main__":
    raise SystemExit(main())
