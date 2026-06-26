"""Check that FAB RL Bridge dependencies and assets are available."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from fab_bridge.agents import agent_cache_dir, agent_status, load_manifest, gh_auth_ok
from fab_bridge.paths import repo_root, talishar_assets_dir, talishar_dir


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


@dataclass
class DoctorReport:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results if r.required)

    def add(self, name: str, ok: bool, detail: str, *, required: bool = True) -> None:
        self.results.append(CheckResult(name=name, ok=ok, detail=detail, required=required))


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _command_version(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or proc.stderr or "").strip().splitlines()[0]


def run_doctor(*, require_docker: bool = True) -> DoctorReport:
    report = DoctorReport()
    root = repo_root()

    py_ok = sys.version_info >= (3, 10)
    report.add(
        "Python",
        py_ok,
        f"{sys.version.split()[0]} ({sys.executable})",
    )

    for mod in ("numpy", "rich", "rlbridge", "fab_tui", "fab_gui"):
        report.add(f"import:{mod}", _module_available(mod), mod)

    report.add(
        "import:torch",
        _module_available("torch"),
        "required for unified agent inference",
        required=False,
    )

    assets = talishar_assets_dir()
    assets_ok = assets.is_dir() and any(assets.glob("*.txt"))
    report.add(
        "Talishar Assets",
        assets_ok,
        str(assets) if assets_ok else f"Missing or empty: {assets}",
    )

    talishar_ok = talishar_dir().is_dir()
    report.add(
        "Talishar source",
        talishar_ok,
        str(talishar_dir()) if talishar_ok else f"Not found: {talishar_dir()}",
    )

    docker = shutil.which("docker")
    docker_ok = False
    docker_detail = "docker not on PATH"
    if docker:
        version = _command_version(["docker", "version", "--format", "{{.Server.Version}}"])
        docker_ok = version is not None
        docker_detail = version or "docker found but daemon not reachable"
    report.add("Docker", docker_ok, docker_detail, required=require_docker)

    cards_db = root / "src" / "flesh_and_blood_rlbridge" / "card_db" / "cards.json"
    report.add(
        "Card database",
        cards_db.is_file(),
        str(cards_db) if cards_db.is_file() else f"Missing: {cards_db}",
    )

    try:
        manifest = load_manifest()
        default_fmt = str(manifest.get("default_format", "silver_age"))
        status = agent_status(agent_cache_dir(), default_fmt)
        agent_ok = bool(status.get("exists"))
        release = status.get("release_id") or "not installed"
        manifest_has_entry = any(
            isinstance(row, dict) and str(row.get("format", "")) == status.get("cache_format")
            for row in manifest.get("agents", [])
        )
        report.add(
            f"Unified agent ({default_fmt})",
            agent_ok,
            f"{status.get('weights_path')} — release: {release}"
            if agent_ok
            else (
                f"Missing — run: fab-bridge agents sync"
                if manifest_has_entry
                else "No published agent in manifest yet"
            ),
            required=require_docker and manifest_has_entry,
        )
    except Exception as exc:  # noqa: BLE001
        report.add(
            "Unified agent",
            False,
            f"Could not check agent cache: {exc}",
            required=require_docker,
        )

    gh_ok, gh_detail = gh_auth_ok()
    report.add(
        "GitHub CLI (optional)",
        gh_ok,
        gh_detail + " — needed for fab-bridge agents publish",
        required=False,
    )

    cmake = shutil.which("cmake")
    report.add(
        "CMake (optional)",
        cmake is not None,
        cmake or "not installed — C++ engine builds will be skipped",
        required=False,
    )

    node = shutil.which("node")
    report.add(
        "Node.js (optional)",
        node is not None,
        _command_version(["node", "--version"]) or "not installed — live play / GIF FE unavailable",
        required=False,
    )

    return report


def print_report(report: DoctorReport) -> int:
    print("FAB RL Bridge — environment check")
    print("=" * 50)
    for item in report.results:
        tag = "OK" if item.ok else ("WARN" if not item.required else "FAIL")
        suffix = "" if item.required else " (optional)"
        print(f"  [{tag:4}] {item.name}{suffix}: {item.detail}")
    print("=" * 50)
    if report.ok:
        print("All required checks passed.")
        return 0
    print("Some required checks failed. Run: fab-bridge init")
    return 1
