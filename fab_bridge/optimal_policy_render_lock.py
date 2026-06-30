"""Cross-process lock so only one optimal-policy live render companion runs per run dir."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

OPTIMAL_POLICY_RENDER_LOCK = "optimal_policy_render_lock.json"
_DEFAULT_STALE_SECONDS = 4 * 60 * 60


def optimal_policy_render_lock_path(run_dir: Path) -> Path:
    return run_dir.expanduser().resolve() / OPTIMAL_POLICY_RENDER_LOCK


def _read_lock(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_lock(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _parse_updated_at(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _lock_is_stale(
    data: dict[str, Any],
    *,
    stale_seconds: float = _DEFAULT_STALE_SECONDS,
) -> bool:
    if not data.get("active"):
        return True
    pid = int(data.get("pid") or 0)
    if pid == os.getpid():
        return False
    if not _pid_alive(pid):
        return True
    updated = _parse_updated_at(data.get("updated_at"))
    if updated is not None:
        age = (datetime.now(timezone.utc) - updated.astimezone(timezone.utc)).total_seconds()
        if age > max(60.0, float(stale_seconds)):
            return True
    return False


def try_acquire_optimal_policy_render_lock(
    run_dir: Path,
    *,
    stale_seconds: float = _DEFAULT_STALE_SECONDS,
) -> bool:
    """Claim the live-render companion slot for *run_dir*. Returns False if another holder is active."""
    path = optimal_policy_render_lock_path(run_dir)
    deadline = time.monotonic() + 0.25
    while True:
        existing = _read_lock(path)
        if existing.get("active") and not _lock_is_stale(existing, stale_seconds=stale_seconds):
            holder = int(existing.get("pid") or 0)
            if holder != os.getpid():
                return False
        _write_lock(
            path,
            {
                "active": True,
                "pid": os.getpid(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        verify = _read_lock(path)
        if int(verify.get("pid") or 0) == os.getpid() and verify.get("active"):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def release_optimal_policy_render_lock(run_dir: Path) -> None:
    """Release the live-render companion slot when held by this process."""
    path = optimal_policy_render_lock_path(run_dir)
    data = _read_lock(path)
    if int(data.get("pid") or 0) not in (0, os.getpid()):
        return
    _write_lock(
        path,
        {
            "active": False,
            "pid": os.getpid(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def optimal_policy_render_lock_holder(run_dir: Path) -> Optional[int]:
    """Return the PID holding the live-render lock, or None if unclaimed/stale."""
    data = _read_lock(optimal_policy_render_lock_path(run_dir))
    if not data.get("active") or _lock_is_stale(data):
        return None
    pid = int(data.get("pid") or 0)
    return pid if pid > 0 else None
