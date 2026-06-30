"""Cross-process lock so live optimal-policy render yields the eval Talishar shard."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

EVAL_SHARD_LOCK = "eval_shard_lock.json"
_DEFAULT_STALE_SECONDS = 4 * 60 * 60


def eval_shard_lock_path(run_dir: Path) -> Path:
    return run_dir.expanduser().resolve() / EVAL_SHARD_LOCK


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


def is_eval_shard_busy(
    run_dir: Path,
    *,
    stale_seconds: float = _DEFAULT_STALE_SECONDS,
) -> bool:
    """True when checkpoint eval holds the dedicated eval backend."""
    path = eval_shard_lock_path(run_dir)
    data = _read_lock(path)
    if not data.get("busy"):
        return False
    updated = _parse_updated_at(data.get("updated_at"))
    if updated is not None:
        age = (datetime.now(timezone.utc) - updated.astimezone(timezone.utc)).total_seconds()
        if age > max(60.0, float(stale_seconds)):
            return False
    return True


def acquire_eval_shard_lock(
    run_dir: Path,
    owner: str,
    *,
    label: str = "",
) -> None:
    path = eval_shard_lock_path(run_dir)
    payload = {
        "busy": True,
        "owner": str(owner),
        "label": str(label or ""),
        "pid": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_lock(path, payload)


def release_eval_shard_lock(run_dir: Path, owner: str) -> None:
    path = eval_shard_lock_path(run_dir)
    data = _read_lock(path)
    if data.get("owner") not in (owner, ""):
        return
    _write_lock(
        path,
        {
            "busy": False,
            "owner": str(owner),
            "label": "",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


@contextmanager
def hold_eval_shard(
    run_dir: Path,
    owner: str,
    *,
    label: str = "",
) -> Iterator[None]:
    """Mark the eval shard busy for the duration of a checkpoint-eval job."""
    acquire_eval_shard_lock(run_dir, owner, label=label)
    try:
        yield
    finally:
        release_eval_shard_lock(run_dir, owner)


def wait_for_eval_shard_idle(
    run_dir: Path,
    *,
    poll_seconds: float = 0.5,
    stop: Optional[Any] = None,
) -> bool:
    """Block until the eval shard is free. Returns False if *stop* is set."""
    while is_eval_shard_busy(run_dir):
        if stop is not None and getattr(stop, "is_set", lambda: False)():
            return False
        time.sleep(max(0.1, float(poll_seconds)))
    return True
