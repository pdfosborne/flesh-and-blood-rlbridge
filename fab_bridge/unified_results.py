"""Helpers for unified random matchup experiment result directories."""

from __future__ import annotations

import json
from pathlib import Path

RUN_MANIFEST = "run_manifest.json"
CHECKPOINT_EVAL_SCOPE = "checkpoint_eval_scope.json"
UNIFIED_SELFPLAY = "unified_selfplay"
MATCHUP_LABEL = "matchup_label.json"

_SKIP_RUN_CHILD_NAMES = frozenset(
    {
        RUN_MANIFEST,
        "training_summary.json",
        "checkpoint_eval_history.json",
        CHECKPOINT_EVAL_SCOPE,
        "eval_dashboard",
    }
)


def is_unified_random_matchup_run(path: Path) -> bool:
    """True when *path* is a ``train_unified_random_matchups.py`` run folder."""
    return (path / RUN_MANIFEST).is_file()


def read_checkpoint_eval_scope(run_dir: Path) -> dict:
    path = run_dir / CHECKPOINT_EVAL_SCOPE
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _looks_like_matchup_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name in _SKIP_RUN_CHILD_NAMES:
        return False
    if (path / MATCHUP_LABEL).is_file():
        return True
    if (path / "checkpoint_eval").is_dir():
        return True
    return any(path.glob(f"{UNIFIED_SELFPLAY}/p1/episode_*/metadata.json"))


def iter_unified_matchup_dirs(run_dir: Path) -> list[Path]:
    """Return matchup output folders under a unified run, newest first."""
    if not run_dir.is_dir():
        return []
    dirs = [child for child in run_dir.iterdir() if _looks_like_matchup_dir(child)]
    return sorted(dirs, key=lambda path: path.stat().st_mtime, reverse=True)


def resolve_latest_unified_matchup_dir(run_dir: Path) -> Path | None:
    """Resolve the active/latest matchup folder for checkpoint evaluation."""
    scope = read_checkpoint_eval_scope(run_dir)
    subdir = str(scope.get("matchup_dir") or "").strip()
    if subdir:
        candidate = run_dir / subdir
        if candidate.is_dir():
            return candidate
    dirs = iter_unified_matchup_dirs(run_dir)
    return dirs[0] if dirs else None


def has_unified_selfplay_checkpoints(run_dir: Path) -> bool:
    matchup_dir = resolve_latest_unified_matchup_dir(run_dir)
    if matchup_dir is None:
        return False
    return any(matchup_dir.glob(f"{UNIFIED_SELFPLAY}/p1/episode_*/metadata.json"))


def iter_unified_checkpoint_metadata(run_dir: Path, role: str) -> list[Path]:
    matchup_dir = resolve_latest_unified_matchup_dir(run_dir)
    if matchup_dir is None:
        return []
    return sorted(
        matchup_dir.glob(f"{UNIFIED_SELFPLAY}/{role}/episode_*/metadata.json"),
        key=lambda path: path.parent.name,
    )


def unified_run_label(run_dir: Path) -> str:
    manifest_path = run_dir / RUN_MANIFEST
    if not manifest_path.is_file():
        return run_dir.name
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return run_dir.name
    fmt = str(data.get("format") or "").strip()
    matchups = data.get("matchups_sampled") or []
    count = len(matchups) if isinstance(matchups, list) else 0
    latest = resolve_latest_unified_matchup_dir(run_dir)
    latest_name = ""
    if latest is not None:
        label_path = latest / MATCHUP_LABEL
        if label_path.is_file():
            try:
                label = json.loads(label_path.read_text(encoding="utf-8"))
                latest_name = str(label.get("name") or "").strip()
            except (json.JSONDecodeError, OSError, TypeError):
                latest_name = latest.name
        else:
            latest_name = latest.name
    parts = []
    if fmt:
        parts.append(fmt.replace("_", " "))
    if count:
        parts.append(f"{count} matchup(s)")
    if latest_name:
        parts.append(f"latest: {latest_name}")
    return " · ".join(parts) if parts else run_dir.name
