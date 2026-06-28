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


def resolve_unified_run_root(path: Path) -> Path:
    """Normalize *path* to a unified run root (accepts matchup subfolders).

    Walks upward until ``run_manifest.json`` is found. Returns *path* unchanged
    when it is not inside a unified run.
    """
    resolved = path.expanduser().resolve()
    current = resolved
    for _ in range(6):
        if is_unified_random_matchup_run(current):
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return resolved


def read_checkpoint_eval_scope(run_dir: Path) -> dict:
    path = run_dir / CHECKPOINT_EVAL_SCOPE
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def active_merged_matchup_dirs(run_dir: Path) -> list[Path]:
    """Matchup folders in the current merged training batch."""
    scope = read_checkpoint_eval_scope(run_dir)
    matchups = scope.get("matchups")
    if isinstance(matchups, list) and matchups:
        dirs: list[Path] = []
        for row in matchups:
            if not isinstance(row, dict):
                continue
            subdir = str(row.get("matchup_dir") or "").strip()
            if not subdir:
                continue
            candidate = run_dir / subdir
            if candidate.is_dir():
                dirs.append(candidate)
        if dirs:
            return dirs
    single = str(scope.get("matchup_dir") or "").strip()
    if single:
        candidate = run_dir / single
        if candidate.is_dir():
            return [candidate]
    return iter_unified_matchup_dirs(run_dir)


def find_latest_merged_unified_bucket(
    run_dir: Path,
    *,
    matchup_dirs: list[Path] | None = None,
) -> tuple[int, dict[Path, Path]] | None:
    """Return ``(episode, {matchup_dir: p1_checkpoint_dir})`` when all matchups align."""
    dirs = list(matchup_dirs or active_merged_matchup_dirs(run_dir))
    if not dirs:
        return None
    per_dir_episodes: dict[Path, set[int]] = {}
    per_dir_ckpts: dict[Path, dict[int, Path]] = {}
    for matchup_dir in dirs:
        episodes: set[int] = set()
        ckpts: dict[int, Path] = {}
        for meta_path in matchup_dir.glob("unified_selfplay/p1/episode_*/metadata.json"):
            episode_name = meta_path.parent.name.removeprefix("episode_")
            try:
                episode = int(episode_name)
            except ValueError:
                continue
            episodes.add(episode)
            ckpts[episode] = meta_path.parent
        if not episodes:
            return None
        per_dir_episodes[matchup_dir] = episodes
        per_dir_ckpts[matchup_dir] = ckpts
    common = set.intersection(*per_dir_episodes.values())
    if not common:
        return None
    best_episode = max(common)
    return best_episode, {
        matchup_dir: per_dir_ckpts[matchup_dir][best_episode]
        for matchup_dir in dirs
    }


def _episode_dir_sort_key(meta_path: Path) -> tuple[int, float]:
    episode_name = meta_path.parent.name.removeprefix("episode_")
    try:
        episode = int(episode_name)
    except ValueError:
        episode = 0
    try:
        mtime = meta_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return episode, mtime


def matchup_dir_from_unified_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the matchup output folder containing a unified self-play checkpoint."""
    parts = checkpoint_dir.resolve().parts
    if UNIFIED_SELFPLAY not in parts:
        return None
    idx = parts.index(UNIFIED_SELFPLAY)
    if idx < 1:
        return None
    return Path(*parts[:idx])


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
    """Resolve the active/latest matchup folder for in-run checkpoint watching."""
    scope = read_checkpoint_eval_scope(run_dir)
    subdir = str(scope.get("matchup_dir") or "").strip()
    if subdir:
        candidate = run_dir / subdir
        if candidate.is_dir():
            return candidate
    dirs = iter_unified_matchup_dirs(run_dir)
    return dirs[0] if dirs else None


def _glob_unified_checkpoint_metadata(
    run_dir: Path,
    role: str,
    *,
    matchup_dir: Path | None = None,
) -> list[Path]:
    if matchup_dir is not None:
        search_roots = [matchup_dir]
    else:
        search_roots = iter_unified_matchup_dirs(run_dir)
    paths: list[Path] = []
    for root in search_roots:
        paths.extend(root.glob(f"{UNIFIED_SELFPLAY}/{role}/episode_*/metadata.json"))
    return sorted(paths, key=_episode_dir_sort_key)


def has_unified_selfplay_checkpoints(run_dir: Path) -> bool:
    return bool(_glob_unified_checkpoint_metadata(run_dir, "p1"))


def iter_unified_checkpoint_metadata(
    run_dir: Path,
    role: str,
    *,
    matchup_dir: Path | None = None,
) -> list[Path]:
    """Collect unified self-play checkpoint metadata paths under a run.

    When *matchup_dir* is set, only that matchup folder is searched (watch mode
    during training). Otherwise every matchup under the run is scanned so the
    globally latest checkpoint can be resolved from the run root.
    """
    return _glob_unified_checkpoint_metadata(run_dir, role, matchup_dir=matchup_dir)


def find_latest_unified_checkpoint_metadata(
    run_dir: Path,
    role: str = "p1",
    *,
    matchup_dir: Path | None = None,
) -> Path | None:
    """Return metadata path for the newest unified checkpoint in *run_dir*."""
    paths = iter_unified_checkpoint_metadata(
        run_dir,
        role,
        matchup_dir=matchup_dir,
    )
    return paths[-1] if paths else None


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
    latest_meta = find_latest_unified_checkpoint_metadata(run_dir, "p1")
    latest_name = ""
    if latest_meta is not None:
        latest_matchup = matchup_dir_from_unified_checkpoint(latest_meta.parent)
        if latest_matchup is not None:
            label_path = latest_matchup / MATCHUP_LABEL
            if label_path.is_file():
                try:
                    label = json.loads(label_path.read_text(encoding="utf-8"))
                    latest_name = str(label.get("name") or "").strip()
                except (json.JSONDecodeError, OSError, TypeError):
                    latest_name = latest_matchup.name
            else:
                latest_name = latest_matchup.name
    parts = []
    if fmt:
        parts.append(fmt.replace("_", " "))
    if count:
        parts.append(f"{count} matchup(s)")
    if latest_name:
        parts.append(f"latest ckpt: {latest_name}")
    return " · ".join(parts) if parts else run_dir.name
