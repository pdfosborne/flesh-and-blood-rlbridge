"""Discover saved training results with evaluable phase-3 checkpoints."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fab_tui.config import REPO_ROOT, RESULTS_ROOT

RESULT_CATEGORY_ROOTS: tuple[tuple[str, Path], ...] = (
    ("experiments", RESULTS_ROOT / "experiments"),
    ("matchup_sims", RESULTS_ROOT / "matchup_sims"),
    ("full_pipeline", RESULTS_ROOT / "full_pipeline"),
    ("sideboard_compare", RESULTS_ROOT / "sideboard_compare"),
)

_RUN_STAMP_SUFFIX = re.compile(r"_(\d{8}_\d{6})$")


@dataclass(frozen=True)
class EvaluableResultsEntry:
    path: Path
    category: str
    label: str
    run_started: str
    run_stamp: str
    run_count: int
    latest_episode: str | None
    mtime: float

    @property
    def display_path(self) -> str:
        try:
            return str(self.path.relative_to(REPO_ROOT))
        except ValueError:
            return str(self.path)

    @property
    def checkpoints_summary(self) -> str:
        if self.latest_episode:
            ep = self.latest_episode.removeprefix("episode_")
            if self.run_count > 1:
                return f"{self.run_count} runs, latest ep {ep}"
            return f"latest ep {ep}"
        if self.run_count:
            return f"{self.run_count} run(s)"
        return "—"


@dataclass(frozen=True)
class CompletedTrainingEntry(EvaluableResultsEntry):
    """A finished training run that still has evaluable phase-3 checkpoints."""

    status_summary: str = "—"

    @property
    def summary(self) -> str:
        parts = [self.status_summary, self.checkpoints_summary]
        return " · ".join(part for part in parts if part and part != "—")


def _parse_run_stamp_from_path(path: Path) -> datetime | None:
    """Parse trailing ``_YYYYMMDD_HHMMSS`` from a results folder name."""
    match = _RUN_STAMP_SUFFIX.search(path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _run_started_from_manifest(path: Path) -> datetime | None:
    manifest_path = path / "candidates_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw = data.get("started_at")
        if not raw:
            return None
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return None


def _run_time_labels(path: Path) -> tuple[str, str]:
    """Return ``(display_datetime, stamp_slug)`` for a results directory."""
    dt = _parse_run_stamp_from_path(path)
    stamp = ""
    match = _RUN_STAMP_SUFFIX.search(path.name)
    if match:
        stamp = match.group(1)

    if dt is None:
        dt = _run_started_from_manifest(path)
        if dt is not None and not stamp:
            stamp = dt.strftime("%Y%m%d_%H%M%S")

    if dt is None:
        try:
            dt = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            return "—", stamp or path.name

    if not stamp:
        stamp = dt.strftime("%Y%m%d_%H%M%S")
    return dt.strftime("%Y-%m-%d %H:%M"), stamp


def _is_sideboard_compare_dir(path: Path) -> bool:
    return (
        (path / "candidates_manifest.json").is_file()
        and (path / "candidates").is_dir()
    )


def _has_phase3_checkpoints(path: Path) -> bool:
    if _is_sideboard_compare_dir(path):
        patterns = (
            "candidates/*/p3_*/p1/episode_*/metadata.json",
            "candidates/*/parallel_seeds/seed_*/p3_*/p1/episode_*/metadata.json",
        )
        return any(path.glob(pattern) for pattern in patterns)
    patterns = (
        "p3_*/p1/episode_*/metadata.json",
        "parallel_seeds/seed_*/p3_*/p1/episode_*/metadata.json",
    )
    return any(path.glob(pattern) for pattern in patterns)


def _checkpoint_summary(path: Path) -> tuple[int, str | None]:
    if _is_sideboard_compare_dir(path):
        p3_dirs = [
            p
            for pattern in (
                "candidates/*/p3_*",
                "candidates/*/parallel_seeds/seed_*/p3_*",
            )
            for p in path.glob(pattern)
            if p.is_dir()
        ]
        meta_globs = (
            "candidates/*/p3_*/p1/episode_*/metadata.json",
            "candidates/*/parallel_seeds/seed_*/p3_*/p1/episode_*/metadata.json",
        )
    else:
        p3_dirs = [
            p
            for pattern in ("p3_*", "parallel_seeds/seed_*/p3_*")
            for p in path.glob(pattern)
            if p.is_dir()
        ]
        meta_globs = (
            "p3_*/p1/episode_*/metadata.json",
            "parallel_seeds/seed_*/p3_*/p1/episode_*/metadata.json",
        )
    latest_episode: str | None = None
    meta_paths = [meta for glob in meta_globs for meta in path.glob(glob)]
    for meta in sorted(
        meta_paths,
        key=lambda p: p.parent.name,
        reverse=True,
    ):
        latest_episode = meta.parent.name
        break
    return len(p3_dirs), latest_episode


def _label_for_results_dir(path: Path) -> str:
    manifest_path = path / "candidates_manifest.json"
    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            hero = str(data.get("hero_id", "") or "").strip()
            opponent = str(data.get("opponent_hero_id", "") or "").strip()
            opp_deck = str(data.get("opponent_deck", "") or "").strip()
            if hero and opponent:
                deck = f" · {opp_deck}" if opp_deck else ""
                return f"{hero} vs {opponent}{deck}"
        except (json.JSONDecodeError, OSError):
            pass
    results_json = path / "results.json"
    if results_json.is_file():
        try:
            data = json.loads(results_json.read_text(encoding="utf-8"))
            p1_deck = str(data.get("p1", {}).get("deck_asset_name") or "").strip()
            p2_deck = str(data.get("p2", {}).get("deck_asset_name") or "").strip()
            if p1_deck and p2_deck:
                return f"{p1_deck} vs {p2_deck}"
        except (json.JSONDecodeError, OSError):
            pass
    return path.name


def _read_json(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _training_target_reached(live_path: Path) -> bool:
    live = _read_json(live_path)
    if not live:
        return False
    completed = int(live.get("episodes_completed", 0) or 0)
    target = int(live.get("target_episodes", 0) or 0)
    return target > 0 and completed >= target


def _is_training_complete(path: Path) -> bool:
    """True when a run looks finished (not an in-progress checkpoint watcher target)."""
    if (path / "sideboard_compare_results.json").is_file():
        return True
    if (path / "results.json").is_file():
        return True
    if any(path.glob("candidates/*/candidate_result.json")):
        return True
    live_paths = list(path.glob("play_training_live.json"))
    live_paths.extend(path.glob("candidates/*/play_training_live.json"))
    live_paths.extend(path.glob("candidates/*/parallel_seeds/seed_*/play_training_live.json"))
    return any(_training_target_reached(live_path) for live_path in live_paths)


def _format_win_rate(value: object) -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return None


def _completion_summary(path: Path) -> str:
    summary_path = path / "sideboard_compare_results.json"
    if summary_path.is_file():
        data = _read_json(summary_path)
        winner = data.get("winner") or {}
        if isinstance(winner, dict):
            cid = str(winner.get("candidate_id") or "winner")
            wr = _format_win_rate(
                winner.get("final_eval_win_rate", winner.get("play_win_rate"))
            )
            if wr:
                return f"complete · {cid} @ {wr}"
            return f"complete · {cid}"
        return "complete"

    results_path = path / "results.json"
    if results_path.is_file():
        data = _read_json(results_path)
        p1 = data.get("p1") or {}
        if isinstance(p1, dict):
            final = p1.get("final_eval")
            if isinstance(final, dict):
                wr = _format_win_rate(final.get("win_rate"))
                if wr:
                    return f"complete · P1 final {wr}"
        return "complete"

    best_candidate: tuple[float, str] | None = None
    for result_path in path.glob("candidates/*/candidate_result.json"):
        row = _read_json(result_path)
        if not row:
            continue
        cid = str(row.get("candidate_id") or result_path.parent.name)
        for key in ("final_eval_win_rate", "latest_checkpoint_eval_win_rate", "play_win_rate"):
            wr = row.get(key)
            if wr is None:
                continue
            try:
                score = float(wr)
            except (TypeError, ValueError):
                continue
            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (score, cid)
            break

    if best_candidate is not None:
        score, cid = best_candidate
        return f"trained · {cid} @ {score:.0%}"

    return "trained"


def discover_evaluable_results(*, limit: int = 25) -> list[EvaluableResultsEntry]:
    """Return result folders that contain phase-3 play checkpoints, newest first."""
    entries: list[EvaluableResultsEntry] = []
    seen: set[Path] = set()

    for category, root in RESULT_CATEGORY_ROOTS:
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if not path.is_dir():
                continue
            resolved = path.resolve()
            if resolved in seen or not _has_phase3_checkpoints(path):
                continue
            seen.add(resolved)
            run_count, latest_episode = _checkpoint_summary(path)
            run_started, run_stamp = _run_time_labels(path)
            entries.append(
                EvaluableResultsEntry(
                    path=resolved,
                    category=category,
                    label=_label_for_results_dir(path),
                    run_started=run_started,
                    run_stamp=run_stamp,
                    run_count=run_count,
                    latest_episode=latest_episode,
                    mtime=path.stat().st_mtime,
                )
            )

    entries.sort(key=lambda entry: entry.mtime, reverse=True)
    return entries[:limit]


def discover_completed_training_runs(*, limit: int = 25) -> list[CompletedTrainingEntry]:
    """Return finished training runs with checkpoints, newest first."""
    entries: list[CompletedTrainingEntry] = []
    seen: set[Path] = set()

    for category, root in RESULT_CATEGORY_ROOTS:
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if not path.is_dir():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            if not _has_phase3_checkpoints(path) or not _is_training_complete(path):
                continue
            seen.add(resolved)
            run_count, latest_episode = _checkpoint_summary(path)
            run_started, run_stamp = _run_time_labels(path)
            entries.append(
                CompletedTrainingEntry(
                    path=resolved,
                    category=category,
                    label=_label_for_results_dir(path),
                    run_started=run_started,
                    run_stamp=run_stamp,
                    run_count=run_count,
                    latest_episode=latest_episode,
                    mtime=path.stat().st_mtime,
                    status_summary=_completion_summary(path),
                )
            )

    entries.sort(key=lambda entry: entry.mtime, reverse=True)
    return entries[:limit]


def list_sideboard_candidate_ids(path: Path) -> list[str]:
    """Return candidate IDs declared in a sideboard compare manifest."""
    manifest_path = path / "candidates_manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    raw = data.get("candidates") or []
    ids: list[str] = []
    if isinstance(raw, list):
        for row in raw:
            if isinstance(row, dict):
                cid = str(row.get("candidate_id", "") or "").strip()
                if cid:
                    ids.append(cid)
    return ids
