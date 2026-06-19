"""Discover saved training results with evaluable phase-3 checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fab_tui.config import REPO_ROOT, RESULTS_ROOT

RESULT_CATEGORY_ROOTS: tuple[tuple[str, Path], ...] = (
    ("experiments", RESULTS_ROOT / "experiments"),
    ("matchup_sims", RESULTS_ROOT / "matchup_sims"),
    ("full_pipeline", RESULTS_ROOT / "full_pipeline"),
)


@dataclass(frozen=True)
class EvaluableResultsEntry:
    path: Path
    category: str
    label: str
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


def _has_phase3_checkpoints(path: Path) -> bool:
    return any(path.glob("p3_*/p1/episode_*/metadata.json"))


def _checkpoint_summary(path: Path) -> tuple[int, str | None]:
    p3_dirs = [p for p in path.glob("p3_*") if p.is_dir()]
    latest_episode: str | None = None
    for meta in sorted(
        path.glob("p3_*/p1/episode_*/metadata.json"),
        key=lambda p: p.parent.name,
        reverse=True,
    ):
        latest_episode = meta.parent.name
        break
    return len(p3_dirs), latest_episode


def _label_for_results_dir(path: Path) -> str:
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
            entries.append(
                EvaluableResultsEntry(
                    path=resolved,
                    category=category,
                    label=_label_for_results_dir(path),
                    run_count=run_count,
                    latest_episode=latest_episode,
                    mtime=path.stat().st_mtime,
                )
            )

    entries.sort(key=lambda entry: entry.mtime, reverse=True)
    return entries[:limit]
