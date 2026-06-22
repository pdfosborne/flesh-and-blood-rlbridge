"""Train multiple RNG seeds in parallel and pick the best agent per role."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAINING_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from runtime_defaults import DEFAULT_PARALLEL_SEEDS  # noqa: E402

SEED_STRIDE = 10_000


def derive_training_seed(base_seed: Optional[int], seed_index: int) -> Optional[int]:
    if base_seed is None:
        return None
    return base_seed + seed_index * SEED_STRIDE


def workers_per_parallel_seed(
    total_workers: Optional[int],
    n_seeds: int,
    *,
    cpu_count: Optional[int] = None,
) -> int:
    """Divide rollout workers across parallel seeds to avoid CPU oversubscription."""
    cores = cpu_count if cpu_count is not None else (os.cpu_count() or 4)
    budget = max(1, int(total_workers)) if total_workers is not None else max(1, cores)
    return max(1, budget // max(1, n_seeds))


def apply_cpu_affinity_for_seed(
    seed_index: int,
    n_seeds: int,
    workers_per_seed: int,
) -> None:
    """Pin a seed subprocess to a disjoint slice of CPU cores when psutil is available."""
    try:
        import psutil
    except ImportError:
        return

    cpu_count = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    per_seed = max(1, workers_per_seed)
    start = (seed_index * per_seed) % cpu_count
    cores = [(start + i) % cpu_count for i in range(min(per_seed, cpu_count))]
    try:
        psutil.Process().cpu_affinity(cores)
    except (AttributeError, NotImplementedError, OSError):
        return


def _process_pool_entry(payload: dict[str, Any]) -> dict[str, Any]:
    """Top-level ProcessPoolExecutor entry (must be picklable)."""
    apply_cpu_affinity_for_seed(
        int(payload["seed_index"]),
        int(payload["n_seeds"]),
        int(payload["workers_per_seed"]),
    )
    handler = str(payload.get("handler", ""))
    if handler == "play_parallel_seed":
        from train_play import execute_play_parallel_seed_job  # noqa: PLC0415

        return execute_play_parallel_seed_job(payload)
    raise ValueError(f"unknown parallel seed handler: {handler!r}")


def average_win_rates(rows: list[tuple[float, float]]) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    return mean(r[0] for r in rows), mean(r[1] for r in rows)


def _parallel_seed_dirs(out_dir: Path) -> list[Path]:
    seeds_root = out_dir / "parallel_seeds"
    if not seeds_root.is_dir():
        return []
    return sorted(p for p in seeds_root.glob("seed_*") if p.is_dir())


def _seed_index_from_dir(seed_dir: Path) -> int:
    name = seed_dir.name
    if name.startswith("seed_"):
        try:
            return int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            pass
    return 0


def load_seed_checkpoint_histories(
    out_dir: Path,
) -> dict[int, list[dict[str, Any]]]:
    """Return per-seed checkpoint eval histories under ``parallel_seeds/``."""
    histories: dict[int, list[dict[str, Any]]] = {}
    for seed_dir in _parallel_seed_dirs(out_dir):
        path = seed_dir / "checkpoint_eval_history.json"
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(raw, list):
            histories[_seed_index_from_dir(seed_dir)] = [
                row for row in raw if isinstance(row, dict)
            ]
    return histories


def merge_parallel_seed_checkpoint_history(
    out_dir: Path,
    *,
    write: bool = True,
) -> list[dict[str, Any]]:
    """Merge per-seed checkpoint evals; best ``p1_win_rate`` wins each episode."""
    histories = load_seed_checkpoint_histories(out_dir)
    if not histories:
        parent = out_dir / "checkpoint_eval_history.json"
        if parent.is_file():
            try:
                raw = json.loads(parent.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
            if isinstance(raw, list):
                return [row for row in raw if isinstance(row, dict)]
        return []

    by_episode: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for seed_idx, rows in histories.items():
        for row in rows:
            ep = int(row.get("episodes_completed", 0) or 0)
            by_episode.setdefault(ep, []).append((seed_idx, row))

    merged: list[dict[str, Any]] = []
    for ep in sorted(by_episode):
        candidates = by_episode[ep]
        best_seed_idx, best_row = max(
            candidates,
            key=lambda item: float(item[1].get("p1_win_rate", 0) or 0),
        )
        record = dict(best_row)
        record["episodes_completed"] = ep
        record["best_p1_seed_index"] = best_seed_idx
        record["parallel_seeds"] = len(candidates)
        record["seed_checkpoint_win_rates"] = {
            str(seed_i): float(row.get("p1_win_rate", 0) or 0)
            for seed_i, row in candidates
        }
        merged.append(record)

    if write and merged:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoint_eval_history.json").write_text(
            json.dumps(merged, indent=2),
            encoding="utf-8",
        )
    return merged


def merge_parallel_seed_training_live(
    out_dir: Path,
    *,
    write: bool = True,
) -> Optional[dict[str, Any]]:
    """Average training live stats across parallel seed runs."""
    lives: list[dict[str, Any]] = []
    for seed_dir in _parallel_seed_dirs(out_dir):
        path = seed_dir / "play_training_live.json"
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(raw, dict):
            lives.append(raw)
    if not lives:
        return None

    episodes = [int(row.get("episodes_completed", 0) or 0) for row in lives]
    targets = [int(row.get("target_episodes", 0) or 0) for row in lives]
    win_rates = [
        float(row["win_rate"])
        for row in lives
        if row.get("win_rate") is not None
    ]
    merged: dict[str, Any] = {
        "episodes_completed": int(round(mean(episodes))) if episodes else 0,
        "target_episodes": int(round(mean(targets))) if targets else 0,
        "win_rate": float(mean(win_rates)) if win_rates else 0.0,
        "parallel_seeds": len(lives),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    for key in ("wins", "losses", "draws", "timeouts"):
        values = [int(row.get(key, 0) or 0) for row in lives if key in row]
        if values:
            merged[key] = int(round(mean(values)))

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "play_training_live.json").write_text(
            json.dumps(merged, indent=2),
            encoding="utf-8",
        )
    return merged


def sync_parallel_seed_dashboard_artifacts(out_dir: Path) -> None:
    """Write merged checkpoint eval + training live files for the HTML dashboard."""
    merge_parallel_seed_checkpoint_history(out_dir, write=True)
    merge_parallel_seed_training_live(out_dir, write=True)


def select_best_agents_by_win_rate(
    rows: list[dict[str, Any]],
    *,
    p1_win_key: str = "p1_win_rate",
    p2_win_key: str = "p2_win_rate",
    p1_agent_key: str = "p1_agent",
    p2_agent_key: str = "p2_agent",
) -> tuple[Any, Any, int, int]:
    """Return best P1/P2 agents and the seed indices they came from."""
    best_p1_row = max(rows, key=lambda row: float(row[p1_win_key]))
    best_p2_row = max(rows, key=lambda row: float(row[p2_win_key]))
    return (
        best_p1_row[p1_agent_key],
        best_p2_row[p2_agent_key],
        int(best_p1_row.get("seed_index", 0)),
        int(best_p2_row.get("seed_index", 0)),
    )


@dataclass
class ParallelSeedSummary:
    n_seeds: int
    avg_p1_win_rate: float
    avg_p2_win_rate: float
    best_p1_seed_index: int
    best_p2_seed_index: int
    seed_rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_seeds": self.n_seeds,
            "avg_p1_win_rate": self.avg_p1_win_rate,
            "avg_p2_win_rate": self.avg_p2_win_rate,
            "best_p1_seed_index": self.best_p1_seed_index,
            "best_p2_seed_index": self.best_p2_seed_index,
            "seeds": [
                {
                    "seed_index": row.get("seed_index"),
                    "seed": row.get("seed"),
                    "p1_win_rate": row.get("p1_win_rate"),
                    "p2_win_rate": row.get("p2_win_rate"),
                    "out_dir": row.get("out_dir"),
                }
                for row in self.seed_rows
            ],
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }


def run_parallel_seed_jobs(
    n_seeds: int,
    base_seed: Optional[int],
    out_dir: Path,
    job_fn: Optional[Callable[[int, Optional[int], Path], dict[str, Any]]] = None,
    *,
    label: str = "training",
    on_seed_complete: Optional[Callable[[dict[str, Any]], None]] = None,
    process_jobs: Optional[list[dict[str, Any]]] = None,
    workers_per_seed: int = 1,
    use_processes: bool = True,
) -> ParallelSeedSummary:
    """Run *n_seeds* independent training jobs and aggregate win rates.

    When *process_jobs* is provided and *use_processes* is True, each seed runs in
    its own subprocess with optional CPU affinity.  Otherwise falls back to threads
    via *job_fn*.
    """
    if n_seeds <= 1:
        raise ValueError("run_parallel_seed_jobs requires n_seeds > 1")

    seeds_root = out_dir / "parallel_seeds"
    seeds_root.mkdir(parents=True, exist_ok=True)
    mode = "process" if (use_processes and process_jobs) else "thread"
    print(
        f"  Parallel seeds: {n_seeds} independent {label} run(s) "
        f"({mode}, {workers_per_seed} worker(s)/seed) → {seeds_root}"
    )

    rows: list[dict[str, Any]] = []

    if use_processes and process_jobs is not None:
        executor_cls = ProcessPoolExecutor
        with executor_cls(max_workers=n_seeds) as pool:
            futures = {
                pool.submit(_process_pool_entry, payload): int(payload["seed_index"])
                for payload in process_jobs
            }
            for fut in as_completed(futures):
                seed_index = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Parallel seed {seed_index} failed during {label}: {exc}"
                    ) from exc
                row.setdefault("seed_index", seed_index)
                rows.append(row)
                if on_seed_complete is not None:
                    on_seed_complete(row)
    else:
        if job_fn is None:
            raise ValueError("run_parallel_seed_jobs requires job_fn or process_jobs")
        with ThreadPoolExecutor(max_workers=n_seeds) as pool:
            futures = {
                pool.submit(
                    job_fn,
                    seed_index,
                    derive_training_seed(base_seed, seed_index),
                    seeds_root / f"seed_{seed_index}",
                ): seed_index
                for seed_index in range(n_seeds)
            }
            for fut in as_completed(futures):
                seed_index = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Parallel seed {seed_index} failed during {label}: {exc}"
                    ) from exc
                row.setdefault("seed_index", seed_index)
                rows.append(row)
                if on_seed_complete is not None:
                    on_seed_complete(row)

    rows.sort(key=lambda row: int(row.get("seed_index", 0)))
    sync_parallel_seed_dashboard_artifacts(out_dir)
    avg_p1, avg_p2 = average_win_rates(
        [(float(r["p1_win_rate"]), float(r["p2_win_rate"])) for r in rows]
    )
    _, _, best_p1_idx, best_p2_idx = select_best_agents_by_win_rate(rows)

    print(f"\n  Seed training win rates ({n_seeds} seeds, avg shown):")
    for row in rows:
        print(
            f"    seed {row.get('seed_index')}: "
            f"p1={float(row['p1_win_rate']):.1%}  "
            f"p2={float(row['p2_win_rate']):.1%}"
        )
    print(
        f"  Average train win%: p1={avg_p1:.1%}  p2={avg_p2:.1%}"
    )
    print(
        f"  Selected for eval: best P1 from seed {best_p1_idx}, "
        f"best P2 from seed {best_p2_idx}"
    )

    summary = ParallelSeedSummary(
        n_seeds=n_seeds,
        avg_p1_win_rate=avg_p1,
        avg_p2_win_rate=avg_p2,
        best_p1_seed_index=best_p1_idx,
        best_p2_seed_index=best_p2_idx,
        seed_rows=rows,
    )
    (out_dir / "parallel_seeds_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2),
        encoding="utf-8",
    )
    return summary
