"""Async logic-vs-logic baselines for unified random matchup training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from checkpoint_eval_async import submit_checkpoint_eval
from fab_bridge.unified_dashboard import LOGIC_VS_LOGIC_BASELINE_NAME
from unified_checkpoint_eval import allocate_eval_episodes

if TYPE_CHECKING:
    from train_dual_agent_common import Matchup


def logic_vs_logic_baseline_path(out_dir: Path, matchup: "Matchup") -> Path:
    from train_dual_agent_common import matchup_out_dir  # noqa: PLC0415

    return matchup_out_dir(out_dir, matchup) / LOGIC_VS_LOGIC_BASELINE_NAME


def load_logic_vs_logic_baseline(
    out_dir: Path,
    matchup: "Matchup",
) -> Optional[dict[str, Any]]:
    path = logic_vs_logic_baseline_path(out_dir, matchup)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return None
    return raw if isinstance(raw, dict) else None


def run_matchup_logic_vs_logic_baseline(
    matchup: "Matchup",
    *,
    out_dir: Path,
    base_url: str,
    game_format: str,
    max_steps: int,
    episodes: int,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Evaluate logic win% vs logic for one matchup and persist the baseline file."""
    from train_dual_agent_common import DEFAULT_TALISHAR_BACKEND  # noqa: PLC0415
    from train_play import evaluate_logic_vs_logic  # noqa: PLC0415

    path = logic_vs_logic_baseline_path(out_dir, matchup)
    if path.is_file():
        existing = load_logic_vs_logic_baseline(out_dir, matchup)
        if existing is not None:
            return existing

    if episodes <= 0:
        return {
            "episodes": 0,
            "p1_wins": 0,
            "p2_wins": 0,
            "draws": 0,
            "timeouts": 0,
            "errors": 0,
            "p1_win_rate": 0.0,
            "p2_win_rate": 0.0,
            "draw_rate": 0.0,
            "timeout_rate": 0.0,
            "p1_policy": "logic",
            "p2_policy": "logic",
        }

    metrics = evaluate_logic_vs_logic(
        matchup,
        base_url=base_url,
        game_format=game_format,
        max_steps=max_steps,
        episodes=episodes,
        seed=seed,
        backend=DEFAULT_TALISHAR_BACKEND,
        eval_label="Logic win% vs logic baseline",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    from fab_bridge.unified_dashboard import maybe_refresh_unified_dashboard  # noqa: PLC0415

    maybe_refresh_unified_dashboard(out_dir, min_interval_seconds=0.0)
    return metrics


def run_batch_logic_vs_logic_baselines(
    matchups: dict[str, "Matchup"],
    *,
    out_dir: Path,
    base_url: str,
    game_format: str,
    max_steps: int,
    total_episodes: int,
    seed: Optional[int] = None,
) -> None:
    """Run logic-vs-logic baselines for all matchups (serial job on eval shard)."""
    keys = list(matchups.keys())
    alloc = allocate_eval_episodes(total_episodes, keys, seed=seed)
    pending = {
        key: matchup
        for key, matchup in matchups.items()
        if alloc.get(key, 0) > 0
        and not logic_vs_logic_baseline_path(out_dir, matchup).is_file()
    }
    if not pending:
        return

    def _run_batch() -> dict[str, Any]:
        results: dict[str, Any] = {}
        for index, (matchup_key, matchup) in enumerate(pending.items()):
            eval_eps = alloc.get(matchup_key, 0)
            if eval_eps <= 0:
                continue
            eval_seed = (
                (seed + index * 10_000) if seed is not None else None
            )
            try:
                results[matchup_key] = run_matchup_logic_vs_logic_baseline(
                    matchup,
                    out_dir=out_dir,
                    base_url=base_url,
                    game_format=game_format,
                    max_steps=max_steps,
                    episodes=eval_eps,
                    seed=eval_seed,
                )
            except Exception as exc:
                print(
                    f"  Logic win% vs logic baseline: skipped matchup "
                    f"'{matchup.name}' due to error: {exc!r}",
                    flush=True,
                )
        return results

    submit_checkpoint_eval(_run_batch, label="logic win% vs logic baselines", run_dir=out_dir)


def submit_batch_logic_vs_logic_baselines(
    matchups: dict[str, "Matchup"],
    *,
    out_dir: Path,
    base_url: str,
    game_format: str,
    max_steps: int,
    total_episodes: int,
    seed: Optional[int] = None,
) -> int:
    """Schedule async logic-vs-logic baselines; return count of matchups queued."""
    keys = list(matchups.keys())
    alloc = allocate_eval_episodes(total_episodes, keys, seed=seed)
    pending_count = sum(
        1
        for key, matchup in matchups.items()
        if alloc.get(key, 0) > 0
        and not logic_vs_logic_baseline_path(out_dir, matchup).is_file()
    )
    if pending_count <= 0:
        return 0

    run_batch_logic_vs_logic_baselines(
        matchups,
        out_dir=out_dir,
        base_url=base_url,
        game_format=game_format,
        max_steps=max_steps,
        total_episodes=total_episodes,
        seed=seed,
    )
    return pending_count
