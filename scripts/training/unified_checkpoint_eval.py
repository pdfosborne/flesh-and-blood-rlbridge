"""Batch-level merged checkpoint eval for unified random matchup training."""

from __future__ import annotations

import json
import math
import random
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from checkpoint_eval_async import submit_checkpoint_eval
from fab_bridge.unified_results import CHECKPOINT_EVAL_SCOPE

if TYPE_CHECKING:
    from train_dual_agent_common import Matchup, PPOAgent
    from flesh_and_blood_rlbridge.talishar_backend_pool import TalisharBackendPool


def allocate_eval_episodes(
    total_episodes: int,
    matchup_keys: list[str],
    *,
    seed: Optional[int] = None,
) -> dict[str, int]:
    """Split *total_episodes* evenly across *matchup_keys* (remainder shuffled).

    The returned counts sum to *total_episodes* and are the self-play eval games
    budget per matchup (not per supplementary eval mode).
    """
    if total_episodes <= 0 or not matchup_keys:
        return {}
    keys = list(matchup_keys)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(keys)
    n = len(keys)
    base, rem = divmod(total_episodes, n)
    return {key: base + (1 if i < rem else 0) for i, key in enumerate(keys)}


def uniform_matchup_schedule(
    total_episodes: int,
    matchup_keys: list[str],
    *,
    seed: Optional[int] = None,
) -> list[str]:
    """Return *total_episodes* independent uniform samples over *matchup_keys*."""
    if total_episodes <= 0 or not matchup_keys:
        return []
    rng = random.Random(seed)
    keys = list(matchup_keys)
    return [rng.choice(keys) for _ in range(int(total_episodes))]


def _mean_and_se(values: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))


def _vs_logic_avg(row: dict[str, Any]) -> Optional[float]:
    vs_logic = row.get("vs_logic")
    if not isinstance(vs_logic, dict):
        return None
    rates: list[float] = []
    for seat_key in ("agent_p1_seat", "agent_p2_seat"):
        seat = vs_logic.get(seat_key)
        if isinstance(seat, dict) and seat.get("agent_win_rate") is not None:
            rates.append(float(seat["agent_win_rate"]))
    if not rates:
        return None
    return sum(rates) / len(rates)


def _build_aggregate(per_matchup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    self_play = [
        float(row.get("p1_win_rate") or 0.0)
        for row in per_matchup.values()
        if row.get("p1_win_rate") is not None
    ]
    vs_logic = [
        rate
        for row in per_matchup.values()
        if (rate := _vs_logic_avg(row)) is not None
    ]
    sp_mean, sp_se = _mean_and_se(self_play)
    vl_mean, vl_se = _mean_and_se(vs_logic)
    total_eval_games = sum(int(row.get("eval_episodes") or 0) for row in per_matchup.values())
    return {
        "self_play_win_rate_mean": sp_mean,
        "self_play_win_rate_se": sp_se,
        "vs_logic_win_rate_mean": vl_mean,
        "vs_logic_win_rate_se": vl_se,
        "total_eval_games": total_eval_games,
    }


class UnifiedCheckpointCoordinator:
    """Gates merged checkpoint eval on all active batch matchups reaching buckets."""

    def __init__(
        self,
        *,
        out_dir: Path,
        matchups: dict[str, "Matchup"],
        base_url: str,
        game_format: str,
        max_steps: int,
        n_episodes: int,
        warmup_episodes: int,
        checkpoint_interval: int,
        checkpoint_eval_episodes: int,
        unified_policy: "PPOAgent",
        policy_snapshot_fn: Callable[[], tuple["PPOAgent", "PPOAgent"]],
        seed: Optional[int] = None,
        eval_backend_pool: Optional["TalisharBackendPool"] = None,
        eval_agent_vs_logic: bool = False,
        eval_logic_vs_logic: bool = True,
    ) -> None:
        self.out_dir = out_dir.expanduser().resolve()
        self.matchups = dict(matchups)
        self.base_url = base_url
        self.game_format = game_format
        self.max_steps = max_steps
        self.n_episodes = int(n_episodes)
        self.warmup_episodes = max(0, int(warmup_episodes))
        self.checkpoint_interval = max(1, int(checkpoint_interval))
        self.checkpoint_eval_episodes = int(checkpoint_eval_episodes)
        self.unified_policy = unified_policy
        self.policy_snapshot_fn = policy_snapshot_fn
        self.seed = seed
        self.eval_backend_pool = eval_backend_pool
        self.eval_agent_vs_logic = bool(eval_agent_vs_logic)
        self.eval_logic_vs_logic = bool(eval_logic_vs_logic)
        self.log: list[dict[str, Any]] = []
        self.first_win_rate: Optional[float] = None
        self.final_win_rate: Optional[float] = None
        self._progress: dict[str, int] = {}
        self._completed_buckets: set[int] = set()
        self._lock = threading.Lock()
        self._history_lock = threading.Lock()

    def report_progress(self, matchup_key: str, episodes_completed: int) -> None:
        if matchup_key not in self.matchups:
            return
        with self._lock:
            self._progress[matchup_key] = max(
                self._progress.get(matchup_key, 0),
                int(episodes_completed),
            )
            if len(self._progress) < len(self.matchups):
                return
            min_ep = min(self._progress[k] for k in self.matchups)
            for bucket_ep, bucket_kind in self._eligible_buckets(min_ep):
                if bucket_ep in self._completed_buckets:
                    continue
                self._completed_buckets.add(bucket_ep)
                self._trigger_eval(bucket_ep, bucket_kind)

    def _eligible_buckets(self, min_ep: int) -> list[tuple[int, str]]:
        buckets: list[tuple[int, str]] = []
        if self.warmup_episodes > 0 and min_ep >= self.warmup_episodes:
            buckets.append((self.warmup_episodes, "warmup"))
        for k in range(1, (self.n_episodes // self.checkpoint_interval) + 1):
            ep = k * self.checkpoint_interval
            if ep > self.n_episodes:
                break
            if min_ep >= ep:
                buckets.append((ep, "periodic"))
        if min_ep >= self.n_episodes:
            buckets.append((self.n_episodes, "final"))
        seen: set[int] = set()
        unique: list[tuple[int, str]] = []
        for ep, kind in buckets:
            if ep in seen:
                continue
            seen.add(ep)
            unique.append((ep, self._bucket_kind(ep)))
        return unique

    def _bucket_kind(self, ep: int) -> str:
        if self.warmup_episodes > 0 and ep == self.warmup_episodes:
            return "warmup"
        if ep >= self.n_episodes:
            return "final"
        return "periodic"

    def _trigger_eval(self, bucket_ep: int, bucket_kind: str) -> None:
        eval_p1, eval_p2 = self.policy_snapshot_fn()
        if eval_p1._shared is None or eval_p2._shared is None:
            raise RuntimeError(
                "Merged checkpoint eval could not clone unified policy weights"
            )
        from train_dual_agent_common import (  # noqa: PLC0415
            _save_unified_selfplay_checkpoint,
        )

        for matchup in self.matchups.values():
            _save_unified_selfplay_checkpoint(
                out_dir=self.out_dir,
                matchup=matchup,
                game_format=self.game_format,
                policy=self.unified_policy,
                episodes_completed=bucket_ep,
                target_episodes=self.n_episodes,
            )
        submit_checkpoint_eval(
            lambda ep=bucket_ep, kind=bucket_kind, p1=eval_p1, p2=eval_p2: self._run_merged_eval(
                episodes_completed=ep,
                bucket_kind=kind,
                eval_p1=p1,
                eval_p2=p2,
            ),
            label=f"merged ep {bucket_ep}",
            run_dir=self.out_dir,
        )

    def _run_merged_eval(
        self,
        *,
        episodes_completed: int,
        bucket_kind: str,
        eval_p1: "PPOAgent",
        eval_p2: "PPOAgent",
    ) -> None:
        from train_dual_agent_common import (  # noqa: PLC0415
            DEFAULT_TALISHAR_BACKEND,
            _resolve_matchup_subdir,
        )
        from unified_logic_baseline import load_logic_vs_logic_baseline  # noqa: PLC0415
        from train_play import (  # noqa: PLC0415
            _evaluate_p1_vs_fixed_opponent,
            evaluate_agent_vs_logic_both_seats,
        )
        from fab_bridge.cpp_eval_live_dashboard import (  # noqa: PLC0415
            unified_checkpoint_eval_live_path,
        )
        from fab_bridge.unified_dashboard import maybe_refresh_unified_dashboard  # noqa: PLC0415

        eval_base_url = self.base_url
        live_progress_path = unified_checkpoint_eval_live_path(self.out_dir)
        keys = list(self.matchups.keys())
        alloc_seed = (
            (self.seed + episodes_completed) if self.seed is not None else None
        )
        schedule_counts = allocate_eval_episodes(
            self.checkpoint_eval_episodes,
            keys,
            seed=alloc_seed,
        )
        per_matchup: dict[str, dict[str, Any]] = {}
        matchups_total = len(keys)
        total_self_play_games = sum(schedule_counts.values())

        try:
            for index, (matchup_key, matchup) in enumerate(self.matchups.items()):
                eval_eps = schedule_counts.get(matchup_key, 0)
                if eval_eps <= 0:
                    continue
                eval_seed = (
                    (self.seed + episodes_completed + index * 10_000)
                    if self.seed is not None
                    else None
                )
                live_progress_extra = {
                    "matchup": matchup.name,
                    "matchup_dir": matchup_key,
                    "merged_eval_index": index + 1,
                    "matchups_total": matchups_total,
                }
                metrics = _evaluate_p1_vs_fixed_opponent(
                    matchup,
                    eval_p1,
                    p2_agent=eval_p2,
                    base_url=eval_base_url,
                    game_format=self.game_format,
                    max_steps=self.max_steps,
                    episodes=eval_eps,
                    seed=eval_seed,
                    backend=DEFAULT_TALISHAR_BACKEND,
                    eval_label="Checkpoint eval (self-play)",
                    live_progress_path=live_progress_path,
                    live_progress_extra=live_progress_extra,
                    backend_pool=self.eval_backend_pool,
                )
                record = {
                    "matchup": matchup.name,
                    "matchup_dir": _resolve_matchup_subdir(self.out_dir, matchup),
                    "episodes_completed": episodes_completed,
                    "target_episodes": self.n_episodes,
                    "eval_episodes": eval_eps,
                    "eval_mode": "self_play",
                    **metrics,
                }
                if self.eval_agent_vs_logic and eval_eps > 0:
                    vs_logic = evaluate_agent_vs_logic_both_seats(
                        matchup,
                        eval_p1,
                        base_url=eval_base_url,
                        game_format=self.game_format,
                        max_steps=self.max_steps,
                        episodes=eval_eps,
                        seed=(
                            (eval_seed + 25_000) if eval_seed is not None else None
                        ),
                        backend=DEFAULT_TALISHAR_BACKEND,
                        eval_label_prefix="Checkpoint eval vs logic",
                        live_progress_path=live_progress_path,
                        live_progress_extra=live_progress_extra,
                        backend_pool=self.eval_backend_pool,
                    )
                    record["vs_logic"] = vs_logic
                if self.eval_logic_vs_logic:
                    baseline = load_logic_vs_logic_baseline(self.out_dir, matchup)
                    if baseline is not None:
                        record["logic_vs_logic"] = baseline
                per_matchup[matchup_key] = record
                self._write_per_matchup_artifacts(matchup, episodes_completed, record)
        except Exception as exc:
            self._append_failed_merged_record(
                episodes_completed=episodes_completed,
                bucket_kind=bucket_kind,
                error=str(exc),
                per_matchup=per_matchup,
                eval_episodes_total=self.checkpoint_eval_episodes,
            )
            maybe_refresh_unified_dashboard(self.out_dir, min_interval_seconds=0.0)
            raise

        aggregate = _build_aggregate(per_matchup)
        merged_record = {
            "eval_mode": "merged",
            "episodes_completed": episodes_completed,
            "target_episodes": self.n_episodes,
            "eval_episodes_total": self.checkpoint_eval_episodes,
            "self_play_games_total": total_self_play_games,
            "matchups_evaluated": len(per_matchup),
            "checkpoint_bucket": bucket_kind,
            "aggregate": aggregate,
            "per_matchup": per_matchup,
        }
        self._append_merged_record(merged_record)
        wr = aggregate.get("self_play_win_rate_mean")
        if wr is not None:
            if self.first_win_rate is None:
                self.first_win_rate = float(wr)
            self.final_win_rate = float(wr)
        self._write_merged_scope()
        print(
            f"  Merged checkpoint eval @ ep {episodes_completed} ({bucket_kind}): "
            f"self-play mean={float(wr or 0.0):.1%} "
            f"({len(per_matchup)} matchup(s), "
            f"{total_self_play_games} self-play game(s) total)"
        )
        maybe_refresh_unified_dashboard(self.out_dir, min_interval_seconds=0.0)

    def _write_per_matchup_artifacts(
        self,
        matchup: "Matchup",
        episodes_completed: int,
        record: dict[str, Any],
    ) -> None:
        from train_dual_agent_common import matchup_out_dir  # noqa: PLC0415

        matchup_dir = matchup_out_dir(self.out_dir, matchup)
        ckpt_dir = matchup_dir / "checkpoint_eval" / f"episode_{episodes_completed:06d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "checkpoint_eval.json").write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )
        history_path = matchup_dir / "checkpoint_eval_history.json"
        history = self._read_json_list(history_path)
        history.append(record)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    def _append_merged_record(self, record: dict[str, Any]) -> None:
        with self._history_lock:
            history_path = self.out_dir / "checkpoint_eval_history.json"
            history = self._read_json_list(history_path)
            history.append(record)
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
            self.log.append(record)

    def _append_failed_merged_record(
        self,
        *,
        episodes_completed: int,
        bucket_kind: str,
        error: str,
        per_matchup: dict[str, dict[str, Any]],
        eval_episodes_total: int,
    ) -> None:
        aggregate = _build_aggregate(per_matchup) if per_matchup else {}
        merged_record = {
            "eval_mode": "merged",
            "status": "error",
            "error": error,
            "episodes_completed": episodes_completed,
            "target_episodes": self.n_episodes,
            "eval_episodes_total": eval_episodes_total,
            "matchups_evaluated": len(per_matchup),
            "checkpoint_bucket": bucket_kind,
            "aggregate": aggregate,
            "per_matchup": per_matchup,
        }
        self._append_merged_record(merged_record)
        print(
            f"  Merged checkpoint eval @ ep {episodes_completed} ({bucket_kind}) FAILED: "
            f"{error[:200]}",
            flush=True,
        )

    def _write_merged_scope(self) -> None:
        from train_dual_agent_common import _resolve_matchup_subdir  # noqa: PLC0415

        matchups_payload = [
            {
                "matchup": matchup.name,
                "matchup_dir": _resolve_matchup_subdir(self.out_dir, matchup),
            }
            for matchup in self.matchups.values()
        ]
        first = matchups_payload[0] if matchups_payload else {}
        payload = {
            "matchup": first.get("matchup", ""),
            "matchup_dir": first.get("matchup_dir", ""),
            "matchups": matchups_payload,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.out_dir / CHECKPOINT_EVAL_SCOPE).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _read_json_list(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, TypeError):
            return []
        if not isinstance(raw, list):
            return []
        return [row for row in raw if isinstance(row, dict)]

    def get_matchup_checkpoint_win_rate(self, matchup_key: str) -> Optional[float]:
        for record in reversed(self.log):
            per = record.get("per_matchup")
            if not isinstance(per, dict):
                continue
            row = per.get(matchup_key)
            if isinstance(row, dict) and row.get("p1_win_rate") is not None:
                return float(row["p1_win_rate"])
        return None

    def get_matchup_checkpoint_history(self, matchup_key: str) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for record in self.log:
            per = record.get("per_matchup")
            if not isinstance(per, dict):
                continue
            row = per.get(matchup_key)
            if isinstance(row, dict):
                history.append(row)
        return history
