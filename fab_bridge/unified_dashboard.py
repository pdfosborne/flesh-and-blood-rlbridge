"""Live HTML dashboard for unified random matchup training runs."""

from __future__ import annotations

import html
import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fab_bridge.cpp_eval_live_dashboard import (
    CPP_EVAL_LIVE_DASHBOARD,
    CPP_EVAL_LIVE_STATE,
    checkpoint_eval_replay_display_label,
    format_checkpoint_eval_replay_heading,
)
from fab_bridge.unified_results import (
    RUN_MANIFEST,
    iter_unified_matchup_dirs,
    is_unified_random_matchup_run,
    read_checkpoint_eval_scope,
)

UNIFIED_DASHBOARD_NAME = "unified_random_matchups_dashboard.html"
UNIFIED_LIVE_STATE = "unified_training_live.json"
LOGIC_VS_LOGIC_BASELINE_NAME = "logic_vs_logic_baseline.json"

_last_dashboard_write: dict[str, float] = {}
_run_state_locks: dict[str, threading.Lock] = {}
_run_state_locks_guard = threading.Lock()


def _run_state_lock(run_dir: Path) -> threading.Lock:
    key = str(run_dir.expanduser().resolve())
    with _run_state_locks_guard:
        lock = _run_state_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _run_state_locks[key] = lock
        return lock


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return None


def _pct(value: object) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _vs_logic_win_rates(row: dict[str, Any]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (agent@P1 seat, agent@P2 seat, average) win rates vs logic policy."""
    vs_logic = row.get("vs_logic")
    if not isinstance(vs_logic, dict):
        return None, None, None
    p1_seat = vs_logic.get("agent_p1_seat")
    p2_seat = vs_logic.get("agent_p2_seat")
    p1_wr: Optional[float] = None
    p2_wr: Optional[float] = None
    if isinstance(p1_seat, dict) and p1_seat.get("agent_win_rate") is not None:
        p1_wr = float(p1_seat["agent_win_rate"])
    if isinstance(p2_seat, dict) and p2_seat.get("agent_win_rate") is not None:
        p2_wr = float(p2_seat["agent_win_rate"])
    if p1_wr is not None and p2_wr is not None:
        return p1_wr, p2_wr, (p1_wr + p2_wr) / 2.0
    if p1_wr is not None:
        return p1_wr, None, p1_wr
    if p2_wr is not None:
        return None, p2_wr, p2_wr
    return None, None, None


def _logic_vs_logic_win_rate(row: dict[str, Any]) -> Optional[float]:
    """P1-seat win rate when both seats use the C++ logic policy."""
    logic_vs_logic = row.get("logic_vs_logic")
    if not isinstance(logic_vs_logic, dict):
        return None
    if logic_vs_logic.get("p1_win_rate") is not None:
        return float(logic_vs_logic["p1_win_rate"])
    return None


def _read_matchup_logic_vs_logic_baseline(matchup_dir: Path) -> Optional[float]:
    """Read once-per-matchup logic-vs-logic baseline from the matchup directory."""
    raw = _read_json(matchup_dir / LOGIC_VS_LOGIC_BASELINE_NAME)
    if isinstance(raw, dict) and raw.get("p1_win_rate") is not None:
        return float(raw["p1_win_rate"])
    return None


def _logic_vs_agent_win_rate(row: dict[str, Any]) -> Optional[float]:
    """Average logic-policy win rate vs the trained agent (both seats)."""
    vs_logic = row.get("vs_logic")
    if not isinstance(vs_logic, dict):
        return None
    p1_seat = vs_logic.get("agent_p1_seat")
    p2_seat = vs_logic.get("agent_p2_seat")
    rates: list[float] = []
    if isinstance(p1_seat, dict) and p1_seat.get("p2_win_rate") is not None:
        rates.append(float(p1_seat["p2_win_rate"]))
    if isinstance(p2_seat, dict) and p2_seat.get("p1_win_rate") is not None:
        rates.append(float(p2_seat["p1_win_rate"]))
    if not rates:
        _, _, agent_avg = _vs_logic_win_rates(row)
        if agent_avg is not None:
            return 1.0 - agent_avg
        return None
    return sum(rates) / len(rates)


def _apply_checkpoint_history_to_row(
    row: dict[str, Any],
    ckpt_hist: list[Any],
) -> None:
    """Fill first/last self-play and vs-logic checkpoint win rates on *row*."""
    entries = [entry for entry in ckpt_hist if isinstance(entry, dict)]
    if not entries:
        return
    first = entries[0]
    last = entries[-1]
    if first.get("p1_win_rate") is not None:
        row["first_checkpoint_win_rate"] = float(first["p1_win_rate"])
    if last.get("p1_win_rate") is not None:
        row["checkpoint_win_rate"] = float(last["p1_win_rate"])
    _, _, first_vs_logic = _vs_logic_win_rates(first)
    _, _, last_vs_logic = _vs_logic_win_rates(last)
    if first_vs_logic is not None:
        row["first_checkpoint_vs_logic_win_rate"] = first_vs_logic
    if last_vs_logic is not None:
        row["checkpoint_vs_logic_win_rate"] = last_vs_logic
    if last.get("episodes_completed") is not None:
        row["episodes_completed"] = int(last["episodes_completed"])


def _resolve_checkpoint_eval_replay_label(
    run_dir: Path,
    *,
    ckpt_history: list[Any],
    live: dict[str, Any],
) -> str:
    """Infer the eval replay engine label for unified training runs."""
    live_state = _read_json(run_dir / CPP_EVAL_LIVE_STATE)
    if isinstance(live_state, dict):
        label = checkpoint_eval_replay_display_label(live_state)
        if label:
            return label
    for row in reversed(ckpt_history):
        if isinstance(row, dict) and row.get("runtime_backend"):
            label = checkpoint_eval_replay_display_label(str(row["runtime_backend"]))
            if label:
                return label
    if isinstance(live, dict) and live.get("runtime_backend"):
        label = checkpoint_eval_replay_display_label(str(live["runtime_backend"]))
        if label:
            return label
    return "Talishar fast"


def _matchup_label(matchup_dir: Path) -> str:
    raw = _read_json(matchup_dir / "matchup_label.json")
    if isinstance(raw, dict):
        name = str(raw.get("name") or "").strip()
        if name:
            return name
    return matchup_dir.name


def _is_matchup_complete(matchup_dir: Path, target_episodes: int) -> bool:
    if (matchup_dir / "cached_unified").is_dir():
        return True
    for results_path in matchup_dir.glob("ppo_*/training_results.json"):
        raw = _read_json(results_path)
        if not isinstance(raw, dict):
            continue
        stats = raw.get("training_stats") or {}
        episodes = int(stats.get("episodes", 0) or 0)
        rewards = raw.get("episode_rewards") or []
        n_target = int(raw.get("n_episodes", 0) or target_episodes or 0)
        if n_target > 0 and max(episodes, len(rewards)) >= n_target:
            return True
    return False


def _matchup_summary_row(matchup_dir: Path, target_episodes: int) -> dict[str, Any]:
    label = _matchup_label(matchup_dir)
    complete = _is_matchup_complete(matchup_dir, target_episodes)
    row: dict[str, Any] = {
        "name": label,
        "subdir": matchup_dir.name,
        "complete": complete,
        "cached": (matchup_dir / "cached_unified").is_dir(),
        "train_p1_win_rate": None,
        "train_p2_win_rate": None,
        "first_checkpoint_win_rate": None,
        "first_checkpoint_vs_logic_win_rate": None,
        "checkpoint_win_rate": None,
        "checkpoint_vs_logic_win_rate": None,
        "episodes_completed": 0,
        "target_episodes": target_episodes,
    }
    for results_path in sorted(matchup_dir.glob("ppo_*/training_results.json")):
        raw = _read_json(results_path)
        if not isinstance(raw, dict):
            continue
        stats = raw.get("training_stats") or {}
        if stats.get("p1_win_rate") is not None:
            row["train_p1_win_rate"] = float(stats["p1_win_rate"])
        if stats.get("p2_win_rate") is not None:
            row["train_p2_win_rate"] = float(stats["p2_win_rate"])
        row["episodes_completed"] = max(
            row["episodes_completed"],
            int(stats.get("episodes", 0) or 0),
            len(raw.get("episode_rewards") or []),
        )
        ckpt_hist = stats.get("checkpoint_eval_history")
        if isinstance(ckpt_hist, list) and ckpt_hist:
            _apply_checkpoint_history_to_row(row, ckpt_hist)
    per_matchup_ckpt = _read_json(matchup_dir / "checkpoint_eval_history.json")
    if isinstance(per_matchup_ckpt, list) and per_matchup_ckpt:
        _apply_checkpoint_history_to_row(row, per_matchup_ckpt)
    return row


def count_completed_matchups(run_dir: Path, target_episodes: int) -> int:
    return sum(
        1
        for matchup_dir in iter_unified_matchup_dirs(run_dir)
        if _is_matchup_complete(matchup_dir, target_episodes)
    )


def update_unified_training_live(run_dir: Path, **fields: Any) -> None:
    """Merge fields into ``unified_training_live.json`` at the run root."""
    if not is_unified_random_matchup_run(run_dir):
        return
    run_dir = run_dir.expanduser().resolve()
    path = run_dir / UNIFIED_LIVE_STATE
    with _run_state_lock(run_dir):
        current: dict[str, Any] = {}
        raw = _read_json(path)
        if isinstance(raw, dict):
            current.update(raw)
        current.update({k: v for k, v in fields.items() if v is not None})
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(path, current)


def update_unified_matchup_live(
    run_dir: Path,
    matchup_key: str,
    **fields: Any,
) -> None:
    """Merge per-matchup fields into ``active_matchups`` in live state."""
    if not is_unified_random_matchup_run(run_dir):
        return
    run_dir = run_dir.expanduser().resolve()
    path = run_dir / UNIFIED_LIVE_STATE
    with _run_state_lock(run_dir):
        current: dict[str, Any] = {}
        raw = _read_json(path)
        if isinstance(raw, dict):
            current.update(raw)
        active_raw = current.get("active_matchups")
        active: dict[str, Any] = (
            dict(active_raw) if isinstance(active_raw, dict) else {}
        )
        row: dict[str, Any] = {}
        existing = active.get(matchup_key)
        if isinstance(existing, dict):
            row.update(existing)
        row.update({k: v for k, v in fields.items() if v is not None})
        active[str(matchup_key)] = row
        current["active_matchups"] = active
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(path, current)


def _checkpoint_point_from_row(row: dict[str, Any]) -> dict[str, Any]:
    vs_p1, vs_p2, vs_avg = _vs_logic_win_rates(row)
    timeout_rate: Optional[float] = None
    if row.get("timeout_rate") is not None:
        timeout_rate = float(row["timeout_rate"])
    return {
        "episode": int(row.get("episodes_completed") or 0),
        "win_rate": float(row.get("p1_win_rate") or 0.0),
        "p1_wins": int(row.get("p1_wins") or 0),
        "p2_wins": int(row.get("p2_wins") or 0),
        "matchup": str(row.get("matchup") or ""),
        "vs_logic_agent_p1": vs_p1,
        "vs_logic_agent_p2": vs_p2,
        "vs_logic_win_rate": vs_avg,
        "logic_vs_agent_win_rate": _logic_vs_agent_win_rate(row),
        "agent_vs_agent_win_rate": float(row.get("p1_win_rate") or 0.0),
        "timeout_rate": timeout_rate,
    }


def _load_matchup_checkpoint_history(matchup_dir: Path) -> list[dict[str, Any]]:
    raw = _read_json(matchup_dir / "checkpoint_eval_history.json")
    if not isinstance(raw, list):
        return []
    return [row for row in raw if isinstance(row, dict)]


def _mean_and_se(values: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))


def aggregate_checkpoint_points(
    histories: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Average checkpoint metrics across matchups at each episode bucket."""
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for _key, hist in histories.items():
        for row in hist:
            if row.get("episodes_completed") is None:
                continue
            point = _checkpoint_point_from_row(row)
            episode = int(point["episode"])
            by_episode.setdefault(episode, []).append(point)

    aggregate: list[dict[str, Any]] = []
    for episode in sorted(by_episode.keys()):
        points = by_episode[episode]
        self_play = [float(p["win_rate"]) for p in points]
        vs_logic = [
            float(p["vs_logic_win_rate"])
            for p in points
            if p.get("vs_logic_win_rate") is not None
        ]
        sp_mean, sp_se = _mean_and_se(self_play)
        vl_mean, vl_se = _mean_and_se(vs_logic)
        aggregate.append(
            {
                "episode": episode,
                "win_rate_mean": sp_mean,
                "win_rate_se": sp_se,
                "vs_logic_mean": vl_mean,
                "vs_logic_se": vl_se,
                "n_matchups": len(points),
            }
        )
    return aggregate


def _latest_checkpoint_rows_for_active(
    run_dir: Path,
    active_matchups: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subdir, info in active_matchups.items():
        if not isinstance(info, dict):
            continue
        hist = _load_matchup_checkpoint_history(run_dir / str(subdir))
        if not hist:
            continue
        latest = hist[-1]
        point = _checkpoint_point_from_row(latest)
        point["matchup"] = str(info.get("name") or point.get("matchup") or subdir)
        point["matchup_dir"] = str(subdir)
        rows.append(point)
    return rows


def collect_unified_run_state(run_dir: Path) -> dict[str, Any]:
    """Scan a unified random matchups run directory into dashboard state."""
    run_dir = run_dir.expanduser().resolve()
    manifest = _read_json(run_dir / RUN_MANIFEST)
    if not isinstance(manifest, dict):
        manifest = {}

    live = _read_json(run_dir / UNIFIED_LIVE_STATE)
    if not isinstance(live, dict):
        live = {}

    scope = read_checkpoint_eval_scope(run_dir)
    ckpt_history = _read_json(run_dir / "checkpoint_eval_history.json")
    if not isinstance(ckpt_history, list):
        ckpt_history = []

    training_summary = _read_json(run_dir / "training_summary.json")
    if not isinstance(training_summary, list):
        training_summary = []

    target_episodes = int(
        live.get("target_episodes")
        or manifest.get("episodes_per_matchup")
        or 0
    )
    matchups_total = int(
        live.get("matchups_total")
        or manifest.get("matchups_requested")
        or len(manifest.get("matchups_sampled") or [])
        or 0
    )
    matchups_completed = int(
        live.get("matchups_completed")
        if live.get("matchups_completed") is not None
        else count_completed_matchups(run_dir, target_episodes)
    )

    parallel_matchups = int(
        live.get("parallel_matchups")
        or manifest.get("parallel_matchups")
        or 1
    )
    batch_index = int(live.get("batch_index") or 0)

    active_raw = live.get("active_matchups")
    active_matchups: dict[str, Any] = (
        dict(active_raw) if isinstance(active_raw, dict) else {}
    )

    current_name = str(live.get("current_matchup") or scope.get("matchup") or "").strip()
    current_subdir = str(
        live.get("current_matchup_dir") or scope.get("matchup_dir") or ""
    ).strip()
    if not current_name and current_subdir:
        current_name = _matchup_label(run_dir / current_subdir)

    if not active_matchups and current_subdir:
        active_matchups = {
            current_subdir: {
                "name": current_name or current_subdir,
                "episodes_completed": int(live.get("episodes_completed") or 0),
                "p1_win_rate": live.get("p1_win_rate"),
                "p2_win_rate": live.get("p2_win_rate"),
                "status": live.get("status") or "training",
            }
        }

    episodes_completed = int(live.get("episodes_completed") or 0)
    if episodes_completed <= 0 and active_matchups:
        episodes_completed = max(
            int(row.get("episodes_completed") or 0)
            for row in active_matchups.values()
            if isinstance(row, dict)
        )

    train_p1_wr = live.get("p1_win_rate")
    train_p2_wr = live.get("p2_win_rate")
    status = str(live.get("status") or "training")
    if (run_dir / "training_summary.json").is_file() and matchups_completed >= matchups_total > 0:
        status = "complete"
    elif not active_matchups and matchups_completed >= matchups_total > 0:
        status = "complete"

    active_histories: dict[str, list[dict[str, Any]]] = {
        str(subdir): _load_matchup_checkpoint_history(run_dir / str(subdir))
        for subdir in active_matchups
        if (run_dir / str(subdir)).is_dir()
    }
    if not active_histories and current_subdir:
        active_histories[current_subdir] = _load_matchup_checkpoint_history(
            run_dir / current_subdir
        )
    if not active_histories and ckpt_history:
        active_histories["__run_root__"] = [
            row for row in ckpt_history if isinstance(row, dict)
        ]

    checkpoint_aggregate_points = aggregate_checkpoint_points(active_histories)
    active_checkpoint_rows = _latest_checkpoint_rows_for_active(
        run_dir, active_matchups
    )
    if not active_checkpoint_rows and ckpt_history:
        latest = ckpt_history[-1]
        if isinstance(latest, dict):
            point = _checkpoint_point_from_row(latest)
            point["matchup"] = str(point.get("matchup") or current_name or "—")
            active_checkpoint_rows = [point]

    checkpoint_points = []
    for row in ckpt_history:
        if not isinstance(row, dict) or row.get("episodes_completed") is None:
            continue
        checkpoint_points.append(_checkpoint_point_from_row(row))

    completed_rows = [
        _matchup_summary_row(matchup_dir, target_episodes)
        for matchup_dir in reversed(iter_unified_matchup_dirs(run_dir))
        if _is_matchup_complete(matchup_dir, target_episodes)
    ]

    overall_pct = 0.0
    if matchups_total > 0:
        if active_matchups and target_episodes > 0 and status == "training":
            intra_values = [
                int(row.get("episodes_completed") or 0) / max(1, target_episodes)
                for row in active_matchups.values()
                if isinstance(row, dict)
            ]
            intra = sum(intra_values) / max(1, len(intra_values)) if intra_values else 0.0
        elif target_episodes > 0 and status == "training":
            intra = episodes_completed / max(1, target_episodes)
        else:
            intra = 0.0
        overall_pct = ((matchups_completed + intra) / matchups_total) * 100.0

    cpp_eval_live_dashboard = run_dir / CPP_EVAL_LIVE_DASHBOARD
    cpp_eval_live_dashboard_path = (
        str(cpp_eval_live_dashboard) if cpp_eval_live_dashboard.is_file() else ""
    )
    all_ckpt_for_label: list[Any] = ckpt_history
    for hist in active_histories.values():
        all_ckpt_for_label = list(hist)
        break
    checkpoint_eval_replay_label = _resolve_checkpoint_eval_replay_label(
        run_dir,
        ckpt_history=all_ckpt_for_label,
        live=live,
    )

    return {
        "run_dir": str(run_dir),
        "format": str(manifest.get("format") or "—"),
        "started_at": str(manifest.get("started_at") or ""),
        "matchups_total": matchups_total,
        "matchups_completed": matchups_completed,
        "matchups_sampled": list(manifest.get("matchups_sampled") or []),
        "target_episodes": target_episodes,
        "parallel_matchups": parallel_matchups,
        "batch_index": batch_index,
        "active_matchups": active_matchups,
        "current_matchup": current_name or "—",
        "current_matchup_dir": current_subdir,
        "episodes_completed": episodes_completed,
        "train_p1_win_rate": train_p1_wr,
        "train_p2_win_rate": train_p2_wr,
        "status": status,
        "checkpoint_points": checkpoint_points,
        "checkpoint_aggregate_points": checkpoint_aggregate_points,
        "active_checkpoint_rows": active_checkpoint_rows,
        "completed_matchups": completed_rows,
        "overall_pct": overall_pct,
        "complete": status == "complete",
        "cpp_eval_live_dashboard_path": cpp_eval_live_dashboard_path,
        "checkpoint_eval_replay_label": checkpoint_eval_replay_label,
    }


def _svg_winrate_chart_with_error_bands(
    points: list[dict[str, Any]],
    *,
    width: int = 520,
    height: int = 200,
    target_episodes: int = 0,
    empty_message: str = "Waiting for checkpoint eval…",
) -> str:
    if not points:
        return (
            f'<svg viewBox="0 0 {width} {height}" class="chart empty">'
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            f'class="chart-empty">{html.escape(empty_message)}</text></svg>'
        )
    pad_l, pad_r, pad_t, pad_b = 36, 12, 12, 28
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    max_x = max(
        int(target_episodes or 0),
        max(int(p.get("episode", 0) or 0) for p in points),
        1,
    )

    def px(ep: float) -> float:
        return pad_l + (ep / max_x) * plot_w

    def py(rate: float) -> float:
        return pad_t + (1.0 - max(0.0, min(1.0, float(rate)))) * plot_h

    def band(mean_key: str, se_key: str, class_prefix: str) -> str:
        upper = []
        lower = []
        for point in points:
            mean = point.get(mean_key)
            se = point.get(se_key)
            if mean is None:
                continue
            spread = float(se or 0.0)
            upper.append(f"{px(point['episode']):.1f},{py(float(mean) + spread):.1f}")
            lower.append(f"{px(point['episode']):.1f},{py(float(mean) - spread):.1f}")
        if not upper:
            return ""
        polygon = " ".join(upper + list(reversed(lower)))
        line = " ".join(
            f"{px(point['episode']):.1f},{py(float(point[mean_key])):.1f}"
            for point in points
            if point.get(mean_key) is not None
        )
        return (
            f'<polygon points="{polygon}" class="{class_prefix}-band"/>'
            f'<polyline points="{line}" class="{class_prefix}-line" fill="none"/>'
        )

    grid = "\n".join(
        f'<line x1="{pad_l}" y1="{py(v):.1f}" x2="{width - pad_r}" y2="{py(v):.1f}" class="chart-grid"/>'
        f'<text x="{pad_l - 6}" y="{py(v) + 4:.1f}" text-anchor="end" class="chart-axis">{int(v * 100)}%</text>'
        for v in (0.0, 0.5, 1.0)
    )
    legend = (
        '<text x="' + str(pad_l) + f'" y="{height - 8}" class="chart-axis">'
        "— self-play mean ± SE  ·  — vs logic mean ± SE</text>"
    )
    return f"""<svg viewBox="0 0 {width} {height}" class="chart">
  {grid}
  {band("win_rate_mean", "win_rate_se", "chart-selfplay")}
  {band("vs_logic_mean", "vs_logic_se", "chart-vslogic")}
  {legend}
</svg>"""


def _svg_winrate_chart(
    points: list[dict[str, Any]],
    *,
    width: int = 520,
    height: int = 160,
    target_episodes: int = 0,
    empty_message: str = "Waiting for checkpoint eval…",
) -> str:
    if not points:
        return (
            f'<svg viewBox="0 0 {width} {height}" class="chart empty">'
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            f'class="chart-empty">{html.escape(empty_message)}</text></svg>'
        )
    pad_l, pad_r, pad_t, pad_b = 36, 12, 12, 24
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    max_x = max(
        int(target_episodes or 0),
        max(int(p.get("episode", 0) or 0) for p in points),
        1,
    )

    def px(ep: float) -> float:
        return pad_l + (ep / max_x) * plot_w

    def py(rate: float) -> float:
        return pad_t + (1.0 - max(0.0, min(1.0, float(rate)))) * plot_h

    poly = " ".join(
        f"{px(p['episode']):.1f},{py(p['win_rate']):.1f}" for p in points
    )
    dots = "\n".join(
        f'<circle cx="{px(p["episode"]):.1f}" cy="{py(p["win_rate"]):.1f}" r="3.5" class="chart-dot">'
        f"<title>ep {p['episode']}: {_pct(p['win_rate'])}</title></circle>"
        for p in points
    )
    grid = "\n".join(
        f'<line x1="{pad_l}" y1="{py(v):.1f}" x2="{width - pad_r}" y2="{py(v):.1f}" class="chart-grid"/>'
        f'<text x="{pad_l - 6}" y="{py(v) + 4:.1f}" text-anchor="end" class="chart-axis">{int(v * 100)}%</text>'
        for v in (0.0, 0.5, 1.0)
    )
    return f"""<svg viewBox="0 0 {width} {height}" class="chart">
  {grid}
  <polyline points="{poly}" class="chart-line" fill="none"/>
  {dots}
</svg>"""


def render_unified_random_matchups_html(
    state: dict[str, Any],
    *,
    auto_refresh_seconds: Optional[float] = None,
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    refresh_tag = ""
    if auto_refresh_seconds and not state.get("complete"):
        refresh_tag = (
            f'<meta http-equiv="refresh" content="{max(1, int(auto_refresh_seconds))}">'
        )

    matchups_total = int(state.get("matchups_total") or 0)
    matchups_completed = int(state.get("matchups_completed") or 0)
    target_eps = int(state.get("target_episodes") or 0)
    overall_pct = float(state.get("overall_pct") or 0.0)
    batch_index = int(state.get("batch_index") or 0)
    status = str(state.get("status") or "training")
    status_label = {
        "training": "Training",
        "between_matchups": "Between matchups",
        "complete": "Complete",
    }.get(status, status.title())

    progress_headline = (
        f"Matchups {matchups_completed}/{matchups_total} · "
        f"{overall_pct:.1f}%"
    )
    if batch_index > 0:
        progress_headline += f" · Batch {batch_index}"

    active_matchups = state.get("active_matchups") or {}
    progress_blocks = ""
    if isinstance(active_matchups, dict) and active_matchups:
        for subdir, row in active_matchups.items():
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or subdir)
            eps = int(row.get("episodes_completed") or 0)
            pct = (eps / max(1, target_eps)) * 100.0 if target_eps else 0.0
            progress_blocks += f"""
      <div class="matchup-progress">
        <div class="progress-label"><span>{html.escape(name)}</span><span>{eps}/{target_eps}</span></div>
        <div class="progress-bar"><div class="progress-fill" style="width:{pct:.1f}%"></div></div>
      </div>"""
    else:
        current_eps = int(state.get("episodes_completed") or 0)
        name = str(state.get("current_matchup") or "—")
        pct = (current_eps / max(1, target_eps)) * 100.0 if target_eps else 0.0
        progress_blocks = f"""
      <div class="matchup-progress">
        <div class="progress-label"><span>{html.escape(name)}</span><span>{current_eps}/{target_eps}</span></div>
        <div class="progress-bar"><div class="progress-fill" style="width:{pct:.1f}%"></div></div>
      </div>"""

    cpp_live_path = str(state.get("cpp_eval_live_dashboard_path") or "").strip()
    replay_heading = format_checkpoint_eval_replay_heading(
        str(state.get("checkpoint_eval_replay_label") or "")
    )
    if cpp_live_path:
        cpp_live_link = (
            f'<p class="muted">{html.escape(replay_heading)}: '
            f'<a href="{html.escape(CPP_EVAL_LIVE_DASHBOARD)}">'
            f"{html.escape(CPP_EVAL_LIVE_DASHBOARD)}</a></p>"
        )
    else:
        cpp_live_link = (
            f'<p class="muted">{html.escape(replay_heading)} appears here during checkpoint eval '
            f"({html.escape(CPP_EVAL_LIVE_DASHBOARD)}).</p>"
        )

    ckpt_chart = _svg_winrate_chart_with_error_bands(
        state.get("checkpoint_aggregate_points") or [],
        target_episodes=target_eps,
    )

    completed_rows = state.get("completed_matchups") or []
    if completed_rows:
        history_rows = "\n".join(
            f"<tr>"
            f"<td>{html.escape(str(row.get('name', '')))}</td>"
            f"<td>{_pct(row.get('first_checkpoint_win_rate'))}</td>"
            f"<td>{_pct(row.get('checkpoint_win_rate'))}</td>"
            f"<td>{_pct(row.get('checkpoint_vs_logic_win_rate'))}</td>"
            f"</tr>"
            for row in completed_rows
        )
        history_table = f"""
<table class="history">
  <thead><tr><th>Matchup</th><th>First self-play ckpt</th><th>Last self-play ckpt</th><th>Last vs logic</th></tr></thead>
  <tbody>{history_rows}</tbody>
</table>"""
    else:
        history_table = '<p class="muted">No completed matchups yet.</p>'

    ckpt_rows = ""
    for row in state.get("active_checkpoint_rows") or []:
        ckpt_rows += (
            f"<tr>"
            f"<td>{html.escape(str(row.get('matchup') or '—'))}</td>"
            f"<td>{int(row.get('episode', 0))}</td>"
            f"<td>{_pct(row.get('vs_logic_win_rate'))}</td>"
            f"<td>{_pct(row.get('logic_vs_agent_win_rate'))}</td>"
            f"<td>{_pct(row.get('agent_vs_agent_win_rate'))}</td>"
            f"<td>{_pct(row.get('timeout_rate'))}</td></tr>"
        )
    ckpt_table = (
        f'<table class="history"><thead><tr>'
        f"<th>Matchup</th>"
        f"<th>Episode</th>"
        f"<th>Agent win% vs logic</th>"
        f"<th>Logic vs agent win%</th>"
        f"<th>Agent win% vs agent</th>"
        f"<th>Timeout %</th>"
        f"</tr></thead><tbody>{ckpt_rows or '<tr><td colspan=\"6\" class=\"muted\">No checkpoint eval yet</td></tr>'}</tbody></table>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Training AI agents with random matchups</title>
  {refresh_tag}
  <style>
    :root {{
      --bg: #0f1419; --surface: #1a2332; --border: #2d3a4d;
      --text: #e7ecf3; --muted: #8b9cb3; --primary: #5b9cff;
      --good: #3ecf8e; --warn: #f0b429;
      --font: "Segoe UI", system-ui, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: var(--font); }}
    .wrap {{ max-width: 960px; margin: 0 auto; padding: 28px 20px 40px; }}
    h1 {{ margin: 0 0 6px; font-size: 1.6rem; }}
    .sub {{ color: var(--muted); margin: 0 0 24px; line-height: 1.5; }}
    .progress-headline {{ font-size: 1.15rem; font-weight: 600; margin: 0 0 16px; }}
    .matchup-progress {{ margin-bottom: 14px; }}
    .matchup-progress:last-child {{ margin-bottom: 0; }}
    .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; margin-bottom: 18px; }}
    .card h2 {{ margin: 0 0 12px; font-size: 1.05rem; }}
    .progress-bar {{ height: 10px; background: #243044; border-radius: 999px; overflow: hidden; }}
    .progress-fill {{ height: 100%; background: linear-gradient(90deg, var(--primary), var(--good)); }}
    .progress-label {{ display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--muted); margin-bottom: 8px; }}
    .status-pill {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.75rem; background: #243044; color: var(--primary); }}
    .status-pill.complete {{ color: var(--good); }}
    .history {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    .history th, .history td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }}
    .history th {{ color: var(--muted); font-weight: 500; }}
    .muted {{ color: var(--muted); }}
    a {{ color: var(--primary); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .chart {{ width: 100%; max-width: 520px; }}
    .chart-selfplay-line {{ stroke: var(--primary); stroke-width: 2; }}
    .chart-selfplay-band {{ fill: rgba(91, 156, 255, 0.18); stroke: none; }}
    .chart-vslogic-line {{ stroke: var(--good); stroke-width: 2; }}
    .chart-vslogic-band {{ fill: rgba(62, 207, 142, 0.15); stroke: none; }}
    .chart-grid {{ stroke: #243044; stroke-width: 1; }}
    .chart-axis {{ fill: var(--muted); font-size: 10px; }}
    .chart-empty {{ fill: var(--muted); font-size: 12px; }}
    footer {{ margin-top: 28px; color: var(--muted); font-size: 0.75rem; border-top: 1px solid var(--border); padding-top: 14px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Training AI agents with random matchups</h1>
    <p class="sub">{html.escape(str(state.get('format', '—')).replace('_', ' '))} ·
      <span class="status-pill{' complete' if status == 'complete' else ''}">{html.escape(status_label)}</span>
    </p>

    <p class="progress-headline">{html.escape(progress_headline)}</p>

    <div class="card">
      <h2>Training progress</h2>
      {progress_blocks}
    </div>

    <div class="card">
      <h2>Checkpoint eval (active batch)</h2>
      {cpp_live_link}
      {ckpt_chart}
      {ckpt_table}
    </div>

    <div class="card">
      <h2>Completed matchups</h2>
      {history_table}
    </div>

    <footer>Generated {generated} · {html.escape(str(state.get('run_dir', '')))}</footer>
  </div>
</body>
</html>"""


def write_unified_random_matchups_dashboard(
    run_dir: Path,
    *,
    auto_refresh_seconds: Optional[float] = None,
) -> Optional[Path]:
    """Collect state and write ``unified_random_matchups_dashboard.html``."""
    run_dir = run_dir.expanduser().resolve()
    if not is_unified_random_matchup_run(run_dir):
        return None
    with _run_state_lock(run_dir):
        state = collect_unified_run_state(run_dir)
        page = render_unified_random_matchups_html(
            state,
            auto_refresh_seconds=auto_refresh_seconds,
        )
        html_path = run_dir / UNIFIED_DASHBOARD_NAME
        html_path.write_text(page, encoding="utf-8")
        return html_path


def maybe_refresh_unified_dashboard(
    run_dir: Path,
    *,
    auto_refresh_seconds: float = 5.0,
    min_interval_seconds: float = 4.0,
) -> Optional[Path]:
    """Throttle dashboard regeneration during hot training loops."""
    if not is_unified_random_matchup_run(run_dir):
        return None
    key = str(run_dir.resolve())
    now = time.monotonic()
    last = _last_dashboard_write.get(key, 0.0)
    if now - last < min_interval_seconds:
        return None
    _last_dashboard_write[key] = now
    return write_unified_random_matchups_dashboard(
        run_dir,
        auto_refresh_seconds=auto_refresh_seconds,
    )
