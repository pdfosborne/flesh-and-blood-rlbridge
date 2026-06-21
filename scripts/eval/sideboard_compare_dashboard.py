#!/usr/bin/env python3
"""Live single-page HTML dashboard for sideboard comparison training.

Polls ``train_sideboard_compare.py`` output directories and regenerates
``sideboard_compare_dashboard.html`` with candidate swaps, training progress,
checkpoint eval win-rate charts, and ETA estimates.

Typical usage (alongside training)::

    python scripts/eval/sideboard_compare_dashboard.py \\
        --out-dir results/sideboard_compare_aurora_vs_briar \\
        --watch --poll-seconds 5
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_ROOT))
import _bootstrap  # noqa: E402

REPO_ROOT = _bootstrap.configure_paths()

_PITCH_SUFFIX = re.compile(r"_(red|blue|yellow|purple)$")
_DASHBOARD_NAME = "sideboard_compare_dashboard.html"


def _slug_to_display_name(card_id: str) -> str:
    token = str(card_id or "").strip()
    if not token:
        return "?"
    pitch = ""
    match = _PITCH_SUFFIX.search(token)
    if match:
        pitch = f" ({match.group(1).title()})"
        token = token[: match.start()]
    words = [w for w in token.replace("-", "_").split("_") if w]
    return " ".join(w.capitalize() for w in words) + pitch


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _episode_num_from_dir(path: Path) -> int:
    name = path.name
    if name.startswith("episode_"):
        try:
            return int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            return 0
    return 0


def _glob_checkpoint_roots(candidate_dir: Path) -> list[Path]:
    roots = sorted(candidate_dir.glob("p3_*/p1"))
    if roots:
        return roots
    return sorted(candidate_dir.glob("*/p1"))


def _latest_training_metadata(candidate_dir: Path) -> Optional[dict[str, Any]]:
    meta_paths: list[Path] = []
    for root in _glob_checkpoint_roots(candidate_dir):
        meta_paths.extend(root.glob("episode_*/metadata.json"))
    if not meta_paths:
        return None
    latest = max(meta_paths, key=_episode_num_from_dir)
    data = _read_json(latest)
    return data if isinstance(data, dict) else None


def _training_win_series(candidate_dir: Path) -> list[dict[str, Any]]:
    """Cumulative training win rate at each saved play checkpoint."""
    history_path = candidate_dir / "play_training_history.json"
    if history_path.is_file():
        raw = _read_json(history_path)
        if isinstance(raw, list):
            return [
                {
                    "episode": int(row.get("episodes_completed", 0) or 0),
                    "win_rate": float(row.get("win_rate", 0.0) or 0.0),
                    "wins": int(row.get("wins", 0) or 0),
                    "losses": int(row.get("losses", 0) or 0),
                    "draws": int(row.get("draws", 0) or 0),
                    "timeouts": int(row.get("timeouts", 0) or 0),
                }
                for row in raw
                if isinstance(row, dict)
            ]

    points: list[dict[str, Any]] = []
    for root in _glob_checkpoint_roots(candidate_dir):
        for meta_path in root.glob("episode_*/metadata.json"):
            data = _read_json(meta_path)
            if not isinstance(data, dict):
                continue
            wr = data.get("win_rate")
            if wr is None:
                continue
            ep = int(
                data.get("episodes_completed")
                or _episode_num_from_dir(meta_path.parent)
            )
            points.append({
                "episode": ep,
                "win_rate": float(wr),
                "wins": int(data.get("wins", 0) or 0),
                "losses": int(data.get("losses", 0) or 0),
                "draws": int(data.get("draws", 0) or 0),
                "timeouts": int(data.get("timeouts", 0) or 0),
            })
    points.sort(key=lambda row: int(row.get("episode", 0) or 0))
    return points


def _live_training_progress(candidate_dir: Path) -> Optional[dict[str, Any]]:
    live_path = candidate_dir / "play_training_live.json"
    if live_path.is_file():
        data = _read_json(live_path)
        if isinstance(data, dict):
            return data
    return None


def _checkpoint_eval_series(candidate_dir: Path) -> list[dict[str, Any]]:
    history_path = candidate_dir / "checkpoint_eval_history.json"
    if history_path.is_file():
        raw = _read_json(history_path)
        if isinstance(raw, list):
            return [row for row in raw if isinstance(row, dict)]

    points: list[dict[str, Any]] = []
    for root in _glob_checkpoint_roots(candidate_dir):
        for eval_path in root.glob("episode_*/checkpoint_eval.json"):
            row = _read_json(eval_path)
            if not isinstance(row, dict):
                continue
            ep = int(row.get("episodes_completed") or _episode_num_from_dir(eval_path.parent))
            row = dict(row)
            row["episodes_completed"] = ep
            points.append(row)
    points.sort(key=lambda row: int(row.get("episodes_completed", 0) or 0))
    return points


def _candidate_status(
    *,
    result: Optional[dict[str, Any]],
    training_meta: Optional[dict[str, Any]],
    skip_final_eval: bool,
) -> str:
    if result is not None:
        if skip_final_eval or result.get("final_eval_win_rate") is not None:
            return "complete"
        if result.get("play_win_rate") is not None:
            return "final_eval"
    if training_meta is not None:
        done = int(training_meta.get("episodes_completed", 0) or 0)
        target = int(training_meta.get("target_episodes", 0) or 0)
        if target > 0 and done >= target:
            return "final_eval"
        if done > 0:
            return "training"
    return "pending"


def _format_duration(seconds: float) -> str:
    if seconds < 0 or not (seconds < float("inf")):
        return "—"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _parse_started_at(manifest: dict[str, Any], out_dir: Path) -> Optional[datetime]:
    raw = manifest.get("started_at")
    if raw:
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            pass
    manifest_path = out_dir / "candidates_manifest.json"
    if manifest_path.is_file():
        try:
            return datetime.fromtimestamp(
                manifest_path.stat().st_mtime,
                tz=timezone.utc,
            )
        except OSError:
            pass
    return None


def _estimate_eta(
    *,
    started_at: Optional[datetime],
    completed_units: float,
    total_units: float,
) -> tuple[Optional[float], str]:
    if started_at is None or completed_units <= 0 or total_units <= 0:
        return None, "—"
    now = datetime.now(timezone.utc)
    elapsed = (now - started_at).total_seconds()
    if elapsed <= 0:
        return None, "—"
    rate = completed_units / elapsed
    remaining = max(0.0, total_units - completed_units)
    if rate <= 0:
        return None, "—"
    eta_seconds = remaining / rate
    return eta_seconds, _format_duration(eta_seconds)


def collect_sideboard_compare_state(out_dir: Path) -> dict[str, Any]:
    """Scan a sideboard compare output directory into dashboard-ready state."""
    out_dir = out_dir.expanduser().resolve()
    manifest_path = out_dir / "candidates_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    if not isinstance(manifest, dict):
        manifest = {}

    summary_path = out_dir / "sideboard_compare_results.json"
    summary = _read_json(summary_path) if summary_path.is_file() else None
    if not isinstance(summary, dict):
        summary = None

    play_episodes = int(manifest.get("play_episodes", 0) or 0)
    final_eval_episodes = int(manifest.get("final_eval_episodes", 0) or 0)
    default_checkpoint_eval_episodes = int(
        manifest.get("checkpoint_eval_episodes", 0) or 0
    )
    skip_final_eval = bool(manifest.get("skip_final_eval", False))
    started_at = _parse_started_at(manifest, out_dir)

    raw_candidates = manifest.get("candidates") or []
    if not isinstance(raw_candidates, list):
        raw_candidates = []

    candidates_dir = out_dir / "candidates"
    candidate_rows: list[dict[str, Any]] = []
    completed_units = 0.0
    total_units = 0.0

    for raw in raw_candidates:
        if not isinstance(raw, dict):
            continue
        cid = str(raw.get("candidate_id", ""))
        label = str(raw.get("label", cid))
        candidate_dir = candidates_dir / cid if cid else None

        result: Optional[dict[str, Any]] = None
        training_meta: Optional[dict[str, Any]] = None
        live_training: Optional[dict[str, Any]] = None
        eval_series: list[dict[str, Any]] = []
        train_series: list[dict[str, Any]] = []
        if candidate_dir and candidate_dir.is_dir():
            result_path = candidate_dir / "candidate_result.json"
            if result_path.is_file():
                loaded = _read_json(result_path)
                if isinstance(loaded, dict):
                    result = loaded
            training_meta = _latest_training_metadata(candidate_dir)
            live_training = _live_training_progress(candidate_dir)
            eval_series = _checkpoint_eval_series(candidate_dir)
            train_series = _training_win_series(candidate_dir)

        status = _candidate_status(
            result=result,
            training_meta=training_meta,
            skip_final_eval=skip_final_eval,
        )

        train_done = 0
        train_target = play_episodes
        if live_training:
            train_done = int(live_training.get("episodes_completed", 0) or 0)
            train_target = int(live_training.get("target_episodes", play_episodes) or play_episodes)
        elif training_meta:
            train_done = int(training_meta.get("episodes_completed", 0) or 0)
            train_target = int(training_meta.get("target_episodes", play_episodes) or play_episodes)
        elif result is not None:
            train_done = play_episodes

        candidate_units = play_episodes
        if not skip_final_eval:
            candidate_units += final_eval_episodes
        total_units += candidate_units

        done_units = min(train_done, play_episodes)
        if not skip_final_eval and result and result.get("final_eval_win_rate") is not None:
            done_units += final_eval_episodes
        elif not skip_final_eval and status == "final_eval":
            done_units += final_eval_episodes * 0.35
        completed_units += done_units

        swaps_raw = raw.get("swaps") or []
        swaps: list[dict[str, str]] = []
        if isinstance(swaps_raw, list):
            for pair in swaps_raw:
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                out_id, in_id = str(pair[0]), str(pair[1])
                swaps.append({
                    "out_id": out_id,
                    "in_id": in_id,
                    "out_name": _slug_to_display_name(out_id),
                    "in_name": _slug_to_display_name(in_id),
                })

        chart_points = [
            _chart_point_with_stderr(
                episode=int(pt.get("episodes_completed", 0) or 0),
                win_rate=float(pt.get("p1_win_rate", 0.0) or 0.0),
                wins=(
                    int(pt["p1_wins"])
                    if "p1_wins" in pt and pt.get("p1_wins") is not None
                    else None
                ),
                losses=int(pt.get("losses", 0) or 0) if "losses" in pt else None,
                draws=int(pt.get("draws", 0) or 0) if "draws" in pt else None,
                timeouts=int(pt.get("timeouts", 0) or 0) if "timeouts" in pt else None,
                n=(
                    int(pt.get("eval_episodes") or pt.get("episodes") or 0)
                    or default_checkpoint_eval_episodes
                    or None
                ),
            )
            for pt in eval_series
        ]
        eval_win_rates = [float(pt["win_rate"]) for pt in chart_points]

        # Import here so dashboard works when only scripts/eval is on sys.path.
        _training_root = _SCRIPTS_ROOT / "training"
        if str(_training_root) not in sys.path:
            sys.path.insert(0, str(_training_root))
        from play_outcome_stats import compute_eval_stability  # noqa: E402

        eval_stability = compute_eval_stability(
            eval_win_rates,
            episodes_completed=train_done if train_done else None,
            target_episodes=train_target if train_target else None,
        )
        train_chart_points = [
            _chart_point_with_stderr(
                episode=int(pt.get("episode", 0) or 0),
                win_rate=float(pt.get("win_rate", 0.0) or 0.0),
                wins=int(pt.get("wins", 0) or 0) if "wins" in pt else None,
                losses=int(pt.get("losses", 0) or 0) if "losses" in pt else None,
                draws=int(pt.get("draws", 0) or 0) if "draws" in pt else None,
                timeouts=int(pt.get("timeouts", 0) or 0) if "timeouts" in pt else None,
            )
            for pt in train_series
        ]

        live_train_wr = None
        train_wins = train_losses = train_draws = train_timeouts = None
        if live_training is not None:
            if live_training.get("win_rate") is not None:
                live_train_wr = float(live_training["win_rate"])
            train_wins = int(live_training.get("wins", 0) or 0)
            train_losses = int(live_training.get("losses", 0) or 0)
            train_draws = int(live_training.get("draws", 0) or 0)
            train_timeouts = int(live_training.get("timeouts", 0) or 0)
        elif training_meta is not None and training_meta.get("win_rate") is not None:
            live_train_wr = float(training_meta["win_rate"])
            train_wins = int(training_meta.get("wins", 0) or 0)
            train_losses = int(training_meta.get("losses", 0) or 0)
            train_draws = int(training_meta.get("draws", 0) or 0)
            train_timeouts = int(training_meta.get("timeouts", 0) or 0)
        elif train_chart_points:
            live_train_wr = float(train_chart_points[-1]["win_rate"])

        play_win_rate = None
        if result and result.get("play_win_rate") is not None:
            play_win_rate = float(result["play_win_rate"])
        elif live_train_wr is not None:
            play_win_rate = live_train_wr

        train_engine_label = _resolve_train_engine_label(
            live_training=live_training,
            training_meta=training_meta,
            manifest=manifest,
        )
        eval_engine_label = _resolve_eval_engine_label(
            eval_series=eval_series,
            manifest=manifest,
            default_checkpoint_eval_episodes=default_checkpoint_eval_episodes,
        )

        candidate_rows.append({
            "candidate_id": cid,
            "label": label,
            "status": status,
            "swaps": swaps,
            "guide_margin": raw.get("guide_margin"),
            "train_done": train_done,
            "train_target": train_target,
            "train_pct": (train_done / train_target * 100.0) if train_target else 0.0,
            "play_win_rate": play_win_rate,
            "train_wins": train_wins,
            "train_losses": train_losses,
            "train_draws": train_draws,
            "train_timeouts": train_timeouts,
            "latest_checkpoint_timeouts": (
                int(eval_series[-1].get("timeouts", 0) or 0) if eval_series else None
            ),
            "latest_checkpoint_win_rate": (
                chart_points[-1]["win_rate"] if chart_points else None
            ),
            "final_eval_win_rate": (
                result.get("final_eval_win_rate") if result else None
            ),
            "final_eval_delta": (
                result.get("final_eval_delta_vs_baseline") if result else None
            ),
            "chart_points": chart_points,
            "train_chart_points": train_chart_points,
            "train_engine_label": train_engine_label,
            "eval_engine_label": eval_engine_label,
            "eval_stability": eval_stability,
        })

    eta_seconds, eta_label = _estimate_eta(
        started_at=started_at,
        completed_units=completed_units,
        total_units=total_units,
    )
    overall_pct = (completed_units / total_units * 100.0) if total_units else 0.0

    winner = None
    if summary and isinstance(summary.get("winner"), dict):
        winner = summary["winner"].get("candidate_id")

    baseline_final = None
    for row in candidate_rows:
        if row.get("candidate_id") == "baseline":
            baseline_final = row.get("final_eval_win_rate")
            break
    if baseline_final is None:
        for row in candidate_rows:
            fe = row.get("final_eval_win_rate")
            if fe is not None:
                baseline_final = fe
                break
    if baseline_final is not None:
        for row in candidate_rows:
            fe = row.get("final_eval_win_rate")
            if fe is not None and row.get("final_eval_delta") is None:
                row["final_eval_delta"] = float(fe) - float(baseline_final)

    return {
        "out_dir": str(out_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at.isoformat() if started_at else None,
        "hero_id": manifest.get("hero_id", ""),
        "opponent_hero_id": manifest.get("opponent_hero_id", ""),
        "opponent_deck": manifest.get("opponent_deck", ""),
        "format": manifest.get("format", ""),
        "play_episodes": play_episodes,
        "final_eval_episodes": final_eval_episodes,
        "skip_final_eval": skip_final_eval,
        "max_parallel": int(manifest.get("max_parallel", 1) or 1),
        "checkpoint_interval": manifest.get("checkpoint_interval"),
        "checkpoint_eval_episodes": manifest.get("checkpoint_eval_episodes"),
        "cpp_engine_dir": manifest.get("cpp_engine_dir"),
        "complete": summary is not None,
        "winner_id": winner,
        "overall_pct": overall_pct,
        "eta_seconds": eta_seconds,
        "eta_label": eta_label,
        "candidates": candidate_rows,
    }


_STATUS_LABELS = {
    "pending": "Queued",
    "training": "Training",
    "final_eval": "Final eval",
    "complete": "Complete",
}


def _pct_text(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.1f}%"


def _chart_engine_label(raw: Any) -> str:
    """Normalize persisted backend labels to dashboard chart titles."""
    token = str(raw or "").strip().lower()
    if not token:
        return ""
    if "cpp" in token or "c++" in token:
        return "cpp engine"
    if "talishar" in token or "http" in token:
        return "talishar engine"
    return ""


def _resolve_train_engine_label(
    *,
    live_training: Optional[dict[str, Any]],
    training_meta: Optional[dict[str, Any]],
    manifest: dict[str, Any],
) -> str:
    for src in (live_training, training_meta):
        if isinstance(src, dict) and src.get("runtime_backend"):
            label = _chart_engine_label(src["runtime_backend"])
            if label:
                return label
    cpp_dir = None
    if isinstance(training_meta, dict):
        cpp_dir = training_meta.get("cpp_engine_dir")
    if not cpp_dir:
        cpp_dir = manifest.get("cpp_engine_dir")
    if cpp_dir and (live_training or training_meta):
        return "cpp engine"
    if live_training or training_meta:
        return "talishar engine"
    if manifest.get("cpp_engine_dir"):
        return "cpp engine"
    return ""


def _resolve_eval_engine_label(
    *,
    eval_series: list[dict[str, Any]],
    manifest: dict[str, Any],
    default_checkpoint_eval_episodes: int,
) -> str:
    for pt in reversed(eval_series):
        if isinstance(pt, dict) and pt.get("runtime_backend"):
            label = _chart_engine_label(pt["runtime_backend"])
            if label:
                return label
    if eval_series:
        return "cpp engine"
    if default_checkpoint_eval_episodes > 0 and manifest.get("cpp_engine_dir"):
        return "cpp engine"
    return ""


def _chart_title(base: str, engine_label: str) -> str:
    if engine_label:
        return f"{base} ({engine_label})"
    return base


def _chart_point_with_stderr(
    *,
    episode: int,
    win_rate: float,
    wins: Optional[int] = None,
    losses: Optional[int] = None,
    draws: Optional[int] = None,
    timeouts: Optional[int] = None,
    n: Optional[int] = None,
) -> dict[str, Any]:
    """Build a chart point with binomial standard error when sample size is known."""
    _training_root = _SCRIPTS_ROOT / "training"
    if str(_training_root) not in sys.path:
        sys.path.insert(0, str(_training_root))
    from play_outcome_stats import (  # noqa: E402
        win_rate_standard_error,
        win_rate_standard_error_from_rate,
    )

    total = int(n) if n else None
    if total is None and wins is not None:
        total = (
            int(wins)
            + int(losses or 0)
            + int(draws or 0)
            + int(timeouts or 0)
        )
    stderr: Optional[float] = None
    if total and total > 0:
        if wins is not None:
            stderr = win_rate_standard_error(int(wins), total)
        else:
            stderr = win_rate_standard_error_from_rate(win_rate, total)
    return {
        "episode": int(episode),
        "win_rate": float(win_rate),
        "n": total,
        "stderr": stderr,
    }


def _delta_text(value: Optional[float]) -> str:
    if value is None:
        return ""
    sign = "+" if value >= 0 else ""
    return f"{sign}{float(value) * 100:.1f}% vs baseline"


def _stability_class(status: Optional[str]) -> str:
    token = str(status or "").strip().lower()
    if token == "converged":
        return "stability-converged"
    if token == "learning":
        return "stability-learning"
    return "stability-insufficient"


def _render_stability_block(stability: Optional[dict[str, Any]]) -> str:
    if not stability or not isinstance(stability, dict):
        return (
            '<div><span class="metric-label">Eval stability</span>'
            '<span class="metric-value">—</span></div>'
        )
    status = str(stability.get("status", ""))
    label = html.escape(str(stability.get("label", "—")))
    detail = html.escape(str(stability.get("detail", "")))
    recommendation = html.escape(str(stability.get("recommendation", "")))
    css = _stability_class(status)
    sufficient = " · sufficient" if stability.get("sufficient") else ""
    return (
        f'<div><span class="metric-label">Eval stability</span>'
        f'<span class="metric-value {css}">{label}{html.escape(sufficient)}</span>'
        f'<span class="stability-detail">{detail}</span>'
        f'<span class="stability-tip">{recommendation}</span></div>'
    )


def _chart_point_tooltip(point: dict[str, Any]) -> str:
    ep = int(point.get("episode", 0) or 0)
    wr = _pct_text(point.get("win_rate"))
    stderr = point.get("stderr")
    if stderr is not None:
        n = point.get("n")
        n_text = f", n={int(n)}" if n else ""
        return f"ep {ep}: {wr} ± {_pct_text(stderr)}{n_text}"
    return f"ep {ep}: {wr}"


def _svg_winrate_chart(
    points: list[dict[str, Any]],
    *,
    width: int = 360,
    height: int = 130,
    target_episodes: int = 0,
    line_class: str = "chart-line",
    dot_class: str = "chart-dot",
    error_bar_class: str = "chart-error-bar",
    empty_message: str = "Waiting for checkpoint eval…",
    aria_label: str = "Win rate chart",
) -> str:
    if not points:
        return (
            f'<svg viewBox="0 0 {width} {height}" class="chart empty" '
            f'aria-label="{html.escape(empty_message)}">'
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
        clamped = max(0.0, min(1.0, float(rate)))
        return pad_t + (1.0 - clamped) * plot_h

    cap_half = 3.0
    error_bars: list[str] = []
    for point in points:
        stderr = point.get("stderr")
        if stderr is None or float(stderr) <= 0:
            continue
        x = px(point["episode"])
        rate = float(point["win_rate"])
        y_high = py(min(1.0, rate + float(stderr)))
        y_low = py(max(0.0, rate - float(stderr)))
        tooltip = html.escape(_chart_point_tooltip(point))
        error_bars.append(
            f'<g class="{error_bar_class}">'
            f'<line x1="{x:.1f}" y1="{y_high:.1f}" x2="{x:.1f}" y2="{y_low:.1f}"/>'
            f'<line x1="{x - cap_half:.1f}" y1="{y_high:.1f}" x2="{x + cap_half:.1f}" y2="{y_high:.1f}"/>'
            f'<line x1="{x - cap_half:.1f}" y1="{y_low:.1f}" x2="{x + cap_half:.1f}" y2="{y_low:.1f}"/>'
            f"<title>{tooltip}</title></g>"
        )
    error_bar_svg = "\n  ".join(error_bars)

    poly = " ".join(
        f"{px(p['episode']):.1f},{py(p['win_rate']):.1f}" for p in points
    )
    dots = "\n".join(
        f'<circle cx="{px(p["episode"]):.1f}" cy="{py(p["win_rate"]):.1f}" r="3.5" '
        f'class="{dot_class}"><title>{html.escape(_chart_point_tooltip(p))}</title></circle>'
        for p in points
    )
    grid = "\n".join(
        f'<line x1="{pad_l}" y1="{py(v):.1f}" x2="{width - pad_r}" y2="{py(v):.1f}" '
        f'class="chart-grid"/>'
        f'<text x="{pad_l - 6}" y="{py(v) + 4:.1f}" text-anchor="end" class="chart-axis">'
        f'{int(v * 100)}%</text>'
        for v in (0.0, 0.5, 1.0)
    )
    return f"""<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{html.escape(aria_label)}">
  {grid}
  <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" y2="{pad_t + plot_h}" class="chart-axis-line"/>
  {error_bar_svg}
  <polyline points="{poly}" class="{line_class}"/>
  {dots}
</svg>"""


def _render_swaps(swaps: list[dict[str, str]]) -> str:
    if not swaps:
        return '<p class="swap-none">No swaps (baseline deck)</p>'
    rows = []
    for swap in swaps:
        rows.append(
            f'<div class="swap-row">'
            f'<span class="swap-out" title="{html.escape(swap["out_id"])}">'
            f'{html.escape(swap["out_name"])}</span>'
            f'<span class="swap-arrow">→</span>'
            f'<span class="swap-in" title="{html.escape(swap["in_id"])}">'
            f'{html.escape(swap["in_name"])}</span>'
            f"</div>"
        )
    return "\n".join(rows)


def _render_candidate_card(row: dict[str, Any], *, play_episodes: int) -> str:
    status = str(row.get("status", "pending"))
    status_label = _STATUS_LABELS.get(status, status.title())
    winner = row.get("is_winner", False)
    card_class = "candidate-card"
    if winner:
        card_class += " winner"
    if status == "training":
        card_class += " active"

    margin = row.get("guide_margin")
    margin_html = ""
    if margin is not None:
        margin_html = f'<span class="meta-pill">guide margin {float(margin):.2f}</span>'

    train_engine = str(row.get("train_engine_label") or "")
    eval_engine = str(row.get("eval_engine_label") or "")
    train_chart_title = _chart_title("Training win rate", train_engine)
    eval_chart_title = _chart_title("Checkpoint eval win rate", eval_engine)

    chart = _svg_winrate_chart(
        row.get("chart_points") or [],
        target_episodes=play_episodes,
        empty_message=_chart_title("Waiting for checkpoint eval…", eval_engine),
        aria_label=eval_chart_title,
    )
    train_chart = _svg_winrate_chart(
        row.get("train_chart_points") or [],
        target_episodes=play_episodes,
        line_class="chart-line chart-line-train",
        dot_class="chart-dot chart-dot-train",
        error_bar_class="chart-error-bar chart-error-bar-train",
        empty_message=_chart_title("Waiting for training checkpoint…", train_engine),
        aria_label=train_chart_title,
    )
    delta = _delta_text(row.get("final_eval_delta"))
    delta_html = f'<span class="delta">{html.escape(delta)}</span>' if delta else ""

    record_bits = []
    if row.get("train_wins") is not None:
        record_bits.append(f"{int(row['train_wins'])}W")
    if row.get("train_losses") is not None:
        record_bits.append(f"{int(row['train_losses'])}L")
    if row.get("train_draws") is not None:
        record_bits.append(f"{int(row['train_draws'])}D")
    if row.get("train_timeouts") is not None:
        record_bits.append(f"{int(row['train_timeouts'])}T")
    ckpt_timeouts = row.get("latest_checkpoint_timeouts")
    ckpt_record = ""
    if ckpt_timeouts is not None:
        ckpt_record = f" · ckpt eval incl. {int(ckpt_timeouts)}T"
    train_record = " · ".join(record_bits)
    train_record_html = (
        f'<span class="train-record">{html.escape(train_record)}{html.escape(ckpt_record)}</span>'
        if train_record or ckpt_record else ""
    )

    return f"""
<article class="{card_class}">
  <header class="candidate-head">
    <div>
      <h3>{html.escape(str(row.get("label", "")))}</h3>
      <p class="candidate-id">{html.escape(str(row.get("candidate_id", "")))}</p>
    </div>
    <span class="status status-{html.escape(status)}">{html.escape(status_label)}</span>
  </header>
  <section class="swaps">
    <h4>Sideboard changes</h4>
    {_render_swaps(row.get("swaps") or [])}
    {margin_html}
  </section>
  <section class="progress-block">
    <div class="progress-label">
      <span>Training</span>
      <span>{int(row.get("train_done", 0))}/{int(row.get("train_target", 0))} episodes</span>
    </div>
    <div class="progress-bar"><div class="progress-fill" style="width:{float(row.get("train_pct", 0)):.1f}%"></div></div>
  </section>
  <section class="metrics">
    <div><span class="metric-label">Train win%</span><span class="metric-value">{_pct_text(row.get("play_win_rate"))}</span>{train_record_html}</div>
    <div><span class="metric-label">Checkpoint eval</span><span class="metric-value">{_pct_text(row.get("latest_checkpoint_win_rate"))}</span></div>
    {_render_stability_block(row.get("eval_stability"))}
    <div><span class="metric-label">Final eval</span><span class="metric-value">{_pct_text(row.get("final_eval_win_rate"))} {delta_html}</span></div>
  </section>
  <section class="chart-block">
    <h4>{html.escape(train_chart_title)}</h4>
    {train_chart}
  </section>
  <section class="chart-block">
    <h4>{html.escape(eval_chart_title)}</h4>
    {chart}
  </section>
</article>"""


def render_sideboard_compare_html(
    state: dict[str, Any],
    *,
    auto_refresh_seconds: Optional[float] = None,
) -> str:
    """Render the single-page dashboard HTML for *state*."""
    hero = html.escape(str(state.get("hero_id", "?")))
    opponent = html.escape(str(state.get("opponent_hero_id", "?")))
    opp_deck = html.escape(str(state.get("opponent_deck", "")))
    game_format = html.escape(str(state.get("format", "")))
    generated = html.escape(str(state.get("generated_at", "")))
    out_dir = html.escape(str(state.get("out_dir", "")))
    play_episodes = int(state.get("play_episodes", 0) or 0)

    winner_id = state.get("winner_id")
    complete = bool(state.get("complete"))
    status_banner = "Complete" if complete else "In progress"
    if winner_id:
        status_banner += f" — winner: {html.escape(str(winner_id))}"

    refresh_meta = ""
    if auto_refresh_seconds and not complete:
        refresh_meta = (
            f'<meta http-equiv="refresh" content="{max(1, int(auto_refresh_seconds))}">'
        )

    candidates = state.get("candidates") or []
    cards_html = "\n".join(
        _render_candidate_card(
            {**row, "is_winner": row.get("candidate_id") == winner_id},
            play_episodes=play_episodes,
        )
        for row in candidates
        if isinstance(row, dict)
    )

    ckpt_info = ""
    if state.get("checkpoint_interval"):
        ckpt_info = (
            f"Checkpoints every {int(state['checkpoint_interval'])} ep · "
            f"{int(state.get('checkpoint_eval_episodes') or 0)} eval games"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {refresh_meta}
  <title>Sideboard Compare — {hero} vs {opponent}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0f1419;
      --surface: #1a2332;
      --surface-variant: #243044;
      --text: #e8eef7;
      --muted: #93a4bd;
      --primary: #5b9cff;
      --primary-light: rgba(91, 156, 255, 0.16);
      --good: #3ecf8e;
      --good-light: rgba(62, 207, 142, 0.14);
      --warn: #f0b429;
      --warn-light: rgba(240, 180, 41, 0.14);
      --bad: #f07178;
      --bad-light: rgba(240, 113, 120, 0.14);
      --border: #2f3f56;
      --shadow-1: 0 1px 3px rgba(0, 0, 0, 0.35), 0 1px 2px rgba(0, 0, 0, 0.25);
      --shadow-2: 0 2px 6px rgba(0, 0, 0, 0.45), 0 1px 3px rgba(0, 0, 0, 0.3);
      --radius: 4px;
      --font: "JetBrains Mono", "Cascadia Code", "Fira Code", "Consolas", "Liberation Mono", ui-monospace, monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--font);
      font-size: 13px;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 24px 20px 48px; }}
    h1 {{
      margin: 0 0 8px;
      font-size: 1.35rem;
      font-weight: 600;
      letter-spacing: -0.02em;
    }}
    h3 {{ margin: 0; font-size: 0.95rem; font-weight: 600; }}
    h4 {{
      margin: 0 0 8px;
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .sub {{ color: var(--muted); margin: 0 0 20px; font-size: 0.85rem; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .stat {{
      background: var(--surface);
      border-radius: var(--radius);
      padding: 16px;
      box-shadow: var(--shadow-1);
    }}
    .stat-label {{
      display: block;
      color: var(--muted);
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .stat-value {{ font-size: 1.25rem; font-weight: 600; margin-top: 6px; }}
    .overall {{
      background: var(--surface);
      border-radius: var(--radius);
      padding: 16px 18px;
      margin-bottom: 20px;
      box-shadow: var(--shadow-1);
    }}
    .progress-bar {{
      height: 4px;
      background: var(--surface-variant);
      border-radius: 2px;
      overflow: hidden;
    }}
    .progress-fill {{
      height: 100%;
      background: var(--primary);
      border-radius: 2px;
      transition: width 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .progress-label {{
      display: flex;
      justify-content: space-between;
      font-size: 0.82rem;
      color: var(--muted);
      margin-bottom: 8px;
      font-weight: 500;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 16px;
    }}
    .candidate-card {{
      background: var(--surface);
      border-radius: var(--radius);
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      box-shadow: var(--shadow-1);
      border-left: 4px solid transparent;
    }}
    .candidate-card.active {{
      border-left-color: var(--primary);
      box-shadow: var(--shadow-2);
    }}
    .candidate-card.winner {{
      border-left-color: var(--good);
      background: linear-gradient(90deg, var(--good-light) 0%, var(--surface) 12%);
    }}
    .candidate-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: start; }}
    .candidate-id {{ margin: 4px 0 0; color: var(--muted); font-size: 0.78rem; }}
    .status {{
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 4px 10px;
      border-radius: var(--radius);
      white-space: nowrap;
    }}
    .status-training {{ color: var(--primary); background: var(--primary-light); }}
    .status-final_eval {{ color: var(--warn); background: var(--warn-light); }}
    .status-complete {{ color: var(--good); background: var(--good-light); }}
    .status-pending {{ color: var(--muted); background: var(--surface-variant); }}
    .swap-row {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 6px; }}
    .swap-out {{ color: var(--bad); font-weight: 500; }}
    .swap-in {{ color: var(--good); font-weight: 500; }}
    .swap-arrow {{ color: var(--muted); }}
    .swap-none {{ color: var(--muted); margin: 0; font-size: 0.85rem; }}
    .meta-pill {{
      display: inline-block;
      margin-top: 8px;
      font-size: 0.72rem;
      color: var(--muted);
      background: var(--surface-variant);
      padding: 4px 10px;
      border-radius: var(--radius);
      font-weight: 500;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
    }}
    .metrics > div {{
      background: var(--surface-variant);
      border-radius: var(--radius);
      padding: 10px 12px;
    }}
    .metric-label {{
      display: block;
      font-size: 0.68rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 600;
    }}
    .metric-value {{ font-size: 0.92rem; font-weight: 600; margin-top: 2px; }}
    .stability-converged {{ color: var(--good); }}
    .stability-learning {{ color: var(--warn); }}
    .stability-insufficient {{ color: var(--muted); }}
    .stability-detail, .stability-tip {{
      display: block;
      font-size: 0.72rem;
      color: var(--muted);
      font-weight: 400;
      margin-top: 4px;
      line-height: 1.4;
    }}
    .delta {{ font-size: 0.75rem; color: var(--muted); font-weight: 500; }}
    .chart-block {{
      background: var(--surface-variant);
      border-radius: var(--radius);
      padding: 12px;
    }}
    .chart {{ width: 100%; height: auto; display: block; }}
    .chart-grid {{ stroke: #2f3f56; stroke-width: 1; }}
    .chart-axis-line {{ stroke: #3d4f66; stroke-width: 1; }}
    .chart-axis {{ fill: var(--muted); font-size: 10px; font-family: var(--font); }}
    .chart-line {{
      fill: none;
      stroke: var(--primary);
      stroke-width: 2;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .chart-line-train {{ stroke: var(--good); }}
    .chart-error-bar {{ stroke: rgba(91, 156, 255, 0.65); stroke-width: 1.5; }}
    .chart-error-bar-train {{ stroke: rgba(62, 207, 142, 0.65); }}
    .chart-dot {{ fill: var(--primary); stroke: var(--surface); stroke-width: 1.5; }}
    .chart-dot-train {{ fill: var(--good); stroke: var(--surface); stroke-width: 1.5; }}
    .train-record {{ display: block; font-size: 0.72rem; color: var(--muted); font-weight: 400; margin-top: 4px; }}
    .chart-empty {{ fill: var(--muted); font-size: 11px; font-family: var(--font); }}
    footer {{
      margin-top: 28px;
      color: var(--muted);
      font-size: 0.75rem;
      padding-top: 16px;
      border-top: 1px solid var(--border);
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Sideboard comparison dashboard</h1>
    <p class="sub">{hero} vs {opponent} ({opp_deck}) · {game_format} · {html.escape(status_banner)}</p>

    <div class="summary">
      <div class="stat"><span class="stat-label">Overall progress</span><span class="stat-value">{float(state.get("overall_pct", 0)):.1f}%</span></div>
      <div class="stat"><span class="stat-label">ETA remaining</span><span class="stat-value">{html.escape(str(state.get("eta_label", "—")))}</span></div>
      <div class="stat"><span class="stat-label">Candidates</span><span class="stat-value">{len(candidates)}</span></div>
      <div class="stat"><span class="stat-label">Parallel</span><span class="stat-value">{int(state.get("max_parallel", 1))}</span></div>
    </div>

    <section class="overall">
      <div class="progress-label"><span>Run progress</span><span>{float(state.get("overall_pct", 0)):.1f}%</span></div>
      <div class="progress-bar"><div class="progress-fill" style="width:{float(state.get("overall_pct", 0)):.1f}%"></div></div>
      <p class="sub" style="margin:10px 0 0">{play_episodes} training episodes/candidate · {html.escape(ckpt_info)}</p>
    </section>

    <div class="grid">
      {cards_html}
    </div>

    <footer>
      Generated {generated} · Output: {out_dir}<br>
      Open this file in a browser; it auto-refreshes while the run is in progress.
    </footer>
  </div>
</body>
</html>"""


def write_sideboard_compare_dashboard(
    out_dir: Path,
    *,
    auto_refresh_seconds: Optional[float] = None,
) -> Path:
    """Collect state and write ``sideboard_compare_dashboard.html``."""
    state = collect_sideboard_compare_state(out_dir)
    page = render_sideboard_compare_html(state, auto_refresh_seconds=auto_refresh_seconds)
    html_path = out_dir / _DASHBOARD_NAME
    html_path.write_text(page, encoding="utf-8")
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live HTML dashboard for sideboard comparison training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", required=True,
        help="Sideboard compare output directory (contains candidates_manifest.json)")
    parser.add_argument("--watch", action="store_true",
        help="Keep polling and regenerating the dashboard until the run completes")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--open-browser", action="store_true",
        help="Open the dashboard in the default browser once")
    parser.add_argument("--no-auto-refresh", action="store_true",
        help="Disable the HTML meta refresh tag")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    refresh = None if args.no_auto_refresh else args.poll_seconds
    opened = False

    print("=" * 62)
    print("  Sideboard Compare Dashboard")
    print("=" * 62)
    print(f"  Watching : {out_dir}")
    print(f"  Output   : {out_dir / _DASHBOARD_NAME}")
    print("=" * 62)

    poll = 0
    while True:
        poll += 1
        html_path = write_sideboard_compare_dashboard(
            out_dir,
            auto_refresh_seconds=refresh,
        )
        state = collect_sideboard_compare_state(out_dir)
        complete = bool(state.get("complete"))
        n_done = sum(1 for c in state.get("candidates", []) if c.get("status") == "complete")
        n_total = len(state.get("candidates", []))
        print(
            f"  [poll {poll}] {html_path.name} updated — "
            f"{state.get('overall_pct', 0):.1f}% overall, "
            f"{n_done}/{n_total} candidates done, "
            f"ETA {state.get('eta_label', '—')}",
            flush=True,
        )

        if args.open_browser and not opened:
            webbrowser.open(html_path.resolve().as_uri())
            opened = True

        if not args.watch or complete:
            break
        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    main()
