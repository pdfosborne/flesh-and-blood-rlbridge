"""Live HTML dashboard for unified random matchup training runs."""

from __future__ import annotations

import html
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fab_bridge.unified_results import (
    RUN_MANIFEST,
    iter_unified_matchup_dirs,
    is_unified_random_matchup_run,
    read_checkpoint_eval_scope,
)

UNIFIED_DASHBOARD_NAME = "unified_random_matchups_dashboard.html"
UNIFIED_LIVE_STATE = "unified_training_live.json"

_last_dashboard_write: dict[str, float] = {}


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
    path = run_dir / UNIFIED_LIVE_STATE
    current: dict[str, Any] = {}
    raw = _read_json(path)
    if isinstance(raw, dict):
        current.update(raw)
    current.update({k: v for k, v in fields.items() if v is not None})
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(current, indent=2), encoding="utf-8")


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

    current_name = str(
        live.get("current_matchup")
        or scope.get("matchup")
        or ""
    ).strip()
    current_subdir = str(
        live.get("current_matchup_dir")
        or scope.get("matchup_dir")
        or ""
    ).strip()
    if not current_name and current_subdir:
        current_name = _matchup_label(run_dir / current_subdir)

    episodes_completed = int(live.get("episodes_completed") or 0)
    if episodes_completed <= 0 and ckpt_history:
        last_ckpt = ckpt_history[-1]
        if isinstance(last_ckpt, dict):
            episodes_completed = int(last_ckpt.get("episodes_completed") or 0)
    if episodes_completed <= 0 and current_subdir:
        row = _matchup_summary_row(run_dir / current_subdir, target_episodes)
        episodes_completed = int(row.get("episodes_completed") or 0)

    train_p1_wr = live.get("p1_win_rate")
    train_p2_wr = live.get("p2_win_rate")
    status = str(live.get("status") or "training")
    if (run_dir / "training_summary.json").is_file() and matchups_completed >= matchups_total > 0:
        status = "complete"
    elif not current_name and matchups_completed >= matchups_total > 0:
        status = "complete"

    checkpoint_points = []
    for row in ckpt_history:
        if not isinstance(row, dict) or row.get("episodes_completed") is None:
            continue
        vs_p1, vs_p2, vs_avg = _vs_logic_win_rates(row)
        checkpoint_points.append(
            {
                "episode": int(row.get("episodes_completed") or 0),
                "win_rate": float(row.get("p1_win_rate") or 0.0),
                "p1_wins": int(row.get("p1_wins") or 0),
                "p2_wins": int(row.get("p2_wins") or 0),
                "matchup": str(row.get("matchup") or ""),
                "vs_logic_agent_p1": vs_p1,
                "vs_logic_agent_p2": vs_p2,
                "vs_logic_win_rate": vs_avg,
            }
        )

    latest_vs_logic_p1: Optional[float] = None
    latest_vs_logic_p2: Optional[float] = None
    latest_vs_logic_avg: Optional[float] = None
    if checkpoint_points:
        latest_vs_logic_p1 = checkpoint_points[-1].get("vs_logic_agent_p1")
        latest_vs_logic_p2 = checkpoint_points[-1].get("vs_logic_agent_p2")
        latest_vs_logic_avg = checkpoint_points[-1].get("vs_logic_win_rate")

    completed_rows = [
        _matchup_summary_row(matchup_dir, target_episodes)
        for matchup_dir in reversed(iter_unified_matchup_dirs(run_dir))
        if _is_matchup_complete(matchup_dir, target_episodes)
    ]

    overall_pct = 0.0
    if matchups_total > 0:
        intra = (
            (episodes_completed / max(1, target_episodes))
            if status == "training" and target_episodes > 0
            else 0.0
        )
        overall_pct = ((matchups_completed + intra) / matchups_total) * 100.0

    return {
        "run_dir": str(run_dir),
        "format": str(manifest.get("format") or "—"),
        "started_at": str(manifest.get("started_at") or ""),
        "matchups_total": matchups_total,
        "matchups_completed": matchups_completed,
        "matchups_sampled": list(manifest.get("matchups_sampled") or []),
        "target_episodes": target_episodes,
        "current_matchup": current_name or "—",
        "current_matchup_dir": current_subdir,
        "episodes_completed": episodes_completed,
        "train_p1_win_rate": train_p1_wr,
        "train_p2_win_rate": train_p2_wr,
        "status": status,
        "checkpoint_points": checkpoint_points,
        "latest_checkpoint_win_rate": (
            checkpoint_points[-1]["win_rate"] if checkpoint_points else None
        ),
        "latest_vs_logic_win_rate": latest_vs_logic_avg,
        "latest_vs_logic_p1_seat": latest_vs_logic_p1,
        "latest_vs_logic_p2_seat": latest_vs_logic_p2,
        "completed_matchups": completed_rows,
        "overall_pct": overall_pct,
        "complete": status == "complete",
    }


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
    current_eps = int(state.get("episodes_completed") or 0)
    train_pct = (current_eps / max(1, target_eps)) * 100.0 if target_eps else 0.0
    overall_pct = float(state.get("overall_pct") or 0.0)
    status = str(state.get("status") or "training")
    status_label = {
        "training": "Training",
        "between_matchups": "Between matchups",
        "complete": "Complete",
    }.get(status, status.title())

    ckpt_chart = _svg_winrate_chart(
        state.get("checkpoint_points") or [],
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
    for row in reversed(state.get("checkpoint_points") or []):
        vs_p1 = row.get("vs_logic_agent_p1")
        vs_p2 = row.get("vs_logic_agent_p2")
        vs_detail = ""
        if vs_p1 is not None or vs_p2 is not None:
            vs_detail = (
                f' title="P1 seat: {_pct(vs_p1)}, P2 seat: {_pct(vs_p2)}"'
            )
        ckpt_rows += (
            f"<tr><td>{int(row.get('episode', 0))}</td>"
            f"<td>{_pct(row.get('win_rate'))}</td>"
            f"<td{vs_detail}>{_pct(row.get('vs_logic_win_rate'))}</td>"
            f"<td>{_pct(vs_p1)}</td>"
            f"<td>{_pct(vs_p2)}</td>"
            f"<td>{int(row.get('p1_wins', 0))}</td>"
            f"<td>{int(row.get('p2_wins', 0))}</td></tr>"
        )
    ckpt_table = (
        f'<table class="history"><thead><tr>'
        f"<th>Episode</th><th>Self-play P1%</th>"
        f"<th>Vs logic avg%</th><th>Vs logic P1 seat</th><th>Vs logic P2 seat</th>"
        f"<th>Self-play P1 wins</th><th>Self-play P2 wins</th>"
        f"</tr></thead><tbody>{ckpt_rows or '<tr><td colspan=\"7\" class=\"muted\">No checkpoint eval yet</td></tr>'}</tbody></table>"
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
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 24px; }}
    .stat {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }}
    .stat-label {{ display: block; color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    .stat-value {{ display: block; font-size: 1.35rem; font-weight: 600; margin-top: 6px; }}
    .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; margin-bottom: 18px; }}
    .card h2 {{ margin: 0 0 12px; font-size: 1.05rem; }}
    .progress-bar {{ height: 10px; background: #243044; border-radius: 999px; overflow: hidden; }}
    .progress-fill {{ height: 100%; background: linear-gradient(90deg, var(--primary), var(--good)); }}
    .progress-label {{ display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--muted); margin-bottom: 8px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-top: 14px; }}
    .metric-label {{ display: block; color: var(--muted); font-size: 0.75rem; }}
    .metric-value {{ font-size: 1.2rem; font-weight: 600; }}
    .status-pill {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.75rem; background: #243044; color: var(--primary); }}
    .status-pill.complete {{ color: var(--good); }}
    .history {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    .history th, .history td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }}
    .history th {{ color: var(--muted); font-weight: 500; }}
    .muted {{ color: var(--muted); }}
    .chart {{ width: 100%; max-width: 520px; }}
    .chart-line {{ stroke: var(--primary); stroke-width: 2; }}
    .chart-dot {{ fill: var(--primary); }}
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

    <div class="summary">
      <div class="stat"><span class="stat-label">Matchups done</span><span class="stat-value">{matchups_completed}/{matchups_total}</span></div>
      <div class="stat"><span class="stat-label">Overall progress</span><span class="stat-value">{overall_pct:.1f}%</span></div>
      <div class="stat"><span class="stat-label">Current train win%</span><span class="stat-value">{_pct(state.get('train_p1_win_rate'))}</span></div>
      <div class="stat"><span class="stat-label">Latest self-play ckpt</span><span class="stat-value">{_pct(state.get('latest_checkpoint_win_rate'))}</span></div>
      <div class="stat"><span class="stat-label">Latest vs logic</span><span class="stat-value">{_pct(state.get('latest_vs_logic_win_rate'))}</span></div>
    </div>

    <div class="card">
      <h2>Current matchup · {html.escape(str(state.get('current_matchup') or '—'))}</h2>
      <div class="progress-label"><span>Training episodes</span><span>{current_eps}/{target_eps}</span></div>
      <div class="progress-bar"><div class="progress-fill" style="width:{train_pct:.1f}%"></div></div>
      <div class="metrics">
        <div><span class="metric-label">P1 seat win%</span><span class="metric-value">{_pct(state.get('train_p1_win_rate'))}</span></div>
        <div><span class="metric-label">P2 seat win%</span><span class="metric-value">{_pct(state.get('train_p2_win_rate'))}</span></div>
        <div><span class="metric-label">Latest self-play ckpt</span><span class="metric-value">{_pct(state.get('latest_checkpoint_win_rate'))}</span></div>
        <div><span class="metric-label">Vs logic (avg)</span><span class="metric-value">{_pct(state.get('latest_vs_logic_win_rate'))}</span></div>
        <div><span class="metric-label">Vs logic P1 / P2 seat</span><span class="metric-value">{_pct(state.get('latest_vs_logic_p1_seat'))} / {_pct(state.get('latest_vs_logic_p2_seat'))}</span></div>
      </div>
    </div>

    <div class="card">
      <h2>Checkpoint eval (current matchup)</h2>
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
