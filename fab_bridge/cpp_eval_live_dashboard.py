"""Live auto-refresh dashboard for C++ engine evaluation replays.

Writes ``cpp_eval_live_state.json`` after each eval step and regenerates
``cpp_eval_live_dashboard.html`` (meta refresh) so you can watch agent
actions during fast C++ checkpoint eval.
"""

from __future__ import annotations

import html
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

CPP_EVAL_LIVE_STATE = "cpp_eval_live_state.json"
CPP_EVAL_LIVE_DASHBOARD = "cpp_eval_live_dashboard.html"
UNIFIED_CHECKPOINT_EVAL_LIVE = "unified_checkpoint_eval_live.json"


def unified_checkpoint_eval_live_path(run_dir: Path) -> Path:
    """Live progress JSON path for unified random-matchup checkpoint eval."""
    return run_dir.expanduser().resolve() / UNIFIED_CHECKPOINT_EVAL_LIVE

_last_html_write: dict[str, float] = {}
_announced_dashboards: set[str] = set()
DEFAULT_TAIL_EVENTS = 25
DEFAULT_AUTO_REFRESH_SECONDS = 1.0
DEFAULT_HTML_MIN_INTERVAL = 0.4


def cpp_eval_live_paths(live_progress_path: Path) -> tuple[Path, Path]:
    """Return (state_json, dashboard_html) paths beside *live_progress_path*."""
    root = live_progress_path.expanduser().resolve().parent
    return root / CPP_EVAL_LIVE_STATE, root / CPP_EVAL_LIVE_DASHBOARD


def _resolve_trace_env(env: Any) -> Any:
    inner = getattr(env, "_cpp_env", None)
    return inner if inner is not None else env


def eval_engine_from_env(env: Any) -> tuple[str, str]:
    """Return ``(engine_key, display_label)`` for a training/eval environment."""
    if bool(getattr(env, "_using_cpp", False)):
        return "cpp", "C++ engine"
    if bool(getattr(env, "_using_fast_talishar", False)):
        backend = str(getattr(env, "talishar_backend", "fast") or "fast")
        return "talishar_fast", f"Talishar {backend}"
    if getattr(env, "_cpp_env", None) is not None:
        return "cpp", "C++ engine"
    return "talishar", "Talishar"


def checkpoint_eval_replay_display_label(
    source: Optional[dict[str, Any] | str] = None,
) -> str:
    """Human-readable engine label for checkpoint eval replay links."""
    if isinstance(source, str):
        token = source.strip().lower()
        if "talishar" in token and "fast" in token:
            return "Talishar fast"
        if "cpp" in token or "c++" in token:
            return "C++ engine"
        if "talishar" in token:
            return "Talishar"
        return ""
    if not isinstance(source, dict):
        return ""
    display = str(source.get("engine_display") or "").strip()
    if display:
        return display
    engine = str(source.get("engine") or "").strip().lower()
    if engine == "cpp":
        return "C++ engine"
    if engine in {"talishar_fast", "fast"}:
        return "Talishar fast"
    if engine == "talishar":
        return "Talishar"
    runtime = str(source.get("runtime_backend") or "").strip()
    if runtime:
        return checkpoint_eval_replay_display_label(runtime)
    return ""


def format_checkpoint_eval_replay_heading(engine_label: str = "") -> str:
    """Build the replay section heading for unified training dashboards."""
    label = checkpoint_eval_replay_display_label(engine_label)
    if label:
        return f"{label} checkpoint eval replay"
    return "Checkpoint eval replay"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def collect_cpp_eval_live_state(
    env: Any,
    *,
    episode: int,
    episodes_total: int,
    step: int,
    action_index: Optional[int] = None,
    eval_label: str = "Eval",
    p1_policy: str = "agent",
    p2_policy: str = "agent",
    aggregate: Optional[dict[str, Any]] = None,
    episode_done: bool = False,
    tail_events: int = DEFAULT_TAIL_EVENTS,
) -> dict[str, Any]:
    """Build a JSON-serialisable live eval snapshot from the current env."""
    trace_env = _resolve_trace_env(env)
    board = {}
    if hasattr(env, "live_display_snapshot"):
        board = env.live_display_snapshot()
    elif hasattr(trace_env, "live_display_snapshot"):
        board = trace_env.live_display_snapshot()

    tracker: dict[str, Any] = {}
    if hasattr(env, "get_combat_tracker_snapshot"):
        tracker = env.get_combat_tracker_snapshot(
            top_k=5,
            tail_events=tail_events,
            tail_log_lines=30,
        )
    elif hasattr(trace_env, "get_combat_tracker_snapshot"):
        tracker = trace_env.get_combat_tracker_snapshot(
            top_k=5,
            tail_events=tail_events,
            tail_log_lines=30,
        )

    recent_events = tracker.get("recent_events") or []
    latest_event = recent_events[-1] if recent_events else None
    chosen_action = None
    if isinstance(latest_event, dict):
        chosen_action = latest_event.get("action")

    agg = dict(aggregate or {})
    complete = bool(
        episode_done and int(agg.get("episodes_completed", 0) or 0) >= episodes_total
    )
    engine_key, engine_display = eval_engine_from_env(env)

    return {
        "engine": engine_key,
        "engine_display": engine_display,
        "eval_label": eval_label,
        "episode": int(episode),
        "episodes_total": int(episodes_total),
        "step": int(step),
        "action_index": action_index,
        "chosen_action": chosen_action,
        "latest_event": latest_event,
        "board": board,
        "recent_events": recent_events,
        "recent_combat_log": tracker.get("recent_combat_log_lines") or [],
        "p1_policy": p1_policy,
        "p2_policy": p2_policy,
        "aggregate": agg,
        "complete": complete,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_cpp_eval_live_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path = state_path.expanduser().resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _format_action(action: Any) -> str:
    if not isinstance(action, dict):
        return "—"
    label = str(action.get("label", "") or "").strip()
    if label:
        return label
    code = action.get("action_code", "")
    zone = str(action.get("zone", "") or "")
    return f"mode={code} ({zone})" if zone else f"mode={code}"


def _hand_cards_html(cards: list[dict[str, Any]], *, acting: bool) -> str:
    if not cards:
        return '<p class="muted">(empty)</p>'
    cls = "hand-card acting-seat" if acting else "hand-card"
    parts = []
    for card in cards:
        name = html.escape(str(card.get("name") or card.get("card_id") or "?"))
        cost = int(card.get("cost", 0) or 0)
        pitch = int(card.get("pitch", 0) or 0)
        power = int(card.get("power", 0) or 0)
        parts.append(
            f'<div class="{cls}">'
            f"<strong>{name}</strong>"
            f'<span class="muted"> cost {cost} · pitch {pitch} · power {power}</span>'
            f"</div>"
        )
    return "\n".join(parts)


def _events_table(events: list[dict[str, Any]]) -> str:
    if not events:
        return '<p class="muted">No steps recorded yet.</p>'
    rows = []
    for event in reversed(events):
        if not isinstance(event, dict):
            continue
        step_no = int(event.get("step", 0) or 0)
        before = event.get("before") if isinstance(event.get("before"), dict) else {}
        action = event.get("action") if isinstance(event.get("action"), dict) else {}
        acting = int(before.get("acting_player_id", 0) or 0)
        turn_no = int(before.get("turn_no", 0) or 0)
        phase = html.escape(str(before.get("phase", "") or ""))
        p_hp = int(before.get("player_health", 0) or 0)
        o_hp = int(before.get("opponent_health", 0) or 0)
        action_text = html.escape(_format_action(action))
        action_class = html.escape(str(event.get("action_class", "") or ""))
        rows.append(
            f"<tr>"
            f"<td>{step_no}</td>"
            f"<td>P{acting}</td>"
            f"<td>{turn_no}</td>"
            f"<td>{phase}</td>"
            f"<td class=\"action-cell\"><span class=\"pill\">{action_class}</span> {action_text}</td>"
            f"<td>{p_hp} / {o_hp}</td>"
            f"</tr>"
        )
    return (
        '<table class="events">'
        "<thead><tr>"
        "<th>Step</th><th>Seat</th><th>Turn</th><th>Phase</th>"
        "<th>Action</th><th>HP (act / opp)</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _legal_actions_html(
    legal_actions: list[dict[str, Any]],
    *,
    highlight_index: Optional[int],
) -> str:
    if not legal_actions:
        return '<p class="muted">No legal actions.</p>'
    rows = []
    for index, action in enumerate(legal_actions):
        if not isinstance(action, dict):
            continue
        row_cls = "chosen" if highlight_index is not None and index == highlight_index else ""
        label = html.escape(_format_action(action))
        zone = html.escape(str(action.get("zone", "") or ""))
        rows.append(
            f'<tr class="{row_cls}">'
            f"<td>[{index}]</td>"
            f"<td>{label}</td>"
            f"<td>{zone}</td>"
            f"</tr>"
        )
    return (
        '<table class="legal">'
        "<thead><tr><th>#</th><th>Label</th><th>Zone</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_cpp_eval_live_html(
    state: dict[str, Any],
    *,
    auto_refresh_seconds: Optional[float] = DEFAULT_AUTO_REFRESH_SECONDS,
) -> str:
    """Render a self-contained HTML page for *state*."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    complete = bool(state.get("complete"))
    refresh_tag = ""
    if auto_refresh_seconds and not complete:
        refresh_tag = (
            f'<meta http-equiv="refresh" content="{max(1, int(auto_refresh_seconds))}">'
        )

    board = state.get("board") if isinstance(state.get("board"), dict) else {}
    agg = state.get("aggregate") if isinstance(state.get("aggregate"), dict) else {}
    episode = int(state.get("episode", 0) or 0)
    episodes_total = int(state.get("episodes_total", 0) or 0)
    step = int(state.get("step", 0) or 0)
    acting = int(board.get("acting_player_id", 0) or 0)
    p1_hp = int(board.get("p1_health", 0) or 0)
    p2_hp = int(board.get("p2_health", 0) or 0)
    turn_no = int(board.get("turn_no", 0) or 0)
    phase = html.escape(str(board.get("phase", "") or "—"))
    deck1 = html.escape(str(board.get("deck1", "") or "P1"))
    deck2 = html.escape(str(board.get("deck2", "") or "P2"))
    eval_label = html.escape(str(state.get("eval_label", "Eval") or "Eval"))
    p1_policy = html.escape(str(state.get("p1_policy", "agent") or "agent"))
    p2_policy = html.escape(str(state.get("p2_policy", "agent") or "agent"))
    engine_display = html.escape(
        checkpoint_eval_replay_display_label(state) or "Eval"
    )

    wins = int(agg.get("wins", 0) or 0)
    losses = int(agg.get("losses", 0) or 0)
    draws = int(agg.get("draws", 0) or 0)
    timeouts = int(agg.get("timeouts", 0) or 0)
    eps_done = int(agg.get("episodes_completed", 0) or 0)

    chosen = state.get("chosen_action")
    chosen_text = html.escape(_format_action(chosen if isinstance(chosen, dict) else {}))
    action_index = state.get("action_index")

    p1_hand = board.get("p1_hand") if isinstance(board.get("p1_hand"), list) else []
    p2_hand = board.get("p2_hand") if isinstance(board.get("p2_hand"), list) else []
    legal_actions = (
        board.get("legal_actions") if isinstance(board.get("legal_actions"), list) else []
    )
    recent_events = (
        state.get("recent_events") if isinstance(state.get("recent_events"), list) else []
    )
    combat_log = (
        state.get("recent_combat_log")
        if isinstance(state.get("recent_combat_log"), list)
        else []
    )

    log_lines = "".join(
        f"<li>{html.escape(str(line))}</li>" for line in combat_log[-20:]
    )
    status_label = "Complete" if complete else "Running"

    p1_acting = acting == 1
    p2_acting = acting == 2

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{engine_display} eval live — {eval_label}</title>
  {refresh_tag}
  <style>
    :root {{
      --bg: #0f1419; --surface: #1a2332; --border: #2d3a4d;
      --text: #e7ecf3; --muted: #8b9cb3; --primary: #5b9cff;
      --ok: #3ecf8e; --warn: #f0b429; --p1: #5b9cff; --p2: #f08c5b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: ui-sans-serif, system-ui, sans-serif;
      background: var(--bg); color: var(--text); line-height: 1.45;
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 1.25rem; }}
    h1 {{ font-size: 1.35rem; margin: 0 0 0.25rem; }}
    h2 {{ font-size: 1rem; margin: 0 0 0.75rem; color: var(--muted); font-weight: 600; }}
    .sub {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 1rem; }}
    .grid {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    .card {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 10px; padding: 1rem;
    }}
    .stats {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem; }}
    .stat {{ min-width: 120px; }}
    .stat-label {{ display: block; color: var(--muted); font-size: 0.8rem; }}
    .stat-value {{ font-size: 1.2rem; font-weight: 600; }}
    .player-panel {{ border-left: 4px solid var(--border); padding-left: 0.75rem; }}
    .player-panel.p1 {{ border-color: var(--p1); }}
    .player-panel.p2 {{ border-color: var(--p2); }}
    .player-panel.acting {{ background: rgba(91, 156, 255, 0.08); border-radius: 6px; padding: 0.75rem; }}
    .hand-card {{ padding: 0.25rem 0; border-bottom: 1px solid var(--border); }}
    .hand-card:last-child {{ border-bottom: none; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    th, td {{ text-align: left; padding: 0.35rem 0.5rem; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 600; }}
    tr.chosen td {{ background: rgba(62, 207, 142, 0.12); }}
    .action-cell {{ max-width: 320px; word-break: break-word; }}
    .pill {{
      display: inline-block; font-size: 0.7rem; padding: 0.1rem 0.35rem;
      border-radius: 4px; background: var(--border); color: var(--muted);
    }}
    .chosen-banner {{
      background: rgba(62, 207, 142, 0.15); border: 1px solid rgba(62, 207, 142, 0.35);
      border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem;
    }}
    .muted {{ color: var(--muted); }}
    .badge {{
      display: inline-block; padding: 0.15rem 0.5rem; border-radius: 999px;
      font-size: 0.75rem; background: var(--border);
    }}
    .badge.running {{ color: var(--primary); }}
    .badge.complete {{ color: var(--ok); }}
    footer {{ margin-top: 1.5rem; color: var(--muted); font-size: 0.8rem; }}
    ul.log {{ margin: 0; padding-left: 1.2rem; font-size: 0.85rem; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{engine_display} eval — live replay</h1>
    <p class="sub">{eval_label} · episode {episode}/{episodes_total} · step {step}
      · <span class="badge {'complete' if complete else 'running'}">{status_label}</span></p>

    <div class="stats">
      <div class="stat"><span class="stat-label">Eval games done</span>
        <span class="stat-value">{eps_done}/{episodes_total}</span></div>
      <div class="stat"><span class="stat-label">W / L / D / T</span>
        <span class="stat-value">{wins} / {losses} / {draws} / {timeouts}</span></div>
      <div class="stat"><span class="stat-label">Turn · phase</span>
        <span class="stat-value">{turn_no} · {phase}</span></div>
      <div class="stat"><span class="stat-label">Priority</span>
        <span class="stat-value">P{acting or '—'}</span></div>
      <div class="stat"><span class="stat-label">Policies</span>
        <span class="stat-value">P1={p1_policy} · P2={p2_policy}</span></div>
    </div>

    <div class="chosen-banner">
      <strong>Latest agent action</strong>
      <span class="muted"> (index {action_index if action_index is not None else '—'})</span>
      <div>{chosen_text}</div>
    </div>

    <div class="grid">
      <div class="card player-panel p1 {'acting' if p1_acting else ''}">
        <h2>P1 — {deck1}</h2>
        <div class="stat-value">{p1_hp} HP</div>
        <p class="muted">deck {int(board.get('p1_deck_size', 0) or 0)} · pitch {int(board.get('p1_pitch_size', 0) or 0)}</p>
        {_hand_cards_html(p1_hand, acting=p1_acting)}
      </div>
      <div class="card player-panel p2 {'acting' if p2_acting else ''}">
        <h2>P2 — {deck2}</h2>
        <div class="stat-value">{p2_hp} HP</div>
        <p class="muted">deck {int(board.get('p2_deck_size', 0) or 0)} · pitch {int(board.get('p2_pitch_size', 0) or 0)}</p>
        {_hand_cards_html(p2_hand, acting=p2_acting)}
      </div>
    </div>

    <div class="card" style="margin-top:1rem">
      <h2>Current legal actions</h2>
      {_legal_actions_html(legal_actions, highlight_index=None)}
    </div>

    <div class="card" style="margin-top:1rem">
      <h2>Recent steps (newest first)</h2>
      {_events_table(recent_events)}
    </div>

    <div class="card" style="margin-top:1rem">
      <h2>Combat log</h2>
      <ul class="log">{log_lines or '<li class="muted">(empty)</li>'}</ul>
    </div>

    <footer>Generated {generated} · auto-refresh {int(auto_refresh_seconds or 0)}s while running</footer>
  </div>
</body>
</html>"""


def write_cpp_eval_live_dashboard(
    state_path: Path,
    html_path: Path,
    *,
    auto_refresh_seconds: Optional[float] = DEFAULT_AUTO_REFRESH_SECONDS,
) -> Path:
    """Read *state_path* and write the HTML dashboard."""
    state = _read_json(state_path)
    html_path = html_path.expanduser().resolve()
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(
        render_cpp_eval_live_html(state, auto_refresh_seconds=auto_refresh_seconds),
        encoding="utf-8",
    )
    return html_path


def maybe_refresh_cpp_eval_live_dashboard(
    state_path: Path,
    html_path: Path,
    *,
    auto_refresh_seconds: float = DEFAULT_AUTO_REFRESH_SECONDS,
    min_interval_seconds: float = DEFAULT_HTML_MIN_INTERVAL,
) -> Optional[Path]:
    """Throttle HTML regeneration during hot eval loops."""
    key = str(html_path.resolve())
    now = time.monotonic()
    last = _last_html_write.get(key, 0.0)
    if now - last < min_interval_seconds:
        return None
    _last_html_write[key] = now
    return write_cpp_eval_live_dashboard(
        state_path,
        html_path,
        auto_refresh_seconds=auto_refresh_seconds,
    )


def update_cpp_eval_live_replay(
    live_progress_path: Path,
    env: Any,
    *,
    episode: int,
    episodes_total: int,
    step: int,
    action_index: Optional[int] = None,
    eval_label: str = "Eval",
    p1_policy: str = "agent",
    p2_policy: str = "agent",
    aggregate: Optional[dict[str, Any]] = None,
    episode_done: bool = False,
    auto_refresh_seconds: float = DEFAULT_AUTO_REFRESH_SECONDS,
    announce: bool = True,
) -> tuple[Path, Optional[Path]]:
    """Write live JSON state and (throttled) regenerate the HTML dashboard."""
    state_path, html_path = cpp_eval_live_paths(live_progress_path)
    state = collect_cpp_eval_live_state(
        env,
        episode=episode,
        episodes_total=episodes_total,
        step=step,
        action_index=action_index,
        eval_label=eval_label,
        p1_policy=p1_policy,
        p2_policy=p2_policy,
        aggregate=aggregate,
        episode_done=episode_done,
    )
    write_cpp_eval_live_state(state_path, state)
    refreshed = maybe_refresh_cpp_eval_live_dashboard(
        state_path,
        html_path,
        auto_refresh_seconds=auto_refresh_seconds,
    )
    if announce and refreshed is not None:
        key = str(html_path.resolve())
        if key not in _announced_dashboards:
            _announced_dashboards.add(key)
            print(f"  C++ eval live dashboard → {html_path}", flush=True)
    return state_path, refreshed
