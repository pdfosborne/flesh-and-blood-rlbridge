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

from fab_bridge.unified_results import (
    RUN_MANIFEST,
    iter_unified_matchup_dirs,
    is_unified_random_matchup_run,
    read_checkpoint_eval_scope,
)
from flesh_and_blood_rlbridge.card_db.fabrary_meta import (
    DEFAULT_PERIOD,
    load_fabrary_meta,
    lookup_all_queues_for_matchup_dir,
)

UNIFIED_DASHBOARD_NAME = "unified_random_matchups_dashboard.html"
UNIFIED_LIVE_STATE = "unified_training_live.json"
LOGIC_VS_LOGIC_BASELINE_NAME = "logic_vs_logic_baseline.json"
POLICY_WEIGHT_HISTORY_LIMIT = 30
TRAIN_WIN_RATE_HISTORY_LIMIT = 500

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


def _hero_display_name(raw: str) -> str:
    slug = str(raw or "").strip().removeprefix("fab_")
    if not slug:
        return "?"
    if "-vs-" in slug:
        slug = slug.split("-vs-", 1)[0]
    slug = slug.replace("-", "_")
    if slug.startswith("precon_"):
        parts = slug.split("_")
        if len(parts) >= 4:
            slug = parts[-1]
        elif len(parts) >= 2:
            slug = parts[-1]
    elif "_" in slug:
        slug = slug.rsplit("_", 1)[-1]
    return slug.replace("_", " ").replace("-", " ").title() or "?"


def _read_matchup_hero_labels(matchup_dir: Path) -> tuple[str, str]:
    raw = _read_json(matchup_dir / "matchup_label.json")
    if isinstance(raw, dict):
        h1 = str(raw.get("p1_hero") or "").strip()
        h2 = str(raw.get("p2_hero") or "").strip()
        if h1 and h2:
            return _hero_display_name(h1), _hero_display_name(h2)
        p1_deck = str(raw.get("p1_deck") or "").strip()
        p2_deck = str(raw.get("p2_deck") or "").strip()
        if p1_deck and p2_deck:
            return _hero_display_name(p1_deck), _hero_display_name(p2_deck)
        name = str(raw.get("name") or "").strip()
        if "-vs-" in name:
            left, right = name.split("-vs-", 1)
            return _hero_display_name(left), _hero_display_name(right)
    return _hero_display_name(matchup_dir.name), "?"


def _format_matchup_hero_title(hero1: str, hero2: str) -> str:
    return f"{hero1} (Hero 1) vs {hero2} (Hero 2)"


def _optional_win_rate(row: dict[str, Any], key: str) -> Optional[float]:
    if row.get(key) is None:
        return None
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return None


def _hero1_win_rate_excluding_timeouts(metrics: dict[str, Any]) -> Optional[float]:
    """Hero 1 win rate over decided games only (excludes timeouts/errors)."""
    heroes = metrics.get("heroes")
    if isinstance(heroes, dict):
        if heroes.get("win_rate_decided") is not None:
            return float(heroes["win_rate_decided"])
        h1_wins = heroes.get("hero1_wins")
        h2_wins = heroes.get("hero2_wins")
        draws = heroes.get("draws", 0)
        if h1_wins is not None and h2_wins is not None:
            decided = int(h1_wins) + int(h2_wins) + int(draws or 0)
            if decided > 0:
                return int(h1_wins) / decided
    if metrics.get("win_rate_decided") is not None:
        return float(metrics["win_rate_decided"])
    p1_wins = metrics.get("p1_wins")
    p2_wins = metrics.get("p2_wins")
    draws = metrics.get("draws", 0)
    if p1_wins is not None and p2_wins is not None:
        decided = int(p1_wins) + int(p2_wins) + int(draws or 0)
        if decided > 0:
            return int(p1_wins) / decided
    return None


def _hero_win_rates_from_metrics(
    metrics: dict[str, Any],
) -> tuple[Optional[float], Optional[float]]:
    """Read nominal hero1/hero2 win rates from a metrics or checkpoint row."""
    heroes = metrics.get("heroes")
    if isinstance(heroes, dict):
        h1 = heroes.get("hero1_win_rate")
        h2 = heroes.get("hero2_win_rate")
        if h1 is not None and h2 is not None:
            return float(h1), float(h2)
    if metrics.get("hero1_win_rate") is not None and metrics.get("hero2_win_rate") is not None:
        return float(metrics["hero1_win_rate"]), float(metrics["hero2_win_rate"])
    p1 = metrics.get("p1_win_rate")
    p2 = metrics.get("p2_win_rate")
    if p1 is None and p2 is None:
        return None, None
    if p1 is not None and p2 is None:
        p1f = float(p1)
        return p1f, max(0.0, 1.0 - p1f)
    if p2 is not None and p1 is None:
        p2f = float(p2)
        return max(0.0, 1.0 - p2f), p2f
    p1f = float(p1)
    p2f = float(p2)
    if bool(metrics.get("deck_swap_eval", False)):
        avg = (p1f + p2f) / 2.0
        return avg, avg
    return p1f, p2f


def _logic_vs_logic_hero1_win_rate(row: dict[str, Any]) -> Optional[float]:
    logic_vs_logic = row.get("logic_vs_logic")
    if not isinstance(logic_vs_logic, dict):
        return None
    return _hero1_win_rate_excluding_timeouts(logic_vs_logic)


def _logic_vs_logic_hero_win_rates(
    row: dict[str, Any],
) -> tuple[Optional[float], Optional[float]]:
    logic_vs_logic = row.get("logic_vs_logic")
    if not isinstance(logic_vs_logic, dict):
        return None, None
    return _hero_win_rates_from_metrics(logic_vs_logic)


def _apply_hero_labels_to_checkpoint_row(
    row: dict[str, Any],
    matchup_dir: Path,
) -> dict[str, Any]:
    hero1, hero2 = _read_matchup_hero_labels(matchup_dir)
    row["hero1_name"] = hero1
    row["hero2_name"] = hero2
    row["matchup"] = _format_matchup_hero_title(hero1, hero2)
    return row


def _vs_logic_win_rates(row: dict[str, Any]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (agent hero1, agent hero2, average) win rates vs logic policy."""
    vs_logic = row.get("vs_logic")
    if not isinstance(vs_logic, dict):
        return None, None, None
    if vs_logic.get("agent_hero1_win_rate") is not None:
        h1 = float(vs_logic["agent_hero1_win_rate"])
        h2 = float(vs_logic.get("agent_hero2_win_rate", 0.0) or 0.0)
        avg = float(vs_logic.get("agent_hero_win_rate", (h1 + h2) / 2.0) or 0.0)
        return h1, h2, avg
    p1_seat = vs_logic.get("agent_p1_seat")
    p2_seat = vs_logic.get("agent_p2_seat")
    p1_wr: Optional[float] = None
    p2_wr: Optional[float] = None
    if isinstance(p1_seat, dict):
        h1, _ = _hero_win_rates_from_metrics(p1_seat)
        if isinstance(p1_seat.get("heroes"), dict):
            agent_h1 = p1_seat["heroes"].get("agent_hero1_win_rate")
            if agent_h1 is not None:
                p1_wr = float(agent_h1)
        if p1_wr is None and p1_seat.get("agent_win_rate") is not None:
            p1_wr = float(p1_seat["agent_win_rate"])
    if isinstance(p2_seat, dict):
        if isinstance(p2_seat.get("heroes"), dict):
            agent_h2 = p2_seat["heroes"].get("agent_hero2_win_rate")
            if agent_h2 is not None:
                p2_wr = float(agent_h2)
        if p2_wr is None and p2_seat.get("agent_win_rate") is not None:
            p2_wr = float(p2_seat["agent_win_rate"])
    if p1_wr is not None and p2_wr is not None:
        return p1_wr, p2_wr, (p1_wr + p2_wr) / 2.0
    if p1_wr is not None:
        return p1_wr, None, p1_wr
    if p2_wr is not None:
        return None, p2_wr, p2_wr
    return None, None, None


def _logic_vs_agent_hero_win_rates(
    row: dict[str, Any],
) -> tuple[Optional[float], Optional[float]]:
    vs_logic = row.get("vs_logic")
    if not isinstance(vs_logic, dict):
        return None, None
    p1_seat = vs_logic.get("agent_p1_seat")
    p2_seat = vs_logic.get("agent_p2_seat")
    logic_h1: Optional[float] = None
    logic_h2: Optional[float] = None
    if isinstance(p1_seat, dict) and p1_seat.get("p2_win_rate") is not None:
        logic_h1 = float(p1_seat["p2_win_rate"])
    if isinstance(p2_seat, dict) and p2_seat.get("p1_win_rate") is not None:
        logic_h2 = float(p2_seat["p1_win_rate"])
    if logic_h1 is None or logic_h2 is None:
        avg = _logic_vs_agent_win_rate(row)
        if avg is not None:
            if logic_h1 is None and logic_h2 is None:
                return avg, avg
            if logic_h1 is None:
                logic_h1 = max(0.0, min(1.0, 2.0 * avg - float(logic_h2)))
            elif logic_h2 is None:
                logic_h2 = max(0.0, min(1.0, 2.0 * avg - float(logic_h1)))
    return logic_h1, logic_h2


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
    record = _read_matchup_logic_vs_logic_baseline_record(matchup_dir)
    if record is not None and record.get("p1_win_rate") is not None:
        return float(record["p1_win_rate"])
    return None


def _read_matchup_logic_vs_logic_baseline_record(
    matchup_dir: Path,
) -> Optional[dict[str, Any]]:
    raw = _read_json(matchup_dir / LOGIC_VS_LOGIC_BASELINE_NAME)
    if isinstance(raw, dict):
        return raw
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
    first_h1, _ = _hero_win_rates_from_metrics(first)
    last_h1, _ = _hero_win_rates_from_metrics(last)
    if first_h1 is not None:
        row["first_checkpoint_win_rate"] = first_h1
    elif first.get("p1_win_rate") is not None:
        row["first_checkpoint_win_rate"] = float(first["p1_win_rate"])
    if last_h1 is not None:
        row["checkpoint_win_rate"] = last_h1
    elif last.get("p1_win_rate") is not None:
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


def record_policy_weight_update(run_dir: Path, summary: dict[str, Any]) -> None:
    """Append a weight summary snapshot to unified live state."""
    if not is_unified_random_matchup_run(run_dir):
        return
    run_dir = run_dir.expanduser().resolve()
    path = run_dir / UNIFIED_LIVE_STATE
    with _run_state_lock(run_dir):
        current: dict[str, Any] = {}
        raw = _read_json(path)
        if isinstance(raw, dict):
            current.update(raw)
        current["policy_weights"] = summary
        history_raw = current.get("policy_weight_history")
        history: list[dict[str, Any]] = (
            list(history_raw) if isinstance(history_raw, list) else []
        )
        history.append(
            {
                "update_count": summary.get("update_count"),
                "l2_norm": summary.get("l2_norm"),
                "fingerprint": summary.get("fingerprint"),
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        current["policy_weight_history"] = history[-POLICY_WEIGHT_HISTORY_LIMIT:]
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
        episode = row.get("episodes_completed")
        hero1_wr = row.get("hero1_win_rate")
        if hero1_wr is None:
            hero1_wr = row.get("p1_win_rate")
        hero2_wr = row.get("hero2_win_rate")
        if hero2_wr is None:
            hero2_wr = row.get("p2_win_rate")
        if episode is not None and hero1_wr is not None:
            _append_matchup_win_rate_history(
                row,
                episode=int(episode),
                hero1_win_rate=float(hero1_wr),
                hero2_win_rate=float(hero2_wr) if hero2_wr is not None else None,
            )
        active[str(matchup_key)] = row
        current["active_matchups"] = active
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(path, current)


def _checkpoint_point_from_row(row: dict[str, Any]) -> dict[str, Any]:
    vs_p1, vs_p2, vs_avg = _vs_logic_win_rates(row)
    logic_h1 = _logic_vs_logic_hero1_win_rate(row)
    logic_vs_agent_h1, logic_vs_agent_h2 = _logic_vs_agent_hero_win_rates(row)
    timeout_rate: Optional[float] = None
    if row.get("timeout_rate") is not None:
        timeout_rate = float(row["timeout_rate"])
    eval_episodes: Optional[int] = None
    if row.get("eval_episodes") is not None:
        eval_episodes = int(row["eval_episodes"])
    agent_h1 = _hero1_win_rate_excluding_timeouts(row)
    if agent_h1 is None:
        agent_h1, _ = _hero_win_rates_from_metrics(row)
    primary_wr = agent_h1 if agent_h1 is not None else float(row.get("p1_win_rate") or 0.0)
    return {
        "episode": int(row.get("episodes_completed") or 0),
        "win_rate": primary_wr,
        "p1_wins": int(row.get("p1_wins") or 0),
        "p2_wins": int(row.get("p2_wins") or 0),
        "matchup": str(row.get("matchup") or ""),
        "vs_logic_agent_p1": vs_p1,
        "vs_logic_agent_p2": vs_p2,
        "vs_logic_win_rate": vs_avg,
        "vs_logic_hero1_win_rate": vs_p1,
        "vs_logic_hero2_win_rate": vs_p2,
        "logic_vs_agent_win_rate": _logic_vs_agent_win_rate(row),
        "logic_vs_agent_hero1_win_rate": logic_vs_agent_h1,
        "logic_vs_agent_hero2_win_rate": logic_vs_agent_h2,
        "agent_vs_agent_win_rate": agent_h1,
        "agent_vs_agent_hero1_win_rate": agent_h1,
        "timeout_rate": timeout_rate,
        "eval_episodes": eval_episodes,
        "logic_vs_logic_win_rate": logic_h1,
        "logic_vs_logic_hero1_win_rate": logic_h1,
    }


def _baseline_checkpoint_row(
    *,
    matchup_dir: Path,
    matchup_name: str,
    subdir: str,
) -> Optional[dict[str, Any]]:
    record = _read_matchup_logic_vs_logic_baseline_record(matchup_dir)
    if record is None:
        return None
    eval_episodes: Optional[int] = None
    if record.get("episodes") is not None:
        eval_episodes = int(record["episodes"])
    logic_h1 = _hero1_win_rate_excluding_timeouts(record)
    if logic_h1 is None and record.get("p1_win_rate") is not None:
        logic_h1 = float(record["p1_win_rate"])
    row = {
        "matchup": matchup_name,
        "matchup_dir": subdir,
        "episode": 0,
        "episode_label": "baseline",
        "eval_episodes": eval_episodes,
        "logic_vs_logic_win_rate": logic_h1,
        "logic_vs_logic_hero1_win_rate": logic_h1,
        "vs_logic_win_rate": None,
        "logic_vs_agent_win_rate": None,
        "agent_vs_agent_win_rate": None,
        "agent_vs_agent_hero1_win_rate": None,
        "timeout_rate": (
            float(record["timeout_rate"])
            if record.get("timeout_rate") is not None
            else None
        ),
    }
    return _apply_hero_labels_to_checkpoint_row(row, matchup_dir)


def _normalize_run_format(raw: object) -> str:
    token = str(raw or "").strip().lower()
    if token in ("classic_constructed", "cc"):
        return "classic_constructed"
    if token in ("silver_age", "sa", "sage"):
        return "silver_age"
    return token or "silver_age"


def _apply_fabrary_meta_to_row(
    row: dict[str, Any],
    matchup_dir: Path,
    *,
    format_name: str,
    meta: dict[str, Any],
) -> None:
    """Attach Fabrary Talishar hero-meta reference win rates (all three queues)."""
    if not matchup_dir.is_dir():
        return
    queues = lookup_all_queues_for_matchup_dir(
        matchup_dir,
        format_name=format_name,
        period=DEFAULT_PERIOD,
        meta=meta,
    )
    for games, prefix in (
        ("competitive", "fabrary_competitive"),
        ("all", "fabrary_all"),
        ("standard", "fabrary_standard"),
    ):
        ref = queues.get(games)
        if not isinstance(ref, dict):
            continue
        if ref.get("p1_win_rate") is not None:
            row[f"{prefix}_p1_win_rate"] = float(ref["p1_win_rate"])
        if games in ("competitive", "all") and ref.get("plays") is not None:
            row[f"{prefix}_plays"] = int(ref["plays"])


def _enrich_checkpoint_rows_with_fabrary_meta(
    run_dir: Path,
    rows: list[dict[str, Any]],
    *,
    format_name: str,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    for row in rows:
        if not isinstance(row, dict):
            continue
        subdir = str(row.get("matchup_dir") or "").strip()
        if not subdir:
            continue
        _apply_fabrary_meta_to_row(
            row,
            run_dir / subdir,
            format_name=format_name,
            meta=meta,
        )
    return rows


def _enrich_completed_rows_with_fabrary_meta(
    run_dir: Path,
    rows: list[dict[str, Any]],
    *,
    format_name: str,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    for row in rows:
        if not isinstance(row, dict):
            continue
        subdir = str(row.get("subdir") or "").strip()
        if not subdir:
            continue
        _apply_fabrary_meta_to_row(
            row,
            run_dir / subdir,
            format_name=format_name,
            meta=meta,
        )
    return rows


def _enrich_checkpoint_rows_with_baselines(
    run_dir: Path,
    rows: list[dict[str, Any]],
    active_matchups: dict[str, Any],
) -> list[dict[str, Any]]:
    """Merge logic-vs-logic baseline metrics into active checkpoint table rows."""
    if not active_matchups:
        return rows
    by_dir = {
        str(row.get("matchup_dir") or ""): dict(row)
        for row in rows
        if str(row.get("matchup_dir") or "")
    }
    enriched: list[dict[str, Any]] = []
    for subdir, info in active_matchups.items():
        if not isinstance(info, dict):
            continue
        matchup_dir = run_dir / str(subdir)
        name = str(info.get("name") or subdir)
        if str(subdir) in by_dir:
            row = by_dir[str(subdir)]
            baseline_record = _read_matchup_logic_vs_logic_baseline_record(matchup_dir)
            if baseline_record is not None:
                baseline_h1 = _hero1_win_rate_excluding_timeouts(baseline_record)
                if baseline_h1 is None and baseline_record.get("p1_win_rate") is not None:
                    baseline_h1 = float(baseline_record["p1_win_rate"])
                if baseline_h1 is not None:
                    row["logic_vs_logic_hero1_win_rate"] = baseline_h1
                    row["logic_vs_logic_win_rate"] = baseline_h1
            if row.get("logic_vs_logic_hero1_win_rate") is None:
                embedded_h1 = _logic_vs_logic_hero1_win_rate(row)
                if embedded_h1 is not None:
                    row["logic_vs_logic_hero1_win_rate"] = embedded_h1
                    row["logic_vs_logic_win_rate"] = embedded_h1
            enriched.append(_apply_hero_labels_to_checkpoint_row(row, matchup_dir))
            continue
        baseline_row = _baseline_checkpoint_row(
            matchup_dir=matchup_dir,
            matchup_name=name,
            subdir=str(subdir),
        )
        if baseline_row is not None:
            enriched.append(baseline_row)
    if enriched:
        return enriched
    return rows


def _merged_aggregate_points(
    ckpt_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in ckpt_history:
        if row.get("eval_mode") != "merged":
            continue
        aggregate = row.get("aggregate")
        if not isinstance(aggregate, dict):
            aggregate = {}
        episode = int(row.get("episodes_completed") or 0)
        win_rate_mean = aggregate.get("self_play_win_rate_mean")
        win_rate_se = aggregate.get("self_play_win_rate_se")
        vs_logic_mean = aggregate.get("vs_logic_win_rate_mean")
        vs_logic_se = aggregate.get("vs_logic_win_rate_se")
        if win_rate_mean is None or vs_logic_mean is None:
            derived = _aggregate_means_from_per_matchup(row)
            if derived is not None:
                win_rate_mean = win_rate_mean if win_rate_mean is not None else derived[0]
                win_rate_se = win_rate_se if win_rate_se is not None else derived[1]
                vs_logic_mean = vs_logic_mean if vs_logic_mean is not None else derived[2]
                vs_logic_se = vs_logic_se if vs_logic_se is not None else derived[3]
        if win_rate_mean is None and vs_logic_mean is None:
            continue
        points.append(
            {
                "episode": episode,
                "win_rate_mean": win_rate_mean,
                "win_rate_se": win_rate_se,
                "vs_logic_mean": vs_logic_mean,
                "vs_logic_se": vs_logic_se,
                "n_matchups": int(row.get("matchups_evaluated") or 0),
            }
        )
    return points


def _aggregate_means_from_per_matchup(
    row: dict[str, Any],
) -> Optional[tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]:
    per_matchup = row.get("per_matchup")
    if not isinstance(per_matchup, dict) or not per_matchup:
        return None
    self_play: list[float] = []
    vs_logic: list[float] = []
    for matchup_row in per_matchup.values():
        if not isinstance(matchup_row, dict):
            continue
        h1, _ = _hero_win_rates_from_metrics(matchup_row)
        if h1 is not None:
            self_play.append(h1)
        elif matchup_row.get("p1_win_rate") is not None:
            self_play.append(float(matchup_row["p1_win_rate"]))
        point = _checkpoint_point_from_row(matchup_row)
        if point.get("vs_logic_win_rate") is not None:
            vs_logic.append(float(point["vs_logic_win_rate"]))
    sp_mean, sp_se = _mean_and_se(self_play)
    vl_mean, vl_se = _mean_and_se(vs_logic)
    if sp_mean is None and vl_mean is None:
        return None
    return sp_mean, sp_se, vl_mean, vl_se


def _checkpoint_matchup_series(
    run_dir: Path,
    active_matchups: dict[str, Any],
    active_histories: dict[str, list[dict[str, Any]]],
    ckpt_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build agent-vs-agent hero-1 win% time series for each active matchup."""
    labels: dict[str, str] = {}
    for subdir, row in active_matchups.items():
        if not isinstance(row, dict):
            continue
        subdir_s = str(subdir)
        matchup_dir = run_dir / subdir_s
        hero1, hero2 = _read_matchup_hero_labels(matchup_dir)
        label = _format_matchup_hero_title(hero1, hero2)
        if not label.strip() or label == "? vs ?":
            label = str(row.get("name") or subdir_s)
        labels[subdir_s] = label

    by_subdir: dict[str, list[dict[str, Any]]] = {}
    for subdir_s, hist in active_histories.items():
        if subdir_s == "__run_root__":
            continue
        for row in hist:
            if row.get("eval_mode") == "merged":
                continue
            if row.get("episodes_completed") is None:
                continue
            point = _checkpoint_point_from_row(row)
            wr = point.get("agent_vs_agent_hero1_win_rate")
            if wr is None:
                continue
            by_subdir.setdefault(subdir_s, []).append(
                {"episode": int(point["episode"]), "win_rate": float(wr)}
            )

    for row in ckpt_history:
        if row.get("eval_mode") != "merged":
            continue
        episode = int(row.get("episodes_completed") or 0)
        per_matchup = row.get("per_matchup")
        if not isinstance(per_matchup, dict):
            continue
        for subdir_s, matchup_row in per_matchup.items():
            if not isinstance(matchup_row, dict):
                continue
            subdir_key = str(subdir_s)
            if subdir_key not in labels:
                labels[subdir_key] = str(matchup_row.get("matchup") or subdir_key)
            point = _checkpoint_point_from_row(matchup_row)
            wr = point.get("agent_vs_agent_hero1_win_rate")
            if wr is None:
                continue
            by_subdir.setdefault(subdir_key, []).append(
                {"episode": episode, "win_rate": float(wr)}
            )

    series_list: list[dict[str, Any]] = []
    for subdir in active_matchups:
        subdir_s = str(subdir)
        points = sorted(by_subdir.get(subdir_s, []), key=lambda p: int(p["episode"]))
        if not points:
            continue
        series_list.append(
            {
                "label": labels.get(subdir_s, subdir_s),
                "subdir": subdir_s,
                "points": points,
            }
        )
    return series_list


def _append_matchup_win_rate_history(
    row: dict[str, Any],
    *,
    episode: int,
    hero1_win_rate: float,
    hero2_win_rate: Optional[float] = None,
) -> None:
    """Append or refresh a training win-rate sample for dashboard charts."""
    history_raw = row.get("win_rate_history")
    history: list[dict[str, Any]] = (
        list(history_raw) if isinstance(history_raw, list) else []
    )
    point: dict[str, Any] = {
        "episode": int(episode),
        "hero1_win_rate": float(hero1_win_rate),
    }
    if hero2_win_rate is not None:
        point["hero2_win_rate"] = float(hero2_win_rate)
    if history and int(history[-1].get("episode", -1)) == int(episode):
        history[-1] = point
    else:
        history.append(point)
    row["win_rate_history"] = history[-TRAIN_WIN_RATE_HISTORY_LIMIT:]


def _training_matchup_series(
    run_dir: Path,
    active_matchups: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build nominal hero-1 training win% time series for each active matchup."""
    series_list: list[dict[str, Any]] = []
    for subdir, row in active_matchups.items():
        if not isinstance(row, dict):
            continue
        subdir_s = str(subdir)
        history_raw = row.get("win_rate_history")
        if not isinstance(history_raw, list):
            continue
        points: list[dict[str, Any]] = []
        for sample in history_raw:
            if not isinstance(sample, dict):
                continue
            wr = sample.get("hero1_win_rate")
            if wr is None:
                wr = sample.get("p1_win_rate")
            if wr is None or sample.get("episode") is None:
                continue
            points.append(
                {"episode": int(sample["episode"]), "win_rate": float(wr)}
            )
        if not points:
            continue
        matchup_dir = run_dir / subdir_s
        hero1, hero2 = _read_matchup_hero_labels(matchup_dir)
        label = _format_matchup_hero_title(hero1, hero2)
        if not label.strip() or label == "? vs ?":
            label = str(row.get("name") or subdir_s)
        series_list.append(
            {
                "label": label,
                "subdir": subdir_s,
                "points": sorted(points, key=lambda p: int(p["episode"])),
            }
        )
    return series_list


def _resolve_checkpoint_aggregate_points(
    ckpt_history: list[dict[str, Any]],
    active_histories: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    merged_points = _merged_aggregate_points(ckpt_history)
    if any(point.get("win_rate_mean") is not None for point in merged_points):
        return merged_points
    fallback = aggregate_checkpoint_points(active_histories)
    if any(point.get("win_rate_mean") is not None for point in fallback):
        return fallback
    return merged_points or fallback


def _checkpoint_rows_have_vs_logic(rows: list[dict[str, Any]]) -> bool:
    return any(
        row.get("vs_logic_win_rate") is not None
        or row.get("logic_vs_agent_win_rate") is not None
        for row in rows
        if isinstance(row, dict)
    )


def _checkpoint_rows_have_logic_vs_logic(rows: list[dict[str, Any]]) -> bool:
    return any(
        row.get("logic_vs_logic_win_rate") is not None
        for row in rows
        if isinstance(row, dict)
    )


def _rows_from_latest_merged(
    ckpt_history: list[dict[str, Any]],
    active_matchups: dict[str, Any],
    *,
    run_dir: Path,
) -> list[dict[str, Any]]:
    for row in reversed(ckpt_history):
        if row.get("eval_mode") != "merged":
            continue
        per_matchup = row.get("per_matchup")
        if not isinstance(per_matchup, dict):
            continue
        rows: list[dict[str, Any]] = []
        for subdir, matchup_row in per_matchup.items():
            if not isinstance(matchup_row, dict):
                continue
            point = _checkpoint_point_from_row(matchup_row)
            point["matchup_dir"] = str(subdir)
            rows.append(
                _apply_hero_labels_to_checkpoint_row(
                    point,
                    run_dir / str(subdir),
                )
            )
        return rows
    return []


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
            if row.get("eval_mode") == "merged":
                per_matchup = row.get("per_matchup")
                if isinstance(per_matchup, dict):
                    for matchup_row in per_matchup.values():
                        if not isinstance(matchup_row, dict):
                            continue
                        if matchup_row.get("episodes_completed") is None:
                            continue
                        point = _checkpoint_point_from_row(matchup_row)
                        episode = int(point["episode"])
                        by_episode.setdefault(episode, []).append(point)
                continue
            if row.get("episodes_completed") is None:
                continue
            point = _checkpoint_point_from_row(row)
            episode = int(point["episode"])
            by_episode.setdefault(episode, []).append(point)

    aggregate: list[dict[str, Any]] = []
    for episode in sorted(by_episode.keys()):
        points = by_episode[episode]
        self_play = [
            float(p.get("agent_vs_agent_hero1_win_rate") or p.get("win_rate") or 0.0)
            for p in points
        ]
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
        point["matchup_dir"] = str(subdir)
        if latest.get("eval_episodes") is not None:
            point["eval_episodes"] = int(latest["eval_episodes"])
        rows.append(
            _apply_hero_labels_to_checkpoint_row(
                point,
                run_dir / str(subdir),
            )
        )
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
    talishar_backends = (
        live.get("talishar_backends")
        or manifest.get("talishar_backends")
        or []
    )
    if not isinstance(talishar_backends, list):
        talishar_backends = []
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

    checkpoint_aggregate_points = _resolve_checkpoint_aggregate_points(
        ckpt_history,
        active_histories,
    )
    checkpoint_matchup_series = _checkpoint_matchup_series(
        run_dir,
        active_matchups,
        active_histories,
        ckpt_history,
    )
    training_matchup_series = _training_matchup_series(run_dir, active_matchups)
    active_checkpoint_rows = _rows_from_latest_merged(
        ckpt_history,
        active_matchups,
        run_dir=run_dir,
    )
    if not active_checkpoint_rows:
        active_checkpoint_rows = _latest_checkpoint_rows_for_active(
            run_dir, active_matchups
        )
    if not active_checkpoint_rows and ckpt_history:
        latest = ckpt_history[-1]
        if isinstance(latest, dict):
            if latest.get("status") == "error" and latest.get("eval_mode") == "merged":
                active_checkpoint_rows = [
                    {
                        "matchup": "merged checkpoint eval (failed)",
                        "episode": int(latest.get("episodes_completed") or 0),
                        "eval_episodes": int(latest.get("eval_episodes_total") or 0),
                        "error": str(latest.get("error") or "unknown"),
                        "timeout_rate": 1.0,
                    }
                ]
            else:
                point = _checkpoint_point_from_row(latest)
                point["matchup"] = str(point.get("matchup") or current_name or "—")
                active_checkpoint_rows = [point]
    active_checkpoint_rows = _enrich_checkpoint_rows_with_baselines(
        run_dir,
        active_checkpoint_rows,
        active_matchups,
    )
    run_format = _normalize_run_format(manifest.get("format"))
    fabrary_meta_doc = load_fabrary_meta()
    active_checkpoint_rows = _enrich_checkpoint_rows_with_fabrary_meta(
        run_dir,
        active_checkpoint_rows,
        format_name=run_format,
        meta=fabrary_meta_doc,
    )
    checkpoint_points: list[dict[str, Any]] = []
    for row in ckpt_history:
        if not isinstance(row, dict) or row.get("episodes_completed") is None:
            continue
        checkpoint_points.append(_checkpoint_point_from_row(row))

    completed_rows = [
        _matchup_summary_row(matchup_dir, target_episodes)
        for matchup_dir in reversed(iter_unified_matchup_dirs(run_dir))
        if _is_matchup_complete(matchup_dir, target_episodes)
    ]
    completed_rows = _enrich_completed_rows_with_fabrary_meta(
        run_dir,
        completed_rows,
        format_name=run_format,
        meta=fabrary_meta_doc,
    )

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

    return {
        "run_dir": str(run_dir),
        "format": str(manifest.get("format") or "—"),
        "started_at": str(manifest.get("started_at") or ""),
        "matchups_total": matchups_total,
        "matchups_completed": matchups_completed,
        "matchups_sampled": list(manifest.get("matchups_sampled") or []),
        "target_episodes": target_episodes,
        "parallel_matchups": parallel_matchups,
        "talishar_backends": talishar_backends,
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
        "checkpoint_matchup_series": checkpoint_matchup_series,
        "training_matchup_series": training_matchup_series,
        "active_checkpoint_rows": active_checkpoint_rows,
        "completed_matchups": completed_rows,
        "overall_pct": overall_pct,
        "complete": status == "complete",
        "policy_weights": live.get("policy_weights"),
        "policy_weight_history": live.get("policy_weight_history") or [],
        "checkpoint_eval_logic_vs_logic": bool(
            manifest.get("checkpoint_eval_logic_vs_logic", True)
        ),
        "checkpoint_eval_agent_vs_logic": bool(
            manifest.get("checkpoint_eval_agent_vs_logic", False)
        ),
        "show_logic_vs_logic_column": bool(
            manifest.get("checkpoint_eval_logic_vs_logic", True)
        )
        or _checkpoint_rows_have_logic_vs_logic(active_checkpoint_rows),
        "show_agent_vs_logic_columns": bool(
            manifest.get("checkpoint_eval_agent_vs_logic", False)
        )
        or _checkpoint_rows_have_vs_logic(active_checkpoint_rows),
        "fabrary_meta": {
            "period": DEFAULT_PERIOD,
            "fetched_at": str(fabrary_meta_doc.get("fetched_at") or ""),
            "source": str(fabrary_meta_doc.get("source") or "fabrary.net/meta-results"),
        },
    }


def _svg_winrate_chart_with_error_bands(
    points: list[dict[str, Any]],
    *,
    width: int = 520,
    height: int = 200,
    target_episodes: int = 0,
    empty_message: str = "Waiting for checkpoint eval…",
    show_vs_logic: bool = True,
) -> str:
    plottable = [
        point
        for point in points
        if point.get("win_rate_mean") is not None
        or (show_vs_logic and point.get("vs_logic_mean") is not None)
    ]
    if not plottable:
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
        max(int(p.get("episode", 0) or 0) for p in plottable),
        1,
    )

    def px(ep: float) -> float:
        return pad_l + (ep / max_x) * plot_w

    def py(rate: float) -> float:
        centered = _centered_win_rate_deviation(rate)
        return pad_t + (0.5 - centered) * plot_h

    def band(mean_key: str, se_key: str, class_prefix: str) -> str:
        upper = []
        lower = []
        for point in plottable:
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
            for point in plottable
            if point.get(mean_key) is not None
        )
        markers = ""
        if line.count(",") <= 1:
            for point in plottable:
                mean = point.get(mean_key)
                if mean is None:
                    continue
                cx = px(point["episode"])
                cy = py(float(mean))
                markers += (
                    f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" '
                    f'class="{class_prefix}-line"/>'
                )
        return (
            f'<polygon points="{polygon}" class="{class_prefix}-band"/>'
            f'<polyline points="{line}" class="{class_prefix}-line" fill="none"/>'
            f"{markers}"
        )

    grid = "\n".join(
        f'<line x1="{pad_l}" y1="{py(v):.1f}" x2="{width - pad_r}" y2="{py(v):.1f}" '
        f'class="{"chart-grid-zero" if v == 0.5 else "chart-grid"}"/>'
        f'<text x="{pad_l - 6}" y="{py(v) + 4:.1f}" text-anchor="end" class="chart-axis">'
        f"{_centered_axis_label(v)}</text>"
        for v in (0.0, 0.5, 1.0)
    )
    has_vs_logic = show_vs_logic and any(
        point.get("vs_logic_mean") is not None for point in plottable
    )
    if has_vs_logic:
        legend = (
            '<text x="' + str(pad_l) + f'" y="{height - 8}" class="chart-axis">'
            "— agent vs agent mean ± SE  ·  — agent vs logic mean ± SE</text>"
        )
        vs_logic_band = band("vs_logic_mean", "vs_logic_se", "chart-vslogic")
    else:
        legend = (
            '<text x="' + str(pad_l) + f'" y="{height - 8}" class="chart-axis">'
            "— agent vs agent mean ± SE</text>"
        )
        vs_logic_band = ""
    return f"""<svg viewBox="0 0 {width} {height}" class="chart">
  {grid}
  {band("win_rate_mean", "win_rate_se", "chart-selfplay")}
  {vs_logic_band}
  {legend}
</svg>"""


_MATCHUP_LINE_COLORS = (
    "#5b9cff",
    "#3ecf8e",
    "#f5a623",
    "#e85d75",
    "#b388ff",
    "#4dd0e1",
    "#ff8a65",
    "#aed581",
)


def _centered_win_rate_deviation(win_rate: float) -> float:
    """Map absolute win rate to deviation from even (50% → 0)."""
    return float(win_rate) - 0.5


def _centered_axis_label(win_rate: float) -> str:
    """Format a grid label where 50% win rate reads as 0%."""
    deviation_pct = _centered_win_rate_deviation(win_rate) * 100.0
    if abs(deviation_pct) < 0.05:
        return "0%"
    sign = "+" if deviation_pct > 0 else ""
    if abs(deviation_pct - round(deviation_pct)) < 0.05:
        return f"{sign}{int(round(deviation_pct))}%"
    return f"{sign}{deviation_pct:.1f}%"


def _svg_centered_matchup_lines_chart(
    series_list: list[dict[str, Any]],
    *,
    width: int = 520,
    height: int = 200,
    target_episodes: int = 0,
    empty_message: str = "Waiting for data…",
) -> str:
    plottable = [
        series
        for series in series_list
        if isinstance(series, dict) and isinstance(series.get("points"), list) and series["points"]
    ]
    if not plottable:
        return (
            f'<svg viewBox="0 0 {width} {height}" class="chart empty">'
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            f'class="chart-empty">{html.escape(empty_message)}</text></svg>'
        )

    pad_l, pad_r, pad_t = 40, 12, 12
    legend_rows = len(plottable) if len(plottable) > 3 else 1
    pad_b = 16 + legend_rows * 14
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    all_points = [
        point
        for series in plottable
        for point in series["points"]
        if isinstance(point, dict) and point.get("episode") is not None
    ]
    max_x = max(
        int(target_episodes or 0),
        max(int(point.get("episode", 0) or 0) for point in all_points),
        1,
    )

    def px(ep: float) -> float:
        return pad_l + (ep / max_x) * plot_w

    def py(rate: float) -> float:
        centered = _centered_win_rate_deviation(rate)
        return pad_t + (0.5 - centered) * plot_h

    grid = "\n".join(
        f'<line x1="{pad_l}" y1="{py(v):.1f}" x2="{width - pad_r}" y2="{py(v):.1f}" '
        f'class="{"chart-grid-zero" if v == 0.5 else "chart-grid"}"/>'
        f'<text x="{pad_l - 6}" y="{py(v) + 4:.1f}" text-anchor="end" class="chart-axis">'
        f"{_centered_axis_label(v)}</text>"
        for v in (0.0, 0.5, 1.0)
    )

    lines = ""
    legend_items: list[str] = []
    for index, series in enumerate(plottable):
        color = _MATCHUP_LINE_COLORS[index % len(_MATCHUP_LINE_COLORS)]
        points = [
            point
            for point in series["points"]
            if isinstance(point, dict) and point.get("win_rate") is not None
        ]
        if not points:
            continue
        poly = " ".join(
            f"{px(point['episode']):.1f},{py(float(point['win_rate'])):.1f}"
            for point in points
        )
        lines += (
            f'<polyline points="{poly}" fill="none" stroke="{color}" '
            f'stroke-width="2" class="chart-matchup-line"/>'
        )
        if poly.count(",") <= 1:
            for point in points:
                cx = px(point["episode"])
                cy = py(float(point["win_rate"]))
                lines += (
                    f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" '
                    f'fill="{color}" class="chart-matchup-dot">'
                    f"<title>{html.escape(str(series.get('label') or ''))} "
                    f"ep {int(point['episode'])}: {_pct(point['win_rate'])} "
                    f"({_centered_axis_label(float(point['win_rate']))} vs even)</title></circle>"
                )
        label = str(series.get("label") or series.get("subdir") or f"Matchup {index + 1}")
        legend_items.append(
            f'<tspan fill="{color}">—</tspan> {html.escape(label)}'
        )

    legend_y = height - 6
    legend_x = pad_l
    if len(legend_items) <= 3:
        legend = (
            f'<text x="{legend_x}" y="{legend_y}" class="chart-axis">'
            + "  ·  ".join(legend_items)
            + "</text>"
        )
    else:
        legend = "\n  ".join(
            f'<text x="{legend_x}" y="{legend_y - (len(legend_items) - 1 - row_idx) * 12}" '
            f'class="chart-axis">{item}</text>'
            for row_idx, item in enumerate(legend_items)
        )

    return f"""<svg viewBox="0 0 {width} {height}" class="chart">
  {grid}
  {lines}
  {legend}
</svg>"""


def _svg_checkpoint_matchup_lines_chart(
    series_list: list[dict[str, Any]],
    *,
    width: int = 520,
    height: int = 200,
    target_episodes: int = 0,
    empty_message: str = "Waiting for checkpoint eval…",
) -> str:
    return _svg_centered_matchup_lines_chart(
        series_list,
        width=width,
        height=height,
        target_episodes=target_episodes,
        empty_message=empty_message,
    )


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
        centered = _centered_win_rate_deviation(rate)
        return pad_t + (0.5 - centered) * plot_h

    poly = " ".join(
        f"{px(p['episode']):.1f},{py(p['win_rate']):.1f}" for p in points
    )
    dots = "\n".join(
        f'<circle cx="{px(p["episode"]):.1f}" cy="{py(p["win_rate"]):.1f}" r="3.5" class="chart-dot">'
        f"<title>ep {p['episode']}: {_pct(p['win_rate'])} "
        f"({_centered_axis_label(float(p['win_rate']))} vs even)</title></circle>"
        for p in points
    )
    grid = "\n".join(
        f'<line x1="{pad_l}" y1="{py(v):.1f}" x2="{width - pad_r}" y2="{py(v):.1f}" '
        f'class="{"chart-grid-zero" if v == 0.5 else "chart-grid"}"/>'
        f'<text x="{pad_l - 6}" y="{py(v) + 4:.1f}" text-anchor="end" class="chart-axis">'
        f"{_centered_axis_label(v)}</text>"
        for v in (0.0, 0.5, 1.0)
    )
    return f"""<svg viewBox="0 0 {width} {height}" class="chart">
  {grid}
  <polyline points="{poly}" class="chart-line" fill="none"/>
  {dots}
</svg>"""


def _svg_l2_norm_sparkline(
    history: list[dict[str, Any]],
    *,
    width: int = 520,
    height: int = 72,
) -> str:
    values = [
        float(row["l2_norm"])
        for row in history
        if isinstance(row, dict) and row.get("l2_norm") is not None
    ]
    if len(values) < 2:
        return ""
    pad_l, pad_r, pad_t, pad_b = 36, 12, 8, 16
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    min_v = min(values)
    max_v = max(values)
    span = max(max_v - min_v, 1e-9)

    def px(i: int) -> float:
        return pad_l + (i / max(1, len(values) - 1)) * plot_w

    def py(v: float) -> float:
        return pad_t + plot_h - ((v - min_v) / span) * plot_h

    poly = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(values))
    return f"""<svg viewBox="0 0 {width} {height}" class="chart sparkline">
  <text x="{pad_l}" y="{height - 4}" class="chart-axis">L2 norm trend ({len(values)} updates)</text>
  <polyline points="{poly}" class="chart-selfplay-line" fill="none"/>
</svg>"""


def _render_policy_weights_card(state: dict[str, Any]) -> str:
    weights = state.get("policy_weights")
    if not isinstance(weights, dict) or not weights.get("initialized"):
        return '<p class="muted">Waiting for unified policy weights…</p>'

    history_raw = state.get("policy_weight_history") or []
    history = [row for row in history_raw if isinstance(row, dict)]
    update_count = weights.get("update_count")
    l2_norm = weights.get("l2_norm")
    fingerprint = str(weights.get("fingerprint") or "—")
    param_count = int(weights.get("param_count") or 0)

    delta_text = ""
    if len(history) >= 2:
        prev = history[-2].get("l2_norm")
        if prev is not None and l2_norm is not None:
            delta = float(l2_norm) - float(prev)
            sign = "+" if delta >= 0 else ""
            delta_text = f" ({sign}{delta:.4f} vs prior update)"

    sparkline = _svg_l2_norm_sparkline(history)

    recent_rows = ""
    for row in reversed(history[-8:]):
        recent_rows += (
            "<tr>"
            f"<td>{html.escape(str(row.get('update_count') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('l2_norm') or '—'))}</td>"
            f"<td><code>{html.escape(str(row.get('fingerprint') or '—'))}</code></td>"
            "</tr>"
        )
    recent_table = ""
    if recent_rows:
        recent_table = (
            '<table class="history weights-history"><thead><tr>'
            "<th>Update</th><th>L2 norm</th><th>Fingerprint</th>"
            f"</tr></thead><tbody>{recent_rows}</tbody></table>"
        )

    return f"""
      <dl class="weights-summary">
        <div><dt>Architecture</dt><dd>d_model={weights.get('d_model')} · layers={weights.get('n_layers')} · heads={weights.get('n_heads')}</dd></div>
        <div><dt>Observation / actions</dt><dd>obs_dim={weights.get('obs_dim')} · n_actions={weights.get('n_actions')}</dd></div>
        <div><dt>Parameters</dt><dd>{param_count:,}</dd></div>
        <div><dt>PPO updates</dt><dd>{html.escape(str(update_count if update_count is not None else '—'))}</dd></div>
        <div><dt>L2 norm</dt><dd>{html.escape(str(l2_norm if l2_norm is not None else '—'))}{html.escape(delta_text)}</dd></div>
        <div><dt>Fingerprint</dt><dd><code>{html.escape(fingerprint)}</code></dd></div>
      </dl>
      {sparkline}
      {recent_table}"""


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
        run_path = Path(str(state.get("run_dir") or ""))
        for subdir, row in active_matchups.items():
            if not isinstance(row, dict):
                continue
            if run_path.is_dir():
                hero_title = _format_matchup_hero_title(
                    *_read_matchup_hero_labels(run_path / str(subdir))
                )
            else:
                hero_title = str(row.get("name") or subdir)
            eps = int(row.get("episodes_completed") or 0)
            pct = (eps / max(1, target_eps)) * 100.0 if target_eps else 0.0
            progress_blocks += f"""
      <div class="matchup-progress">
        <div class="progress-label"><span>{html.escape(hero_title)}</span><span>{eps}/{target_eps}</span></div>
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

    ckpt_chart = _svg_checkpoint_matchup_lines_chart(
        state.get("checkpoint_matchup_series") or [],
        target_episodes=target_eps,
    )
    train_chart = _svg_centered_matchup_lines_chart(
        state.get("training_matchup_series") or [],
        target_episodes=target_eps,
        empty_message="Waiting for training rollouts…",
    )
    policy_weights_card = _render_policy_weights_card(state)

    completed_rows = state.get("completed_matchups") or []
    if completed_rows:
        history_rows = "\n".join(
            f"<tr>"
            f"<td>{html.escape(str(row.get('name', '')))}</td>"
            f"<td>{_pct(row.get('first_checkpoint_win_rate'))}</td>"
            f"<td>{_pct(row.get('checkpoint_win_rate'))}</td>"
            f"<td>{_pct(row.get('checkpoint_vs_logic_win_rate'))}</td>"
            f"<td>{_pct(row.get('fabrary_competitive_p1_win_rate'))}</td>"
            f"</tr>"
            for row in completed_rows
        )
        history_table = f"""
<table class="history">
  <thead><tr><th>Matchup</th><th>First self-play ckpt</th><th>Last self-play ckpt</th><th>Last vs logic</th><th>Fabrary comp (H1%)</th></tr></thead>
  <tbody>{history_rows}</tbody>
</table>"""
    else:
        history_table = '<p class="muted">No completed matchups yet.</p>'

    show_logic_vs_logic = bool(state.get("show_logic_vs_logic_column"))
    show_agent_vs_logic = bool(state.get("show_agent_vs_logic_columns"))
    ckpt_rows = ""
    for row in state.get("active_checkpoint_rows") or []:
        eval_games = row.get("eval_episodes")
        eval_games_cell = (
            str(int(eval_games)) if eval_games is not None else "—"
        )
        episode_cell = str(row.get("episode_label") or int(row.get("episode", 0)))
        cells = [
            f"<td>{html.escape(str(row.get('matchup') or '—'))}</td>",
            f"<td>{html.escape(episode_cell)}</td>",
            f"<td>{eval_games_cell}</td>",
        ]
        if show_logic_vs_logic:
            cells.extend(
                [
                    f"<td>{_pct(row.get('logic_vs_logic_hero1_win_rate'))}</td>",
                ]
            )
        if show_agent_vs_logic:
            cells.extend(
                [
                    f"<td>{_pct(row.get('vs_logic_hero1_win_rate'))}</td>",
                    f"<td>{_pct(row.get('logic_vs_agent_hero1_win_rate'))}</td>",
                ]
            )
        cells.extend(
            [
                f"<td>{_pct(row.get('agent_vs_agent_hero1_win_rate'))}</td>",
                f"<td>{_pct(row.get('timeout_rate'))}</td>",
                f"<td>{_pct(row.get('fabrary_competitive_p1_win_rate'))}</td>",
                f"<td>{row.get('fabrary_competitive_plays') if row.get('fabrary_competitive_plays') is not None else '—'}</td>",
                f"<td>{_pct(row.get('fabrary_all_p1_win_rate'))}</td>",
                f"<td>{row.get('fabrary_all_plays') if row.get('fabrary_all_plays') is not None else '—'}</td>",
            ]
        )
        ckpt_rows += f"<tr>{''.join(cells)}</tr>"
    header_cells = ["Matchup", "Episode", "Eval games"]
    if show_logic_vs_logic:
        header_cells.extend(
            [
                "Logic vs logic (Hero 1 win%)",
            ]
        )
    if show_agent_vs_logic:
        header_cells.extend(
            [
                "Vs logic win% (Hero 1)",
                "Logic win% vs agent (Hero 1)",
            ]
        )
    header_cells.extend(
        [
            "Agent vs agent (Hero 1 win%)",
            "Timeout %",
            "Fabrary comp (H1%)",
            "Fabrary comp (n)",
            "Fabrary all (H1%)",
            "Fabrary all (n)",
        ]
    )
    colspan = len(header_cells)
    header_row = "".join(f"<th>{html.escape(label)}</th>" for label in header_cells)
    empty_ckpt_row = (
        f'<tr><td colspan="{colspan}" class="muted">No checkpoint eval yet</td></tr>'
    )
    ckpt_table = (
        f'<table class="history"><thead><tr>{header_row}</tr></thead>'
        f"<tbody>{ckpt_rows or empty_ckpt_row}</tbody></table>"
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
    .weights-summary {{ display: grid; gap: 10px; margin: 0 0 14px; }}
    .weights-summary > div {{ display: grid; grid-template-columns: 160px 1fr; gap: 10px; font-size: 0.9rem; }}
    .weights-summary dt {{ margin: 0; color: var(--muted); }}
    .weights-summary dd {{ margin: 0; }}
    .weights-history {{ margin-top: 12px; }}
    .sparkline {{ margin-top: 8px; }}
    code {{ font-family: Consolas, "Courier New", monospace; font-size: 0.82rem; color: #c5d4ea; }}
    .muted {{ color: var(--muted); }}
    a {{ color: var(--primary); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .chart {{ width: 100%; max-width: 100%; }}
    .chart-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; margin-bottom: 16px; }}
    .chart-panel h3 {{ margin: 0 0 8px; font-size: 0.92rem; color: var(--muted); font-weight: 600; }}
    .chart-caption {{ margin: 0 0 12px; font-size: 0.82rem; color: var(--muted); }}
    .chart-selfplay-line {{ stroke: var(--primary); stroke-width: 2; }}
    .chart-selfplay-band {{ fill: rgba(91, 156, 255, 0.18); stroke: none; }}
    .chart-vslogic-line {{ stroke: var(--good); stroke-width: 2; }}
    .chart-vslogic-band {{ fill: rgba(62, 207, 142, 0.15); stroke: none; }}
    .chart-grid {{ stroke: #243044; stroke-width: 1; }}
    .chart-grid-zero {{ stroke: #3d516d; stroke-width: 1.5; }}
    .chart-axis {{ fill: var(--muted); font-size: 10px; }}
    .chart-empty {{ fill: var(--muted); font-size: 12px; }}
    footer {{ margin-top: 28px; color: var(--muted); font-size: 0.75rem; border-top: 1px solid var(--border); padding-top: 14px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Training AI agents with random matchups</h1>
    <p class="sub">{html.escape(str(state.get('format', '—')).replace('_', ' '))} ·
      <span class="status-pill{' complete' if status == 'complete' else ''}">{html.escape(status_label)}</span><br>
      <span class="muted">Fabrary reference = Talishar online hero meta (fabrary.net); deck matchups mapped via hero_id</span>
    </p>

    <p class="progress-headline">{html.escape(progress_headline)}</p>

    <div class="card">
      <h2>Training progress</h2>
      {progress_blocks}
    </div>

    <div class="card">
      <h2>Win rate charts (active batch)</h2>
      <p class="chart-caption">Y axis centered at even matchups: 0% = 50% Hero 1 win rate, +10% = 60%, −10% = 40%.</p>
      <div class="chart-row">
        <div class="chart-panel">
          <h3>Training rollouts</h3>
          {train_chart}
        </div>
        <div class="chart-panel">
          <h3>Checkpoint eval</h3>
          {ckpt_chart}
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Checkpoint eval (active batch)</h2>
      {ckpt_table}
    </div>

    <div class="card">
      <h2>Completed matchups</h2>
      {history_table}
    </div>

    <div class="card">
      <h2>Agent weights</h2>
      {policy_weights_card}
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
