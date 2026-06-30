"""Diagnostic logging when eval/render episodes appear stuck on Talishar."""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from flesh_and_blood_rlbridge.combat_log_tracker import (
    extract_talishar_chat_log_lines,
    talishar_gamestate_revert_detected,
)
from flesh_and_blood_rlbridge.state_loop_guard import (
    board_state_fingerprint,
    decision_point_fingerprint,
)
from flesh_and_blood_rlbridge.talishar_default_policy import _is_pass_action

from fab_bridge.unified_training_debug import shard_label

_LOCK = threading.Lock()
_ENABLED = False
_RUN_DIR: Optional[Path] = None
_LOG_PATH: Optional[Path] = None


def is_enabled() -> bool:
    if _ENABLED:
        return True
    token = os.environ.get("FAB_ANTI_STUCK_LOGGING", "").strip().lower()
    return token in {"1", "true", "yes", "on"}


def configure(*, run_dir: Path | str, enabled: bool) -> Path | None:
    """Enable anti-stuck logging for a run directory."""
    global _ENABLED, _RUN_DIR, _LOG_PATH  # noqa: PLW0603
    env_on = os.environ.get("FAB_ANTI_STUCK_LOGGING", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    with _LOCK:
        _ENABLED = bool(enabled) or env_on
        _RUN_DIR = Path(run_dir).expanduser().resolve()
        _LOG_PATH = _RUN_DIR / "anti_stuck_reports.jsonl" if _ENABLED else None
        if _ENABLED and _LOG_PATH is not None:
            _RUN_DIR.mkdir(parents=True, exist_ok=True)
            _write_unlocked(
                "anti_stuck",
                "Anti-stuck logging enabled",
                log_path=str(_LOG_PATH),
            )
        return _LOG_PATH


def log_path() -> Optional[Path]:
    return _LOG_PATH


def read_from_manifest(run_dir: Path) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return False
    return bool(data.get("anti_stuck_logging"))


def _write_unlocked(category: str, message: str, **details: Any) -> None:
    if not _ENABLED or _LOG_PATH is None:
        return
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "message": message,
    }
    if details:
        record["details"] = details
    line = json.dumps(record, default=str) + "\n"
    with _LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line)
    print(f"  [anti-stuck] {message}", flush=True)


def log_event(category: str, message: str, **details: Any) -> None:
    if not is_enabled():
        return
    with _LOCK:
        _write_unlocked(category, message, **details)


@dataclass(frozen=True)
class AntiStuckConfig:
    pass_streak: int = 8
    no_progress_steps: int = 6
    repeat_streak: int = 5
    loop_guard_streak: int = 4


def _legal_actions_from_obs(obs_data: dict[str, Any]) -> list[dict[str, Any]]:
    legal = obs_data.get("legal_actions")
    if isinstance(legal, list) and legal:
        return [a for a in legal if isinstance(a, dict)]
    entries = obs_data.get("legalActions") or []
    out: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        out.append(
            {
                "action_code": int(entry.get("action_code", 0) or 0),
                "label": str(entry.get("label", "") or ""),
                "zone": str(entry.get("zone", "") or ""),
                "button_input": str(entry.get("button_input", "") or ""),
            }
        )
    return out


def _resolve_submitted_action(
    obs_data: dict[str, Any],
    action: Any,
) -> dict[str, Any]:
    legal = _legal_actions_from_obs(obs_data)
    if isinstance(action, int) and 0 <= action < len(legal):
        return legal[action]
    if isinstance(action, dict):
        return {
            "action_code": int(action.get("action_code", 0) or 0),
            "label": str(action.get("label", "") or ""),
            "zone": str(action.get("zone", "") or ""),
            "button_input": str(action.get("button_input", "") or ""),
        }
    return {
        "action_code": 0,
        "label": str(action),
        "zone": "",
        "button_input": "",
    }


def _obs_state_dict(obs_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "turnNo": int(obs_data.get("turnNo", 0) or 0),
        "turnPhase": str(obs_data.get("turnPhase", "") or ""),
        "playerHealth": obs_data.get("playerHealth", 0),
        "opponentHealth": obs_data.get("opponentHealth", 0),
        "playerHand": obs_data.get("playerHand", []),
        "playerPitchCount": obs_data.get("playerPitchCount", 0),
        "playerHandSize": obs_data.get("playerHandSize", 0),
        "opponentHandSize": obs_data.get("opponentHandSize", 0),
        "chatLog": obs_data.get("chatLog", ""),
    }


def _progress_signature(
    *,
    obs_data: dict[str, Any],
    server_report: dict[str, Any],
) -> str:
    turn_no = int(server_report.get("turn_no", obs_data.get("turnNo", 0)) or 0)
    phase = str(server_report.get("phase", obs_data.get("turnPhase", "")) or "")
    p1_hp = float(
        server_report.get("player_health", obs_data.get("playerHealth", 0)) or 0
    )
    p2_hp = float(
        server_report.get("opponent_health", obs_data.get("opponentHealth", 0)) or 0
    )
    board_fp = str(server_report.get("board_fingerprint", "") or "")
    if not board_fp:
        board_fp = board_state_fingerprint(_obs_state_dict(obs_data))
    return f"{turn_no}|{phase}|{int(p1_hp + p2_hp)}|{board_fp}"


def _compact_legal_actions(legal_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "action_code": int(a.get("action_code", 0) or 0),
            "label": str(a.get("label", "") or ""),
            "zone": str(a.get("zone", "") or ""),
        }
        for a in legal_actions[:40]
    ]


@dataclass
class AntiStuckMonitor:
    """Per-episode stuck-pattern detector for eval/render loops."""

    config: AntiStuckConfig = field(default_factory=AntiStuckConfig)
    _context: dict[str, Any] = field(default_factory=dict, init=False)
    _step: int = field(default=0, init=False)
    _recent_actions: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=10),
        init=False,
    )
    _logged_kinds: set[str] = field(default_factory=set, init=False)
    _active_kinds: set[str] = field(default_factory=set, init=False)
    _pass_loop_streak: int = field(default=0, init=False)
    _last_progress_signature: str = field(default="", init=False)
    _decision_loop_streak: int = field(default=0, init=False)
    _last_decision_fingerprint: str = field(default="", init=False)
    _loop_guard_streak: int = field(default=0, init=False)
    _last_server_report: dict[str, Any] = field(default_factory=dict, init=False)

    def begin_episode(self, **context: Any) -> None:
        self._context = dict(context)
        self._step = 0
        self._recent_actions.clear()
        self._logged_kinds.clear()
        self._active_kinds.clear()
        self._pass_loop_streak = 0
        self._last_progress_signature = ""
        self._decision_loop_streak = 0
        self._last_decision_fingerprint = ""
        self._loop_guard_streak = 0
        self._last_server_report = {}

    def observe_step(
        self,
        env: Any,
        *,
        obs_data: dict[str, Any],
        step_info: dict[str, Any],
        action: Any,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        if not is_enabled():
            return

        self._step += 1
        submitted = _resolve_submitted_action(obs_data, action)
        self._recent_actions.append(
            {
                "step": self._step,
                "action_code": submitted.get("action_code"),
                "label": submitted.get("label"),
                "zone": submitted.get("zone"),
            }
        )

        server_report = self._fetch_server_report(env)
        self._last_server_report = server_report
        legal_actions = _legal_actions_from_obs(obs_data)
        if not legal_actions and server_report.get("legal_actions"):
            legal_actions = list(server_report["legal_actions"])

        state_for_fp = _obs_state_dict(obs_data)
        decision_fp = decision_point_fingerprint(state_for_fp, legal_actions)
        progress_sig = _progress_signature(
            obs_data=obs_data,
            server_report=server_report,
        )
        game_over = bool(
            terminated
            or truncated
            or server_report.get("game_over")
            or obs_data.get("gameOver")
        )
        legal_count = int(
            server_report.get("legal_count", len(legal_actions)) or 0
        )

        self._update_pass_loop(submitted, progress_sig)
        self._update_decision_loop(decision_fp)
        self._update_loop_guard(step_info)

        pending: list[tuple[str, dict[str, Any]]] = []

        if not game_over and legal_count == 0:
            self._set_active("no_legal_actions", True)
            pending.append(("no_legal_actions", {}))
        else:
            self._set_active("no_legal_actions", False)

        repeat_streak = int(step_info.get("repeat_streak", 0) or 0)
        info_decision_streak = int(step_info.get("decision_loop_streak", 0) or 0)
        if (
            repeat_streak >= self.config.repeat_streak
            or info_decision_streak >= self.config.repeat_streak
            or self._loop_guard_streak >= self.config.loop_guard_streak
        ):
            self._set_active("action_loop", True)
            pending.append(
                (
                    "action_loop",
                    {
                        "repeat_streak": repeat_streak,
                        "decision_loop_streak": info_decision_streak,
                        "loop_guard_streak": self._loop_guard_streak,
                    },
                )
            )
        else:
            self._set_active("action_loop", False)

        if server_report.get("gamestate_revert") or talishar_gamestate_revert_detected(
            state_for_fp
        ):
            self._set_active("gamestate_revert", True)
            pending.append(("gamestate_revert", {}))
        else:
            self._set_active("gamestate_revert", False)

        for stuck_kind, extra in pending:
            self._maybe_log(
                stuck_kind,
                env=env,
                obs_data=obs_data,
                step_info=step_info,
                server_report=server_report,
                legal_actions=legal_actions,
                extra=extra,
            )
        self._flush_pending_loop_logs(
            env,
            obs_data=obs_data,
            step_info=step_info,
            server_report=server_report,
            legal_actions=legal_actions,
        )

    def finish_episode(self, outcome: str) -> None:
        if not is_enabled() or not self._active_kinds:
            return
        log_event(
            "anti_stuck",
            "episode_stuck_summary",
            outcome=outcome,
            episode=self._context.get("episode"),
            mode=self._context.get("mode"),
            step=self._step,
            active_kinds=sorted(self._active_kinds),
            p1_deck=self._context.get("p1_deck"),
            p2_deck=self._context.get("p2_deck"),
            matchup=self._context.get("matchup"),
            shard=self._context.get("shard"),
            recent_actions=list(self._recent_actions),
            server_report=self._last_server_report,
        )

    def _fetch_server_report(self, env: Any) -> dict[str, Any]:
        getter = getattr(env, "get_server_report", None)
        if callable(getter):
            try:
                report = getter()
                if isinstance(report, dict):
                    return report
            except Exception:
                return {}
        return {}

    def _update_pass_loop(
        self,
        submitted: dict[str, Any],
        progress_sig: str,
    ) -> None:
        if _is_pass_action(submitted) and progress_sig == self._last_progress_signature:
            self._pass_loop_streak += 1
        else:
            self._pass_loop_streak = 1 if _is_pass_action(submitted) else 0
        self._last_progress_signature = progress_sig

        if self._pass_loop_streak >= self.config.pass_streak:
            self._set_active("pass_loop", True)
        else:
            self._set_active("pass_loop", False)

    def _update_decision_loop(self, decision_fp: str) -> None:
        if decision_fp == self._last_decision_fingerprint:
            self._decision_loop_streak += 1
        else:
            self._decision_loop_streak = 1
        self._last_decision_fingerprint = decision_fp

        if self._decision_loop_streak >= self.config.no_progress_steps:
            self._set_active("decision_loop", True)
        else:
            self._set_active("decision_loop", False)

    def _update_loop_guard(self, step_info: dict[str, Any]) -> None:
        if bool(step_info.get("loop_guard_forced_pass")):
            self._loop_guard_streak += 1
        else:
            self._loop_guard_streak = 0

    def _set_active(self, kind: str, active: bool) -> None:
        if active:
            self._active_kinds.add(kind)
        else:
            self._active_kinds.discard(kind)

    def _maybe_log(
        self,
        stuck_kind: str,
        *,
        env: Any,
        obs_data: dict[str, Any],
        step_info: dict[str, Any],
        server_report: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        if stuck_kind in self._logged_kinds:
            return
        if stuck_kind not in self._active_kinds:
            return
        self._logged_kinds.add(stuck_kind)

        combat_log = list(server_report.get("combat_log") or [])
        if not combat_log:
            combat_log = extract_talishar_chat_log_lines(obs_data.get("chatLog", ""))[-40:]

        combat_tracker: dict[str, Any] = {}
        getter = getattr(env, "get_combat_tracker_snapshot", None)
        if callable(getter):
            try:
                combat_tracker = getter(tail_events=10, tail_log_lines=20)
            except Exception:
                combat_tracker = {}

        details: dict[str, Any] = {
            "stuck_kind": stuck_kind,
            "episode": self._context.get("episode"),
            "mode": self._context.get("mode"),
            "step": self._step,
            "p1_deck": self._context.get("p1_deck"),
            "p2_deck": self._context.get("p2_deck"),
            "matchup": self._context.get("matchup"),
            "game_format": self._context.get("game_format"),
            "shard": self._context.get("shard"),
            "server_report": server_report,
            "combat_log": combat_log,
            "step_info": {
                "repeat_streak": step_info.get("repeat_streak"),
                "loop_guard_forced_pass": step_info.get("loop_guard_forced_pass"),
                "loop_guard_reason": step_info.get("loop_guard_reason"),
                "turn_steps": step_info.get("turn_steps"),
                "decision_loop_streak": step_info.get("decision_loop_streak"),
                "acting_player_id": obs_data.get("actingPlayerID"),
                "phase": obs_data.get("turnPhase"),
            },
            "recent_actions": list(self._recent_actions),
            "legal_actions": _compact_legal_actions(legal_actions),
        }
        if stuck_kind == "pass_loop":
            details["pass_loop_streak"] = self._pass_loop_streak
            details["progress_signature"] = self._last_progress_signature
        if stuck_kind == "decision_loop":
            details["decision_loop_streak"] = self._decision_loop_streak
            details["decision_fingerprint"] = self._last_decision_fingerprint
        if combat_tracker:
            details["combat_tracker"] = combat_tracker
        if extra:
            details.update(extra)

        log_event("anti_stuck", f"{stuck_kind} detected", **details)

    def _flush_pending_loop_logs(
        self,
        env: Any,
        *,
        obs_data: dict[str, Any],
        step_info: dict[str, Any],
        server_report: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> None:
        for kind in ("pass_loop", "decision_loop"):
            if kind in self._active_kinds:
                self._maybe_log(
                    kind,
                    env=env,
                    obs_data=obs_data,
                    step_info=step_info,
                    server_report=server_report,
                    legal_actions=legal_actions,
                )


def monitor_for_episode(
    *,
    config: AntiStuckConfig,
    episode: int,
    mode: str,
    p1_deck: str,
    p2_deck: str,
    base_url: str,
    game_format: str = "",
    matchup: str = "",
) -> AntiStuckMonitor:
    monitor = AntiStuckMonitor(config=config)
    monitor.begin_episode(
        episode=episode,
        mode=mode,
        p1_deck=p1_deck,
        p2_deck=p2_deck,
        base_url=base_url,
        shard=shard_label(base_url),
        game_format=game_format,
        matchup=matchup,
    )
    return monitor
