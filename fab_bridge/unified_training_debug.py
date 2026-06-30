"""Structured debug logging for unified random matchup training runs."""

from __future__ import annotations

import json
import os
import re
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

_LOCK = threading.Lock()
_ENABLED = False
_RUN_DIR: Optional[Path] = None
_LOG_PATH: Optional[Path] = None

_HOST_PORT_RE = re.compile(r"host='([^']+)'.*port=(\d+)")


def is_enabled() -> bool:
    if _ENABLED:
        return True
    token = os.environ.get("FAB_UNIFIED_DEBUG", "").strip().lower()
    return token in {"1", "true", "yes", "on"}


def configure(*, run_dir: Path | str, enabled: bool) -> Path | None:
    """Enable debug logging for a unified run directory."""
    global _ENABLED, _RUN_DIR, _LOG_PATH  # noqa: PLW0603
    env_on = os.environ.get("FAB_UNIFIED_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    with _LOCK:
        _ENABLED = bool(enabled) or env_on
        _RUN_DIR = Path(run_dir).expanduser().resolve()
        _LOG_PATH = _RUN_DIR / "unified_training_debug.jsonl" if _ENABLED else None
        if _ENABLED and _LOG_PATH is not None:
            _RUN_DIR.mkdir(parents=True, exist_ok=True)
            _write_unlocked(
                "debug",
                "Unified training debug logging enabled",
                log_path=str(_LOG_PATH),
            )
        return _LOG_PATH


def log_path() -> Optional[Path]:
    return _LOG_PATH


def _write_unlocked(category: str, message: str, **details: Any) -> None:
    """Append one JSONL record. Caller must hold ``_LOCK``."""
    if not _ENABLED or _LOG_PATH is None:
        return
    log_path = _LOG_PATH
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "message": message,
    }
    if details:
        record["details"] = details
    line = json.dumps(record, default=str) + "\n"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)
    print(f"  [debug:{category}] {message}", flush=True)


def log_event(category: str, message: str, **details: Any) -> None:
    if not is_enabled():
        return
    with _LOCK:
        _write_unlocked(category, message, **details)


def log_exception(
    category: str,
    message: str,
    exc: BaseException,
    **details: Any,
) -> None:
    if not is_enabled():
        return
    payload = {
        **details,
        "exc_type": type(exc).__name__,
        "exc_repr": repr(exc),
        "traceback": traceback.format_exc(),
        **connection_details_from_exception(exc),
    }
    with _LOCK:
        _write_unlocked(category, message, **payload)


def connection_details_from_exception(exc: BaseException) -> dict[str, Any]:
    """Best-effort host/port/url extraction from HTTP client exceptions."""
    out: dict[str, Any] = {}
    text = repr(exc)
    match = _HOST_PORT_RE.search(text)
    if match:
        out["host"] = match.group(1)
        out["port"] = int(match.group(2))
    for attr in ("request", "url"):
        value = getattr(exc, attr, None)
        if value is not None:
            out[attr] = str(value)
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        nested = connection_details_from_exception(cause)
        for key, value in nested.items():
            out.setdefault(key, value)
    return out


def shard_label(base_url: str) -> str:
    parts = urlsplit(str(base_url).strip())
    host = parts.hostname or "localhost"
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return f"{host}:{port}"


def audit_matchup_decks(
    matchup: Any,
    *,
    game_format: str,
    matchup_dir: Path | None = None,
    assets_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Validate deck assets for a matchup; log and return issue records."""
    if not is_enabled():
        return []

    from flesh_and_blood_rlbridge.deck_context import _read_asset_deck  # noqa: PLC0415
    from flesh_and_blood_rlbridge.talishar_deck_assets import (  # noqa: PLC0415
        load_guide_sideboard_record,
        resolve_matchup_equipment_header,
        resolve_talishar_deck_stem,
    )

    if assets_path is None:
        from fab_bridge.paths import repo_root  # noqa: PLC0415

        assets_path = repo_root() / "Talishar" / "Assets"
    assets = Path(assets_path)
    guide = load_guide_sideboard_record(matchup_dir) if matchup_dir else {}
    issues: list[dict[str, Any]] = []

    for role, deck_stem, hero in (
        ("p1", getattr(matchup, "p1_deck", ""), getattr(matchup, "p1_hero", "")),
        ("p2", getattr(matchup, "p2_deck", ""), getattr(matchup, "p2_hero", "")),
    ):
        stem = str(deck_stem or "").strip()
        if not stem:
            issue = {
                "role": role,
                "matchup": getattr(matchup, "name", ""),
                "problem": "missing_deck_stem",
            }
            issues.append(issue)
            log_event("deck_validation", f"{role} deck stem missing", **issue)
            continue
        try:
            resolved = resolve_talishar_deck_stem(assets, stem)
            hero_id, counts = _read_asset_deck(assets, resolved)
        except Exception as exc:  # noqa: BLE001
            issue = {
                "role": role,
                "matchup": getattr(matchup, "name", ""),
                "deck_stem": stem,
                "problem": "deck_read_failed",
            }
            issues.append(issue)
            log_exception(
                "deck_validation",
                f"Could not read {role} deck asset",
                exc,
                **issue,
            )
            continue

        card_total = sum(int(v) for v in counts.values() if int(v) > 0)
        unique_cards = len([k for k, v in counts.items() if int(v) > 0])
        if card_total < 40:
            issue = {
                "role": role,
                "matchup": getattr(matchup, "name", ""),
                "deck_stem": resolved,
                "problem": "deck_too_small",
                "card_total": card_total,
                "unique_cards": unique_cards,
            }
            issues.append(issue)
            log_event(
                "deck_validation",
                f"{role} deck has only {card_total} cards",
                **issue,
            )

        header = resolve_matchup_equipment_header(
            role=role,
            hero_id=hero or hero_id,
            deck_stem=resolved,
            assets_dir=assets,
            fallback=hero_id,
            guide_sideboard=guide,
        )
        tokens = [t for t in str(header or "").split() if t.strip()]
        if len(tokens) < 2:
            issue = {
                "role": role,
                "matchup": getattr(matchup, "name", ""),
                "deck_stem": resolved,
                "problem": "equipment_header_incomplete",
                "equipment_header": header,
                "token_count": len(tokens),
            }
            issues.append(issue)
            log_event(
                "equipment",
                f"{role} equipment header looks incomplete",
                **issue,
            )

        log_event(
            "deck_validation",
            f"{role} deck audit ok",
            matchup=getattr(matchup, "name", ""),
            role=role,
            deck_stem=resolved,
            hero_id=hero_id,
            card_total=card_total,
            unique_cards=unique_cards,
            equipment_header=header,
            game_format=game_format,
        )

    return issues


def log_render_observation(
    obs: Any,
    *,
    message: str,
    **details: Any,
) -> None:
    if not is_enabled():
        return
    obs_data: dict[str, Any] = {}
    if isinstance(obs, str):
        try:
            parsed = json.loads(obs)
            if isinstance(parsed, dict):
                obs_data = parsed
        except json.JSONDecodeError:
            obs_data = {"raw_preview": obs[:500]}
    elif isinstance(obs, dict):
        obs_data = obs

    turn_no = int(obs_data.get("turnNo", 0) or 0)
    acting = obs_data.get("actingPlayerID")
    p1_hp = obs_data.get("p1Health", obs_data.get("playerHealth"))
    p2_hp = obs_data.get("p2Health", obs_data.get("opponentHealth"))
    suspicious = turn_no <= 0 or p1_hp in (0, "0", None) or p2_hp in (0, "0", None)
    payload = {
        **details,
        "turn_no": turn_no,
        "acting_player_id": acting,
        "p1_hp": p1_hp,
        "p2_hp": p2_hp,
        "legal_actions": len(obs_data.get("legalActions", []) or []),
        "suspicious_init": suspicious,
    }
    category = "render_init" if suspicious else "render"
    log_event(category, message, **payload)


def read_debug_from_manifest(run_dir: Path) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return False
    return bool(data.get("debug_training"))


def log_shard_pool(
    *,
    urls_before: list[str],
    urls_after: list[str],
    failed_health: list[str],
) -> None:
    if not is_enabled():
        return
    dropped = [u for u in urls_before if u not in urls_after]
    if dropped or failed_health:
        log_event(
            "connection",
            "Talishar shard pool health filter",
            urls_before=urls_before,
            urls_after=urls_after,
            dropped_shards=dropped,
            failed_health=failed_health,
            shard_labels_before=[shard_label(u) for u in urls_before],
            shard_labels_after=[shard_label(u) for u in urls_after],
        )
