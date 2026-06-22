"""Shared fixed-width observation features for C++ and Talishar training.

The schema mirrors the generated C++ ``GameState::fast_observation_vector``:
16 scalar features followed by 8 hand slots of ``cost, pitch, power, defense``.
Keep this file and ``scripts/cpp/generate_cpp_engine.py`` in sync.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

FAST_HAND_SLOTS = 8
FAST_OBS_DIM = 16 + FAST_HAND_SLOTS * 4
FAST_ACTION_CAPACITY = 32

_PHASE_TO_VALUE = {
    "start": 0,
    "startturn": 0,
    "m": 1,
    "main": 1,
    "p": 2,
    "pitch": 2,
    "a": 3,
    "attack": 3,
    "b": 4,
    "block": 4,
    "d": 4,
    "defense": 4,
    "damage": 5,
    "endphase": 6,
    "end": 6,
    "over": 7,
}


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _scaled(value: Any, denom: float) -> float:
    return float(_to_int(value)) / denom


def _card_key(card: dict[str, Any]) -> str:
    return str(
        card.get("cardNumber")
        or card.get("cardID")
        or card.get("card_id")
        or card.get("id")
        or ""
    )


def _load_card_stats() -> dict[str, tuple[int, int, int, int]]:
    path = Path(__file__).parent / "card_db" / "cards.json"
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    stats: dict[str, tuple[int, int, int, int]] = {}
    for rec in records if isinstance(records, list) else []:
        if not isinstance(rec, dict):
            continue
        cid = str(rec.get("id", "") or "")
        if not cid:
            continue
        stats[cid] = (
            _to_int(rec.get("cost")),
            _to_int(rec.get("pitch")),
            _to_int(rec.get("power")),
            _to_int(rec.get("defense")),
        )
    return stats


_CARD_STATS = _load_card_stats()


def _card_stat(card: dict[str, Any], key: str, default: int = 0) -> int:
    aliases = {
        "cost": ("cost", "resource"),
        "pitch": ("pitch",),
        "power": ("power", "attack"),
        "defense": ("defense", "block"),
    }
    for alias in aliases.get(key, (key,)):
        if card.get(alias) not in (None, ""):
            return _to_int(card.get(alias), default)
    stats = _CARD_STATS.get(_card_key(card))
    if stats is None:
        return default
    index = {"cost": 0, "pitch": 1, "power": 2, "defense": 3}[key]
    return stats[index]


def _phase_value(phase: Any) -> int:
    if isinstance(phase, dict):
        phase = phase.get("turnPhase", "")
    token = str(phase or "").strip().lower()
    return _PHASE_TO_VALUE.get(token, 1)


def fast_observation_vector(
    state: dict[str, Any],
    legal_actions: list[Any] | None,
    *,
    acting_player_id: int | None = None,
    p1_health: int | None = None,
    p2_health: int | None = None,
    winner: int = -1,
    game_over: bool = False,
    consecutive_passes: int = 0,
) -> np.ndarray:
    """Return the canonical 48-float fast observation vector."""
    acting = int(acting_player_id or state.get("actingPlayerID", 1) or 1)
    player_hp = _to_int(state.get("playerHealth"))
    opponent_hp = _to_int(state.get("opponentHealth"))
    if p1_health is None or p2_health is None:
        if acting == 1:
            p1_health = player_hp
            p2_health = opponent_hp
        else:
            p1_health = opponent_hp
            p2_health = player_hp

    hand = state.get("playerHand", [])
    if not isinstance(hand, list):
        hand = []
    legal_count = len(legal_actions or [])

    out: list[float] = [
        float(acting),
        _scaled(player_hp, 40.0),
        _scaled(opponent_hp, 40.0),
        _scaled(p1_health, 40.0),
        _scaled(p2_health, 40.0),
        _scaled(state.get("turnNo", state.get("turn_no", 0)), 100.0),
        float(_phase_value(state.get("turnPhase", state.get("turn_phase", "")))) / 10.0,
        _scaled(state.get("playerHandSize", len(hand)), 10.0),
        _scaled(state.get("opponentHandSize", state.get("opponent_hand_size", 0)), 10.0),
        _scaled(state.get("playerDeckCount", state.get("player_deck_count", 0)), 80.0),
        _scaled(state.get("opponentDeckCount", state.get("opponent_deck_count", 0)), 80.0),
        _scaled(state.get("playerPitchCount", state.get("player_pitch_count", 0)), 20.0),
        float(legal_count) / float(FAST_ACTION_CAPACITY),
        float(consecutive_passes) / 20.0,
        1.0 if game_over else 0.0,
        float(int(winner) + 1) / 3.0,
    ]

    for slot in range(FAST_HAND_SLOTS):
        card = hand[slot] if slot < len(hand) and isinstance(hand[slot], dict) else None
        if card is None:
            out.extend((0.0, 0.0, 0.0, 0.0))
            continue
        out.extend(
            (
                _card_stat(card, "cost") / 10.0,
                _card_stat(card, "pitch") / 4.0,
                _card_stat(card, "power") / 12.0,
                _card_stat(card, "defense") / 5.0,
            )
        )

    return np.asarray(out, dtype=np.float64)


def fast_observation_payload(vec: np.ndarray) -> list[float]:
    """JSON-serialisable representation of a fast observation vector."""
    return [float(x) for x in np.asarray(vec, dtype=np.float64).reshape(-1)]
