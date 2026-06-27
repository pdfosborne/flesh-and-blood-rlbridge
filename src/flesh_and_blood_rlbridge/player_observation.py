"""Player-fair fixed-width observation vector for Talishar and C++ training.

Layout is the single source of truth; ``scripts/cpp/generate_cpp_engine.py`` mirrors
these constants when emitting ``GameState::player_observation_vector``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .card_conditionals import evaluate_conditional_active
from .card_vocab import (
    card_index_normalized,
    format_index_normalized,
    hero_index_normalized,
    vocab_size,
)
from .deck_context import EpisodeContext, hero_from_equipment

PLAYER_OBS_SCHEMA_VERSION = 2
ACTION_CAPACITY = 128

DECK_SLOTS = 80
HAND_SLOTS = 8
HAND_SLOT_DIM = 7  # card_idx, cost, pitch, power, defense, playable, conditional_active

ZONE_SLOT_DIM = 5  # card_idx, counters, tapped, face_down, conditional_active

# (zone_name, max_slots) per player perspective
ZONE_SPECS: tuple[tuple[str, int], ...] = (
    ("equipment", 4),
    ("arsenal", 1),
    ("discard", 8),
    ("pitch", 12),
    ("banish", 6),
    ("auras", 4),
    ("allies", 4),
    ("items", 4),
)

COMBAT_SCALAR_COUNT = 6
COMBAT_CHAIN_SLOTS = 8
COMBAT_CHAIN_SLOT_DIM = 3  # card_idx, controller, conditional_active

SCALAR_COUNT = 26

CONTEXT_DIM = 4 + DECK_SLOTS

_ZONES_PER_PLAYER = sum(n for _, n in ZONE_SPECS) * ZONE_SLOT_DIM
ZONE_SLOTS_PER_PLAYER = sum(n for _, n in ZONE_SPECS)

# Slice offsets for tokenizing the flat observation vector (attention policy).
HERO_SELF_OFF = 0
HERO_OPP_OFF = 1
FORMAT_OFF = 2
FIRST_PLAYER_OFF = 3
DECK_OFF = 4
DECK_END = CONTEXT_DIM
SCALAR_OFF = CONTEXT_DIM
SCALAR_END = CONTEXT_DIM + SCALAR_COUNT
HAND_OFF = SCALAR_END
HAND_END = HAND_OFF + HAND_SLOTS * HAND_SLOT_DIM
ZONE_OFF = HAND_END
ZONE_END = ZONE_OFF + 2 * _ZONES_PER_PLAYER
COMBAT_SCALAR_OFF = ZONE_END
COMBAT_SCALAR_END = COMBAT_SCALAR_OFF + COMBAT_SCALAR_COUNT
COMBAT_CHAIN_OFF = COMBAT_SCALAR_END
COMBAT_CHAIN_END = COMBAT_CHAIN_OFF + COMBAT_CHAIN_SLOTS * COMBAT_CHAIN_SLOT_DIM

PLAYER_OBS_DIM = (
    CONTEXT_DIM
    + SCALAR_COUNT
    + HAND_SLOTS * HAND_SLOT_DIM
    + 2 * _ZONES_PER_PLAYER
    + COMBAT_SCALAR_COUNT
    + COMBAT_CHAIN_SLOTS * COMBAT_CHAIN_SLOT_DIM
)

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

_POPUP_NONE = 0
_POPUP_PITCH = 1
_POPUP_MULTICHOOSE = 2
_POPUP_YESNO = 3
_POPUP_BUTTON = 4
_POPUP_OTHER = 5

_PITCH_RESOURCES_RE = re.compile(r"\((\d+)\s+of\s+(\d+)\)", re.I)


@dataclass(frozen=True)
class NormalizedCard:
    card_id: str = ""
    cost: int = 0
    pitch: int = 0
    power: int = 0
    defense: int = 0
    counters: int = 0
    tapped: bool = False
    face_down: bool = False
    playable: bool = False
    controller: int = 0
    card_visible: bool = True


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
    from pathlib import Path
    import json

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


def _is_face_down(card: dict[str, Any]) -> bool:
    cid = _card_key(card).lower()
    if "cardback" in cid or cid in {"", "wtr000"}:
        return True
    facing = str(card.get("facing", "") or "").upper()
    if facing == "DOWN":
        return True
    overlay = str(card.get("overlay", "") or "").lower()
    if overlay in {"disabled", "hidden"}:
        return True
    mod = str(card.get("mod", "") or card.get("modifier", "") or "").lower()
    return mod.startswith("down") or mod == "facedown"


def _reveal_card_id(*, side: str, zone: str, card: dict[str, Any]) -> bool:
    if side == "self" and zone == "arsenal":
        cid = _card_key(card).lower()
        return bool(cid and "cardback" not in cid and cid not in {"", "wtr000"})
    return not _is_face_down(card)


def normalize_talishar_card(
    card: Any,
    *,
    playable: bool = False,
    side: str = "self",
    zone: str = "hand",
) -> NormalizedCard:
    if not isinstance(card, dict):
        return NormalizedCard()
    face_down = _is_face_down(card)
    visible = _reveal_card_id(side=side, zone=zone, card=card)
    cid = _card_key(card) if visible else ""
    counters = _to_int(card.get("counters") or card.get("counter") or 0)
    if counters <= 0 and isinstance(card.get("countersMap"), dict):
        counters = sum(_to_int(v) for v in card["countersMap"].values())
    tapped = bool(card.get("tapped") or card.get("isFrozen"))
    return NormalizedCard(
        card_id=cid,
        cost=_card_stat(card, "cost"),
        pitch=_card_stat(card, "pitch"),
        power=_card_stat(card, "power"),
        defense=_card_stat(card, "defense"),
        counters=counters,
        tapped=tapped,
        face_down=face_down,
        playable=playable or bool(_to_int(card.get("action", 0))),
        controller=_to_int(card.get("controller"), 0),
        card_visible=visible,
    )


def normalize_zone_cards(
    cards: Any,
    *,
    playable_ids: Optional[set[str]] = None,
    side: str = "self",
    zone: str = "hand",
) -> list[NormalizedCard]:
    if not isinstance(cards, list):
        return []
    out: list[NormalizedCard] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        cid = _card_key(card)
        playable = bool(playable_ids and cid and cid in playable_ids)
        out.append(
            normalize_talishar_card(card, playable=playable, side=side, zone=zone)
        )
    return out


def _popup_type(state: dict[str, Any], phase_token: str) -> int:
    popup = state.get("playerInputPopUp", {})
    if isinstance(popup, dict) and popup.get("active"):
        title = ""
        inner = popup.get("popup", {})
        if isinstance(inner, dict):
            title = str(inner.get("title", "") or "").lower()
        phase = phase_token.lower()
        if phase in {"p", "pitch"} or "pitch" in title:
            return _POPUP_PITCH
        if "multichoose" in phase or "choose" in title:
            return _POPUP_MULTICHOOSE
        if phase == "yesno" or "yes" in title:
            return _POPUP_YESNO
        if popup.get("buttons"):
            return _POPUP_BUTTON
        return _POPUP_OTHER
    if phase_token.lower() in {"p", "pitch"}:
        return _POPUP_PITCH
    return _POPUP_NONE


def _parse_pitch_resources(state: dict[str, Any], phase_token: str) -> tuple[int, int]:
    available = _to_int(state.get("playerPitchCount", state.get("player_pitch_count", 0)))
    required = available
    if phase_token.lower() not in {"p", "pitch"}:
        return available, required
    prompt = state.get("playerPrompt", {})
    if isinstance(prompt, dict):
        match = _PITCH_RESOURCES_RE.search(str(prompt.get("helpText", "") or ""))
        if match:
            return _to_int(match.group(1)), _to_int(match.group(2))
    return available, required


def _conditional_active(
    card: NormalizedCard,
    state: dict[str, Any],
    *,
    zone: str,
    side: str,
    phase: str,
) -> float:
    return evaluate_conditional_active(
        card.card_id,
        state,
        zone=zone,
        side=side,
        phase=phase,
        card_visible=card.card_visible and bool(card.card_id),
    )


def _encode_card_slot(
    card: NormalizedCard,
    state: dict[str, Any],
    *,
    zone: str,
    side: str,
    phase: str,
) -> list[float]:
    cond = _conditional_active(card, state, zone=zone, side=side, phase=phase)
    if not card.card_visible or not card.card_id:
        return [
            0.0,
            _scaled(card.counters, 10.0),
            1.0 if card.tapped else 0.0,
            1.0 if card.face_down else 0.0,
            cond if card.card_visible else 0.0,
        ]
    return [
        card_index_normalized(card.card_id),
        _scaled(card.counters, 10.0),
        1.0 if card.tapped else 0.0,
        1.0 if card.face_down else 0.0,
        cond,
    ]


def _encode_hand_slot(
    card: Optional[NormalizedCard],
    state: dict[str, Any],
    *,
    phase: str,
) -> list[float]:
    if card is None:
        return [0.0] * HAND_SLOT_DIM
    cond = _conditional_active(card, state, zone="hand", side="self", phase=phase)
    return [
        card_index_normalized(card.card_id) if card.card_id else 0.0,
        _scaled(card.cost, 10.0),
        _scaled(card.pitch, 4.0),
        _scaled(card.power, 12.0),
        _scaled(card.defense, 5.0),
        1.0 if card.playable else 0.0,
        cond,
    ]


def _encode_zone_block(
    cards: list[NormalizedCard],
    max_slots: int,
    state: dict[str, Any],
    *,
    zone: str,
    side: str,
    phase: str,
) -> list[float]:
    out: list[float] = []
    for slot in range(max_slots):
        card = cards[slot] if slot < len(cards) else None
        if card is None:
            out.extend([0.0] * ZONE_SLOT_DIM)
        else:
            out.extend(_encode_card_slot(card, state, zone=zone, side=side, phase=phase))
    return out


def _encode_context(episode: Optional[EpisodeContext]) -> list[float]:
    if episode is None:
        return [0.0] * CONTEXT_DIM
    self_hero = episode.self_hero_id
    opp_hero = episode.opp_hero_id
    out: list[float] = [
        hero_index_normalized(self_hero),
        hero_index_normalized(opp_hero),
        format_index_normalized(episode.format),
        1.0 if int(episode.first_player) == 1 else 2.0,
    ]
    deck_indices = episode.self_deck_indices()
    for slot in range(DECK_SLOTS):
        if slot < len(deck_indices):
            out.append(float(deck_indices[slot]) / float(max(vocab_size(), 1)))
        else:
            out.append(0.0)
    return out


def _zone_state_keys(side: str, zone: str) -> str:
    prefix = "player" if side == "self" else "opponent"
    mapping = {
        "equipment": f"{prefix}Equipment",
        "arsenal": f"{prefix}Arse",
        "discard": f"{prefix}Discard",
        "pitch": f"{prefix}Pitch",
        "banish": f"{prefix}Banish",
        "auras": f"{prefix}Auras",
        "allies": f"{prefix}Allies",
        "items": f"{prefix}Items",
    }
    return mapping[zone]


def _encode_combat(state: dict[str, Any], *, phase: str) -> list[float]:
    out: list[float] = [0.0] * (COMBAT_SCALAR_COUNT + COMBAT_CHAIN_SLOTS * COMBAT_CHAIN_SLOT_DIM)
    link = state.get("activeChainLink", {})
    if not isinstance(link, dict):
        return out
    out[0] = _scaled(link.get("totalPower", 0), 20.0)
    out[1] = _scaled(link.get("totalDefense", 0), 20.0)
    out[2] = 1.0 if link.get("goAgain") else 0.0
    out[3] = 1.0 if link.get("piercing") else 0.0
    out[4] = _scaled(link.get("numRequiredEquipBlock", 0), 5.0)
    out[5] = _scaled(link.get("damagePrevention", 0), 20.0)

    reactions = link.get("reactions", [])
    if not isinstance(reactions, list):
        reactions = []
    attack = link.get("attackingCard")
    chain_cards: list[Any] = []
    if isinstance(attack, dict):
        chain_cards.append(attack)
    chain_cards.extend(reactions)
    base = COMBAT_SCALAR_COUNT
    for slot in range(COMBAT_CHAIN_SLOTS):
        offset = base + slot * COMBAT_CHAIN_SLOT_DIM
        if slot >= len(chain_cards):
            continue
        norm = normalize_talishar_card(chain_cards[slot], zone="combat", side="self")
        out[offset] = card_index_normalized(norm.card_id) if norm.card_id else 0.0
        ctrl = norm.controller or _to_int(chain_cards[slot].get("controller"), 0)
        out[offset + 1] = float(ctrl) / 2.0 if ctrl else 0.0
        out[offset + 2] = _conditional_active(
            norm, state, zone="combat", side="self", phase=phase
        )
    return out


def player_observation_vector(
    state: dict[str, Any],
    legal_actions: list[Any] | None,
    *,
    episode_context: Optional[EpisodeContext] = None,
    acting_player_id: int | None = None,
    p1_health: int | None = None,
    p2_health: int | None = None,
    winner: int = -1,
    game_over: bool = False,
    consecutive_passes: int = 0,
    raw_talishar_state: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Return the canonical player-fair observation vector."""
    raw = raw_talishar_state if isinstance(raw_talishar_state, dict) else state
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

    phase_token = state.get("turnPhase", state.get("turn_phase", ""))
    if isinstance(phase_token, dict):
        phase_token = phase_token.get("turnPhase", "")
    phase_str = str(phase_token or "")

    hand_raw = state.get("playerHand", [])
    if not isinstance(hand_raw, list):
        hand_raw = []
    playable_hand: set[str] = set()
    for card in hand_raw:
        if isinstance(card, dict) and _to_int(card.get("action", 0)) == 27:
            cid = _card_key(card)
            if cid:
                playable_hand.add(cid)
    hand = normalize_zone_cards(
        hand_raw, playable_ids=playable_hand, side="self", zone="hand"
    )

    resources_avail, resources_required = _parse_pitch_resources(raw, phase_str)
    legal_count = len(legal_actions or [])

    if episode_context is not None and not episode_context.self_hero_id:
        episode_context = EpisodeContext(
            self_hero_id=hero_from_equipment(raw.get("playerEquipment", [])),
            opp_hero_id=hero_from_equipment(raw.get("opponentEquipment", []))
            or episode_context.opp_hero_id,
            format=episode_context.format,
            self_deck_counts=dict(episode_context.self_deck_counts),
            first_player=_to_int(raw.get("firstPlayer"), episode_context.first_player),
        )
    elif episode_context is None:
        episode_context = EpisodeContext(
            self_hero_id=hero_from_equipment(raw.get("playerEquipment", [])),
            opp_hero_id=hero_from_equipment(raw.get("opponentEquipment", [])),
            format=str(state.get("game_format", "silver_age") or "silver_age"),
            first_player=_to_int(raw.get("firstPlayer"), 1),
        )

    scalars: list[float] = [
        float(acting) / 2.0,
        _scaled(player_hp, 40.0),
        _scaled(opponent_hp, 40.0),
        _scaled(p1_health, 40.0),
        _scaled(p2_health, 40.0),
        _scaled(state.get("turnNo", state.get("turn_no", 0)), 100.0),
        float(_phase_value(phase_token)) / 10.0,
        _scaled(state.get("playerHandSize", len(hand)), 10.0),
        _scaled(state.get("opponentHandSize", state.get("opponent_hand_size", 0)), 10.0),
        _scaled(state.get("playerDeckCount", state.get("player_deck_count", 0)), 80.0),
        _scaled(state.get("opponentDeckCount", state.get("opponent_deck_count", 0)), 80.0),
        _scaled(state.get("playerPitchCount", state.get("player_pitch_count", 0)), 20.0),
        _scaled(raw.get("opponentPitchCount", state.get("opponent_pitch_count", 0)), 20.0),
        _scaled(resources_avail, 20.0),
        _scaled(resources_required, 20.0),
        _scaled(raw.get("playerAP", 0), 10.0),
        _scaled(raw.get("opponentAP", 0), 10.0),
        float(legal_count) / float(ACTION_CAPACITY),
        float(consecutive_passes) / 20.0,
        1.0 if game_over else 0.0,
        float(int(winner) + 1) / 3.0,
        1.0 if raw.get("canPassPhase", state.get("canPassPhase", True)) else 0.0,
        float(_popup_type(raw, phase_str)) / 10.0,
        1.0 if raw.get("amIActivePlayer", state.get("amIActivePlayer", False)) else 0.0,
        1.0 if state.get("havePriority", True) else 0.0,
        1.0 if raw.get("turnPlayer", acting) == acting else 0.0,
    ]
    if len(scalars) != SCALAR_COUNT:
        raise RuntimeError(f"scalar block size mismatch: {len(scalars)} != {SCALAR_COUNT}")

    out: list[float] = []
    out.extend(_encode_context(episode_context))
    out.extend(scalars)
    for slot in range(HAND_SLOTS):
        card = hand[slot] if slot < len(hand) else None
        out.extend(_encode_hand_slot(card, raw, phase=phase_str))

    for side in ("self", "opp"):
        for zone_name, max_slots in ZONE_SPECS:
            key = _zone_state_keys(side, zone_name)
            cards = normalize_zone_cards(raw.get(key, []), side=side, zone=zone_name)
            out.extend(
                _encode_zone_block(
                    cards, max_slots, raw, zone=zone_name, side=side, phase=phase_str
                )
            )

    out.extend(_encode_combat(raw, phase=phase_str))
    if len(out) != PLAYER_OBS_DIM:
        raise RuntimeError(f"observation dim mismatch: {len(out)} != {PLAYER_OBS_DIM}")
    return np.asarray(out, dtype=np.float64)


def player_observation_payload(vec: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(vec, dtype=np.float64).reshape(-1)]


# Backward-compatible aliases during migration (removed once all imports updated)
FAST_HAND_SLOTS = HAND_SLOTS
FAST_OBS_DIM = PLAYER_OBS_DIM
FAST_ACTION_CAPACITY = ACTION_CAPACITY


def observation_payload(vec: np.ndarray) -> list[float]:
    return player_observation_payload(vec)
