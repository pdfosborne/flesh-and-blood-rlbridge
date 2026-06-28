"""Full game-state parity between Talishar HTTP and the C++ engine.

Compares absolute P1/P2 snapshots (HP, phase, zones, combat chain, resources)
in addition to the agent-facing observation contract.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .obs_alignment import (
    align_observation_for_cpp_training,
    observation_vectors_aligned,
)
from .player_observation import PLAYER_OBS_DIM

# Taxonomy codes for generator feedback.
TAXONOMY_PHASE = "phase_transition"
TAXONOMY_LEGAL = "legal_actions"
TAXONOMY_ZONE_PREFIX = "zone_"
TAXONOMY_COMBAT = "combat_chain"
TAXONOMY_CARD_EFFECT = "card_effect"
TAXONOMY_DECK_INIT = "deck_init"
TAXONOMY_OBS_CONTRACT = "obs_contract"
TAXONOMY_CPP_UNIMPLEMENTED = "cpp_zone_unimplemented"
TAXONOMY_REWARD = "reward"
TAXONOMY_TERMINATION = "termination"

OBSERVATION_KEYS = (
    "actingPlayerID",
    "selfPlay",
    "playerHealth",
    "opponentHealth",
    "turnNo",
    "turnPhase",
    "havePriority",
    "playerHandSize",
    "opponentHandSize",
    "playerDeckCount",
    "opponentDeckCount",
    "playerPitchCount",
    "playerHand",
    "legalActions",
)
OBSERVATION_SCALAR_KEYS = OBSERVATION_KEYS[:-2]
OBSERVATION_NUMERIC_KEYS = {
    "actingPlayerID",
    "playerHealth",
    "opponentHealth",
    "turnNo",
    "playerHandSize",
    "opponentHandSize",
    "playerDeckCount",
    "opponentDeckCount",
    "playerPitchCount",
}
OBS_LEGAL_ACTION_KEYS = ("index", "label", "zone")
INFO_LEGAL_ACTION_KEYS = ("action_code", "button_input", "card_id", "zone", "label")
INFO_CONTRACT_KEYS = (
    "legal_actions",
    "player_hp",
    "opponent_hp",
    "acting_player_id",
    "self_play",
    "turn",
    "repeat_streak",
    "repeat_penalty",
)
REWARD_TOLERANCE = 0.001
OBS_VEC_TOLERANCE = 0.05

# Talishar placeholder IDs that must not be synced into the C++ deck model.
HIDDEN_CARD_IDS = frozenset(
    {
        "CardBack",
        "cardback",
        "NOCARD",
        "nocard",
        "blank",
        "Blank",
    }
)


def is_syncable_card_id(card_id: Any) -> bool:
    """True when *card_id* is a real deck card (not a Talishar placeholder)."""
    text = str(card_id or "").strip()
    if not text:
        return False
    return text not in HIDDEN_CARD_IDS and not text.lower().startswith("cardback")

# Zones compared in Tier B (absolute P1/P2 keys in normalized state).
ZONE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("p1_hand", "playerHand", "hand"),
    ("p2_hand", "opponentHand", "hand"),
    ("p1_deck", "playerDeck", "deck"),
    ("p2_deck", "opponentDeck", "deck"),
    ("p1_discard", "playerDiscard", "discard"),
    ("p2_discard", "opponentDiscard", "discard"),
    ("p1_equipment", "playerEquipment", "equipment"),
    ("p2_equipment", "opponentEquipment", "equipment"),
    ("p1_arsenal", "playerArse", "arsenal"),
    ("p2_arsenal", "opponentArse", "arsenal"),
    ("p1_pitch", "playerPitch", "pitch"),
    ("p2_pitch", "opponentPitch", "pitch"),
    ("p1_banish", "playerBanish", "banish"),
    ("p2_banish", "opponentBanish", "banish"),
)

CPP_IMPLEMENTED_ZONES = {
    "hand",
    "deck",
    "discard",
    "equipment",
    "arsenal",
    "pitch",
    "banish",
}

# Talishar phases not yet modeled 1:1 in the C++ stub.
PHASE_EQUIVALENTS: dict[str, frozenset[str]] = {
    "INSTANT": frozenset({"INSTANT", "P", "M", "B", "A", "D"}),
    "D": frozenset({"D", "B", "INSTANT"}),
    "A": frozenset({"A", "B", "INSTANT"}),
}


def _talishar_action_points(phase: str, resources: int) -> int:
    """Map internal pitch pool to Talishar ``playerAP`` (only nonzero in main)."""
    if str(phase or "").upper() in {"M", "STARTTURN"}:
        return max(0, int(resources))
    return 0


def _phases_match(tal_phase: str, cpp_phase: str) -> bool:
    tal = str(tal_phase or "").upper()
    cpp = str(cpp_phase or "").upper()
    if tal == cpp:
        return True
    equivalents = PHASE_EQUIVALENTS.get(tal)
    return equivalents is not None and cpp in equivalents


@dataclass
class StateDiscrepancy:
    category: str
    taxonomy: str
    description: str
    talishar_value: Any
    cpp_value: Any
    card_id: str = ""
    zone: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "taxonomy": self.taxonomy,
            "description": self.description,
            "talishar_value": self.talishar_value,
            "cpp_value": self.cpp_value,
            "card_id": self.card_id,
            "zone": self.zone,
        }


@dataclass
class GameStateCompareResult:
    ok: bool
    discrepancies: list[StateDiscrepancy] = field(default_factory=list)

    def first_taxonomy(self) -> str:
        if not self.discrepancies:
            return ""
        return self.discrepancies[0].taxonomy


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _card_id_from_entry(card: Any) -> str:
    if not isinstance(card, dict):
        return ""
    for key in ("cardID", "cardNumber", "cardId", "card_id"):
        value = card.get(key)
        if value and is_syncable_card_id(value):
            return str(value)
    return ""


def _zone_card_ids(cards: Any) -> list[str]:
    if not isinstance(cards, list):
        return []
    return [_card_id_from_entry(card) for card in cards if _card_id_from_entry(card)]


def _phase_from_talishar_state(state: dict[str, Any]) -> str:
    turn_phase = state.get("turnPhase", "")
    if isinstance(turn_phase, dict):
        return str(turn_phase.get("turnPhase", "") or "").upper()
    return str(turn_phase or "").upper()


def _normalize_talishar_player_state(state: dict[str, Any], player_id: int) -> dict[str, Any]:
    """Convert a Talishar HTTP snapshot to absolute P1/P2 fields."""
    acting = _safe_int(state.get("actingPlayerID", player_id), player_id)
    if acting not in (1, 2):
        acting = player_id

    def p1_val(key: str, opp_key: str) -> Any:
        if acting == 1:
            return state.get(key)
        return state.get(opp_key)

    p1_health = _safe_int(p1_val("playerHealth", "opponentHealth"))
    p2_health = _safe_int(
        state.get("opponentHealth") if acting == 1 else state.get("playerHealth")
    )

    return {
        "acting_player_id": acting,
        "p1_health": p1_health,
        "p2_health": p2_health,
        "turn_no": _safe_int(state.get("turnNo", 0)),
        "phase": _phase_from_talishar_state(state),
        "p1_hand_size": len(_zone_card_ids(p1_val("playerHand", "opponentHand"))),
        "p2_hand_size": len(
            _zone_card_ids(
                state.get("opponentHand") if acting == 1 else state.get("playerHand")
            )
        ),
        "p1_deck_count": _safe_int(p1_val("playerDeckCount", "opponentDeckCount")),
        "p2_deck_count": _safe_int(
            state.get("opponentDeckCount") if acting == 1 else state.get("playerDeckCount")
        ),
        "p1_pitch_count": _safe_int(p1_val("playerPitchCount", "opponentPitchCount")),
        "p2_pitch_count": _safe_int(
            state.get("opponentPitchCount") if acting == 1 else state.get("playerPitchCount")
        ),
        "p1_resources": _safe_int(p1_val("playerAP", "opponentAP")),
        "p2_resources": _safe_int(
            state.get("opponentAP") if acting == 1 else state.get("playerAP")
        ),
        "priority_player": _safe_int(state.get("turnPlayer", acting), acting),
        "p1_hand": _zone_card_ids(p1_val("playerHand", "opponentHand")),
        "p2_hand": _zone_card_ids(
            state.get("opponentHand") if acting == 1 else state.get("playerHand")
        ),
        "p1_deck": _zone_card_ids(p1_val("playerDeck", "opponentDeck")),
        "p2_deck": _zone_card_ids(
            state.get("opponentDeck") if acting == 1 else state.get("playerDeck")
        ),
        "p1_discard": _zone_card_ids(p1_val("playerDiscard", "opponentDiscard")),
        "p2_discard": _zone_card_ids(
            state.get("opponentDiscard") if acting == 1 else state.get("playerDiscard")
        ),
        "p1_equipment": _zone_card_ids(p1_val("playerEquipment", "opponentEquipment")),
        "p2_equipment": _zone_card_ids(
            state.get("opponentEquipment") if acting == 1 else state.get("playerEquipment")
        ),
        "p1_arsenal": _zone_card_ids(p1_val("playerArse", "opponentArse")),
        "p2_arsenal": _zone_card_ids(
            state.get("opponentArse") if acting == 1 else state.get("opponentArse")
        ),
        "p1_pitch": _zone_card_ids(p1_val("playerPitch", "opponentPitch")),
        "p2_pitch": _zone_card_ids(
            state.get("opponentPitch") if acting == 1 else state.get("playerPitch")
        ),
        "p1_banish": _zone_card_ids(p1_val("playerBanish", "opponentBanish")),
        "p2_banish": _zone_card_ids(
            state.get("opponentBanish") if acting == 1 else state.get("playerBanish")
        ),
        "combat_chain": _extract_combat_chain(state),
        "pending_attack_power": _safe_int(state.get("pendingAttackPower", 0)),
        "pending_block_value": _safe_int(state.get("pendingBlockValue", 0)),
        "game_over": bool(state.get("winner", 0)),
        "winner": _safe_int(state.get("winner", -1), -1),
    }


def _extract_combat_chain(state: dict[str, Any]) -> list[dict[str, Any]]:
    chain = state.get("combatChain") or state.get("combat_chain") or []
    if not isinstance(chain, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in chain:
        if not isinstance(entry, dict):
            continue
        out.append(
            {
                "card_id": _card_id_from_entry(entry),
                "power": _safe_int(entry.get("power", entry.get("attack", 0))),
                "defense": _safe_int(entry.get("defense", 0)),
            }
        )
    return out


def _absolute_talishar_snapshot(
    p1_state: dict[str, Any],
    p2_state: dict[str, Any],
    *,
    acting_player_id: int,
) -> dict[str, Any]:
    """Build absolute P1/P2 fields from per-player Talishar HTTP snapshots."""
    p1_hand = _zone_card_ids(p1_state.get("playerHand", []))
    p2_hand = _zone_card_ids(p2_state.get("playerHand", []))
    p1_discard = _zone_card_ids(p1_state.get("playerDiscard", []))
    p2_discard = _zone_card_ids(p2_state.get("playerDiscard", []))
    p1_equipment = _zone_card_ids(p1_state.get("playerEquipment", []))
    p2_equipment = _zone_card_ids(p2_state.get("playerEquipment", []))
    p1_arsenal = _zone_card_ids(p1_state.get("playerArse", []))
    p2_arsenal = _zone_card_ids(p2_state.get("playerArse", []))
    p1_pitch = _zone_card_ids(p1_state.get("playerPitch", []))
    p2_pitch = _zone_card_ids(p2_state.get("playerPitch", []))

    phase = _phase_from_talishar_state(p1_state) or _phase_from_talishar_state(p2_state)
    turn_no = _safe_int(p1_state.get("turnNo", p2_state.get("turnNo", 0)))

    return {
        "acting_player_id": acting_player_id,
        "p1_health": _safe_int(p1_state.get("playerHealth", 0)),
        "p2_health": _safe_int(p2_state.get("playerHealth", 0)),
        "turn_no": turn_no,
        "phase": phase,
        "p1_hand_size": len(p1_hand),
        "p2_hand_size": len(p2_hand),
        "p1_deck_count": _safe_int(p1_state.get("playerDeckCount", 0)),
        "p2_deck_count": _safe_int(p2_state.get("playerDeckCount", 0)),
        "p1_pitch_count": _safe_int(p1_state.get("playerPitchCount", 0)),
        "p2_pitch_count": _safe_int(p2_state.get("playerPitchCount", 0)),
        "p1_resources": _safe_int(p1_state.get("playerAP", 0)),
        "p2_resources": _safe_int(p2_state.get("playerAP", 0)),
        "priority_player": _safe_int(
            p1_state.get("turnPlayer", p2_state.get("turnPlayer", acting_player_id)),
            acting_player_id,
        ),
        "p1_hand": p1_hand,
        "p2_hand": p2_hand,
        "p1_deck": [],  # Talishar hides deck order over HTTP
        "p2_deck": [],
        "p1_discard": p1_discard,
        "p2_discard": p2_discard,
        "p1_equipment": p1_equipment,
        "p2_equipment": p2_equipment,
        "p1_arsenal": p1_arsenal,
        "p2_arsenal": p2_arsenal,
        "p1_pitch": p1_pitch,
        "p2_pitch": p2_pitch,
        "p1_banish": _zone_card_ids(p1_state.get("playerBanish", [])),
        "p2_banish": _zone_card_ids(p2_state.get("playerBanish", [])),
        "combat_chain": _extract_combat_chain(p1_state) or _extract_combat_chain(p2_state),
        "pending_attack_power": _safe_int(
            p1_state.get("pendingAttackPower", p2_state.get("pendingAttackPower", 0))
        ),
        "pending_block_value": _safe_int(
            p1_state.get("pendingBlockValue", p2_state.get("pendingBlockValue", 0))
        ),
        "game_over": bool(p1_state.get("winner") or p2_state.get("winner")),
        "winner": _safe_int(p1_state.get("winner", p2_state.get("winner", -1)), -1),
    }


def extract_talishar_state(env: Any, *, player_id: int = 1) -> dict[str, Any]:
    """Build absolute P1/P2 snapshot from Talishar HTTP for both players."""
    fetch_state = getattr(env, "_fetch_state", None)
    by_player: dict[int, dict[str, Any]] = {}
    if callable(fetch_state):
        for pid in (1, 2):
            try:
                raw = fetch_state(player_id=pid, last_update=0)
                if isinstance(raw, dict) and raw:
                    by_player[pid] = raw
            except Exception:
                continue

    acting = _safe_int(
        getattr(env, "_acting_player_id", None) or player_id,
        player_id,
    )

    if 1 in by_player and 2 in by_player:
        return _absolute_talishar_snapshot(
            by_player[1],
            by_player[2],
            acting_player_id=acting,
        )

    fallback = by_player.get(acting) or by_player.get(1) or getattr(env, "_last_state", None)
    if isinstance(fallback, dict) and fallback:
        return _normalize_talishar_player_state(fallback, acting)
    return {}


def extract_cpp_state(cpp_env: Any) -> dict[str, Any]:
    """Read C++ GameState without Talishar overlay."""
    export_fn = getattr(cpp_env, "export_game_state", None)
    if callable(export_fn):
        return export_fn(absolute=True)
    snapshot_fn = getattr(getattr(cpp_env, "_gs", None), "snapshot_state", None)
    if callable(snapshot_fn):
        return dict(snapshot_fn())
    clear = getattr(cpp_env, "clear_talishar_state", None)
    if callable(clear):
        clear()
    raw_fn = getattr(cpp_env, "_raw_state_from_gs", None)
    if callable(raw_fn):
        return _cpp_raw_to_absolute(cpp_env, raw_fn())
    return {}


def _cpp_raw_to_absolute(cpp_env: Any, raw: dict[str, Any]) -> dict[str, Any]:
    gs = getattr(cpp_env, "_gs", None)
    if gs is None:
        return {}
    acting = _safe_int(getattr(cpp_env, "_acting_player", 1), 1)
    p1_health = _safe_int(getattr(gs, "p1_health", 0))
    p2_health = _safe_int(getattr(gs, "p2_health", 0))
    phase_code = ""
    phase_fn = getattr(cpp_env, "_phase_code", None)
    if callable(phase_fn):
        phase_code = str(phase_fn() or "").upper()

    def zone_cards(prefix: str, zone: str) -> list[str]:
        attr = f"{prefix}_{zone}"
        cards = getattr(gs, attr, None)
        if cards is None:
            return []
        ids: list[str] = []
        for card in list(cards):
            cid = getattr(card, "card_id", None) or (
                card.get("card_id") if isinstance(card, dict) else ""
            )
            if cid:
                ids.append(str(cid))
        return ids

    combat_chain: list[dict[str, Any]] = []
    chain_attr = getattr(gs, "combat_chain", None)
    if isinstance(chain_attr, list):
        for entry in chain_attr:
            if isinstance(entry, dict):
                combat_chain.append(entry)
            else:
                combat_chain.append(
                    {
                        "card_id": str(getattr(entry, "card_id", "")),
                        "power": _safe_int(getattr(entry, "power", 0)),
                        "defense": _safe_int(getattr(entry, "defense", 0)),
                    }
                )

    return {
        "acting_player_id": acting,
        "p1_health": p1_health,
        "p2_health": p2_health,
        "turn_no": _safe_int(getattr(gs, "turn_no", 0)),
        "phase": phase_code,
        "p1_hand_size": _safe_int(getattr(gs, "p1_hand_size", 0)),
        "p2_hand_size": _safe_int(getattr(gs, "p2_hand_size", 0)),
        "p1_deck_count": _safe_int(getattr(gs, "p1_deck_size", 0)),
        "p2_deck_count": _safe_int(getattr(gs, "p2_deck_size", 0)),
        "p1_pitch_count": _safe_int(getattr(gs, "p1_resources", 0)),
        "p2_pitch_count": _safe_int(getattr(gs, "p2_resources", 0)),
        "p1_resources": _talishar_action_points(
            phase_code, _safe_int(getattr(gs, "p1_action_points", 0))
        ),
        "p2_resources": _talishar_action_points(
            phase_code, _safe_int(getattr(gs, "p2_action_points", 0))
        ),
        "priority_player": _safe_int(getattr(gs, "priority", 0)) + 1,
        "p1_hand": zone_cards("p1", "hand"),
        "p2_hand": zone_cards("p2", "hand"),
        "p1_deck": zone_cards("p1", "deck"),
        "p2_deck": zone_cards("p2", "deck"),
        "p1_discard": zone_cards("p1", "discard"),
        "p2_discard": zone_cards("p2", "discard"),
        "p1_equipment": zone_cards("p1", "equipment"),
        "p2_equipment": zone_cards("p2", "equipment"),
        "p1_arsenal": zone_cards("p1", "arsenal"),
        "p2_arsenal": zone_cards("p2", "arsenal"),
        "p1_pitch": zone_cards("p1", "pitch"),
        "p2_pitch": zone_cards("p2", "pitch"),
        "p1_banish": zone_cards("p1", "banish"),
        "p2_banish": zone_cards("p2", "banish"),
        "combat_chain": combat_chain,
        "pending_attack_power": _safe_int(getattr(gs, "pending_attack_power", 0)),
        "pending_block_value": _safe_int(getattr(gs, "pending_block_value", 0)),
        "game_over": bool(getattr(gs, "game_over", False)),
        "winner": _safe_int(getattr(gs, "winner", -1), -1),
    }


def _compare_zone_multisets(
    tal_ids: list[str],
    cpp_ids: list[str],
    *,
    zone_name: str,
) -> Optional[StateDiscrepancy]:
    tal_counter = Counter(tal_ids)
    cpp_counter = Counter(cpp_ids)
    if tal_counter == cpp_counter:
        return None
    diff_card = ""
    for card_id in sorted(set(tal_counter) | set(cpp_counter)):
        if tal_counter[card_id] != cpp_counter[card_id]:
            diff_card = card_id
            break
    return StateDiscrepancy(
        category="game_state",
        taxonomy=f"{TAXONOMY_ZONE_PREFIX}{zone_name}",
        description=(
            f"zone {zone_name} mismatch: Talishar={dict(tal_counter)}, C++={dict(cpp_counter)}"
        ),
        talishar_value=tal_ids,
        cpp_value=cpp_ids,
        card_id=diff_card,
        zone=zone_name,
    )


def compare_game_states(
    tal: dict[str, Any],
    cpp: dict[str, Any],
    *,
    acting_player_id: Optional[int] = None,
    compare_zones: bool = True,
    compare_combat: bool = True,
) -> GameStateCompareResult:
    """Layered Tier A/B/C game-state comparison."""
    discrepancies: list[StateDiscrepancy] = []

    tier_a_keys = (
        ("p1_health", "p1_health", TAXONOMY_CARD_EFFECT),
        ("p2_health", "p2_health", TAXONOMY_CARD_EFFECT),
        ("turn_no", "turn_no", TAXONOMY_PHASE),
        ("phase", "phase", TAXONOMY_PHASE),
        ("acting_player_id", "acting_player_id", TAXONOMY_PHASE),
        ("priority_player", "priority_player", TAXONOMY_PHASE),
        ("p1_hand_size", "p1_hand_size", TAXONOMY_DECK_INIT),
        ("p2_hand_size", "p2_hand_size", TAXONOMY_DECK_INIT),
        ("p1_deck_count", "p1_deck_count", TAXONOMY_DECK_INIT),
        ("p2_deck_count", "p2_deck_count", TAXONOMY_DECK_INIT),
        ("p1_pitch_count", "p1_pitch_count", TAXONOMY_CARD_EFFECT),
        ("p2_pitch_count", "p2_pitch_count", TAXONOMY_CARD_EFFECT),
    )

    for key, _label, taxonomy in tier_a_keys:
        tal_v = tal.get(key)
        cpp_v = cpp.get(key)
        if key == "phase" and _phases_match(str(tal_v or ""), str(cpp_v or "")):
            continue
        if key in ("p1_pitch_count", "p2_pitch_count"):
            tal_phase = str(tal.get("phase") or "").upper()
            cpp_phase = str(cpp.get("phase") or "").upper()
            if tal_phase in ("P", "PITCH") or cpp_phase in ("P", "PITCH"):
                continue
        if tal_v != cpp_v:
            discrepancies.append(
                StateDiscrepancy(
                    category="game_state",
                    taxonomy=taxonomy,
                    description=f"{key}: Talishar={tal_v!r}, C++={cpp_v!r}",
                    talishar_value=tal_v,
                    cpp_value=cpp_v,
                )
            )

    if compare_zones:
        for abs_key, _tal_key, zone_name in ZONE_SPECS:
            tal_ids = tal.get(abs_key, [])
            cpp_ids = cpp.get(abs_key, [])
            if not isinstance(tal_ids, list):
                tal_ids = []
            if not isinstance(cpp_ids, list):
                cpp_ids = []

            # Talishar HTTP never exposes deck order; compare counts only.
            if zone_name == "deck":
                tal_count_key = abs_key.replace("_deck", "_deck_count")
                tal_count = _safe_int(tal.get(tal_count_key, len(tal_ids)))
                cpp_count = _safe_int(cpp.get(tal_count_key, len(cpp_ids)))
                if tal_count != cpp_count:
                    discrepancies.append(
                        StateDiscrepancy(
                            category="game_state",
                            taxonomy=f"{TAXONOMY_ZONE_PREFIX}deck_count",
                            description=(
                                f"deck count {abs_key}: Talishar={tal_count}, C++={cpp_count}"
                            ),
                            talishar_value=tal_count,
                            cpp_value=cpp_count,
                            zone="deck",
                        )
                    )
                continue

            if zone_name not in CPP_IMPLEMENTED_ZONES and tal_ids:
                discrepancies.append(
                    StateDiscrepancy(
                        category="game_state",
                        taxonomy=TAXONOMY_CPP_UNIMPLEMENTED,
                        description=f"C++ zone {zone_name} not implemented; Talishar has {len(tal_ids)} card(s)",
                        talishar_value=tal_ids,
                        cpp_value=cpp_ids,
                        zone=zone_name,
                    )
                )
                continue

            # Talishar HTTP often omits freshly banished tokens until the next poll.
            if zone_name == "banish" and not tal_ids and cpp_ids:
                continue

            disc = _compare_zone_multisets(tal_ids, cpp_ids, zone_name=zone_name)
            if disc is not None:
                discrepancies.append(disc)

    if compare_combat:
        tal_chain = tal.get("combat_chain", [])
        cpp_chain = cpp.get("combat_chain", [])
        tal_attack = _safe_int(tal.get("pending_attack_power", 0))
        cpp_attack = _safe_int(cpp.get("pending_attack_power", 0))
        tal_block = _safe_int(tal.get("pending_block_value", 0))
        cpp_block = _safe_int(cpp.get("pending_block_value", 0))

        if tal_attack != cpp_attack or tal_block != cpp_block:
            discrepancies.append(
                StateDiscrepancy(
                    category="game_state",
                    taxonomy=TAXONOMY_COMBAT,
                    description=(
                        f"combat totals: attack Talishar={tal_attack} C++={cpp_attack}, "
                        f"block Talishar={tal_block} C++={cpp_block}"
                    ),
                    talishar_value={"attack": tal_attack, "block": tal_block},
                    cpp_value={"attack": cpp_attack, "block": cpp_block},
                )
            )

        if tal_chain != cpp_chain and (tal_chain or cpp_chain):
            discrepancies.append(
                StateDiscrepancy(
                    category="game_state",
                    taxonomy=TAXONOMY_COMBAT,
                    description="combat_chain contents differ",
                    talishar_value=tal_chain,
                    cpp_value=cpp_chain,
                )
            )

        tal_phase = str(tal.get("phase", "") or "").upper()
        cpp_phase = str(cpp.get("phase", "") or "").upper()
        if tal_phase in {"M", "STARTTURN"} and cpp_phase in {"M", "STARTTURN"}:
            tal_r1 = _safe_int(tal.get("p1_resources", 0))
            tal_r2 = _safe_int(tal.get("p2_resources", 0))
            cpp_r1 = _safe_int(cpp.get("p1_resources", 0))
            cpp_r2 = _safe_int(cpp.get("p2_resources", 0))
            if tal_r1 != cpp_r1 or tal_r2 != cpp_r2:
                discrepancies.append(
                    StateDiscrepancy(
                        category="game_state",
                        taxonomy=TAXONOMY_CARD_EFFECT,
                        description=(
                            f"resources: P1 Talishar={tal_r1} C++={cpp_r1}, "
                            f"P2 Talishar={tal_r2} C++={cpp_r2}"
                        ),
                        talishar_value={"p1": tal_r1, "p2": tal_r2},
                        cpp_value={"p1": cpp_r1, "p2": cpp_r2},
                    )
                )

    if acting_player_id is not None:
        tal_acting = _safe_int(tal.get("acting_player_id", 0))
        if tal_acting != acting_player_id:
            discrepancies.append(
                StateDiscrepancy(
                    category="game_state",
                    taxonomy=TAXONOMY_PHASE,
                    description=f"acting_player_id expected {acting_player_id}, Talishar={tal_acting}",
                    talishar_value=tal_acting,
                    cpp_value=acting_player_id,
                )
            )

    return GameStateCompareResult(ok=not discrepancies, discrepancies=discrepancies)


def _normalise_action(action: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    normalised: dict[str, Any] = {}
    for field_name in fields:
        value = action.get(field_name)
        if field_name in {"index", "action_code"}:
            value = _safe_int(value)
        elif field_name in {"button_input", "card_id", "zone", "label"}:
            value = str(value or "")
        normalised[field_name] = value
    return normalised


def compare_legal_actions(
    legal_tal: list[dict[str, Any]],
    legal_cpp: list[dict[str, Any]],
    *,
    fields: tuple[str, ...] = INFO_LEGAL_ACTION_KEYS,
) -> tuple[bool, str]:
    if len(legal_tal) != len(legal_cpp):
        return False, f"count mismatch: Talishar={len(legal_tal)}, C++={len(legal_cpp)}"

    for index, (tal_action, cpp_action) in enumerate(zip(legal_tal, legal_cpp)):
        if not isinstance(tal_action, dict) or not isinstance(cpp_action, dict):
            return False, f"action[{index}] must be object in both lists"
        tal_normalised = _normalise_action(tal_action, fields)
        cpp_normalised = _normalise_action(cpp_action, fields)
        if tal_normalised != cpp_normalised:
            return (
                False,
                f"action[{index}] mismatch: Talishar={tal_normalised}, C++={cpp_normalised}",
            )
    return True, ""


def _parse_observation(source: str, observation: Any) -> tuple[Optional[dict[str, Any]], str]:
    if isinstance(observation, str):
        try:
            parsed = json.loads(observation)
        except json.JSONDecodeError as exc:
            return None, f"{source} observation is invalid JSON: {exc}"
    else:
        parsed = observation
    if not isinstance(parsed, dict):
        return None, f"{source} observation is {type(parsed).__name__}; expected JSON object"
    return parsed, ""


def compare_observations(
    obs_tal: Any,
    obs_cpp: Any,
    *,
    align_obs: bool = True,
) -> tuple[bool, str]:
    tal, msg = _parse_observation("Talishar", obs_tal)
    if tal is None:
        return False, msg
    cpp, msg = _parse_observation("C++", obs_cpp)
    if cpp is None:
        return False, msg

    mismatches: list[str] = []
    for key in OBSERVATION_SCALAR_KEYS:
        tal_value = tal.get(key)
        cpp_value = cpp.get(key)
        if key == "turnPhase" and _phases_match(str(tal_value or ""), str(cpp_value or "")):
            continue
        if key in OBSERVATION_NUMERIC_KEYS:
            tal_value = _safe_int(tal_value)
            cpp_value = _safe_int(cpp_value)
        elif key in {"selfPlay", "havePriority"}:
            tal_value = bool(tal_value)
            cpp_value = bool(cpp_value)
        if tal_value != cpp_value:
            mismatches.append(f"{key}: Talishar={tal_value!r}, C++={cpp_value!r}")

    tal_legal = tal.get("legalActions", [])
    cpp_legal = cpp.get("legalActions", [])
    success, msg = compare_legal_actions(
        tal_legal if isinstance(tal_legal, list) else [],
        cpp_legal if isinstance(cpp_legal, list) else [],
        fields=OBS_LEGAL_ACTION_KEYS,
    )
    if not success:
        mismatches.append(f"legalActions {msg}")

    if mismatches:
        return False, "; ".join(mismatches[:10])

    tal_vec = tal.get("observationVec")
    cpp_vec = cpp.get("observationVec")
    if isinstance(tal_vec, list) and isinstance(cpp_vec, list):
        if not align_obs:
            return True, ""
        if len(tal_vec) != PLAYER_OBS_DIM or len(cpp_vec) != PLAYER_OBS_DIM:
            return False, f"observationVec dim mismatch: {len(tal_vec)} vs {len(cpp_vec)}"
        tal_arr = np.asarray(tal_vec, dtype=np.float64)
        cpp_arr = np.asarray(cpp_vec, dtype=np.float64)
        if align_obs:
            tal_arr = align_observation_for_cpp_training(tal_arr)
            cpp_arr = align_observation_for_cpp_training(cpp_arr)
        ok, vec_msg = observation_vectors_aligned(tal_arr, cpp_arr, atol=OBS_VEC_TOLERANCE)
        if not ok:
            return False, vec_msg

    return True, ""


def compare_rewards(reward_tal: float, reward_cpp: float) -> tuple[bool, str]:
    diff = abs(float(reward_tal) - float(reward_cpp))
    if diff <= REWARD_TOLERANCE:
        return True, f"diff={diff:.6f}"
    return False, f"Talishar={reward_tal}, C++={reward_cpp}, diff={diff:.6f}"


def compare_info_contract(info_tal: dict[str, Any], info_cpp: dict[str, Any]) -> tuple[bool, str]:
    for key in INFO_CONTRACT_KEYS:
        if key not in info_tal and key not in info_cpp:
            continue
        if key not in info_tal or key not in info_cpp:
            return False, f"{key!r} presence mismatch"
        if key == "legal_actions":
            success, msg = compare_legal_actions(info_tal[key], info_cpp[key])
            if not success:
                return False, f"legal_actions {msg}"
            continue
        if key == "repeat_penalty":
            success, msg = compare_rewards(float(info_tal[key]), float(info_cpp[key]))
            if not success:
                return False, f"repeat_penalty mismatch: {msg}"
            continue
        if key == "turn":
            if int(info_tal[key] or 0) != int(info_cpp[key] or 0):
                return False, f"{key!r} mismatch: Talishar={info_tal[key]}, C++={info_cpp[key]}"
            continue
        if info_tal[key] != info_cpp[key]:
            return False, f"{key!r} mismatch: Talishar={info_tal[key]}, C++={info_cpp[key]}"
    return True, ""


def compare_agent_contract(
    step_tal: Any,
    step_cpp: Any,
    *,
    align_obs: bool = True,
) -> list[StateDiscrepancy]:
    """Compare observation, info, rewards, and termination between step results."""
    discrepancies: list[StateDiscrepancy] = []

    ok, msg = compare_observations(
        step_tal.observation,
        step_cpp.observation,
        align_obs=align_obs,
    )
    if not ok:
        discrepancies.append(
            StateDiscrepancy(
                category="observation",
                taxonomy=TAXONOMY_OBS_CONTRACT,
                description=msg,
                talishar_value=step_tal.observation,
                cpp_value=step_cpp.observation,
            )
        )

    ok, msg = compare_rewards(step_tal.reward, step_cpp.reward)
    if not ok:
        discrepancies.append(
            StateDiscrepancy(
                category="reward",
                taxonomy=TAXONOMY_REWARD,
                description=msg,
                talishar_value=step_tal.reward,
                cpp_value=step_cpp.reward,
            )
        )

    if bool(step_tal.terminated) != bool(step_cpp.terminated):
        discrepancies.append(
            StateDiscrepancy(
                category="termination",
                taxonomy=TAXONOMY_TERMINATION,
                description=f"terminated: Talishar={step_tal.terminated}, C++={step_cpp.terminated}",
                talishar_value=step_tal.terminated,
                cpp_value=step_cpp.terminated,
            )
        )

    if bool(step_tal.truncated) != bool(step_cpp.truncated):
        discrepancies.append(
            StateDiscrepancy(
                category="truncation",
                taxonomy=TAXONOMY_TERMINATION,
                description=f"truncated: Talishar={step_tal.truncated}, C++={step_cpp.truncated}",
                talishar_value=step_tal.truncated,
                cpp_value=step_cpp.truncated,
            )
        )

    info_tal = getattr(step_tal, "info", {}) or {}
    info_cpp = getattr(step_cpp, "info", {}) or {}
    ok, msg = compare_info_contract(info_tal, info_cpp)
    if not ok:
        taxonomy = TAXONOMY_LEGAL if "legal_actions" in msg else TAXONOMY_OBS_CONTRACT
        discrepancies.append(
            StateDiscrepancy(
                category="legal_actions" if taxonomy == TAXONOMY_LEGAL else "info",
                taxonomy=taxonomy,
                description=msg,
                talishar_value=info_tal,
                cpp_value=info_cpp,
            )
        )

    return discrepancies


def build_initial_sync_payload(tal_state: dict[str, Any]) -> dict[str, Any]:
    """Extract hands, deck order, equipment, and seed hints from Talishar baseline."""
    acting = _safe_int(tal_state.get("actingPlayerID", 1), 1)
    fetch_fn = tal_state.get("_fetch_both")
    if callable(fetch_fn):
        p1_state, p2_state = fetch_fn()
        normalized = _absolute_talishar_snapshot(
            p1_state,
            p2_state,
            acting_player_id=acting,
        )
        resource_pools = {
            1: _safe_int(p1_state.get("playerPitchCount", 0)),
            2: _safe_int(p2_state.get("playerPitchCount", 0)),
        }
        action_points = {
            1: _safe_int(p1_state.get("playerAP", 0)),
            2: _safe_int(p2_state.get("playerAP", 0)),
        }
        if action_points[1] == 0:
            action_points[1] = _safe_int(p2_state.get("opponentAP", 0))
        if action_points[2] == 0:
            action_points[2] = _safe_int(p1_state.get("opponentAP", 0))
    else:
        normalized = _normalize_talishar_player_state(tal_state, acting)
        resource_pools = {
            1: _safe_int(normalized.get("p1_pitch_count", 0)),
            2: _safe_int(normalized.get("p2_pitch_count", 0)),
        }
        action_points = {1: 0, 2: 0}
    if not any(int(v or 0) > 0 for v in action_points.values()):
        ap = _safe_int(tal_state.get("playerAP", 0))
        if ap > 0 and acting in (1, 2):
            action_points[acting] = ap
    opening_hands = {
        1: normalized.get("p1_hand", []),
        2: normalized.get("p2_hand", []),
    }
    return {
        "opening_hands": opening_hands,
        "deck_orders": {
            1: normalized.get("p1_deck", []),
            2: normalized.get("p2_deck", []),
        },
        "equipment": {
            1: normalized.get("p1_equipment", []),
            2: normalized.get("p2_equipment", []),
        },
        "resources": resource_pools,
        "action_points": action_points,
        "acting_player_id": acting,
        "turn_no": normalized.get("turn_no", 0),
        "phase": normalized.get("phase", ""),
    }
