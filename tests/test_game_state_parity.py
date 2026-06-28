"""Offline tests for game_state_parity helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.game_state_parity import (
    TAXONOMY_COMBAT,
    TAXONOMY_ZONE_PREFIX,
    build_initial_sync_payload,
    compare_game_states,
    compare_legal_actions,
)


def test_compare_game_states_detects_hp_mismatch() -> None:
    tal = {"p1_health": 20, "p2_health": 20, "turn_no": 1, "phase": "M"}
    cpp = {"p1_health": 19, "p2_health": 20, "turn_no": 1, "phase": "M"}
    result = compare_game_states(tal, cpp)
    assert not result.ok
    assert any(d.taxonomy == "card_effect" for d in result.discrepancies)


def test_compare_zone_multiset_mismatch() -> None:
    tal = {
        "p1_health": 20,
        "p2_health": 20,
        "turn_no": 1,
        "phase": "M",
        "p1_hand": ["snatch_red", "snatch_red"],
        "p2_hand": [],
        "p1_deck_count": 0,
        "p2_deck_count": 0,
        "p1_pitch_count": 0,
        "p2_pitch_count": 0,
    }
    cpp = dict(tal)
    cpp["p1_hand"] = ["snatch_red"]
    result = compare_game_states(tal, cpp)
    assert not result.ok
    assert any(d.taxonomy == f"{TAXONOMY_ZONE_PREFIX}hand" for d in result.discrepancies)


def test_compare_combat_chain_totals() -> None:
    tal = {
        "p1_health": 20,
        "p2_health": 20,
        "turn_no": 1,
        "phase": "B",
        "pending_attack_power": 6,
        "pending_block_value": 2,
        "combat_chain": [{"card_id": "wtr001", "power": 6, "defense": 0}],
        "p1_hand_size": 0,
        "p2_hand_size": 0,
        "p1_deck_count": 0,
        "p2_deck_count": 0,
        "p1_pitch_count": 0,
        "p2_pitch_count": 0,
    }
    cpp = dict(tal)
    cpp["pending_block_value"] = 0
    result = compare_game_states(tal, cpp)
    assert not result.ok
    assert any(d.taxonomy == TAXONOMY_COMBAT for d in result.discrepancies)


def test_build_initial_sync_payload_extracts_hands() -> None:
    state = {
        "actingPlayerID": 1,
        "playerHealth": 20,
        "opponentHealth": 20,
        "turnNo": 1,
        "turnPhase": {"turnPhase": "M"},
        "playerHand": [{"cardNumber": "WTR001"}, {"cardNumber": "WTR002"}],
        "opponentHand": [{"cardNumber": "WTR003"}],
        "playerDeckCount": 30,
        "opponentDeckCount": 30,
    }
    payload = build_initial_sync_payload(state)
    assert payload["opening_hands"][1] == ["WTR001", "WTR002"]
    assert payload["opening_hands"][2] == ["WTR003"]


def test_build_initial_sync_payload_cross_view_action_points() -> None:
    state = {
        "actingPlayerID": 1,
        "_fetch_both": lambda: (
            {"playerAP": 1, "opponentAP": 0, "playerPitchCount": 0, "opponentPitchCount": 0},
            {"playerAP": 0, "opponentAP": 0, "playerPitchCount": 0, "opponentPitchCount": 0},
        ),
    }
    payload = build_initial_sync_payload(state)
    assert payload["action_points"][1] == 1
    assert payload["action_points"][2] == 0


def test_cardback_filtered_from_sync_payload() -> None:
    from flesh_and_blood_rlbridge.game_state_parity import (
        build_initial_sync_payload,
        is_syncable_card_id,
    )

    assert not is_syncable_card_id("CardBack")
    state = {
        "actingPlayerID": 1,
        "playerHand": [{"cardNumber": "CardBack"}, {"cardNumber": "snatch_red"}],
        "opponentHand": [{"cardNumber": "CardBack"}],
        "playerDeckCount": 30,
        "opponentDeckCount": 30,
        "turnPhase": {"turnPhase": "M"},
    }
    payload = build_initial_sync_payload(state)
    assert payload["opening_hands"][1] == ["snatch_red"]
    assert payload["opening_hands"][2] == []


def test_compare_legal_actions_order_sensitive() -> None:
    tal = [{"action_code": 99, "button_input": "", "card_id": "", "zone": "button", "label": "Pass"}]
    cpp = [{"action_code": 27, "button_input": "0", "card_id": "x", "zone": "hand", "label": "Play"}]
    ok, msg = compare_legal_actions(tal, cpp)
    assert not ok
    assert "count mismatch" in msg or "mismatch" in msg
