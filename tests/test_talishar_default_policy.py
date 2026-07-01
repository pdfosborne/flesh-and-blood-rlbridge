import sys
from pathlib import Path

# Allow tests from repo root without package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.talishar_default_policy import choose_talishar_action_index


def test_defense_window_prefers_highest_block() -> None:
    state = {
        "turnPhase": {"turnPhase": "D", "caption": "Defend"},
        "playerHand": [
            {
                "action": 27,
                "actionDataOverride": "0",
                "label": "Blue Card",
                "defense": 2,
                "power": 1,
            },
            {
                "action": 27,
                "actionDataOverride": "1",
                "label": "Red Card",
                "defense": 3,
                "power": 3,
            },
        ],
    }
    legal = [
        {
            "action_code": 27,
            "button_input": "0",
            "zone": "hand",
            "label": "Blue Card",
        },
        {
            "action_code": 27,
            "button_input": "1",
            "zone": "hand",
            "label": "Red Card",
        },
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "label": "Pass",
        },
    ]
    assert choose_talishar_action_index(legal, state) == 1


def test_offense_window_prefers_highest_attack() -> None:
    state = {
        "turnPhase": {"turnPhase": "M", "caption": "Choose an action"},
        "playerHand": [
            {
                "action": 27,
                "actionDataOverride": "0",
                "label": "Swing Small",
                "power": 2,
                "defense": 3,
                "cost": 0,
            },
            {
                "action": 27,
                "actionDataOverride": "1",
                "label": "Swing Big",
                "power": 5,
                "defense": 2,
                "cost": 1,
            },
        ],
    }
    legal = [
        {
            "action_code": 27,
            "button_input": "0",
            "zone": "hand",
            "label": "Swing Small",
        },
        {
            "action_code": 27,
            "button_input": "1",
            "zone": "hand",
            "label": "Swing Big",
        },
        {
            "action_code": 99,
            "button_input": "",
            "zone": "button",
            "label": "Pass",
        },
    ]
    assert choose_talishar_action_index(legal, state) == 1


def test_popup_prefers_yes_like_confirmation() -> None:
    legal = [
        {
            "action_code": 1,
            "button_input": "n",
            "zone": "popup",
            "label": "No",
        },
        {
            "action_code": 1,
            "button_input": "y",
            "zone": "popup",
            "label": "Yes",
        },
    ]
    assert choose_talishar_action_index(legal, {}) == 1


def test_only_pass_returns_zero() -> None:
    legal = [{"action_code": 99, "button_input": "", "zone": "button", "label": "Pass"}]
    assert choose_talishar_action_index(legal, {}) == 0


def test_confirm_phase_always_passes() -> None:
    """In INSTANT/A/ARS phases the policy must pass immediately (no pitch needed)."""
    for phase_code in ("INSTANT", "A", "ARS", "maychoosemultizone"):
        state = {"turnPhase": {"turnPhase": phase_code}}
        legal = [
            {"action_code": 27, "button_input": "0", "zone": "hand", "label": "some_card"},
            {"action_code": 99, "button_input": "", "zone": "button", "label": "Pass"},
        ]
        result = choose_talishar_action_index(legal, state)
        assert result == 1, f"Expected pass (index 1) in phase {phase_code}, got {result}"


def test_pitch_phase_pitches_hand_card_before_passing() -> None:
    """In P (pitch) phase the policy must pitch a hand card, not pass immediately."""
    state = {"turnPhase": {"turnPhase": "P"}}
    legal = [
        {"action_code": 27, "button_input": "0", "zone": "hand", "label": "some_red_card"},
        {"action_code": 10000, "button_input": "", "zone": "button", "label": "Cancel"},
        {"action_code": 99, "button_input": "", "zone": "button", "label": "Pass"},
    ]
    result = choose_talishar_action_index(legal, state)
    assert result == 0, f"Expected to pitch hand card (index 0) in phase P, got {result}"


def test_pitch_phase_passes_when_no_pitch_cards() -> None:
    """In P phase with no pitchable hand cards, policy should pass to confirm."""
    state = {"turnPhase": {"turnPhase": "P"}}
    legal = [
        {"action_code": 10000, "button_input": "", "zone": "button", "label": "Cancel"},
        {"action_code": 99, "button_input": "", "zone": "button", "label": "Pass"},
    ]
    result = choose_talishar_action_index(legal, state)
    assert result == 1, f"Expected pass (index 1) in P with no pitch cards, got {result}"


def test_choosemultizone_picks_popup_card() -> None:
    """In choosemultizone the policy must pick the first popup card."""
    state = {"turnPhase": {"turnPhase": "choosemultizone"}}
    legal = [
        {"action_code": 99,  "button_input": "",  "zone": "button", "label": "Pass"},
        {"action_code": 16,  "button_input": "0", "zone": "popup",  "label": "some_card"},
        {"action_code": 16,  "button_input": "1", "zone": "popup",  "label": "other_card"},
    ]
    result = choose_talishar_action_index(legal, state)
    assert result == 1, f"Expected popup card (index 1) in choosemultizone, got {result}"


def test_main_phase_prefers_higher_power_attack() -> None:
    """Main phase: policy should prefer the stronger attack when both are legal."""
    state = {
        "turnPhase": {"turnPhase": "M"},
        "playerHand": [
            {"action": 27, "actionDataOverride": "0", "label": "oasis_respite_red", "power": 0},
            {"action": 27, "actionDataOverride": "1", "label": "other_attack_red", "power": 4},
        ],
    }
    legal = [
        {"action_code": 27, "button_input": "0", "zone": "hand", "label": "oasis_respite_red"},
        {"action_code": 27, "button_input": "1", "zone": "hand", "label": "other_attack_red"},
        {"action_code": 99, "button_input": "",  "zone": "button", "label": "Pass"},
    ]
    result = choose_talishar_action_index(legal, state)
    assert result == 1, f"Expected other_attack_red (index 1), got {result}"


def test_choosehand_picks_first_hand_card() -> None:
    """CHOOSEHAND: must pick a hand card (action=16), not Pass (which is a no-op)."""
    state = {"turnPhase": {"turnPhase": "CHOOSEHAND"}}
    legal = [
        {"action_code": 16, "button_input": "0", "zone": "hand", "label": "springboard_somersault_yellow"},
        {"action_code": 16, "button_input": "1", "zone": "hand", "label": "oasis_respite_red"},
        # mode=99 would be silently ignored by CanPassPhase=0, so no Pass button exposed
    ]
    result = choose_talishar_action_index(legal, state)
    assert result == 0, f"Expected first hand card (index 0), got {result}"


def test_choosehandcancel_uses_cancel_when_no_hand_cards() -> None:
    """CHOOSEHANDCANCEL with no hand cards: must use Cancel (10000), not Pass."""
    state = {"turnPhase": {"turnPhase": "CHOOSEHANDCANCEL"}}
    legal = [
        {"action_code": 10000, "button_input": "", "zone": "button", "label": "Cancel"},
    ]
    result = choose_talishar_action_index(legal, state)
    assert result == 0, f"Expected Cancel (index 0), got {result}"


def test_buttoninput_picks_mode17_button() -> None:
    """BUTTONINPUT: must pick a mode=17 popup button, not Pass (no-op)."""
    state = {"turnPhase": {"turnPhase": "BUTTONINPUT"}}
    legal = [
        {"action_code": 17, "button_input": "0", "zone": "popup", "label": "Option A"},
        {"action_code": 17, "button_input": "1", "zone": "popup", "label": "Option B"},
    ]
    result = choose_talishar_action_index(legal, state)
    assert result == 0, f"Expected first popup button (index 0), got {result}"


def test_buttoninputnopass_picks_mode17_button() -> None:
    """BUTTONINPUTNOPASS: same as BUTTONINPUT — pick mode=17 button."""
    state = {"turnPhase": {"turnPhase": "BUTTONINPUTNOPASS"}}
    legal = [
        {"action_code": 17, "button_input": "42", "zone": "popup", "label": "Choice"},
    ]
    result = choose_talishar_action_index(legal, state)
    assert result == 0, f"Expected popup button (index 0), got {result}"


# ── Block phase tests ─────────────────────────────────────────────────────────

def _hand_with_attackers(n: int) -> list[dict]:
    """n attack cards (def=2) in hand — not usable as blockers."""
    return [
        {"action": 27, "actionDataOverride": str(i), "label": f"Attacker{i}", "defense": 2, "power": 4}
        for i in range(n)
    ]

def _hand_with_blocker_and_attackers(n_attackers: int) -> list[dict]:
    """One real blocker (def=3, cost=1, resource=1) plus n attack cards."""
    return [
        {"action": 27, "actionDataOverride": "0", "label": "Blocker", "defense": 3, "power": 1, "cost": 1, "resource": 1},
        *[
            {"action": 27, "actionDataOverride": str(i + 1), "label": f"Attacker{i}", "defense": 2, "power": 4}
            for i in range(n_attackers)
        ],
    ]

def _block_legal_from_hand(hand: list[dict]) -> list[dict]:
    """Legal action list matching the given hand, plus Pass."""
    legal = [
        {"action_code": 27, "button_input": str(i), "zone": "hand", "label": c["label"]}
        for i, c in enumerate(hand)
    ]
    legal.append({"action_code": 99, "button_input": "", "zone": "button", "label": "Pass"})
    return legal


def test_block_uses_dedicated_blocker_card() -> None:
    """B phase: plays the card with defense ≥ 3 when attackers are plentiful."""
    hand = _hand_with_blocker_and_attackers(n_attackers=4)  # 4 > _MIN_HAND_FOR_ATTACK=3
    state = {
        "turnPhase": {"turnPhase": "B"},
        "playerHand": hand,
        "activeChainLink": {},
    }
    result = choose_talishar_action_index(_block_legal_from_hand(hand), state)
    assert result == 0, f"Expected blocker (index 0), got {result}"


def test_block_passes_when_hand_is_mostly_attackers_and_thin() -> None:
    """B phase: ≤ _MIN_HAND_FOR_ATTACK attack cards left → skip block, preserve hand."""
    hand = _hand_with_blocker_and_attackers(n_attackers=3)  # exactly _MIN_HAND_FOR_ATTACK=3
    state = {
        "turnPhase": {"turnPhase": "B"},
        "playerHand": hand,
        "activeChainLink": {},
    }
    result = choose_talishar_action_index(_block_legal_from_hand(hand), state)
    pass_idx = len(hand)  # Pass is last in the legal list
    assert result == pass_idx, f"Expected pass (index {pass_idx}) with thin hand, got {result}"


def test_block_passes_when_no_dedicated_blockers() -> None:
    """B phase: only attack cards in hand (def < 3) → pass, keep them for offence."""
    hand = _hand_with_attackers(4)
    state = {
        "turnPhase": {"turnPhase": "B"},
        "playerHand": hand,
        "activeChainLink": {},
    }
    result = choose_talishar_action_index(_block_legal_from_hand(hand), state)
    pass_idx = len(hand)
    assert result == pass_idx, f"Expected pass (index {pass_idx}) with no blockers, got {result}"


def test_block_prefers_highest_defense_blocker() -> None:
    """B phase: two blocker-quality cards → picks the one with higher defense."""
    hand = [
        {"action": 27, "actionDataOverride": "0", "label": "WeakBlocker",   "defense": 3, "power": 1, "cost": 1, "resource": 1},
        {"action": 27, "actionDataOverride": "1", "label": "StrongBlocker",  "defense": 5, "power": 1, "cost": 1, "resource": 1},
        # extra attackers so hand threshold is satisfied (need > _MIN_HAND_FOR_ATTACK=3)
        {"action": 27, "actionDataOverride": "2", "label": "Attacker1", "defense": 2, "power": 4},
        {"action": 27, "actionDataOverride": "3", "label": "Attacker2", "defense": 2, "power": 4},
        {"action": 27, "actionDataOverride": "4", "label": "Attacker3", "defense": 2, "power": 4},
        {"action": 27, "actionDataOverride": "5", "label": "Attacker4", "defense": 2, "power": 4},
    ]
    legal = _block_legal_from_hand(hand)
    state = {"turnPhase": {"turnPhase": "B"}, "playerHand": hand, "activeChainLink": {}}
    result = choose_talishar_action_index(legal, state)
    assert result == 1, f"Expected StrongBlocker (index 1), got {result}"


# ── max_pitch_value / min_resource_cost filter tests ─────────────────────────

def _hand_with_mixed_pitch_cards() -> list[dict]:
    """One red blocker (pitch=1, def=3), one blue blocker (pitch=3, def=3), plus attackers."""
    return [
        # index 0: red blocker — low pitch, good blocker
        {"action": 27, "actionDataOverride": "0", "label": "RedBlocker",  "defense": 3, "power": 3, "resource": 1, "cost": 1},
        # index 1: blue blocker — high pitch, good blocker
        {"action": 27, "actionDataOverride": "1", "label": "BlueBlocker", "defense": 3, "power": 1, "resource": 3, "cost": 1},
        # indices 2-5: attackers (def<3, should not block)
        {"action": 27, "actionDataOverride": "2", "label": "Atk1", "defense": 2, "power": 4, "resource": 1, "cost": 1},
        {"action": 27, "actionDataOverride": "3", "label": "Atk2", "defense": 2, "power": 4, "resource": 1, "cost": 1},
        {"action": 27, "actionDataOverride": "4", "label": "Atk3", "defense": 2, "power": 4, "resource": 1, "cost": 1},
        {"action": 27, "actionDataOverride": "5", "label": "Atk4", "defense": 2, "power": 4, "resource": 1, "cost": 1},
    ]


def test_block_max_pitch_value_filters_blue_cards() -> None:
    """max_pitch_value=1 → only red (pitch=1) card eligible; blue (pitch=3) skipped."""
    hand = _hand_with_mixed_pitch_cards()
    legal = _block_legal_from_hand(hand)
    state = {"turnPhase": {"turnPhase": "B"}, "playerHand": hand, "activeChainLink": {}}
    result = choose_talishar_action_index(legal, state, max_pitch_value=1)
    assert result == 0, f"Expected RedBlocker (index 0) with max_pitch_value=1, got {result}"


def test_block_max_pitch_value_unrestricted_allows_any_blocker() -> None:
    """max_pitch_value=999 (explicit override) → both blocker cards eligible; picks highest def."""
    hand = _hand_with_mixed_pitch_cards()
    legal = _block_legal_from_hand(hand)
    state = {"turnPhase": {"turnPhase": "B"}, "playerHand": hand, "activeChainLink": {}}
    # Both have def=3, so first eligible wins (red at index 0)
    result = choose_talishar_action_index(legal, state, max_pitch_value=999)
    assert result in (0, 1), f"Expected a blocker (0 or 1) with no pitch filter, got {result}"


def test_block_min_resource_cost_filters_free_cards() -> None:
    """min_resource_cost=1 → free cards (cost=0) skipped; only paid blockers used."""
    hand = [
        # cost=0 blocker — should be filtered out
        {"action": 27, "actionDataOverride": "0", "label": "FreeBlocker", "defense": 3, "power": 1, "resource": 1, "cost": 0},
        # cost=1 blocker — passes the filter
        {"action": 27, "actionDataOverride": "1", "label": "PaidBlocker", "defense": 3, "power": 1, "resource": 1, "cost": 1},
        # extra attackers
        {"action": 27, "actionDataOverride": "2", "label": "Atk1", "defense": 2, "power": 4, "resource": 1, "cost": 1},
        {"action": 27, "actionDataOverride": "3", "label": "Atk2", "defense": 2, "power": 4, "resource": 1, "cost": 1},
        {"action": 27, "actionDataOverride": "4", "label": "Atk3", "defense": 2, "power": 4, "resource": 1, "cost": 1},
        {"action": 27, "actionDataOverride": "5", "label": "Atk4", "defense": 2, "power": 4, "resource": 1, "cost": 1},
    ]
    legal = _block_legal_from_hand(hand)
    state = {"turnPhase": {"turnPhase": "B"}, "playerHand": hand, "activeChainLink": {}}
    result = choose_talishar_action_index(legal, state, min_resource_cost=1)
    assert result == 1, f"Expected PaidBlocker (index 1) with min_resource_cost=1, got {result}"


def test_block_combined_filters_fall_through_to_pass() -> None:
    """Both filters applied and no card satisfies them → pass."""
    hand = [
        # pitch=3 (blue) and cost=0: fails max_pitch_value=1
        {"action": 27, "actionDataOverride": "0", "label": "BlueBlocker", "defense": 3, "power": 1, "resource": 3, "cost": 0},
        # extra attackers
        {"action": 27, "actionDataOverride": "1", "label": "Atk1", "defense": 2, "power": 4, "resource": 1, "cost": 1},
        {"action": 27, "actionDataOverride": "2", "label": "Atk2", "defense": 2, "power": 4, "resource": 1, "cost": 1},
        {"action": 27, "actionDataOverride": "3", "label": "Atk3", "defense": 2, "power": 4, "resource": 1, "cost": 1},
        {"action": 27, "actionDataOverride": "4", "label": "Atk4", "defense": 2, "power": 4, "resource": 1, "cost": 1},
    ]
    legal = _block_legal_from_hand(hand)
    state = {"turnPhase": {"turnPhase": "B"}, "playerHand": hand, "activeChainLink": {}}
    result = choose_talishar_action_index(legal, state, max_pitch_value=1)
    pass_idx = len(hand)
    assert result == pass_idx, f"Expected pass (index {pass_idx}) when all blockers filtered, got {result}"


def test_has_card_for_arsenal_add_attack_in_hand() -> None:
    from flesh_and_blood_rlbridge.talishar_default_policy import has_card_for_arsenal_add

    state = {
        "playerHand": [{"cardNumber": "nimblism_red", "action": 4}],
        "playerEquipment": [],
    }
    assert has_card_for_arsenal_add("nimblism_red", state)


def test_has_card_for_arsenal_add_utility_item_only_in_hand_is_invalid() -> None:
    from flesh_and_blood_rlbridge.talishar_default_policy import has_card_for_arsenal_add

    state = {
        "playerHand": [
            {"cardNumber": "swiftstrike_bracers", "action": 4},
            {"cardNumber": "swiftstrike_bracers", "action": 4},
        ],
        "playerEquipment": [],
    }
    assert not has_card_for_arsenal_add("swiftstrike_bracers", state)


def test_ars_policy_passes_when_only_invalid_arsenal_targets() -> None:
    state = {
        "turnPhase": {"turnPhase": "ARS"},
        "playerHand": [
            {
                "cardNumber": "swiftstrike_bracers",
                "action": 4,
                "actionDataOverride": "swiftstrike_bracers",
            },
            {
                "cardNumber": "swiftstrike_bracers",
                "action": 4,
                "actionDataOverride": "swiftstrike_bracers",
            },
        ],
        "playerEquipment": [],
    }
    legal = [
        {
            "action_code": 4,
            "button_input": "swiftstrike_bracers",
            "zone": "hand",
            "card_id": "swiftstrike_bracers",
            "label": "Swiftstrike Bracers",
        },
        {
            "action_code": 4,
            "button_input": "swiftstrike_bracers",
            "zone": "hand",
            "card_id": "swiftstrike_bracers",
            "label": "Swiftstrike Bracers",
        },
        {"action_code": 99, "button_input": "", "zone": "button", "label": "Pass"},
    ]
    assert choose_talishar_action_index(legal, state) == 2
