from check_cpp_vs_talishar_parity import (
    _card_ids_from_state_hand,
    _is_parity_baseline_state,
    _pick_pregame_advance_index,
)


class _PhaseEnv:
    @staticmethod
    def _phase_str(state):
        turn_phase = state.get("turnPhase", {})
        if isinstance(turn_phase, dict):
            return str(turn_phase.get("turnPhase", ""))
        return ""


def test_card_ids_from_state_hand_reads_card_number() -> None:
    state = {
        "playerHand": [
            {"cardNumber": "WTR001", "label": "Surging Strike"},
            {"cardID": "ARC002"},
        ]
    }
    assert _card_ids_from_state_hand(state) == ["WTR001", "ARC002"]


def test_is_parity_baseline_state_requires_main_phase_and_hand() -> None:
    env = _PhaseEnv()
    assert _is_parity_baseline_state(
        env,
        {
            "turnPhase": {"turnPhase": "M"},
            "playerHand": [{"cardNumber": "WTR001"}],
            "playerDeckCount": 30,
            "opponentDeckCount": 30,
        },
    )
    assert not _is_parity_baseline_state(
        env,
        {
            "turnPhase": {"turnPhase": "startgame"},
            "playerHand": [],
            "playerDeckCount": 0,
            "opponentDeckCount": 0,
        },
    )


def test_pick_pregame_advance_index_prefers_pass() -> None:
    legal = [
        {"action_code": 3, "label": "Equip"},
        {"action_code": 99, "label": "Confirm"},
    ]
    assert _pick_pregame_advance_index(legal) == 1
