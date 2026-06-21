from check_cpp_vs_talishar_parity import (
    _card_ids_from_state_hand,
    _is_parity_baseline_state,
    _json_safe_value,
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


def test_json_safe_value_converts_ndarray() -> None:
    import json

    import numpy as np

    payload = _json_safe_value({
        "obs": np.array([1.0, 2.5, 3.0]),
        "nested": {"weights": np.array([[1, 2], [3, 4]])},
    })
    json.dumps(payload)
    assert payload["obs"] == [1.0, 2.5, 3.0]
    assert payload["nested"]["weights"] == [[1, 2], [3, 4]]


def test_json_safe_value_does_not_reparse_plain_strings() -> None:
    import json

    payload = _json_safe_value({
        "label": "snatch_red",
        "note": "not json",
        "nested": {"action": "warrior_s_valor_red"},
    })
    assert payload["label"] == "snatch_red"
    json.dumps(payload)
