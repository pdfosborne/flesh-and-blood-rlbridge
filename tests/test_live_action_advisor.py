"""Tests for live action coaching helpers."""

from __future__ import annotations

import json

import numpy as np

from flesh_and_blood_rlbridge.frontend_action_overlay import (
    ActionCoachHint,
    overlay_hints_payload,
    playwright_update_overlay_script,
)
from flesh_and_blood_rlbridge.live_action_advisor import (
    _legal_entries_from_obs,
    compute_agent_policy_scores,
    normalize_legal_entries,
)


class _FakeActor:
    def __init__(self, logits: list[float]) -> None:
        self._logits = np.asarray(logits, dtype=np.float64)

    def predict(self, _obs_vec: np.ndarray) -> np.ndarray:
        return self._logits


class _FakeAgent:
    def __init__(self, logits: list[float]) -> None:
        self._mask_actions = True
        self._actor = _FakeActor(logits)

    def _obs_to_vec(self, obs: object) -> np.ndarray:
        return np.zeros(8, dtype=np.float64)


def test_legal_entries_from_obs_reads_talishar_shape() -> None:
    obs = json.dumps(
        {
            "legalActions": [
                {"index": 0, "label": "pass_red", "zone": "button"},
                {"index": 1, "label": "attack_red", "zone": "hand"},
            ]
        }
    )
    entries = _legal_entries_from_obs(obs)
    assert len(entries) == 2
    assert entries[1]["label"] == "attack_red"


def test_compute_agent_policy_scores_marks_best_action() -> None:
    obs = json.dumps(
        {
            "legalActions": [
                {"index": 0, "label": "pass", "zone": "button"},
                {"index": 1, "label": "card_a", "zone": "hand"},
            ]
        }
    )
    agent = _FakeAgent([0.1, 2.5])
    hints = compute_agent_policy_scores(agent, obs)
    assert len(hints) == 2
    assert hints[0].policy_pct is not None
    assert hints[1].is_best is True
    assert hints[0].is_best is False
    assert hints[1].policy_pct > hints[0].policy_pct


def test_normalize_legal_entries_from_talishar_env_shape() -> None:
    legal = [
        {
            "action_code": 27,
            "button_input": "0",
            "card_id": "lightning_press_red",
            "zone": "hand",
            "label": "lightning_press_red",
        },
        {
            "action_code": 99,
            "button_input": "",
            "card_id": "",
            "zone": "button",
            "label": "Pass",
        },
    ]
    entries = normalize_legal_entries(legal)
    assert len(entries) == 2
    assert entries[0]["index"] == 0
    assert entries[0]["match_text"] == "lightning_press_red"
    assert entries[1]["label"] == "Pass"


def test_overlay_payload_round_trip() -> None:
    payload = overlay_hints_payload(
        [
            ActionCoachHint(
                index=1,
                label="attack_red",
                policy_pct=0.72,
                win_pct=0.61,
                is_best=True,
            )
        ]
    )
    parsed = json.loads(payload)
    assert parsed[0]["label"] == "attack_red"
    assert parsed[0]["isBest"] is True
    assert "function" in playwright_update_overlay_script() or "(" in playwright_update_overlay_script()
