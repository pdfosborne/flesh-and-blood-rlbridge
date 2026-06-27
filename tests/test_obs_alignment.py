import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.obs_alignment import (
    align_observation_for_cpp_training,
    merge_talishar_raw_state,
    observation_vectors_aligned,
)
from flesh_and_blood_rlbridge.player_observation import (
    COMBAT_CHAIN_END,
    COMBAT_SCALAR_OFF,
    PLAYER_OBS_DIM,
)


def test_align_zeros_combat_block() -> None:
    vec = np.ones(PLAYER_OBS_DIM, dtype=np.float64)
    aligned = align_observation_for_cpp_training(vec)
    assert float(aligned[COMBAT_SCALAR_OFF:COMBAT_CHAIN_END].sum()) == 0.0


def test_aligned_vectors_match_when_only_combat_differs() -> None:
    tal = np.zeros(PLAYER_OBS_DIM, dtype=np.float64)
    cpp = np.zeros(PLAYER_OBS_DIM, dtype=np.float64)
    tal[COMBAT_SCALAR_OFF] = 0.5
    ok, _ = observation_vectors_aligned(tal, cpp, atol=0.01)
    assert ok


def test_merge_talishar_raw_state_prefers_http_zones() -> None:
    cpp = {"playerEquipment": [{"cardNumber": "cpp_card"}]}
    tal = {"playerEquipment": [{"cardNumber": "tal_card"}], "combatChain": [{"cardID": "x"}]}
    merged = merge_talishar_raw_state(cpp, tal)
    assert merged["playerEquipment"][0]["cardNumber"] == "tal_card"
    assert merged["combatChain"][0]["cardID"] == "x"
