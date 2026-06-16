import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flesh_and_blood_rlbridge.cpp_engine_environment import CppEngineEnvironment


class _FakeGS:
    def __init__(self) -> None:
        self.sync_calls: list[tuple[int, list[str]]] = []
        self.registered = False
        self.inited = False

    def register_all_cards(self) -> None:
        self.registered = True

    def init_standard_decks(self) -> None:
        self.inited = True

    def sync_opening_hand(self, player_idx: int, card_ids: list[str]) -> None:
        self.sync_calls.append((player_idx, list(card_ids)))


def test_new_gamestate_applies_opening_hands_from_options() -> None:
    env = object.__new__(CppEngineEnvironment)
    env._fab = type("_Fab", (), {"GameState": _FakeGS})()

    gs = env._new_gamestate(
        {
            "opening_hands": {
                1: ["WTR001", "WTR002"],
                2: ["ARC001"],
            }
        }
    )

    assert gs.registered is True
    assert gs.inited is True
    assert gs.sync_calls == [
        (0, ["WTR001", "WTR002"]),
        (1, ["ARC001"]),
    ]


def test_new_gamestate_skips_empty_opening_hands() -> None:
    env = object.__new__(CppEngineEnvironment)
    env._fab = type("_Fab", (), {"GameState": _FakeGS})()

    gs = env._new_gamestate({"opening_hands": {1: [], 2: ["ARC001"]}})

    assert gs.sync_calls == [(1, ["ARC001"])]
