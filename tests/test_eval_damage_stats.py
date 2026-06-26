from scripts.training.eval_damage_stats import (
    EvalDamageAccumulator,
    merge_damage_breakdowns,
)


def _event(
    *,
    log_lines: list[str] | None = None,
    before_hp: tuple[int, int] = (20, 20),
    after_hp: tuple[int, int] = (20, 20),
    acting: int = 1,
    action_class: str = "other",
    card_id: str = "",
    zone: str = "hand",
) -> dict:
    return {
        "before": {
            "acting_player_id": acting,
            "player_health": before_hp[0] if acting == 1 else before_hp[1],
            "opponent_health": before_hp[1] if acting == 1 else before_hp[0],
            "turn_no": 1,
        },
        "after": {
            "acting_player_id": acting,
            "player_health": after_hp[0] if acting == 1 else after_hp[1],
            "opponent_health": after_hp[1] if acting == 1 else after_hp[0],
            "turn_no": 1,
        },
        "action": {
            "card_id": card_id,
            "action_code": 27,
            "label": card_id,
            "zone": zone,
        },
        "action_class": action_class,
        "combat_log_delta": log_lines or [],
    }


def test_player_took_damage_attributes_to_last_attack() -> None:
    acc = EvalDamageAccumulator(deck_card_ids={"sink_below_surge_red"})
    acc.ingest_trace(
        [
            _event(acting=1, action_class="attack", card_id="sink_below_surge_red"),
            _event(log_lines=["Player 2 took 4 damage"]),
        ]
    )
    result = acc.to_dict()
    assert result["total_dealt"] == 4
    assert result["cards_dealt"] == [
        {"card_id": "sink_below_surge_red", "damage": 4, "avg_damage": 4.0}
    ]


def test_opponent_attack_counted_as_damage_taken() -> None:
    acc = EvalDamageAccumulator()
    acc.ingest_trace(
        [
            _event(acting=2, action_class="attack", card_id="enigma_chimera_red"),
            _event(log_lines=["Player 1 took 3 damage"]),
        ]
    )
    result = acc.to_dict()
    assert result["total_taken"] == 3
    assert result["cards_taken_from"] == [
        {"card_id": "enigma_chimera_red", "damage": 3, "avg_damage": 3.0}
    ]


def test_merge_damage_breakdowns_adds_unattributed_bucket() -> None:
    merged = merge_damage_breakdowns(
        [
            {
                "total_dealt": 10,
                "total_taken": 4,
                "cards_dealt": [],
                "cards_taken_from": [],
            }
        ]
    )
    assert merged["cards_dealt"] == [
        {"card_id": "(unattributed)", "damage": 10, "avg_damage": 10.0}
    ]
    assert merged["cards_taken_from"] == [
        {"card_id": "(unattributed)", "damage": 4, "avg_damage": 4.0}
    ]


def test_cpp_hp_log_line_attributes_to_last_card() -> None:
    acc = EvalDamageAccumulator(deck_card_ids={"lightning_surge_red"})
    acc.ingest_trace(
        [
            _event(
                acting=1,
                action_class="other",
                card_id="lightning_surge_red",
                zone="hand",
            ),
            _event(log_lines=["HP P1 20->20 | P2 20->16"]),
        ]
    )
    result = acc.to_dict()
    assert result["total_dealt"] == 4
    assert result["cards_dealt"] == [
        {"card_id": "lightning_surge_red", "damage": 4, "avg_damage": 4.0}
    ]
    merged = merge_damage_breakdowns(
        [
            {
                "total_dealt": 4,
                "total_taken": 2,
                "cards_dealt": [{"card_id": "a", "damage": 4}],
                "cards_taken_from": [{"card_id": "b", "damage": 2}],
            },
            {
                "total_dealt": 6,
                "total_taken": 1,
                "cards_dealt": [{"card_id": "a", "damage": 3}, {"card_id": "c", "damage": 3}],
                "cards_taken_from": [{"card_id": "b", "damage": 1}],
            },
        ]
    )
    assert merged["episodes"] == 2
    assert merged["total_dealt"] == 10
    assert merged["total_taken"] == 3
    assert merged["cards_dealt"][0] == {"card_id": "a", "damage": 7, "avg_damage": 3.5}
    assert merged["cards_taken_from"][0] == {"card_id": "b", "damage": 3, "avg_damage": 1.5}
