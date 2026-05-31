#!/usr/bin/env python3
"""Systematically re-parse gap cards and run targeted runtime checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flesh_and_blood_rlbridge import effects
from flesh_and_blood_rlbridge.environment import CombatState, FleshAndBloodEnvironment

CARDS_JSON = ROOT / "src/flesh_and_blood_rlbridge/card_db/unimplemented_cards.json"
EFFECTS_JSON = ROOT / "src/flesh_and_blood_rlbridge/card_db/unimplemented_effects.json"


def load_gap_cards() -> list[dict]:
    path = CARDS_JSON if CARDS_JSON.exists() else EFFECTS_JSON
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("cards", [])


def filler_deck(*extra: str, size: int = 36) -> list[str]:
    base = ["sink_below_red", "sink_below_yellow", "sink_below_blue", "razor_reflex_red"]
    deck = list(extra)
    i = 0
    while len(deck) < size:
        deck.append(base[i % len(base)])
        i += 1
    return deck[:size]


def make_env(
    agent_hero: str,
    opp_hero: str = "hero_dorinthea_ironsong",
    agent_deck: list[str] | None = None,
    seed: int = 1,
) -> FleshAndBloodEnvironment:
    env = FleshAndBloodEnvironment(
        seed=seed,
        agent_hero_id=agent_hero,
        opponent_hero_id=opp_hero,
        two_phase_deckbuild=False,
        self_play=True,
        opponent_type="random",
        deck_size=36,
    )
    env.reset()
    env._start_match(
        agent_hero_id=agent_hero,
        opponent_hero_id=opp_hero,
        agent_deck_style="balanced",
        opponent_deck_style="balanced",
        agent_deck_ids=agent_deck,
        opponent_deck_ids=filler_deck(),
    )
    return env


def rescan_card(card_row: dict) -> tuple[str, list[dict], list[dict]]:
    """Return (status, unimplemented, implemented) from fresh parse."""
    text = card_row.get("text") or ""
    keywords = list(card_row.get("keywords") or [])
    card_types = tuple(card_row.get("card_types") or ())
    parsed = effects.parse_all_interactions(text, keywords=keywords, card_types=card_types)
    trigs = (*parsed["triggers"], *parsed["keyword_triggers"])
    abilities = parsed["abilities"]
    mods = parsed["modifiers"]
    play_costs = parsed.get("play_costs") or ()
    action_damage, action_arcane = parsed["action_damage"]

    impl: list[dict] = []
    unimpl: list[dict] = []
    if action_damage > 0:
        kind = "arcane_damage" if action_arcane else "damage"
        impl.append({"source": "action", "kind": kind, "raw": f"deal {action_damage} damage"})
    for pc in play_costs:
        row = {"source": "play_cost", "kind": pc.kind, "raw": pc.raw}
        (impl if pc.implemented else unimpl).append(row)
    for t in trigs:
        row = {"source": "trigger", "when": t.when, "kind": t.effect.kind, "raw": t.effect.raw}
        (impl if t.effect.implemented else unimpl).append(row)
    for a in abilities:
        row = {"source": "activated", "kind": a.effect.kind, "raw": a.effect.raw, "ability": a.raw[:120]}
        (impl if a.implemented else unimpl).append(row)

    has_any_parse = bool(trigs or abilities or mods or play_costs)
    if unimpl and not impl and not mods:
        status = "unparsed" if not has_any_parse else "unimplemented_only"
    elif unimpl:
        status = "partial"
    elif has_any_parse or mods:
        status = "implemented"
    else:
        status = "unparsed"
    return status, unimpl, impl


def test_lay_to_rest() -> list[str]:
    issues: list[str] = []
    env = make_env("hero_arakni_web_of_deceit", agent_deck=filler_deck("lay-to-rest-1"))
    p0, p1 = env._players[0], env._players[1]
    p1.banished = ["sink_below_yellow"]
    card = env._cards["lay-to-rest-1"]
    eff = next(
        e
        for e in effects.parse_triggers(card.text)
        if e.when == "on_hit" and e.effect.kind == "turn_banished_face"
    ).effect
    env._apply_effect(0, card, eff, None, 0)
    if "sink_below_yellow" not in p1.face_down_banished:
        issues.append("Lay to Rest: should turn opponent banished card face-down")
    return issues


def test_sunken_treasure_defend() -> list[str]:
    issues: list[str] = []
    env = make_env("hero_riptide_lurker_of_the_deep", agent_deck=filler_deck("sunken_treasure_blue"))
    p0, p1 = env._players[0], env._players[1]
    p1.discard = ["sink_below_yellow"]
    card = env._cards.get("sunken_treasure_blue") or env._cards.get("sunken-treasure-3")
    if card is None:
        return ["Sunken Treasure: card id not found in db"]
    eff = next(
        e
        for e in effects.parse_triggers(card.text)
        if e.when == "on_defend" and e.effect.kind == "turn_banished_face"
    ).effect
    gold_before = p0.gold + p0.tokens.get("gold", 0)
    env._apply_effect(0, card, eff, None, 0)
    if "sink_below_yellow" not in p1.face_down_discard:
        issues.append("Sunken Treasure: should turn gy card face-down")
    gold_after = p0.gold + p0.tokens.get("gold", 0)
    if gold_after <= gold_before:
        issues.append("Sunken Treasure: yellow gy card should create Gold token")
    return issues


def test_drag_down() -> list[str]:
    issues: list[str] = []
    env = make_env("hero_dorinthea_ironsong", agent_deck=filler_deck("drag-down-1"))
    p0 = env._players[0]
    card = env._cards["drag-down-1"]
    combat = CombatState(
        attacker=1,
        defender=0,
        attack_card_id="wrecking_ball_red",
        attack_power=6,
        blocks=[],
    )
    env._pending_combat = combat
    p0.hand = ["drag-down-1"]
    p0.resources = 2
    env._phase = "defense_reaction"
    env._active_player = 0
    env._do_defense_reaction(0, 0, 0)
    if combat.opposing_power_mod >= 0:
        issues.append(
            f"Drag Down: attack should get -3 power on chain; opposing_power_mod={combat.opposing_power_mod}"
        )
    return issues


def test_buzzsaw_trap() -> list[str]:
    issues: list[str] = []
    env = make_env(
        "hero_riptide_lurker_of_the_deep",
        agent_deck=filler_deck("buzzsaw-trap-3"),
    )
    p0 = env._players[0]
    p0.arsenal = ["buzzsaw-trap-3"]
    combat = CombatState(
        attacker=1,
        defender=0,
        attack_card_id="wrecking_ball_red",
        attack_power=8,
        blocks=[],
    )
    env._pending_combat = combat
    env._phase = "defense"
    env._do_trap_defend(0, 0)
    if not combat.block_power_gain:
        issues.append("Buzzsaw Trap: should block power gain when attack power > base")
    return issues


RUNTIME_TESTS = [
    ("Lay to Rest (banished face-down)", test_lay_to_rest),
    ("Sunken Treasure (gy face-down + gold)", test_sunken_treasure_defend),
    ("Drag Down (-power on chain)", test_drag_down),
    ("Buzzsaw Trap (block power gain)", test_buzzsaw_trap),
]


def main() -> int:
    gap_cards = load_gap_cards()
    print(f"Loaded {len(gap_cards)} gap cards from catalog")

    rescan_fixed = 0
    still_gaps = 0
    for row in gap_cards:
        old_status = row.get("status")
        new_status, unimpl, impl = rescan_card(row)
        if new_status == "implemented" and old_status != "implemented":
            rescan_fixed += 1
        elif new_status != "implemented":
            still_gaps += 1

    print(f"Re-scan: {rescan_fixed} cards now fully implemented (of {len(gap_cards)} in catalog)")
    print(f"Re-scan: {still_gaps} cards still have gaps when re-parsed today")

    all_issues: list[str] = []
    for name, fn in RUNTIME_TESTS:
        try:
            found = fn()
        except Exception as exc:
            found = [f"EXCEPTION {type(exc).__name__}: {exc}"]
        if found:
            print(f"FAIL {name}:")
            for item in found:
                print(f"  - {item}")
            all_issues.extend(found)
        else:
            print(f"OK   {name}")

    return 1 if all_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
