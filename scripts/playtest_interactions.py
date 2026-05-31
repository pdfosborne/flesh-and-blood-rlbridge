#!/usr/bin/env python3
"""Targeted interaction checks against en-fab-cr.txt expectations."""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flesh_and_blood_rlbridge import effects
from flesh_and_blood_rlbridge.environment import CombatState, FleshAndBloodEnvironment


def filler_deck(*extra: str, size: int = 36) -> list[str]:
    base = [
        "sink_below_red",
        "sink_below_yellow",
        "sink_below_blue",
        "razor_reflex_red",
    ]
    deck = list(extra)
    i = 0
    while len(deck) < size:
        deck.append(base[i % len(base)])
        i += 1
    return deck[:size]


def make_env(
    agent_hero: str,
    opp_hero: str,
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


def test_reckless_swing_defreact() -> list[str]:
    """CR 175: discard random as additional cost, then conditional damage to attacking hero."""
    issues: list[str] = []
    env = make_env(
        "hero_dorinthea_ironsong",
        "hero_rhinar_reckless_rampage",
        filler_deck("reckless-swing-3"),
    )
    p0, p1 = env._players[0], env._players[1]
    p0.hand = ["reckless-swing-3", "wrecking_ball_red"]
    env._pending_combat = CombatState(
        attacker=1,
        defender=0,
        attack_card_id="wrecking_ball_red",
        attack_power=6,
        blocks=[],
    )
    env._phase = "defense_reaction"
    env._active_player = 0
    hand_before = len(p0.hand)
    opp_life_before = p1.life
    env._do_defense_reaction(0, 0, 0)
    if len(p0.hand) >= hand_before:
        issues.append(
            "Reckless Swing: defense reaction did not discard a random card (CR 175 leading cost)"
        )
    if not p0.class_state.get("DiscardedPower6ThisTurn"):
        issues.append("Reckless Swing: 6+ power discard not recorded")
    if p1.life >= opp_life_before:
        issues.append(
            f"Reckless Swing: attacking hero should take 2 damage; life {opp_life_before}->{p1.life}; "
            f"event={env._last_event!r}"
        )
    return issues


def test_system_failure() -> list[str]:
    issues: list[str] = []
    env = make_env(
        "hero_kassai_cintari_sellsword",
        "hero_dorinthea_ironsong",
        filler_deck("system-failure-2"),
    )
    p0, p1 = env._players[0], env._players[1]
    eq_id = p1.equipment[0]
    env._add_permanent_counter(1, eq_id, "steam", 3)
    card = env._cards["system-failure-2"]
    p0.resources = 2
    p0.action_points = 1
    ctrl_life_before = p1.life
    env._phase = "action"
    env._active_player = 0
    env._resolve_direct_effects(0, card, 0, False, False, False, False, (), 0)
    removed = p0.class_state.get("LastSteamRemovedCount", 0)
    if removed < 2:
        issues.append(f"System Failure: expected 2+ steam removed, got {removed}")
    if removed >= 2 and p1.life >= ctrl_life_before:
        issues.append("System Failure: controller should take 2 damage when 2+ steam removed")
    if env._permanent_counters(1, eq_id).get("steam", 0) > 0:
        issues.append("System Failure: steam counters should be cleared from target")
    return issues


def test_riptide_trap_trigger() -> list[str]:
    """CR 8.3.28 ambush + Riptide: trap trigger deals 1 to attacking hero."""
    issues: list[str] = []
    env = make_env(
        "hero_riptide_lurker_of_the_deep",
        "hero_dorinthea_ironsong",
        filler_deck("tripwire-trap-1"),
    )
    p0, p1 = env._players[0], env._players[1]
    p0.arsenal = ["tripwire-trap-1"]
    env._pending_combat = CombatState(
        attacker=1,
        defender=0,
        attack_card_id="wrecking_ball_red",
        attack_power=6,
        blocks=[],
    )
    env._phase = "defense"
    env._active_player = 0
    opp_life_before = p1.life
    env._do_trap_defend(0, 0)
    if p0.arsenal:
        issues.append("Trap defend should empty arsenal")
    if p1.life != opp_life_before - 1:
        issues.append(
            f"Riptide trap trigger: attacking hero should take 1 damage; "
            f"life {opp_life_before}->{p1.life}; event={env._last_event!r}"
        )
    if not env._pending_combat.hit_effects_blocked:
        issues.append("Tripwire Trap: should block hit effects on defend")
    return issues


def test_bloodrot_trap_reaction() -> list[str]:
    issues: list[str] = []
    env = make_env(
        "hero_riptide_lurker_of_the_deep",
        "hero_dorinthea_ironsong",
        filler_deck("bloodrot-trap-1"),
    )
    p0, p1 = env._players[0], env._players[1]
    p0.arsenal = ["bloodrot-trap-1"]
    env._pending_combat = CombatState(
        attacker=1,
        defender=0,
        attack_card_id="wrecking_ball_red",
        attack_power=6,
        blocks=[],
        attack_reactions_played=1,
    )
    env._phase = "defense"
    env._active_player = 0
    env._do_trap_defend(0, 0)
    token_key = next(
        (k for k in p1.tokens if "bloodrot" in k.lower() or "pox" in k.lower()),
        None,
    )
    if token_key is None or p1.tokens.get(token_key, 0) < 1:
        issues.append(
            f"Bloodrot Trap: should create Bloodrot Pox when attacker played a reaction; "
            f"tokens={p1.tokens!r}"
        )
    return issues


def test_tripwire_hand_blocked() -> list[str]:
    """Tripwire can only be played from arsenal — not from hand during defreact step."""
    issues: list[str] = []
    env = make_env(
        "hero_riptide_lurker_of_the_deep",
        "hero_dorinthea_ironsong",
        filler_deck("tripwire-trap-1"),
    )
    p0 = env._players[0]
    p0.hand = ["tripwire-trap-1"]
    env._pending_combat = CombatState(
        attacker=1,
        defender=0,
        attack_card_id="wrecking_ball_red",
        attack_power=6,
        blocks=[],
    )
    env._phase = "defense_reaction"
    env._active_player = 0
    if env._legal_defense_reactions(0):
        issues.append("Tripwire Trap: arsenal-only trap should not be legal from hand")
    return issues


def test_riptide_stash() -> list[str]:
    issues: list[str] = []
    env = make_env(
        "hero_riptide_lurker_of_the_deep",
        "hero_dorinthea_ironsong",
        filler_deck("wrecking_ball_red"),
    )
    p0 = env._players[0]
    p0.hand = ["wrecking_ball_red", "sink_below_red"]
    p0.arsenal = []
    p0.resources = 2
    p0.action_points = 1
    env._phase = "action"
    env._active_player = 0
    env._record_play_counters(0, env._cards["wrecking_ball_red"], True, False, False, None, from_hand=True)
    if not env._pending_optionals:
        issues.append("Riptide: play from hand should queue optional arsenal stash")
    return issues


def test_verdance_gold_arcane() -> list[str]:
    issues: list[str] = []
    env = make_env(
        "hero_verdance_thorn_of_the_rose",
        "hero_dorinthea_ironsong",
    )
    p0, p1 = env._players[0], env._players[1]
    p0.banished = ["amulet_of_earth_blue"] * 8
    env._phase = "action"
    env._active_player = 0
    opp_life = p1.life
    proxy = env._cards[p0.weapon_id or p0.equipment[0]]
    env._apply_effect(0, proxy, effects.Effect("gain_gold", 1, target="self"), None, 0)
    if not env._pending_optionals:
        issues.append(
            f"Verdance: with {env._count_earth_in_banish(0)} earth banished, "
            "gain gold should offer optional arcane damage"
        )
    elif p1.life < opp_life:
        pass  # auto-resolved optional — acceptable in auto sim
    return issues


def run_random_games(n: int = 50, seed: int = 100) -> list[str]:
    issues: list[str] = []
    rng = random.Random(seed)
    heroes = [
        "hero_rhinar_reckless_rampage",
        "hero_dorinthea_ironsong",
        "hero_kassai_cintari_sellsword",
        "hero_riptide_lurker_of_the_deep",
        "hero_verdance_thorn_of_the_rose",
        "hero_bravo_showstopper",
        "hero_viserai_rune_blood",
    ]
    notice_re = re.compile(r"not implemented|interaction not implemented", re.I)
    for i in range(n):
        h0, h1 = rng.sample(heroes, 2)
        env = make_env(h0, h1, seed=rng.randint(0, 10**9))
        for _ in range(300):
            if env._is_terminal():
                break
            legal = env._legal_actions()
            if not legal:
                break
            env.step(rng.choice(legal))
            for msg in env._notices:
                if notice_re.search(msg) and msg not in issues:
                    issues.append(f"game {i}: {msg}")
    return issues


def main() -> int:
    all_issues: list[str] = []
    for name, fn in [
        ("Reckless Swing (defense reaction)", test_reckless_swing_defreact),
        ("System Failure", test_system_failure),
        ("Riptide stash", test_riptide_stash),
        ("Riptide trap trigger", test_riptide_trap_trigger),
        ("Bloodrot trap + reaction", test_bloodrot_trap_reaction),
        ("Tripwire arsenal-only", test_tripwire_hand_blocked),
        ("Verdance gold arcane", test_verdance_gold_arcane),
    ]:
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

    print("\nRandom games (50):")
    random_issues = run_random_games(50)
    if random_issues:
        for item in random_issues[:15]:
            print(f"  - {item}")
        all_issues.extend(random_issues)
    else:
        print("  No unimplemented-interaction notices")

    return 1 if all_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
