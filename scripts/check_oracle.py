#!/usr/bin/env python3
"""Run a sequence of games and validate every combat resolution against Talishar rules.

Usage
-----
::

    # Offline rule-book check (no Docker required):
    python scripts/check_oracle.py

    # With a local Talishar Docker instance:
    TALISHAR_URL=http://localhost python scripts/check_oracle.py

    # More games / verbose output:
    python scripts/check_oracle.py --games 20 --verbose

Options
-------
--games N       Number of complete games to play (default: 5)
--turns N       Maximum turns per game (default: 30)
--seed N        RNG seed for reproducibility (default: 42)
--hero A:B      Hero IDs to use, colon-separated (default: auto)
--verbose       Print every combat result, not just mismatches
--stop-on-fail  Abort as soon as the first mismatch is found
"""

from __future__ import annotations

import argparse
import sys
import os

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flesh_and_blood_rlbridge.gameplay_environment import FleshAndBloodGameplayEnvironment
from flesh_and_blood_rlbridge.talishar_oracle import (
    CombatSnapshot,
    OracleHook,
    TalisharOracle,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", type=int, default=5, metavar="N")
    p.add_argument("--turns", type=int, default=30, metavar="N")
    p.add_argument("--seed", type=int, default=42, metavar="N")
    p.add_argument("--hero", type=str, default="", metavar="A:B",
                   help="e.g. hero_dorinthea_ironsong:hero_rhinar_reckless_rampage")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--stop-on-fail", action="store_true")
    return p.parse_args()


def run_game(
    game_idx: int,
    *,
    seed: int,
    max_turns: int,
    agent_hero_id: str,
    opponent_hero_id: str,
    hook: OracleHook,
    verbose: bool,
    stop_on_fail: bool,
) -> tuple[int, int]:
    """Play one game and return (checks, mismatches)."""
    env = FleshAndBloodGameplayEnvironment(
        seed=seed + game_idx,
        agent_hero_id=agent_hero_id,
        opponent_hero_id=opponent_hero_id,
        max_turns=max_turns,
        opponent_type="random",
    )
    env.reset()
    hook.attach(env)  # registers on_combat_resolve callback
    checks_before = len(hook.results)

    # Use a separate RNG so we don't disturb the environment's internal stream.
    action_rng = __import__("random").Random(seed + game_idx + 9999)
    done = False
    while not done:
        legal = env._legal_actions()
        action = action_rng.choice(legal) if legal else "pass"
        step_result = env.step(action)
        done = step_result.terminated or step_result.truncated

    checks = len(hook.results) - checks_before
    mismatches = 0
    for r in hook.results[checks_before:]:
        if not r.match:
            mismatches += 1
            print(f"[game {game_idx + 1}] MISMATCH")
            print(r.report())
            if stop_on_fail:
                return checks, mismatches
        elif verbose:
            print(
                f"[game {game_idx + 1}] OK  "
                f"power={r.actual_total_power}  block={r.actual_total_power - r.actual_raw_damage}  "
                f"raw={r.actual_raw_damage}  dmg={r.actual_mitigated}"
            )

    return checks, mismatches


def main() -> None:
    args = parse_args()

    # Determine heroes
    if args.hero:
        parts = args.hero.split(":")
        agent_hero = parts[0].strip()
        opponent_hero = parts[1].strip() if len(parts) > 1 else parts[0].strip()
    else:
        # Use heroes that appear in a broad range of scenarios
        agent_hero = "hero_dorinthea_ironsong"
        opponent_hero = "hero_rhinar_reckless_rampage"

    oracle = TalisharOracle()
    hook = OracleHook(oracle)

    if oracle.server_available:
        print(f"Talishar server detected at {oracle._client.base_url}")
    else:
        print("No Talishar server — using local rule-book validation only.")

    total_checks = 0
    total_mismatches = 0

    for i in range(args.games):
        checks, mismatches = run_game(
            i,
            seed=args.seed,
            max_turns=args.turns,
            agent_hero_id=agent_hero,
            opponent_hero_id=opponent_hero,
            hook=hook,
            verbose=args.verbose,
            stop_on_fail=args.stop_on_fail,
        )
        total_checks += checks
        total_mismatches += mismatches
        print(f"Game {i + 1}/{args.games}: {checks} combats checked, {mismatches} mismatches")
        if mismatches and args.stop_on_fail:
            break

    print()
    print(hook.summary())
    print(f"\nTotal: {total_checks} combat checks, {total_mismatches} mismatches")

    if total_mismatches > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
