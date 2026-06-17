#!/usr/bin/env python
"""Manual test script for TalisharEngineEnvironment.

Usage:
    TALISHAR_URL=http://localhost:8080/game python -m pytest tests/test_talishar_env_smoke.py

The script plays one full episode with random actions and prints a summary.
"""

import os
import sys

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

if "TALISHAR_URL" not in os.environ:
    os.environ["TALISHAR_URL"] = "http://localhost:8080/game"

from flesh_and_blood_rlbridge import FLESH_AND_BLOOD_TALISHAR_V0  # noqa: E402

print(f"Connecting to {os.environ['TALISHAR_URL']} ...")
env = FLESH_AND_BLOOD_TALISHAR_V0.create(render_mode="ansi")

# ── reset ─────────────────────────────────────────────────────────────────────
result = env.reset()
state = result.observation

print("\n=== Episode start ===")
print(f"Initial state = {state}")
print(env.render().text)

# ── play until done ───────────────────────────────────────────────────────────
total_reward = 0.0
step_no = 0

while True:
    action = env.sample_action()
    step = env.step(action)
    state = step.observation
    total_reward += step.reward
    step_no += 1

    if step_no % 5 == 0 or step.terminated or step.truncated:
        info = step.info
        print(
            f"\n=== Step {step_no} ===\n"
            f"State = {state}  \n ----------------------- \n"
            f"Action={action:<4}  reward={step.reward:+.3f}"
            f"  P1 HP={info['player_hp']}  Opp HP={info['opponent_hp']}"
            f"  turn={info['turn']}"
        )
        print(env.render().text)

    if step.terminated or step.truncated:
        outcome = "terminated" if step.terminated else "truncated"
        print(f"\n=== Episode {outcome} after {step_no} steps ===")
        print(f"Total reward: {total_reward:+.3f}")
        print(env.render().text)
        break

env.close()
print("\nDone.")
