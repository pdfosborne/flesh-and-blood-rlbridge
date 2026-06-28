#!/usr/bin/env python3
"""Benchmark Talishar HTTP legacy vs fast backend vs optional C++."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flesh_and_blood_rlbridge.talishar_engine_environment import TalisharEngineEnvironment
from flesh_and_blood_rlbridge.talishar_fast_client import DEFAULT_TALISHAR_URL


def _bench_steps(env: TalisharEngineEnvironment, *, steps: int) -> tuple[float, int]:
    env.reset()
    t0 = time.perf_counter()
    completed = 0
    for _ in range(steps):
        try:
            if env.supports_fast_training:
                state = env.fast_step_index(0)
                if state.get("terminated") or state.get("truncated"):
                    env.fast_reset()
            else:
                out = env.step("0")
                if out.terminated or out.truncated:
                    env.reset()
            completed += 1
        except Exception as exc:
            print(f"  step failed: {exc}")
            break
    elapsed = time.perf_counter() - t0
    return elapsed, completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_TALISHAR_URL)
    parser.add_argument("--deck1", default="Ira")
    parser.add_argument("--deck2", default="Ira")
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    configs = [
        ("http (legacy)", {"talishar_backend": "http", "use_cpp_engine": False}),
        ("fast (Talishar)", {"talishar_backend": "fast", "use_cpp_engine": False}),
        ("cpp (opt-in)", {"talishar_backend": "auto", "use_cpp_engine": True}),
    ]

    print(f"Benchmark — {args.deck1} vs {args.deck2}  ({args.steps} steps)")
    print(f"Talishar URL: {args.base_url}\n")

    for label, kw in configs:
        try:
            env = TalisharEngineEnvironment(
                base_url=args.base_url,
                local_deck_name=args.deck1,
                opponent_deck_name=args.deck2,
                self_play=True,
                max_turns=500,
                cpp_obs_alignment=False,
                **kw,
            )
            backend = getattr(env, "talishar_backend", "?")
            elapsed, completed = _bench_steps(env, steps=args.steps)
            if completed == 0:
                print(f"  {label:20s}  backend={backend:8s}  FAILED")
                continue
            ms = elapsed / completed * 1000
            print(
                f"  {label:20s}  backend={backend:8s}  "
                f"{completed / elapsed:6.1f} steps/s  ({ms:6.1f} ms/step)"
            )
        except Exception as exc:
            print(f"  {label:20s}  ERROR: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
