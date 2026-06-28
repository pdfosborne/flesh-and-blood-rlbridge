#!/usr/bin/env python3
"""Benchmark Talishar HTTP legacy vs fast backend vs optional C++."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

_SCRIPTS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_ROOT))
import _bootstrap  # noqa: E402

_bootstrap.configure_paths()

from flesh_and_blood_rlbridge.cpp_engine_environment import (  # noqa: E402
    explain_cpp_engine_unavailable,
)
from flesh_and_blood_rlbridge.talishar_engine_environment import TalisharEngineEnvironment
from flesh_and_blood_rlbridge.talishar_fast_client import DEFAULT_TALISHAR_URL
from runtime_defaults import normalize_rollout_mode  # noqa: E402

_REPO_ROOT = _SCRIPTS_ROOT.parent
_CPP_CACHE_DIR = _REPO_ROOT / "results" / "cpp_engines"


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


def _build_cpp_engine(*, deck1: str, deck2: str) -> int:
    import subprocess

    cmd = [
        sys.executable,
        str(_SCRIPTS_ROOT / "cpp" / "build_cpp_engine_for_matchup.py"),
        "--deck1",
        deck1,
        "--deck2",
        deck2,
        "--no-server",
    ]
    print(f"Building C++ engine for {deck1} vs {deck2}...")
    completed = subprocess.run(cmd, cwd=str(_REPO_ROOT), check=False)
    return int(completed.returncode)


def _bench_rollout_modes(
    *,
    base_url: str,
    deck1: str,
    deck2: str,
    workers: int,
    max_steps: int,
    steps_per_episode: int,
) -> None:
    from train_dual_agent_common import (  # noqa: PLC0415
        _run_parallel_fast_episode_batch,
    )
    from rl_agents.ppo import PPOAgent  # noqa: PLC0415
    from flesh_and_blood_rlbridge.player_observation import (  # noqa: PLC0415
        ACTION_CAPACITY,
        PLAYER_OBS_DIM,
    )

    envs = [
        TalisharEngineEnvironment(
            base_url=base_url,
            local_deck_name=deck1,
            opponent_deck_name=deck2,
            self_play=True,
            max_turns=max_steps,
            talishar_backend="fast",
            rl_training_mode=True,
        )
        for _ in range(workers)
    ]
    p1 = PPOAgent(n_actions=ACTION_CAPACITY, obs_dim=PLAYER_OBS_DIM)
    p2 = PPOAgent(n_actions=ACTION_CAPACITY, obs_dim=PLAYER_OBS_DIM)
    p1._init_nets(PLAYER_OBS_DIM)
    p2._init_nets(PLAYER_OBS_DIM)

    modes = ["batched", "threaded_episodes", "batched_concurrent"]
    print(f"\nRollout modes — {workers} worker(s), up to {steps_per_episode} steps/ep")
    for mode in modes:
        total_steps = 0
        t0 = time.perf_counter()
        try:
            results = _run_parallel_fast_episode_batch(
                envs[:workers],
                p1,
                p2,
                max_steps=steps_per_episode,
                warmup=True,
                episode_indices=list(range(workers)),
                seed_base=0,
                rollout_mode=normalize_rollout_mode(mode),
                max_workers=workers,
            )
            total_steps = sum(int(r.get("steps", 0) or 0) for r in results)
        except Exception as exc:
            print(f"  {mode:20s}  ERROR: {exc}")
            continue
        elapsed = time.perf_counter() - t0
        if total_steps <= 0:
            print(f"  {mode:20s}  FAILED")
            continue
        print(
            f"  {mode:20s}  {total_steps / elapsed:6.1f} steps/s  "
            f"({elapsed / max(1, len(results)) * 1000:.0f} ms/ep)"
        )
    for env in envs:
        try:
            env.close()
        except Exception:
            pass


def _bench_rlstep_profile(
    *,
    base_url: str,
    deck1: str,
    deck2: str,
    steps: int,
) -> int:
    import os
    import statistics

    os.environ["FAB_RLSTEP_PROFILE"] = "1"
    env = TalisharEngineEnvironment(
        base_url=base_url,
        local_deck_name=deck1,
        opponent_deck_name=deck2,
        self_play=True,
        max_turns=500,
        talishar_backend="fast",
        rl_training_mode=True,
    )
    if not env.supports_fast_training:
        print("RLStep profile requires fast backend with RLStep overlay")
        return 1
    timings: dict[str, list[float]] = {}
    try:
        env.fast_reset()
        client = env._fast_client
        assert client is not None
        for _ in range(steps):
            payload = {
                "gameName": env._game_name or "",
                "playerID": env._acting_player_id,
                "authKey": env._auth_key_for(env._acting_player_id),
                "mode": 99,
                "trainingMode": True,
                "slimResponse": True,
                "profileTimings": True,
            }
            resp = client.post_rlstep(payload)
            if not resp.get("success"):
                print(f"RLStep failed: {resp}")
                return 1
            section = resp.get("timingsMs") or {}
            if isinstance(section, dict):
                for key, value in section.items():
                    try:
                        timings.setdefault(str(key), []).append(float(value))
                    except (TypeError, ValueError):
                        pass
            env._apply_rlstep_states(resp)
    finally:
        env.close()
    if not timings:
        print("No timingsMs returned — restart web-server after overlay deploy")
        return 1
    print(f"RLStep profile — {deck1} vs {deck2}  ({steps} pass steps)")
    for key in sorted(timings):
        values = timings[key]
        med = statistics.median(values)
        print(f"  {key:12s}  median {med:6.1f} ms")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_TALISHAR_URL)
    parser.add_argument("--deck1", default="Ira")
    parser.add_argument("--deck2", default="Ira")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument(
        "--rollout-mode",
        default=None,
        choices=["batched", "threaded_episodes", "batched_concurrent"],
        help="When set with --workers, benchmark parallel rollout modes only",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--episode-steps", type=int, default=30)
    parser.add_argument(
        "--build-cpp",
        action="store_true",
        help="Build the C++ engine for --deck1 vs --deck2 before benchmarking",
    )
    parser.add_argument(
        "--profile-rlstep",
        action="store_true",
        help="Print median RLStep PHP timingsMs (requires training overlay)",
    )
    args = parser.parse_args()

    if args.profile_rlstep:
        return _bench_rlstep_profile(
            base_url=args.base_url,
            deck1=args.deck1,
            deck2=args.deck2,
            steps=args.steps,
        )

    if args.build_cpp:
        rc = _build_cpp_engine(deck1=args.deck1, deck2=args.deck2)
        if rc != 0:
            print(f"C++ engine build failed (exit {rc})\n")

    if args.workers > 0 or args.rollout_mode:
        workers = max(1, args.workers or 4)
        _bench_rollout_modes(
            base_url=args.base_url,
            deck1=args.deck1,
            deck2=args.deck2,
            workers=workers,
            max_steps=max(args.steps, args.episode_steps),
            steps_per_episode=args.episode_steps,
        )
        return 0

    configs = [
        ("http (legacy)", {"talishar_backend": "http", "use_cpp_engine": False}),
        ("fast (Talishar)", {"talishar_backend": "fast", "use_cpp_engine": False}),
        (
            "cpp (opt-in)",
            {
                "talishar_backend": "fast",
                "use_cpp_engine": True,
                "require_cpp": True,
            },
        ),
    ]

    print(f"Benchmark — {args.deck1} vs {args.deck2}  ({args.steps} steps)")
    print(f"Talishar URL: {args.base_url}\n")

    for label, kw in configs:
        require_cpp = bool(kw.pop("require_cpp", False))
        try:
            with warnings.catch_warnings():
                if require_cpp:
                    warnings.simplefilter("ignore", RuntimeWarning)
                env = TalisharEngineEnvironment(
                base_url=args.base_url,
                local_deck_name=args.deck1,
                opponent_deck_name=args.deck2,
                self_play=True,
                max_turns=500,
                cpp_obs_alignment=False,
                cpp_engine_cache_dir=str(_CPP_CACHE_DIR),
                **kw,
            )
            if require_cpp and not getattr(env, "_using_cpp", False):
                reason = explain_cpp_engine_unavailable(
                    args.deck1,
                    args.deck2,
                    _CPP_CACHE_DIR,
                )
                print(f"  {label:20s}  UNAVAILABLE: {reason}")
                env.close()
                continue
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
