#!/usr/bin/env python3
"""Train briefly on C++ engine, then verify Talishar transfer via parity + policy agreement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from flesh_and_blood_rlbridge.obs_alignment import (  # noqa: E402
    align_observation_for_cpp_training,
    observation_vectors_aligned,
)
from flesh_and_blood_rlbridge.cpp_engine_environment import (  # noqa: E402
    CppEngineEnvironment,
    is_cpp_engine_available,
)
from flesh_and_blood_rlbridge.player_observation import PLAYER_OBS_DIM  # noqa: E402
from flesh_and_blood_rlbridge.talishar_engine_environment import TalisharEngineEnvironment  # noqa: E402
from rl_agents.ppo import PPOAgent  # noqa: E402
from scripts.cpp.check_cpp_vs_talishar_parity import (  # noqa: E402
    _align_cpp_reset_result,
    _build_talishar_reset_snapshot,
    _choose_action,
    _compare_reset,
    _compare_step,
    _cpp_inner_env,
    _opening_hands_from_talishar,
    _hand_playability_from_talishar,
    _reset_talishar_for_parity,
    _talishar_parity_snapshot,
    _talishar_action_descriptor,
    compare_observations,
    run_parity_check,
)
from train_dual_agent_common import Matchup, train_matchup  # noqa: E402


def _obs_vec(env: Any) -> np.ndarray:
    vec = getattr(env, "_last_observation_vec", None)
    if vec is not None:
        return np.asarray(vec, dtype=np.float64)
    info = getattr(env, "_last_info", {}) or {}
    vec = info.get("observation_vec")
    if vec is not None:
        return np.asarray(vec, dtype=np.float64)
    raise RuntimeError("observation vector unavailable")


def _legal_count(env: Any) -> int:
    info = getattr(env, "_last_info", {}) or {}
    return int(info.get("legal_count", 0) or 0)


def _policy_action(agent: PPOAgent, obs_vec: np.ndarray, legal_count: int) -> int:
    logits, _ = agent.predict_policy_value(obs_vec)
    if 0 < legal_count < len(logits):
        masked = np.array(logits, dtype=np.float64, copy=True)
        masked[legal_count:] = -1e9
        logits = masked
    return int(np.argmax(logits))


def _run_talishar_eval(
    agent: PPOAgent,
    *,
    base_url: str,
    game_format: str,
    deck1: str,
    deck2: str,
    episodes: int,
    max_steps: int,
) -> dict[str, Any]:
    from flesh_and_blood_rlbridge.talishar_engine_environment import (  # noqa: PLC0415
        run_talishar_eval_episode,
    )

    env = TalisharEngineEnvironment(
        base_url=base_url,
        game_format=game_format,
        local_deck_name=deck1,
        opponent_deck_name=deck2,
        max_turns=max_steps,
        self_play=True,
        use_cpp_engine=False,
        cpp_obs_alignment=True,
    )
    outcomes: list[str] = []
    steps_list: list[int] = []
    errors: list[str] = []
    try:
        for ep in range(episodes):
            try:
                result = run_talishar_eval_episode(
                    env,
                    agent,
                    max_steps,
                    seed=ep,
                    p2_agent=agent,
                )
                won = result.get("deck_player_won")
                if won is True:
                    outcomes.append("win")
                elif won is False:
                    outcomes.append("loss")
                else:
                    outcomes.append("draw_or_timeout")
                steps_list.append(int(result.get("steps", 0) or 0))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"ep{ep}: {exc}")
                outcomes.append("error")
    finally:
        env.close()
    return {
        "episodes": episodes,
        "wins": outcomes.count("win"),
        "losses": outcomes.count("loss"),
        "errors": errors,
        "avg_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "finite_forward_pass": len(errors) == 0,
    }


def run_policy_agreement_episode(
    agent: PPOAgent,
    env_tal: TalisharEngineEnvironment,
    env_cpp: CppEngineEnvironment,
    *,
    max_steps: int,
) -> dict[str, Any]:
    """Mirror Talishar steps on C++ and compare policy actions on both obs vectors."""
    _reset_talishar_for_parity(env_tal, env_cpp)
    opening_hands = _opening_hands_from_talishar(env_tal)
    hand_playability = _hand_playability_from_talishar(env_tal)
    acting_player_id = int(getattr(env_tal, "_acting_player_id", 1) or 1)
    reset_cpp = env_cpp.reset(
        options={
            "opening_hands": opening_hands,
            "hand_playability": hand_playability,
            "acting_player_id": acting_player_id,
        }
    )
    reset_tal = _build_talishar_reset_snapshot(env_tal)
    reset_cpp = _align_cpp_reset_result(env_cpp, reset_tal, reset_cpp)

    steps = 0
    policy_matches = 0
    policy_mismatches: list[str] = []
    obs_vec_mismatches: list[str] = []
    observation = reset_tal.observation

    for step in range(1, max_steps + 1):
        action_index, action_label = _choose_action(env_tal, observation, stress=False)
        try:
            action_descriptor = _talishar_action_descriptor(env_tal, action_index)
            step_tal = env_tal.step(str(action_index))
            set_mirror = getattr(_cpp_inner_env(env_cpp), "set_talishar_mirror_state", None)
            if callable(set_mirror):
                set_mirror(
                    _talishar_parity_snapshot(
                        step_tal,
                        raw_state=getattr(env_tal, "_last_state", None),
                    )
                )
            step_cpp = env_cpp.step(action_descriptor)
        except Exception as exc:
            return {
                "steps": steps,
                "policy_matches": policy_matches,
                "policy_mismatches": policy_mismatches,
                "obs_vec_mismatches": obs_vec_mismatches,
                "error": str(exc),
                "terminated_early": True,
            }

        ok_obs, obs_msg = compare_observations(step_tal.observation, step_cpp.observation)
        if not ok_obs:
            obs_vec_mismatches.append(f"step {step}: {obs_msg}")

        tal_vec = _obs_vec(env_tal)
        cpp_vec = _obs_vec(env_cpp)
        ok_aligned, align_msg = observation_vectors_aligned(tal_vec, cpp_vec, atol=0.05)
        if not ok_aligned:
            obs_vec_mismatches.append(f"step {step}: {align_msg}")

        legal_tal = len(step_tal.info.get("legal_actions", []) or [])
        legal_cpp = len(step_cpp.info.get("legal_actions", []) or [])
        legal = max(legal_tal, legal_cpp, 1)
        act_tal = _policy_action(agent, align_observation_for_cpp_training(tal_vec), legal)
        act_cpp = _policy_action(agent, align_observation_for_cpp_training(cpp_vec), legal)
        steps += 1
        if act_tal == act_cpp:
            policy_matches += 1
        else:
            policy_mismatches.append(
                f"step {step} after {action_label!r}: tal_action={act_tal} cpp_action={act_cpp}"
            )

        if bool(step_tal.terminated) or bool(step_tal.truncated):
            break
        observation = step_tal.observation

    return {
        "steps": steps,
        "policy_matches": policy_matches,
        "policy_mismatches": policy_mismatches,
        "obs_vec_mismatches": obs_vec_mismatches,
        "error": "",
        "terminated_early": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck1", default="Ira")
    parser.add_argument("--deck2", default="Ira")
    parser.add_argument("--format", default="silver_age")
    parser.add_argument("--train-episodes", type=int, default=20)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-env-parity", action="store_true")
    parser.add_argument("--weights-path", default=None)
    parser.add_argument("--parity-episodes", type=int, default=2)
    parser.add_argument("--policy-steps", type=int, default=40)
    parser.add_argument("--talishar-url", default="http://localhost:8080/game")
    parser.add_argument("--cpp-engine-dir", default=None)
    parser.add_argument("--out-dir", default="results/parity_checks/agent_transfer_v2")
    args = parser.parse_args()

    engine_dir = Path(args.cpp_engine_dir) if args.cpp_engine_dir else None
    if engine_dir is None:
        from flesh_and_blood_rlbridge.cpp_engine_environment import get_engine_dir  # noqa: PLC0415

        engine_dir = get_engine_dir(args.deck1, args.deck2)
    if not is_cpp_engine_available(engine_dir):
        print(f"C++ engine not available at {engine_dir}", file=sys.stderr)
        return 2

    out_root = Path(args.out_dir)
    train_dir = out_root / "cpp_training"
    train_dir.mkdir(parents=True, exist_ok=True)

    matchup = Matchup(
        name="ira-vs-ira-transfer",
        p1_deck=args.deck1,
        p2_deck=args.deck2,
        description="C++ training smoke for transfer parity",
        p1_hero="ira",
        p2_hero="ira",
        cpp_engine_dir=str(engine_dir),
    )

    if args.skip_train:
        if not args.weights_path:
            print("--weights-path required with --skip-train", file=sys.stderr)
            return 2
        weights_path = Path(args.weights_path)
    else:
        print(f"Training {args.train_episodes} episodes on C++ ({engine_dir})...")
        summary = train_matchup(
            matchup,
            base_url=args.talishar_url,
            n_episodes=args.train_episodes,
            max_steps=80,
            out_dir=train_dir,
            eval_env_ids={},
            game_format=args.format,
            warmup_episodes=0,
            warmup_baseline_eval_episodes=0,
            checkpoint_eval_episodes=0,
            build_cpp_engine=False,
            require_cpp_engine=True,
            n_workers=1,
        )
        weights_path = Path(summary["p1"]["package_dir"]) / "weights" / "agent_weights.json"
    print(f"Using weights: {weights_path}")

    agent = PPOAgent()
    agent.load(str(weights_path))

    parity_dir = out_root / f"{args.deck1}_vs_{args.deck2}"
    if args.skip_env_parity:
        print("Skipping env parity check (--skip-env-parity)")
        report = type("R", (), {
            "episodes_passed": 0, "episodes_run": 0, "episodes_failed": 0,
            "discrepancies_found": -1, "total_steps": 0,
        })()
        parity_code = -1
    else:
        print(f"Running env parity ({args.parity_episodes} episodes)...")
        report, parity_code = run_parity_check(
            deck1=args.deck1,
            deck2=args.deck2,
            game_format=args.format,
            episodes=args.parity_episodes,
            mode="multi-step",
            steps_per_episode=30,
            talishar_url=args.talishar_url,
            cpp_engine_dir=str(engine_dir),
            cpp_engine_deck1=args.deck1,
            cpp_engine_deck2=args.deck2,
            out_dir=parity_dir,
            write_reports=True,
            verbose=True,
        )

    env_tal = TalisharEngineEnvironment(
        base_url=args.talishar_url,
        game_format=args.format,
        local_deck_name=args.deck1,
        opponent_deck_name=args.deck2,
        max_turns=80,
        self_play=True,
        use_cpp_engine=False,
        cpp_obs_alignment=True,
    )
    env_cpp = CppEngineEnvironment(
        engine_dir=str(engine_dir),
        deck1=args.deck1,
        deck2=args.deck2,
        max_turns=80,
    )

    print("Running policy agreement on mirrored states...")
    policy_result = run_policy_agreement_episode(
        agent,
        env_tal,
        env_cpp,
        max_steps=args.policy_steps,
    )
    env_tal.close()
    env_cpp.close()

    print("Running Talishar HTTP eval with C++-trained weights...")
    tal_eval = _run_talishar_eval(
        agent,
        base_url=args.talishar_url,
        game_format=args.format,
        deck1=args.deck1,
        deck2=args.deck2,
        episodes=5,
        max_steps=80,
    )

    result = {
        "weights_path": str(weights_path),
        "train_episodes": args.train_episodes,
        "parity_exit_code": parity_code,
        "parity_episodes_passed": report.episodes_passed,
        "parity_episodes_failed": report.episodes_failed,
        "parity_discrepancies": report.discrepancies_found,
        "policy_agreement": policy_result,
        "talishar_eval": tal_eval,
    }
    result_path = out_root / "agent_transfer_report.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print("  Agent C++ → Talishar transfer report")
    print("=" * 72)
    print(f"  Training weights     : {weights_path}")
    print(f"  Env parity           : {report.episodes_passed}/{report.episodes_run} episodes passed")
    print(f"  Parity discrepancies : {report.discrepancies_found}")
    print(
        f"  Policy agreement     : {policy_result['policy_matches']}/{policy_result['steps']} steps"
    )
    if policy_result["policy_mismatches"]:
        print(f"  Policy mismatches    : {len(policy_result['policy_mismatches'])}")
        for row in policy_result["policy_mismatches"][:5]:
            print(f"    - {row}")
    if policy_result["obs_vec_mismatches"]:
        print(f"  Obs vec mismatches   : {len(policy_result['obs_vec_mismatches'])}")
        for row in policy_result["obs_vec_mismatches"][:5]:
            print(f"    - {row}")
    print(
        f"  Talishar HTTP eval   : {tal_eval['wins']}W / {tal_eval['losses']}L "
        f"over {tal_eval['episodes']} episodes (avg {tal_eval['avg_steps']:.1f} steps)"
    )
    if tal_eval["errors"]:
        print(f"  Talishar eval errors : {len(tal_eval['errors'])}")
        for row in tal_eval["errors"][:3]:
            print(f"    - {row}")
    print(f"  Full report          : {result_path}")
    print("=" * 72)

    ok = (
        tal_eval["finite_forward_pass"]
        and tal_eval["errors"] == []
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
