#!/usr/bin/env python3
"""Print side-by-side C++ vs Talishar agent I/O samples.

Runs a short episode on both engines and prints what an RL agent receives
(observation JSON) and produces (action index / legal-action descriptor) at
reset and after each step. Useful for eyeballing parity before running the
full ``run_parity_check.py`` suite.

Examples::

    # Default Ira vs Briar, 5 steps, Talishar on localhost
    python scripts/cpp/compare_engine_interaction_sample.py

    # Explicit matchup + compiled engine directory
    python scripts/cpp/compare_engine_interaction_sample.py \\
        --deck1 BriarSAGEPrecon --deck2 FaiSAGEPrecon \\
        --cpp-engine-dir results/cpp_engines/BriarSAGEPrecon_vs_FaiSAGEPrecon-e5b7fb33625fb9e1 \\
        --steps 8 --seed 42

    # Write machine-readable trace to JSON
    python scripts/cpp/compare_engine_interaction_sample.py --json-out sample_trace.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_ROOT))
import _bootstrap  # noqa: E402

_bootstrap.configure_paths()

from check_cpp_vs_talishar_parity import (  # noqa: E402
    INFO_CONTRACT_KEYS,
    OBSERVATION_SCALAR_KEYS,
    _align_cpp_reset_result,
    _build_talishar_reset_snapshot,
    _choose_action,
    _compare_reset,
    _compare_step,
    _create_parity_envs,
    _cpp_inner_env,
    _hand_playability_from_talishar,
    _json_safe_value,
    _legal_actions_from_observation,
    _opening_hands_from_talishar,
    _parse_observation,
    _reset_talishar_for_parity,
    _talishar_action_descriptor,
    _talishar_parity_snapshot,
    compare_info_contract,
    compare_observations,
    compare_rewards,
)
from check_cpp_vs_talishar_parity import ParityReport  # noqa: E402


def _default_talishar_url() -> str:
    return os.environ.get("TALISHAR_URL", "http://localhost:8080/game")


@dataclass
class InteractionFrame:
    """One reset or step snapshot from both engines."""

    label: str
    step: int
    action_index: Optional[int] = None
    action_label: str = ""
    talishar: dict[str, Any] = field(default_factory=dict)
    cpp: dict[str, Any] = field(default_factory=dict)
    parity: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "step": self.step,
            "action_index": self.action_index,
            "action_label": self.action_label,
            "talishar": self.talishar,
            "cpp": self.cpp,
            "parity": self.parity,
        }


def _observation_summary(observation: Any) -> dict[str, Any]:
    parsed, error = _parse_observation("obs", observation)
    if parsed is None:
        return {"error": error}

    hand = parsed.get("playerHand", [])
    hand_preview: list[dict[str, Any]] = []
    if isinstance(hand, list):
        for card in hand[:8]:
            if isinstance(card, dict):
                hand_preview.append(
                    {
                        "cardID": card.get("cardID") or card.get("cardNumber") or "",
                        "label": card.get("label", ""),
                        "action": card.get("action", 0),
                        "actionDataOverride": card.get("actionDataOverride", ""),
                    }
                )
        if len(hand) > 8:
            hand_preview.append({"note": f"... {len(hand) - 8} more cards"})

    legal = parsed.get("legalActions", [])
    legal_preview: list[dict[str, Any]] = []
    if isinstance(legal, list):
        for action in legal[:12]:
            if isinstance(action, dict):
                legal_preview.append(
                    {
                        "index": action.get("index"),
                        "label": action.get("label", ""),
                        "zone": action.get("zone", ""),
                    }
                )
        if len(legal) > 12:
            legal_preview.append({"note": f"... {len(legal) - 12} more actions"})

    summary: dict[str, Any] = {
        key: parsed.get(key)
        for key in OBSERVATION_SCALAR_KEYS
        if key in parsed
    }
    summary["playerHand"] = hand_preview
    summary["legalActions"] = legal_preview
    return summary


def _info_summary(info: Any) -> dict[str, Any]:
    if not isinstance(info, dict):
        return {}
    out: dict[str, Any] = {}
    for key in INFO_CONTRACT_KEYS:
        if key not in info:
            continue
        value = info[key]
        if key == "legal_actions" and isinstance(value, list):
            out[key] = [
                {
                    "action_code": action.get("action_code"),
                    "button_input": action.get("button_input", ""),
                    "card_id": action.get("card_id", ""),
                    "zone": action.get("zone", ""),
                    "label": action.get("label", ""),
                }
                for action in value[:12]
                if isinstance(action, dict)
            ]
            if len(value) > 12:
                out[key].append({"note": f"... {len(value) - 12} more"})
        else:
            out[key] = value
    if "engine" in info:
        out["engine"] = info["engine"]
    return out


def _step_summary(result: Any) -> dict[str, Any]:
    return {
        "reward": getattr(result, "reward", None),
        "terminated": bool(getattr(result, "terminated", False)),
        "truncated": bool(getattr(result, "truncated", False)),
        "observation": _observation_summary(getattr(result, "observation", {})),
        "info": _info_summary(getattr(result, "info", {})),
    }


def _reset_summary(result: Any) -> dict[str, Any]:
    return {
        "observation": _observation_summary(getattr(result, "observation", {})),
        "info": _info_summary(getattr(result, "info", {})),
    }


def _parity_flags(reset_or_step_tal: Any, reset_or_step_cpp: Any) -> dict[str, Any]:
    obs_ok, obs_msg = compare_observations(
        getattr(reset_or_step_tal, "observation", {}),
        getattr(reset_or_step_cpp, "observation", {}),
    )
    info_tal = getattr(reset_or_step_tal, "info", {}) or {}
    info_cpp = getattr(reset_or_step_cpp, "info", {}) or {}
    info_ok, info_msg = compare_info_contract(info_tal, info_cpp)

    flags: dict[str, Any] = {
        "observation_match": obs_ok,
        "info_match": info_ok,
    }
    if not obs_ok:
        flags["observation_detail"] = obs_msg
    if not info_ok:
        flags["info_detail"] = info_msg

    if hasattr(reset_or_step_tal, "reward"):
        reward_ok, reward_msg = compare_rewards(
            float(getattr(reset_or_step_tal, "reward", 0.0)),
            float(getattr(reset_or_step_cpp, "reward", 0.0)),
        )
        flags["reward_match"] = reward_ok
        if not reward_ok:
            flags["reward_detail"] = reward_msg
        flags["termination_match"] = bool(getattr(reset_or_step_tal, "terminated", False)) == bool(
            getattr(reset_or_step_cpp, "terminated", False)
        )
        flags["truncation_match"] = bool(getattr(reset_or_step_tal, "truncated", False)) == bool(
            getattr(reset_or_step_cpp, "truncated", False)
        )
    return flags


def _status_mark(ok: bool) -> str:
    return "OK" if ok else "DIFF"


def _print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _print_frame(frame: InteractionFrame) -> None:
    parity = frame.parity
    obs_mark = _status_mark(bool(parity.get("observation_match", True)))
    info_mark = _status_mark(bool(parity.get("info_match", True)))
    print()
    print(f"--- {frame.label} (step {frame.step}) ---")
    if frame.action_index is not None:
        print(f"Agent action: index={frame.action_index} label={frame.action_label!r}")
    print(f"Parity: observation={obs_mark}  info={info_mark}", end="")
    if "reward_match" in parity:
        reward_mark = _status_mark(bool(parity.get("reward_match", True)))
        term_mark = _status_mark(bool(parity.get("termination_match", True)))
        trunc_mark = _status_mark(bool(parity.get("truncation_match", True)))
        print(f"  reward={reward_mark}  terminated={term_mark}  truncated={trunc_mark}", end="")
    print()
    if parity.get("observation_detail"):
        print(f"  observation: {parity['observation_detail']}")
    if parity.get("info_detail"):
        print(f"  info: {parity['info_detail']}")
    if parity.get("reward_detail"):
        print(f"  reward: {parity['reward_detail']}")

    print()
    print("  Talishar observation (agent input):")
    print(json.dumps(frame.talishar.get("observation", {}), indent=2, sort_keys=True))
    print("  C++ observation (agent input):")
    print(json.dumps(frame.cpp.get("observation", {}), indent=2, sort_keys=True))

    print()
    print("  Talishar info.legal_actions (action descriptors):")
    print(json.dumps(frame.talishar.get("info", {}).get("legal_actions", []), indent=2))
    print("  C++ info.legal_actions (action descriptors):")
    print(json.dumps(frame.cpp.get("info", {}).get("legal_actions", []), indent=2))

    if "reward" in frame.talishar or "reward" in frame.cpp:
        print()
        print(
            f"  Talishar step: reward={frame.talishar.get('reward')} "
            f"terminated={frame.talishar.get('terminated')} "
            f"truncated={frame.talishar.get('truncated')}"
        )
        print(
            f"  C++ step:      reward={frame.cpp.get('reward')} "
            f"terminated={frame.cpp.get('terminated')} "
            f"truncated={frame.cpp.get('truncated')}"
        )


def _pick_action_index(
    env_tal: Any,
    observation: Any,
    *,
    seed: Optional[int],
    action_index: Optional[int],
    pass_only: bool,
) -> tuple[int, str]:
    legal = _legal_actions_from_observation(observation)
    if not legal:
        return 0, "<no legal actions>"

    if action_index is not None:
        index = max(0, min(int(action_index), len(legal) - 1))
    elif pass_only:
        index = 0
        for i, action in enumerate(legal):
            label = str(action.get("label", "") or "").strip().casefold()
            zone = str(action.get("zone", "") or "").strip().casefold()
            if label == "pass" or (
                zone == "button" and int(action.get("action_code", 0) or 0) == 99
            ):
                index = i
                break
    else:
        if seed is not None:
            random.seed(seed + len(legal))
        index, _ = _choose_action(env_tal, observation, stress=seed is not None)

    label = str(legal[index].get("label", "") or "")
    return index, label


def run_interaction_sample(
    *,
    deck1: str,
    deck2: str,
    game_format: str = "silver_age",
    steps: int = 5,
    talishar_url: str = "",
    cpp_engine_dir: str = "",
    cpp_engine_cache_dir: str = "",
    cpp_engine_deck1: str = "",
    cpp_engine_deck2: str = "",
    seed: Optional[int] = None,
    action_index: Optional[int] = None,
    pass_only: bool = False,
    json_out: Optional[Path] = None,
    max_turns: int = 2000,
) -> tuple[list[InteractionFrame], int]:
    """Run a short aligned episode and return frames plus exit code."""
    url = talishar_url or _default_talishar_url()
    frames: list[InteractionFrame] = []
    report = ParityReport(
        matchup=f"{deck1} vs {deck2}",
        format=game_format,
        mode="interaction-sample",
        episodes_requested=1,
    )

    _print_header("C++ vs Talishar — Agent I/O Sample")
    print(f"  Matchup : {deck1} vs {deck2}")
    print(f"  Format  : {game_format}")
    print(f"  Steps   : {steps}")
    print(f"  Talishar: {url}")
    if cpp_engine_dir:
        print(f"  C++ dir : {cpp_engine_dir}")
    if seed is not None:
        print(f"  Seed    : {seed}")
    if action_index is not None:
        print(f"  Action  : fixed index {action_index}")
    elif pass_only:
        print("  Action  : pass when available")

    try:
        env_tal, env_cpp = _create_parity_envs(
            deck1=deck1,
            deck2=deck2,
            game_format=game_format,
            max_turns=max_turns,
            talishar_url=url,
            cpp_engine_cache_dir=cpp_engine_cache_dir or None,
            cpp_engine_dir=cpp_engine_dir or None,
            cpp_engine_deck1=cpp_engine_deck1 or None,
            cpp_engine_deck2=cpp_engine_deck2 or None,
        )
    except Exception as exc:
        print(f"\n[ERROR] Could not create environments: {exc}")
        return frames, 2

    exit_code = 0
    try:
        print("\nResetting and aligning opening state...")
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

        reset_frame = InteractionFrame(
            label="RESET — agent input after env.reset()",
            step=0,
            talishar=_reset_summary(reset_tal),
            cpp=_reset_summary(reset_cpp),
            parity=_parity_flags(reset_tal, reset_cpp),
        )
        frames.append(reset_frame)
        _print_frame(reset_frame)
        if not _compare_reset(reset_tal, reset_cpp, report, episode=1):
            exit_code = 1

        observation = reset_tal.observation
        for step in range(1, steps + 1):
            chosen_index, chosen_label = _pick_action_index(
                env_tal,
                observation,
                seed=seed,
                action_index=action_index,
                pass_only=pass_only,
            )
            action_descriptor = _talishar_action_descriptor(env_tal, chosen_index)

            step_tal = env_tal.step(str(chosen_index))
            set_mirror = getattr(_cpp_inner_env(env_cpp), "set_talishar_mirror_state", None)
            if callable(set_mirror):
                set_mirror(_talishar_parity_snapshot(step_tal))
            step_cpp = env_cpp.step(action_descriptor)

            step_frame = InteractionFrame(
                label=f"STEP {step} — after agent action",
                step=step,
                action_index=chosen_index,
                action_label=chosen_label,
                talishar=_step_summary(step_tal),
                cpp=_step_summary(step_cpp),
                parity=_parity_flags(step_tal, step_cpp),
            )
            frames.append(step_frame)
            _print_frame(step_frame)

            if not _compare_step(
                step_tal,
                step_cpp,
                report,
                episode=1,
                step=step,
                action_index=chosen_index,
                action_label=chosen_label,
            ):
                exit_code = 1

            observation = step_tal.observation
            if bool(step_tal.terminated) or bool(step_tal.truncated):
                print(
                    f"\nEpisode ended at step {step} "
                    f"(terminated={step_tal.terminated}, truncated={step_tal.truncated})"
                )
                break
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user")
        exit_code = 130
    except Exception as exc:
        print(f"\n[ERROR] Sample run failed: {exc}")
        exit_code = 2
    finally:
        env_tal.close()
        env_cpp.close()

    if report.discrepancies_found:
        print(
            f"\nSummary: {report.discrepancies_found} parity difference(s) "
            f"across {len(frames)} frame(s)"
        )
    else:
        print(f"\nSummary: all {len(frames)} frame(s) matched between engines")

    if json_out is not None:
        payload = {
            "matchup": f"{deck1} vs {deck2}",
            "format": game_format,
            "steps_requested": steps,
            "frames": [_json_safe_value(frame.to_dict()) for frame in frames],
            "discrepancies": report.discrepancies,
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON trace: {json_out}")

    return frames, exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--deck1", default="Ira", help="P1 Talishar local deck name")
    parser.add_argument("--deck2", default="Briar", help="P2 Talishar local deck name")
    parser.add_argument(
        "--format",
        default="silver_age",
        choices=["silver_age", "classic_constructed"],
        help="Game format",
    )
    parser.add_argument("--steps", type=int, default=5, help="Agent steps after reset")
    parser.add_argument("--max-turns", type=int, default=2000, help="Episode truncation limit")
    parser.add_argument("--talishar-url", default="", help="Talishar server base URL")
    parser.add_argument("--cpp-engine-dir", default="", help="Compiled fab_engine directory")
    parser.add_argument(
        "--cpp-engine-cache-dir",
        default="",
        help="Cache root containing compiled matchup subdirectories",
    )
    parser.add_argument("--cpp-engine-deck1", default="", help="Deck1 name for C++ cache lookup")
    parser.add_argument("--cpp-engine-deck2", default="", help="Deck2 name for C++ cache lookup")
    parser.add_argument("--seed", type=int, default=None, help="Random action seed (optional)")
    parser.add_argument(
        "--action-index",
        type=int,
        default=None,
        help="Always choose this legal-action index (overrides --seed / --pass-only)",
    )
    parser.add_argument(
        "--pass-only",
        action="store_true",
        help="Prefer the Pass action (mode 99) when available",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Write full trace JSON to this path (default: no file)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    json_out = Path(args.json_out) if args.json_out else None
    _, exit_code = run_interaction_sample(
        deck1=args.deck1,
        deck2=args.deck2,
        game_format=args.format,
        steps=max(0, int(args.steps)),
        talishar_url=args.talishar_url,
        cpp_engine_dir=args.cpp_engine_dir,
        cpp_engine_cache_dir=args.cpp_engine_cache_dir,
        cpp_engine_deck1=args.cpp_engine_deck1,
        cpp_engine_deck2=args.cpp_engine_deck2,
        seed=args.seed,
        action_index=args.action_index,
        pass_only=bool(args.pass_only),
        json_out=json_out,
        max_turns=args.max_turns,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
