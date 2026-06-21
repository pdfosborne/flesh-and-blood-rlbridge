#!/usr/bin/env python3
"""Watch and evaluate Phase 3 checkpoints for a matchup results directory."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runtime_defaults import RUNTIME  # noqa: E402

from runscripts._common import (
    REPO_ROOT,
    SCRIPTS_EVAL,
    assets_path,
    env_or_default,
    find_cpp_engine_dir,
    matchup_label_from_dir_name,
    run_python,
    talishar_url,
)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "-?" in argv or "/?" in argv:
        argv = ["--help"]

    results_dir = Path(
        os.environ.get("RESULTS_DIR", str(REPO_ROOT / "results" / "matchup_sims" / "briar_vs_riptide"))
    )
    assets = assets_path()
    url = talishar_url()
    _eval = RUNTIME.eval_dashboard
    episodes = env_or_default("EPISODES", str(_eval.episodes))
    parallel_workers = env_or_default("PARALLEL_WORKERS", str(_eval.parallel_workers))
    max_steps = env_or_default("MAX_STEPS", str(_eval.max_steps))
    render_max_steps = env_or_default("RENDER_MAX_STEPS", str(_eval.render_max_steps))
    poll_seconds = env_or_default("POLL_SECONDS", str(_eval.poll_seconds))
    stall_no_damage_turns = env_or_default("STALL_NO_DAMAGE_TURNS", str(_eval.stall_no_damage_turns))
    stall_low_hand_turns = env_or_default("STALL_LOW_HAND_TURNS", str(_eval.stall_low_hand_turns))
    stall_max_single_low_hand_turns = env_or_default(
        "STALL_MAX_SINGLE_LOW_HAND_TURNS",
        str(_eval.stall_max_single_low_hand_turns),
    )
    stall_min_attack_hand = env_or_default("STALL_MIN_ATTACK_HAND", str(_eval.stall_min_attack_hand))
    gif_fps = env_or_default("GIF_FPS", str(_eval.gif_fps))

    parity_args: list[str] = []
    cpp_engine_dir = os.environ.get("CPP_ENGINE_DIR")
    if cpp_engine_dir:
        parity_args = ["--parity-cpp-engine-dir", cpp_engine_dir]
    else:
        label = matchup_label_from_dir_name(results_dir.name)
        if label:
            discovered = find_cpp_engine_dir(label)
            if discovered is not None:
                print(f"  Parity C++ engine: {discovered}")
                parity_args = ["--parity-cpp-engine-dir", str(discovered)]

    cmd = [
        "--results-dir",
        str(results_dir),
        "--assets-path",
        str(assets),
        "--talishar-url",
        url,
        "--episodes",
        episodes,
        "--parallel-workers",
        parallel_workers,
        "--max-steps",
        max_steps,
        "--render-max-steps",
        render_max_steps,
        "--stall-no-damage-turns",
        stall_no_damage_turns,
        "--stall-low-hand-turns",
        stall_low_hand_turns,
        "--stall-max-single-low-hand-turns",
        stall_max_single_low_hand_turns,
        "--stall-min-attack-hand",
        stall_min_attack_hand,
        "--watch",
        "--poll-seconds",
        poll_seconds,
        "--gif-fps",
        gif_fps,
        *parity_args,
        *argv,
    ]
    return run_python(SCRIPTS_EVAL / "eval_phase3_checkpoint.py", *cmd)


if __name__ == "__main__":
    raise SystemExit(main())
