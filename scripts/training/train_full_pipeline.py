#!/usr/bin/env python3
"""Full three-phase FaB RL training pipeline (orchestrator).

Fast default workflow:
  1. SideboardGuidePolicy picks a game deck per opponent (no sideboard RL).
  2. Train play agents on those decks.
  3. Iterations: deckbuild pool → guide sideboard → play train.

For standalone phase training, use:

* ``train_deckbuild_sideboard.py`` — Phases 1 and 2 (pool + sideboard RL)
* ``train_play.py`` — Phase 3 (play agents + optional final eval)

* ``train_deckbuild_sideboard.py`` — Phases 1 and 2 (pool + sideboard)
* ``train_play.py`` — Phase 3 (play agents + optional final eval)

    python scripts/training/train_full_pipeline.py --iterations 3

    # Or run phases separately:
    python scripts/training/train_deckbuild_sideboard.py --iterations 3
    python scripts/training/train_play.py --iterations 3
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_ROOT))
import _bootstrap  # noqa: E402

REPO_ROOT = _bootstrap.configure_paths()
_RL_SRC = Path("~/Documents/RL/rlbridge/src").expanduser()
if str(_RL_SRC) not in sys.path:
    sys.path.insert(0, str(_RL_SRC))

from flesh_and_blood_rlbridge.opponent_deck import (  # noqa: E402
    normalize_talishar_asset_name,
    resolve_opponent_deck_name,
)

from train_deckbuild_sideboard import _build_phase_agents  # noqa: E402
from train_pipeline_common import (  # noqa: E402
    DEFAULT_EQUIPMENT_HEADER,
    DEFAULT_FORMAT,
    DEFAULT_HERO_CLASS,
    DEFAULT_HERO_ID,
    DEFAULT_OPPONENT_DECK,
    DEFAULT_OPPONENT_HERO,
    OUT_DIR,
    PhaseAgents,
    _load_agent,
    _load_starting_deck,
    _save_all_agents,
    _write_deck_file,
    _write_results_json,
    apply_deck_state,
    apply_guide_sideboard_for_matchup,
    greedy_game_deck_cut,
    load_deck_state,
    min_deck_size_for_format,
    resolve_assets_path,
    run_phase1_deckbuilder,
    save_deck_state,
)
from train_play import (  # noqa: E402
    DEFAULT_CHECKPOINT_EVAL_EPISODES,
    DEFAULT_CHECKPOINT_INTERVAL_PCT,
    DEFAULT_PARALLEL_SEEDS,
    DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
    DEFAULT_WARMUP_EPISODES,
    auto_detect_workers,
    run_final_evaluation,
    run_phase3_play,
)
from runtime_defaults import RUNTIME  # noqa: E402
from cpp_engine_matchup import ensure_cpp_engine_for_matchup  # noqa: E402

# Re-export symbols used by eval_phase3_checkpoint.py and other tools.
from train_play import (  # noqa: E402,F401
    _ensure_playwright,
    _frames_to_gif,
    _infer_render_outcome,
    _prepare_render_dir,
    _save_end_state_frame,
    _save_state_image,
)
from train_pipeline_common import _write_deck_file  # noqa: E402,F401


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full three-phase FaB RL training: Deckbuilder, Sideboard, Play",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--format", default=DEFAULT_FORMAT,
        choices=["silver_age", "classic_constructed", "blitz", "upf"])
    parser.add_argument("--hero-id", default=DEFAULT_HERO_ID)
    parser.add_argument("--hero-class", default=DEFAULT_HERO_CLASS)
    parser.add_argument("--equipment-header", default=DEFAULT_EQUIPMENT_HEADER)
    parser.add_argument("--opponent-mode", default="preset",
        choices=["preset", "mirror", "dual"])
    parser.add_argument("--opponent-deck", default=DEFAULT_OPPONENT_DECK)
    parser.add_argument("--opponent-hero-id", default=DEFAULT_OPPONENT_HERO)
    parser.add_argument("--p2-hero-id", default="dorinthea_ironsong")
    parser.add_argument("--p2-hero-class", default="Warrior")
    parser.add_argument("--p2-equipment-header",
        default="dorinthea_ironsong dori_equipment_sword dori_equipment_sword "
                "helm_of_avarice gauntlet_of_might ironrot_legs valor_boots")

    parser.add_argument("--deckbuild-episodes", type=int, default=RUNTIME.full_pipeline.deckbuild_episodes)
    parser.add_argument("--play-episodes", type=int, default=RUNTIME.full_pipeline.play_episodes)
    parser.add_argument("--play-checkpoint-interval", type=int, default=None,
        help="Fixed checkpoint interval in episodes (default: 10%% of --play-episodes)")
    parser.add_argument("--checkpoint-interval-pct", type=float,
        default=DEFAULT_CHECKPOINT_INTERVAL_PCT,
        help="Checkpoint every N%% of play episodes when interval is unset")
    parser.add_argument("--checkpoint-eval-episodes", type=int,
        default=DEFAULT_CHECKPOINT_EVAL_EPISODES,
        help="C++ eval games vs opponent agent (greedy) at each checkpoint (0=off)")
    parser.add_argument("--max-build-steps", type=int, default=RUNTIME.full_pipeline.max_build_steps)
    parser.add_argument("--max-sideboard-steps", type=int, default=RUNTIME.full_pipeline.max_sideboard_steps)
    parser.add_argument("--max-play-steps", type=int, default=RUNTIME.play.max_play_steps)
    parser.add_argument("--num-eval-games", type=int, default=RUNTIME.full_pipeline.num_eval_games,
        help="Internal eval games for deckbuilder phase")
    parser.add_argument("--num-sideboard-episodes", type=int, default=RUNTIME.full_pipeline.num_sideboard_episodes,
        help="Sideboard episodes inside deckbuilder finalize (guide policy, max 1)")
    parser.add_argument("--warmup-episodes", type=int, default=DEFAULT_WARMUP_EPISODES)
    parser.add_argument("--warmup-baseline-eval-episodes", type=int,
        default=DEFAULT_WARMUP_BASELINE_EVAL_EPISODES)
    parser.add_argument("--workers", type=int, default=None,
        help="Parallel C++ game sessions for play training (auto-detected when omitted)")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--parallel-seeds",
        type=int,
        default=DEFAULT_PARALLEL_SEEDS,
        help=(
            "Independent RNG seeds for play training; training win%% is averaged "
            f"and best P1/P2 agents are used for eval (default: "
            f"{DEFAULT_PARALLEL_SEEDS}; use 1 to disable)"
        ),
    )
    parser.add_argument("--iterations", type=int, default=RUNTIME.full_pipeline.iterations)

    parser.add_argument("--p1-deckbuilder", default=None)
    parser.add_argument("--p1-sideboard", default=None)
    parser.add_argument("--p1-play", default=None)
    parser.add_argument("--p2-deckbuilder", default=None)
    parser.add_argument("--p2-sideboard", default=None)
    parser.add_argument("--p2-play", default=None)
    parser.add_argument("--p1-starting-deck", default=None)
    parser.add_argument("--p2-starting-deck", default=None)
    parser.add_argument("--p1-fixed-deck", default=None)
    parser.add_argument("--p2-fixed-deck", default=None)

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--results-json", default=None)
    parser.add_argument("--talishar-url",
        default=os.environ.get("TALISHAR_URL", "http://localhost:8080/game"))
    parser.add_argument("--talishar-fe-url",
        default=os.environ.get("TALISHAR_FE_URL", "http://localhost:5173"))
    parser.add_argument("--cpp-engine-dir", default=None)
    parser.add_argument("--assets-path", default=os.environ.get("TALISHAR_ASSETS_PATH", ""))
    parser.add_argument(
        "--no-build-cpp-engine",
        action="store_true",
        help="Do not auto-build a C++ engine when none is cached",
    )
    parser.add_argument("--final-eval-episodes", type=int, default=RUNTIME.full_pipeline.final_eval_episodes)
    parser.add_argument("--final-eval-max-steps", type=int, default=RUNTIME.full_pipeline.final_eval_max_steps)
    parser.add_argument(
        "--render-gif",
        action="store_true",
        help="After final eval, render a Talishar FE rollout GIF (slow; off by default)",
    )
    parser.add_argument("--gif-fps", type=float, default=RUNTIME.play.gif_fps)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_path = resolve_assets_path(args.assets_path or None)
    args.opponent_deck = normalize_talishar_asset_name(args.opponent_deck, assets_path)
    if args.opponent_mode == "mirror":
        args.opponent_hero_id = args.hero_id

    cpp_opponent = (
        args.p2_hero_id if args.opponent_mode == "dual" else args.opponent_deck
    )
    if not args.no_build_cpp_engine:
        _use_existing = False
        if args.cpp_engine_dir:
            try:
                if str(REPO_ROOT / "src") not in sys.path:
                    sys.path.insert(0, str(REPO_ROOT / "src"))
                from flesh_and_blood_rlbridge.cpp_engine_environment import (  # noqa: PLC0415
                    is_cpp_engine_available,
                )
                _use_existing = is_cpp_engine_available(args.cpp_engine_dir)
            except Exception:
                _use_existing = False
        if not _use_existing:
            args.cpp_engine_dir = ensure_cpp_engine_for_matchup(
                args.hero_id,
                cpp_opponent,
                assets_path=assets_path,
                talishar_url=args.talishar_url,
                deck1_json=Path(args.p1_starting_deck) if args.p1_starting_deck else (
                    Path(args.p1_fixed_deck) if args.p1_fixed_deck else None
                ),
                deck2_json=Path(args.p2_starting_deck) if args.p2_starting_deck else (
                    Path(args.p2_fixed_deck) if args.p2_fixed_deck else None
                ),
                build=True,
            ) or args.cpp_engine_dir

    if args.workers is None:
        args.workers = auto_detect_workers(
            hero_id=args.hero_id,
            p2_hero_id=args.p2_hero_id,
            cpp_engine_dir=args.cpp_engine_dir,
            assets_path=assets_path,
        )

    try:
        import torch as _torch
        _gpu_label = (
            f"GPU ({_torch.cuda.get_device_name(0)})"
            if _torch.cuda.is_available() else "CPU"
        )
    except ImportError:
        _gpu_label = "CPU (torch not available)"
    print(f"  [device] PPO gradient updates: {_gpu_label}")
    print(f"  [workers] Parallel game sessions: {args.workers}")

    min_warmup = max(1, math.ceil(args.play_episodes / 10))
    warmup_eps = min(max(args.warmup_episodes, min_warmup), args.play_episodes)

    results_json = Path(args.results_json) if args.results_json else out_dir / "results.json"
    min_size = min_deck_size_for_format(args.format)

    p1, p2 = _build_phase_agents(args)

    state = load_deck_state(out_dir)
    if state:
        apply_deck_state(p1, p2, state)

    p1_starting_deck = _load_starting_deck(args.p1_starting_deck)
    p2_starting_deck = _load_starting_deck(args.p2_starting_deck)
    p1_fixed_deck = _load_starting_deck(args.p1_fixed_deck)
    p2_fixed_deck = _load_starting_deck(args.p2_fixed_deck)

    if args.opponent_mode == "dual" and p2 is not None:
        p1_opponents = [args.p2_hero_id]
        p2_opponents = [args.hero_id]
        p1_opp_key = args.p2_hero_id
    else:
        p1_opponents = [args.opponent_hero_id]
        p2_opponents = []
        p1_opp_key = args.opponent_hero_id

    if p1_starting_deck and not p1.card_pool:
        p1.card_pool = dict(p1_starting_deck)
        p1.active_decks[p1_opp_key] = greedy_game_deck_cut(p1_starting_deck, min_size)
    if p2 is not None and p2_starting_deck and not p2.card_pool:
        p2.card_pool = dict(p2_starting_deck)
        p2.active_decks[args.hero_id] = greedy_game_deck_cut(p2_starting_deck, min_size)

    if p1_fixed_deck:
        p1.card_pool = dict(p1_fixed_deck)
        if sum(p1_fixed_deck.values()) >= min_size:
            p1.active_decks[p1_opp_key] = greedy_game_deck_cut(p1_fixed_deck, min_size)
    if p2 is not None and p2_fixed_deck:
        p2.card_pool = dict(p2_fixed_deck)
        if sum(p2_fixed_deck.values()) >= min_size:
            p2.active_decks[args.hero_id] = greedy_game_deck_cut(p2_fixed_deck, min_size)

    print(
        f"\n{'='*62}\n"
        f"  Full Pipeline Training (guide sideboard → play → deckbuild)\n"
        f"  Format: {args.format}  |  Hero: {args.hero_id}\n"
        f"  Opponent mode: {args.opponent_mode}\n"
        f"  Sideboard: SideboardGuidePolicy (no RL, no C++ eval)\n"
        f"  Outer iterations: {args.iterations}\n"
        f"{'='*62}"
    )

    def _p1_opponent_deck() -> str:
        return resolve_opponent_deck_name(
            player_hero_id=args.hero_id,
            opponent_mode=args.opponent_mode,
            preset_opponent_deck=args.opponent_deck,
            opponent_agents=p2,
            opponent_hero_id=args.p2_hero_id,
            assets_path=assets_path,
            min_deck_size=min_size,
            write_deck_file=_write_deck_file,
            opponent_equipment_header=args.p2_equipment_header,
        )

    def _p2_opponent_deck() -> str:
        return resolve_opponent_deck_name(
            player_hero_id=args.p2_hero_id,
            opponent_mode=args.opponent_mode,
            preset_opponent_deck=args.opponent_deck,
            opponent_agents=p1,
            opponent_hero_id=args.hero_id,
            assets_path=assets_path,
            min_deck_size=min_size,
            write_deck_file=_write_deck_file,
            opponent_equipment_header=args.equipment_header,
        )

    p1_eval_opponent_hero = (
        args.p2_hero_id if args.opponent_mode == "dual" else args.opponent_hero_id
    )

    def _apply_p1_guide_sideboard() -> None:
        if p1_fixed_deck and sum(p1_fixed_deck.values()) >= min_size:
            p1.active_decks[p1_opp_key] = greedy_game_deck_cut(p1_fixed_deck, min_size)
            return
        apply_guide_sideboard_for_matchup(
            p1, p1_opponents,
            hero_id=args.hero_id,
            hero_class=args.hero_class,
            equipment_header=args.equipment_header,
            game_format=args.format,
            opponent_deck_name=_p1_opponent_deck(),
            max_sideboard_steps=args.max_sideboard_steps,
            assets_path=assets_path,
            base_url=args.talishar_url,
            cpp_engine_dir=args.cpp_engine_dir,
        )

    def _apply_p2_guide_sideboard() -> None:
        if p2 is None:
            return
        if p2_fixed_deck and sum(p2_fixed_deck.values()) >= min_size:
            p2.active_decks[args.hero_id] = greedy_game_deck_cut(p2_fixed_deck, min_size)
            return
        apply_guide_sideboard_for_matchup(
            p2, p2_opponents,
            hero_id=args.p2_hero_id,
            hero_class=args.p2_hero_class,
            equipment_header=args.p2_equipment_header,
            game_format=args.format,
            opponent_deck_name=_p2_opponent_deck(),
            max_sideboard_steps=args.max_sideboard_steps,
            assets_path=assets_path,
            base_url=args.talishar_url,
            cpp_engine_dir=args.cpp_engine_dir,
        )

    def _run_play_phase(label: str) -> None:
        nonlocal p1, p2
        print(f"\n{'#'*62}\n  {label}\n{'#'*62}")
        p1_wr, p2_wr = run_phase3_play(
            p1,
            p2 if args.opponent_mode == "dual" else None,
            opponent_mode=args.opponent_mode,
            game_format=args.format,
            p1_hero_id=args.hero_id,
            p2_hero_id=args.p2_hero_id,
            p1_equipment_header=args.equipment_header,
            p2_equipment_header=args.p2_equipment_header,
            p1_opponent_hero_id=(
                args.p2_hero_id if args.opponent_mode == "dual" else args.opponent_hero_id
            ),
            p1_opponent_deck_name=args.opponent_deck,
            n_episodes=args.play_episodes,
            max_play_steps=args.max_play_steps,
            warmup_episodes=warmup_eps,
            warmup_baseline_eval_episodes=args.warmup_baseline_eval_episodes,
            n_workers=args.workers,
            assets_path=assets_path,
            base_url=args.talishar_url,
            out_dir=out_dir,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            seed=args.seed,
            cpp_engine_dir=args.cpp_engine_dir,
            checkpoint_interval=args.play_checkpoint_interval,
            checkpoint_interval_pct=args.checkpoint_interval_pct,
            checkpoint_eval_episodes=args.checkpoint_eval_episodes,
            parallel_seeds=args.parallel_seeds,
        )
        p1.last_play_win_rate = p1_wr
        if p2 is not None:
            p2.last_play_win_rate = p2_wr

    # ── Stage 1: guide sideboard on starting pool, then initial play ────────
    if p1.card_pool:
        print(f"\n{'#'*62}\n  STAGE 1 — Guide sideboard + initial play\n{'#'*62}")
        _apply_p1_guide_sideboard()
        if args.opponent_mode == "dual":
            _apply_p2_guide_sideboard()
        _run_play_phase("STAGE 1 — Play (starting pool)")

    for iteration in range(1, args.iterations + 1):
        print(f"\n\n{'#'*62}\n  ITERATION {iteration} / {args.iterations}\n{'#'*62}")

        if not p1_fixed_deck:
            run_phase1_deckbuilder(
                p1,
                hero_id=args.hero_id,
                hero_class=args.hero_class,
                equipment_header=args.equipment_header,
                game_format=args.format,
                opponent_deck_name=_p1_opponent_deck(),
                opponent_hero_id=p1_eval_opponent_hero,
                n_episodes=args.deckbuild_episodes,
                max_build_steps=args.max_build_steps,
                num_eval_games=args.num_eval_games,
                num_sideboard_episodes=args.num_sideboard_episodes,
                assets_path=assets_path,
                base_url=args.talishar_url,
                render=args.render,
                cpp_engine_dir=args.cpp_engine_dir,
                starting_deck=p1.card_pool if iteration > 1 else p1_starting_deck,
                play_reward=p1.last_play_win_rate,
            )

            if args.opponent_mode == "dual" and p2 is not None and not p2_fixed_deck:
                run_phase1_deckbuilder(
                    p2,
                    hero_id=args.p2_hero_id,
                    hero_class=args.p2_hero_class,
                    equipment_header=args.p2_equipment_header,
                    game_format=args.format,
                    opponent_deck_name=_p2_opponent_deck(),
                    opponent_hero_id=args.hero_id,
                    n_episodes=args.deckbuild_episodes,
                    max_build_steps=args.max_build_steps,
                    num_eval_games=args.num_eval_games,
                    num_sideboard_episodes=args.num_sideboard_episodes,
                    assets_path=assets_path,
                    base_url=args.talishar_url,
                    render=args.render,
                    cpp_engine_dir=args.cpp_engine_dir,
                    starting_deck=p2.card_pool if iteration > 1 else p2_starting_deck,
                    play_reward=p2.last_play_win_rate if p2 else 0.0,
                )
        elif p1_fixed_deck:
            p1.card_pool = dict(p1_fixed_deck)
            print(f"\n  [p1] Deckbuild skipped — fixed deck ({sum(p1_fixed_deck.values())} cards)")

        _apply_p1_guide_sideboard()
        if args.opponent_mode == "dual":
            _apply_p2_guide_sideboard()

        _run_play_phase(f"ITERATION {iteration} — Play")

        _save_all_agents(p1, out_dir)
        if p2 is not None:
            _save_all_agents(p2, out_dir)
        save_deck_state(out_dir, p1=p1, p2=p2, game_format=args.format, opponent_mode=args.opponent_mode)
        _write_results_json(
            results_json,
            game_format=args.format,
            opponent_mode=args.opponent_mode,
            p1=p1,
            p2=p2,
            iterations=iteration,
        )

    final_eval_dir = out_dir / "final_eval"
    final_eval_dir.mkdir(parents=True, exist_ok=True)
    p1_eval = run_final_evaluation(
        p1,
        p2 if args.opponent_mode == "dual" else None,
        hero_id=args.hero_id,
        equipment_header=args.equipment_header,
        opponent_equipment_header=args.p2_equipment_header,
        game_format=args.format,
        opponent_deck_name=args.opponent_deck,
        opponent_hero_id=(
            args.opponent_hero_id if args.opponent_mode != "dual" else args.p2_hero_id
        ),
        opponent_mode=args.opponent_mode,
        num_eval_episodes=args.final_eval_episodes,
        max_steps=args.final_eval_max_steps,
        assets_path=assets_path,
        base_url=args.talishar_url,
        fe_url=args.talishar_fe_url,
        out_dir=final_eval_dir,
        render_gif=args.render_gif,
        gif_fps=args.gif_fps,
    )
    p2_eval: Optional[dict[str, Any]] = None
    if args.opponent_mode == "dual" and p2 is not None:
        p2_eval = run_final_evaluation(
            p2, p1,
            hero_id=args.p2_hero_id,
            equipment_header=args.p2_equipment_header,
            opponent_equipment_header=args.equipment_header,
            game_format=args.format,
            opponent_deck_name=args.opponent_deck,
            opponent_hero_id=args.hero_id,
            opponent_mode=args.opponent_mode,
            num_eval_episodes=args.final_eval_episodes,
            max_steps=args.final_eval_max_steps,
            assets_path=assets_path,
            base_url=args.talishar_url,
            fe_url=args.talishar_fe_url,
            out_dir=final_eval_dir,
            render_gif=args.render_gif,
            gif_fps=args.gif_fps,
        )

    _write_results_json(
        results_json,
        game_format=args.format,
        opponent_mode=args.opponent_mode,
        p1=p1,
        p2=p2,
        iterations=args.iterations,
        p1_final_eval=p1_eval,
        p2_final_eval=p2_eval,
    )

    print(f"\n  Training complete → {out_dir}")


if __name__ == "__main__":
    main()
