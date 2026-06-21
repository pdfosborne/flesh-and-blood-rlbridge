#!/usr/bin/env python3
"""Deckbuilder and sideboard training (Phases 1 and 2).

Trains deck construction and sideboard selection agents.  Play win-rate
feedback from a prior ``train_play.py`` run (via ``deck_state.json`` or
``results.json``) can be passed back as a terminal reward bonus.

Typical workflow::

    # 1. Build pools and sideboard decks
    python scripts/training/train_deckbuild_sideboard.py --iterations 3

    # 2. Train play agents on the resulting decks
    python scripts/training/train_play.py --iterations 3

    # Or run the full alternating loop:
    python scripts/training/train_full_pipeline.py --iterations 3
"""

from __future__ import annotations

import argparse
import json
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

from flesh_and_blood_rlbridge.opponent_deck import resolve_opponent_deck_name  # noqa: E402

from train_pipeline_common import (  # noqa: E402
    DEFAULT_EQUIPMENT_HEADER,
    DEFAULT_FORMAT,
    DEFAULT_HERO_CLASS,
    DEFAULT_HERO_ID,
    DEFAULT_OPPONENT_DECK,
    DEFAULT_OPPONENT_HERO,
    DEFAULT_SIDEBOARD_EVAL_GAMES,
    DEFAULT_SIDEBOARD_BATCH_SIZE,
    DEFAULT_SIDEBOARD_CONVERGENCE_WINDOW,
    DEFAULT_SIDEBOARD_CONVERGENCE_STD,
    OUT_DIR,
    PhaseAgents,
    _load_agent,
    _load_starting_deck,
    _save_all_agents,
    _write_deck_file,
    _write_results_json,
    apply_deck_state,
    greedy_game_deck_cut,
    load_deck_state,
    load_play_feedback,
    min_deck_size_for_format,
    resolve_assets_path,
    run_phase1_deckbuilder,
    run_phase2_sideboard,
    save_deck_state,
)


def _build_phase_agents(args: argparse.Namespace) -> tuple[PhaseAgents, Optional[PhaseAgents]]:
    p1 = PhaseAgents(
        player="p1",
        deckbuilder=_load_agent(args.p1_deckbuilder),
        sideboard=_load_agent(args.p1_sideboard),
        play=_load_agent(args.p1_play),
        equipment_header=args.equipment_header,
    )
    p2: Optional[PhaseAgents] = None
    if args.opponent_mode == "dual":
        p2 = PhaseAgents(
            player="p2",
            deckbuilder=_load_agent(args.p2_deckbuilder),
            sideboard=_load_agent(args.p2_sideboard),
            play=_load_agent(args.p2_play),
            equipment_header=args.p2_equipment_header,
        )
    return p1, p2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train deckbuilder and sideboard agents (Phases 1 and 2)",
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

    parser.add_argument("--deckbuild-episodes", type=int, default=50)
    parser.add_argument("--sideboard-episodes", type=int, default=20)
    parser.add_argument("--max-build-steps", type=int, default=200)
    parser.add_argument("--max-sideboard-steps", type=int, default=100)
    parser.add_argument("--num-eval-games", type=int, default=3,
        help="Internal eval games for deckbuilder phase (not sideboard)")
    parser.add_argument("--sideboard-eval-games", type=int,
        default=DEFAULT_SIDEBOARD_EVAL_GAMES)
    parser.add_argument("--sideboard-batch-size", type=int,
        default=DEFAULT_SIDEBOARD_BATCH_SIZE)
    parser.add_argument("--sideboard-convergence-window", type=int,
        default=DEFAULT_SIDEBOARD_CONVERGENCE_WINDOW)
    parser.add_argument("--sideboard-convergence-std", type=float,
        default=DEFAULT_SIDEBOARD_CONVERGENCE_STD)
    parser.add_argument("--max-sideboard-rounds", type=int, default=10)
    parser.add_argument("--num-sideboard-episodes", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--render", action="store_true")

    parser.add_argument("--p1-deckbuilder", default=None)
    parser.add_argument("--p1-sideboard", default=None)
    parser.add_argument("--p1-play", default=None,
        help="Optional play agent used during internal deck/sideboard eval games")
    parser.add_argument("--p2-deckbuilder", default=None)
    parser.add_argument("--p2-sideboard", default=None)
    parser.add_argument("--p2-play", default=None)

    parser.add_argument("--p1-starting-deck", default=None)
    parser.add_argument("--p2-starting-deck", default=None)
    parser.add_argument("--p1-fixed-deck", default=None)
    parser.add_argument("--p2-fixed-deck", default=None)

    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--results-json", default=None)
    parser.add_argument("--deck-state", default=None,
        help="Path to deck_state.json for load/save (default: <out-dir>/deck_state.json)")
    parser.add_argument("--talishar-url",
        default=os.environ.get("TALISHAR_URL", "http://localhost:8080/game"))
    parser.add_argument("--cpp-engine-dir", default=None)
    parser.add_argument("--assets-path", default=os.environ.get("TALISHAR_ASSETS_PATH", ""))
    parser.add_argument("--play-reward-p1", type=float, default=None,
        help="Override P1 play win-rate feedback (default: load from deck_state/results)")
    parser.add_argument("--play-reward-p2", type=float, default=None,
        help="Override P2 play win-rate feedback (dual mode)")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_path = resolve_assets_path(args.assets_path or None)
    results_json = Path(args.results_json) if args.results_json else out_dir / "results.json"
    min_size = min_deck_size_for_format(args.format)

    p1, p2 = _build_phase_agents(args)

    deck_state_path = Path(args.deck_state) if args.deck_state else out_dir / "deck_state.json"
    if deck_state_path.is_file():
        try:
            state = json.loads(deck_state_path.read_text(encoding="utf-8"))
            apply_deck_state(p1, p2, state)
            print(f"  Loaded deck state from {deck_state_path}")
        except Exception as exc:
            print(f"  WARNING: could not load deck state: {exc}")

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
        f"  Deckbuild + Sideboard Training (sideboard-first)\n"
        f"  Format: {args.format}  |  Hero: {args.hero_id}\n"
        f"  Opponent mode: {args.opponent_mode}\n"
        f"  Sideboard eval: {args.sideboard_eval_games} C++ games/ep\n"
        f"  Iterations: {args.iterations}\n"
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

    if p1.card_pool and not p1.sideboard_converged:
        print(f"\n{'#'*62}\n  STAGE 1 — Sideboard-first (fixed pool)\n{'#'*62}")
        for sb_round in range(1, args.max_sideboard_rounds + 1):
            print(f"\n  Sideboard round {sb_round}/{args.max_sideboard_rounds}")
            run_phase2_sideboard(
                p1,
                p1_opponents,
                hero_id=args.hero_id,
                hero_class=args.hero_class,
                equipment_header=args.equipment_header,
                game_format=args.format,
                opponent_deck_name=_p1_opponent_deck(),
                n_episodes_per_opponent=args.sideboard_episodes,
                max_sideboard_steps=args.max_sideboard_steps,
                sideboard_eval_games=args.sideboard_eval_games,
                assets_path=assets_path,
                base_url=args.talishar_url,
                render=args.render,
                cpp_engine_dir=args.cpp_engine_dir,
                sideboard_batch_size=args.sideboard_batch_size,
                convergence_window=args.sideboard_convergence_window,
                convergence_std=args.sideboard_convergence_std,
            )
            if p1.sideboard_converged:
                break
        if not p1.sideboard_converged:
            print("  WARNING: sideboard did not converge — enabling deckbuild anyway")
            p1.sideboard_converged = True

    for iteration in range(1, args.iterations + 1):
        print(f"\n\n{'#'*62}\n  ITERATION {iteration} / {args.iterations}\n{'#'*62}")

        if args.play_reward_p1 is not None:
            p1_play_reward = args.play_reward_p1
        else:
            p1_play_reward, _ = load_play_feedback(out_dir)
        if p2 is not None:
            if args.play_reward_p2 is not None:
                p2_play_reward = args.play_reward_p2
            else:
                _, p2_play_reward = load_play_feedback(out_dir)
        else:
            p2_play_reward = 0.0

        deckbuild_enabled = (
            (p1.sideboard_converged or not p1.card_pool)
            and not p1_fixed_deck
        )
        if deckbuild_enabled:
            p1_warm = p1.card_pool if iteration > 1 else p1_starting_deck
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
                starting_deck=p1_warm,
                play_reward=p1_play_reward,
            )
            p1.sideboard_converged = False
        elif p1_fixed_deck:
            p1.card_pool = dict(p1_fixed_deck)
            print(f"\n  [p1] Deckbuild skipped — fixed deck ({sum(p1_fixed_deck.values())} cards)")

        if args.opponent_mode == "dual" and p2 is not None:
            if (p2.sideboard_converged or not p2.card_pool) and not p2_fixed_deck:
                p2_warm = p2.card_pool if iteration > 1 else p2_starting_deck
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
                    starting_deck=p2_warm,
                    play_reward=p2_play_reward,
                )
                p2.sideboard_converged = False
            elif p2_fixed_deck:
                p2.card_pool = dict(p2_fixed_deck)
                print(f"\n  [p2] Deckbuild skipped — fixed deck ({sum(p2_fixed_deck.values())} cards)")

        if p1_fixed_deck and sum(p1_fixed_deck.values()) >= min_size:
            p1.active_decks[p1_opp_key] = greedy_game_deck_cut(p1_fixed_deck, min_size)
            print(f"\n  [p1] Sideboard skipped — fixed deck is game-ready")
        else:
            run_phase2_sideboard(
                p1,
                p1_opponents,
                hero_id=args.hero_id,
                hero_class=args.hero_class,
                equipment_header=args.equipment_header,
                game_format=args.format,
                opponent_deck_name=_p1_opponent_deck(),
                n_episodes_per_opponent=args.sideboard_episodes,
                max_sideboard_steps=args.max_sideboard_steps,
                sideboard_eval_games=args.sideboard_eval_games,
                assets_path=assets_path,
                base_url=args.talishar_url,
                render=args.render,
                cpp_engine_dir=args.cpp_engine_dir,
                sideboard_batch_size=args.sideboard_batch_size,
                convergence_window=args.sideboard_convergence_window,
                convergence_std=args.sideboard_convergence_std,
                play_reward=p1_play_reward,
            )

        if args.opponent_mode == "dual" and p2 is not None:
            if p2_fixed_deck and sum(p2_fixed_deck.values()) >= min_size:
                p2.active_decks[args.hero_id] = greedy_game_deck_cut(p2_fixed_deck, min_size)
                print(f"\n  [p2] Sideboard skipped — fixed deck is game-ready")
            else:
                run_phase2_sideboard(
                    p2,
                    p2_opponents,
                    hero_id=args.p2_hero_id,
                    hero_class=args.p2_hero_class,
                    equipment_header=args.p2_equipment_header,
                    game_format=args.format,
                    opponent_deck_name=_p2_opponent_deck(),
                    n_episodes_per_opponent=args.sideboard_episodes,
                    max_sideboard_steps=args.max_sideboard_steps,
                    sideboard_eval_games=args.sideboard_eval_games,
                    assets_path=assets_path,
                    base_url=args.talishar_url,
                    render=args.render,
                    cpp_engine_dir=args.cpp_engine_dir,
                    sideboard_batch_size=args.sideboard_batch_size,
                    convergence_window=args.sideboard_convergence_window,
                    convergence_std=args.sideboard_convergence_std,
                    play_reward=p2_play_reward,
                )

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

    print(f"\n  Deckbuild + sideboard training complete → {out_dir}")
    print(f"  Deck state written → {deck_state_path}")
    print("  Next: python scripts/training/train_play.py --out-dir", out_dir)


if __name__ == "__main__":
    main()
