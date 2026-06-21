#!/usr/bin/env python3
"""Silver Age (SAGE) dual deckbuilder — guide sideboard + play + deckbuild.

Two heroes co-train deckbuilder and play in dual mode.
Sideboard selection uses SideboardGuidePolicy (no sideboard RL training).

Defaults: Aurora (P1) vs Briar (P2).

    python runscripts/sage_deckbuilder.py
    python runscripts/sage_deckbuilder.py dorinthea kayo
    python runscripts/sage_deckbuilder.py --p1-hero aurora --p2-hero briar \\
        --p1-deck path/to/aurora.json --p2-deck 01KGZPKM6NBVNFYEEWWS4SGFQ7

From main.py:

    python main.py sage                    # interactive deck picker (TTY)
    python main.py sage --defaults         # Aurora vs Briar, no prompts
    python main.py sage --list-decks       # show available SAGE precons
    python main.py sage dorinthea kayo     # explicit heroes
    python main.py preset sage-deckbuilder -- dorinthea kayo

Set TALISHAR_URL / TALISHAR_ASSETS_PATH / FABRARY_API_KEY in the environment as needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runscripts._common import (
    REPO_ROOT,
    SCRIPTS_TRAINING,
    assets_path,
    build_cpp_engine_for_matchup,
    find_cpp_engine_dir_for_decks,
    hero_slug,
    load_results_json,
    optional_cpp_engine_arg,
    optional_play_batch_size_arg,
    optional_train_workers_arg,
    print_banner,
    print_final_eval,
    read_deck_meta,
    resolve_hero_starting_deck,
    run_python,
    talishar_url,
    title_case_token,
)

# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_P1_HERO = "aurora"
DEFAULT_P2_HERO = "briar"
DEFAULT_FORMAT = "silver_age"

DECKBUILD_EPISODES = 1000
NUM_EVAL_GAMES = 3

PLAY_EPISODES = 10000
ITERATIONS = 20
PLAY_WORKERS: int | None = None
PLAY_BATCH_SIZE: int | None = None

FINAL_EVAL_EPISODES = 100
FINAL_EVAL_MAX_STEPS = 200
GIF_FPS = 1.0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Silver Age (SAGE) dual deckbuilder pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "p1_hero",
        nargs="?",
        default=DEFAULT_P1_HERO,
        help="P1 hero id or slug (default: aurora)",
    )
    parser.add_argument(
        "p2_hero",
        nargs="?",
        default=DEFAULT_P2_HERO,
        help="P2 hero id or slug (default: briar)",
    )
    parser.add_argument(
        "--p1-hero",
        dest="p1_hero_flag",
        default=None,
        help="Override P1 hero (alternative to positional arg)",
    )
    parser.add_argument(
        "--p2-hero",
        dest="p2_hero_flag",
        default=None,
        help="Override P2 hero (alternative to positional arg)",
    )
    parser.add_argument(
        "--p1-deck",
        default=None,
        help="P1 warm-start: FaBrary URL/slug or local deck JSON path",
    )
    parser.add_argument(
        "--p2-deck",
        default=None,
        help="P2 warm-start: FaBrary URL/slug or local deck JSON path",
    )
    parser.add_argument(
        "--format",
        default=DEFAULT_FORMAT,
        choices=["silver_age", "sage", "classic_constructed", "blitz", "upf"],
        help="Game format (sage is an alias for silver_age)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Results directory (default: results/sage_<p1>_vs_<p2>)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel C++ play sessions (auto-detected from CPU count when omitted)",
    )
    parser.add_argument(
        "--play-batch-size",
        type=int,
        default=None,
        help="Episodes per parallel batch (defaults to --workers)",
    )
    return parser.parse_args(argv)


def _resolved_heroes(args: argparse.Namespace) -> tuple[str, str]:
    p1 = args.p1_hero_flag or args.p1_hero
    p2 = args.p2_hero_flag or args.p2_hero
    return hero_slug(p1), hero_slug(p2)


def _pipeline_format(fmt: str) -> str:
    token = fmt.strip().lower()
    if token in {"sage", "silver_age"}:
        return "silver_age"
    return token


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    p1_token, p2_token = _resolved_heroes(args)
    game_format = _pipeline_format(args.format)

    talishar_url_value = talishar_url()
    assets = assets_path()
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else REPO_ROOT / "results" / f"sage_{p1_token}_vs_{p2_token}"
    )
    deck_dir = out_dir / "sage_decks"
    results_json = out_dir / "results.json"
    deck_dir.mkdir(parents=True, exist_ok=True)

    p1_label = title_case_token(p1_token)
    p2_label = title_case_token(p2_token)

    print_banner(
        f"{p1_label} vs {p2_label} — Silver Age — Play + Deckbuild Pipeline",
        width=56,
    )
    print(f"  Talishar URL : {talishar_url_value}")
    print(f"  Assets path  : {assets}")
    print(f"  Output dir   : {out_dir}")
    print(f"  Format       : {game_format}")
    print()
    print("  Stage 1: SideboardGuidePolicy → initial play training")
    print(f"  Then {ITERATIONS} iterations: deckbuild → guide sideboard → play")
    print()

    p1_deck_path = resolve_hero_starting_deck(
        p1_token,
        deck_source=args.p1_deck,
        deck_dir=deck_dir,
        assets_path=assets,
        label=p1_label,
        game_format=game_format,
    )
    p2_deck_path = resolve_hero_starting_deck(
        p2_token,
        deck_source=args.p2_deck,
        deck_dir=deck_dir,
        assets_path=assets,
        label=p2_label,
        game_format=game_format,
    )

    if p1_deck_path is None or p2_deck_path is None:
        print()
        print("ERROR: could not resolve warm-start decks for both heroes.")
        print("  Provide --p1-deck / --p2-deck or ensure Talishar SAGE precons exist.")
        return 1

    p1_meta = read_deck_meta(p1_deck_path, default_format=game_format)
    p2_meta = read_deck_meta(p2_deck_path, default_format=game_format)

    starting_deck_args = [
        "--p1-starting-deck",
        str(p1_deck_path),
        "--p2-starting-deck",
        str(p2_deck_path),
    ]

    print()
    print(f"  Building C++ engine for {p1_label} vs {p2_label}...")
    print()

    build_rc = build_cpp_engine_for_matchup(
        deck1=p1_meta.hero_id or p1_token,
        deck2=p2_meta.hero_id or p2_token,
        talishar_url_value=talishar_url_value,
        deck1_json=p1_deck_path,
        deck2_json=p2_deck_path,
        no_server=True,
    )
    cpp_engine_dir = None
    if build_rc == 0:
        cpp_engine_dir = find_cpp_engine_dir_for_decks(
            p1_meta.hero_id or p1_token,
            p2_meta.hero_id or p2_token,
            deck1_json=p1_deck_path,
            deck2_json=p2_deck_path,
        )
        if cpp_engine_dir is None:
            print("  WARNING: C++ build reported success but engine directory not found.")
        else:
            print(f"  Engine directory: {cpp_engine_dir}")
            print("  C++ engine build succeeded.")
            print()
    else:
        print()
        print(f"  WARNING: C++ engine build failed (exit {build_rc}).")
        print("  Continuing — training will fall back to HTTP Talishar.")
        print()

    print()
    print("  Starting train_full_pipeline.py ...")
    print()

    train_args = [
        "--format",
        game_format,
        "--hero-id",
        p1_meta.hero_id or p1_token,
        "--hero-class",
        p1_meta.hero_class,
        "--equipment-header",
        p1_meta.equipment_header,
        "--opponent-mode",
        "dual",
        "--p2-hero-id",
        p2_meta.hero_id or p2_token,
        "--p2-hero-class",
        p2_meta.hero_class,
        "--p2-equipment-header",
        p2_meta.equipment_header,
        "--deckbuild-episodes",
        str(DECKBUILD_EPISODES),
        "--play-episodes",
        str(PLAY_EPISODES),
        "--num-eval-games",
        str(NUM_EVAL_GAMES),
        "--iterations",
        str(ITERATIONS),
        "--final-eval-episodes",
        str(FINAL_EVAL_EPISODES),
        "--final-eval-max-steps",
        str(FINAL_EVAL_MAX_STEPS),
        "--gif-fps",
        str(GIF_FPS),
        "--talishar-url",
        talishar_url_value,
        "--assets-path",
        str(assets),
        "--out-dir",
        str(out_dir),
        "--results-json",
        str(results_json),
        *optional_cpp_engine_arg(cpp_engine_dir),
        *optional_train_workers_arg(args.workers if args.workers is not None else PLAY_WORKERS),
        *optional_play_batch_size_arg(
            args.play_batch_size if args.play_batch_size is not None else PLAY_BATCH_SIZE
        ),
        *starting_deck_args,
    ]
    rc = run_python(SCRIPTS_TRAINING / "train_full_pipeline.py", *train_args)
    if rc != 0:
        print()
        print(f"ERROR: train_full_pipeline.py exited with code {rc}")
        return rc

    print_banner("Training complete", width=56)
    results = load_results_json(results_json)
    if results is not None:
        print(f"  Results -> {results_json}")
        print()
        print(f"  {p1_label} (P1)")
        print_final_eval("", results.get("p1", {}))
        print()
        print(f"  {p2_label} (P2)")
        print_final_eval("", results.get("p2", {}))

    print()
    print(f"  Agents saved to: {out_dir}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
