#!/usr/bin/env python3
"""Compare guide-policy sideboard swaps via parallel play training.

Resolves a starting deck, generates sideboard candidates, and runs
``train_sideboard_compare.py`` to pick the best list vs a fixed opponent.

    python runscripts/sideboard_compare.py aurora briar \\
        --starting-deck path/to/aurora.json

    python runscripts/sideboard_compare.py --hero aurora --opponent briar \\
        --starting-deck path/to/aurora.json --num-options 4 --max-parallel 2
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
    optional_cpp_engine_arg,
    optional_play_batch_size_arg,
    optional_train_workers_arg,
    print_banner,
    read_deck_meta,
    resolve_hero_starting_deck,
    run_python,
    start_sideboard_compare_dashboard,
    stop_background_process,
    talishar_url,
    title_case_token,
)

DEFAULT_NUM_OPTIONS = 4
DEFAULT_MAX_PARALLEL = 2
DEFAULT_PLAY_EPISODES = 10000
DEFAULT_FORMAT = "silver_age"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sideboard variants with parallel play training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("hero", nargs="?", default="aurora", help="Hero slug")
    parser.add_argument("opponent", nargs="?", default="briar", help="Opponent slug")
    parser.add_argument("--hero", dest="hero_flag", default=None)
    parser.add_argument("--opponent", dest="opponent_flag", default=None)
    parser.add_argument("--starting-deck", default=None,
        help="Deck JSON (FaBrary export or local path)")
    parser.add_argument("--card-pool", default=None,
        help="Optional registered pool JSON")
    parser.add_argument("--format", default=DEFAULT_FORMAT,
        choices=["silver_age", "sage", "classic_constructed", "blitz", "upf"])
    parser.add_argument("--num-options", type=int, default=DEFAULT_NUM_OPTIONS)
    parser.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL)
    parser.add_argument("--play-episodes", type=int, default=DEFAULT_PLAY_EPISODES)
    parser.add_argument("--checkpoint-interval-pct", type=float, default=5.0,
        help="Checkpoint every N%% of play episodes")
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=100,
        help="C++ eval games with fixed opponent at each checkpoint (0=off)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--play-batch-size", type=int, default=None)
    parser.add_argument("--opponent-deck", default=None,
        help="Talishar Assets deck stem (default: <Opponent>SAGEPrecon when present)")
    parser.add_argument("--no-dashboard", action="store_true",
        help="Do not launch the live HTML dashboard watcher")
    parser.add_argument("--dashboard-poll-seconds", type=float, default=5.0)
    return parser.parse_args(argv)


def _pipeline_format(fmt: str) -> str:
    token = fmt.strip().lower()
    if token in {"sage", "silver_age"}:
        return "silver_age"
    return token


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    hero_token = hero_slug(args.hero_flag or args.hero)
    opponent_token = hero_slug(args.opponent_flag or args.opponent)
    game_format = _pipeline_format(args.format)

    talishar_url_value = talishar_url()
    assets = assets_path()
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else REPO_ROOT / "results" / f"sideboard_compare_{hero_token}_vs_{opponent_token}"
    )
    deck_dir = out_dir / "decks"
    deck_dir.mkdir(parents=True, exist_ok=True)

    hero_label = title_case_token(hero_token)
    opponent_label = title_case_token(opponent_token)

    print_banner(
        f"{hero_label} sideboard compare vs {opponent_label}",
        width=58,
    )
    print(f"  Options       : {args.num_options}")
    print(f"  Max parallel  : {args.max_parallel}")
    print(f"  Play episodes : {args.play_episodes}")
    print(f"  Checkpoints   : every {args.checkpoint_interval_pct:g}%  "
          f"| eval {args.checkpoint_eval_episodes} games @ fixed opponent")
    print(f"  Output        : {out_dir}")
    print()

    deck_path = resolve_hero_starting_deck(
        hero_token,
        deck_source=args.starting_deck,
        deck_dir=deck_dir,
        assets_path=assets,
        label=hero_label,
        game_format=game_format,
    )
    if deck_path is None:
        print("ERROR: could not resolve starting deck.")
        print("  Pass --starting-deck or ensure a SAGE precon exists in Assets.")
        return 1

    meta = read_deck_meta(deck_path, default_format=game_format)
    opponent_deck = args.opponent_deck or f"{opponent_label}SAGEPrecon"

    print(f"  Building C++ engine for {hero_label} vs {opponent_label}...")
    build_rc = build_cpp_engine_for_matchup(
        deck1=meta.hero_id or hero_token,
        deck2=opponent_token,
        talishar_url_value=talishar_url_value,
        deck1_json=deck_path,
        no_server=True,
    )
    cpp_engine_dir = None
    if build_rc == 0:
        cpp_engine_dir = find_cpp_engine_dir_for_decks(
            meta.hero_id or hero_token,
            opponent_token,
            deck1_json=deck_path,
        )
        if cpp_engine_dir:
            print(f"  Engine directory: {cpp_engine_dir}")
    else:
        print(f"  WARNING: C++ build failed (exit {build_rc}); continuing without engine.")

    train_args = [
        "--format", game_format,
        "--hero-id", meta.hero_id or hero_token,
        "--hero-class", meta.hero_class,
        "--equipment-header", meta.equipment_header,
        "--opponent-hero-id", opponent_token,
        "--opponent-deck", opponent_deck,
        "--starting-deck", str(deck_path),
        "--num-options", str(args.num_options),
        "--max-parallel", str(args.max_parallel),
        "--play-episodes", str(args.play_episodes),
        "--checkpoint-interval-pct", str(args.checkpoint_interval_pct),
        "--checkpoint-eval-episodes", str(args.checkpoint_eval_episodes),
        "--out-dir", str(out_dir),
        "--cache-dir", str(REPO_ROOT / "results" / "agent_cache"),
        "--talishar-url", talishar_url_value,
        "--assets-path", str(assets),
        *optional_cpp_engine_arg(cpp_engine_dir),
        *optional_train_workers_arg(args.workers),
        *optional_play_batch_size_arg(args.play_batch_size),
    ]
    if args.card_pool:
        train_args.extend(["--card-pool", args.card_pool])

    dashboard_proc = None
    if not args.no_dashboard:
        dashboard_proc = start_sideboard_compare_dashboard(
            out_dir,
            poll_seconds=args.dashboard_poll_seconds,
        )
        print(f"  Dashboard     : {out_dir / 'sideboard_compare_dashboard.html'}")
        print()

    try:
        rc = run_python(SCRIPTS_TRAINING / "train_sideboard_compare.py", *train_args)
    finally:
        stop_background_process(dashboard_proc)

    if rc != 0:
        print(f"ERROR: train_sideboard_compare.py exited with code {rc}")
        return rc

    print_banner("Sideboard comparison complete", width=58)
    print(f"  Results -> {out_dir / 'sideboard_compare_results.json'}")
    if not args.no_dashboard:
        print(f"  Dashboard -> {out_dir / 'sideboard_compare_dashboard.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
