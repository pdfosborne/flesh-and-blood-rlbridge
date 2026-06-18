#!/usr/bin/env python3
"""Simulate a Flesh and Blood deck matchup and report win percentages.

Given two decks (FaBrary URLs/slugs or local JSON files), this script:

  1. Fetches deck data from FaBrary (or uses local JSON files directly).
  2. Parses hero IDs, classes, and equipment headers from the deck JSON.
  3. Builds the C++ engine for fast simulation (optional but recommended).
  4. Handles sideboard automatically.
  5. Trains play agents for both decks via Phase 3 (no deckbuilding).
  6. Runs a final evaluation to report win percentages.

Set TALISHAR_URL / TALISHAR_ASSETS_PATH / FABRARY_API_KEY in the environment as needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from flesh_and_blood_rlbridge.cpp_engine_environment import is_cpp_engine_available

from runscripts._common import (
    MIN_DECK_SIZES,
    REPO_ROOT,
    SCRIPTS_TRAINING,
    assets_path,
    build_cpp_engine_for_matchup,
    describe_deck_size,
    fetch_fabrary_deck,
    find_cpp_engine_dir_for_decks,
    optional_cpp_engine_arg,
    optional_train_workers_arg,
    print_banner,
    print_matchup_player_result,
    read_deck_meta,
    resolve_deck_source,
    run_python,
    talishar_url,
    title_case_token,
)

# ─── Default configuration ───────────────────────────────────────────────────

DEFAULT_DECK1_SOURCE = "https://fabrary.net/decks/01KTBBVEZE0TPDAZ74Z4D787G4"
DEFAULT_DECK2_SOURCE = "https://fabrary.net/decks/01KR0XXRF5MESBQWQH7FW5Y8MG"
DEFAULT_FORMAT = "silver_age"
MAX_PLAY_STEPS = 500
PLAY_EPISODES = 1_000_000
FINAL_EVAL_EPISODES = 100
FINAL_EVAL_MAX_STEPS = 500
SIDEBOARD_EPISODES = 30
NUM_EVAL_GAMES = 50
WARMUP_EPISODES = 50
ITERATIONS = 1
PLAY_WORKERS: int | None = None


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "deck1_source",
        nargs="?",
        default=DEFAULT_DECK1_SOURCE,
        help="FaBrary URL, 26-char deck slug, or path to a local JSON file (deck 1 / P1).",
    )
    parser.add_argument(
        "deck2_source",
        nargs="?",
        default=DEFAULT_DECK2_SOURCE,
        help="FaBrary URL, 26-char deck slug, or path to a local JSON file (deck 2 / P2).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))

    talishar_url_value = talishar_url()
    assets = assets_path()
    matchup_root = REPO_ROOT / "results" / "matchup_sims"
    deck_dir = matchup_root / "decks"
    deck_dir.mkdir(parents=True, exist_ok=True)

    print_banner("FaB Deck Matchup Simulator")
    print(f"  Deck 1 : {args.deck1_source}")
    print(f"  Deck 2 : {args.deck2_source}")
    print(f"  Format : {DEFAULT_FORMAT}")
    print()

    deck1_info = resolve_deck_source(args.deck1_source, deck_dir)
    deck2_info = resolve_deck_source(args.deck2_source, deck_dir)

    deck1_ok = True
    deck2_ok = True
    if deck1_info.slug:
        deck1_ok = fetch_fabrary_deck(deck1_info.slug, deck1_info.local_path, "Deck 1")
    else:
        print(f"  [Deck 1] Using local file: {deck1_info.local_path}")
    if deck2_info.slug:
        deck2_ok = fetch_fabrary_deck(deck2_info.slug, deck2_info.local_path, "Deck 2")
    else:
        print(f"  [Deck 2] Using local file: {deck2_info.local_path}")

    if not deck1_ok or not deck1_info.local_path.is_file():
        print("ERROR: Deck 1 JSON not available. Aborting.")
        return 1
    if not deck2_ok or not deck2_info.local_path.is_file():
        print("ERROR: Deck 2 JSON not available. Aborting.")
        return 1

    p1_meta = read_deck_meta(deck1_info.local_path, DEFAULT_FORMAT)
    p2_meta = read_deck_meta(deck2_info.local_path, DEFAULT_FORMAT)

    matchup_label = f"{p1_meta.short_name}_vs_{p2_meta.short_name}"
    out_dir = matchup_root / matchup_label
    results_json = out_dir / "results.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    effective_format = p1_meta.fmt
    min_size = MIN_DECK_SIZES.get(effective_format, 40)

    print()
    print(f"  Deck 1 : {p1_meta.name}")
    print(f"           Hero: {p1_meta.hero_id} | Class: {p1_meta.hero_class}")
    print(f"           Cards in JSON deck: {p1_meta.total_cards}")
    describe_deck_size(p1_meta, min_size)
    print()
    print(f"  Deck 2 : {p2_meta.name}")
    print(f"           Hero: {p2_meta.hero_id} | Class: {p2_meta.hero_class}")
    print(f"           Cards in JSON deck: {p2_meta.total_cards}")
    describe_deck_size(p2_meta, min_size)
    print()
    print(f"  Output dir : {out_dir}")
    print()

    build_deck1 = title_case_token(p1_meta.short_name)
    build_deck2 = title_case_token(p2_meta.short_name)
    print(f"  Building C++ engine for {p1_meta.short_name} vs {p2_meta.short_name}...")
    print()

    build_rc = build_cpp_engine_for_matchup(
        deck1=build_deck1,
        deck2=build_deck2,
        talishar_url_value=talishar_url_value,
        deck1_json=deck1_info.local_path,
        deck2_json=deck2_info.local_path,
        no_server=True,
    )
    cpp_engine_dir = None
    if build_rc == 0:
        cpp_engine_dir = find_cpp_engine_dir_for_decks(
            build_deck1,
            build_deck2,
            deck1_json=deck1_info.local_path,
            deck2_json=deck2_info.local_path,
        )
        if cpp_engine_dir is None:
            print("  WARNING: C++ build reported success but engine directory not found.")
        elif not is_cpp_engine_available(cpp_engine_dir):
            print(
                f"  WARNING: C++ engine at {cpp_engine_dir} is not importable — "
                "forcing rebuild for the active Python."
            )
            rebuild_rc = build_cpp_engine_for_matchup(
                deck1=build_deck1,
                deck2=build_deck2,
                talishar_url_value=talishar_url_value,
                deck1_json=deck1_info.local_path,
                deck2_json=deck2_info.local_path,
                no_server=True,
            )
            if rebuild_rc == 0:
                cpp_engine_dir = find_cpp_engine_dir_for_decks(
                    build_deck1,
                    build_deck2,
                    deck1_json=deck1_info.local_path,
                    deck2_json=deck2_info.local_path,
                )
        else:
            print(f"  Engine directory: {cpp_engine_dir}")
            print("  C++ engine ready.")
            print()
    else:
        print()
        print("  WARNING: C++ engine build failed -- falling back to HTTP Talishar.")
        print()

    print("  Starting simulation ...")
    print("  Both decks are FIXED (no deckbuilding phase).")
    print("  Phase 3 (play) trains both agents then runs final evaluation.")
    print()

    train_args = [
        "--format",
        effective_format,
        "--hero-id",
        p1_meta.hero_id,
        "--hero-class",
        p1_meta.hero_class,
        "--equipment-header",
        p1_meta.equipment_header,
        "--opponent-mode",
        "dual",
        "--p2-hero-id",
        p2_meta.hero_id,
        "--p2-hero-class",
        p2_meta.hero_class,
        "--p2-equipment-header",
        p2_meta.equipment_header,
        "--opponent-hero-id",
        p2_meta.hero_id,
        "--deckbuild-episodes",
        "0",
        "--sideboard-episodes",
        str(SIDEBOARD_EPISODES),
        "--play-episodes",
        str(PLAY_EPISODES),
        "--max-play-steps",
        str(MAX_PLAY_STEPS),
        "--num-eval-games",
        str(NUM_EVAL_GAMES),
        "--warmup-episodes",
        str(WARMUP_EPISODES),
        "--warmup-baseline-eval-episodes",
        "20",
        "--iterations",
        str(ITERATIONS),
        "--final-eval-episodes",
        str(FINAL_EVAL_EPISODES),
        "--final-eval-max-steps",
        str(FINAL_EVAL_MAX_STEPS),
        "--gif-fps",
        "2.0",
        "--talishar-url",
        talishar_url_value,
        "--assets-path",
        str(assets),
        "--out-dir",
        str(out_dir),
        "--results-json",
        str(results_json),
        "--p1-fixed-deck",
        str(deck1_info.local_path),
        "--p2-fixed-deck",
        str(deck2_info.local_path),
        *optional_cpp_engine_arg(cpp_engine_dir),
        *optional_train_workers_arg(PLAY_WORKERS),
    ]
    rc = run_python(SCRIPTS_TRAINING / "train_full_pipeline.py", *train_args)
    if rc != 0:
        print()
        print(f"ERROR: simulation exited with code {rc}")
        return rc

    print_banner(f"Simulation Results -- {matchup_label}")
    if results_json.is_file():
        results = json.loads(results_json.read_text(encoding="utf-8"))
        print_matchup_player_result(
            "Deck 1 (P1)",
            p1_meta.name,
            results.get("p1", {}),
            final_eval_episodes=FINAL_EVAL_EPISODES,
        )
        print_matchup_player_result(
            "Deck 2 (P2)",
            p2_meta.name,
            results.get("p2", {}),
            final_eval_episodes=FINAL_EVAL_EPISODES,
        )

        p1_final_rate = None
        p2_final_rate = None
        p1 = results.get("p1", {})
        p2 = results.get("p2", {})
        if isinstance(p1.get("final_eval"), dict):
            p1_final_rate = float(p1["final_eval"].get("win_rate", 0.0))
        elif p1.get("win_rates"):
            p1_final_rate = float(p1["win_rates"][-1])
        if isinstance(p2.get("final_eval"), dict):
            p2_final_rate = float(p2["final_eval"].get("win_rate", 0.0))
        elif p2.get("win_rates"):
            p2_final_rate = float(p2["win_rates"][-1])

        if p1_final_rate is not None and p2_final_rate is not None:
            p1_pct = round(p1_final_rate * 100, 1)
            p2_pct = round(p2_final_rate * 100, 1)
            print()
            print(f"  Head-to-head : P1 {p1_pct}%  vs  P2 {p2_pct}%")
            if p1_final_rate > p2_final_rate:
                print(f"  Verdict      : Deck 1 ({p1_meta.name}) wins the matchup")
            elif p2_final_rate > p1_final_rate:
                print(f"  Verdict      : Deck 2 ({p2_meta.name}) wins the matchup")
            else:
                print("  Verdict      : Even matchup")
    else:
        print(f"  WARNING: results.json not found at {results_json}")

    print()
    print(f"  Full results : {results_json}")
    print(f"  Output dir   : {out_dir}")
    print(f"  Render GIFs  : {out_dir / 'final_eval' / 'p1_optimal_policy.gif'}")
    print(f"               : {out_dir / 'final_eval' / 'p2_optimal_policy.gif'}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
