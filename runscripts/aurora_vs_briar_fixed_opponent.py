#!/usr/bin/env python3
"""Find the best Aurora deck against a specific fixed Briar opponent deck.

Aurora trains all three phases (deckbuilder + sideboard + play agent).
Briar is fixed to a known FaBrary deck — no deckbuilding, no sideboarding.

Fixed opponent deck (Briar):
    https://fabrary.net/decks/01KTBBVEZE0TPDAZ74Z4D787G4

Aurora warm-start deck:
    https://fabrary.net/decks/01KST88R7JVEQ73M82ZA0PJ9RN

Set TALISHAR_URL / TALISHAR_ASSETS_PATH / FABRARY_API_KEY in the environment as needed.
"""

from __future__ import annotations

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
    fetch_fabrary_deck,
    find_cpp_engine_dir,
    load_results_json,
    optional_cpp_engine_arg,
    optional_train_workers_arg,
    print_banner,
    print_final_eval,
    run_python,
    talishar_url,
)

# ─── Configuration ───────────────────────────────────────────────────────────

TALISHAR_URL_VALUE = talishar_url()
ASSETS_PATH = assets_path()
OUT_DIR = REPO_ROOT / "results" / "aurora_vs_briar_fixed"
DECK_DIR = OUT_DIR / "sage_decks"
RESULTS_JSON = OUT_DIR / "results.json"

AURORA_SLUG = "01KST88R7JVEQ73M82ZA0PJ9RN"
BRIAR_FIXED_SLUG = "01KTBBVEZE0TPDAZ74Z4D787G4"
AURORA_DECK_JSON = DECK_DIR / "aurora_warmstart_deck.json"
BRIAR_FIXED_DECK_JSON = DECK_DIR / "briar_fixed_deck.json"

AURORA_HERO_ID = "aurora"
AURORA_HERO_CLASS = "Runeblade"
AURORA_EQUIPMENT = (
    "aurora star_fall aether_ironweave spellbound_creepers "
    "aether_crackers crown_of_dichotomy"
)

BRIAR_HERO_ID = "briar"
BRIAR_HERO_CLASS = "Runeblade"
BRIAR_EQUIPMENT = (
    "briar star_fall aether_ironweave spellbound_creepers "
    "aether_crackers crown_of_dichotomy"
)

DECKBUILD_EPISODES = 10
SIDEBOARD_EPISODES = 10
PLAY_EPISODES = 100
NUM_EVAL_GAMES = 20
NUM_SIDEBOARD_EPISODES = 10
ITERATIONS = 3
PLAY_WORKERS: int | None = None

FINAL_EVAL_EPISODES = 100
FINAL_EVAL_MAX_STEPS = 200
GIF_FPS = 1.0


def main() -> int:
    DECK_DIR.mkdir(parents=True, exist_ok=True)

    print_banner("Aurora (training) vs Briar (fixed deck) — Silver Age")
    print("  Mode         : Aurora trains all phases; Briar is FIXED")
    print(f"  Talishar URL : {TALISHAR_URL_VALUE}")
    print(f"  Assets path  : {ASSETS_PATH}")
    print(f"  Output dir   : {OUT_DIR}")
    print()

    aurora_fetched = fetch_fabrary_deck(AURORA_SLUG, AURORA_DECK_JSON, "Aurora (warm-start)")
    briar_fetched = fetch_fabrary_deck(
        BRIAR_FIXED_SLUG, BRIAR_FIXED_DECK_JSON, "Briar  (fixed opponent)"
    )

    extra_args: list[str] = []
    if aurora_fetched and AURORA_DECK_JSON.is_file():
        extra_args.extend(["--p1-starting-deck", str(AURORA_DECK_JSON)])

    if briar_fetched and BRIAR_FIXED_DECK_JSON.is_file():
        extra_args.extend(["--p2-fixed-deck", str(BRIAR_FIXED_DECK_JSON)])
        print()
        print(f"  Briar opponent deck pinned to: {BRIAR_FIXED_DECK_JSON}")
        print("  Phase 1 (deckbuilder) and Phase 2 (sideboard) will be skipped for Briar.")
    else:
        print()
        print("  WARNING: Briar fixed deck not available — Briar will train deckbuilder/sideboard too.")

    print()
    print("  Building C++ engine for Aurora vs Briar...")
    print()

    build_rc = build_cpp_engine_for_matchup(
        deck1=AURORA_HERO_ID,
        deck2=BRIAR_HERO_ID,
        talishar_url_value=TALISHAR_URL_VALUE,
        deck1_json=AURORA_DECK_JSON if AURORA_DECK_JSON.is_file() else None,
        deck2_json=BRIAR_FIXED_DECK_JSON if BRIAR_FIXED_DECK_JSON.is_file() else None,
        no_server=True,
    )
    cpp_engine_dir = None
    if build_rc == 0:
        cpp_engine_dir = find_cpp_engine_dir("aurora_vs_briar")
        if cpp_engine_dir is None:
            print("  WARNING: C++ build reported success but engine directory not found.")
        else:
            print(f"  Engine directory: {cpp_engine_dir}")
            print("  C++ engine build succeeded.")
            print()
    else:
        print()
        print("  WARNING: C++ engine build failed — falling back to HTTP Talishar.")
        print()

    print("  Starting train_full_pipeline.py ...")
    print("  Aurora: trains Phase 1 (deckbuilder) + Phase 2 (sideboard) + Phase 3 (play)")
    print("  Briar:  fixed deck — only Phase 3 (play agent) trains")
    print()

    train_args = [
        "--format",
        "silver_age",
        "--hero-id",
        AURORA_HERO_ID,
        "--hero-class",
        AURORA_HERO_CLASS,
        "--equipment-header",
        AURORA_EQUIPMENT,
        "--opponent-mode",
        "dual",
        "--p2-hero-id",
        BRIAR_HERO_ID,
        "--p2-hero-class",
        BRIAR_HERO_CLASS,
        "--p2-equipment-header",
        BRIAR_EQUIPMENT,
        "--deckbuild-episodes",
        str(DECKBUILD_EPISODES),
        "--sideboard-episodes",
        str(SIDEBOARD_EPISODES),
        "--play-episodes",
        str(PLAY_EPISODES),
        "--num-eval-games",
        str(NUM_EVAL_GAMES),
        "--num-sideboard-episodes",
        str(NUM_SIDEBOARD_EPISODES),
        "--iterations",
        str(ITERATIONS),
        "--final-eval-episodes",
        str(FINAL_EVAL_EPISODES),
        "--final-eval-max-steps",
        str(FINAL_EVAL_MAX_STEPS),
        "--gif-fps",
        str(GIF_FPS),
        "--talishar-url",
        TALISHAR_URL_VALUE,
        "--assets-path",
        str(ASSETS_PATH),
        "--out-dir",
        str(OUT_DIR),
        "--results-json",
        str(RESULTS_JSON),
        *optional_cpp_engine_arg(cpp_engine_dir),
        *optional_train_workers_arg(PLAY_WORKERS),
        *extra_args,
    ]
    rc = run_python(SCRIPTS_TRAINING / "train_full_pipeline.py", *train_args)
    if rc != 0:
        print()
        print(f"ERROR: train_full_pipeline.py exited with code {rc}")
        return rc

    print_banner("Training complete")
    results = load_results_json(RESULTS_JSON)
    if results is not None:
        print(f"  Results -> {RESULTS_JSON}")
        print()
        print("  Aurora (P1 — trained deckbuilder + sideboard + play)")
        print_final_eval("", results.get("p1", {}))
        print()
        print("  Briar (P2 — fixed deck, play agent only)")
        print_final_eval("", results.get("p2", {}))

    print()
    print(f"  Agents + results saved to: {OUT_DIR}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
