#!/usr/bin/env python3
"""Train and evaluate the full 3-phase RL pipeline for Aurora vs Briar (Silver Age).

Phase 1 — Deckbuilder: optimise the 55-card registered pool for each hero
Phase 2 — Sideboard:   select the 40-card game deck vs the opposing hero
Phase 3 — Play:        co-evolution self-play (dual mode)

FaBrary decks used as warm-start pools:
    Aurora: https://fabrary.net/decks/01KST88R7JVEQ73M82ZA0PJ9RN
    Briar:  https://fabrary.net/decks/01KGZPKM6NBVNFYEEWWS4SGFQ7

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
OUT_DIR = REPO_ROOT / "results" / "auroraPaulvanGijssel_vs_briarAjanell"
DECK_DIR = OUT_DIR / "sage_decks"
RESULTS_JSON = OUT_DIR / "results.json"

AURORA_SLUG = "01KST88R7JVEQ73M82ZA0PJ9RN"
BRIAR_SLUG = "01KGZPKM6NBVNFYEEWWS4SGFQ7"
AURORA_DECK_JSON = DECK_DIR / "aurora_PaulvanGijssel_deck.json"
BRIAR_DECK_JSON = DECK_DIR / "briar_Ajanell_deck.json"

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

DECKBUILD_EPISODES = 1000
SIDEBOARD_EPISODES = 1000
PLAY_EPISODES = 10000
NUM_EVAL_GAMES = 1000
NUM_SIDEBOARD_EPISODES = 1000
ITERATIONS = 20
PLAY_WORKERS: int | None = None

FINAL_EVAL_EPISODES = 100
FINAL_EVAL_MAX_STEPS = 200
GIF_FPS = 1.0


def main() -> int:
    DECK_DIR.mkdir(parents=True, exist_ok=True)

    print_banner("Aurora vs Briar — Silver Age — 3-Phase RL Pipeline", width=56)
    print(f"  Talishar URL : {TALISHAR_URL_VALUE}")
    print(f"  Assets path  : {ASSETS_PATH}")
    print(f"  Output dir   : {OUT_DIR}")
    print()

    aurora_fetched = fetch_fabrary_deck(AURORA_SLUG, AURORA_DECK_JSON, "Aurora")
    briar_fetched = fetch_fabrary_deck(BRIAR_SLUG, BRIAR_DECK_JSON, "Briar")

    starting_deck_args: list[str] = []
    if aurora_fetched and AURORA_DECK_JSON.is_file():
        starting_deck_args.extend(["--p1-starting-deck", str(AURORA_DECK_JSON)])
    if briar_fetched and BRIAR_DECK_JSON.is_file():
        starting_deck_args.extend(["--p2-starting-deck", str(BRIAR_DECK_JSON)])

    print()
    print("  Building C++ engine for Aurora vs Briar...")
    print()

    build_rc = build_cpp_engine_for_matchup(
        deck1=AURORA_HERO_ID,
        deck2=BRIAR_HERO_ID,
        talishar_url_value=TALISHAR_URL_VALUE,
        deck1_json=AURORA_DECK_JSON if AURORA_DECK_JSON.is_file() else None,
        deck2_json=BRIAR_DECK_JSON if BRIAR_DECK_JSON.is_file() else None,
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
            print(
                "  Expected runtime backend: C++ engine "
                "(with automatic HTTP fallback if unavailable per matchup)."
            )
            print()
    else:
        print()
        print(f"  WARNING: C++ engine build failed (exit {build_rc}).")
        print("  Continuing — training will fall back to HTTP Talishar.")
        print("  Fix errors above and re-run to get the speed benefit.")
        print()

    print()
    print("  Starting train_full_pipeline.py ...")
    if cpp_engine_dir is not None:
        print(
            "  Backend selection: C++ preferred; "
            "train_full_pipeline.py will print actual runtime backend."
        )
    else:
        print(
            "  Backend selection: HTTP Talishar expected; "
            "train_full_pipeline.py will print actual runtime backend."
        )
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
        *starting_deck_args,
    ]
    rc = run_python(SCRIPTS_TRAINING / "train_full_pipeline.py", *train_args)
    if rc != 0:
        print()
        print(f"ERROR: train_full_pipeline.py exited with code {rc}")
        return rc

    print_banner("Training complete", width=56)
    results = load_results_json(RESULTS_JSON)
    if results is not None:
        print(f"  Results -> {RESULTS_JSON}")
        print()
        print("  Aurora (P1)")
        print_final_eval("", results.get("p1", {}))
        print()
        print("  Briar (P2)")
        print_final_eval("", results.get("p2", {}))

    print()
    print(f"  Agents saved to: {OUT_DIR}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
