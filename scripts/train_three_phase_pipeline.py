#!/usr/bin/env python3
"""Three-phase FaB pipeline: Deckbuilder → Sideboard → Play.

Phase 1 – **Deckbuilder agent** constructs a registered card pool.
  * Silver Age: builds 55-card pool (40 main + 15 inventory).
  * Classic Constructed: builds 80-card pool (60+ main + 20 sideboard).

Phase 2 – **Sideboard agent** selects the per-opponent game deck.
  * Given the built pool and an opponent hero identity, the agent decides
    which cards move into the active 40 (SA) / 60 (CC) card game deck.
  * Runs once per opponent in ``opponent_heroes``.

Phase 3 – **Play agent** (already implemented in existing training scripts)
  evaluates the final deck win rate via Talishar self-play.

Usage
-----
    # Silver Age (Ira vs Dorinthea + Fai)
    python scripts/train_three_phase_pipeline.py

    # Classic Constructed
    python scripts/train_three_phase_pipeline.py --format classic_constructed \\
        --hero-class Ninja --hero-id ira_crimson_haze

    # Provide an existing agent checkpoint to pre-load the deckbuilder
    python scripts/train_three_phase_pipeline.py \\
        --deckbuilder-agent results/deckbuild_agents/ira_sa_deckbuilder.pkl \\
        --sideboard-agent  results/sideboard_agents/ira_sa_sideboard.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flesh_and_blood_rlbridge import (  # noqa: E402
    TalisharDeckBuilderEnvironment,
    TalisharSideboardEnvironment,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_FORMAT = "silver_age"
_DEFAULT_HERO_ID = "ira_crimson_haze"
_DEFAULT_HERO_CLASS = "Ninja"
_DEFAULT_EQUIPMENT_HEADER = (
    "ira_crimson_haze harmonized_kodachi harmonized_kodachi "
    "blade_beckoner_helm blood_scent tearing_shuko pouncing_paws"
)

# Opponents to sideboard for
_DEFAULT_OPPONENTS = [
    "dorinthea_ironsong",
    "fai_rising_rebellion",
    "bravo_showman",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_agent(path: str | None) -> Any | None:
    """Load a pickled agent from *path*, or return None."""
    if path is None or not Path(path).exists():
        return None
    import pickle  # noqa: PLC0415

    with open(path, "rb") as fh:
        return pickle.load(fh)  # noqa: S301


def _save_agent(agent: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    import pickle  # noqa: PLC0415

    with open(path, "wb") as fh:
        pickle.dump(agent, fh)


def run_deckbuilder_phase(
    *,
    game_format: str,
    hero_id: str,
    hero_class: str,
    equipment_header: str,
    n_episodes: int,
    max_build_steps: int,
    num_eval_games: int,
    agent: Any | None,
    out_path: str | None,
    render: bool,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Run Phase 1: build the card pool.

    Returns ``(card_pool, pool_by_id)`` from the final episode.
    """
    print(
        f"\n{'='*60}\n"
        f"PHASE 1 — Deckbuilder  [{hero_id} / {game_format}]\n"
        f"{'='*60}"
    )

    env = TalisharDeckBuilderEnvironment(
        hero_id=hero_id,
        hero_class=hero_class,
        hero_equipment_header=equipment_header,
        game_format=game_format,
        num_eval_games=num_eval_games,
        max_build_steps=max_build_steps,
        render_mode="ansi" if render else None,
    )

    best_reward = float("-inf")
    best_pool: dict[str, int] = {}

    for ep in range(n_episodes):
        result = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            # Use the provided agent or fall back to random action sampling.
            if agent is not None and hasattr(agent, "act"):
                action = agent.act(result.observation)
            else:
                avail = json.loads(result.observation).get("availableActions", [])
                import random  # noqa: PLC0415
                action = random.choice(avail) if avail else "finalize"

            step = env.step(action)
            ep_reward += step.reward
            done = step.terminated or step.truncated

            if render and not done:
                render_result = env.render()
                if render_result.text:
                    print(render_result.text)

            result.observation = step.observation

        pool = env.get_card_pool()
        pool_size = sum(pool.values())
        print(
            f"  Ep {ep+1:>4}/{n_episodes}  reward={ep_reward:+.3f}  "
            f"pool={pool_size}"
        )

        if ep_reward > best_reward and pool_size >= env._min_deck_size:
            best_reward = ep_reward
            best_pool = dict(pool)

    if out_path and agent is not None:
        _save_agent(agent, out_path)
        print(f"  Deckbuilder agent saved → {out_path}")

    print(f"\n  Best pool: {sum(best_pool.values())} cards  (reward {best_reward:+.3f})")
    return best_pool, env._pool_by_id


def run_sideboard_phase(
    *,
    card_pool: dict[str, int],
    pool_by_id: dict[str, Any],
    game_format: str,
    hero_id: str,
    equipment_header: str,
    opponent_heroes: list[str],
    n_episodes_per_opponent: int,
    max_sideboard_steps: int,
    num_eval_games: int,
    agent: Any | None,
    out_path: str | None,
    render: bool,
) -> dict[str, dict[str, int]]:
    """Run Phase 2: sideboard selection for each opponent.

    Returns a dict mapping ``opponent_hero_id → selected game deck``.
    """
    print(
        f"\n{'='*60}\n"
        f"PHASE 2 — Sideboard  [{hero_id} / {game_format}]\n"
        f"{'='*60}"
    )

    results: dict[str, dict[str, int]] = {}

    for opponent in opponent_heroes:
        print(f"\n  Opponent: {opponent}")

        env = TalisharSideboardEnvironment(
            card_pool=card_pool,
            pool_by_id=pool_by_id,
            opponent_hero_id=opponent,
            hero_id=hero_id,
            hero_equipment_header=equipment_header,
            game_format=game_format,
            num_eval_games=num_eval_games,
            max_sideboard_steps=max_sideboard_steps,
            render_mode="ansi" if render else None,
        )

        best_reward = float("-inf")
        best_deck: dict[str, int] = {}

        for ep in range(n_episodes_per_opponent):
            result = env.reset()
            ep_reward = 0.0
            done = False

            while not done:
                if agent is not None and hasattr(agent, "act"):
                    action = agent.act(result.observation)
                else:
                    avail = json.loads(result.observation).get("availableActions", [])
                    import random  # noqa: PLC0415
                    action = random.choice(avail) if avail else "finalize"

                step = env.step(action)
                ep_reward += step.reward
                done = step.terminated or step.truncated

                if render and not done:
                    render_result = env.render()
                    if render_result.text:
                        print(render_result.text)

                result.observation = step.observation

            deck = env.get_active_deck()
            deck_size = sum(deck.values())
            print(
                f"    Ep {ep+1:>4}/{n_episodes_per_opponent}  "
                f"reward={ep_reward:+.3f}  deck={deck_size}"
            )

            if ep_reward > best_reward and deck_size >= env._min_deck_size:
                best_reward = ep_reward
                best_deck = dict(deck)

        results[opponent] = best_deck
        print(
            f"  → Best deck vs {opponent}: "
            f"{sum(best_deck.values())} cards  (reward {best_reward:+.3f})"
        )

    if out_path and agent is not None:
        _save_agent(agent, out_path)
        print(f"\n  Sideboard agent saved → {out_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-phase FaB pipeline: Deckbuilder → Sideboard → Play",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--format", default=_DEFAULT_FORMAT)
    parser.add_argument("--hero-id", default=_DEFAULT_HERO_ID)
    parser.add_argument("--hero-class", default=_DEFAULT_HERO_CLASS)
    parser.add_argument("--equipment-header", default=_DEFAULT_EQUIPMENT_HEADER)
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=_DEFAULT_OPPONENTS,
        help="Opponent hero IDs to sideboard for",
    )

    # Phase 1 — deckbuilder
    parser.add_argument("--deckbuild-episodes", type=int, default=50)
    parser.add_argument("--max-build-steps", type=int, default=200)
    parser.add_argument("--deckbuilder-agent", default=None, help="Path to agent pkl")
    parser.add_argument(
        "--deckbuilder-out",
        default=str(_REPO_ROOT / "results" / "deckbuild_agents" / "deckbuilder.pkl"),
    )

    # Phase 2 — sideboard
    parser.add_argument("--sideboard-episodes", type=int, default=30)
    parser.add_argument("--max-sideboard-steps", type=int, default=100)
    parser.add_argument("--sideboard-agent", default=None, help="Path to agent pkl")
    parser.add_argument(
        "--sideboard-out",
        default=str(_REPO_ROOT / "results" / "sideboard_agents" / "sideboard.pkl"),
    )

    # Shared
    parser.add_argument("--num-eval-games", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--results-json",
        default=str(_REPO_ROOT / "results" / "three_phase_results.json"),
        help="Path to write the per-opponent deck selection summary JSON",
    )

    args = parser.parse_args()

    db_agent = _load_agent(args.deckbuilder_agent)
    sb_agent = _load_agent(args.sideboard_agent)

    # ── Phase 1: build the pool ───────────────────────────────────────────────
    card_pool, pool_by_id = run_deckbuilder_phase(
        game_format=args.format,
        hero_id=args.hero_id,
        hero_class=args.hero_class,
        equipment_header=args.equipment_header,
        n_episodes=args.deckbuild_episodes,
        max_build_steps=args.max_build_steps,
        num_eval_games=args.num_eval_games,
        agent=db_agent,
        out_path=args.deckbuilder_out,
        render=args.render,
    )

    if not card_pool:
        print(
            "\nDeckbuilder produced an empty pool — check the card DB and format rules."
        )
        sys.exit(1)

    # ── Phase 2: sideboard for each opponent ─────────────────────────────────
    decks_by_opponent = run_sideboard_phase(
        card_pool=card_pool,
        pool_by_id=pool_by_id,
        game_format=args.format,
        hero_id=args.hero_id,
        equipment_header=args.equipment_header,
        opponent_heroes=args.opponents,
        n_episodes_per_opponent=args.sideboard_episodes,
        max_sideboard_steps=args.max_sideboard_steps,
        num_eval_games=args.num_eval_games,
        agent=sb_agent,
        out_path=args.sideboard_out,
        render=args.render,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Format : {args.format}")
    print(f"Hero   : {args.hero_id}")
    print(f"Pool   : {sum(card_pool.values())} cards")
    for opp, deck in decks_by_opponent.items():
        print(f"  vs {opp:<35} → {sum(deck.values())} cards in game deck")

    # Write results JSON
    out_json = {
        "format": args.format,
        "hero_id": args.hero_id,
        "card_pool": card_pool,
        "decks_by_opponent": {
            opp: deck for opp, deck in decks_by_opponent.items()
        },
    }
    results_path = Path(args.results_json)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print(f"\nResults written → {results_path}")


if __name__ == "__main__":
    main()
