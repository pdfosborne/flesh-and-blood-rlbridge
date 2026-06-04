#!/usr/bin/env python3
"""Full three-phase FaB RL training pipeline.

Phase 1 — Deckbuilder agent
    Builds a registered card pool (55 cards for Silver Age, 80 for Classic
    Constructed) card by card.  On finalize, Phase 2 runs automatically inside
    the deckbuilder's evaluation step to select the best game deck, which is
    then used for Phase 3 play.

Phase 2 — Sideboard agent
    Given the built pool and an opponent hero identity, selects which cards
    form the active game deck (≥40 / ≥60 cards depending on format).  Can be
    pre-trained separately or left as random (acts as a baseline).

Phase 3 — Play agent
    Plays Talishar self-play games with the sidebaorded deck.  Supports three
    opponent modes:
      ``preset``  — fixed Talishar Assets deck (CombatDummy-style baseline)
      ``mirror``  — the same built deck on both sides (mirror match)
      ``dual``    — a full second pipeline (deckbuilder + sideboard + play)
                    trains in parallel for both players simultaneously

Usage
-----
    # Quickstart: Silver Age, Ira vs preset Dorinthea opponent
    python scripts/train_full_pipeline.py

    # Mirror match (Ira vs copy of own built deck)
    python scripts/train_full_pipeline.py --opponent-mode mirror

    # Dual / co-evolution: both players train all three phases simultaneously
    python scripts/train_full_pipeline.py --opponent-mode dual \\
        --p2-hero-id dorinthea_ironsong --p2-hero-class Warrior

    # Classic Constructed
    python scripts/train_full_pipeline.py --format classic_constructed \\
        --hero-class Ninja --hero-id ira_crimson_haze \\
        --opponent-mode dual --p2-hero-id dorinthea_ironsong --p2-hero-class Warrior

    # Resume from saved agents
    python scripts/train_full_pipeline.py \\
        --p1-deckbuilder results/full_pipeline/p1_deckbuilder.pkl \\
        --p1-sideboard   results/full_pipeline/p1_sideboard.pkl \\
        --p1-play        results/full_pipeline/p1_play.pkl

Windows (PowerShell):
    $env:TALISHAR_URL="http://localhost:8080/game"
    python scripts/train_full_pipeline.py --opponent-mode dual
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import uuid
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
_FAB_SRC = _REPO_ROOT / "src"
_RL_SRC = Path("~/Documents/RL/rlbridge/src").expanduser()

for _p in (_FAB_SRC, _RL_SRC, str(_SCRIPTS_DIR)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from flesh_and_blood_rlbridge import (  # noqa: E402
    TalisharDeckBuilderEnvironment,
    TalisharSideboardEnvironment,
    TalisharEngineEnvironment,
)

try:
    from train_dual_agent_common import (  # noqa: E402
        make_agent,
        make_env,
        Matchup,
        run_matchup_training,
        train_agents_from_both_perspectives,
        train_agents_from_both_perspectives_parallel,
        _evaluate_policy_pair,
        _save_warmup_handoff_checkpoint,
        DEFAULT_N_EPISODES,
        DEFAULT_WARMUP_EPISODES,
        DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
    )
    from episode_cache import EpisodeCache  # noqa: E402
    _DUAL_AGENT_AVAILABLE = True
except ImportError:
    _DUAL_AGENT_AVAILABLE = False
    DEFAULT_N_EPISODES = 300
    DEFAULT_WARMUP_EPISODES = 50
    DEFAULT_WARMUP_BASELINE_EVAL_EPISODES = 20

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
_DEFAULT_OPPONENT_DECK = "Ira"
_DEFAULT_OPPONENT_HERO = "dorinthea_ironsong"

_OUT_DIR = _REPO_ROOT / "results" / "full_pipeline"


# ---------------------------------------------------------------------------
# Agent persistence
# ---------------------------------------------------------------------------


def _load_agent(path: Optional[str]) -> Optional[Any]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("rb") as fh:
            agent = pickle.load(fh)  # noqa: S301
        print(f"  Loaded agent from {p}")
        return agent
    except Exception as exc:
        print(f"  WARNING: could not load agent from {p}: {exc}")
        return None


def _load_starting_deck(path: Optional[str]) -> Optional[dict[str, int]]:
    """Load a starting deck from a JSON file produced by fetch_fabrary_deck.py.

    The JSON must contain a ``"deck"`` key mapping card IDs to counts.
    Returns ``None`` if ``path`` is falsy or the file cannot be read.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: starting-deck file not found: {p}")
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        deck: dict[str, int] = data.get("deck", {})
        if not deck:
            print(f"  WARNING: 'deck' field is empty in {p}")
            return None
        total = sum(deck.values())
        print(f"  Loaded starting deck from {p}  ({total} cards)")
        return {str(k): int(v) for k, v in deck.items()}
    except Exception as exc:
        print(f"  WARNING: could not load starting deck from {p}: {exc}")
        return None


def _save_agent(agent: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(agent, fh)
    print(f"  Saved agent → {path}")


# ---------------------------------------------------------------------------
# Phase data containers
# ---------------------------------------------------------------------------


@dataclass
class PhaseAgents:
    """Holds all three phase agents for one player."""
    player: str  # "p1" or "p2"
    deckbuilder: Optional[Any] = None
    sideboard: Optional[Any] = None
    play: Optional[Any] = None
    # Outputs from completed phases
    card_pool: dict[str, int] = field(default_factory=dict)
    pool_by_id: dict[str, Any] = field(default_factory=dict)
    # Per-opponent active decks produced by the sideboard phase
    active_decks: dict[str, dict[str, int]] = field(default_factory=dict)
    # Deck file name written to Talishar Assets for play evaluation
    deck_asset_name: str = ""
    win_rates: list[float] = field(default_factory=list)
    # Win rate from the most recent Phase-3 play evaluation.
    # This is the TRUE reward signal that feeds back into Phase 1 & 2:
    # deck construction and sideboard selection are only meaningful if the
    # resulting deck wins in actual play.
    last_play_win_rate: float = 0.0


# ---------------------------------------------------------------------------
# Phase 1 — Deckbuilder
# ---------------------------------------------------------------------------


def run_phase1_deckbuilder(
    agents: PhaseAgents,
    *,
    hero_id: str,
    hero_class: str,
    equipment_header: str,
    game_format: str,
    opponent_deck_name: str,
    opponent_hero_id: str,
    n_episodes: int,
    max_build_steps: int,
    num_eval_games: int,
    num_sideboard_episodes: int,
    assets_path: Optional[str],
    base_url: str,
    render: bool,
    starting_deck: Optional[dict[str, int]] = None,
    play_reward: float = 0.0,
) -> None:
    """Train the deckbuilder agent and store the best pool on ``agents``.

    ``play_reward`` is the win rate from the most recent Phase-3 play
    evaluation (0.0 on the first iteration when no play has run yet).
    When non-zero it is added to the environment reward as the terminal
    signal — so the deckbuilder learns to build pools that win in actual
    play rather than just passing a quick internal evaluation.

    On iterations 2+ (play_reward > 0) we reduce ``num_eval_games`` to 1
    so the internal Talishar eval is fast; the real score comes from play.
    """
    # On iteration 1 (play_reward == 0.0) skip internal eval entirely — there
    # is no play baseline yet so any eval score would be noise.  Reward will
    # be 0.0 for any valid pool.  On later iterations use a lightweight eval;
    # the real signal comes from Phase-3 play win rate.
    effective_eval_games = (
        0 if play_reward == 0.0 else max(1, num_eval_games // 3)
    )
    print(
        f"\n{'='*62}\n"
        f"  PHASE 1 — Deckbuilder  [{agents.player} / {hero_id} / {game_format}]\n"
        f"{'='*62}"
    )
    if play_reward > 0.0:
        print(
            f"  [{agents.player}] Play feedback: last win rate = {play_reward:.1%}  "
            f"(added as terminal reward bonus)"
        )

    env = TalisharDeckBuilderEnvironment(
        hero_id=hero_id,
        hero_class=hero_class,
        hero_equipment_header=equipment_header,
        game_format=game_format,
        num_eval_games=effective_eval_games,
        opponent_deck_name=opponent_deck_name,
        opponent_hero_id=opponent_hero_id,
        sideboard_agent=agents.sideboard,
        num_sideboard_episodes=num_sideboard_episodes,
        base_url=base_url,
        talishar_assets_path=assets_path,
        max_build_steps=max_build_steps,
        starting_deck=starting_deck,
        render_mode="ansi" if render else None,
    )

    best_reward = float("-inf")
    best_pool: dict[str, int] = {}

    for ep in range(1, n_episodes + 1):
        result = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            if agents.deckbuilder is not None and hasattr(agents.deckbuilder, "act"):
                action = agents.deckbuilder.act(result.observation)
            else:
                avail = json.loads(result.observation).get("availableActions", [])
                import random  # noqa: PLC0415
                action = random.choice(avail) if avail else "finalize"

            step = env.step(action)
            ep_reward += step.reward
            done = step.terminated or step.truncated

            if render and not done:
                r = env.render()
                if r.text:
                    print(r.text)

            result.observation = step.observation

        pool = env.get_card_pool()
        pool_size = sum(pool.values())
        valid = pool_size >= env._min_deck_size

        # Add play win rate as terminal reward bonus so the deckbuilder
        # learns to build pools that win in actual play, not just pass
        # internal evaluation.  On iteration 1 play_reward == 0.0 so this
        # has no effect.
        if play_reward != 0.0 and valid:
            ep_reward += play_reward

        if ep % max(1, n_episodes // 10) == 0 or ep == n_episodes:
            play_tag = f"  play_bonus={play_reward:+.3f}" if play_reward != 0.0 else ""
            print(
                f"  [{agents.player}] Ep {ep:>4}/{n_episodes}  "
                f"reward={ep_reward:+.3f}  pool={pool_size}  valid={valid}"
                + play_tag
            )

        if valid and ep_reward > best_reward:
            best_reward = ep_reward
            best_pool = dict(pool)

    if not best_pool:
        # Fallback: use last pool even if suboptimal
        best_pool = env.get_card_pool()

    agents.card_pool = best_pool
    agents.pool_by_id = env._pool_by_id
    print(
        f"\n  [{agents.player}] Best pool: {sum(best_pool.values())} cards  "
        f"(reward {best_reward:+.3f})"
    )


# ---------------------------------------------------------------------------
# Phase 2 — Sideboard
# ---------------------------------------------------------------------------


def run_phase2_sideboard(
    agents: PhaseAgents,
    opponent_hero_ids: list[str],
    *,
    hero_id: str,
    equipment_header: str,
    game_format: str,
    opponent_deck_name: str,
    n_episodes_per_opponent: int,
    max_sideboard_steps: int,
    num_eval_games: int,
    assets_path: Optional[str],
    base_url: str,
    render: bool,
    play_reward: float = 0.0,
) -> None:
    """Train the sideboard agent for each opponent, store best decks on ``agents``.

    ``play_reward`` is the win rate from the most recent Phase-3 play
    evaluation.  When non-zero it is added as a terminal reward bonus so
    the sideboard agent learns to select decks that win in play, not just
    decks that score well on quick internal evaluations.
    """
    # On iteration 1 (play_reward == 0.0) skip internal eval — no play
    # baseline exists yet.  Reward is 0.0 for any valid deck selection.
    # On later iterations use a lightweight eval; play win rate is the signal.
    effective_eval_games = (
        0 if play_reward == 0.0 else max(1, num_eval_games // 3)
    )
    print(
        f"\n{'='*62}\n"
        f"  PHASE 2 — Sideboard  [{agents.player} / {hero_id} / {game_format}]\n"
        f"{'='*62}"
    )

    if play_reward != 0.0:
        print(
            f"  [play feedback] last play win-rate = {play_reward:+.3f}  "
            f"(will be added as terminal bonus; eval_games reduced to {effective_eval_games})"
        )

    if not agents.card_pool:
        print(f"  [{agents.player}] WARNING: empty card pool — skipping sideboard phase")
        return

    for opponent in opponent_hero_ids:
        print(f"\n  [{agents.player}] Opponent: {opponent}")

        env = TalisharSideboardEnvironment(
            card_pool=agents.card_pool,
            pool_by_id=agents.pool_by_id,
            opponent_hero_id=opponent,
            hero_id=hero_id,
            hero_equipment_header=equipment_header,
            game_format=game_format,
            num_eval_games=effective_eval_games,
            opponent_deck_name=opponent_deck_name,
            eval_p1_agent=agents.play,
            base_url=base_url,
            talishar_assets_path=assets_path,
            max_sideboard_steps=max_sideboard_steps,
            render_mode="ansi" if render else None,
        )
        best_reward = float("-inf")
        best_deck: dict[str, int] = {}

        for ep in range(1, n_episodes_per_opponent + 1):
            result = env.reset()
            ep_reward = 0.0
            done = False

            while not done:
                if agents.sideboard is not None and hasattr(agents.sideboard, "act"):
                    action = agents.sideboard.act(result.observation)
                else:
                    # Greedy fallback: fill deck to min size, then finalize.
                    # A pure-random agent too often moves cards back out and
                    # never reaches a valid 40-card deck.
                    obs_data = json.loads(result.observation)
                    deck_sz = obs_data.get("deckSize", 0)
                    min_sz = obs_data.get("minDeckSize", 40)
                    avail = obs_data.get("availableActions", [])
                    if deck_sz < min_sz:
                        move_acts = [a for a in avail if a.startswith("move_to_deck:")]
                        action = move_acts[0] if move_acts else "finalize"
                    else:
                        action = "finalize"

                step = env.step(action)
                ep_reward += step.reward
                done = step.terminated or step.truncated

                if render and not done:
                    r = env.render()
                    if r.text:
                        print(r.text)

                result.observation = step.observation

            deck = env.get_active_deck()
            deck_size = sum(deck.values())
            valid = deck_size >= env._min_deck_size

            # Add play win-rate as terminal bonus so good play-results
            # reinforce the sideboard choices that led to them.
            if valid and play_reward != 0.0:
                ep_reward += play_reward

            if ep % max(1, n_episodes_per_opponent // 5) == 0 or ep == n_episodes_per_opponent:
                play_tag = f"  play_bonus={play_reward:+.3f}" if play_reward != 0.0 else ""
                print(
                    f"    [{agents.player}] Ep {ep:>4}/{n_episodes_per_opponent}  "
                    f"reward={ep_reward:+.3f}  deck={deck_size}  valid={valid}{play_tag}"
                )

            if valid and ep_reward > best_reward:
                best_reward = ep_reward
                best_deck = dict(deck)

        agents.active_decks[opponent] = best_deck if best_deck else dict(agents.card_pool)
        print(
            f"  [{agents.player}] → vs {opponent}: "
            f"{sum(agents.active_decks[opponent].values())} cards  "
            f"(reward {best_reward:+.3f})"
        )


# ---------------------------------------------------------------------------
# Phase 3 — Play  (preset / mirror)
# ---------------------------------------------------------------------------


def _write_deck_file(
    deck: dict[str, int],
    equipment_header: str,
    deck_name: str,
    assets_path: str,
) -> Path:
    card_ids: list[str] = []
    for card_id, count in sorted(deck.items()):
        card_ids.extend([card_id] * count)
    content = f"{equipment_header}\n{' '.join(card_ids)}\n"
    out_path = Path(assets_path) / f"{deck_name}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def run_phase3_play_preset(
    agents: PhaseAgents,
    opponent_hero_id: str,
    *,
    game_format: str,
    opponent_deck_name: str,
    equipment_header: str,
    n_episodes: int,
    max_play_steps: int,
    assets_path: str,
    base_url: str,
    render: bool,
) -> float:
    """Legacy preset play — delegates to run_phase3_play."""
    return run_phase3_play(
        agents, None,
        opponent_mode="preset",
        game_format=game_format,
        p1_hero_id="",
        p2_hero_id="",
        p1_equipment_header=equipment_header,
        p2_equipment_header="",
        p1_opponent_hero_id=opponent_hero_id,
        p1_opponent_deck_name=opponent_deck_name,
        n_episodes=n_episodes,
        max_play_steps=max_play_steps,
        warmup_episodes=DEFAULT_WARMUP_EPISODES,
        warmup_baseline_eval_episodes=DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
        n_workers=1,
        assets_path=assets_path,
        base_url=base_url,
        out_dir=Path(assets_path).parent,
        cache_dir=None,
        seed=None,
    )[0]


def run_phase3_play_mirror(
    agents: PhaseAgents,
    *,
    game_format: str,
    equipment_header: str,
    n_episodes: int,
    max_play_steps: int,
    assets_path: str,
    base_url: str,
    render: bool,
) -> float:
    """Legacy mirror play — delegates to run_phase3_play."""
    return run_phase3_play(
        agents, None,
        opponent_mode="mirror",
        game_format=game_format,
        p1_hero_id="",
        p2_hero_id="",
        p1_equipment_header=equipment_header,
        p2_equipment_header=equipment_header,
        p1_opponent_hero_id="",
        p1_opponent_deck_name="",
        n_episodes=n_episodes,
        max_play_steps=max_play_steps,
        warmup_episodes=DEFAULT_WARMUP_EPISODES,
        warmup_baseline_eval_episodes=DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
        n_workers=1,
        assets_path=assets_path,
        base_url=base_url,
        out_dir=Path(assets_path).parent,
        cache_dir=None,
        seed=None,
    )[0]


def run_phase3_play_dual(
    p1: PhaseAgents,
    p2: PhaseAgents,
    *,
    game_format: str,
    p1_equipment_header: str,
    p2_equipment_header: str,
    n_episodes: int,
    max_play_steps: int,
    assets_path: str,
    base_url: str,
    render: bool,
) -> tuple[float, float]:
    """Legacy dual play — delegates to run_phase3_play."""
    return run_phase3_play(
        p1, p2,
        opponent_mode="dual",
        game_format=game_format,
        p1_hero_id="",
        p2_hero_id="",
        p1_equipment_header=p1_equipment_header,
        p2_equipment_header=p2_equipment_header,
        p1_opponent_hero_id="",
        p1_opponent_deck_name="",
        n_episodes=n_episodes,
        max_play_steps=max_play_steps,
        warmup_episodes=DEFAULT_WARMUP_EPISODES,
        warmup_baseline_eval_episodes=DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
        n_workers=1,
        assets_path=assets_path,
        base_url=base_url,
        out_dir=Path(assets_path).parent,
        cache_dir=None,
        seed=None,
    )


# ---------------------------------------------------------------------------
# Phase 3 — Play  (unified, uses train_dual_agent_common warmup infrastructure)
# ---------------------------------------------------------------------------


def run_phase3_play(
    p1: PhaseAgents,
    p2: Optional[PhaseAgents],
    *,
    opponent_mode: str,               # "preset" | "mirror" | "dual"
    game_format: str,
    p1_hero_id: str,
    p2_hero_id: str,
    p1_equipment_header: str,
    p2_equipment_header: str,
    p1_opponent_hero_id: str,         # used to pick the right sideborded deck
    p1_opponent_deck_name: str,       # Talishar Assets deck name (preset mode)
    n_episodes: int,
    max_play_steps: int,
    warmup_episodes: int,             # default-policy warmup before PPO
    warmup_baseline_eval_episodes: int,
    n_workers: int,
    assets_path: str,
    base_url: str,
    out_dir: Path,
    cache_dir: Optional[Path],
    seed: Optional[int] = None,
) -> tuple[float, float]:
    """Co-evolution play using train_dual_agent_common warmup + episode-cache infrastructure.

    Both players always get PPO updates (in preset/mirror mode the "opponent"
    agent is discarded afterwards; only ``p1.play`` is updated).

    Returns ``(p1_win_rate, p2_win_rate)``.
    """
    print(
        f"\n{'='*62}\n"
        f"  PHASE 3 — Play ({opponent_mode})  [{p1.player}"
        + (f" vs {p2.player}" if p2 else "")
        + f"]\n{'='*62}"
    )

    if not _DUAL_AGENT_AVAILABLE:
        print("  WARNING: train_dual_agent_common not available — using fallback loop")
        return _run_phase3_fallback(
            p1, p2,
            opponent_mode=opponent_mode,
            game_format=game_format,
            p1_equipment_header=p1_equipment_header,
            p2_equipment_header=p2_equipment_header,
            p1_opponent_hero_id=p1_opponent_hero_id,
            p1_opponent_deck_name=p1_opponent_deck_name,
            n_episodes=n_episodes,
            max_play_steps=max_play_steps,
            assets_path=assets_path,
            base_url=base_url,
        )

    # ── select game decks ─────────────────────────────────────────────────────
    p1_game_deck = (
        p1.active_decks.get(p1_opponent_hero_id)
        or next(iter(p1.active_decks.values()), {})
        or p1.card_pool
    )
    if not p1_game_deck:
        print(f"  [{p1.player}] No deck available — skipping play phase")
        return 0.0, 0.0

    if opponent_mode == "dual" and p2 is not None:
        p2_game_deck = (
            p2.active_decks.get(p1_hero_id)
            or next(iter(p2.active_decks.values()), {})
            or p2.card_pool
        )
        if not p2_game_deck:
            print(f"  [{p2.player}] No deck available — skipping play phase")
            return 0.0, 0.0
    elif opponent_mode == "mirror":
        p2_game_deck = p1_game_deck
    else:
        p2_game_deck = None  # preset mode: p2_deck_name refers to assets file

    # ── write deck files ──────────────────────────────────────────────────────
    p1_deck_name = f"rl_p3_p1_{uuid.uuid4().hex[:8]}"
    p1_deck_file = _write_deck_file(p1_game_deck, p1_equipment_header, p1_deck_name, assets_path)

    if opponent_mode == "preset":
        p2_deck_name = p1_opponent_deck_name
        p2_deck_file: Optional[Path] = None
    elif opponent_mode == "mirror":
        p2_deck_name = p1_deck_name          # same file — mirror match
        p2_deck_file = None
    else:                                     # dual
        p2_deck_name = f"rl_p3_p2_{uuid.uuid4().hex[:8]}"
        p2_deck_file = _write_deck_file(
            p2_game_deck, p2_equipment_header, p2_deck_name, assets_path  # type: ignore[arg-type]
        )

    # ── Matchup + EpisodeCache ────────────────────────────────────────────────
    matchup = Matchup(
        name=f"p3_{p1_deck_name[-8:]}-vs-{p2_deck_name[-8:]}",
        p1_deck=p1_deck_name,
        p2_deck=p2_deck_name,
        description=f"Phase 3 play ({opponent_mode}): {p1.player} vs "
                    + (p2.player if p2 else p1_opponent_deck_name),
        p1_hero=p1_hero_id.replace("_", "-"),
        p2_hero=p2_hero_id.replace("_", "-"),
    )

    cache_root = cache_dir or (out_dir.parent / "agent_cache")
    episode_cache = EpisodeCache(cache_root=cache_root, game_format=game_format)

    _ep_cache_info = episode_cache.info(p1_deck_name, p2_deck_name)
    print(
        f"  Episode cache: {_ep_cache_info['total_episodes']} stored episode(s) "
        f"(skip threshold: {episode_cache.warmup_skip_threshold})"
    )

    # ── create / reuse play agents ────────────────────────────────────────────
    p1_agent = p1.play if p1.play is not None else make_agent(seed=seed)
    p2_seed = (seed + 1) if seed is not None else None
    p2_agent = (
        (p2.play if p2.play is not None else make_agent(seed=p2_seed))
        if (opponent_mode == "dual" and p2 is not None)
        else make_agent(seed=p2_seed)
    )
    p1_tiers: list[Any] = [p1_agent]
    p2_tiers: list[Any] = [p2_agent]

    live_path: Optional[Path] = out_dir / "play_live_state.png"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── training ──────────────────────────────────────────────────────────────
    p1_rewards: list[float] = []
    p2_rewards: list[float] = []

    try:
        if n_workers > 1:
            # Parallel: warmup + PPO in one call; baseline eval afterwards
            p1_r, p2_r, _ = train_agents_from_both_perspectives_parallel(
                matchup=matchup,
                base_url=base_url,
                game_format=game_format,
                p1_tiers=p1_tiers,
                p2_tiers=p2_tiers,
                n_episodes=n_episodes,
                max_steps=max_play_steps,
                seed=seed,
                warmup_episodes=warmup_episodes,
                n_workers=n_workers,
                live_state_image_path=live_path,
                episode_cache=episode_cache,
            )
            p1_rewards.extend(p1_r)
            p2_rewards.extend(p2_r)
            if warmup_episodes > 0 and warmup_baseline_eval_episodes > 0:
                _run_warmup_baseline(
                    matchup, p1_agent, p2_agent,
                    base_url=base_url, game_format=game_format,
                    max_steps=max_play_steps, out_dir=out_dir,
                    episodes=warmup_baseline_eval_episodes, seed=seed,
                )

        else:
            # Serial: explicit warmup phase → baseline eval → PPO phase
            env = make_env(
                matchup, base_url=base_url, game_format=game_format,
                max_turns=max_play_steps,
            )
            try:
                warmup_count = min(warmup_episodes, n_episodes)
                if warmup_count > 0:
                    print(f"  Warmup: {warmup_count} episode(s) with default policy…")
                    w_p1, w_p2, _ = train_agents_from_both_perspectives(
                        env, p1_tiers, p2_tiers,
                        n_episodes=warmup_count,
                        max_steps=max_play_steps,
                        seed=seed,
                        warmup_episodes=warmup_count,
                        live_state_image_path=live_path,
                        episode_cache=episode_cache,
                        p1_deck=p1_deck_name,
                        p2_deck=p2_deck_name,
                    )
                    p1_rewards.extend(w_p1)
                    p2_rewards.extend(w_p2)
                    if warmup_baseline_eval_episodes > 0:
                        _run_warmup_baseline(
                            matchup, p1_agent, p2_agent,
                            base_url=base_url, game_format=game_format,
                            max_steps=max_play_steps, out_dir=out_dir,
                            episodes=warmup_baseline_eval_episodes, seed=seed,
                        )

                remaining = n_episodes - warmup_count
                if remaining > 0:
                    print(f"  PPO: {remaining} episode(s) with learned policy…")
                    r_seed = (seed + warmup_count) if seed is not None else None
                    r_p1, r_p2, _ = train_agents_from_both_perspectives(
                        env, p1_tiers, p2_tiers,
                        n_episodes=remaining,
                        max_steps=max_play_steps,
                        seed=r_seed,
                        warmup_episodes=0,
                        live_state_image_path=live_path,
                        episode_cache=episode_cache,
                        p1_deck=p1_deck_name,
                        p2_deck=p2_deck_name,
                    )
                    p1_rewards.extend(r_p1)
                    p2_rewards.extend(r_p2)
            finally:
                env.close()

    finally:
        # Clean up temp deck files
        for f in [p1_deck_file] + ([p2_deck_file] if p2_deck_file else []):
            try:
                if f.exists():
                    f.unlink(missing_ok=True)
            except Exception:
                pass

    # ── update agents ─────────────────────────────────────────────────────────
    p1.play = p1_agent
    if opponent_mode == "dual" and p2 is not None:
        p2.play = p2_agent

    # ── win rates from reward signs ───────────────────────────────────────────
    p1_wr = (
        sum(1 for r in p1_rewards if r > 0) / max(1, len(p1_rewards))
        if p1_rewards else 0.0
    )
    p2_wr = (
        sum(1 for r in p2_rewards if r > 0) / max(1, len(p2_rewards))
        if p2_rewards else 0.0
    )
    p1.win_rates.append(p1_wr)
    if opponent_mode == "dual" and p2 is not None:
        p2.win_rates.append(p2_wr)

    print(f"\n  Win rates: p1={p1_wr:.1%}  p2={p2_wr:.1%}")
    return p1_wr, p2_wr


def _run_warmup_baseline(
    matchup: "Matchup",
    p1_agent: Any,
    p2_agent: Any,
    *,
    base_url: str,
    game_format: str,
    max_steps: int,
    out_dir: Path,
    episodes: int,
    seed: Optional[int],
) -> None:
    """Evaluate P1/P2 policies after warmup and save a handoff checkpoint."""
    print(f"  Warmup baseline eval: {episodes} episode(s)…")
    baseline = _evaluate_policy_pair(
        matchup,
        base_url=base_url,
        game_format=game_format,
        max_steps=max_steps,
        p1_policy=p1_agent,
        p2_policy=p2_agent,
        episodes=episodes,
        seed=(seed + 100_000) if seed is not None else None,
    )
    ckpt_dir = _save_warmup_handoff_checkpoint(
        out_dir=out_dir,
        matchup=matchup,
        p1_policy=p1_agent,
        p2_policy=p2_agent,
        baseline=baseline,
    )
    print(
        f"  Warmup baseline: P1 win%={baseline['p1_win_rate'] * 100:.1f}  "
        f"P2 win%={baseline['p2_win_rate'] * 100:.1f}  "
        f"draw%={baseline['draw_rate'] * 100:.1f}"
    )
    print(f"  Warmup checkpoint → {ckpt_dir}")


def _run_phase3_fallback(
    p1: PhaseAgents,
    p2: Optional[PhaseAgents],
    *,
    opponent_mode: str,
    game_format: str,
    p1_equipment_header: str,
    p2_equipment_header: str,
    p1_opponent_hero_id: str,
    p1_opponent_deck_name: str,
    n_episodes: int,
    max_play_steps: int,
    assets_path: str,
    base_url: str,
) -> tuple[float, float]:
    """Simple fallback play loop used when train_dual_agent_common is unavailable."""
    p1_game_deck = (
        p1.active_decks.get(p1_opponent_hero_id)
        or next(iter(p1.active_decks.values()), {})
        or p1.card_pool
    )
    if not p1_game_deck:
        return 0.0, 0.0

    if opponent_mode == "dual" and p2 is not None:
        p2_game_deck = next(iter(p2.active_decks.values()), {}) or p2.card_pool
    elif opponent_mode == "mirror":
        p2_game_deck = p1_game_deck
    else:
        p2_game_deck = None

    p1_deck_name = f"rl_fb_p1_{uuid.uuid4().hex[:8]}"
    p1_deck_file = _write_deck_file(p1_game_deck, p1_equipment_header, p1_deck_name, assets_path)
    if opponent_mode == "preset":
        p2_deck_name = p1_opponent_deck_name
        p2_deck_file = None
    elif opponent_mode == "mirror":
        p2_deck_name = p1_deck_name
        p2_deck_file = None
    else:
        p2_deck_name = f"rl_fb_p2_{uuid.uuid4().hex[:8]}"
        p2_deck_file = _write_deck_file(
            p2_game_deck, p2_equipment_header, p2_deck_name, assets_path  # type: ignore[arg-type]
        )

    p1_wins = 0
    try:
        for ep in range(1, n_episodes + 1):
            env = TalisharEngineEnvironment(
                base_url=base_url,
                game_format=game_format,
                local_deck_name=p1_deck_name,
                opponent_deck_name=p2_deck_name,
                max_turns=max_play_steps,
                self_play=True,
            )
            try:
                result = env.reset()
                done = False
                while not done:
                    obs_data = json.loads(result.observation)
                    acting_player = obs_data.get("actingPlayerID", 1)
                    agent = (p1.play if acting_player == 1 else (p2.play if p2 else None))
                    if agent is not None and hasattr(agent, "act"):
                        action = agent.act(result.observation)
                    else:
                        action = env.sample_action()
                    step = env.step(action)
                    done = step.terminated or step.truncated
                    result.observation = step.observation

                obs_data = json.loads(result.observation)
                if obs_data.get("playerHealth", 0) > 0 and obs_data.get("opponentHealth", 0) <= 0:
                    p1_wins += 1
            finally:
                env.close()
    finally:
        for f in [p1_deck_file] + ([p2_deck_file] if p2_deck_file else []):
            try:
                if f and f.exists():
                    f.unlink(missing_ok=True)
            except Exception:
                pass

    p1_wr = p1_wins / max(1, n_episodes)
    p2_wr = 1.0 - p1_wr
    p1.win_rates.append(p1_wr)
    if opponent_mode == "dual" and p2 is not None:
        p2.win_rates.append(p2_wr)
    print(f"\n  Fallback win rates: p1={p1_wr:.1%}  p2={p2_wr:.1%}")
    return p1_wr, p2_wr


# ---------------------------------------------------------------------------
# Save / summary helpers
# ---------------------------------------------------------------------------


def _save_all_agents(agents: PhaseAgents, out_dir: Path) -> None:
    prefix = out_dir / agents.player
    if agents.deckbuilder is not None:
        _save_agent(agents.deckbuilder, prefix.parent / f"{agents.player}_deckbuilder.pkl")
    if agents.sideboard is not None:
        _save_agent(agents.sideboard, prefix.parent / f"{agents.player}_sideboard.pkl")
    if agents.play is not None:
        _save_agent(agents.play, prefix.parent / f"{agents.player}_play.pkl")


# ---------------------------------------------------------------------------
# Final evaluation — eval games + optimal-policy render + GIF
# ---------------------------------------------------------------------------


def _frames_to_gif(frame_paths: list[Path], gif_path: Path, fps: float = 3.0) -> None:
    """Assemble PNG frame paths into an animated GIF (requires Pillow)."""
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        print("  WARNING: Pillow not installed — skipping GIF assembly.")
        print("           Install with: pip install Pillow")
        return
    frames: list[Any] = []
    for p in frame_paths:
        try:
            frames.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    if not frames:
        return
    duration_ms = max(1, int(1000.0 / fps))
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
    )
    print(f"  GIF saved ({len(frames)} frames, {fps} fps) → {gif_path}")


def _save_state_image(env: Any, obs: Any, out_path: Path) -> bool:
    """Save one render frame.  Falls back to a text dump image if needed."""
    import base64  # noqa: PLC0415

    # Try rgb_array render (returns base64-encoded PNG via env.render())
    try:
        rr = env.render()
        b64 = getattr(rr, "data", None)
        if b64:
            out_path.write_bytes(base64.b64decode(b64))
            return True
    except Exception:
        pass

    # Text-based fallback using Pillow
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415
        obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
        lines = [
            f"Turn {obs_data.get('turnNo', '?')}  Phase {obs_data.get('turnPhase', '?')}",
            f"Acting player: {obs_data.get('actingPlayerID', '?')}",
            f"P1 HP: {obs_data.get('playerHealth', '?')}   "
            f"P2 HP: {obs_data.get('opponentHealth', '?')}",
            f"Legal actions: {len(obs_data.get('legalActions', []) or [])}",
            f"Prompt: {obs_data.get('prompt', '')}",
        ]
        font = ImageFont.load_default()
        img = Image.new("RGB", (800, 200), color=(18, 18, 18))
        draw = ImageDraw.Draw(img)
        for i, line in enumerate(lines):
            draw.text((10, 10 + i * 30), line, fill=(235, 235, 235), font=font)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        return True
    except Exception:
        return False


def run_final_evaluation(
    agents: PhaseAgents,
    opponent_agents: Optional[PhaseAgents],
    *,
    hero_id: str,
    equipment_header: str,
    game_format: str,
    opponent_deck_name: str,
    opponent_hero_id: str,
    opponent_mode: str,
    num_eval_episodes: int,
    max_steps: int,
    assets_path: str,
    base_url: str,
    out_dir: Path,
    render_gif: bool = True,
    gif_fps: float = 3.0,
) -> dict[str, Any]:
    """Full final evaluation for one player's pipeline.

    Steps
    -----
    1. Select the best sidebaorded game deck (or fall back to card pool).
    2. Write a temporary deck file to Talishar Assets.
    3. Run ``num_eval_episodes`` games with the trained play agent (greedy),
       recording win / loss / draw for each episode.
    4. Render one full rollout with ``rgb_array`` (or text fallback), saving
       per-step PNGs and assembling them into an animated GIF.
    5. Write ``final_eval.json`` to ``out_dir``.

    Returns the summary dict.
    """
    player = agents.player
    print(
        f"\n{'='*62}\n"
        f"  FINAL EVALUATION  [{player} / {hero_id}]\n"
        f"{'='*62}"
    )

    # ── select game deck ──────────────────────────────────────────────────────
    game_deck = (
        agents.active_decks.get(opponent_hero_id)
        or next(iter(agents.active_decks.values()), {})
        or agents.card_pool
    )
    if not game_deck:
        print(f"  [{player}] No game deck available — skipping final eval")
        return {"skipped": True, "reason": "no_game_deck"}

    deck_total = sum(game_deck.values())
    print(f"  [{player}] Game deck: {deck_total} cards  (vs {opponent_hero_id})")

    # ── write deck file ───────────────────────────────────────────────────────
    deck_name = f"rl_final_{player}_{uuid.uuid4().hex[:8]}"
    deck_file = _write_deck_file(game_deck, equipment_header, deck_name, assets_path)

    # For dual mode, use the opponent's sidebaorded deck; otherwise use preset.
    if opponent_mode == "dual" and opponent_agents is not None:
        opp_deck = (
            opponent_agents.active_decks.get(hero_id)
            or next(iter(opponent_agents.active_decks.values()), {})
            or opponent_agents.card_pool
        )
        opp_name = f"rl_final_{opponent_agents.player}_{uuid.uuid4().hex[:8]}"
        opp_file = _write_deck_file(
            opp_deck,
            # Use opponent equipment header stored in the first key of active_decks
            # (we don't store it separately, so we pass the deck name)
            deck_name,  # placeholder — overridden by opp_name below
            opp_name,
            assets_path,
        )
    else:
        opp_name = opponent_deck_name
        opp_file = None

    # ── evaluation games ──────────────────────────────────────────────────────
    wins = 0
    losses = 0
    draws = 0
    episode_log: list[dict[str, Any]] = []

    try:
        for ep in range(1, num_eval_episodes + 1):
            env = TalisharEngineEnvironment(
                base_url=base_url,
                game_format=game_format,
                local_deck_name=deck_name,
                opponent_deck_name=opp_name,
                max_turns=max_steps,
                self_play=True,
                render_mode=None,
            )
            try:
                result = env.reset()
                obs = result.observation
                done = False
                steps = 0
                while not done:
                    if agents.play is not None and hasattr(agents.play, "act_greedy"):
                        action = agents.play.act_greedy(obs)
                    elif agents.play is not None and hasattr(agents.play, "act"):
                        action = agents.play.act(obs)
                    else:
                        action = env.sample_action()
                    step = env.step(action)
                    obs = step.observation
                    done = step.terminated or step.truncated
                    steps += 1

                obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
                p1_hp = float(obs_data.get("playerHealth", 0) or 0)
                p2_hp = float(obs_data.get("opponentHealth", 0) or 0)
                if p1_hp > p2_hp:
                    wins += 1
                    outcome = "win"
                elif p2_hp > p1_hp:
                    losses += 1
                    outcome = "loss"
                else:
                    draws += 1
                    outcome = "draw"

                episode_log.append({"episode": ep, "outcome": outcome, "steps": steps,
                                     "p1_hp": p1_hp, "p2_hp": p2_hp})
                wr = wins / ep
                print(
                    f"  [{player}] Ep {ep:>3}/{num_eval_episodes}  "
                    f"{outcome:<4}  steps={steps:3d}  win_rate={wr:.1%}"
                )
            finally:
                env.close()

    except Exception as exc:
        print(f"  [{player}] Eval error: {exc}")

    total = max(1, num_eval_episodes)
    win_rate = wins / total
    print(
        f"\n  [{player}] Final win rate: {win_rate:.1%}  "
        f"({wins}W / {losses}L / {draws}D  over {num_eval_episodes} games)"
    )

    # ── render rollout ────────────────────────────────────────────────────────
    render_dir = out_dir / f"{player}_final_render"
    render_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / f"{player}_optimal_policy.gif"

    frame_paths: list[Path] = []
    render_steps = 0
    render_terminated = False
    render_truncated = False

    print(f"\n  [{player}] Rendering optimal-policy rollout → {render_dir}")
    try:
        env = TalisharEngineEnvironment(
            base_url=base_url,
            game_format=game_format,
            local_deck_name=deck_name,
            opponent_deck_name=opp_name,
            max_turns=max_steps,
            self_play=True,
            render_mode="rgb_array",
        )
        try:
            result = env.reset()
            obs = result.observation
            frame_path = render_dir / "frame_0000_reset.png"
            if _save_state_image(env, obs, frame_path):
                frame_paths.append(frame_path)

            for step_no in range(1, max_steps + 1):
                # Greedy policy — try act_greedy first, fall back to act
                if agents.play is not None and hasattr(agents.play, "act_greedy"):
                    action = agents.play.act_greedy(obs)
                elif agents.play is not None and hasattr(agents.play, "act"):
                    action = agents.play.act(obs)
                else:
                    action = env.sample_action()

                # For dual: route to the correct agent based on acting player
                if opponent_mode == "dual" and opponent_agents is not None:
                    obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
                    acting = obs_data.get("actingPlayerID", 1)
                    if acting != 1:
                        if opponent_agents.play is not None and hasattr(opponent_agents.play, "act_greedy"):
                            action = opponent_agents.play.act_greedy(obs)
                        elif opponent_agents.play is not None and hasattr(opponent_agents.play, "act"):
                            action = opponent_agents.play.act(obs)

                step = env.step(action)
                obs = step.observation
                render_terminated = bool(step.terminated)
                render_truncated = bool(step.truncated)
                render_steps = step_no

                obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
                acting = obs_data.get("actingPlayerID", "?")
                fname = f"frame_{step_no:04d}_p{acting}.png"
                if _save_state_image(env, obs, render_dir / fname):
                    frame_paths.append(render_dir / fname)

                if render_terminated or render_truncated:
                    break
        finally:
            env.close()
    except Exception as exc:
        print(f"  [{player}] Render error: {exc}")

    print(f"  [{player}] Saved {len(frame_paths)} frames")

    if render_gif and frame_paths:
        _frames_to_gif(frame_paths, gif_path, fps=gif_fps)

    # ── write final_eval.json ─────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "player": player,
        "hero_id": hero_id,
        "opponent_hero_id": opponent_hero_id,
        "opponent_mode": opponent_mode,
        "format": game_format,
        "game_deck_size": deck_total,
        "eval": {
            "episodes": num_eval_episodes,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "episode_log": episode_log,
        },
        "render": {
            "frames_dir": str(render_dir),
            "frames_saved": len(frame_paths),
            "steps": render_steps,
            "terminated": render_terminated,
            "truncated": render_truncated,
            "gif": str(gif_path) if (render_gif and frame_paths) else None,
        },
    }
    eval_json_path = out_dir / f"{player}_final_eval.json"
    eval_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  [{player}] Final eval written → {eval_json_path}")

    # ── cleanup temp deck files ───────────────────────────────────────────────
    for f in [deck_file] + ([opp_file] if opp_file else []):
        try:
            if f.exists():
                f.unlink()
        except Exception:
            pass

    return summary


def _write_results_json(
    out_path: Path,
    *,
    game_format: str,
    opponent_mode: str,
    p1: PhaseAgents,
    p2: Optional[PhaseAgents],
    iterations: int,
    p1_final_eval: Optional[dict[str, Any]] = None,
    p2_final_eval: Optional[dict[str, Any]] = None,
) -> None:
    data: dict[str, Any] = {
        "format": game_format,
        "opponent_mode": opponent_mode,
        "iterations": iterations,
        "p1": {
            "pool_size": sum(p1.card_pool.values()),
            "active_decks": {
                opp: sum(d.values()) for opp, d in p1.active_decks.items()
            },
            "win_rates": p1.win_rates,
        },
    }
    if p1_final_eval:
        data["p1"]["final_eval"] = {
            k: v for k, v in p1_final_eval.get("eval", {}).items()
            if k != "episode_log"
        }
        data["p1"]["final_eval_gif"] = p1_final_eval.get("render", {}).get("gif")
    if p2 is not None:
        data["p2"] = {
            "pool_size": sum(p2.card_pool.values()),
            "active_decks": {
                opp: sum(d.values()) for opp, d in p2.active_decks.items()
            },
            "win_rates": p2.win_rates,
        }
        if p2_final_eval:
            data["p2"]["final_eval"] = {
                k: v for k, v in p2_final_eval.get("eval", {}).items()
                if k != "episode_log"
            }
            data["p2"]["final_eval_gif"] = p2_final_eval.get("render", {}).get("gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n  Results written → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full three-phase FaB RL training: Deckbuilder → Sideboard → Play",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── hero / format ─────────────────────────────────────────────────────────
    parser.add_argument("--format", default=_DEFAULT_FORMAT,
        choices=["silver_age", "classic_constructed", "blitz", "upf"])
    parser.add_argument("--hero-id", default=_DEFAULT_HERO_ID)
    parser.add_argument("--hero-class", default=_DEFAULT_HERO_CLASS)
    parser.add_argument("--equipment-header", default=_DEFAULT_EQUIPMENT_HEADER)

    # ── opponent ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--opponent-mode", default="preset",
        choices=["preset", "mirror", "dual"],
        help=(
            "preset  — fixed Talishar Assets deck (default)  |  "
            "mirror  — same built deck on both sides  |  "
            "dual    — full second pipeline trains simultaneously"
        ),
    )
    parser.add_argument("--opponent-deck", default=_DEFAULT_OPPONENT_DECK,
        help="Talishar Assets deck name for preset opponent")
    parser.add_argument("--opponent-hero-id", default=_DEFAULT_OPPONENT_HERO,
        help="Opponent hero ID (used for sideboard observations)")

    # ── dual / p2 ─────────────────────────────────────────────────────────────
    parser.add_argument("--p2-hero-id", default="dorinthea_ironsong")
    parser.add_argument("--p2-hero-class", default="Warrior")
    parser.add_argument("--p2-equipment-header",
        default="dorinthea_ironsong dori_equipment_sword dori_equipment_sword "
                "helm_of_avarice gauntlet_of_might ironrot_legs valor_boots")

    # ── training volumes ──────────────────────────────────────────────────────
    parser.add_argument("--deckbuild-episodes", type=int, default=50,
        help="Deckbuilder episodes per outer iteration")
    parser.add_argument("--sideboard-episodes", type=int, default=20,
        help="Sideboard episodes per opponent per outer iteration")
    parser.add_argument("--play-episodes", type=int, default=30,
        help="Play episodes per outer iteration")
    parser.add_argument("--max-build-steps", type=int, default=200)
    parser.add_argument("--max-sideboard-steps", type=int, default=100)
    parser.add_argument("--max-play-steps", type=int, default=60)
    parser.add_argument("--num-eval-games", type=int, default=3,
        help="Talishar games per deckbuilder/sideboard finalize (lower = faster)")
    parser.add_argument("--num-sideboard-episodes", type=int, default=5,
        help="Sideboard episodes run *inside* each deckbuilder evaluation step")

    # ── phase-3 warmup (train_dual_agent_common) ──────────────────────────────
    parser.add_argument(
        "--warmup-episodes", type=int, default=DEFAULT_WARMUP_EPISODES,
        help=(
            "Default-policy (random) warmup episodes at the start of Phase 3 "
            "play training.  These episodes populate the episode cache and seed "
            "behavioural-cloning before PPO takes over."
        ),
    )
    parser.add_argument(
        "--warmup-baseline-eval-episodes", type=int,
        default=DEFAULT_WARMUP_BASELINE_EVAL_EPISODES,
        help=(
            "Evaluation games run after the warmup phase to capture a "
            "before-PPO baseline win rate.  Results are saved to a checkpoint."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Parallel game sessions for Phase 3 play training.  "
            "≥2 enables the parallel path in train_dual_agent_common."
        ),
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help=(
            "Agent / episode cache root directory.  "
            "Defaults to <out-dir>/../agent_cache."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Optional global RNG seed for reproducible training.",
    )

    # ── outer iterations ──────────────────────────────────────────────────────
    parser.add_argument("--iterations", type=int, default=3,
        help=(
            "Number of outer loop iterations.  Each iteration runs all three "
            "phases in sequence so later phases can feed back into earlier ones "
            "(e.g. a better play agent improves sideboard reward signals)"
        ),
    )

    # ── agent checkpoints ─────────────────────────────────────────────────────
    parser.add_argument("--p1-deckbuilder", default=None)
    parser.add_argument("--p1-sideboard", default=None)
    parser.add_argument("--p1-play", default=None)
    parser.add_argument("--p2-deckbuilder", default=None)
    parser.add_argument("--p2-sideboard", default=None)
    parser.add_argument("--p2-play", default=None)

    # ── warm-start decks (JSON files from fetch_fabrary_deck.py) ─────────────
    parser.add_argument(
        "--p1-starting-deck", default=None,
        help=(
            "Path to a JSON file produced by fetch_fabrary_deck.py.  "
            "The 'deck' field is used as the deckbuilder warm-start pool so "
            "training begins from a known deck rather than an empty slate."
        ),
    )
    parser.add_argument(
        "--p2-starting-deck", default=None,
        help="Same as --p1-starting-deck but for player 2.",
    )

    # ── misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out-dir", default=str(_OUT_DIR))
    parser.add_argument("--results-json", default=None,
        help="Override path for results JSON (default: <out-dir>/results.json)")
    parser.add_argument("--talishar-url",
        default=os.environ.get("TALISHAR_URL", "http://localhost:8080/game"))
    parser.add_argument("--assets-path",
        default=os.environ.get("TALISHAR_ASSETS_PATH", ""))

    # ── final evaluation ──────────────────────────────────────────────────────
    parser.add_argument(
        "--final-eval-episodes", type=int, default=20,
        help="Number of games to play in the post-training final evaluation.",
    )
    parser.add_argument(
        "--final-eval-max-steps", type=int, default=60,
        help="Max game steps per episode in the final evaluation.",
    )
    parser.add_argument(
        "--no-render-gif", action="store_true",
        help="Skip rendering the animated GIF of the optimal-policy rollout.",
    )
    parser.add_argument(
        "--gif-fps", type=float, default=3.0,
        help="Frames per second for the rendered GIF.",
    )

    args = parser.parse_args()

      
    # Ensure warmup is at least 1/5 of play episodes (but no more than play episodes)
    min_warmup = max(1, math.ceil(args.play_episodes / 5))
    warmup_eps = min(max(args.warmup_episodes, min_warmup), args.play_episodes)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_path = args.assets_path or str(
        Path(__file__).resolve().parent.parent / "Talishar" / "Assets"
    )
    results_json = Path(args.results_json) if args.results_json else out_dir / "results.json"

    # ── build agent containers ────────────────────────────────────────────────
    p1 = PhaseAgents(
        player="p1",
        deckbuilder=_load_agent(args.p1_deckbuilder),
        sideboard=_load_agent(args.p1_sideboard),
        play=_load_agent(args.p1_play),
    )
    p2: Optional[PhaseAgents] = None
    if args.opponent_mode == "dual":
        p2 = PhaseAgents(
            player="p2",
            deckbuilder=_load_agent(args.p2_deckbuilder),
            sideboard=_load_agent(args.p2_sideboard),
            play=_load_agent(args.p2_play),
        )

    # ── warm-start (FaBrary) decks ────────────────────────────────────────────
    p1_starting_deck: Optional[dict[str, int]] = _load_starting_deck(
        getattr(args, "p1_starting_deck", None)
    )
    p2_starting_deck: Optional[dict[str, int]] = _load_starting_deck(
        getattr(args, "p2_starting_deck", None)
    )

    # ── opponent list for sideboard phase ────────────────────────────────────
    # In preset/mirror mode the sideboard agent trains for one fixed opponent.
    # In dual mode each player trains against the other's hero.
    if args.opponent_mode == "dual" and p2 is not None:
        p1_opponents = [args.p2_hero_id]
        p2_opponents = [args.hero_id]
    else:
        p1_opponents = [args.opponent_hero_id]

    print(
        f"\n{'='*62}\n"
        f"  Full Pipeline Training\n"
        f"  Format: {args.format}  |  Hero: {args.hero_id}\n"
        f"  Opponent mode: {args.opponent_mode}\n"
        f"  Outer iterations: {args.iterations}\n"
        f"{'='*62}"
    )

    # ── Seed active_decks from starting decks (cold-start Phase 3) ───────────
    # Phase 3 (play) runs first in each iteration.  On iteration 1 no pool
    # has been built yet, so pre-populate card_pool and active_decks from the
    # FaBrary warm-start decks so there is a playable deck on the first run.
    def _greedy_game_deck_cut(pool: dict[str, int], min_size: int) -> dict[str, int]:
        """Take the first min_size cards from pool (deterministic greedy cut)."""
        game_deck: dict[str, int] = {}
        remaining = min_size
        for card_id, count in pool.items():
            if remaining <= 0:
                break
            take = min(count, remaining)
            game_deck[card_id] = take
            remaining -= take
        return game_deck

    _min_size = 40 if args.format == "silver_age" else 60
    if p1_starting_deck and not p1.card_pool:
        p1.card_pool = dict(p1_starting_deck)
        cold_deck = _greedy_game_deck_cut(p1_starting_deck, _min_size)
        cold_key = args.p2_hero_id if args.opponent_mode == "dual" else args.opponent_hero_id
        p1.active_decks[cold_key] = cold_deck
        print(f"  [p1] Cold-start deck: {sum(cold_deck.values())} cards from starting pool")

    if p2 is not None and p2_starting_deck and not p2.card_pool:
        p2.card_pool = dict(p2_starting_deck)
        cold_deck2 = _greedy_game_deck_cut(p2_starting_deck, _min_size)
        p2.active_decks[args.hero_id] = cold_deck2
        print(f"  [p2] Cold-start deck: {sum(cold_deck2.values())} cards from starting pool")

    # ── outer training loop ───────────────────────────────────────────────────
    for iteration in range(1, args.iterations + 1):
        print(f"\n\n{'#'*62}")
        print(f"  ITERATION {iteration} / {args.iterations}")
        print(f"{'#'*62}")

        # ── Phase 3 FIRST: Play ───────────────────────────────────────────────
        # Evaluate the *current* decks (random / loaded on iter 1, trained on
        # later iterations).  The returned win-rates become the reward signal
        # that drives Phase 1 (deckbuilder) and Phase 2 (sideboard).
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
                args.p2_hero_id if args.opponent_mode == "dual"
                else args.opponent_hero_id
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
        )
        p1.last_play_win_rate = p1_wr
        if p2 is not None:
            p2.last_play_win_rate = p2_wr

        # ── Phase 1: Deckbuilder ───────────────────────────────────────────────
        # On iteration 1 use the FaBrary warm-start; on later iterations feed
        # back the best pool from the previous iteration.
        p1_warm = p1.card_pool if iteration > 1 else p1_starting_deck
        run_phase1_deckbuilder(
            p1,
            hero_id=args.hero_id,
            hero_class=args.hero_class,
            equipment_header=args.equipment_header,
            game_format=args.format,
            opponent_deck_name=args.opponent_deck,
            opponent_hero_id=args.opponent_hero_id,
            n_episodes=args.deckbuild_episodes,
            max_build_steps=args.max_build_steps,
            num_eval_games=args.num_eval_games,
            num_sideboard_episodes=args.num_sideboard_episodes,
            assets_path=assets_path,
            base_url=args.talishar_url,
            render=args.render,
            starting_deck=p1_warm,
            play_reward=p1.last_play_win_rate,
        )

        if args.opponent_mode == "dual" and p2 is not None:
            p2_warm = p2.card_pool if iteration > 1 else p2_starting_deck
            run_phase1_deckbuilder(
                p2,
                hero_id=args.p2_hero_id,
                hero_class=args.p2_hero_class,
                equipment_header=args.p2_equipment_header,
                game_format=args.format,
                opponent_deck_name=args.opponent_deck,
                opponent_hero_id=args.hero_id,
                n_episodes=args.deckbuild_episodes,
                max_build_steps=args.max_build_steps,
                num_eval_games=args.num_eval_games,
                num_sideboard_episodes=args.num_sideboard_episodes,
                assets_path=assets_path,
                base_url=args.talishar_url,
                render=args.render,
                starting_deck=p2_warm,
                play_reward=p2.last_play_win_rate,
            )

        # ── Phase 2: Sideboard ────────────────────────────────────────────────
        run_phase2_sideboard(
            p1,
            p1_opponents,
            hero_id=args.hero_id,
            equipment_header=args.equipment_header,
            game_format=args.format,
            opponent_deck_name=args.opponent_deck,
            n_episodes_per_opponent=args.sideboard_episodes,
            max_sideboard_steps=args.max_sideboard_steps,
            num_eval_games=args.num_eval_games,
            assets_path=assets_path,
            base_url=args.talishar_url,
            render=args.render,
            play_reward=p1.last_play_win_rate,
        )

        if args.opponent_mode == "dual" and p2 is not None:
            run_phase2_sideboard(
                p2,
                p2_opponents,
                hero_id=args.p2_hero_id,
                equipment_header=args.p2_equipment_header,
                game_format=args.format,
                opponent_deck_name=args.opponent_deck,
                n_episodes_per_opponent=args.sideboard_episodes,
                max_sideboard_steps=args.max_sideboard_steps,
                num_eval_games=args.num_eval_games,
                assets_path=assets_path,
                base_url=args.talishar_url,
                render=args.render,
                play_reward=p2.last_play_win_rate,
            )

        # ── Save after each iteration ─────────────────────────────────────────
        _save_all_agents(p1, out_dir)
        if p2 is not None:
            _save_all_agents(p2, out_dir)

        _write_results_json(
            results_json,
            game_format=args.format,
            opponent_mode=args.opponent_mode,
            p1=p1,
            p2=p2,
            iterations=iteration,
        )

    # ── Final evaluation ──────────────────────────────────────────────────────
    print(f"\n\n{'#'*62}")
    print("  FINAL EVALUATION — best deck → sideboard → optimal policy")
    print(f"{'#'*62}")

    final_eval_dir = out_dir / "final_eval"
    final_eval_dir.mkdir(parents=True, exist_ok=True)

    p1_eval = run_final_evaluation(
        p1,
        p2 if args.opponent_mode == "dual" else None,
        hero_id=args.hero_id,
        equipment_header=args.equipment_header,
        game_format=args.format,
        opponent_deck_name=args.opponent_deck,
        opponent_hero_id=args.opponent_hero_id if args.opponent_mode != "dual"
                         else args.p2_hero_id,
        opponent_mode=args.opponent_mode,
        num_eval_episodes=args.final_eval_episodes,
        max_steps=args.final_eval_max_steps,
        assets_path=assets_path,
        base_url=args.talishar_url,
        out_dir=final_eval_dir,
        render_gif=not args.no_render_gif,
        gif_fps=args.gif_fps,
    )

    p2_eval: Optional[dict[str, Any]] = None
    if args.opponent_mode == "dual" and p2 is not None:
        p2_eval = run_final_evaluation(
            p2,
            p1,
            hero_id=args.p2_hero_id,
            equipment_header=args.p2_equipment_header,
            game_format=args.format,
            opponent_deck_name=args.opponent_deck,
            opponent_hero_id=args.hero_id,
            opponent_mode=args.opponent_mode,
            num_eval_episodes=args.final_eval_episodes,
            max_steps=args.final_eval_max_steps,
            assets_path=assets_path,
            base_url=args.talishar_url,
            out_dir=final_eval_dir,
            render_gif=not args.no_render_gif,
            gif_fps=args.gif_fps,
        )

    # Merge final eval results into the main results JSON
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

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*62}")
    print("  TRAINING COMPLETE")
    print(f"{'='*62}")
    print(f"  Format        : {args.format}")
    print(f"  Opponent mode : {args.opponent_mode}")
    print(f"  Iterations    : {args.iterations}")
    print(f"\n  P1 ({args.hero_id})")
    print(f"    Pool size   : {sum(p1.card_pool.values())} cards")
    for opp, deck in p1.active_decks.items():
        print(f"    Deck vs {opp:<32}: {sum(deck.values())} cards")
    if p1.win_rates:
        print(f"    Win rates   : {[f'{w:.1%}' for w in p1.win_rates]}")
    p1_wr = p1_eval.get("eval", {}).get("win_rate")
    if p1_wr is not None:
        print(f"    Final eval  : {p1_wr:.1%}  ({args.final_eval_episodes} games)")
    p1_gif = p1_eval.get("render", {}).get("gif")
    if p1_gif:
        print(f"    Render GIF  : {p1_gif}")

    if p2 is not None:
        print(f"\n  P2 ({args.p2_hero_id})")
        print(f"    Pool size   : {sum(p2.card_pool.values())} cards")
        for opp, deck in p2.active_decks.items():
            print(f"    Deck vs {opp:<32}: {sum(deck.values())} cards")
        if p2.win_rates:
            print(f"    Win rates   : {[f'{w:.1%}' for w in p2.win_rates]}")
        if p2_eval:
            p2_wr = p2_eval.get("eval", {}).get("win_rate")
            if p2_wr is not None:
                print(f"    Final eval  : {p2_wr:.1%}  ({args.final_eval_episodes} games)")
            p2_gif = p2_eval.get("render", {}).get("gif")
            if p2_gif:
                print(f"    Render GIF  : {p2_gif}")

    print(f"\n  Agents saved   → {out_dir}")
    print(f"  Final eval dir → {final_eval_dir}")
    print(f"  Results JSON   → {results_json}")


if __name__ == "__main__":
    main()
