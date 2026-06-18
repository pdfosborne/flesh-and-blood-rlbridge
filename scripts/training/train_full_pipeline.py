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

                try:
                    # Navigate and give the FE time to hydrate and fetch game state
                    page.goto(game_url, timeout=20_000, wait_until="domcontentloaded")
                    # Extra delay mirrors _open_playwright_page in the engine so
                    # the FE can complete client-side routing + API calls.
                    page.wait_for_timeout(5000)

                    # Try multiple consent button labels in case a different
                    # widget is used (match common variants).
                    for lbl in ("Agree", "Accept", "Accept All Cookies", "Accept All", "OK", "Got it"):
                        try:
                            btn = page.locator("button", has_text=lbl).first
                            if btn.is_visible(timeout=500):
                                btn.click()
                                page.wait_for_timeout(1500)
                                break
                        except Exception:
                            pass

                    # Wait for the game board to mount (not the home page)
                    board_sel = "#root .game, #root [class*='board'], #root [class*='Board'], #root [class*='play'], #root [class*='Play']"
                    page.wait_for_selector(board_sel, timeout=15_000)
                    # Wait for at least one IMG inside the board so equipment art
                    # / card images have started loading. If no <img> appears, the
                    # selector times out and we fall back to the normal screenshot.
                    try:
                        page.wait_for_selector(f"{board_sel} img", timeout=8_000)
                        # little extra time to let the image render fully
                        page.wait_for_timeout(400)
                    except Exception:
                        pass
                except Exception as exc:
                    print(f"  [{player_label}] WARNING: FE page load error ({exc}) — continuing with fallback screenshots")

                # Screenshot after reset (give a moment for images to load)
                frame_path = render_dir / "frame_0000_reset.png"
                try:
                    page.wait_for_timeout(500)
                    page.screenshot(path=str(frame_path), full_page=False)
                    frame_paths.append(frame_path)
                except Exception:
                    pass
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
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_ROOT))
import _bootstrap  # noqa: E402

_REPO_ROOT = _bootstrap.configure_paths()
_RL_SRC = Path("~/Documents/RL/rlbridge/src").expanduser()
if str(_RL_SRC) not in sys.path:
    sys.path.insert(0, str(_RL_SRC))

from flesh_and_blood_rlbridge import (  # noqa: E402
    TalisharDeckBuilderEnvironment,
    TalisharSideboardEnvironment,
    TalisharEngineEnvironment,
)
from flesh_and_blood_rlbridge.opponent_deck import (  # noqa: E402
    normalize_talishar_asset_name,
    resolve_opponent_deck_name,
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


def _runtime_backend_label(env: Any) -> str:
    """Return a friendly backend label for a Talishar environment instance."""
    try:
        if bool(getattr(env, "_using_cpp", False)):
            return "C++ engine"
    except Exception:
        pass
    return "HTTP Talishar"


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
    # Equipment header string written into the deck file (e.g. "ira_crimson_haze harmonized_kodachi ...")
    equipment_header: str = ""
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
    cpp_engine_dir: Optional[str] = None,
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
        cpp_engine_dir=cpp_engine_dir,
    )

    best_reward = float("-inf")
    best_pool: dict[str, int] = {}

    for ep in range(1, n_episodes + 1):
        print(f"  [{agents.player}] Ep {ep:>4}/{n_episodes}")
        result = env.reset()
        ep_reward = 0.0
        done = False

        step_count = 0
        while not done:
            if step_count % 100 == 0:
                print(f"  [{agents.player}] Ep {ep:>4}/{n_episodes} - Step {step_count}")
            if agents.deckbuilder is not None and hasattr(agents.deckbuilder, "act"):
                action = agents.deckbuilder.act(result.observation)
            else:
                obs_data = json.loads(result.observation)
                deck_sz  = obs_data.get("deckSize", 0)
                pool_sz  = obs_data.get("targetPoolSize", 55)
                avail    = obs_data.get("availableActions", [])
                if deck_sz < pool_sz:
                    # Greedily add cards until pool is full — one evaluate per episode
                    add_acts = [a for a in avail if a.startswith("add:")]
                    action = add_acts[0] if add_acts else "finalize"
                else:
                    # Pool is full — finalize immediately
                    action = "finalize"

            step = env.step(action)
            ep_reward += step.reward
            done = step.terminated or step.truncated

            if render and not done:
                r = env.render()
                if r.text:
                    print(r.text)

            result.observation = step.observation
            step_count += 1

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
    cpp_engine_dir: Optional[str] = None,
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
            cpp_engine_dir=cpp_engine_dir,
        )
        best_reward = float("-inf")
        best_deck: dict[str, int] = {}

        for ep in range(1, n_episodes_per_opponent + 1):
            print(f"  [{agents.player}] Ep {ep:>4}/{n_episodes_per_opponent}")
            result = env.reset()
            ep_reward = 0.0
            done = False
            step_count = 0
            while not done:
                if step_count % 100 == 0:
                    print(f"  [{agents.player}] Ep {ep:>4}/{n_episodes_per_opponent} - Step {step_count}")
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
                step_count += 1

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
    # ── Equipment fallback extraction ─────────────────────────────────────────
    # When the header only contains a hero ID (no equipment pieces), scan the
    # deck list for equipment-type cards and promote them to the header line.
    #
    # This handles decks fetched via FaBrary's GraphQL endpoint which returns
    # all registered cards (including equipment) in deckCards with no zone
    # information, so fetch_fabrary_deck.py ends up storing everything in
    # `deck` and leaving equipment_header as just the hero ID.
    #
    # Heuristic rules (safe because FaB play-cards always carry a colour
    # suffix: _red / _blue / _yellow / _purple):
    #   1. Any card with a colour suffix  →  always a play card.
    #   2. Cards matching equipment slot patterns without a colour suffix
    #      →  equipment; lifted out of the deck into the header.
    _EQUIP_SLOT_PATS: dict[str, list[str]] = {
        "weapon": [
            # Use specific multi-character fragments only — "blade" alone is too
            # broad and matches equipment names like "blade_beckoner_gauntlets".
            "kodachi", "dawnblade", "rosetta", "galaxia", "pistol",
            "sword", "axe", "staff", "bow", "harpoon",
            "scimitar", "cracked_bauble", "bauble", "quiver", "death_dealer",
        ],
        "head": [
            "helm", "hood", "crown", "cap", "headband", "goggles",
            "mask", "hat", "brow", "visor", "tiara", "circlet",
        ],
        "chest": [
            "coat", "robe", "vest", "chestplate", "jacket", "shirt",
            "tunic", "cuirass", "cloak", "cape", "mantle", "doublet",
        ],
        "arms": [
            "gauntlet", "glove", "bracer", "vambrace", "wrist",
            "bangle", "shuko", "sleeve", "handwrap", "sedative",
        ],
        "legs": [
            "boots", "greaves", "pants", "leggings", "leg", "sabaton",
            "footwrap", "shin", "paws",
        ],
    }
    # Known equipment cards whose names give no keyword clue.
    # Checked before pattern matching so they are never mis-placed in the
    # play deck (e.g. "garland_of_spring" = Runeblade chest,
    # "star_fall" = Runeblade sword 1H).
    _KNOWN_EQUIP: dict[str, str] = {
        # Runeblade
        "star_fall":                   "weapon",
        "nebula_blade":                "weapon",
        "talishar_the_lost_prince":    "weapon",
        "aether_ironweave":            "chest",
        "garland_of_spring":           "chest",
        "ironhide_plate":              "chest",
        "spellbound_creepers":         "legs",
        "aether_crackers":             "arms",
        "nullrune_gloves":             "arms",
        # Blade Beckoner equipment — "blade" pattern in weapon list would
        # match these incorrectly if it were included.
        "blade_beckoner_gauntlets":    "arms",
        "blade_beckoner_helm":         "head",
        "blade_beckoner_boots":        "legs",
        # Ranger / Riptide
        "quiver_of_a_thousand_arrows": "weapon",
        # Generic
        "nullrune_robe":               "chest",
        "nullrune_hood":               "head",
    }
    _COLOUR_SUFFIXES = ("_red", "_blue", "_yellow", "_purple")

    def _equip_slot(cid: str) -> str:
        known = _KNOWN_EQUIP.get(cid)
        if known:
            return known
        if cid.endswith(_COLOUR_SUFFIXES):
            return "deck"
        for slot, pats in _EQUIP_SLOT_PATS.items():
            for pat in pats:
                if pat in cid:
                    return slot
        return "deck"

    header_parts = (equipment_header or "").split()
    if len(header_parts) <= 1:
        # Only a hero ID (or empty) — extract equipment from the deck
        hero = header_parts[0] if header_parts else ""
        slot_cards: dict[str, list[str]] = {s: [] for s in _EQUIP_SLOT_PATS}
        play_deck: dict[str, int] = {}
        for card_id, count in deck.items():
            s = _equip_slot(card_id)
            if s in slot_cards:
                slot_cards[s].extend([card_id] * count)
            else:
                play_deck[card_id] = count
        found: list[str] = []
        for slot in ("weapon", "head", "chest", "arms", "legs"):
            found.extend(slot_cards[slot])
        if found:
            equipment_header = (hero + " " + " ".join(found)).strip()
            deck = play_deck
            print(
                f"  [deck] Extracted {len(found)} equipment card(s) from deck "
                f"into header: {found}"
            )

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
    cpp_engine_dir: Optional[str] = None,
    checkpoint_interval: int = 10000,
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
    p1.deck_asset_name = p1_deck_name

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
    if p2 is not None:
        p2.deck_asset_name = p2_deck_name

    # ── C++ engine lookup key (original hero IDs, not UUID deck names) ───────
    # The C++ engine is compiled under the original hero/asset IDs
    # (e.g. "aurora_vs_briar"), but Phase 3 writes deck files with UUID-based
    # names.  Pass the hero IDs as override lookup keys so the engine is found.
    _cpp_deck1 = p1_hero_id or None
    _cpp_deck2 = p2_hero_id or _cpp_deck1

    # ── backend visibility (C++ vs HTTP) ────────────────────────────────────
    probe_env = TalisharEngineEnvironment(
        base_url=base_url,
        game_format=game_format,
        local_deck_name=p1_deck_name,
        opponent_deck_name=p2_deck_name,
        max_turns=max_play_steps,
        self_play=True,
        render_mode=None,
        cpp_engine_deck1=_cpp_deck1,
        cpp_engine_deck2=_cpp_deck2,
        cpp_engine_dir=cpp_engine_dir,
    )
    try:
        print(f"  Runtime backend (Phase 3): {_runtime_backend_label(probe_env)}")
        use_cpp_backend = bool(getattr(probe_env, "_using_cpp", False))
        if cpp_engine_dir and not use_cpp_backend:
            raise RuntimeError(
                f"C++ engine required (--cpp-engine-dir={cpp_engine_dir}) but "
                f"failed to load for Python {sys.version_info.major}.{sys.version_info.minor}. "
                "Rebuild with:\n"
                f"  python scripts/cpp/build_cpp_engine_for_matchup.py "
                f"--deck1 {_cpp_deck1} --deck2 {_cpp_deck2} "
                f"--deck1-json <p1.json> --deck2-json <p2.json> --no-server"
            )
        if not use_cpp_backend and n_workers > 1:
            print(
                f"  WARNING: HTTP Talishar cannot run {n_workers} parallel game sessions — "
                "capping workers to 1."
            )
            n_workers = 1
    finally:
        probe_env.close()

    # ── Matchup + EpisodeCache ────────────────────────────────────────────────
    matchup = Matchup(
        name=f"p3_{p1_deck_name[-8:]}-vs-{p2_deck_name[-8:]}",
        p1_deck=p1_deck_name,
        p2_deck=p2_deck_name,
        description=f"Phase 3 play ({opponent_mode}): {p1.player} vs "
                    + (p2.player if p2 else p1_opponent_deck_name),
        p1_hero=p1_hero_id.replace("_", "-"),
        p2_hero=p2_hero_id.replace("_", "-"),
        cpp_engine_deck1=_cpp_deck1,
        cpp_engine_deck2=_cpp_deck2,
        cpp_engine_dir=cpp_engine_dir,
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
    total_completed = 0
    warmup_remaining = min(warmup_episodes, n_episodes)
    baseline_saved = False

    def _maybe_save_checkpoint(force: bool = False) -> None:
        if checkpoint_interval <= 0:
            if not force:
                return
        elif not force and total_completed % checkpoint_interval != 0:
            return
        if total_completed <= 0:
            return
        _save_phase3_play_checkpoints(
            out_dir=out_dir,
            matchup=matchup,
            game_format=game_format,
            p1_agent=p1_agent,
            p2_agent=p2_agent,
            p1_rewards=p1_rewards,
            p2_rewards=p2_rewards,
            episodes_completed=total_completed,
            total_target_episodes=n_episodes,
            opponent_mode=opponent_mode,
            p1_deck_cards=p1_game_deck,
            p2_deck_cards=p2_game_deck,
            p1_equipment_header=p1_equipment_header,
            p2_equipment_header=p2_equipment_header,
            p1_opponent_deck_name=p1_opponent_deck_name,
        )

    try:
        if n_workers > 1:
            while total_completed < n_episodes:
                remaining = n_episodes - total_completed
                chunk_size = remaining
                if checkpoint_interval > 0:
                    chunk_size = min(chunk_size, checkpoint_interval)
                chunk_warmup = min(warmup_remaining, chunk_size)
                chunk_seed = (seed + total_completed) if seed is not None else None
                mode_label = "warmup" if chunk_warmup == chunk_size else ("mixed" if chunk_warmup > 0 else "ppo")
                print(
                    f"  Parallel chunk: {chunk_size} episode(s) "
                    f"[{mode_label}] starting at ep {total_completed + 1}"
                )
                p1_r, p2_r, _ = train_agents_from_both_perspectives_parallel(
                    matchup=matchup,
                    base_url=base_url,
                    game_format=game_format,
                    p1_tiers=p1_tiers,
                    p2_tiers=p2_tiers,
                    n_episodes=chunk_size,
                    max_steps=max_play_steps,
                    seed=chunk_seed,
                    warmup_episodes=chunk_warmup,
                    n_workers=n_workers,
                    live_state_image_path=live_path,
                    episode_cache=episode_cache,
                )
                p1_rewards.extend(p1_r)
                p2_rewards.extend(p2_r)
                total_completed += chunk_size
                warmup_remaining -= chunk_warmup
                if (
                    not baseline_saved
                    and warmup_episodes > 0
                    and warmup_baseline_eval_episodes > 0
                    and warmup_remaining <= 0
                ):
                    _run_warmup_baseline(
                        matchup, p1_agent, p2_agent,
                        base_url=base_url, game_format=game_format,
                        max_steps=max_play_steps, out_dir=out_dir,
                        episodes=warmup_baseline_eval_episodes, seed=seed,
                    )
                    baseline_saved = True
                _maybe_save_checkpoint()

        else:
            # Serial: explicit warmup/PPO chunks so long runs can checkpoint.
            env = make_env(
                matchup, base_url=base_url, game_format=game_format,
                max_turns=max_play_steps,
            )
            try:
                while total_completed < n_episodes:
                    remaining = n_episodes - total_completed
                    chunk_size = remaining
                    if checkpoint_interval > 0:
                        chunk_size = min(chunk_size, checkpoint_interval)
                    chunk_warmup = min(warmup_remaining, chunk_size)
                    chunk_seed = (seed + total_completed) if seed is not None else None
                    if chunk_warmup == chunk_size:
                        print(f"  Warmup chunk: {chunk_size} episode(s) starting at ep {total_completed + 1}…")
                    elif chunk_warmup > 0:
                        print(f"  Mixed chunk: {chunk_size} episode(s) ({chunk_warmup} warmup) starting at ep {total_completed + 1}…")
                    else:
                        print(f"  PPO chunk: {chunk_size} episode(s) starting at ep {total_completed + 1}…")

                    c_p1, c_p2, _ = train_agents_from_both_perspectives(
                        env, p1_tiers, p2_tiers,
                        n_episodes=chunk_size,
                        max_steps=max_play_steps,
                        seed=chunk_seed,
                        warmup_episodes=chunk_warmup,
                        live_state_image_path=live_path,
                        episode_cache=episode_cache,
                        p1_deck=p1_deck_name,
                        p2_deck=p2_deck_name,
                    )
                    p1_rewards.extend(c_p1)
                    p2_rewards.extend(c_p2)
                    total_completed += chunk_size
                    warmup_remaining -= chunk_warmup

                    if (
                        not baseline_saved
                        and warmup_episodes > 0
                        and warmup_baseline_eval_episodes > 0
                        and warmup_remaining <= 0
                    ):
                        _run_warmup_baseline(
                            matchup, p1_agent, p2_agent,
                            base_url=base_url, game_format=game_format,
                            max_steps=max_play_steps, out_dir=out_dir,
                            episodes=warmup_baseline_eval_episodes, seed=seed,
                        )
                        baseline_saved = True
                    _maybe_save_checkpoint()
            finally:
                env.close()

        _maybe_save_checkpoint(force=True)

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
    backend_printed = False
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
                if not backend_printed:
                    print(f"  Runtime backend (fallback play): {_runtime_backend_label(env)}")
                    backend_printed = True
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


def _save_play_checkpoint_package(
    *,
    agent: Any,
    out_dir: Path,
    matchup: "Matchup",
    game_format: str,
    role: str,
    episodes_completed: int,
    total_target_episodes: int,
    reward_history: list[float],
    deck_cards: dict[str, int],
    equipment_header: str,
    opponent_mode: str,
    opponent_deck_name: str,
) -> Optional[Path]:
    """Persist a discoverable phase-3 checkpoint package under results/."""
    if not hasattr(agent, "save"):
        return None

    checkpoint_dir = out_dir / matchup.name / role / f"episode_{episodes_completed:06d}"
    weights_dir = checkpoint_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    try:
        agent.save(weights_dir / "agent_weights.json")
    except Exception as exc:
        print(f"  [{role}] WARNING: could not save play checkpoint at ep {episodes_completed}: {exc}")
        return None

    rewards = reward_history[:episodes_completed]
    avg_reward = float(sum(rewards) / len(rewards)) if rewards else 0.0
    metadata = {
        "checkpoint_type": "phase3_play",
        "created_at": datetime.now().isoformat(),
        "matchup": matchup.name,
        "role": role,
        "game_format": game_format,
        "weights_file": "agent_weights.json",
        "episodes_completed": episodes_completed,
        "target_episodes": total_target_episodes,
        "p1_deck": matchup.p1_deck,
        "p2_deck": matchup.p2_deck,
        "p1_hero": matchup.p1_hero,
        "p2_hero": matchup.p2_hero,
        "cpp_engine_deck1": matchup.cpp_engine_deck1,
        "cpp_engine_deck2": matchup.cpp_engine_deck2,
        "cpp_engine_dir": matchup.cpp_engine_dir,
        "avg_reward": avg_reward,
        "opponent_mode": opponent_mode,
        "opponent_deck_name": opponent_deck_name,
        "deck_spec": {
            "equipment_header": equipment_header,
            "cards": deck_cards,
        },
    }
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return checkpoint_dir


def _ensure_hero_in_header(equipment_header: str, hero_id: str) -> str:
    """Guarantee ``hero_id`` is the first token of ``equipment_header``.

    Talishar's deck file parser requires the hero card ID as the very first
    token on line 1.  When it is absent the hero portrait is never loaded.
    ``hero_id`` uses underscores; convert dashes just in case.
    """
    hero = hero_id.replace("-", "_").strip()
    header = (equipment_header or "").strip()
    if hero and not header.startswith(hero):
        header = (hero + " " + header).strip()
    return header


def _save_phase3_play_checkpoints(
    *,
    out_dir: Path,
    matchup: "Matchup",
    game_format: str,
    p1_agent: Any,
    p2_agent: Any,
    p1_rewards: list[float],
    p2_rewards: list[float],
    episodes_completed: int,
    total_target_episodes: int,
    opponent_mode: str,
    p1_deck_cards: dict[str, int],
    p2_deck_cards: Optional[dict[str, int]],
    p1_equipment_header: str,
    p2_equipment_header: str,
    p1_opponent_deck_name: str,
) -> None:
    # Always store the hero ID as the first token so eval scripts can
    # reconstruct a valid deck file without needing external hero metadata.
    _p1_header = _ensure_hero_in_header(p1_equipment_header, matchup.p1_hero)
    _p2_header = _ensure_hero_in_header(p2_equipment_header, matchup.p2_hero)

    p1_ckpt = _save_play_checkpoint_package(
        agent=p1_agent,
        out_dir=out_dir,
        matchup=matchup,
        game_format=game_format,
        role="p1",
        episodes_completed=episodes_completed,
        total_target_episodes=total_target_episodes,
        reward_history=p1_rewards,
        deck_cards=p1_deck_cards,
        equipment_header=_p1_header,
        opponent_mode=opponent_mode,
        opponent_deck_name=p1_opponent_deck_name,
    )
    if p1_ckpt is not None:
        print(f"  [p1] Phase-3 checkpoint → {p1_ckpt}")

    p2_ckpt = _save_play_checkpoint_package(
        agent=p2_agent,
        out_dir=out_dir,
        matchup=matchup,
        game_format=game_format,
        role="p2",
        episodes_completed=episodes_completed,
        total_target_episodes=total_target_episodes,
        reward_history=p2_rewards,
        deck_cards=p2_deck_cards or {},
        equipment_header=_p2_header,
        opponent_mode=opponent_mode,
        opponent_deck_name=(matchup.p1_deck if opponent_mode == "dual" else p1_opponent_deck_name),
    )
    if p2_ckpt is not None:
        print(f"  [p2] Phase-3 checkpoint → {p2_ckpt}")


# ---------------------------------------------------------------------------
# Final evaluation — eval games + optimal-policy render + GIF
# ---------------------------------------------------------------------------


def _ensure_playwright() -> None:
    """Install Playwright + Chromium if not already available."""
    try:
        import playwright  # noqa: F401
    except ImportError:
        import subprocess, sys  # noqa: PLC0415
        print("  [render] Installing playwright Python package…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
    # Ensure Chromium browser binaries are present
    try:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415
        with sync_playwright() as pw:
            pw.chromium.launch(headless=True)  # will throw if binaries missing
    except Exception:
        import subprocess, sys  # noqa: PLC0415
        print("  [render] Installing Playwright Chromium browser…")
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])


def _prepare_render_dir(render_dir: Path) -> None:
    """Delete stale frames from a prior rollout before writing new ones."""
    import shutil  # noqa: PLC0415

    if render_dir.exists():
        shutil.rmtree(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)


def _render_game_with_talishar_frontend(
    *,
    agents: Any,
    opponent_agents: Optional[Any],
    opponent_mode: str,
    base_url: str,
    fe_url: str,
    game_format: str,
    deck_name: str,
    opp_name: str,
    max_steps: int,
    render_dir: Path,
    player_label: str,
) -> tuple[list[Path], str]:
    """Play one game via the HTTP Talishar backend and screenshot the live
    Talishar frontend after every step.

    Uses ``render_mode='rgb_array'`` on the environment so that
    ``TalisharEngineEnvironment`` manages its own Playwright browser thread —
    the same approach used (and confirmed working) in
    ``train_eval_render_pipeline.py``.  On ``reset()`` the engine navigates to
    the frontend with ``domcontentloaded`` + a 5-second settle wait, then
    queues a screenshot (with a 1.5 s render delay) on every ``env.render()``
    call, which gives equipment card art time to load.

    Returns:
        ``(frame_paths, outcome)`` where *outcome* is ``win`` / ``loss`` /
        ``draw`` / ``timeout`` from P1's perspective.  *frame_paths* includes a
        final annotated end-state frame when capture succeeds.
    """
    _ensure_playwright()
    _prepare_render_dir(render_dir)
    frame_paths: list[Path] = []
    outcome = "timeout"

    try:
        env = TalisharEngineEnvironment(
            base_url=base_url,
            frontend_url=fe_url,           # passed directly to env — no manual browser
            game_format=game_format,
            local_deck_name=deck_name,
            opponent_deck_name=opp_name,
            max_turns=max_steps,
            self_play=True,
            render_mode="rgb_array",       # engine owns the Playwright worker
            use_cpp_engine=False,          # HTTP backend required so FE can connect
            enable_combat_tracker=True,
        )
        try:
            result = env.reset()           # _open_playwright_page() runs here
            obs = result.observation

            # Frame 0 — board state after reset (equipment is visible here)
            frame_path = render_dir / "frame_0000_reset.png"
            if _save_state_image(env, obs, frame_path):
                frame_paths.append(frame_path)
                print(f"  [{player_label}] Frame 0 saved (reset)")

            done = False
            step_no = 0
            terminated = False
            truncated = False
            while not done and step_no < max_steps:
                step_no += 1

                obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
                acting = obs_data.get("actingPlayerID", 1)

                # Route action to the correct agent
                active_agents = agents
                if opponent_mode == "dual" and opponent_agents is not None and acting != 1:
                    active_agents = opponent_agents

                if active_agents.play is not None and hasattr(active_agents.play, "act_greedy"):
                    action = active_agents.play.act_greedy(obs)
                elif active_agents.play is not None and hasattr(active_agents.play, "act"):
                    action = active_agents.play.act(obs)
                else:
                    action = env.sample_action()

                step = env.step(action)
                obs = step.observation
                terminated = bool(step.terminated)
                truncated = bool(step.truncated)
                done = terminated or truncated

                fname = f"frame_{step_no:04d}_p{acting}.png"
                fpath = render_dir / fname
                if _save_state_image(env, obs, fpath):
                    frame_paths.append(fpath)

            outcome = _infer_render_outcome(
                obs, terminated=terminated, truncated=truncated,
            )
            end_path = render_dir / f"frame_{step_no + 1:04d}_end_{outcome}.png"
            if _save_end_state_frame(env, obs, end_path, outcome=outcome, steps=step_no):
                frame_paths.append(end_path)
                print(f"  [{player_label}] End frame saved ({outcome})")

        finally:
            env.close()

    except Exception as exc:
        print(f"  [{player_label}] Render error: {exc}")

    print(f"  [{player_label}] Saved {len(frame_paths)} frames → {render_dir}  ({outcome})")
    return frame_paths, outcome


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
    # Pause longer on the final end-state frame so the outcome banner is readable.
    end_hold_ms = max(duration_ms * 3, 2000)
    durations = [duration_ms] * len(frames)
    if len(durations) > 1:
        durations[-1] = end_hold_ms
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=durations,
    )
    print(f"  GIF saved ({len(frames)} frames, {fps} fps) → {gif_path}")


def _infer_render_outcome(
    obs: Any,
    *,
    terminated: bool,
    truncated: bool,
) -> str:
    """Classify a rendered rollout as win/loss/draw/timeout from P1's perspective."""
    if terminated:
        obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
        p1_hp = float(obs_data.get("playerHealth", 0) or 0)
        p2_hp = float(obs_data.get("opponentHealth", 0) or 0)
        if p1_hp > p2_hp:
            return "win"
        if p2_hp > p1_hp:
            return "loss"
        return "draw"
    if truncated:
        return "timeout"
    return "timeout"


def _save_end_state_frame(
    env: Any,
    obs: Any,
    out_path: Path,
    *,
    outcome: str,
    steps: int = 0,
) -> bool:
    """Save a final board screenshot with a game-end outcome banner."""
    import base64  # noqa: PLC0415
    import io  # noqa: PLC0415

    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415
    except ImportError:
        return _save_state_image(env, obs, out_path)

    img = None
    try:
        rr = env.render()
        b64 = getattr(rr, "data", None)
        if b64:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception:
        pass

    if img is None:
        tmp = out_path.with_suffix(".tmp.png")
        if not _save_state_image(env, obs, tmp):
            return False
        try:
            img = Image.open(tmp).convert("RGB")
        finally:
            tmp.unlink(missing_ok=True)

    obs_data = json.loads(obs) if isinstance(obs, str) else (obs or {})
    p1_hp = obs_data.get("playerHealth", "?")
    p2_hp = obs_data.get("opponentHealth", "?")

    labels: dict[str, tuple[str, tuple[int, int, int]]] = {
        "win": ("WIN", (34, 197, 94)),
        "loss": ("LOSS", (239, 68, 68)),
        "draw": ("DRAW", (250, 204, 21)),
        "timeout": ("TIMEOUT", (249, 115, 22)),
        "stall_timeout": ("STALL TIMEOUT", (249, 115, 22)),
    }
    label, color = labels.get(outcome, (outcome.upper().replace("_", " "), (200, 200, 200)))

    draw = ImageDraw.Draw(img)
    width, height = img.size
    banner_h = max(72, height // 8)
    draw.rectangle([(0, height - banner_h), (width, height)], fill=(16, 16, 16))
    try:
        title_font = ImageFont.truetype("arial.ttf", max(28, banner_h // 3))
        sub_font = ImageFont.truetype("arial.ttf", max(16, banner_h // 5))
    except Exception:
        title_font = ImageFont.load_default()
        sub_font = title_font

    draw.text((24, height - banner_h + 10), label, fill=color, font=title_font)
    draw.text(
        (24, height - banner_h + 44),
        f"P1 {p1_hp} HP  |  P2 {p2_hp} HP  |  {steps} steps",
        fill=(220, 220, 220),
        font=sub_font,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return True


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
    opponent_equipment_header: str = "",
    game_format: str,
    opponent_deck_name: str,
    opponent_hero_id: str,
    opponent_mode: str,
    num_eval_episodes: int,
    max_steps: int,
    assets_path: str,
    base_url: str,
    fe_url: str,
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
    4. Render one full rollout by screenshotting the live Talishar frontend
       (Playwright + Chromium headless) after every game step, saving
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
        # Resolve the opponent equipment header: explicit param > PhaseAgents
        # field > fall back to P1's header so the deck file is always valid.
        _opp_equip = (
            opponent_equipment_header
            or getattr(opponent_agents, "equipment_header", "")
            or equipment_header
        )
        opp_file = _write_deck_file(
            opp_deck,
            _opp_equip,
            opp_name,
            assets_path,
        )
    else:
        opp_name = normalize_talishar_asset_name(opponent_deck_name, assets_path)
        opp_file = None

    # ── evaluation games ──────────────────────────────────────────────────────
    wins = 0
    losses = 0
    draws = 0
    episode_log: list[dict[str, Any]] = []

    backend_printed = False
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
                enable_combat_tracker=True,
            )
            try:
                if not backend_printed:
                    print(f"  [{player}] Runtime backend (final eval): {_runtime_backend_label(env)}")
                    backend_printed = True
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

    # ── render rollout (Playwright + Talishar frontend) ──────────────────────
    render_dir = out_dir / f"{player}_final_render"
    gif_path = out_dir / f"{player}_optimal_policy.gif"

    print(f"\n  [{player}] Rendering optimal-policy rollout via Talishar FE → {render_dir}")
    frame_paths, render_outcome = _render_game_with_talishar_frontend(
        agents=agents,
        opponent_agents=opponent_agents,
        opponent_mode=opponent_mode,
        base_url=base_url,
        fe_url=fe_url,
        game_format=game_format,
        deck_name=deck_name,
        opp_name=opp_name,
        max_steps=max_steps,
        render_dir=render_dir,
        player_label=player,
    )
    render_steps = max(0, len(frame_paths) - 1)

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
            "outcome": render_outcome,
            "terminated": render_outcome in ("win", "loss", "draw"),
            "truncated": render_outcome == "timeout",
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
            "deck_asset_name": p1.deck_asset_name,
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
            "deck_asset_name": p2.deck_asset_name,
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
    parser.add_argument("--play-checkpoint-interval", type=int, default=10000,
        help="Save phase-3 play weights every N episodes into results/.")
    parser.add_argument("--max-build-steps", type=int, default=200)
    parser.add_argument("--max-sideboard-steps", type=int, default=100)
    parser.add_argument("--max-play-steps", type=int, default=200)
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
        "--workers", type=int, default=None,
        help=(
            "Parallel game sessions for Phase 3 play training.  "
            ">=2 enables the parallel path in train_dual_agent_common.  "
            "Default: auto (4 for C++ engine, 1 for HTTP Talishar).  "
            "The C++ engine is thread-safe so high worker counts are safe; "
            "PPO gradient updates always run on GPU when CUDA is available."
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
    parser.add_argument(
        "--p1-fixed-deck", default=None,
        help=(
            "Path to a JSON file (fetch_fabrary_deck.py format) to pin P1 to a "
            "fixed card pool every iteration.  Phase 1 (deckbuilder) is skipped "
            "entirely.  Phase 2 (sideboard) is also skipped when the deck already "
            "meets the minimum play size (e.g. a 40-card Silver Age game deck)."
        ),
    )
    parser.add_argument(
        "--p2-fixed-deck", default=None,
        help=(
            "Same as --p1-fixed-deck but for P2 (dual mode).  Useful for "
            "evaluating a training P1 agent against a known fixed opponent deck."
        ),
    )

    # ── misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out-dir", default=str(_OUT_DIR))
    parser.add_argument("--results-json", default=None,
        help="Override path for results JSON (default: <out-dir>/results.json)")
    parser.add_argument("--talishar-url",
        default=os.environ.get("TALISHAR_URL", "http://localhost:8080/game"))
    parser.add_argument("--talishar-fe-url",
        default=os.environ.get("TALISHAR_FE_URL", "http://localhost:5173"),
        help="Talishar frontend URL for Playwright render screenshots (default: http://localhost:5173)")
    parser.add_argument("--cpp-engine-dir",
        default=None,
        help="Path to compiled C++ engine directory for fast deckbuild/sideboard eval (no HTTP).")
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

    # ── auto-detect worker count ──────────────────────────────────────────────
    # C++ engine: each worker has its own GameState (thread-safe, no shared
    # mutable state).  Use physical CPU cores capped at 8 for C++ engine,
    # always 1 for HTTP Talishar (server is the bottleneck).
    if args.workers is None:
        import os as _os
        import glob as _glob
        _cpp_cache = _os.path.join(str(_REPO_ROOT), "results", "cpp_engines")
        _cpp_deck1 = getattr(args, "hero_id", "") or ""
        _cpp_deck2 = getattr(args, "p2_hero_id", "") or _cpp_deck1
        _cpp_key   = f"{_cpp_deck1}_vs_{_cpp_deck2}"

        # Prefer the explicitly-passed engine dir; fall back to auto-discovery.
        _explicit = getattr(args, "cpp_engine_dir", None)
        if _explicit and _os.path.isdir(_explicit):
            _cpp_dir = _explicit
        else:
            # Exact match first, then hashed variant (e.g. aurora_vs_briar-<hash>)
            _exact = _os.path.join(_cpp_cache, _cpp_key)
            if _os.path.isdir(_exact):
                _cpp_dir = _exact
            else:
                _candidates = sorted(
                    _glob.glob(_os.path.join(_cpp_cache, f"{_cpp_key}-*")),
                    key=_os.path.getmtime, reverse=True,
                )
                _cpp_dir = _candidates[0] if _candidates else _exact

        _has_cpp = False
        if _os.path.isdir(_cpp_dir):
            try:
                if str(_REPO_ROOT / "src") not in sys.path:
                    sys.path.insert(0, str(_REPO_ROOT / "src"))
                from flesh_and_blood_rlbridge.cpp_engine_environment import (  # noqa: PLC0415
                    is_cpp_engine_available,
                    load_fab_engine,
                )
                if is_cpp_engine_available(_cpp_dir):
                    load_fab_engine(_cpp_dir)
                    _has_cpp = True
            except Exception:
                _has_cpp = False
        if _has_cpp:
            _cpu_count = max(1, _os.cpu_count() or 4)
            args.workers = min(_cpu_count, 16)
            print(f"  [auto] C++ engine detected -> {args.workers} parallel workers "
                  f"(set --workers to override)")
        else:
            args.workers = 1
            print("  [auto] No C++ engine found -> 1 worker (HTTP Talishar)")

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_path = args.assets_path or str(
        _REPO_ROOT / "Talishar" / "Assets"
    )
    args.opponent_deck = normalize_talishar_asset_name(args.opponent_deck, assets_path)
    if args.opponent_mode == "mirror":
        args.opponent_hero_id = args.hero_id
    elif not args.opponent_hero_id:
        args.opponent_hero_id = (
            args.p2_hero_id if args.opponent_mode == "dual" else _DEFAULT_OPPONENT_HERO
        )
    results_json = Path(args.results_json) if args.results_json else out_dir / "results.json"

    # ── build agent containers ────────────────────────────────────────────────
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

    # ── fixed decks (skip deckbuilding; also skip sideboard when game-ready) ──
    # A fixed deck overrides the deckbuilder every iteration.  If it already
    # meets the minimum play size no sideboard RL is needed — active_decks is
    # seeded directly from the fixed deck via a greedy cut.
    _p1_opp_key = args.p2_hero_id if args.opponent_mode == "dual" else args.opponent_hero_id
    p1_fixed_deck: Optional[dict[str, int]] = _load_starting_deck(
        getattr(args, "p1_fixed_deck", None)
    )
    p2_fixed_deck: Optional[dict[str, int]] = _load_starting_deck(
        getattr(args, "p2_fixed_deck", None)
    )
    if p1_fixed_deck:
        p1.card_pool = dict(p1_fixed_deck)
        _p1_fd_sz = sum(p1_fixed_deck.values())
        if _p1_fd_sz >= _min_size:
            p1.active_decks[_p1_opp_key] = _greedy_game_deck_cut(p1_fixed_deck, _min_size)
        print(
            f"  [p1] Fixed deck: {_p1_fd_sz} cards — Phase 1 skipped every iteration"
            + (" (Phase 2 also skipped — deck is game-ready)" if _p1_fd_sz >= _min_size else "")
        )
    if p2 is not None and p2_fixed_deck:
        p2.card_pool = dict(p2_fixed_deck)
        _p2_fd_sz = sum(p2_fixed_deck.values())
        if _p2_fd_sz >= _min_size:
            p2.active_decks[args.hero_id] = _greedy_game_deck_cut(p2_fixed_deck, _min_size)
        print(
            f"  [p2] Fixed deck: {_p2_fd_sz} cards — Phase 1 skipped every iteration"
            + (" (Phase 2 also skipped — deck is game-ready)" if _p2_fd_sz >= _min_size else "")
        )

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
            cpp_engine_dir=args.cpp_engine_dir,
            checkpoint_interval=args.play_checkpoint_interval,
        )
        p1.last_play_win_rate = p1_wr
        if p2 is not None:
            p2.last_play_win_rate = p2_wr

        # ── Phase 1: Deckbuilder ───────────────────────────────────────────────
        p1_eval_opponent_hero = (
            args.p2_hero_id if args.opponent_mode == "dual" else args.opponent_hero_id
        )

        def _p1_opponent_deck() -> str:
            return resolve_opponent_deck_name(
                player_hero_id=args.hero_id,
                opponent_mode=args.opponent_mode,
                preset_opponent_deck=args.opponent_deck,
                opponent_agents=p2,
                opponent_hero_id=args.p2_hero_id,
                assets_path=assets_path,
                min_deck_size=_min_size,
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
                min_deck_size=_min_size,
                write_deck_file=_write_deck_file,
                opponent_equipment_header=args.equipment_header,
            )

        if p1_fixed_deck:
            # Fixed deck supplied — deckbuilding is always skipped for p1.
            p1.card_pool = dict(p1_fixed_deck)
            print(f"\n  [p1] Phase 1 skipped — using fixed deck ({sum(p1_fixed_deck.values())} cards)")
        else:
            # On iteration 1 use the FaBrary warm-start; on later iterations feed
            # back the best pool from the previous iteration.
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
                play_reward=p1.last_play_win_rate,
            )

        if args.opponent_mode == "dual" and p2 is not None:
            if p2_fixed_deck:
                p2.card_pool = dict(p2_fixed_deck)
                print(f"\n  [p2] Phase 1 skipped — using fixed deck ({sum(p2_fixed_deck.values())} cards)")
            else:
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
                    play_reward=p2.last_play_win_rate,
                )

        # ── Phase 2: Sideboard ────────────────────────────────────────────────
        if p1_fixed_deck and sum(p1_fixed_deck.values()) >= _min_size:
            # Fixed deck already meets minimum play size — cut it directly and
            # skip the sideboard RL phase for p1.
            _p1_game_deck = _greedy_game_deck_cut(p1_fixed_deck, _min_size)
            p1.active_decks[_p1_opp_key] = _p1_game_deck
            print(
                f"\n  [p1] Phase 2 skipped — fixed deck is game-ready "
                f"({sum(p1_fixed_deck.values())} ≥ {_min_size}, "
                f"game deck: {sum(_p1_game_deck.values())} cards)"
            )
        else:
            run_phase2_sideboard(
                p1,
                p1_opponents,
                hero_id=args.hero_id,
                equipment_header=args.equipment_header,
                game_format=args.format,
                opponent_deck_name=_p1_opponent_deck(),
                n_episodes_per_opponent=args.sideboard_episodes,
                max_sideboard_steps=args.max_sideboard_steps,
                num_eval_games=args.num_eval_games,
                assets_path=assets_path,
                base_url=args.talishar_url,
                render=args.render,
                cpp_engine_dir=args.cpp_engine_dir,
                play_reward=p1.last_play_win_rate,
            )

        if args.opponent_mode == "dual" and p2 is not None:
            if p2_fixed_deck and sum(p2_fixed_deck.values()) >= _min_size:
                _p2_game_deck = _greedy_game_deck_cut(p2_fixed_deck, _min_size)
                p2.active_decks[args.hero_id] = _p2_game_deck
                print(
                    f"\n  [p2] Phase 2 skipped — fixed deck is game-ready "
                    f"({sum(p2_fixed_deck.values())} ≥ {_min_size}, "
                    f"game deck: {sum(_p2_game_deck.values())} cards)"
                )
            else:
                run_phase2_sideboard(
                    p2,
                    p2_opponents,
                    hero_id=args.p2_hero_id,
                    equipment_header=args.p2_equipment_header,
                    game_format=args.format,
                    opponent_deck_name=_p2_opponent_deck(),
                    n_episodes_per_opponent=args.sideboard_episodes,
                    max_sideboard_steps=args.max_sideboard_steps,
                    num_eval_games=args.num_eval_games,
                    assets_path=assets_path,
                    base_url=args.talishar_url,
                    render=args.render,
                    cpp_engine_dir=args.cpp_engine_dir,
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
        opponent_equipment_header=args.p2_equipment_header,
        game_format=args.format,
        opponent_deck_name=args.opponent_deck,
        opponent_hero_id=args.opponent_hero_id if args.opponent_mode != "dual"
                         else args.p2_hero_id,
        opponent_mode=args.opponent_mode,
        num_eval_episodes=args.final_eval_episodes,
        max_steps=args.final_eval_max_steps,
        assets_path=assets_path,
        base_url=args.talishar_url,
        fe_url=args.talishar_fe_url,
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
