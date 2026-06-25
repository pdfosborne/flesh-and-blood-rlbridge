"""Shared types and helpers for the FaB training pipeline."""

from __future__ import annotations

import json
import os
import pickle
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_ROOT))
import _bootstrap  # noqa: E402

REPO_ROOT = _bootstrap.configure_paths()
_RL_SRC = Path("~/Documents/RL/rlbridge/src").expanduser()
if str(_RL_SRC) not in sys.path:
    sys.path.insert(0, str(_RL_SRC))

from flesh_and_blood_rlbridge import (  # noqa: E402
    TalisharDeckBuilderEnvironment,
    TalisharSideboardEnvironment,
)
from flesh_and_blood_rlbridge.sideboard_guide_policy import (  # noqa: E402
    RankedSwap,
    SideboardGuidePolicy,
    apply_sideboard_swap,
    enumerate_ranked_swaps,
    simulate_guide_sideboard_deck,
)
from flesh_and_blood_rlbridge.opponent_deck import (  # noqa: E402
    hero_class_for_id,
    normalize_talishar_asset_name,
    resolve_opponent_deck_name,
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
_DEFAULT_OPPONENT_DECK = "Ira"
_DEFAULT_OPPONENT_HERO = "dorinthea_ironsong"

_OUT_DIR = REPO_ROOT / "results" / "full_pipeline"
_AGENT_CACHE_DIR = REPO_ROOT / "results" / "agent_cache"


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
    last_play_win_rate: float = 0.0
    # Sideboard-first pipeline: True once sideboard win rates stabilise on the
    # current card pool, unlocking deckbuilder training.
    sideboard_converged: bool = False
    # Per-opponent rolling sideboard eval win rates (from C++ play episodes).
    sideboard_win_rates: dict[str, list[float]] = field(default_factory=dict)


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

    ensure_pool_metadata(
        agents,
        hero_id=hero_id,
        hero_class=hero_class,
        game_format=game_format,
    )
    agents.sideboard = SideboardGuidePolicy(pool_by_id=agents.pool_by_id)

    env = TalisharDeckBuilderEnvironment(
        hero_id=hero_id,
        hero_class=hero_class,
        hero_equipment_header=equipment_header,
        game_format=game_format,
        num_eval_games=effective_eval_games,
        opponent_deck_name=opponent_deck_name,
        opponent_hero_id=opponent_hero_id,
        sideboard_agent=agents.sideboard,
        num_sideboard_episodes=max(1, min(num_sideboard_episodes, 1)),
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

DEFAULT_SIDEBOARD_EVAL_GAMES = 1000
DEFAULT_SIDEBOARD_BATCH_SIZE = 4
DEFAULT_SIDEBOARD_CONVERGENCE_WINDOW = 5
DEFAULT_SIDEBOARD_CONVERGENCE_STD = 0.02


def ensure_pool_metadata(
    agents: PhaseAgents,
    *,
    hero_id: str,
    hero_class: str,
    game_format: str,
) -> None:
    """Populate ``agents.pool_by_id`` when only a card pool dict is available."""
    if agents.pool_by_id:
        return
    builder = TalisharDeckBuilderEnvironment(
        hero_id=hero_id,
        hero_class=hero_class,
        game_format=game_format,
    )
    agents.pool_by_id = dict(builder._pool_by_id)


def _sideboard_policy(agents: PhaseAgents) -> Any:
    if agents.sideboard is not None and hasattr(agents.sideboard, "act"):
        return agents.sideboard
    guide = SideboardGuidePolicy(pool_by_id=agents.pool_by_id)
    agents.sideboard = guide
    print(f"  [{agents.player}] Using SideboardGuidePolicy warm-start")
    return guide


def _reward_to_win_rate(reward: float) -> float:
    """Map sideboard finalize reward ``win_rate * 2 - 1`` back to ``[0, 1]``."""
    return max(0.0, min(1.0, (reward + 1.0) / 2.0))


def sideboard_convergence_reached(
    history: list[float],
    *,
    window: int,
    max_std: float,
) -> bool:
    if len(history) < window:
        return False
    recent = history[-window:]
    if len(recent) < 2:
        return False
    mean = sum(recent) / len(recent)
    variance = sum((x - mean) ** 2 for x in recent) / len(recent)
    return variance ** 0.5 <= max_std


def _run_one_sideboard_episode(
    env: TalisharSideboardEnvironment,
    policy: Any,
    *,
    max_sideboard_steps: int,
    render: bool,
) -> tuple[float, dict[str, int], bool, Optional[float]]:
    result = env.reset()
    ep_reward = 0.0
    finalize_reward: Optional[float] = None
    done = False
    steps = 0
    while not done and steps < max_sideboard_steps:
        action = policy.act(result.observation)
        step = env.step(action)
        ep_reward += step.reward
        if step.terminated:
            finalize_reward = step.reward
        done = step.terminated or step.truncated
        if render and not done:
            r = env.render()
            if r.text:
                print(r.text)
        result.observation = step.observation
        steps += 1
    deck = env.get_active_deck()
    valid = sum(deck.values()) >= env._min_deck_size
    win_rate: Optional[float] = None
    if valid and finalize_reward is not None and finalize_reward > -0.99:
        win_rate = _reward_to_win_rate(finalize_reward)
    return ep_reward, deck, valid, win_rate


def run_phase2_sideboard(
    agents: PhaseAgents,
    opponent_hero_ids: list[str],
    *,
    hero_id: str,
    hero_class: str,
    equipment_header: str,
    game_format: str,
    opponent_deck_name: str,
    n_episodes_per_opponent: int,
    max_sideboard_steps: int,
    sideboard_eval_games: int,
    assets_path: Optional[str],
    base_url: str,
    render: bool,
    cpp_engine_dir: Optional[str] = None,
    sideboard_batch_size: int = DEFAULT_SIDEBOARD_BATCH_SIZE,
    convergence_window: int = DEFAULT_SIDEBOARD_CONVERGENCE_WINDOW,
    convergence_std: float = DEFAULT_SIDEBOARD_CONVERGENCE_STD,
    play_reward: float = 0.0,
) -> bool:
    """Train the sideboard agent for each opponent; return True if all converged.

    Each episode: agent selects a play deck, then C++ engine runs
    ``sideboard_eval_games`` play episodes with the fixed Talishar default
    policy (``env.sample_action``).  Reward is win rate only; episode ends
    on finalize.

    When ``play_reward`` is non-zero it is added as a small terminal bonus
    after sideboard eval (legacy feedback from play training).
    """
    print(
        f"\n{'='*62}\n"
        f"  PHASE 2 — Sideboard  [{agents.player} / {hero_id} / {game_format}]\n"
        f"  Eval: {sideboard_eval_games} C++ games/ep  |  "
        f"batch={sideboard_batch_size}  |  fixed default play policy\n"
        f"{'='*62}"
    )

    if not agents.card_pool:
        print(f"  [{agents.player}] WARNING: empty card pool — skipping sideboard phase")
        return False

    ensure_pool_metadata(
        agents,
        hero_id=hero_id,
        hero_class=hero_class,
        game_format=game_format,
    )
    policy = _sideboard_policy(agents)

    all_converged = True
    batch_size = max(1, sideboard_batch_size)

    for opponent in opponent_hero_ids:
        print(f"\n  [{agents.player}] Opponent: {opponent}")
        history = agents.sideboard_win_rates.setdefault(opponent, [])

        env = TalisharSideboardEnvironment(
            card_pool=agents.card_pool,
            pool_by_id=agents.pool_by_id,
            opponent_hero_id=opponent,
            hero_id=hero_id,
            hero_equipment_header=equipment_header,
            game_format=game_format,
            num_eval_games=sideboard_eval_games,
            opponent_deck_name=opponent_deck_name,
            eval_p1_agent=None,
            eval_p2_agent=None,
            base_url=base_url,
            talishar_assets_path=assets_path,
            max_sideboard_steps=max_sideboard_steps,
            render_mode="ansi" if render else None,
            cpp_engine_dir=cpp_engine_dir,
        )
        best_reward = float("-inf")
        best_deck: dict[str, int] = {}

        ep = 0
        while ep < n_episodes_per_opponent:
            batch_end = min(ep + batch_size, n_episodes_per_opponent)
            batch_rewards: list[float] = []
            batch_win_rates: list[float] = []

            for batch_ep in range(ep + 1, batch_end + 1):
                ep_reward, deck, valid, win_rate = _run_one_sideboard_episode(
                    env,
                    policy,
                    max_sideboard_steps=max_sideboard_steps,
                    render=render,
                )
                if valid and play_reward != 0.0:
                    ep_reward += play_reward

                deck_size = sum(deck.values())
                if valid and win_rate is not None:
                    batch_win_rates.append(win_rate)
                    history.append(win_rate)
                    compare_reward = (win_rate * 2.0 - 1.0) + (
                        play_reward if play_reward else 0.0
                    )
                    if compare_reward > best_reward:
                        best_reward = compare_reward
                        best_deck = dict(deck)
                batch_rewards.append(ep_reward)

                if render or batch_ep == batch_end:
                    wr_tag = (
                        f"  win%={win_rate:.1%}"
                        if valid and win_rate is not None
                        else ""
                    )
                    print(
                        f"    [{agents.player}] Ep {batch_ep:>4}/{n_episodes_per_opponent}  "
                        f"reward={ep_reward:+.3f}  deck={deck_size}  valid={valid}{wr_tag}"
                    )

            ep = batch_end
            if batch_win_rates:
                batch_mean = sum(batch_win_rates) / len(batch_win_rates)
                print(
                    f"    [{agents.player}] Batch {ep // batch_size}: "
                    f"mean win%={batch_mean:.1%}  "
                    f"(last {min(len(history), convergence_window)} "
                    f"σ={_rolling_std(history, convergence_window):.3f})"
                )

        agents.active_decks[opponent] = best_deck if best_deck else dict(agents.card_pool)
        opp_converged = sideboard_convergence_reached(
            history,
            window=convergence_window,
            max_std=convergence_std,
        )
        if not opp_converged:
            all_converged = False
        conv_tag = "converged" if opp_converged else "still training"
        print(
            f"  [{agents.player}] → vs {opponent}: "
            f"{sum(agents.active_decks[opponent].values())} cards  "
            f"(best reward {best_reward:+.3f}, {conv_tag})"
        )

    agents.sideboard_converged = all_converged
    if all_converged:
        print(f"  [{agents.player}] Sideboard converged — deckbuilding enabled")
    return all_converged


def apply_guide_sideboard_for_matchup(
    agents: PhaseAgents,
    opponent_hero_ids: list[str],
    *,
    hero_id: str,
    hero_class: str,
    equipment_header: str,
    game_format: str,
    opponent_deck_name: str,
    max_sideboard_steps: int,
    assets_path: Optional[str],
    base_url: str,
    cpp_engine_dir: Optional[str] = None,
) -> None:
    """Pick one game deck per opponent via SideboardGuidePolicy (no RL, no C++ eval).

    Runs a single sideboard episode per opponent using the Silver Age guide
    heuristics.  Skips sideboard-agent training and the expensive C++ play
    eval that ``run_phase2_sideboard`` uses for convergence tracking.
    """
    print(
        f"\n{'='*62}\n"
        f"  Sideboard (guide policy)  [{agents.player} / {hero_id} / {game_format}]\n"
        f"  1 episode/opponent  |  no C++ eval  |  SideboardGuidePolicy\n"
        f"{'='*62}"
    )

    if not agents.card_pool:
        print(f"  [{agents.player}] WARNING: empty card pool — skipping sideboard")
        return

    ensure_pool_metadata(
        agents,
        hero_id=hero_id,
        hero_class=hero_class,
        game_format=game_format,
    )
    policy = SideboardGuidePolicy(pool_by_id=agents.pool_by_id)
    agents.sideboard = policy

    min_size = min_deck_size_for_format(game_format)

    for opponent in opponent_hero_ids:
        print(f"\n  [{agents.player}] Opponent: {opponent}")
        env = TalisharSideboardEnvironment(
            card_pool=agents.card_pool,
            pool_by_id=agents.pool_by_id,
            opponent_hero_id=opponent,
            hero_id=hero_id,
            hero_equipment_header=equipment_header,
            game_format=game_format,
            num_eval_games=0,
            opponent_deck_name=opponent_deck_name,
            eval_p1_agent=None,
            eval_p2_agent=None,
            base_url=base_url,
            talishar_assets_path=assets_path,
            max_sideboard_steps=max_sideboard_steps,
            cpp_engine_dir=cpp_engine_dir,
        )
        _, deck, valid, _ = _run_one_sideboard_episode(
            env,
            policy,
            max_sideboard_steps=max_sideboard_steps,
            render=False,
        )
        if valid and sum(deck.values()) >= min_size:
            agents.active_decks[opponent] = dict(deck)
        else:
            agents.active_decks[opponent] = greedy_game_deck_cut(
                agents.card_pool, min_size
            )
            print(
                f"  [{agents.player}] Guide sideboard invalid — "
                f"using greedy {min_size}-card cut"
            )
        deck_size = sum(agents.active_decks[opponent].values())
        print(
            f"  [{agents.player}] → vs {opponent}: {deck_size} cards "
            f"(SideboardGuidePolicy)"
        )

    agents.sideboard_converged = True


def _rolling_std(history: list[float], window: int) -> float:
    if len(history) < 2:
        return 1.0
    recent = history[-window:]
    if len(recent) < 2:
        return 1.0
    mean = sum(recent) / len(recent)
    variance = sum((x - mean) ** 2 for x in recent) / len(recent)
    return variance ** 0.5


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

    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    from flesh_and_blood_rlbridge.card_db.talishar_card_ids import (  # noqa: PLC0415
        sanitize_deck_for_talishar,
    )

    deck, deck_warnings = sanitize_deck_for_talishar(deck)
    for warning in deck_warnings:
        print(f"  [deck] {warning}")

    if equipment_header:
        from flesh_and_blood_rlbridge.card_db.talishar_card_ids import (  # noqa: PLC0415
            TalisharCardIdResolver,
        )

        resolver = TalisharCardIdResolver()
        header_parts = equipment_header.split()
        if header_parts:
            fixed_header: list[str] = []
            for idx, part in enumerate(header_parts):
                resolved = resolver.resolve(part) if idx > 0 else part
                if idx > 0 and resolved is None:
                    print(f"  [deck] Dropped unknown equipment id from header: {part}")
                    continue
                fixed_header.append(resolved if resolved is not None else part)
            equipment_header = " ".join(fixed_header)

    card_ids: list[str] = []
    for card_id, count in sorted(deck.items()):
        card_ids.extend([card_id] * count)
    content = f"{equipment_header}\n{' '.join(card_ids)}\n"
    out_path = Path(assets_path) / f"{deck_name}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path

def _save_all_agents(agents: PhaseAgents, out_dir: Path) -> None:
    prefix = out_dir / agents.player
    if agents.deckbuilder is not None:
        _save_agent(agents.deckbuilder, prefix.parent / f"{agents.player}_deckbuilder.pkl")
    if agents.sideboard is not None:
        _save_agent(agents.sideboard, prefix.parent / f"{agents.player}_sideboard.pkl")
    if agents.play is not None:
        _save_agent(agents.play, prefix.parent / f"{agents.player}_play.pkl")


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
        fe = {
            k: v for k, v in p1_final_eval.get("eval", {}).items()
            if k != "episode_log"
        }
        analysis = p1_final_eval.get("analysis") or {}
        charts = analysis.get("charts") or {}
        if charts.get("hp_by_turn"):
            fe["hp_chart"] = charts["hp_by_turn"]
        matchup_deck = p1_final_eval.get("matchup_deck") or {}
        if matchup_deck.get("json"):
            fe["matchup_deck_json"] = matchup_deck["json"]
        if matchup_deck.get("image"):
            fe["matchup_deck_image"] = matchup_deck["image"]
        data["p1"]["final_eval"] = fe
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
            fe2 = {
                k: v for k, v in p2_final_eval.get("eval", {}).items()
                if k != "episode_log"
            }
            analysis2 = p2_final_eval.get("analysis") or {}
            charts2 = analysis2.get("charts") or {}
            if charts2.get("hp_by_turn"):
                fe2["hp_chart"] = charts2["hp_by_turn"]
            matchup_deck2 = p2_final_eval.get("matchup_deck") or {}
            if matchup_deck2.get("json"):
                fe2["matchup_deck_json"] = matchup_deck2["json"]
            if matchup_deck2.get("image"):
                fe2["matchup_deck_image"] = matchup_deck2["image"]
            data["p2"]["final_eval"] = fe2
            data["p2"]["final_eval_gif"] = p2_final_eval.get("render", {}).get("gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n  Results written → {out_path}")

def min_deck_size_for_format(game_format: str) -> int:
    fmt = str(game_format or "silver_age").lower().replace(" ", "_")
    if fmt in {"silver_age", "sage", "blitz"}:
        return 40
    return 60


_CARDS_DB_PATH = (
    REPO_ROOT / "src" / "flesh_and_blood_rlbridge" / "card_db" / "cards.json"
)
_cards_db_index: Optional[dict[str, dict[str, Any]]] = None


def _load_cards_db_index() -> dict[str, dict[str, Any]]:
    """Lazy index of ``cards.json`` by Talishar underscore card id."""
    global _cards_db_index
    if _cards_db_index is not None:
        return _cards_db_index
    index: dict[str, dict[str, Any]] = {}
    try:
        for rec in json.loads(_CARDS_DB_PATH.read_text(encoding="utf-8")):
            if not isinstance(rec, dict):
                continue
            cid = str(rec.get("id") or "").strip()
            if cid:
                index[cid] = rec
    except Exception:
        pass
    _cards_db_index = index
    return index


def sideboard_from_pool(
    card_pool: dict[str, int],
    game_deck: dict[str, int],
) -> dict[str, int]:
    """Cards registered in the pool but not included in the game deck."""
    sideboard: dict[str, int] = {}
    for card_id in set(card_pool) | set(game_deck):
        remaining = int(card_pool.get(card_id, 0)) - int(game_deck.get(card_id, 0))
        if remaining > 0:
            sideboard[card_id] = remaining
    return sideboard


@dataclass(frozen=True)
class SideboardCandidate:
    """One sideboard variant to train and compare in play."""

    candidate_id: str
    label: str
    game_deck: dict[str, int]
    swaps: tuple[tuple[str, str], ...] = ()
    guide_margin: Optional[float] = None
    equipment_header: Optional[str] = None


def _deck_signature(deck: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(k), int(v)) for k, v in deck.items() if int(v) > 0))


def load_deck_and_pool_from_json(
    deck_json_path: Optional[str],
    *,
    card_pool_path: Optional[str] = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Load a game deck and registered card pool from JSON export files.

    When ``deck`` and ``sideboard`` keys are both present, the pool is their
    union.  When only ``deck`` is present and it is larger than the format
    minimum, that dict is treated as both pool and starting game deck.
    """
    if not deck_json_path:
        return {}, {}

    data = json.loads(Path(deck_json_path).read_text(encoding="utf-8"))
    game_deck = {
        str(k): int(v) for k, v in (data.get("deck") or {}).items() if int(v) > 0
    }
    sideboard_cards = {
        str(k): int(v) for k, v in (data.get("sideboard") or {}).items() if int(v) > 0
    }

    if card_pool_path:
        pool_data = json.loads(Path(card_pool_path).read_text(encoding="utf-8"))
        if pool_data.get("card_pool"):
            card_pool = {
                str(k): int(v) for k, v in pool_data["card_pool"].items() if int(v) > 0
            }
        elif pool_data.get("deck"):
            card_pool = {
                str(k): int(v) for k, v in pool_data["deck"].items() if int(v) > 0
            }
        else:
            card_pool = dict(game_deck)
    elif sideboard_cards:
        card_pool = dict(game_deck)
        for cid, count in sideboard_cards.items():
            card_pool[cid] = card_pool.get(cid, 0) + count
    else:
        card_pool = dict(game_deck)

    return game_deck, card_pool


def generate_sideboard_candidates(
    card_pool: dict[str, int],
    game_deck: dict[str, int],
    opponent_hero_id: str,
    pool_by_id: dict[str, Any],
    *,
    hero_id: str,
    game_format: str,
    num_options: int,
    include_baseline: bool = True,
    include_guide_full: bool = True,
    min_swap_margin: float = 0.75,
) -> list[SideboardCandidate]:
    """Build sideboard variants for play comparison using guide policy swaps."""
    if num_options <= 0:
        return []

    min_size = min_deck_size_for_format(game_format)
    normalized_deck = greedy_game_deck_cut(game_deck, min_size) if game_deck else {}
    if sum(normalized_deck.values()) < min_size:
        normalized_deck = greedy_game_deck_cut(card_pool, min_size)

    candidates: list[SideboardCandidate] = []
    seen: set[tuple[tuple[str, int], ...]] = set()

    def _add(candidate: SideboardCandidate) -> None:
        if len(candidates) >= num_options:
            return
        sig = _deck_signature(candidate.game_deck)
        if sig in seen or sum(candidate.game_deck.values()) < min_size:
            return
        seen.add(sig)
        candidates.append(candidate)

    if include_baseline and normalized_deck:
        _add(
            SideboardCandidate(
                candidate_id="baseline",
                label="Starting deck",
                game_deck=dict(normalized_deck),
            )
        )

    if include_guide_full and card_pool:
        guide_deck = simulate_guide_sideboard_deck(
            card_pool,
            opponent_hero_id,
            hero_id=hero_id,
            game_format=game_format,
            pool_by_id=pool_by_id,
        )
        if guide_deck:
            _add(
                SideboardCandidate(
                    candidate_id="guide_full",
                    label="Guide policy (full sideboard)",
                    game_deck=dict(guide_deck),
                )
            )

    ranked_swaps: list[RankedSwap] = enumerate_ranked_swaps(
        card_pool,
        normalized_deck,
        opponent_hero_id,
        pool_by_id,
        min_margin=min_swap_margin,
    )
    swap_index = 0
    for swap in ranked_swaps:
        if len(candidates) >= num_options:
            break
        swapped = apply_sideboard_swap(
            normalized_deck,
            card_pool,
            swap.out_card,
            swap.in_card,
        )
        if not swapped:
            continue
        swap_index += 1
        _add(
            SideboardCandidate(
                candidate_id=f"swap_{swap_index:02d}",
                label=(
                    f"Swap {swap.out_card} → {swap.in_card} "
                    f"(margin {swap.margin:+.2f})"
                ),
                game_deck=swapped,
                swaps=((swap.out_card, swap.in_card),),
                guide_margin=swap.margin,
            )
        )

    return candidates


def load_sideboard_candidates_from_json(
    path: str | Path,
    *,
    card_pool: dict[str, int],
    min_deck_size: int,
) -> tuple[list[SideboardCandidate], dict[str, int]]:
    """Load comparison candidates written by the TUI sideboard picker."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pool = dict(card_pool)
    if data.get("card_pool"):
        for cid, count in data["card_pool"].items():
            pool[str(cid)] = int(count)
    default_equipment_header = str(data.get("equipment_header") or "").strip() or None

    candidates: list[SideboardCandidate] = []
    for raw in data.get("candidates") or []:
        game_deck = {
            str(k): int(v) for k, v in (raw.get("game_deck") or {}).items() if int(v) > 0
        }
        if sum(game_deck.values()) < min_deck_size:
            continue
        swaps_raw = raw.get("swaps") or []
        swaps = tuple(
            (str(pair[0]), str(pair[1]))
            for pair in swaps_raw
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        )
        raw_equipment = str(raw.get("equipment_header") or "").strip()
        equipment_header = raw_equipment or default_equipment_header
        candidates.append(
            SideboardCandidate(
                candidate_id=str(raw.get("candidate_id") or f"candidate_{len(candidates)}"),
                label=str(raw.get("label") or raw.get("candidate_id") or "candidate"),
                game_deck=game_deck,
                swaps=swaps,
                guide_margin=(
                    float(raw["guide_margin"])
                    if raw.get("guide_margin") is not None
                    else None
                ),
                equipment_header=equipment_header,
            )
        )
    return candidates, pool


def _card_sort_key(
    card_id: str,
    *,
    pool_by_id: dict[str, dict[str, Any]],
    cards_db: dict[str, dict[str, Any]],
) -> tuple[int, str, str]:
    meta = pool_by_id.get(card_id) or cards_db.get(card_id) or {}
    pitch = meta.get("pitch")
    try:
        pitch_key = -int(pitch) if pitch is not None else 0
    except (TypeError, ValueError):
        pitch_key = 0
    name = str(meta.get("name") or card_id).lower()
    return (pitch_key, name, card_id)


def _format_card_entries(
    cards: dict[str, int],
    *,
    pool_by_id: dict[str, dict[str, Any]],
    image_base_url: str,
) -> list[dict[str, Any]]:
    """Sorted card list with display names and Talishar WebP image URLs."""
    cards_db = _load_cards_db_index()
    base = image_base_url.rstrip("/")
    ordered = sorted(
        cards.items(),
        key=lambda item: _card_sort_key(item[0], pool_by_id=pool_by_id, cards_db=cards_db),
    )
    entries: list[dict[str, Any]] = []
    for card_id, count in ordered:
        if count <= 0:
            continue
        meta = pool_by_id.get(card_id) or cards_db.get(card_id) or {}
        entry: dict[str, Any] = {
            "id": card_id,
            "name": str(meta.get("name") or card_id.replace("_", " ").title()),
            "count": int(count),
            "image_url": f"{base}/WebpImages/{card_id}.webp",
        }
        if meta.get("pitch") is not None:
            entry["pitch"] = int(meta["pitch"])
        if meta.get("type_line"):
            entry["type_line"] = str(meta["type_line"])
        entries.append(entry)
    return entries


def _equipment_entries(
    equipment_header: str,
    hero_id: str,
    *,
    pool_by_id: dict[str, dict[str, Any]],
    image_base_url: str,
) -> list[dict[str, Any]]:
    parts = (equipment_header or "").split()
    if len(parts) <= 1:
        return []
    hero_token = hero_id.replace("-", "_").lower()
    equip_ids = [
        part for part in parts[1:]
        if part.replace("-", "_").lower() != hero_token
    ]
    if not equip_ids:
        return []
    return _format_card_entries(
        {equip_id: 1 for equip_id in equip_ids},
        pool_by_id=pool_by_id,
        image_base_url=image_base_url,
    )


def build_matchup_deck_export(
    agents: PhaseAgents,
    *,
    hero_id: str,
    opponent_hero_id: str,
    game_format: str,
    equipment_header: str,
    game_deck: dict[str, int],
    image_base_url: str,
) -> dict[str, Any]:
    """Build a clean JSON export of pool, game deck, and sideboard for a matchup."""
    card_pool = agents.card_pool or dict(game_deck)
    ensure_pool_metadata(
        agents,
        hero_id=hero_id,
        hero_class=hero_class_for_id(hero_id),
        game_format=game_format,
    )
    pool_by_id = agents.pool_by_id
    sideboard = sideboard_from_pool(card_pool, game_deck)
    image_base = image_base_url.rstrip("/")

    return {
        "player": agents.player,
        "hero_id": hero_id,
        "opponent_hero_id": opponent_hero_id,
        "matchup": f"{hero_id} vs {opponent_hero_id}",
        "format": game_format,
        "equipment_header": equipment_header,
        "equipment": _equipment_entries(
            equipment_header,
            hero_id,
            pool_by_id=pool_by_id,
            image_base_url=image_base,
        ),
        "pool_size": sum(card_pool.values()),
        "game_deck_size": sum(game_deck.values()),
        "sideboard_size": sum(sideboard.values()),
        "card_pool": {
            "total_cards": sum(card_pool.values()),
            "cards": _format_card_entries(
                card_pool,
                pool_by_id=pool_by_id,
                image_base_url=image_base,
            ),
        },
        "game_deck": {
            "total_cards": sum(game_deck.values()),
            "cards": _format_card_entries(
                game_deck,
                pool_by_id=pool_by_id,
                image_base_url=image_base,
            ),
        },
        "sideboard": {
            "total_cards": sum(sideboard.values()),
            "cards": _format_card_entries(
                sideboard,
                pool_by_id=pool_by_id,
                image_base_url=image_base,
            ),
        },
        "image_base_url": image_base,
    }


def greedy_game_deck_cut(
    pool: dict[str, int],
    min_size: int,
    *,
    max_copies: int | None = None,
) -> dict[str, int]:
    """Take the first min_size cards from pool (deterministic greedy cut)."""
    game_deck: dict[str, int] = {}
    remaining = min_size
    for card_id, count in pool.items():
        if remaining <= 0:
            break
        available = int(count)
        if max_copies is not None:
            available = min(available, max_copies)
        take = min(available, remaining)
        if take > 0:
            game_deck[card_id] = take
            remaining -= take
    return game_deck


def resolve_assets_path(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return str(REPO_ROOT / "Talishar" / "Assets")


def deck_state_path(out_dir: Path) -> Path:
    return out_dir / "deck_state.json"


def save_deck_state(
    out_dir: Path,
    *,
    p1: PhaseAgents,
    p2: Optional[PhaseAgents],
    game_format: str,
    opponent_mode: str,
) -> Path:
    """Persist full card pools and active decks for cross-script handoff."""
    path = deck_state_path(out_dir)
    data: dict[str, Any] = {
        "format": game_format,
        "opponent_mode": opponent_mode,
        "p1": {
            "card_pool": p1.card_pool,
            "active_decks": p1.active_decks,
            "last_play_win_rate": p1.last_play_win_rate,
            "win_rates": p1.win_rates,
            "sideboard_converged": p1.sideboard_converged,
            "sideboard_win_rates": p1.sideboard_win_rates,
        },
    }
    if p2 is not None:
        data["p2"] = {
            "card_pool": p2.card_pool,
            "active_decks": p2.active_decks,
            "last_play_win_rate": p2.last_play_win_rate,
            "win_rates": p2.win_rates,
            "sideboard_converged": p2.sideboard_converged,
            "sideboard_win_rates": p2.sideboard_win_rates,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_deck_state(out_dir: Path) -> Optional[dict[str, Any]]:
    path = deck_state_path(out_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def apply_deck_state(
    p1: PhaseAgents,
    p2: Optional[PhaseAgents],
    state: dict[str, Any],
) -> None:
    p1_data = state.get("p1", {})
    if p1_data.get("card_pool"):
        p1.card_pool = {str(k): int(v) for k, v in p1_data["card_pool"].items()}
    if p1_data.get("active_decks"):
        p1.active_decks = {
            str(k): {str(cid): int(c) for cid, c in deck.items()}
            for k, deck in p1_data["active_decks"].items()
        }
    p1.last_play_win_rate = float(p1_data.get("last_play_win_rate", 0.0) or 0.0)
    p1.win_rates = list(p1_data.get("win_rates", []))
    p1.sideboard_converged = bool(p1_data.get("sideboard_converged", False))
    raw_sb = p1_data.get("sideboard_win_rates", {})
    if raw_sb:
        p1.sideboard_win_rates = {
            str(k): [float(x) for x in v]
            for k, v in raw_sb.items()
        }
    if p2 is not None and "p2" in state:
        p2_data = state["p2"]
        if p2_data.get("card_pool"):
            p2.card_pool = {str(k): int(v) for k, v in p2_data["card_pool"].items()}
        if p2_data.get("active_decks"):
            p2.active_decks = {
                str(k): {str(cid): int(c) for cid, c in deck.items()}
                for k, deck in p2_data["active_decks"].items()
            }
        p2.last_play_win_rate = float(p2_data.get("last_play_win_rate", 0.0) or 0.0)
        p2.win_rates = list(p2_data.get("win_rates", []))
        p2.sideboard_converged = bool(p2_data.get("sideboard_converged", False))
        raw_sb2 = p2_data.get("sideboard_win_rates", {})
        if raw_sb2:
            p2.sideboard_win_rates = {
                str(k): [float(x) for x in v]
                for k, v in raw_sb2.items()
            }


def load_play_feedback(out_dir: Path) -> tuple[float, float]:
    """Read last play win rates from deck_state.json or results.json."""
    state = load_deck_state(out_dir)
    if state:
        p1_wr = float(state.get("p1", {}).get("last_play_win_rate", 0.0) or 0.0)
        p2_wr = float(state.get("p2", {}).get("last_play_win_rate", 0.0) or 0.0)
        return p1_wr, p2_wr
    results = out_dir / "results.json"
    if results.is_file():
        try:
            data = json.loads(results.read_text(encoding="utf-8"))
            p1_rates = data.get("p1", {}).get("win_rates", [])
            p2_rates = data.get("p2", {}).get("win_rates", [])
            return (
                float(p1_rates[-1]) if p1_rates else 0.0,
                float(p2_rates[-1]) if p2_rates else 0.0,
            )
        except Exception:
            pass
    return 0.0, 0.0


DEFAULT_FORMAT = _DEFAULT_FORMAT
DEFAULT_HERO_ID = _DEFAULT_HERO_ID
DEFAULT_HERO_CLASS = _DEFAULT_HERO_CLASS
DEFAULT_EQUIPMENT_HEADER = _DEFAULT_EQUIPMENT_HEADER
DEFAULT_OPPONENT_DECK = _DEFAULT_OPPONENT_DECK
DEFAULT_OPPONENT_HERO = _DEFAULT_OPPONENT_HERO
OUT_DIR = _OUT_DIR
DEFAULT_AGENT_CACHE_DIR = _AGENT_CACHE_DIR
