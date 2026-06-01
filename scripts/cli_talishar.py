#!/usr/bin/env python
"""Interactive CLI for playing Flesh and Blood via the Talishar engine.

You control Player 1 against the Talishar CombatDummy AI (Player 2).
Optionally load a pickled RL agent to watch it play automatically.

Usage
-----
Human vs CombatDummy (with deck selection):
    python scripts/cli_talishar.py

Watch a saved agent play:
    python scripts/cli_talishar.py --agent path/to/agent.pkl

Options
-------
    --url URL       Talishar base URL  (default: $TALISHAR_URL or http://localhost)
    --deck NAME     Skip selection and use a pre-built deck from Assets/  (e.g. Ira)
    --format FMT    Game format: blitz, classic_constructed, ...  (default: blitz)
    --agent FILE    Pickle file containing a trained rlbridge agent
    --episodes N    Number of episodes to play  (default: 1)
    --delay SECS    Seconds between actions in agent mode  (default: 0.5)
    --max-steps N   Maximum steps per episode before truncation  (default: 120)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_FAB_DB_DIR = Path(__file__).resolve().parent.parent / "src" / "flesh_and_blood_rlbridge" / "card_db"
_FABRARY_DECKS_PATH = _FAB_DB_DIR / "fabrary_decks.json"
_CARDS_DB_PATH = _FAB_DB_DIR / "cards.json"
_TALISHAR_ID_RE = re.compile(r"^[a-z][a-z0-9_]+$")

_SEP = "─" * 62


def _phase_label(phase: str) -> str:
    return phase if phase else "Setup"


def _display_state(env: Any, obs_json: str) -> list[dict[str, Any]]:
    """Print the current board state and return the full legal-actions list."""
    obs = json.loads(obs_json)
    state = env._last_state

    print()
    print(_SEP)
    print(
        f"  Turn {obs['turnNo']:>2}  |  Phase: {_phase_label(obs['turnPhase'])}"
    )
    print(
        f"  Your HP : {obs['playerHealth']:>3}   Deck: {obs['playerDeckCount']:>2}"
        f"   Pitch pile: {obs['playerPitchCount']:>2}"
    )
    print(
        f"  Opp  HP : {obs['opponentHealth']:>3}   Opp hand: {obs['opponentHandSize']:>2}"
    )
    print(_SEP)

    # Hand
    hand = state.get("playerHand", [])
    if hand:
        print(f"  HAND  ({len(hand)} cards):")
        for card in hand:
            label = card.get("label") or card.get("cardNumber") or "?"
            playable = "►" if card.get("action", 0) else " "
            print(f"    {playable} {label}")
    else:
        print("  HAND  (empty)")

    # Prompt text (e.g. "Choose a target", "Block?")
    prompt = state.get("playerPrompt", {})
    if isinstance(prompt, dict):
        for key in ("promptText", "text", "message"):
            txt = prompt.get(key, "")
            if txt:
                print(f"\n  Prompt: {txt}")
                break

    # Popup
    popup = state.get("playerInputPopUp", {})
    if isinstance(popup, dict) and popup.get("active"):
        popup_title = popup.get("title") or popup.get("label") or "Choose:"
        print(f"\n  Popup: {popup_title}")

    # Legal actions
    legal = env._extract_legal_actions(state)
    print(f"\n  Actions ({len(legal)}):")
    for i, a in enumerate(legal):
        zone_tag = f"[{a['zone']}]"
        print(f"    {i:>3}.  {a['label']:<40} {zone_tag}")

    print(_SEP)
    return legal


def _human_pick(legal: list[dict[str, Any]]) -> str:
    """Prompt the user to pick an action; returns the index string."""
    while True:
        try:
            raw = input("  Choose action (number, 'r'=random, 'q'=quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted.")
            sys.exit(0)

        if raw.lower() == "q":
            print("Quitting.")
            sys.exit(0)
        if raw.lower() == "r":
            import random
            return str(random.randrange(len(legal)))
        try:
            idx = int(raw)
            if 0 <= idx < len(legal):
                return str(idx)
            print(f"  Please enter a number between 0 and {len(legal) - 1}.")
        except ValueError:
            print("  Invalid input.")


def _agent_pick(agent: Any, obs: str) -> str:
    """Ask the agent for an action."""
    if hasattr(agent, "act_greedy"):
        return str(agent.act_greedy(obs))
    return str(agent.act(obs))


# ---------------------------------------------------------------------------
# Deck selection helpers
# ---------------------------------------------------------------------------

def _load_fabrary_decks() -> list[dict]:
    """Return the deck list from fabrary_decks.json."""
    if not _FABRARY_DECKS_PATH.exists():
        return []
    data = json.loads(_FABRARY_DECKS_PATH.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return list(data.get("decks", []))


def _select_deck(decks: list[dict]) -> Optional[dict]:
    """Show an interactive deck menu.  Returns selected deck dict or None to skip."""
    if not decks:
        return None

    # Group by format
    groups: dict[str, list[tuple[int, dict]]] = {}
    for i, d in enumerate(decks):
        fmt = str(d.get("format", "unknown"))
        groups.setdefault(fmt, []).append((i, d))

    print()
    print(_SEP)
    print("  Deck selection  (Enter or 'q' to use the default deck)")
    print(_SEP)
    fmt_labels = {"silver_age": "Silver Age", "classic_constructed": "Classic Constructed"}
    for fmt, entries in groups.items():
        label = fmt_labels.get(fmt, fmt.replace("_", " ").title())
        print(f"\n  [{label}]")
        for idx, d in entries:
            style = d.get("style", "")
            style_tag = f"  ({style})" if style else ""
            print(f"    {idx:>3d}  {d['name']}{style_tag}")

    print()
    while True:
        try:
            raw = input(f"  Deck number [0-{len(decks)-1}] or q/Enter to skip: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw == "" or raw.lower() == "q":
            return None
        try:
            idx = int(raw)
            if 0 <= idx < len(decks):
                return decks[idx]
        except ValueError:
            pass
        print("  Invalid — enter a number from the list, or press Enter to skip.")


def _build_assets_hero_map(assets_path: str) -> dict[str, str]:
    """Scan Assets/*.txt and map hero-card-id → full header line."""
    result: dict[str, str] = {}
    p = Path(assets_path)
    if not p.is_dir():
        return result
    for txt_file in sorted(p.glob("*.txt")):
        try:
            first_line = txt_file.read_text(encoding="utf-8").splitlines()[0].strip()
            if not first_line:
                continue
            hero_id = first_line.split()[0]
            result[hero_id] = first_line
        except Exception:
            continue
    return result


def _get_hero_header(hero_id: str, assets_map: dict[str, str]) -> str:
    """Return the deck-file header line for *hero_id*.

    Tries (in order): direct match, match after stripping 'hero_' prefix.
    Falls back to the bare hero-card-id with no equipment.
    """
    base = hero_id.removeprefix("hero_")
    return assets_map.get(hero_id) or assets_map.get(base) or base


def _resolve_deck_cards(deck_entry: dict) -> list[str]:
    """Resolve fabrary card names to Talishar card IDs.

    Prefers the red (pitch-1) variant when multiple pitches exist.
    Returns a flat list with duplicates for cards with count > 1.
    """
    if not _CARDS_DB_PATH.exists():
        return []
    cards_data = json.loads(_CARDS_DB_PATH.read_text(encoding="utf-8"))

    # Build name.lower() → sorted-by-pitch card list
    name_map: dict[str, list[dict]] = {}
    for c in cards_data:
        cid = c.get("id", "")
        if not _TALISHAR_ID_RE.match(cid):
            continue
        name = c.get("name", "").lower()
        name_map.setdefault(name, []).append(c)
    for entry_list in name_map.values():
        entry_list.sort(key=lambda c: c.get("pitch") or 99)

    result: list[str] = []
    unresolved: list[str] = []
    for card_entry in deck_entry.get("cards", []):
        name = str(card_entry.get("name", "")).lower()
        count = int(card_entry.get("count", 1))
        candidates = name_map.get(name)
        if candidates:
            result.extend([candidates[0]["id"]] * count)
        else:
            unresolved.append(card_entry.get("name", name))
    if unresolved:
        print(f"  Warning: {len(unresolved)} card(s) not resolved: {', '.join(unresolved[:5])}"
              + (" ..." if len(unresolved) > 5 else ""))
    return result


def _write_temp_deck(
    deck_entry: dict,
    assets_path: str,
    assets_map: dict[str, str],
) -> Optional[Path]:
    """Resolve a fabrary deck and write it to Assets/ as a temp file.

    Returns the Path of the written file, or None on failure.
    """
    hero_id = str(deck_entry.get("hero_id", ""))
    hero_header = _get_hero_header(hero_id, assets_map)
    card_ids = _resolve_deck_cards(deck_entry)
    if not card_ids:
        print("  Error: deck resolved to zero cards; using default.")
        return None
    deck_name = f"cli_deck_{uuid.uuid4().hex[:8]}"
    content = f"{hero_header}\n{' '.join(card_ids)}\n"
    out_path = Path(assets_path) / f"{deck_name}.txt"
    out_path.write_text(content, encoding="utf-8")
    return out_path


def run_episode(
    env: Any,
    *,
    agent: Optional[Any],
    delay: float,
    episode_no: int,
) -> dict[str, Any]:
    mode = "agent" if agent is not None else "human"
    print(f"\n{'═' * 62}")
    print(f"  Episode {episode_no}  |  mode: {mode}")
    print(f"{'═' * 62}")

    result = env.reset()
    obs = result.observation

    total_reward = 0.0
    step_no = 0

    while True:
        legal = _display_state(env, obs)

        if agent is not None:
            action = _agent_pick(agent, obs)
            label = legal[int(action)]["label"] if 0 <= int(action) < len(legal) else action
            print(f"  Agent chose: [{action}] {label}")
            if delay > 0:
                time.sleep(delay)
        else:
            action = _human_pick(legal)

        step = env.step(action)
        obs = step.observation
        total_reward += step.reward
        step_no += 1

        info = step.info
        if step.terminated or step.truncated:
            # Final state display
            _display_state(env, obs)
            outcome = "TERMINATED" if step.terminated else "TRUNCATED"
            player_hp = info.get("player_hp", 0)
            opp_hp = info.get("opponent_hp", 0)
            if step.terminated:
                if opp_hp <= 0 < player_hp:
                    result_str = "YOU WIN! 🎉" if agent is None else "AGENT WINS!"
                elif player_hp <= 0 < opp_hp:
                    result_str = "You lost." if agent is None else "Agent lost."
                else:
                    result_str = "Draw."
            else:
                result_str = f"Truncated after {step_no} steps."

            print(f"\n  {outcome} — {result_str}")
            print(f"  Steps: {step_no}  |  Total reward: {total_reward:+.3f}")
            print(f"  Final HP — You: {player_hp}  Opp: {opp_hp}")
            break

    return {
        "episode": episode_no,
        "steps": step_no,
        "total_reward": total_reward,
        "player_hp": info.get("player_hp", 0),
        "opponent_hp": info.get("opponent_hp", 0),
        "terminated": step.terminated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play Flesh and Blood interactively via the Talishar engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default=None, help="Talishar base URL")
    parser.add_argument(
        "--deck",
        default=None,
        help="Skip selection and use this deck from Assets/ (e.g. Ira, FaiCC)",
    )
    parser.add_argument("--format", default="blitz", dest="game_format", help="Game format (default: blitz)")
    parser.add_argument("--agent", default=None, metavar="FILE", help="Pickled RL agent to auto-play")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes (default: 1)")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between agent actions (default: 0.5)")
    parser.add_argument("--max-steps", type=int, default=120, dest="max_steps", help="Max steps per episode (default: 120)")
    args = parser.parse_args()

    # Resolve Talishar URL and Assets path
    base_url = args.url or os.environ.get("TALISHAR_URL", "http://localhost")
    assets_path = os.environ.get(
        "TALISHAR_ASSETS_PATH",
        str(Path.home() / "Documents" / "flesh-and-blood" / "Talishar" / "Assets"),
    )

    # Load agent if requested
    loaded_agent: Optional[Any] = None
    if args.agent:
        with open(args.agent, "rb") as fh:
            loaded_agent = pickle.load(fh)  # noqa: S301
        print(f"Loaded agent from {args.agent}: {type(loaded_agent).__name__}")

    # --- Deck selection ---
    temp_deck_file: Optional[Path] = None  # written temp file to clean up
    deck_name: str = args.deck or "Ira"

    if args.deck is None:
        # Interactive deck selection from fabrary_decks.json
        fabrary_decks = _load_fabrary_decks()
        if fabrary_decks:
            selected = _select_deck(fabrary_decks)
            if selected is not None:
                assets_map = _build_assets_hero_map(assets_path)
                written = _write_temp_deck(selected, assets_path, assets_map)
                if written is not None:
                    temp_deck_file = written
                    deck_name = written.stem
                    print(f"  Using deck: {selected['name']}  →  {deck_name}.txt")
                else:
                    print("  Deck write failed; falling back to default 'Ira'.")
            else:
                print("  No deck selected; using default 'Ira'.")
        else:
            print(f"  fabrary_decks.json not found; using default 'Ira'.")

    # Import env (after sys.path is set)
    from flesh_and_blood_rlbridge import FLESH_AND_BLOOD_TALISHAR_V0  # noqa: E402

    print(f"\nConnecting to {base_url}  |  deck: {deck_name}  |  format: {args.game_format}")
    env = FLESH_AND_BLOOD_TALISHAR_V0.create(
        render_mode="ansi",
        base_url=base_url,
        local_deck_name=deck_name,
        game_format=args.game_format,
        max_turns=args.max_steps,
    )

    # Summary across episodes
    results: list[dict[str, Any]] = []
    try:
        for ep in range(1, args.episodes + 1):
            ep_result = run_episode(
                env,
                agent=loaded_agent,
                delay=args.delay,
                episode_no=ep,
            )
            results.append(ep_result)

    finally:
        env.close()
        if temp_deck_file is not None and temp_deck_file.exists():
            try:
                temp_deck_file.unlink()
            except Exception:
                pass  # best-effort cleanup

    if len(results) > 1:
        wins = sum(1 for r in results if r["opponent_hp"] <= 0 and r["player_hp"] > 0)
        mean_reward = sum(r["total_reward"] for r in results) / len(results)
        print(f"\n{'═' * 62}")
        print(f"  Summary: {len(results)} episodes  |  wins: {wins}  |  mean reward: {mean_reward:+.3f}")
        print(f"{'═' * 62}")


if __name__ == "__main__":
    main()

