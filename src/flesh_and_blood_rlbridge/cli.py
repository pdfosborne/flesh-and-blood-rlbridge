"""Interactive command-line client for playing Flesh and Blood matches.

Pick a deck and an opponent, then play the Talishar-inspired simulation turn by
turn. At every decision point a pre-trained RL agent (PPO) suggests the action
it considers best. Trained agents are cached under ``card_db/agent_cache`` and
shared with the ``fab_*`` MCP tools, so a previously trained matchup is reused
automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Optional

from .gameplay_environment import FleshAndBloodGameplayEnvironment

_CACHE_DIR = Path(__file__).with_name("card_db") / "agent_cache"

_BAR = "=" * 64


# ----------------------------------------------------------------------------
# Deck options
# ----------------------------------------------------------------------------
def _list_deck_options(fmt: str) -> list[dict[str, Any]]:
    """Return the raw (unscored) deck options for a format."""
    env = FleshAndBloodGameplayEnvironment(format=fmt, render_mode=None)
    try:
        return list(env._deck_options_for_format())  # noqa: SLF001
    finally:
        env.close()


# ----------------------------------------------------------------------------
# Agent cache (shared with the fab_* MCP tools)
# ----------------------------------------------------------------------------
def _matchup_cache_key(
    fmt: str,
    agent_type: str,
    deck: dict[str, Any],
    matchup: dict[str, Any],
) -> str:
    return "|".join(
        [
            str(fmt),
            str(agent_type),
            str(deck.get("key", "")),
            str(matchup.get("key", "")),
            str(deck.get("hero_id", "")),
            str(matchup.get("hero_id", "")),
            str(deck.get("style", "balanced")),
            str(matchup.get("style", "balanced")),
            str(int(deck.get("deck_size", 40) or 40)),
        ]
    )


def _cache_paths(key: str) -> tuple[Path, Path]:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return _CACHE_DIR / f"{digest}.pkl", _CACHE_DIR / f"{digest}.json"


def _load_cached_agent(key: str) -> Optional[Any]:
    agent_path, _ = _cache_paths(key)
    if not agent_path.exists():
        return None
    try:
        with agent_path.open("rb") as fh:
            return pickle.load(fh)  # noqa: S301
    except Exception:
        return None


def _save_cached_agent(
    key: str,
    agent: Any,
    *,
    fmt: str,
    deck: dict[str, Any],
    matchup: dict[str, Any],
    agent_type: str,
    episodes: int,
) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    agent_path, meta_path = _cache_paths(key)
    with agent_path.open("wb") as fh:
        pickle.dump(agent, fh)
    metadata = {
        "cache_key": key,
        "format_name": fmt,
        "deck_key": deck.get("key", ""),
        "matchup_key": matchup.get("key", ""),
        "inner_agent_type": agent_type,
        "total_train_episodes": int(episodes),
        "last_win_rate": 0.5,
        "converged": False,
        "env_kwargs": {
            "format": fmt,
            "agent_hero_id": deck.get("hero_id", ""),
            "opponent_hero_id": matchup.get("hero_id", ""),
            "deck_size": int(deck.get("deck_size", 40) or 40),
            "agent_deck_style": deck.get("style", "balanced"),
            "opponent_deck_style": matchup.get("style", "balanced"),
        },
        "source": "fab-play cli",
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _train_agent(
    deck: dict[str, Any],
    matchup: dict[str, Any],
    *,
    fmt: str,
    episodes: int,
    max_steps: int,
    seed: int,
    agent_type: str = "ppo",
) -> Any:
    from rlbridge.rl_agents.ppo import PPOAgent

    agent = PPOAgent(hidden_size=64, seed=seed)
    train_env = FleshAndBloodGameplayEnvironment(
        seed=seed,
        agent_hero_id=str(deck["hero_id"]),
        opponent_hero_id=str(matchup["hero_id"]),
        deck_size=int(deck.get("deck_size", 40) or 40),
        agent_deck_style=str(deck.get("style", "balanced")),
        opponent_deck_style=str(matchup.get("style", "balanced")),
        format=fmt,
        render_mode=None,
    )
    try:
        agent.train(train_env, n_episodes=episodes, max_steps=max_steps, seed=seed)
    finally:
        train_env.close()

    key = _matchup_cache_key(fmt, agent_type, deck, matchup)
    _save_cached_agent(
        key, agent, fmt=fmt, deck=deck, matchup=matchup, agent_type=agent_type, episodes=episodes
    )
    return agent


# ----------------------------------------------------------------------------
# Play environment + suggestions
# ----------------------------------------------------------------------------
def _build_play_env(
    deck: dict[str, Any],
    matchup: dict[str, Any],
    *,
    fmt: str,
    opponent_type: str,
    max_turns: int,
    seed: int,
) -> tuple[FleshAndBloodGameplayEnvironment, dict[str, Any]]:
    env = FleshAndBloodGameplayEnvironment(
        seed=seed,
        agent_hero_id=str(deck["hero_id"]),
        opponent_hero_id=str(matchup["hero_id"]),
        max_turns=max_turns,
        deck_size=int(deck.get("deck_size", 40) or 40),
        agent_deck_style=str(deck.get("style", "balanced")),
        opponent_deck_style=str(matchup.get("style", "balanced")),
        format=fmt,
        opponent_type=opponent_type,
        render_mode="ansi",
    )
    reset = env.reset(seed=seed)
    obs = reset.observation

    a_ids = deck.get("_card_ids")
    o_ids = matchup.get("_card_ids")
    if isinstance(a_ids, list) or isinstance(o_ids, list):
        env._start_match(  # noqa: SLF001
            agent_hero_id=str(deck["hero_id"]),
            opponent_hero_id=str(matchup["hero_id"]),
            agent_deck_style=str(deck.get("style", "balanced")),
            opponent_deck_style=str(matchup.get("style", "balanced")),
            agent_deck_ids=a_ids if isinstance(a_ids, list) else None,
            opponent_deck_ids=o_ids if isinstance(o_ids, list) else None,
        )
        obs = env._observation()  # noqa: SLF001
    return env, obs


def _suggest(agent: Any, obs: Any, legal: list[str]) -> tuple[Optional[str], Optional[float]]:
    """Return the action the pre-trained agent recommends and its confidence.

    Agents now choose from all legal actions in the state (via legal-action
    masking), so ``act_greedy`` returns the recommended action itself; older
    agents may still return an integer index, which we map onto ``legal``.
    """
    if agent is None or not legal:
        return None, None
    try:
        recommended = agent.act_greedy(obs)
    except Exception:
        return None, None

    if isinstance(recommended, int):
        action = legal[recommended % len(legal)]
    else:
        action = recommended if recommended in legal else None
    if action is None:
        return None, None
    idx = legal.index(action)

    confidence: Optional[float] = None
    try:
        import numpy as np

        vec = agent._obs_to_vec(obs)  # noqa: SLF001
        logits = agent._actor.predict(vec)  # noqa: SLF001
        if hasattr(agent, "_masked_logits"):
            logits = agent._masked_logits(logits, obs)  # noqa: SLF001
        shifted = logits - np.max(logits)
        probs = np.exp(shifted) / np.exp(shifted).sum()
        confidence = float(probs[idx])
    except Exception:
        confidence = None
    return action, confidence


# ----------------------------------------------------------------------------
# Interactive prompts
# ----------------------------------------------------------------------------
def _choose_deck(options: list[dict[str, Any]], role: str) -> dict[str, Any]:
    print(
        f"\nSelect the {role} deck ({len(options)} available)."
        " Type a search term to filter (or press Enter to list all)."
    )
    page_size = 30
    while True:
        query = input(f"  {role} search> ").strip().lower()
        matches = (
            [o for o in options if query in o["label"].lower() or query in str(o["key"]).lower()]
            if query
            else list(options)
        )
        if not matches:
            print("  No matches; try a different term.")
            continue
        shown = matches[:page_size]
        for i, opt in enumerate(shown):
            tag = " [fabrary]" if opt.get("_card_ids") else ""
            print(f"    [{i:>2}] {opt['label']} ({opt['deck_size']} cards){tag}")
        if len(matches) > len(shown):
            print(f"    ... {len(matches) - len(shown)} more; refine your search to narrow down.")
        pick = input(f"  Pick {role} [0-{len(shown) - 1}], or Enter to search again: ").strip()
        if pick.isdigit() and 0 <= int(pick) < len(shown):
            chosen = shown[int(pick)]
            print(f"  -> {role} deck: {chosen['label']}")
            return chosen
        print("  (No valid index entered — searching again.)")


def _find_deck(options: list[dict[str, Any]], key: str, role: str) -> dict[str, Any]:
    match = next((o for o in options if str(o.get("key")) == key), None)
    if match is None:
        raise SystemExit(f"Error: {role} deck key {key!r} not found for this format.")
    return match


# ----------------------------------------------------------------------------
# Human-readable rendering
# ----------------------------------------------------------------------------
def _hand_by_index(obs: dict[str, Any]) -> dict[int, dict[str, Any]]:
    agent = obs.get("agent") if isinstance(obs.get("agent"), dict) else {}
    hand = agent.get("hand") if isinstance(agent.get("hand"), list) else []
    return {int(c["index"]): c for c in hand if "index" in c}


def _clean_name(name: Any) -> str:
    """Strip any leading slug prefix (e.g. ``wage-gold-1---Wage Gold``)."""
    text = str(name or "Unknown")
    if "---" in text:
        text = text.rsplit("---", 1)[-1]
    return text.strip() or "Unknown"


def _card_line(card: dict[str, Any]) -> str:
    types = "/".join(t for t in card.get("card_types", []) if t) or "card"
    keywords = card.get("keywords") or []
    kw = f"  {{{', '.join(keywords)}}}" if keywords else ""
    line = (
        f"{_clean_name(card.get('name'))} "
        f"- cost {card.get('cost', 0)}, pitch {card.get('pitch', 0)}, "
        f"attack {card.get('power', 0)}, defense {card.get('defense', 0)} ({types}){kw}"
    )
    text = _clean_text(card.get("text"))
    if text:
        wrapped = text.replace("\n", "\n          ")
        line += f"\n          {wrapped}"
    return line


def _clean_text(text: Any) -> str:
    """Strip FAB markup tokens so card text reads cleanly in the terminal."""
    raw = str(text or "")
    if not raw.strip():
        return ""
    raw = raw.replace("{br}", "\n").replace("**", "")
    for tag in ("{i}", "{/i}"):
        raw = raw.replace(tag, "")
    for token, word in (("{p}", " power"), ("{r}", " resource"), ("{h}", " life"), ("{d}", " defense")):
        raw = raw.replace(token, word)
    lines = [" ".join(line.split()) for line in raw.split("\n")]
    return "\n".join(line for line in lines if line)


def _describe_action(action: str, obs: dict[str, Any]) -> str:
    """Turn a raw legal action string into a natural-language description."""
    phase = obs.get("phase")
    if action == "pass":
        if phase == "defense":
            return "Pass - take the attack unblocked (no more blocks)"
        if phase == "reaction":
            return "Pass - resolve the attack (play no reactions)"
        return "Pass - end your turn (let the opponent act)"

    parts = action.split()
    if len(parts) == 2 and parts[0] in {"play", "block", "pitch", "reaction"} and parts[1].isdigit():
        card = _hand_by_index(obs).get(int(parts[1]))
        if card is None:
            return action
        name = _clean_name(card.get("name"))
        if parts[0] == "play":
            return f'Play "{name}" - attack for {card.get("power", 0)} (cost {card.get("cost", 0)})'
        if parts[0] == "pitch":
            return f'Pitch "{name}" - gain {card.get("pitch", 0)} resources'
        if parts[0] == "reaction":
            return f'React with "{name}" - +{card.get("power", 0)} attack (cost {card.get("cost", 0)})'
        return f'Block with "{name}" - prevents {card.get("defense", 0)} damage'
    return action


def _render_board(obs: dict[str, Any], win_agent: float, win_opp: float) -> str:
    agent = obs.get("agent") if isinstance(obs.get("agent"), dict) else {}
    opp = obs.get("opponent") if isinstance(obs.get("opponent"), dict) else {}
    lines: list[str] = [
        _BAR,
        f"Turn {obs.get('turn')}  |  {obs.get('format')}  |  Phase: {obs.get('phase')}",
        f"Win chance - You: {win_agent:.0%}   Opponent: {win_opp:.0%}",
    ]
    if obs.get("last_event"):
        lines.append(f"Last: {obs['last_event']}")

    lines.append("-" * 64)
    lines.append(f"OPPONENT - {opp.get('hero', '?')}")
    lines.append(
        f"  Life {opp.get('life', 0)}  |  Hand {opp.get('hand_size', 0)} (hidden)  |  "
        f"Deck {opp.get('deck', 0)}  |  Discard {opp.get('discard', 0)}  |  "
        f"Resources {opp.get('resources', 0)}  |  AP {opp.get('action_points', 0)}"
    )

    pending = obs.get("pending_combat")
    if isinstance(pending, dict):
        attacker = "Opponent" if int(pending.get("attacker", 0) or 0) == 1 else "You"
        flags = []
        if pending.get("go_again"):
            flags.append("go again")
        if pending.get("dominate"):
            flags.append("dominate")
        tag = f" [{', '.join(flags)}]" if flags else ""
        lines.append("-" * 64)
        lines.append(
            f"  [COMBAT] {attacker} attacking with {_clean_name(pending.get('attack_card', '?'))}{tag} "
            f"- power {pending.get('attack_power', 0)}, blocked so far {pending.get('total_block', 0)}"
        )

    lines.append("-" * 64)
    lines.append(f"YOU - {agent.get('hero', '?')}")
    lines.append(
        f"  Life {agent.get('life', 0)}  |  Deck {agent.get('deck', 0)}  |  "
        f"Discard {agent.get('discard', 0)}  |  Resources {agent.get('resources', 0)}  |  "
        f"AP {agent.get('action_points', 0)}"
    )
    lines.append("  Your hand:")
    hand = agent.get("hand") if isinstance(agent.get("hand"), list) else []
    if not hand:
        lines.append("    (empty)")
    for card in hand:
        lines.append(f"    [{card.get('index')}] {_card_line(card)}")
    lines.append(_BAR)
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Game loop
# ----------------------------------------------------------------------------
def _win_probabilities(env: Any, obs: dict[str, Any]) -> tuple[float, float]:
    try:
        return env._estimate_win_probabilities(obs)  # noqa: SLF001
    except Exception:
        return 0.5, 0.5


def _play(env: FleshAndBloodGameplayEnvironment, obs: dict[str, Any], agent: Any) -> None:
    print("\n" + _render_board(obs, *_win_probabilities(env, obs)))

    while True:
        legal = list(obs.get("legal_actions") or [])
        if not legal:
            break

        if legal == ["pass"]:
            print("\n(Only 'pass' is available - auto-passing.)")
            out = env.step("pass")
            obs = out.observation
            print("\n" + _render_board(obs, *_win_probabilities(env, obs)))
            if out.terminated or out.truncated:
                break
            continue

        suggestion, confidence = _suggest(agent, obs, legal)

        print("\nYour move:")
        for i, action in enumerate(legal):
            mark = "  <-- suggested" if (suggestion is not None and action == suggestion) else ""
            print(f"  [{i}] {_describe_action(action, obs)}{mark}")
        if suggestion is not None:
            conf = f" (confidence {confidence:.0%})" if confidence is not None else ""
            print(f"Pre-trained agent suggests: {_describe_action(suggestion, obs)}{conf}")
        else:
            print("(No trained agent loaded - playing without suggestions.)")

        raw = input("Choose [number, Enter = suggested, q = quit]: ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            print("Quitting match.")
            return
        if raw == "":
            if suggestion is None:
                print("No suggestion available; please enter a choice.")
                continue
            action: Any = suggestion
        elif raw.isdigit() and 0 <= int(raw) < len(legal):
            action = legal[int(raw)]
        else:
            action = raw

        out = env.step(action)
        obs = out.observation
        print("\n" + _render_board(obs, *_win_probabilities(env, obs)))
        if out.info.get("error"):
            print(f"(!) {out.info['error']}")
        if out.terminated or out.truncated:
            break

    _announce_result(obs)


def _announce_result(obs: dict[str, Any]) -> None:
    print("\n" + _BAR)
    agent_info = obs.get("agent") if isinstance(obs.get("agent"), dict) else {}
    opp_info = obs.get("opponent") if isinstance(obs.get("opponent"), dict) else {}
    agent_life = float(agent_info.get("life", 0))
    opp_life = float(opp_info.get("life", 0))
    if opp_life <= 0 < agent_life:
        print("RESULT: You win!")
    elif agent_life <= 0 < opp_life:
        print("RESULT: You lose.")
    else:
        print(f"RESULT: Match ended (your life {agent_life:g}, opponent life {opp_life:g}).")
    print(_BAR)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fab-play",
        description="Play a Flesh and Blood match with pre-trained agent suggestions.",
    )
    parser.add_argument(
        "--format",
        default="silver_age",
        help="Game format (silver_age or classic_constructed). Default: silver_age.",
    )
    parser.add_argument("--deck", help="Deck key to play (skips the interactive picker).")
    parser.add_argument("--opponent", help="Opponent deck key (skips the interactive picker).")
    parser.add_argument(
        "--opponent-type",
        default="preset_logic",
        choices=["preset_logic", "random"],
        help="Scripted opponent policy. Default: preset_logic.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed. Default: 0.")
    parser.add_argument("--max-turns", type=int, default=60, help="Max turns. Default: 60.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=40,
        help="Episodes to train a fresh agent if none is cached. Default: 40.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=200,
        help="Max steps per training episode. Default: 200.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force training a new agent even if a cached one exists.",
    )
    parser.add_argument(
        "--no-suggest",
        action="store_true",
        help="Disable agent suggestions (no loading or training).",
    )
    args = parser.parse_args(argv)

    print(_BAR)
    print("  Flesh and Blood — interactive match (Talishar-inspired)")
    print(_BAR)

    try:
        options = _list_deck_options(args.format)
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading deck options for format {args.format!r}: {exc}")
        return 1
    if not options:
        print(f"No deck options found for format {args.format!r}.")
        return 1

    deck = _find_deck(options, args.deck, "your") if args.deck else _choose_deck(options, "your")
    opponent = (
        _find_deck(options, args.opponent, "opponent")
        if args.opponent
        else _choose_deck(options, "opponent")
    )

    agent: Any = None
    if not args.no_suggest:
        key = _matchup_cache_key(args.format, "ppo", deck, opponent)
        agent = None if args.retrain else _load_cached_agent(key)
        if agent is not None:
            print("\nLoaded a pre-trained agent for this matchup from the cache.")
        else:
            prompt = (
                f"\nNo cached agent for this matchup. Train one now "
                f"({args.episodes} episodes)? [Y/n]: "
            )
            answer = input(prompt).strip().lower()
            if answer in {"", "y", "yes"}:
                print("Training agent — this may take a moment...")
                try:
                    agent = _train_agent(
                        deck,
                        opponent,
                        fmt=args.format,
                        episodes=args.episodes,
                        max_steps=args.train_steps,
                        seed=args.seed,
                    )
                    print("Training complete; agent cached for next time.")
                except Exception as exc:  # noqa: BLE001
                    print(f"Training failed ({exc}); continuing without suggestions.")
                    agent = None
            else:
                print("Continuing without agent suggestions.")

    print(f"\nMatchup: {deck['label']}  vs  {opponent['label']}")
    env, obs = _build_play_env(
        deck,
        opponent,
        fmt=args.format,
        opponent_type=args.opponent_type,
        max_turns=args.max_turns,
        seed=args.seed,
    )
    try:
        _play(env, obs, agent)
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted; exiting.")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
