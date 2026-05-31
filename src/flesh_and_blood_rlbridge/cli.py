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

from . import effects
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


def _pitch_label(pitch: Any) -> str:
    """FAB pitch color: blue=3, yellow=2, red=1."""
    match int(pitch or 0):
        case 3:
            return "(B)"
        case 2:
            return "(Y)"
        case 1:
            return "(R)"
        case _:
            return ""


def _card_display_name(card: Optional[dict[str, Any]], *, fallback: str = "Unknown") -> str:
    """Card name with pitch-color label when pitch is 1/2/3."""
    if not isinstance(card, dict):
        return fallback
    name = _clean_name(card.get("name"))
    label = _pitch_label(card.get("pitch"))
    return f"{name} {label}" if label else name


def _card_line(card: dict[str, Any]) -> str:
    types = "/".join(t for t in card.get("card_types", []) if t) or "card"
    keywords = card.get("keywords") or []
    kw = f"  {{{', '.join(keywords)}}}" if keywords else ""
    line = (
        f"{_card_display_name(card)} "
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


def _effect_hints(text: Any) -> list[str]:
    """Short summaries of the implementable triggered effects on a card."""
    hints: list[str] = []
    for trig in effects.parse_triggers(str(text or "")):
        e = trig.effect
        if not e.implemented:
            continue
        if e.kind == "next_attack_power":
            cap = f" (cost<={e.max_cost})" if 0 <= e.max_cost < 99 else ""
            hints.append(f"next attack +{e.amount} power{cap}")
        elif e.kind == "banish_combo":
            rider = f"+{e.amount} power" + (" & go again" if e.go_again else "")
            hints.append(f"may banish {e.banish_name}: {rider}")
        elif e.kind in ("damage", "arcane_damage"):
            kind = "arcane " if e.kind == "arcane_damage" else ""
            hints.append(f"{trig.when.replace('_', ' ')}: deal {e.amount} {kind}dmg")
        elif e.kind == "power":
            hints.append(f"{trig.when.replace('_', ' ')}: +{e.amount} power")
        elif e.kind == "draw":
            hints.append(f"{trig.when.replace('_', ' ')}: draw {e.amount}")
        elif e.kind == "go_again":
            hints.append(f"{trig.when.replace('_', ' ')}: go again")
        elif e.kind == "next_action_go_again":
            hints.append("next action: go again")
        elif e.kind in ("dominate", "intimidate"):
            hints.append(f"{trig.when.replace('_', ' ')}: {e.kind}")
        elif e.kind == "create_token":
            hints.append(f"{trig.when.replace('_', ' ')}: create {e.token_name} token")
        elif e.kind == "create_banished":
            hints.append(f"{trig.when.replace('_', ' ')}: create {e.banish_name} in banish")
        elif e.kind in ("banish_top", "banish_defending"):
            hints.append(f"{trig.when.replace('_', ' ')}: banish top of deck")
    for ab in effects.parse_activated_abilities(str(text or "")):
        e = ab.effect
        if not e.implemented:
            continue
        if e.kind == "next_attack_power":
            hints.append(f"ability: next attack +{e.amount} power")
        elif e.kind == "draw":
            hints.append("ability: draw")
        elif e.kind == "go_again":
            hints.append("ability: gain action point")
    for mod in effects.parse_play_modifiers(str(text or "")):
        if mod.go_again and mod.condition:
            hints.append(f"if {mod.condition}: go again")
        if mod.next_action_go_again:
            hints.append(f"next {mod.next_action_go_again} action(s): go again")
        if mod.next_attack_power:
            hints.append(f"next attack +{mod.next_attack_power} power")
    return hints


def _describe_action(action: str, obs: dict[str, Any]) -> str:
    """Turn a raw legal action string into a natural-language description."""
    phase = obs.get("phase")
    decision = obs.get("optional_decision") if isinstance(obs.get("optional_decision"), dict) else None
    if action == "use_optional":
        if decision:
            return f'Use optional - {decision.get("description")}'
        return "Use optional effect"
    if action.startswith("choose "):
        choice = obs.get("choice_decision") if isinstance(obs.get("choice_decision"), dict) else None
        if choice:
            idx = int(action.split()[1])
            for opt in choice.get("options") or []:
                if opt.get("index") == idx:
                    return f'Choose "{opt.get("name")}" - {choice.get("prompt", "card choice")}'
        return action
    if action == "pass":
        if phase == "choice":
            return "Decline - skip this card choice"
        if phase == "optional":
            return "Decline - skip this optional effect"
        if phase == "defense":
            return "Pass - take the attack unblocked (no more blocks)"
        if phase == "reaction":
            return "Pass - resolve the attack (play no reactions)"
        if phase == "arsenal":
            return "Skip - don't stash (proceed to draw)"
        return "Pass - end attacks (arsenal stash step next)"

    arena = obs.get("agent", {}).get("arena") if isinstance(obs.get("agent"), dict) else None
    arena = arena if isinstance(arena, dict) else {}

    if action == "weapon":
        w = arena.get("weapon", {})
        return f'Attack with {_clean_name(w.get("name"))} - {w.get("attack", 0)} damage (cost {w.get("cost", 0)})'

    parts = action.split()
    if len(parts) == 2 and parts[1].isdigit():
        idx = int(parts[1])
        if parts[0] == "ability":
            abilities = arena.get("abilities") or []
            entry = next((a for a in abilities if a.get("index") == idx), None)
            if entry is not None:
                return f'Activate {_clean_name(entry.get("source"))} - {_clean_text(entry.get("summary"))}'
            return action
        if parts[0] == "stash":
            card = _hand_by_index(obs).get(idx)
            name = _card_display_name(card, fallback=f"card {idx}") if card else f"card {idx}"
            return f'Stash "{name}" in arsenal - keep it for next turn'
        if parts[0] == "blockgear":
            equipment = arena.get("equipment") or []
            piece = equipment[idx] if 0 <= idx < len(equipment) else None
            if piece is not None:
                fragile = " (destroyed after blocking)" if {"battleworn", "blade_break"} & set(piece.get("keywords") or []) else ""
                return f'Block with {_clean_name(piece.get("name"))} [{piece.get("slot")}] - prevents {piece.get("defense", 0)} damage{fragile}'
            return action

    if len(parts) == 2 and parts[0] in {"play", "banishplay", "block", "pitch", "reaction"} and parts[1].isdigit():
        if parts[0] == "banishplay":
            agent = obs.get("agent") if isinstance(obs.get("agent"), dict) else {}
            banished = agent.get("banished") if isinstance(agent.get("banished"), list) else []
            card = next((c for c in banished if c.get("index") == int(parts[1])), None)
            if card is None:
                return action
            name = _clean_name(card.get("name"))
            return f'Play "{name}" from banished zone - attack action (cost 0)'
        card = _hand_by_index(obs).get(int(parts[1]))
        if card is None:
            return action
        name = _card_display_name(card)
        if parts[0] == "play":
            cost = card.get("cost", 0)
            damage = card.get("damage", card.get("power", 0))
            is_attack = "attack_action" in (card.get("card_types") or [])
            if damage and card.get("arcane"):
                effect = f"deal {damage} arcane damage"
            elif damage:
                effect = f"attack for {damage}" if is_attack else f"deal {damage} damage"
            else:
                effect = "play action"
            extras = []
            if card.get("fusion_element"):
                state = "ready" if card.get("fusable") else f"needs a {card['fusion_element']} card"
                extras.append(f"fusion: {state}")
            if "go_again" in (card.get("keywords") or []):
                extras.append("go again")
            extras.extend(_effect_hints(card.get("text")))
            tag = f" [{', '.join(extras)}]" if extras else ""
            return f'Play "{name}" - {effect} (cost {cost}){tag}'
        if parts[0] == "pitch":
            return f'Pitch "{name}" - gain {card.get("pitch", 0)} resources'
        if parts[0] == "reaction":
            return f'React with "{name}" - +{card.get("power", 0)} attack (cost {card.get("cost", 0)})'
        return f'Block with "{name}" - prevents {card.get("defense", 0)} damage'
    return action


def _arena_lines(arena: dict[str, Any], reveal: bool) -> list[str]:
    """Render the equipment / weapon / hero / arsenal zones."""
    if not isinstance(arena, dict):
        return []
    out: list[str] = []
    equipment = arena.get("equipment") or []
    if equipment:
        pieces = []
        for e in equipment:
            mark = "*" if e.get("used") else ""
            pieces.append(f"{e.get('slot')}: {_clean_name(e.get('name'))} (def {e.get('defense', 0)}){mark}")
        out.append("  Equipment: " + "  |  ".join(pieces))
    weapon = arena.get("weapon") or {}
    if weapon.get("name"):
        used = " [used]" if weapon.get("used") else ""
        attack = weapon.get("attack", 0)
        no_swing = " (no weapon attack — play attack actions from hand)" if attack <= 0 else ""
        out.append(
            f"  Weapon: {_clean_name(weapon.get('name'))} - attack {attack}, "
            f"cost {weapon.get('cost', 0)}{used}{no_swing}"
        )
    if arena.get("hero_ability_text"):
        ability = _clean_text(arena.get("hero_ability_text"))
        first = ability.split("\n")[0]
        flag = " [activatable]" if arena.get("hero_ability_usable") else ""
        out.append(f"  Hero ability: {first}{flag}")
    if reveal:
        for ab in arena.get("abilities") or []:
            cost = int(ab.get("cost") or 0)
            cost_note = f", costs {cost} resources" if cost else ""
            flag = " [ready]" if ab.get("usable") else (f" [pitch for {cost} resources first]" if cost else " [unavailable]")
            out.append(
                f"    > ability {ab.get('index')}: {_clean_name(ab.get('source'))} - "
                f"{_clean_text(ab.get('summary'))}{cost_note}{flag}"
            )
        arsenal = arena.get("arsenal") or []
        out.append(f"  Arsenal: {', '.join(_clean_name(n) for n in arsenal) if arsenal else '(empty)'}")
    else:
        out.append(f"  Arsenal: {arena.get('arsenal_size', 0)} card(s) (hidden)")
    return out


def _format_block_line(block: dict[str, Any]) -> str:
    name = _clean_name(block.get("name"))
    defense = int(block.get("defense") or 0)
    if block.get("kind") == "equipment":
        slot = block.get("slot") or "gear"
        return f"{name} [{slot}] (def {defense})"
    return f"{name} (def {defense})"


def _token_display_name(key: Any) -> str:
    text = " ".join(str(key or "").strip().lower().split())
    return text.title() if text else "Token"


def _format_player_tokens(player: dict[str, Any]) -> str:
    parts: list[str] = []
    tokens = player.get("tokens") if isinstance(player.get("tokens"), dict) else {}
    gold = int(player.get("gold") or 0)
    for key, count in sorted(tokens.items()):
        amount = int(count or 0)
        if amount <= 0:
            continue
        if str(key).strip().lower() == "gold":
            continue
        parts.append(f"{_token_display_name(key)} x{amount}")
    token_gold = int(tokens.get("gold") or 0) if isinstance(tokens.get("gold"), (int, float)) else 0
    total_gold = max(gold, token_gold)
    if total_gold > 0:
        parts.append(f"Gold x{total_gold}")
    return ", ".join(parts) if parts else "—"


def _board_token_lines(agent: dict[str, Any], opp: dict[str, Any]) -> list[str]:
    agent_text = _format_player_tokens(agent)
    opp_text = _format_player_tokens(opp)
    if agent_text == "—" and opp_text == "—":
        return []
    return [
        "-" * 64,
        "  [TOKENS ON BOARD]",
        f"  Opponent: {opp_text}",
        f"  You: {agent_text}",
    ]


def _last_attack_lines(last_combat: dict[str, Any]) -> list[str]:
    if int(last_combat.get("attacker", -1)) != 0:
        return []
    attack = _clean_name(last_combat.get("attack_card"))
    power = int(last_combat.get("attack_power") or 0)
    block = int(last_combat.get("total_block") or 0)
    damage = int(last_combat.get("damage") or 0)
    outcome = f"{damage} damage dealt" if damage > 0 else "no damage dealt"
    lines = [
        f"  [YOUR LAST ATTACK] {attack} — {power} power vs {block} block → {outcome}",
    ]
    blocks = last_combat.get("blocks") if isinstance(last_combat.get("blocks"), list) else []
    opp_blocks = [b for b in blocks if isinstance(b, dict) and int(b.get("player", -1)) == 1]
    if opp_blocks:
        parts = ", ".join(_format_block_line(b) for b in opp_blocks)
        lines.append(f"  Opponent blocked with: {parts}")
    else:
        lines.append("  Opponent blocked with: (none)")
    return lines


def _format_opponent_pitch_list(turn: dict[str, Any]) -> str:
    pitches = turn.get("pitches") if isinstance(turn.get("pitches"), list) else []
    parts: list[str] = []
    for pitch in pitches:
        if not isinstance(pitch, dict):
            continue
        name = _card_display_name(pitch)
        value = int(pitch.get("pitch") or 0)
        parts.append(f"{name} (+{value})" if value > 0 else name)
    return ", ".join(parts)


def _format_opponent_attack_action(turn: dict[str, Any]) -> str:
    attack = turn.get("attack_card")
    if not attack:
        return ""
    power = int(turn.get("attack_power") or 0)
    label = _clean_name(attack)
    if turn.get("is_weapon"):
        return f"attacks with weapon {label} ({power} power)"
    return f"attacks with {label} ({power} power)"


def _format_opponent_turn_summary(turn: dict[str, Any]) -> str:
    pitch_text = _format_opponent_pitch_list(turn)
    attack_text = _format_opponent_attack_action(turn)
    if pitch_text and attack_text:
        return f"pitched {pitch_text} → {attack_text}"
    if attack_text:
        return attack_text
    if pitch_text:
        return f"pitched {pitch_text}"
    return ""


def _last_opponent_turn_lines(turn: dict[str, Any]) -> list[str]:
    summary = _format_opponent_turn_summary(turn)
    if not summary:
        return []
    return [f"  [OPPONENT'S LAST TURN] {summary}"]


def _render_board(obs: dict[str, Any], win_agent: float, win_opp: float) -> str:
    agent = obs.get("agent") if isinstance(obs.get("agent"), dict) else {}
    opp = obs.get("opponent") if isinstance(obs.get("opponent"), dict) else {}
    lines: list[str] = [
        _BAR,
        f"Turn {obs.get('turn')}  |  {obs.get('format')}  |  {obs.get('flow_segment') or obs.get('phase')}",
        f"Win chance - You: {win_agent:.0%}   Opponent: {win_opp:.0%}",
    ]
    chain_links = int(obs.get("combat_chain_links") or 0)
    if chain_links > 0 and obs.get("phase") == "action":
        lines.append(f"  Combat chain: {chain_links} link(s) resolved this attack phase")
    if obs.get("last_event"):
        lines.append(f"Last: {obs['last_event']}")
    last_combat = obs.get("last_combat")
    if isinstance(last_combat, dict):
        lines.extend(_last_attack_lines(last_combat))
    last_opp_turn = obs.get("last_opponent_turn")
    pending = obs.get("pending_combat")
    opp_attacking = isinstance(pending, dict) and int(pending.get("attacker", 0) or 0) == 1
    if isinstance(last_opp_turn, dict) and not opp_attacking:
        lines.extend(_last_opponent_turn_lines(last_opp_turn))
    for notice in obs.get("notices") or []:
        lines.append(f"  (!) {notice}")
    decision = obs.get("optional_decision")
    if isinstance(decision, dict):
        lines.append(f"  (?) Optional: {decision.get('card')} - may {decision.get('description')}")
    choice = obs.get("choice_decision")
    if isinstance(choice, dict):
        opts = ", ".join(
            f"{o.get('index')}: {_clean_name(o.get('name'))}" for o in (choice.get("options") or [])
        )
        lines.append(f"  (?) Choice: {choice.get('prompt')} [{opts}]")
    if obs.get("phase") == "arsenal":
        lines.append("  (i) Attack phase over — stash one card face-down in arsenal, or skip to draw.")

    lines.append("-" * 64)
    lines.append(f"OPPONENT - {opp.get('hero', '?')}")
    lines.append(
        f"  Life {opp.get('life', 0)}  |  Hand {opp.get('hand_size', 0)} (hidden)  |  "
        f"Deck {opp.get('deck', 0)}  |  Discard {opp.get('discard', 0)}  |  "
        f"Resources {opp.get('resources', 0)}  |  AP {opp.get('action_points', 0)}"
    )
    lines.extend(_arena_lines(opp.get("arena") or {}, reveal=False))
    lines.extend(_board_token_lines(agent, opp))

    if isinstance(pending, dict):
        attacker = "Opponent" if int(pending.get("attacker", 0) or 0) == 1 else "You"
        flags = []
        if pending.get("go_again"):
            flags.append("go again")
        if pending.get("dominate"):
            flags.append("dominate")
        link = pending.get("chain_link")
        if link and int(link) > 1:
            flags.append(f"chain link {link}")
        tag = f" [{', '.join(flags)}]" if flags else ""
        attack_name = _clean_name(pending.get("attack_card", "?"))
        power = pending.get("attack_power", 0)
        blocked = pending.get("total_block", 0)
        lines.append("-" * 64)
        if attacker == "Opponent" and isinstance(last_opp_turn, dict):
            turn_summary = _format_opponent_turn_summary(last_opp_turn)
            if turn_summary:
                lines.append(
                    f"  [COMBAT] Opponent {turn_summary}{tag} "
                    f"— blocked so far {blocked}"
                )
            else:
                lines.append(
                    f"  [COMBAT] Opponent attacking with {attack_name}{tag} "
                    f"- power {power}, blocked so far {blocked}"
                )
        else:
            lines.append(
                f"  [COMBAT] {attacker} attacking with {attack_name}{tag} "
                f"- power {power}, blocked so far {blocked}"
            )
        blocks = pending.get("blocks") if isinstance(pending.get("blocks"), list) else []
        if blocks:
            opp_blocks = [
                b for b in blocks if isinstance(b, dict) and int(b.get("player", -1)) == 1
            ]
            your_blocks = [
                b for b in blocks if isinstance(b, dict) and int(b.get("player", -1)) == 0
            ]
            if opp_blocks:
                parts = ", ".join(_format_block_line(b) for b in opp_blocks)
                lines.append(f"  Opponent blocks with: {parts}")
            if your_blocks:
                parts = ", ".join(_format_block_line(b) for b in your_blocks)
                lines.append(f"  You block with: {parts}")

    lines.append("-" * 64)
    lines.append(f"YOU - {agent.get('hero', '?')}")
    lines.append(
        f"  Life {agent.get('life', 0)}  |  Deck {agent.get('deck', 0)}  |  "
        f"Discard {agent.get('discard', 0)}  |  Resources {agent.get('resources', 0)}  |  "
        f"AP {agent.get('action_points', 0)}"
    )
    banished = agent.get("banished") if isinstance(agent.get("banished"), list) else []
    if banished:
        names = [_clean_name(c.get("name")) for c in banished if isinstance(c, dict)]
        lines.append(f"  Banished: {', '.join(names) if names else '(empty)'}")
    lines.extend(_arena_lines(agent.get("arena") or {}, reveal=True))
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


def _action_phase_hints(obs: dict[str, Any]) -> list[str]:
    """Explain why attack / play options may be missing during the action phase."""
    if obs.get("phase") != "action":
        return []
    legal = list(obs.get("legal_actions") or [])
    if any(a == "weapon" or a.startswith("play ") for a in legal):
        return []

    agent = obs.get("agent") if isinstance(obs.get("agent"), dict) else {}
    hand = agent.get("hand") if isinstance(agent.get("hand"), list) else []
    resources = int(agent.get("resources") or 0)
    pitchable = sum(int(c.get("pitch") or 0) for c in hand if int(c.get("pitch") or 0) > 0)

    attacks = [
        c for c in hand
        if isinstance(c, dict) and "attack_action" in (c.get("card_types") or [])
    ]
    hints: list[str] = []
    if not attacks:
        hints.append(
            "No attack action cards in hand — defense reactions can't be played on your turn. "
            "Pitch to dig for attacks, activate your weapon ability, or pass."
        )
    elif all(int(c.get("cost") or 0) > resources for c in attacks):
        hints.append(
            f"You have attack actions but need resources (have {resources}, "
            f"can pitch for up to {pitchable} more this turn)."
        )

    arena = agent.get("arena") if isinstance(agent.get("arena"), dict) else {}
    for ab in arena.get("abilities") or []:
        if ab.get("usable"):
            continue
        cost = int(ab.get("cost") or 0)
        if cost > resources and cost <= resources + pitchable:
            hints.append(
                f"Ability \"{_clean_name(ab.get('source'))}\" needs {cost} resources — pitch first, then activate."
            )
    return hints


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

        for hint in _action_phase_hints(obs):
            print(f"  (i) {hint}")

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
