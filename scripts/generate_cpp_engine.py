#!/usr/bin/env python3
"""Generate a self-contained C++ FAB engine for a specific deck matchup.

Steps performed:
  1. Create a temp Talishar game to discover card IDs for both decks
     (or supply --no-server to skip this and rely on PHP scan alone).
  2. Scan the Talishar PHP source files to extract metadata + logic snippets
     for each discovered card ID.
  3. Emit C++ source files:
       gamestate.h / gamestate.cpp   — core game-state structs & phase logic
       cards.h                        — one inline stub per card (PHP logic as
                                        comments above each C++ function body)
       register_cards.cpp             — wires stubs into GameState::register_all_cards()
       bindings.cpp                   — pybind11 Python bindings
       CMakeLists.txt                 — CMake build config
    build.sh / build.ps1          — convenience build helpers
       card_manifest.json             — discovered metadata for reference

Build the output with:
    cd <out_dir>
    pip install pybind11
    cmake -B build .
    cmake --build build --config Release
    # copy fab_engine*.pyd / fab_engine*.so to <out_dir>

Usage:
    python scripts/generate_cpp_engine.py \\
        --talishar-src Talishar \\
        --deck1 Ira --deck2 Ira \\
        --out results/cpp_engines/Ira_vs_Ira \\
        [--base-url http://localhost]

    # Without a running server
    python scripts/generate_cpp_engine.py \\
        --talishar-src Talishar \\
        --deck1 BriarSAGEPrecon --deck2 DorintheSAGEPrecon \\
        --no-server \\
        --out results/cpp_engines/BriarSAGEPrecon_vs_DorintheSAGEPrecon
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

# ── Card metadata ──────────────────────────────────────────────────────────────

@dataclass
class CardMeta:
    card_id: str                      # e.g. "WTR010"
    name: str = ""
    cost: int = 0
    pitch: int = 0
    power: int = 0
    defense: int = 0
    card_type: str = ""               # "attack", "reaction", "equipment", ...
    php_source_file: str = ""         # relative path of PHP file
    php_snippet: str = ""             # extracted PHP logic block


@dataclass
class DeckAssetInfo:
    hero_id: str = ""
    equipment_ids: list[str] = field(default_factory=list)
    deck_counts: dict[str, int] = field(default_factory=dict)


# ── Talishar API helpers ───────────────────────────────────────────────────────

def _post(session: requests.Session, base: str, path: str, payload: dict) -> dict:
    resp = session.post(base + path, json=payload, timeout=20)
    text = resp.text
    idx = text.find("{")
    if idx > 0:
        text = text[idx:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _get(session: requests.Session, base: str, path: str, params: dict) -> dict:
    resp = session.get(base + path, params=params, timeout=20)
    text = resp.text
    idx = text.find("{")
    if idx > 0:
        text = text[idx:]
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def create_game(
    session: requests.Session, base: str, deck1: str, deck2: str
) -> tuple[str, str, str]:
    payload = {
        "deckName": deck1,
        "opponentDeckName": deck2,
        "format": "silver_age",
        "visibility": "private",
        "selfPlay": "1",
    }
    resp = _post(session, base, "/APIs/CreateLocalGame.php", payload)
    game_name = str(resp.get("gameName", ""))
    p1_auth = str(resp.get("authKey", ""))
    p2_auth = str(resp.get("p2AuthKey", ""))
    if not game_name:
        raise RuntimeError(f"CreateLocalGame failed: {resp}")
    return game_name, p1_auth, p2_auth


def fetch_full_state(
    session: requests.Session,
    base: str,
    game_name: str,
    player_id: int,
    auth_key: str,
) -> dict:
    return _get(session, base, "/GetNextTurn.php", {
        "gameName": game_name,
        "playerID": str(player_id),
        "authKey": auth_key,
        "lastUpdate": "0",
    })


def collect_card_ids_from_state(state: dict) -> set[str]:
    """Pull every cardNumber that appears anywhere in a Talishar game-state dict."""
    ids: set[str] = set()
    zone_keys = [
        "playerHand", "playerDeck", "playerDiscard", "playerEquipment",
        "playerArse", "playerAuras", "playerAllies", "playerItems",
        "playerPermanents", "playerBanish",
        "opponentHand", "opponentDeck", "opponentDiscard",
    ]
    for zk in zone_keys:
        for card in state.get(zk, []):
            if isinstance(card, dict):
                cid = str(card.get("cardNumber", "")).strip()
                if cid:
                    ids.add(cid)
    for key in ("playerHero", "opponentHero"):
        hero = state.get(key, {})
        if isinstance(hero, dict):
            cid = str(hero.get("cardNumber", "")).strip()
            if cid:
                ids.add(cid)
    return ids


def discover_deck_cards(
    base_url: str, deck1: str, deck2: str, passes: int = 5
) -> tuple[set[str], set[str]]:
    """Start a temp game and fetch state for both players to discover card IDs."""
    session = requests.Session()
    print(f"  Creating temp game ({deck1} vs {deck2})…")
    game_name, p1_auth, p2_auth = create_game(session, base_url, deck1, deck2)
    _get(session, base_url, "/Start.php", {"gameName": game_name, "playerID": "1"})
    time.sleep(1.0)

    p1_ids: set[str] = set()
    p2_ids: set[str] = set()
    for _ in range(passes):
        s1 = fetch_full_state(session, base_url, game_name, 1, p1_auth)
        s2 = fetch_full_state(session, base_url, game_name, 2, p2_auth)
        p1_ids |= collect_card_ids_from_state(s1)
        p2_ids |= collect_card_ids_from_state(s2)
        time.sleep(0.25)

    session.close()
    print(f"  P1 card IDs ({len(p1_ids)}): {sorted(p1_ids)[:8]} {'…' if len(p1_ids) > 8 else ''}")
    print(f"  P2 card IDs ({len(p2_ids)}): {sorted(p2_ids)[:8]} {'…' if len(p2_ids) > 8 else ''}")
    return p1_ids, p2_ids


def resolve_deck_from_json(talishar_src: Path, deck_json_path: str) -> set[str]:
    """Parse a FaBrary/FABdb deck JSON and return the set of card IDs."""
    return set(resolve_deck_counts_from_json(talishar_src, deck_json_path).keys())


def resolve_deck_counts_from_json(talishar_src: Path, deck_json_path: str) -> dict[str, int]:
    """Parse a FaBrary/FABdb deck JSON and return {Talishar card-name ID: count}.

    The JSON format stores card names (e.g. ``"arcanic_crackle_red": 2``) in
    ``deck`` and ``sideboard`` dicts.  These names are the IDs exposed by
    Talishar observations, so keep them as the generated engine's runtime IDs.
    """
    try:
        with open(deck_json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"  WARNING: could not read deck JSON {deck_json_path}: {exc}")
        return {}

    counts: dict[str, int] = {}
    for zone in ("deck", "sideboard"):
        zone_data = data.get(zone, {})
        if isinstance(zone_data, dict):
            items = zone_data.items()
        elif isinstance(zone_data, list):
            items = ((n, 1) for n in zone_data)
        else:
            continue
        for name, cnt in items:
            card_name = str(name or "").strip()
            if card_name:
                counts[card_name] = counts.get(card_name, 0) + int(cnt)

    print(f"  Resolved {len(counts)} card ID(s) ({sum(counts.values())} total) "
          f"from deck JSON: {deck_json_path}")
    return counts


# ── Asset deck-file resolution ─────────────────────────────────────────────────

_NAME_TO_ID_RE = re.compile(r'"([\w_]+)"\s*=>\s*"([A-Z]{2,4}\d+)"')


def _build_name_to_id(talishar_src: Path) -> dict[str, str]:
    """Parse GeneratedCardDictionaries.php to build a card-name→card-ID map."""
    dict_file = talishar_src / "GeneratedCode" / "GeneratedCardDictionaries.php"
    if not dict_file.exists():
        return {}
    text = dict_file.read_text(encoding="utf-8", errors="replace")
    return dict(_NAME_TO_ID_RE.findall(text))


def _build_id_to_name(talishar_src: Path) -> dict[str, str]:
    """Parse GeneratedCardDictionaries.php to build a card-ID→card-name map."""
    name_to_id = _build_name_to_id(talishar_src)
    return {v: k for k, v in name_to_id.items()}


def _parse_generated_stat_block(text: str, fn_name: str) -> dict[str, int]:
    """Extract name→value mapping from a Generated*Value PHP match() function."""
    start = text.find(f"function {fn_name}(")
    if start == -1:
        return {}
    # Find the next closing brace at the same depth
    end = text.find("\n}", start)
    if end == -1:
        end = start + 20_000  # fallback
    block = text[start:end]
    return {name: int(val) for name, val in _MATCH_STAT_RE.findall(block)}


def _build_generated_stats(talishar_src: Path) -> dict[str, dict[str, int]]:
    """Build a card-name→{power,pitch,cost,defense} map from GeneratedCardDictionaries.php."""
    dict_file = talishar_src / "GeneratedCode" / "GeneratedCardDictionaries.php"
    if not dict_file.exists():
        return {}
    text = dict_file.read_text(encoding="utf-8", errors="replace")
    power   = _parse_generated_stat_block(text, "GeneratedPowerValue")
    pitch   = _parse_generated_stat_block(text, "GeneratedPitchValue")
    cost    = _parse_generated_stat_block(text, "GeneratedCardCost")
    defense = _parse_generated_stat_block(text, "GeneratedBlockValue")
    all_names = set(power) | set(pitch) | set(cost) | set(defense)
    return {
        name: {
            "power":   power.get(name, 0),
            "pitch":   pitch.get(name, 0),
            "cost":    cost.get(name, 0),
            "defense": defense.get(name, 0),
        }
        for name in all_names
    }


def _build_character_stats(talishar_src: Path) -> tuple[dict[str, int], dict[str, int]]:
    """Build Talishar character health/intellect maps from generated dictionaries."""
    dict_file = talishar_src / "GeneratedCode" / "GeneratedCardDictionaries.php"
    if not dict_file.exists():
        return {}, {}
    text = dict_file.read_text(encoding="utf-8", errors="replace")
    health = _parse_generated_stat_block(text, "GeneratedCharacterHealth")
    intellect = _parse_generated_stat_block(text, "GeneratedCharacterIntellect")
    return health, intellect


def _hero_health(hero_id: str, health_by_id: dict[str, int]) -> int:
    return health_by_id.get(hero_id, 20)


def _hero_intellect(hero_id: str, intellect_by_id: dict[str, int]) -> int:
    return intellect_by_id.get(hero_id, 4)


def resolve_deck_asset_info(talishar_src: Path, deck_name: str) -> DeckAssetInfo:
    """Read Talishar/Assets/<deck_name>.txt preserving hero/equipment setup."""
    asset_file = talishar_src / "Assets" / f"{deck_name}.txt"
    if not asset_file.exists():
        return DeckAssetInfo()
    lines = [
        line.strip()
        for line in asset_file.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    setup_cards = lines[0].split() if lines else []
    deck_cards = "\n".join(lines[1:]).split()
    counts: dict[str, int] = {}
    for name in deck_cards:
        card_name = str(name or "").strip()
        if card_name:
            counts[card_name] = counts.get(card_name, 0) + 1
    return DeckAssetInfo(
        hero_id=setup_cards[0] if setup_cards else "",
        equipment_ids=setup_cards[1:],
        deck_counts=counts,
    )


def resolve_deck_counts_from_assets(talishar_src: Path, deck_name: str) -> dict[str, int]:
    """Read Talishar/Assets/<deck_name>.txt and return {Talishar card-name ID: count}.

    Falls back gracefully if the file or dictionary is missing.
    """
    return resolve_deck_asset_info(talishar_src, deck_name).deck_counts


def resolve_deck_from_assets(talishar_src: Path, deck_name: str) -> set[str]:
    """Read Talishar/Assets/<deck_name>.txt and return the set of card IDs."""
    return set(resolve_deck_counts_from_assets(talishar_src, deck_name).keys())


# ── PHP source mining ──────────────────────────────────────────────────────────

_CARD_ID_RE = re.compile(r'\$cardID\s*==\s*["\']([A-Z]{3}\d+)["\']')
_COST_RE    = re.compile(r'["\']cost["\']\s*=>\s*(\d+)')
_PITCH_RE   = re.compile(r'["\']pitch["\']\s*=>\s*(\d+)')
_POWER_RE   = re.compile(r'["\']power["\']\s*=>\s*(\d+)')
_DEF_RE     = re.compile(r'["\']defense["\']\s*=>\s*(\d+)')
_NAME_RE    = re.compile(r'["\']name["\']\s*=>\s*["\']([^"\']+)["\']')
_TYPE_RE    = re.compile(r'["\']type_text["\']\s*=>\s*["\']([^"\']+)["\']')

# Matches PHP match() entries: "card_name" => integer_value
_MATCH_STAT_RE = re.compile(r'"([\w_]+)"\s*=>\s*(-?\d+)')


def _extract_block_for_card(php_text: str, card_id: str, window: int = 60) -> str:
    lines = php_text.splitlines()
    for i, line in enumerate(lines):
        if card_id in line:
            start = max(0, i - 5)
            end = min(len(lines), i + window)
            return "\n".join(lines[start:end])
    return ""


def _parse_card_array(php_text: str, card_id: str) -> CardMeta:
    meta = CardMeta(card_id=card_id)
    idx = php_text.find(f'"{card_id}"')
    if idx == -1:
        idx = php_text.find(f"'{card_id}'")
    if idx == -1:
        return meta
    block = php_text[max(0, idx - 100): idx + 800]
    if m := _NAME_RE.search(block):
        meta.name = m.group(1)
    if m := _COST_RE.search(block):
        meta.cost = int(m.group(1))
    if m := _PITCH_RE.search(block):
        meta.pitch = int(m.group(1))
    if m := _POWER_RE.search(block):
        meta.power = int(m.group(1))
    if m := _DEF_RE.search(block):
        meta.defense = int(m.group(1))
    if m := _TYPE_RE.search(block):
        meta.card_type = m.group(1).lower()
    return meta


def scan_php_sources(
    talishar_src: Path, card_ids: set[str]
) -> dict[str, CardMeta]:
    """Walk all PHP files and extract metadata + logic for each card_id."""
    metas: dict[str, CardMeta] = {cid: CardMeta(card_id=cid) for cid in card_ids}

    # ── Fast path: read stats from Generated*Value functions (keyed by card name) ──
    id_to_name = _build_id_to_name(talishar_src)
    gen_stats = _build_generated_stats(talishar_src)
    stats_applied = 0
    for cid, meta in metas.items():
        card_name = id_to_name.get(cid, cid)
        if card_name and card_name in gen_stats:
            s = gen_stats[card_name]
            meta.name = meta.name or card_name
            meta.cost    = s["cost"]
            meta.pitch   = s["pitch"]
            meta.power   = s["power"]
            meta.defense = s["defense"]
            stats_applied += 1
    print(f"  Stats from GeneratedCardDictionaries: {stats_applied}/{len(card_ids)} cards")

    # ── Scan PHP files for card-type and PHP logic snippets ─────────────────
    php_files = list(talishar_src.rglob("*.php"))
    print(f"  Scanning {len(php_files)} PHP files for {len(card_ids)} card IDs…")

    for php_file in php_files:
        try:
            text = php_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        found_here = set(_CARD_ID_RE.findall(text)) | {
            cid for cid in card_ids if cid in text
        }
        for cid in found_here & set(metas.keys()):
            m = _parse_card_array(text, cid)
            ex = metas[cid]
            ex.name = m.name or ex.name
            # Only override stats from PHP if they look richer (non-zero)
            if m.cost and not ex.cost:    ex.cost    = m.cost
            if m.pitch and not ex.pitch:  ex.pitch   = m.pitch
            if m.power and not ex.power:  ex.power   = m.power
            if m.defense and not ex.defense: ex.defense = m.defense
            ex.card_type = m.card_type or ex.card_type
            ex.php_source_file = ex.php_source_file or str(
                php_file.relative_to(talishar_src)
            )
            if not ex.php_snippet:
                ex.php_snippet = _extract_block_for_card(text, cid)

    found = sum(1 for m in metas.values() if m.php_source_file)
    print(f"  PHP source found for {found}/{len(card_ids)} cards")
    return metas


# ── C++ template strings ───────────────────────────────────────────────────────

_GAMESTATE_H = """\
#pragma once
// AUTO-GENERATED — do not edit manually
// Matchup: {deck1} vs {deck2}
// Generated: {timestamp}

#include <array>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ── Card ─────────────────────────────────────────────────────────────────────

struct Card {{
    std::string card_id;
    std::string name;
    int  cost    = 0;
    int  pitch   = 0;
    int  power   = 0;
    int  defense = 0;
    std::string card_type;  // "attack", "reaction", "equipment", ...
    std::string zone;
    int  action_code = 0;
}};

// ── Player ────────────────────────────────────────────────────────────────────

struct PlayerState {{
    int health    = 20;
    int intellect = 4;
    int resources = 0;
    std::vector<Card> hand;
    std::vector<Card> deck;
    std::vector<Card> discard;
    std::vector<Card> equipment;
    std::vector<Card> arsenal;
    std::vector<Card> pitch_zone;
    std::string hero_card_id;
}};

// ── Phase ─────────────────────────────────────────────────────────────────────

enum class TurnPhase : int {{
    START = 0, MAIN, PITCH, ATTACK, BLOCK, DAMAGE, END, OVER
}};

// ── Legal action ──────────────────────────────────────────────────────────────

struct LegalAction {{
    int         action_code  = 0;
    std::string button_input;
    std::string card_id;
    std::string zone;
    std::string label;
}};

// ── Game state ────────────────────────────────────────────────────────────────

struct GameState {{
    std::array<PlayerState, 2> players;  // [0]=P1, [1]=P2
    TurnPhase   phase     = TurnPhase::START;
    int         turn_no   = 0;
    int         priority  = 0;   // 0=P1, 1=P2
    bool        game_over = false;
    int         winner    = -1;  // -1=none, 0=P1, 1=P2
    // Stalemate detection: if both players do nothing but pass this many
    // times in a row without any card being played, declare a draw.
    // Prevents infinite loops when card stubs are not yet implemented.
    int         consecutive_passes     = 0;
    int         max_consecutive_passes = 20;

    using EffectFn = std::function<void(GameState&, int /*player_idx*/)>;
    std::unordered_map<std::string, EffectFn> effects;

    // Core API
    std::vector<LegalAction> get_legal_actions() const;
    void apply_action(const LegalAction& action);
    void register_all_cards();
    void init_standard_decks();   // deal opening hands from pre-built deck lists
    void sync_opening_hand(int player_idx, const std::vector<std::string>& card_ids);
    void set_priority(int player_idx);

private:
    void _advance_phase();
    void _draw_cards(int player_idx, int n);
}};
"""

_GAMESTATE_CPP = """\
// AUTO-GENERATED — do not edit manually
#include "gamestate.h"
#include "cards.h"
#include <algorithm>
#include <random>

std::vector<LegalAction> GameState::get_legal_actions() const {{
    std::vector<LegalAction> actions;
    const auto& p = players[priority];

    // Play affordable cards from hand: pitch available = sum of pitch of all
    // OTHER cards (by position) in hand.
    for (size_t i = 0; i < p.hand.size(); ++i) {{
        int avail = 0;
        for (size_t j = 0; j < p.hand.size(); ++j) {{
            if (j != i) avail += p.hand[j].pitch;
        }}
        if (p.hand[i].cost <= avail) {{
            actions.push_back(LegalAction{{27, std::to_string(i), p.hand[i].card_id, "hand", p.hand[i].name}});
        }}
    }}
    // Always include pass
    actions.push_back(LegalAction{{99, "", "", "button", "Pass"}});
    return actions;
}}

void GameState::apply_action(const LegalAction& action) {{
    if (action.action_code == 99) {{
        // Pass: count consecutive passes for stalemate detection
        consecutive_passes += 1;
        if (consecutive_passes >= max_consecutive_passes) {{
            game_over = true;
            winner    = -1;  // draw
            return;
        }}
        _advance_phase();
        return;
    }}
    if (action.action_code == 27) {{
        // Any card played resets the stalemate counter
        consecutive_passes = 0;
        auto& hand = players[priority].hand;
        size_t idx = static_cast<size_t>(std::stoi(action.button_input));
        if (idx < hand.size()) {{
            Card card = hand[idx];
            hand.erase(hand.begin() + idx);
            auto it = effects.find(card.card_id);
            if (it != effects.end()) {{
                it->second(*this, priority);
            }}
        }}
        // Check for game over immediately after damage — this ensures the
        // reward is attributed to the player who dealt lethal (priority
        // has NOT yet switched).
        for (int i = 0; i < 2; ++i) {{
            if (players[i].health <= 0) {{
                game_over = true;
                winner    = 1 - i;
            }}
        }}
        return;
    }}
    throw std::runtime_error("Unknown action_code: " + std::to_string(action.action_code));
}}

void GameState::_draw_cards(int player_idx, int n) {{
    auto& p = players[player_idx];
    for (int d = 0; d < n; ++d) {{
        if (p.deck.empty()) {{
            // Recycle discard into deck and reshuffle
            if (p.discard.empty()) break;
            p.deck = p.discard;
            p.discard.clear();
            std::shuffle(p.deck.begin(), p.deck.end(),
                         std::mt19937{{std::random_device{{}}()}});
        }}
        if (!p.deck.empty()) {{
            p.hand.push_back(p.deck.back());
            p.deck.pop_back();
        }}
    }}
}}

void GameState::sync_opening_hand(int player_idx, const std::vector<std::string>& card_ids) {{
    if (player_idx < 0 || player_idx >= 2) {{
        throw std::runtime_error("player_idx must be 0 or 1");
    }}

    auto& p = players[player_idx];
    std::vector<Card> available;
    available.reserve(p.hand.size() + p.deck.size());
    available.insert(available.end(), p.hand.begin(), p.hand.end());
    available.insert(available.end(), p.deck.begin(), p.deck.end());

    std::vector<Card> synced_hand;
    synced_hand.reserve(card_ids.size());
    for (const auto& card_id : card_ids) {{
        auto it = std::find_if(available.begin(), available.end(), [&](const Card& c) {{
            return c.card_id == card_id;
        }});
        if (it == available.end()) {{
            throw std::runtime_error("Cannot sync opening hand; card not found: " + card_id);
        }}
        synced_hand.push_back(*it);
        available.erase(it);
    }}

    p.hand = synced_hand;
    p.deck = available;
}}

void GameState::set_priority(int player_idx) {{
    if (player_idx < 0 || player_idx >= 2) {{
        throw std::runtime_error("player_idx must be 0 or 1");
    }}
    priority = player_idx;
}}

void GameState::_advance_phase() {{
    switch (phase) {{
        case TurnPhase::START:  phase = TurnPhase::MAIN;   break;
        case TurnPhase::MAIN:   phase = TurnPhase::END;    break;
        case TurnPhase::ATTACK: phase = TurnPhase::BLOCK;  break;
        case TurnPhase::BLOCK:  phase = TurnPhase::DAMAGE; break;
        case TurnPhase::DAMAGE: phase = TurnPhase::END;    break;
        case TurnPhase::END: {{
            // Discard active player's remaining hand
            auto& active = players[priority];
            for (auto& c : active.hand) active.discard.push_back(c);
            active.hand.clear();
            // Switch priority
            phase    = TurnPhase::MAIN;
            turn_no += 1;
            priority = 1 - priority;
            // New active player draws 4 cards (intellect)
            _draw_cards(priority, 4);
            break;
        }}
        default: break;
    }}
    for (int i = 0; i < 2; ++i) {{
        if (players[i].health <= 0) {{
            game_over = true;
            winner    = 1 - i;
        }}
    }}
}}
"""

_CARDS_H_HEADER = """\
#pragma once
// AUTO-GENERATED — do not edit manually
// Card effect stubs for matchup: {deck1} vs {deck2}
// {n_cards} unique cards detected
//
// Each function applies `power` damage to the opponent by default.
// To implement richer card effects, edit the function body.
// Common translations from PHP:
//   $gamestate->playerHealth -= N        ->  gs.players[player_idx].health -= N;
//   $gamestate->opponentHealth -= N      ->  gs.players[1-player_idx].health -= N;
//   AddCardToHand($cardID, $playerID)    ->  (add Card to gs.players[...].hand)
//   DrawCard($playerID, N)               ->  (call gs._draw_cards(player_idx, N))

#include "gamestate.h"
#include <stdexcept>

"""

_CARD_STUB = """\
// ┌─ {card_id} : {name} ─────────────────────────────────────────────────────
// │ cost={cost}  pitch={pitch}  power={power}  defense={defense}  type={card_type}
// │ PHP source: {php_source_file}
// │
// │ Extracted PHP logic:
{php_comment}
// └──────────────────────────────────────────────────────────────────────────
inline void effect_{card_id}(GameState& gs, int player_idx) {{
    // Default effect: deal 'power' damage to the opponent.
    // For equipment/reactions this is a safe no-op when power == 0.
    int opp_idx = 1 - player_idx;
    gs.players[opp_idx].health -= {power};
}}

"""

_REGISTER_CPP = """\
// AUTO-GENERATED — do not edit manually
#include "gamestate.h"
#include "cards.h"

void GameState::register_all_cards() {{
{registrations}
}}
"""

_INIT_DECKS_CPP = """\
// AUTO-GENERATED — do not edit manually
#include "gamestate.h"
#include "cards.h"
#include <algorithm>
#include <random>

// Helper: push N copies of a Card into a vector
static void _push_card(std::vector<Card>& v, const Card& c, int n) {{
    for (int i = 0; i < n; ++i) v.push_back(c);
}}

void GameState::init_standard_decks() {{
    phase = TurnPhase::MAIN;
    turn_no = 0;
    priority = 0;
    game_over = false;
    winner = -1;
    consecutive_passes = 0;

    players[0].health = {p1_health};
    players[0].intellect = {p1_intellect};
    players[0].resources = 0;
    players[0].hero_card_id = "{p1_hero}";
    players[0].deck.clear();
    players[0].hand.clear();
    players[0].discard.clear();
    players[0].equipment.clear();
    players[0].arsenal.clear();
    players[0].pitch_zone.clear();

    players[1].health = {p2_health};
    players[1].intellect = {p2_intellect};
    players[1].resources = 0;
    players[1].hero_card_id = "{p2_hero}";
    players[1].deck.clear();
    players[1].hand.clear();
    players[1].discard.clear();
    players[1].equipment.clear();
    players[1].arsenal.clear();
    players[1].pitch_zone.clear();

    // ── P1 setup ({deck1}) ───────────────────────────────────────────────
{p1_setup}

    // ── P1 deck ({deck1}) ────────────────────────────────────────────────
{p1_cards}

    // ── P2 setup ({deck2}) ───────────────────────────────────────────────
{p2_setup}

    // ── P2 deck ({deck2}) ────────────────────────────────────────────────
{p2_cards}

    // Shuffle both decks
    auto rng0 = std::mt19937{{std::random_device{{}}()}};
    auto rng1 = std::mt19937{{std::random_device{{}}()}};
    std::shuffle(players[0].deck.begin(), players[0].deck.end(), rng0);
    std::shuffle(players[1].deck.begin(), players[1].deck.end(), rng1);

    // Deal starting hands from Talishar character intellect.
    _draw_cards(0, players[0].intellect);
    _draw_cards(1, players[1].intellect);
}}
"""

_BINDINGS_CPP = """\
// AUTO-GENERATED — do not edit manually
// pybind11 Python bindings — zero HTTP, direct C++ function calls
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gamestate.h"

namespace py = pybind11;

PYBIND11_MODULE(fab_engine, m) {{
    m.doc() = "FAB engine — {deck1} vs {deck2}";

    py::class_<Card>(m, "Card")
        .def_readonly("card_id", &Card::card_id)
        .def_readonly("name", &Card::name)
        .def_readonly("cost", &Card::cost)
        .def_readonly("pitch", &Card::pitch)
        .def_readonly("power", &Card::power)
        .def_readonly("defense", &Card::defense)
        .def_readonly("card_type", &Card::card_type)
        .def_readonly("zone", &Card::zone)
        .def_readonly("action_code", &Card::action_code);

    py::class_<LegalAction>(m, "LegalAction")
        .def_readonly("action_code",  &LegalAction::action_code)
        .def_readonly("button_input", &LegalAction::button_input)
        .def_readonly("card_id",      &LegalAction::card_id)
        .def_readonly("zone",         &LegalAction::zone)
        .def_readonly("label",        &LegalAction::label)
        .def("__repr__", [](const LegalAction& a) {{
            return "<LegalAction " + a.card_id + " mode=" +
                   std::to_string(a.action_code) + ">";
        }});

    py::class_<PlayerState>(m, "PlayerState")
        .def_readonly("health",    &PlayerState::health)
        .def_readonly("resources", &PlayerState::resources)
        .def_property_readonly("hand_size",
            [](const PlayerState& p) {{ return (int)p.hand.size(); }})
        .def_property_readonly("deck_size",
            [](const PlayerState& p) {{ return (int)p.deck.size(); }});

    py::class_<GameState>(m, "GameState")
        .def(py::init<>())
        .def("register_all_cards", &GameState::register_all_cards)
        .def("init_standard_decks", &GameState::init_standard_decks)
        .def("sync_opening_hand", &GameState::sync_opening_hand)
        .def("set_priority", &GameState::set_priority)
        .def("get_legal_actions",  &GameState::get_legal_actions)
        .def("apply_action",       &GameState::apply_action)
        .def_property_readonly("game_over",
            [](const GameState& g) {{ return g.game_over; }})
        .def_property_readonly("winner",
            [](const GameState& g) {{ return g.winner; }})
        .def_property_readonly("turn_no",
            [](const GameState& g) {{ return g.turn_no; }})
        .def_property_readonly("priority",
            [](const GameState& g) {{ return g.priority; }})
        .def_property_readonly("p1_health",
            [](const GameState& g) {{ return g.players[0].health; }})
        .def_property_readonly("p2_health",
            [](const GameState& g) {{ return g.players[1].health; }})
        .def_property_readonly("p1_hand_size",
            [](const GameState& g) {{ return (int)g.players[0].hand.size(); }})
        .def_property_readonly("p2_hand_size",
            [](const GameState& g) {{ return (int)g.players[1].hand.size(); }})
        .def_property_readonly("p1_hand",
            [](const GameState& g) {{ return g.players[0].hand; }})
        .def_property_readonly("p2_hand",
            [](const GameState& g) {{ return g.players[1].hand; }})
        .def_property_readonly("p1_deck_size",
            [](const GameState& g) {{ return (int)g.players[0].deck.size(); }})
        .def_property_readonly("p2_deck_size",
            [](const GameState& g) {{ return (int)g.players[1].deck.size(); }})
        .def_property_readonly("p1_pitch_size",
            [](const GameState& g) {{ return (int)g.players[0].pitch_zone.size(); }})
        .def_property_readonly("p2_pitch_size",
            [](const GameState& g) {{ return (int)g.players[1].pitch_zone.size(); }})
        .def_property_readonly("consecutive_passes",
            [](const GameState& g) {{ return g.consecutive_passes; }})
        .def_readwrite("max_consecutive_passes", &GameState::max_consecutive_passes);
}}
"""

_CMAKELISTS = """\
cmake_minimum_required(VERSION 3.18)
project(fab_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Locate the Python interpreter / headers (honours the active venv / conda env)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Locate pybind11 — pass -Dpybind11_DIR=$(python -m pybind11 --cmakedir) on the
# cmake command line when pybind11 was installed via pip.
find_package(pybind11 REQUIRED)

pybind11_add_module(fab_engine
    gamestate.cpp
    register_cards.cpp
    init_decks.cpp
    bindings.cpp
)
target_include_directories(fab_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Copy the built module next to the source so Python can import it directly
add_custom_command(TARGET fab_engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:fab_engine> ${CMAKE_CURRENT_SOURCE_DIR}/
    COMMENT "Copying fab_engine module to source directory"
)
"""

_BUILD_SH = """\
#!/usr/bin/env bash
# Build the fab_engine C++ module
set -e
pip install pybind11 --quiet
cmake -B build -DCMAKE_BUILD_TYPE=Release .
cmake --build build --config Release
echo "Build complete. Module: $(ls fab_engine*.so fab_engine*.pyd 2>/dev/null | head -1)"
"""

_BUILD_PS1 = """\
# Build the fab_engine C++ module (PowerShell)
pip install pybind11 --quiet
cmake -B build -DCMAKE_BUILD_TYPE=Release .
cmake --build build --config Release
Write-Host "Build complete."
"""

_README = """\
# {deck1} vs {deck2} — C++ FAB Engine
Generated: {timestamp}

## Cards requiring implementation

{card_table}

## Build

```bash
pip install pybind11
cmake -B build .
cmake --build build --config Release
```

## Usage

```python
from cpp_engine_environment import CppEngineEnvironment
env = CppEngineEnvironment(engine_dir="{out_dir_abs}")
obs, info = env.reset()
result = env.step(0)
```

## Implementation guide

Open `cards.h`. Each card stub has the PHP logic as comments above the C++
function body. Translate PHP → C++:

| PHP | C++ |
|-----|-----|
| `$gamestate->playerHealth -= N` | `gs.players[player_idx].health -= N;` |
| `$gamestate->opponentHealth -= N` | `gs.players[1-player_idx].health -= N;` |

Remove the `throw` line once implemented.
"""


# ── Generator ──────────────────────────────────────────────────────────────────

def generate(
    out_dir: Path,
    deck1: str,
    deck2: str,
    metas: dict[str, CardMeta],
    p1_ids: set[str],
    p2_ids: set[str],
    p1_counts: dict[str, int] | None = None,
    p2_counts: dict[str, int] | None = None,
    p1_asset_info: DeckAssetInfo | None = None,
    p2_asset_info: DeckAssetInfo | None = None,
    character_health: dict[str, int] | None = None,
    character_intellect: dict[str, int] | None = None,
) -> None:
    # Default: 2 copies of each unique card if counts not provided.
    # Empty dicts are treated as missing because callers initialise counts
    # before trying live discovery / asset resolution.
    if not p1_counts:
        p1_counts = {cid: 2 for cid in p1_ids}
    if not p2_counts:
        p2_counts = {cid: 2 for cid in p2_ids}
    p1_asset_info = p1_asset_info or DeckAssetInfo(deck_counts=p1_counts)
    p2_asset_info = p2_asset_info or DeckAssetInfo(deck_counts=p2_counts)
    character_health = character_health or {}
    character_intellect = character_intellect or {}

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_ids = sorted(metas.keys())

    # gamestate.h
    (out_dir / "gamestate.h").write_text(
        _GAMESTATE_H.format(deck1=deck1, deck2=deck2, timestamp=ts), encoding="utf-8"
    )

    # gamestate.cpp  (template uses {{ / }} escapes — call .format() to render)
    (out_dir / "gamestate.cpp").write_text(_GAMESTATE_CPP.format(), encoding="utf-8")

    # cards.h
    cards_h = _CARDS_H_HEADER.format(
        deck1=deck1, deck2=deck2, n_cards=len(all_ids)
    )
    for cid in all_ids:
        m = metas[cid]
        php_lines = m.php_snippet.splitlines() if m.php_snippet else ["(no PHP source found)"]
        php_comment = "\n".join(f"// │   {ln}" for ln in php_lines[:50])
        if len(php_lines) > 50:
            php_comment += f"\n// │   … ({len(php_lines) - 50} more lines)"
        cards_h += _CARD_STUB.format(
            card_id=cid,
            name=m.name or cid,
            cost=m.cost,
            pitch=m.pitch,
            power=m.power,
            defense=m.defense,
            card_type=m.card_type or "unknown",
            php_source_file=m.php_source_file or "not found",
            php_comment=php_comment,
        )
    (out_dir / "cards.h").write_text(cards_h, encoding="utf-8")

    # register_cards.cpp
    reg_lines = "\n".join(
        f'    effects["{cid}"] = effect_{cid};' for cid in all_ids
    )
    (out_dir / "register_cards.cpp").write_text(
        _REGISTER_CPP.format(registrations=reg_lines), encoding="utf-8"
    )

    # init_decks.cpp — hard-codes the two deck lists and deals opening hands
    def _card_init_line(
        cid: str,
        count: int,
        player_vec: str,
        *,
        zone: str = "deck",
        action_code: int = 27,
    ) -> str:
        m = metas.get(cid, CardMeta(card_id=cid))
        safe_name = (m.name or cid).replace('"', '\\"')
        safe_type = (m.card_type or "unknown").replace('"', '\\"')
        safe_zone = zone.replace('"', '\\"')
        return (
            f'    _push_card({player_vec}, '
            f'Card{{"{cid}", "{safe_name}", {m.cost}, {m.pitch}, {m.power}, '
            f'{m.defense}, "{safe_type}", "{safe_zone}", {action_code}}}, {count});'
        )

    def _setup_init_lines(info: DeckAssetInfo, player_vec: str) -> str:
        lines = []
        for cid in info.equipment_ids:
            lines.append(_card_init_line(cid, 1, player_vec, zone="equipment", action_code=0))
        return "\n".join(lines)

    p1_lines = "\n".join(
        _card_init_line(cid, cnt, "players[0].deck")
        for cid, cnt in sorted(p1_counts.items())
        if cid in metas
    )
    p2_lines = "\n".join(
        _card_init_line(cid, cnt, "players[1].deck")
        for cid, cnt in sorted(p2_counts.items())
        if cid in metas
    )
    p1_setup_lines = _setup_init_lines(p1_asset_info, "players[0].equipment")
    p2_setup_lines = _setup_init_lines(p2_asset_info, "players[1].equipment")
    (out_dir / "init_decks.cpp").write_text(
        _INIT_DECKS_CPP.format(
            deck1=deck1, deck2=deck2,
            p1_hero=p1_asset_info.hero_id.replace('"', '\\"'),
            p2_hero=p2_asset_info.hero_id.replace('"', '\\"'),
            p1_health=_hero_health(p1_asset_info.hero_id, character_health),
            p2_health=_hero_health(p2_asset_info.hero_id, character_health),
            p1_intellect=_hero_intellect(p1_asset_info.hero_id, character_intellect),
            p2_intellect=_hero_intellect(p2_asset_info.hero_id, character_intellect),
            p1_setup=p1_setup_lines or "    // (no setup cards)",
            p2_setup=p2_setup_lines or "    // (no setup cards)",
            p1_cards=p1_lines or "    // (no cards)",
            p2_cards=p2_lines or "    // (no cards)",
        ),
        encoding="utf-8",
    )

    # bindings.cpp
    (out_dir / "bindings.cpp").write_text(
        _BINDINGS_CPP.format(deck1=deck1, deck2=deck2), encoding="utf-8"
    )

    # CMakeLists.txt
    (out_dir / "CMakeLists.txt").write_text(_CMAKELISTS, encoding="utf-8")

    # build scripts
    (out_dir / "build.sh").write_text(_BUILD_SH, encoding="utf-8")
    (out_dir / "build.ps1").write_text(_BUILD_PS1, encoding="utf-8")

    # card_manifest.json
    manifest = {
        "deck1": deck1,
        "deck2": deck2,
        "generated": ts,
            "p1_hero": p1_asset_info.hero_id,
            "p2_hero": p2_asset_info.hero_id,
            "p1_health": _hero_health(p1_asset_info.hero_id, character_health),
            "p2_health": _hero_health(p2_asset_info.hero_id, character_health),
            "p1_intellect": _hero_intellect(p1_asset_info.hero_id, character_intellect),
            "p2_intellect": _hero_intellect(p2_asset_info.hero_id, character_intellect),
        "p1_cards": sorted(p1_ids),
        "p2_cards": sorted(p2_ids),
        "cards": {
            cid: {
                "name": m.name,
                "cost": m.cost,
                "pitch": m.pitch,
                "power": m.power,
                "defense": m.defense,
                "type": m.card_type,
                "php_file": m.php_source_file,
                "php_found": bool(m.php_snippet),
            }
            for cid, m in metas.items()
        },
    }
    (out_dir / "card_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # README
    card_table_rows = []
    for cid in all_ids:
        m = metas[cid]
        php_flag = "✓" if m.php_snippet else "✗"
        card_table_rows.append(
            f"| {php_flag} | `{cid}` | {m.name or '?'} | "
            f"cost={m.cost} pitch={m.pitch} power={m.power} |"
        )
    card_table = (
        "| PHP | ID | Name | Stats |\n|-----|-----|------|-------|\n"
        + "\n".join(card_table_rows)
    )
    (out_dir / "README.md").write_text(
        _README.format(
            deck1=deck1,
            deck2=deck2,
            timestamp=ts,
            card_table=card_table,
            out_dir_abs=str(out_dir.resolve()),
        ),
        encoding="utf-8",
    )

    # Summary
    implemented = sum(1 for m in metas.values() if m.php_snippet)
    print(f"\nOK  Generated {len(all_ids)} card stubs "
          f"({implemented} with extracted PHP logic)")
    print(f"   Output : {out_dir.resolve()}")
    print(f"\n   Next steps:")
    print(f"   1. Open {out_dir / 'cards.h'} -- translate PHP -> C++ for each stub")
    print(f"   2. cd {out_dir} && cmake -B build . && cmake --build build")
    print(f"   3. Use CppEngineEnvironment(engine_dir=r'{out_dir.resolve()}')")
    print(f"\n   Top cards to implement first:")
    for cid in all_ids[:12]:
        m = metas[cid]
        flag = "Y" if m.php_snippet else "N"
        print(f"   [{flag}]  {cid:10s}  {(m.name or '?'):30s}  "
              f"cost={m.cost} pitch={m.pitch} power={m.power}")
    print(
        f"\n   Reset metadata: P1 {p1_asset_info.hero_id or '?'} "
        f"hp={_hero_health(p1_asset_info.hero_id, character_health)} "
        f"int={_hero_intellect(p1_asset_info.hero_id, character_intellect)}; "
        f"P2 {p2_asset_info.hero_id or '?'} "
        f"hp={_hero_health(p2_asset_info.hero_id, character_health)} "
        f"int={_hero_intellect(p2_asset_info.hero_id, character_intellect)}"
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--talishar-src", default="Talishar",
        help="Path to Talishar PHP source root (default: Talishar/)"
    )
    parser.add_argument("--deck1", default="Ira", help="P1 deck name")
    parser.add_argument("--deck2", default="Ira", help="P2 deck name")
    parser.add_argument(
        "--deck1-json", default=None,
        help="Path to a FaBrary/FABdb deck JSON for P1 (deck + sideboard card names "
             "are resolved to card IDs and merged into the engine).  Optional but "
             "strongly recommended — without it the engine only contains hero cards."
    )
    parser.add_argument(
        "--deck2-json", default=None,
        help="Path to a FaBrary/FABdb deck JSON for P2 (same as --deck1-json)."
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory (default: results/cpp_engines/<deck1>_vs_<deck2>)"
    )
    parser.add_argument(
        "--base-url", default=None,
        help="Talishar server URL (default: $TALISHAR_URL or http://localhost)"
    )
    parser.add_argument(
        "--no-server", action="store_true",
        help="Skip live card discovery; use PHP scan only"
    )
    args = parser.parse_args()

    base_url = args.base_url or os.environ.get("TALISHAR_URL", "http://localhost")
    talishar_src = Path(args.talishar_src)
    matchup_key = f"{args.deck1}_vs_{args.deck2}"
    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "results" / "cpp_engines" / matchup_key
    )

    print("=== FAB C++ Engine Generator ===")
    print(f"  Matchup : {args.deck1} vs {args.deck2}")
    print(f"  PHP src : {talishar_src.resolve()}")
    print(f"  Output  : {out_dir.resolve()}")
    if args.deck1_json:
        print(f"  P1 JSON : {args.deck1_json}")
    if args.deck2_json:
        print(f"  P2 JSON : {args.deck2_json}")
    print()

    # Step 1: card discovery
    p1_ids: set[str] = set()
    p2_ids: set[str] = set()
    p1_counts: dict[str, int] = {}
    p2_counts: dict[str, int] = {}
    p1_asset_info = DeckAssetInfo()
    p2_asset_info = DeckAssetInfo()

    if not args.no_server:
        print("Step 1: Discovering cards via live Talishar game...")
        try:
            p1_ids, p2_ids = discover_deck_cards(base_url, args.deck1, args.deck2)
        except Exception as exc:
            print(f"  WARNING: live discovery failed ({exc})")
            print("  Falling back to PHP scan only")
    else:
        print("Step 1: Skipped (--no-server)")
        # Try to resolve deck card IDs from Talishar/Assets/<deck>.txt files
        if talishar_src.exists():
            p1_asset_info = resolve_deck_asset_info(talishar_src, args.deck1)
            p2_asset_info = resolve_deck_asset_info(talishar_src, args.deck2)
            if p1_asset_info.deck_counts:
                p1_counts.update(p1_asset_info.deck_counts)
                p1_ids |= set(p1_asset_info.deck_counts) | set(p1_asset_info.equipment_ids)
                if p1_asset_info.hero_id:
                    p1_ids.add(p1_asset_info.hero_id)
                print(f"  Resolved {len(p1_asset_info.deck_counts)} card ID(s) for {args.deck1} from Assets txt")
            if p2_asset_info.deck_counts:
                p2_counts.update(p2_asset_info.deck_counts)
                p2_ids |= set(p2_asset_info.deck_counts) | set(p2_asset_info.equipment_ids)
                if p2_asset_info.hero_id:
                    p2_ids.add(p2_asset_info.hero_id)
                print(f"  Resolved {len(p2_asset_info.deck_counts)} card ID(s) for {args.deck2} from Assets txt")

    # Supplement with FaBrary deck JSON files (deck + sideboard card pools)
    if args.deck1_json:
        json_counts = resolve_deck_counts_from_json(talishar_src, args.deck1_json)
        before = len(p1_ids)
        p1_ids |= set(json_counts.keys())
        p1_counts.update(json_counts)
        print(f"  P1: added {len(p1_ids) - before} new card ID(s) from deck JSON "
              f"(total {len(p1_ids)}, {sum(p1_counts.values())} cards)")
    if args.deck2_json:
        json_counts = resolve_deck_counts_from_json(talishar_src, args.deck2_json)
        before = len(p2_ids)
        p2_ids |= set(json_counts.keys())
        p2_counts.update(json_counts)
        print(f"  P2: added {len(p2_ids) - before} new card ID(s) from deck JSON "
              f"(total {len(p2_ids)}, {sum(p2_counts.values())} cards)")

    # Step 2: PHP scan
    if talishar_src.exists():
        p1_asset_info = resolve_deck_asset_info(talishar_src, args.deck1)
        p2_asset_info = resolve_deck_asset_info(talishar_src, args.deck2)
        if p1_asset_info.deck_counts:
            p1_ids |= set(p1_asset_info.deck_counts.keys()) | set(p1_asset_info.equipment_ids)
            if p1_asset_info.hero_id:
                p1_ids.add(p1_asset_info.hero_id)
            p1_counts.update(p1_asset_info.deck_counts)
            print(f"  Resolved {len(p1_asset_info.deck_counts)} card ID(s) for {args.deck1} from Assets txt "
                  f"({sum(p1_asset_info.deck_counts.values())} cards)")
        if p2_asset_info.deck_counts:
            p2_ids |= set(p2_asset_info.deck_counts.keys()) | set(p2_asset_info.equipment_ids)
            if p2_asset_info.hero_id:
                p2_ids.add(p2_asset_info.hero_id)
            p2_counts.update(p2_asset_info.deck_counts)
            print(f"  Resolved {len(p2_asset_info.deck_counts)} card ID(s) for {args.deck2} from Assets txt "
                  f"({sum(p2_asset_info.deck_counts.values())} cards)")

        all_ids = p1_ids | p2_ids
        print("\nStep 2: Scanning PHP source…")
        if not all_ids:
            print("  No asset deck files found — scanning all PHP card definitions…")
            for php_file in talishar_src.rglob("*.php"):
                try:
                    text = php_file.read_text(encoding="utf-8", errors="replace")
                    all_ids |= set(_CARD_ID_RE.findall(text))
                except OSError:
                    continue
            p1_ids = all_ids
            print(f"  Found {len(all_ids)} total card IDs in PHP source")
        metas = scan_php_sources(talishar_src, all_ids)
    else:
        all_ids = p1_ids | p2_ids
        print(f"\nStep 2: WARNING — Talishar source not found at {talishar_src}")
        print("  Generating stub-only output (no PHP logic to extract)")
        metas = {cid: CardMeta(card_id=cid) for cid in all_ids}

    if not metas:
        print("\nERROR: No cards found. Check --talishar-src and/or run Talishar.")
        sys.exit(1)

    # Step 3: generate
    print(f"\nStep 3: Generating C++ for {len(metas)} cards…")
    character_health, character_intellect = _build_character_stats(talishar_src)
    generate(
        out_dir,
        args.deck1,
        args.deck2,
        metas,
        p1_ids,
        p2_ids,
        p1_counts,
        p2_counts,
        p1_asset_info,
        p2_asset_info,
        character_health,
        character_intellect,
    )


if __name__ == "__main__":
    main()
