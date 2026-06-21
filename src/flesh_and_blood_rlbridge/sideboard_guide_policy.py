"""Silver Age sideboard warm-start policy.

Heuristics derived from the FaB Silver Age sideboarding guide:
https://fabtcg.com/articles/silver-age-sideboarding-guide/

The guide emphasises matchup-specific swaps:
* **Arcane** (Wizard / Ice / Runeblade casters): bring Nullrune equipment,
  arcane barrier cards, and cards that punish spell-heavy lines; cut generic
  low-impact equipment.
* **Fatigue** (Guardian, Briar, long games): bring deck damage, recursive
  threats, and cards that push damage through attrition; cut slow clash packages.
* **Aggro** (Fai, Kayo, Azalea): bring defense reactions and high-defense
  cards; cut slow setup and clash cards that lose tempo.
* **Defense-reaction** (Dorinthea, Warrior): bring high-power attacks and
  cards that punish blocking; cut low-impact cards that trade poorly.
* **Assassin / on-hit** (Arakni): bring cards that go tall or punish wide
  setups; cut cards weak into stealth / on-hit lines.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

# Hero-id substring → matchup archetype (Silver Age guide groupings).
_HERO_ARCHETYPE: tuple[tuple[str, str], ...] = (
    ("iyslander", "arcane"),
    ("viserai", "arcane"),
    ("aurora", "arcane"),
    ("kano", "arcane"),
    ("kassai", "defense_reaction"),
    ("dorinthea", "defense_reaction"),
    ("bravo", "fatigue"),
    ("briar", "fatigue"),
    ("kayo", "aggro"),
    ("fai", "aggro"),
    ("azalea", "aggro"),
    ("arakni", "assassin"),
    ("enigma", "combo"),
    ("ira", "aggro"),
    ("dash", "aggro"),
    ("rhinar", "aggro"),
    ("lexi", "aggro"),
    ("oldhim", "fatigue"),
    ("boltyn", "aggro"),
    ("prism", "arcane"),
    ("chane", "fatigue"),
    ("lebby", "arcane"),
)

# Per-archetype card scoring weights (positive = prefer in deck vs this opponent).
_ARCHETYPE_WEIGHTS: dict[str, dict[str, float]] = {
    "arcane": {
        "keyword:arcane_barrier": 4.0,
        "keyword:ward": 2.5,
        "id:nullrune": 5.0,
        "id:topsy": 3.0,
        "id:amulet_of": 2.0,
        "type:defense_reaction": 1.5,
        "high_defense": 1.0,
        "id:ironrot": -2.0,
        "id:crude": -1.5,
        "keyword:clash": -1.0,
    },
    "fatigue": {
        "keyword:dominate": 2.5,
        "keyword:go_again": 1.5,
        "high_power": 2.0,
        "deck_damage": 3.0,
        "keyword:clash": -2.0,
        "type:defense_reaction": -0.5,
        "low_power": -1.0,
    },
    "aggro": {
        "type:defense_reaction": 4.0,
        "keyword:defense_reaction": 4.0,
        "high_defense": 3.0,
        "keyword:clash": -2.5,
        "low_defense": -1.5,
        "deck_damage": 1.0,
    },
    "defense_reaction": {
        "high_power": 3.5,
        "keyword:dominate": 2.5,
        "keyword:go_again": 1.5,
        "type:defense_reaction": -1.5,
        "low_power": -1.0,
    },
    "assassin": {
        "high_power": 2.5,
        "keyword:dominate": 2.0,
        "type:defense_reaction": 1.0,
        "high_defense": 1.5,
        "keyword:stealth": -1.0,
    },
    "combo": {
        "high_power": 2.0,
        "keyword:go_again": 2.5,
        "type:defense_reaction": 2.0,
        "high_defense": 1.5,
    },
}

_DEFAULT_ARCHETYPE = "aggro"
_SWAP_MARGIN = 0.75


@lru_cache(maxsize=1)
def _load_card_db() -> dict[str, dict[str, Any]]:
    db_path = Path(__file__).parent / "card_db" / "cards.json"
    try:
        records = json.loads(db_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for rec in records:
        cid = rec.get("id")
        if cid:
            out[str(cid)] = rec
            hyphen = str(cid).replace("_", "-")
            if hyphen not in out:
                out[hyphen] = rec
    return out


def classify_opponent_archetype(opponent_hero_id: str) -> str:
    """Map an opponent hero id to a Silver Age sideboard archetype."""
    hero = opponent_hero_id.lower().replace("-", "_")
    for prefix, archetype in _HERO_ARCHETYPE:
        if prefix in hero:
            return archetype
    return _DEFAULT_ARCHETYPE


def _card_meta(card_id: str, pool_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if card_id in pool_by_id:
        return pool_by_id[card_id]
    db = _load_card_db()
    return db.get(card_id) or db.get(card_id.replace("_", "-"), {})


def score_card_for_archetype(
    card_id: str,
    meta: dict[str, Any],
    archetype: str,
) -> float:
    """Return a sideboard-in score for *card_id* vs *archetype* (higher = keep in deck)."""
    weights = _ARCHETYPE_WEIGHTS.get(archetype, _ARCHETYPE_WEIGHTS[_DEFAULT_ARCHETYPE])
    score = 0.0
    cid = card_id.lower()
    keywords = {str(k).lower() for k in meta.get("keywords", [])}
    card_types = {str(t).lower() for t in meta.get("card_types", [])}
    defense = int(meta.get("defense") or 0)
    power = int(meta.get("power") or 0)
    text = str(meta.get("text") or "").lower()

    for key, weight in weights.items():
        if key.startswith("keyword:"):
            kw = key.split(":", 1)[1]
            if kw in keywords or kw.replace("_", " ") in text:
                score += weight
        elif key.startswith("id:"):
            frag = key.split(":", 1)[1]
            if frag in cid:
                score += weight
        elif key.startswith("type:"):
            ct = key.split(":", 1)[1]
            if ct in card_types:
                score += weight
        elif key == "high_defense" and defense >= 4:
            score += weight
        elif key == "low_defense" and 0 < defense < 3:
            score += weight
        elif key == "high_power" and power >= 4:
            score += weight
        elif key == "low_power" and 0 < power < 3:
            score += weight
        elif key == "deck_damage" and (
            "deck" in text and ("damage" in text or "destroy" in text)
        ):
            score += weight

    # Baseline: playable cards with reasonable stats are neutral-positive.
    if defense >= 3:
        score += 0.25
    if power >= 3:
        score += 0.25
    return score


class SideboardGuidePolicy:
    """Warm-start sideboard agent using Silver Age matchup heuristics."""

    def __init__(
        self,
        pool_by_id: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        self._pool_by_id = pool_by_id or {}

    def act(self, observation: str | dict[str, Any]) -> str:
        obs = json.loads(observation) if isinstance(observation, str) else observation
        avail = list(obs.get("availableActions", []))
        if not avail:
            return "finalize"

        opponent = str(obs.get("opponentHero", ""))
        archetype = classify_opponent_archetype(opponent)
        deck_size = int(obs.get("deckSize", 0))
        min_size = int(obs.get("minDeckSize", 40))
        max_size = int(obs.get("maxDeckSize", min_size))

        deck_cards = {
            str(c["id"]): int(c.get("count", 1))
            for c in obs.get("deck", [])
        }
        sb_cards = {
            str(c["id"]): int(c.get("count", 1))
            for c in obs.get("sideboard", [])
        }

        def _score(cid: str) -> float:
            meta = _card_meta(cid, self._pool_by_id)
            return score_card_for_archetype(cid, meta, archetype)

        move_in = [a for a in avail if a.startswith("move_to_deck:")]
        move_out = [a for a in avail if a.startswith("move_to_sideboard:")]

        if deck_size < min_size and move_in:
            best = max(
                move_in,
                key=lambda a: _score(a.split(":", 1)[1]),
            )
            return best

        if deck_size >= min_size and deck_size < max_size and move_in:
            best = max(
                move_in,
                key=lambda a: _score(a.split(":", 1)[1]),
            )
            return best

        if deck_size >= min_size and move_out and move_in:
            worst_deck = min(
                (cid for cid in deck_cards if deck_cards[cid] > 0),
                key=_score,
                default=None,
            )
            best_sb = max(
                (a.split(":", 1)[1] for a in move_in),
                key=_score,
                default=None,
            )
            if worst_deck and best_sb:
                if _score(best_sb) - _score(worst_deck) >= _SWAP_MARGIN:
                    out_action = f"move_to_sideboard:{worst_deck}"
                    if out_action in avail:
                        return out_action
                    in_action = f"move_to_deck:{best_sb}"
                    if in_action in avail:
                        return in_action

        if "finalize" in avail and deck_size >= min_size:
            return "finalize"

        if move_in:
            return max(move_in, key=lambda a: _score(a.split(":", 1)[1]))
        if move_out:
            return min(move_out, key=lambda a: _score(a.split(":", 1)[1]))
        return avail[0]
