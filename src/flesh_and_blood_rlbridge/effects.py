"""Card-effect trigger parsing for the Flesh and Blood simulator.

This mirrors the role of Talishar's per-card logic that reads/writes
``ClassState`` on triggers, but works generically off rules text. We only
parse a small, unambiguous vocabulary of triggered effects; anything we
recognize as a trigger but cannot model faithfully is reported as an
``unimplemented`` effect so the engine can surface a clear notice instead of
silently approximating it.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

from . import tokens as token_defs

# Match one or more {r} symbols, or a numeric cost.
_RESOURCE_RE = r"((?:\{r\})+|\d+)"

# Effect kinds the engine knows how to resolve faithfully.
SUPPORTED_EFFECTS = {
    "go_again",
    "damage",
    "arcane_damage",
    "power",
    "draw",
    "next_naa_go_again",
    "next_action_go_again",
    "next_attack_power",
    "banish_combo",
    "dominate",
    "intimidate",
    "overpower",
    "create_token",
    "create_banished",
    "create_in_hand",
    "banish_top",
    "banish_defending",
    "banish_graveyard",
    "banish_arsenal",
    "discard",
    "destroy_top",
    "destroy_arsenal",
    "destroy_hand",
    "gain_resources",
    "damage_floor",
    "prevent_damage",
    "ward",
    "opt",
    "mark",
    "put_soul",
    "reduce_defense",
    "clash",
    "amp",
    "reload",
    "unless_pay",
    "search",
    "boost",
    "reveal_top",
    "put_bottom",
    "enable_banish_play",
    "for_each",
    "transcend",
    "contract",
    "blood_debt",
    "attack",  # weapon swing placeholder
    "gain_gold",
    "return_gy_to_deck",
    "cost_reduction",
    "silence",
    "wager",
    "look_deck",
    "reveal_hand",
    "choose_card",
    "choose_mode",
    "modify_attack_power",
    "named_power_bonus",
    "stash_hand",
    "put_counter",
    "arcane_barrier",
    "steal_equipment",
    "steal_aura",
    "steal_token",
    "block_arsenal_play",
    "block_weapon_attacks",
    "block_power_gain",
    "reduce_next_power_gain",
    "block_hit_effects",
    "block_defense_reactions",
    "lose_gold",
    "destroy_item",
    "crowd_cheer",
    "crowd_boo",
    "play_as_instant",
    "grant_may_play",
    "return_gy_to_hand",
    "remove_counter",
    "opponent_cost_increase",
    "grant_draconic",
    "halve_base_power",
    "put_deck_top_arsenal",
    "put_hand_top",
    "retrieve_gy",
    "steal_ally",
    "put_item_in_arena",
    "put_arrow_arsenal",
    "galvanize",
    "modular_equip",
    "next_defense_bonus",
    "gain_life",
    "lose_life",
    "gain_action_point",
    "extra_weapon_attack",
    "cost_per_token",
    "next_ability_cost_reduction",
    "lose_all_abilities",
    "heave",
    "scrap",
    "additional_pay",
    "pitch_pay",
    "pitch_bonus",
    "equip_inventory",
    "extra_bow_activations",
    "lose_game",
    "schedule_end_phase",
    "extra_turn",
    "enable_gy_play",
    "fusion",
    "grant_hit_bonus",
    "hit_bonus_damage",
    "hit_rider",
    "freeze",
    "create_token_triple",
    "upkeep_or_destroy",
    "play_power_cap",
    "gy_to_bottom",
    "grant_light_block",
    "transform_equip",
    "transform_token",
    "counts_as_gold",
    "extra_attack_targets",
    "play_from_deck_top",
    "intellect_mod",
    "dagger_damage",
    "return_self_hand",
    "return_arsenal_hand",
    "turn_arsenal_face_up",
    "pitch_deck_top",
    "weapon_swing_cost_reduction",
    "taunt",
    "block_opponent_hit_effects",
    "block_gold_gain",
    "lose_phantasm",
    "prevention_reduction",
    "chain_defend",
    "turn_equipment_face_up",
    "name_card",
    "transform_hero",
    "grade_increase",
    "return_arena_tapped",
    "play_restriction",
    "random_banished_pick",
    "turn_banished_face",
    "damage_redirect",
    "inventory_to_hand",
    "block_pitch_color",
    "pitch_restriction",
    "banish_gy_variable",
    "banish_hand_play",
    "banish_self_play",
    "block_arcane_prevention",
    "reveal_named_hand",
    # New effects added to cover previously unimplemented cards
    "lose_life_per_hand_card",    # Vipox: lose {h} equal to opponent hand count
    "lose_life_per_dagger_hit",   # Stab Wound: lose X{h} per dagger chain hit
    "power_per_block",            # Tenacity: +X{p} per defending card on chain
    "limit_actions_next_turn",    # Red in the Ledger: cap opponent actions next turn
    "look_face_down",             # Fact-Finding Mission: peek at face-down card
    "set_next_instant_return_self",   # Gone in a Flash: next instant triggers return attack to hand
    "set_next_instant_return_aura",   # Blast to Oblivion: next instant triggers return target aura
    "reveal_for_blue_bonus",      # Attune with Cosmic Vibrations: reveal top, +3p/+3d if blue
    "reveal_top_draconic_power",  # Red Hot: reveal X=draconic links, +{p} per high-cost card
}

# Only mark as unimplemented when these appear and no specific parser matched.
_COMPLEX_MARKERS = (
    "as though",
    "for each",
    "if you do",
)

# Play-time conditions the environment can evaluate.
CONDITION_ALIASES = {
    "life_less": (
        r"if you have less \{h\} than an opposing hero",
        r"if you have less life than an opposing hero",
    ),
    "empty_deck": (r"if you have no cards in your deck",),
    "empty_arsenal": (
        r"if you have no cards in your arsenal",
        r"and you have no cards in your arsenal",
    ),
    "played_red_other": (r"if you(?:'ve| have) played another red card this turn",),
    "played_blue_other": (r"if you(?:'ve| have) played another blue card this turn",),
    "pitched_power_6": (
        r"if there is a card with 6 or more \{p\} in your pitch zone",
        r"if there is a card with 6 or more power in your pitch zone",
    ),
    "last_action_lightning": (
        r"if the last action card you played this turn was lightning",
    ),
    "controlled_seismic_surge": (
        r"if you(?:'ve| have) controlled a seismic surge token this turn",
    ),
    "combo_red": (r"if an? red attack action card was the last attack this combat chain",),
    "combo_yellow": (r"if an? yellow attack action card was the last attack this combat chain",),
    "combo_blue": (r"if an? blue attack action card was the last attack this combat chain",),
    "combo_draconic": (r"if a draconic attack was the last attack this combat chain",),
    "cheered_this_turn": (r"if you(?:'ve| have) been cheered this turn",),
    "less_gold_than_opponent": (
        r"if you have less \{g\} than them",
        r"if you have less \{g\} than an opposing hero",
    ),
    "more_gold_than_opponent": (
        r"if you have more \{g\} than them",
        r"if you have more \{g\} than an opposing hero",
    ),
    "scrapped_this_play": (r"if it scrapped a card", r"if this scrapped a card"),
    "discarded_power6_turn": (r"if the discarded card has 6 or more \{p\}",),
}

# Mandatory gates parsed from "Play this only if ..." (evaluated in environment.py).
PLAY_GATE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"play this only if you have (\d+) or more evos equipped", "evos_ge"),
    (r"play this only if you've been dealt damage this turn", "damage_taken_turn"),
    (r"play this only if you've boosted this turn", "boosted_turn"),
    (r"play this only if you control exactly (\d+) runechants", "runechants_eq"),
    (
        r"play this only if you've discarded a card with 6 or more \{p\} this turn",
        "discarded_power6_turn",
    ),
    (
        r"play this only if you've pitched a card with 6 or more \{p\} this turn",
        "pitched_power6_turn",
    ),
    (
        r"play this only if a card with 6 or more \{p\} has been put into your banished zone this turn",
        "banished_power6_turn",
    ),
    (
        r"play this only if you've played (\d+) or more cards with blood debt this turn",
        "blood_debt_played_ge",
    ),
    (
        r"play this only if there are (\d+) or more cards with blood debt in your banished zone",
        "blood_debt_banish_ge",
    ),
    (
        r"play this only if an illusionist attack action card you control has been destroyed by phantasm this turn",
        "phantasm_destroyed_turn",
    ),
    (r"play this only if you control (\d+) or more fealty tokens", "fealty_ge"),
    (r"play this only if you control (\d+) or more draconic chain links", "draconic_links_ge"),
    (
        r"play this only if you've attacked (\d+) or more times with weapons this turn",
        "weapon_attacks_ge",
    ),
    (r"play this only if you've wagered this chain link", "wagered_chain"),
    (
        r"play this only if an attack action card is defending this chain link",
        "defending_action_chain",
    ),
    (r"play this only if you've dealt \{p\} damage this turn", "dealt_damage_turn"),
    (
        r"play this only if a yellow card has been put into your soul this turn",
        "yellow_soul_turn",
    ),
)


@dataclass(frozen=True)
class Effect:
    kind: str          # one of SUPPORTED_EFFECTS or "unimplemented"
    amount: int = 0
    raw: str = ""
    max_cost: int = -1     # cost cap for next_attack_power (-1 = no cap)
    banish_name: str = ""  # graveyard card to banish for banish_combo / create_banished
    token_name: str = ""   # token to create
    target: str = "opponent"  # banish_top target: opponent | self
    playable_banished: bool = False  # created banished card playable this turn
    go_again: bool = False  # rider grants go again (banish_combo / compound)
    optional: bool = False  # "you may ..." -> player chooses whether to apply
    condition: str = ""    # play-time gate; empty = always applies

    @property
    def implemented(self) -> bool:
        return self.kind in SUPPORTED_EFFECTS


@dataclass(frozen=True)
class Trigger:
    when: str          # on_play | on_attack | on_hit | when_fused | on_leave | ...
    effect: Effect
    raw: str
    threshold: int = 0  # crush (>=) / surge (>) damage threshold


@dataclass(frozen=True)
class PlayModifier:
    """Inline bonus applied when a card is played (not a separate trigger step)."""

    power: int = 0
    defense: int = 0
    go_again: bool = False
    dominate: bool = False
    next_attack_power: int = 0
    next_attack_max_cost: int = -1
    next_action_go_again: int = 0
    condition: str = ""
    condition_arg: str = ""
    raw: str = ""


@dataclass(frozen=True)
class ActivatedAbility:
    """An activated ability on a permanent (equipment / weapon / hero)."""

    effect: Effect
    cost: int = 0                  # resource ({r}) cost
    uses_action_point: bool = True  # "Action" costs 1 AP; "Instant" costs 0
    once_per_turn: bool = False
    destroy_source: bool = False
    discard_source: bool = False
    counter_name: str = ""
    counter_cost: int = 0
    grants_go_again: bool = False
    raw: str = ""

    @property
    def implemented(self) -> bool:
        return self.effect.implemented


def _merge_split_clauses(body: str) -> str:
    """Rejoin clauses split by '.' that belong together (pitch-pay riders, etc.)."""
    merges = (
        (
            r"(you may pay up to (?:\{r\})+)\.\s*(create that many [^.]+)",
            r"\1, \2",
        ),
        (
            r"(you may remove a \w+ counter from an aura you control)\.\s*"
            r"(if you do,?\s*this gets \+\d+\{d\}[^.]*)",
            r"\1, \2",
        ),
        (
            r"(when this defends, you may remove a suspense counter from an aura you control)\.\s*"
            r"(if you do,?\s*gain (?:\{r\})+(?:\{r\})*)",
            r"\1, \2",
        ),
        (
            r"(put a face-up card from your arsenal on the bottom of your deck)\.\s*"
            r"(if you do,?\s*[^.]*)",
            r"\1, \2",
        ),
        (
            r"(you may pay (?:\{r\})+)\.\s*(if you do,?\s*it gets \+\d+\{d\}[^.]*)",
            r"\1, \2",
        ),
        (
            r"(when this is defended by \d+ or more cards,?\s*you may pay (?:\{r\})+)\.\s*"
            r"(if you do,?\s*it gets \+\d+\{p\}[^.]*)",
            r"\1, \2",
        ),
        (
            r"(you may put a card from your hand on the bottom of your deck)\.\s*"
            r"(if you do,?\s*draw a card)",
            r"\1, \2",
        ),
        (
            r"((?:when )?(?:this|it) defends(?: a \w+ attack)?,?\s*clash[^.]+)\.\s*"
            r"((?:the )?winner creates an? [^.]+ token)",
            r"\1, \2",
        ),
        (
            r"(when this defends,?\s*you may turn a face-down card with crush in your arsenal face-up)\.\s*"
            r"(if you do,?\s*put a \+\d+\{p\} counter on it)",
            r"\1, \2",
        ),
        (
            r"(when this defends,?\s*you may reveal an instant card from your hand)\.\s*"
            r"(if you do,?\s*deal \d+ arcane damage[^.]*)",
            r"\1, \2",
        ),
        (
            r"(when this defends,?\s*you may reveal an instant card from your hand)\.\s*"
            r"(if you do,?\s*create an? [^.]+ token)",
            r"\1, \2",
        ),
        (
            r"(when this defends,?\s*you may remove a gold counter from treasure island)\.\s*"
            r"(if you do and you are a thief,?\s*create a gold token)",
            r"\1, \2",
        ),
        (
            r"(when this defends,?\s*you may turn a card in a graveyard face-down)\.\s*"
            r"(if it'?s yellow,?\s*create a gold token)",
            r"\1, \2",
        ),
        (
            r"(whenever an arrow is put face-up into your arsenal from your deck,?\s*you may pay (?:\{r\})+)\.\s*"
            r"(if you do,?\s*put an aim counter on it)",
            r"\1, \2",
        ),
        (
            r"(whenever you beat chest,?\s*you may pay (?:\{r\})+ and destroy this)\.\s*"
            r"(if you do,?\s*draw a card)",
            r"\1, \2",
        ),
        (
            r"(whenever a weapon attack you control hits,?\s*you may pay (?:\{r\})+)\.\s*"
            r"(if you do,?\s*create an? [^.]+ token)",
            r"\1, \2",
        ),
        (
            r"(at the start of your turn, you may put a card from your hand into your soul)\.\s*"
            r"(if it'?s an illusionist card, create a spectral shield token)",
            r"\1, \2",
        ),
        (
            r"(whenever a lightning or elemental attack you control is defended by a card from hand,?\s*"
            r"you may destroy this)\.\s*(if you do,?\s*the attack gets \+(\d+)\{p\})",
            r"\1, \2",
        ),
    )
    for pattern, repl in merges:
        body = re.sub(pattern, repl, body, flags=re.I)
    return body


def _clean(text: str) -> str:
    body = str(text or "")
    body = body.replace("{br}", ". ").replace("**", "")
    body = body.replace("{i}", "").replace("{/i}", "")
    return _merge_split_clauses(body)


def _is_activated_ability_clause(clause: str) -> bool:
    return bool(
        re.search(
            r"(?:once per turn\s+)?(?:action|instant|attack reaction)\s*(?:[-—–]|(?:\{r\})+|\d+)",
            clause,
            re.I,
        )
    )


def _parse_dagger_damage_text(text: str, *, full_context: str = "") -> Effect | None:
    """Parse dagger-deals-damage effects (Blood Runs Deep, Hurl, Flick Knives, etc.)."""
    c = " ".join(text.lower().split())
    ctx = " ".join((full_context or text).lower().split())
    m = re.search(
        r"(each|target) dagger you control(?: that isn't on the active chain link)? "
        r"deals (\d+) damage to (?:target hero|the defending hero|them)",
        c,
    )
    if not m:
        return None
    amount = int(m.group(2))
    each = m.group(1) == "each"
    not_on_chain = (
        "isn't on the active chain link" in c or "not on the active chain link" in c
    )
    destroy = bool(re.search(r"destroy the daggers?", ctx))
    if "defending hero" in c:
        target = "defender"
    elif "to them" in c:
        target = "attacked"
    else:
        target = "opponent"
    riders: list[str] = []
    if each:
        riders.append("each")
    if not_on_chain:
        riders.append("not_on_chain")
    if destroy:
        riders.append("destroy")
    return Effect(
        "dagger_damage",
        amount,
        text.strip(),
        target=target,
        banish_name=":".join(riders),
    )


def _resource_amount(text: str) -> int:
    """Count {r} symbols or parse a numeric resource cost."""
    txt = str(text or "")
    if "{r}" in txt:
        return max(1, txt.count("{r}"))
    m = re.search(r"\d+", txt)
    return max(1, int(m.group(0))) if m else 1


def _extend_trigger_clause(body: str, clause: str) -> str:
    """Rejoin clauses split at '.' before 'If you do' riders (pitch pay, etc.)."""
    base = clause.strip().rstrip(".")
    if not base:
        return clause
    m = re.search(
        re.escape(base) + r"\.\s*(if you do,?\s*[^.]+(?:\.|$))",
        body,
        re.I,
    )
    if m:
        return m.group(0).strip().rstrip(".")
    return clause.strip()


def _parse_trigger_clause(clause: str, *, optional: bool = False) -> Effect | None:
    """Parse common trigger riders not covered by generic effect parsers."""
    c = " ".join(clause.lower().split())
    raw = clause.strip()
    if not c:
        return None

    opt = optional or "you may" in c

    m = re.search(
        r"you may pay ((?:\{r\})+|\d+)\.?\s*if you do,?\s*this gets \+(\d+)\{p\}",
        c,
    )
    if m:
        return Effect(
            "pitch_pay",
            _resource_amount(m.group(1)),
            raw,
            max_cost=int(m.group(2)),
            banish_name="power",
            optional=True,
        )

    if re.search(r"choose red, yellow, or blue", c):
        return Effect("block_pitch_color", raw=raw, target="opponent")

    if re.search(
        r"until the end of their next turn, they can't pitch or play cards with base cost 0",
        c,
    ):
        return Effect("pitch_restriction", banish_name="no_cost_zero", target="opponent", raw=raw)

    if re.search(
        r"it gets \+x\{p\}, where x is twice the number of cards in all pitch zones",
        c,
    ):
        return Effect("power", 0, banish_name="all_pitch_zones_x2", raw=raw)

    m = re.search(
        r"it gets \+(\d+)\{p\} for each hyper driver destroyed this way",
        c,
    )
    if m:
        return Effect(
            "power",
            int(m.group(1)),
            banish_name="hyper_drivers_destroyed",
            raw=raw,
        )

    if re.match(r"^it gets \+x\{p\}\.?$", c):
        return Effect("power", 0, banish_name="last_banish_count", raw=raw)

    if re.search(
        r"the next time they defend with 1 or more reaction cards this turn, "
        r"those cards get -1\{d\} while defending",
        c,
    ):
        return Effect(
            "reduce_defense",
            1,
            target="opponent",
            banish_name="next_defend_reactions",
            raw=raw,
        )

    if re.search(
        r"the next time they defend with 1 or more equipment this turn, "
        r"those equipment get -1\{d\} while defend",
        c,
    ):
        return Effect(
            "reduce_defense",
            1,
            target="opponent",
            banish_name="next_defend_equipment",
            raw=raw,
        )

    if re.search(
        r"the next time they defend with 1 or more attack action cards this turn, "
        r"those cards get -1\{d\} while",
        c,
    ):
        return Effect(
            "reduce_defense",
            1,
            target="opponent",
            banish_name="next_defend_actions",
            raw=raw,
        )

    if re.search(
        r"you may put a yellow card from a graveyard on the bottom of its owner's deck",
        c,
    ):
        rider = "yellow_any_gy"
        if re.search(r"if you do,?\s*create a gold token", c):
            rider += ":create_gold"
        return Effect("put_bottom", banish_name=rider, optional=opt, raw=raw)

    if re.search(
        r"you may deal that much damage to another ally controlled by the same hero",
        c,
    ):
        return Effect("damage", 0, optional=True, banish_name="cleave", raw=raw)

    if re.search(r"\{u\} all cogs you control", c):
        return Effect("destroy_item", banish_name="all_cogs", raw=raw)

    if re.search(
        r"if a draconic attack was the last attack this combat chain, banish this\.?\s*"
        r"if you do,?\s*you may play it this turn",
        c,
    ):
        return Effect(
            "banish_self_play",
            optional=True,
            condition="combo_draconic",
            raw=raw,
        )

    if re.match(r"^banish this\.?$", c):
        return Effect(
            "banish_self_play",
            optional=True,
            playable_banished=True,
            raw=raw,
        )

    if re.match(
        r"^banish this\.?\s*if you do,?\s*you may play it this turn\.?$",
        c,
    ):
        return Effect(
            "banish_self_play",
            optional=True,
            playable_banished=True,
            raw=raw,
        )

    if re.search(
        r"action card effects you control that deal arcane damage, instead deal that much arcane damage plus 1",
        c,
    ):
        return Effect("amp", 1, banish_name="action_arcane_turn", raw=raw)

    if re.search(
        r"until end of turn if an attack would deal damage, instead it deals that much damage plus 1",
        c,
    ):
        return Effect("amp", 1, banish_name="attack_damage_turn", raw=raw)

    if re.search(r"deal damage to them equal to the number of equipment they control", c):
        return Effect("damage", 0, banish_name="equipment_count", target="opponent", raw=raw)

    if re.match(r"^name another card\.?$", c):
        return Effect("name_card", raw=raw)

    m = re.search(
        r"attack action cards with that name get \+(\d+)\{p\} this combat chain",
        c,
    )
    if m:
        return Effect(
            "named_power_bonus",
            int(m.group(1)),
            banish_name="this_chain",
            raw=raw,
        )

    if re.search(r"put (?:it|this) into (?:its owner'?s|your) hand", c):
        return Effect("return_self_hand", raw=raw)

    if re.match(r"^name a card\.?$", c):
        return Effect("name_card", raw=raw)

    if re.search(r"look at a face-down card in their (?:arsenal|equipment)", c):
        return Effect("look_face_down", raw=raw, optional=True)

    if re.search(
        r"put a mechanologist item from your hand into the arena with cost less than or equal to the number of times you'?ve? boosted",
        c,
    ):
        return Effect("put_item_in_arena", 0, optional=True, banish_name="boost_count_cost", raw=raw)

    if re.search(
        r"the next time you play an instant card this chain link,?\s*you may return this to its owner'?s hand",
        c,
    ):
        return Effect("set_next_instant_return_self", optional=True, raw=raw)

    if re.search(
        r"the next time you play an instant card this chain link,?\s*you may return target aura",
        c,
    ):
        return Effect("set_next_instant_return_aura", optional=True, raw=raw)

    m = re.search(
        r"gain control of an item with cost (\d+) or less they control\.?\s*"
        r"otherwise,?\s*draw a card",
        c,
    )
    if m:
        return Effect(
            "steal_equipment",
            int(m.group(1)),
            target="opponent",
            banish_name="or_draw",
            raw=raw,
        )

    m = re.search(r"gain control of an item with cost (\d+) or less they control", c)
    if m:
        return Effect(
            "steal_equipment",
            int(m.group(1)),
            target="opponent",
            banish_name="item",
            raw=raw,
        )

    if re.search(r"\{u\} your hero", c):
        return Effect("destroy_item", banish_name="your_hero", raw=raw)

    if re.search(
        r"they can't prevent arcane damage from sources you control this turn",
        c,
    ):
        return Effect("block_arcane_prevention", target="opponent", raw=raw)

    m = re.search(
        r"you may banish an attack action card from your hand with cost less than "
        r"the number of draconic chain links you control",
        c,
    )
    if m:
        c_ext = " ".join(raw.lower().split())
        power = 0
        pm = re.search(r"if you do,?\s*it gets \+(\d+)\{p\}", c_ext)
        if pm:
            power = int(pm.group(1))
        go_again = bool(re.search(r"if you do,?\s*it gets go again", c_ext))
        return Effect(
            "banish_hand_play",
            power,
            optional=True,
            banish_name="attack:draconic_links",
            go_again=go_again,
            playable_banished=True,
            raw=raw,
        )

    if re.search(r"you may reveal any number of crouching tigers from your hand", c):
        return Effect(
            "reveal_named_hand",
            optional=True,
            banish_name="Crouching Tiger:1:go_again,2:power:1,3:draw:1",
            raw=raw,
        )

    create = _parse_create_banish(c)
    if create is not None:
        return create

    m = re.search(r"^when this chain link resolves,?\s*(.+)$", c)
    if m:
        rest = m.group(1).strip(" .")
        dm = re.search(r"draw (a|an|one|\d+) cards?", rest)
        if dm:
            n = 1 if dm.group(1) in ("a", "an", "one") else int(dm.group(1))
            return Effect("draw", n, raw=raw, optional=opt)

    m = re.search(r"draw (a|an|one|\d+) cards?", c)
    if m:
        n = 1 if m.group(1) in ("a", "an", "one") else int(m.group(1))
        return Effect("draw", n, raw=raw, optional=opt)

    m = re.search(r"they discard (\d+|a|an|one) cards?", c)
    if m:
        n = 1 if m.group(1) in ("a", "an", "one") else int(m.group(1))
        return Effect("discard", n, target="opponent", raw=raw)

    return None


def _parse_pitch_pay(clause: str, *, optional: bool = True) -> Effect | None:
    """Optional pitch-from-hand costs on triggers (CR: pay {r} = pitch cards)."""
    c = " ".join(clause.lower().split())
    if not c:
        return None

    m = re.search(rf"you may pay up to {_RESOURCE_RE}", c)
    if m:
        max_pay = _resource_amount(m.group(1))
        rest = c[m.end():]
        tm = re.search(r"create that many (.+?) tokens?", rest)
        if tm:
            return Effect(
                "pitch_pay",
                max_pay,
                token_name=tm.group(1).strip(),
                optional=optional,
                raw=clause.strip(),
            )
        if "intimidate" in rest:
            return Effect(
                "pitch_pay",
                max_pay,
                banish_name="intimidate",
                optional=optional,
                raw=clause.strip(),
            )
        return Effect(
            "pitch_pay",
            max_pay,
            optional=optional,
            raw=clause.strip(),
        )

    m = re.search(
        r"you may pay ((?:\{r\})+|\d+)(?:\.|,)\s*if you do,?\s*it gets \+(\d+)\{d\}",
        c,
    )
    if m:
        return Effect(
            "pitch_pay",
            _resource_amount(m.group(1)),
            clause.strip(),
            max_cost=int(m.group(2)),
            banish_name="defense",
            optional=optional,
        )

    m = re.search(
        r"you may pay ((?:\{r\})+|\d+)\.?\s*if you do,?\s*the attack gets \+(\d+)\{p\} and \*\*overpower\*\*",
        c,
    )
    if m:
        return Effect(
            "pitch_pay",
            _resource_amount(m.group(1)),
            clause.strip(),
            max_cost=int(m.group(2)),
            banish_name="power_overpower",
            optional=optional,
        )

    m = re.search(
        r"when this is defended by \d+ or more cards,?\s*you may pay ((?:\{r\})+|\d+)(?:\.|,)\s*"
        r"if you do,?\s*it gets \+(\d+)\{p\}",
        c,
    )
    if m:
        return Effect(
            "pitch_pay",
            _resource_amount(m.group(1)),
            clause.strip(),
            max_cost=int(m.group(2)),
            banish_name="power",
            optional=optional,
        )

    m = re.search(r"you may pay ((?:\{r\})+|\d+) and destroy this[,.\s]*if you do,?\s*draw", c)
    if m:
        return Effect(
            "pitch_pay",
            _resource_amount(m.group(1)),
            banish_name="destroy_draw",
            optional=optional,
            raw=clause.strip(),
        )

    m = re.search(r"you may pay ((?:\{r\})+|\d+)[,.\s]*if you do,?\s*put an aim counter on it", c)
    if m:
        return Effect(
            "pitch_pay",
            _resource_amount(m.group(1)),
            banish_name="aim_counter",
            optional=optional,
            raw=clause.strip(),
        )

    m = re.search(
        r"you may pay ((?:\{r\})+|\d+)\.?\s*if you do,?\s*gain ((?:\{r\})+|\d+)",
        c,
    )
    if m:
        gain = _resource_amount(m.group(2))
        return Effect(
            "pitch_pay",
            _resource_amount(m.group(1)),
            banish_name=f"gain_resources:{gain}",
            optional=optional,
            raw=clause.strip(),
        )

    m = re.search(
        r"you may pay ((?:\{r\})+|\d+)\.?\s*if you do,?\s*create an? (.+?) token",
        c,
    )
    if m:
        return Effect(
            "pitch_pay",
            _resource_amount(m.group(1)),
            token_name=m.group(2).strip(),
            optional=optional,
            raw=clause.strip(),
        )

    m = re.search(
        r"you may pay ((?:\{r\})+|\d+)\.?\s*if you do,?\s*choose (.+?)\.?\s*this gets the chosen name",
        c,
    )
    if m:
        return Effect(
            "pitch_pay",
            _resource_amount(m.group(1)),
            banish_name="choose_name:" + m.group(2).strip(),
            optional=optional,
            raw=clause.strip(),
        )

    m = re.search(
        r"you may pay ((?:\{r\})+|\d+)\.?\s*if you do,?\s*(?:it|this) gets \+(\d+)\{p\}",
        c,
    )
    if m:
        return Effect(
            "pitch_pay",
            _resource_amount(m.group(1)),
            clause.strip(),
            max_cost=int(m.group(2)),
            banish_name="power",
            optional=optional,
        )

    m = re.search(
        r"you may pay ((?:\{r\})+|\d+)\.?\s*if you do,?\s*this gets \+(\d+)\{p\}",
        c,
    )
    if m:
        return Effect(
            "pitch_pay",
            _resource_amount(m.group(1)),
            clause.strip(),
            max_cost=int(m.group(2)),
            banish_name="power",
            optional=optional,
        )

    m = re.search(r"you may pay ((?:\{r\})+|\d+)\.?\s*if you do", c)
    if m and "choose" not in c and "this gets" not in c:
        return Effect(
            "pitch_pay",
            _resource_amount(m.group(1)),
            optional=optional,
            raw=clause.strip(),
        )

    return None


def _detect_condition(clause: str) -> tuple[str, str]:
    """Return (condition_key, remainder) from a leading conditional clause."""
    low = clause.lower()
    combo = token_defs.parse_combo_condition(clause)
    if combo:
        m = re.search(CONDITION_ALIASES[combo][0], low)
        if m:
            return combo, clause[m.end():].lstrip(" ,.")
    for key, patterns in CONDITION_ALIASES.items():
        for pat in patterns:
            m = re.search(pat, low)
            if not m:
                continue
            remainder = clause[m.end():].lstrip(" ,.")
            return key, remainder
    return "", clause


def _singular_card_name(name: str) -> str:
    """Normalize plural create-in-hand names (Crouching Tigers -> Crouching Tiger)."""
    n = " ".join(str(name or "").strip().split())
    if n.endswith("s") and not n.endswith("ss") and len(n) > 3:
        return n[:-1]
    return n


def _parse_create_in_hand(clause: str) -> Effect | None:
    c = " ".join(clause.lower().split())
    m = re.search(r"create (\d+) (.+?) in your hand", c)
    if m:
        return Effect(
            "create_in_hand",
            int(m.group(1)),
            banish_name=_singular_card_name(m.group(2).strip()),
            target="self",
            raw=clause.strip(),
        )
    m = re.search(r"create a (.+?) in your hand", c)
    if m:
        return Effect(
            "create_in_hand",
            banish_name=m.group(1).strip(),
            target="self",
            raw=clause.strip(),
        )
    return None


def _parse_create_banish(clause: str) -> Effect | None:
    """Parse create-token / create-banished / banish-top clauses."""
    hand = _parse_create_in_hand(clause)
    if hand is not None:
        return hand
    c = " ".join(clause.lower().split())
    m = re.search(r"create a crouching tiger in your banished zone", c)
    if m:
        return Effect(
            "create_banished",
            banish_name="Crouching Tiger",
            playable_banished="you may play it this turn" in c or "you may play them this turn" in c,
            raw=clause.strip(),
        )
    m = re.search(r"create (\d+) (.+?)s in your banished zone", c)
    if m:
        return Effect(
            "create_banished",
            int(m.group(1)),
            clause.strip(),
            banish_name=m.group(2).strip(),
            playable_banished="you may play them this turn" in c,
        )
    m = re.search(
        r"create x (.+?)s in your banished zone, where x is the number of (.+?) you control",
        c,
    )
    if m:
        return Effect(
            "create_banished",
            banish_name=m.group(1).strip(),
            token_name=m.group(2).strip(),
            playable_banished="you may play them this turn" in c,
            raw=clause.strip(),
        )
    m = re.search(
        r"create x (.+?) tokens under target hero'?s control",
        c,
    )
    if m:
        return Effect(
            "create_token",
            token_name=m.group(1).strip(),
            target="opponent",
            banish_name="pitch_value",
            raw=clause.strip(),
        )
    m = re.search(r"create[ds]? (\d+) (.+?) tokens?(?: under (?:target|their control))?", c)
    if m:
        target = "opponent" if "under" in c and ("target" in c or "their" in c) else "self"
        return Effect(
            "create_token",
            int(m.group(1)),
            clause.strip(),
            token_name=m.group(2).strip(),
            target=target,
        )
    m = re.search(r"create[ds]? a (.+?) token", c)
    if m:
        name = m.group(1).strip()
        if name and " in your banished zone" not in name and " under their control" not in name:
            return Effect("create_token", token_name=name, raw=clause.strip())
    if re.search(r"create[ds]? an? (.+?) token under their control", c):
        m = re.search(r"create[ds]? an? (.+?) token under their control", c)
        return Effect("create_token", token_name=m.group(1).strip(), target="opponent", raw=clause.strip())
    if re.search(r"banish the top card of their deck and a defending card", c):
        return Effect("banish_defending", raw=clause.strip())
    if re.search(r"banish the top (\d+) cards? of (?:their|target) deck", c):
        m = re.search(r"banish the top (\d+) cards? of (?:their|target) deck", c)
        return Effect("banish_top", int(m.group(1)), clause.strip(), target="opponent")
    if re.search(r"banish the top (\d+) cards? of your deck", c):
        m = re.search(r"banish the top (\d+) cards? of your deck", c)
        return Effect("banish_top", int(m.group(1)), clause.strip(), target="self")
    if re.search(r"banish a card in their arsenal", c):
        return Effect("banish_arsenal", raw=clause.strip(), target="opponent")
    if re.search(r"banish the top card of (?:their|target opposing hero'?s?|the defending hero'?s?) deck", c):
        return Effect("banish_top", target="opponent", raw=clause.strip())
    if re.search(r"banish the top card of your deck", c):
        return Effect("banish_top", target="self", raw=clause.strip())
    if re.search(r"destroy the top card of (?:their|target) deck", c):
        return Effect("destroy_top", target="opponent", raw=clause.strip())
    if re.search(r"destroy the top card of your deck", c):
        return Effect("destroy_top", target="self", raw=clause.strip())
    if re.search(r"destroy all cards in their arsenal", c):
        return Effect("destroy_arsenal", target="opponent", raw=clause.strip())
    if re.search(r"destroy a card in their arsenal", c):
        return Effect("destroy_arsenal", target="opponent", amount=1, raw=clause.strip())
    if re.search(r"banish a card from their graveyard", c):
        return Effect("banish_graveyard", target="opponent", amount=1, raw=clause.strip())
    return None


def _parse_grant_may_play(
    c: str, raw: str, optional: bool, condition: str
) -> Effect | None:
    """Parse 'you may play X from Y as though it were an instant' style grants."""

    def g(**kw) -> Effect:
        opt = kw.pop("optional", optional)
        cond = kw.pop("condition", condition)
        text = kw.pop("raw", raw)
        kind = kw.pop("kind", "grant_may_play")
        amount = kw.pop("amount", 0)
        return Effect(kind, amount, text, optional=opt, condition=cond, **kw)

    if re.search(
        r"you may play a non-attack action card this chain link as though it were an instant",
        c,
    ):
        return g(target="hand", banish_name="non-attack", condition="chain_only")

    if re.search(r"you may play auras this turn as though they were instants", c):
        return g(target="hand", banish_name="aura")

    if re.search(
        r"if it's not your turn, you may play blue non-attack action cards from your arsenal as though they were instants",
        c,
    ):
        return g(target="arsenal", amount=3, banish_name="non-attack", condition="not_own_turn")

    if re.search(r"you may play evos from your banished zone", c):
        return g(target="banished", banish_name="evo", condition="static")

    if re.search(
        r"you may play blue cards from that hero's banished zone without paying their \{r\} cost",
        c,
    ):
        return g(target="opponent_banished", amount=3, condition="free_cost")

    if re.search(r"you may play face-up cards from their arsenal", c):
        return g(target="opponent_arsenal", banish_name="face_up")

    m = re.search(
        r"you may play your next (\w+) non-attack action card this turn as though it were an instant",
        c,
    )
    if m:
        return g(target="hand", banish_name="non-attack", token_name=m.group(1).lower())

    if re.search(r"play your next .+ as though it were an instant", c):
        return g(target="hand", banish_name="non-attack")

    m = re.search(r"you may play (.+?) as though it were an instant", c)
    if m:
        return g(target="hand", banish_name=m.group(1).strip().lower(), condition="static")

    return None


def _parse_extended_clause(clause: str, *, optional: bool, condition: str) -> Effect | None:
    """Parse common FAB effect phrases across all unimplemented categories."""
    c = " ".join(clause.lower().split())
    raw = clause.strip()

    def eff(kind: str, amount: int = 0, **kw) -> Effect:
        opt = kw.pop("optional", optional)
        cond = kw.pop("condition", condition)
        text = kw.pop("raw", raw)
        return Effect(kind, amount, text, optional=opt, condition=cond, **kw)

    if c == "attack" or c.startswith("attack ") or re.match(r"^attack\"?\.?$", c):
        return eff("attack")

    dagger = _parse_dagger_damage_text(clause, full_context=clause)
    if dagger is not None:
        return Effect(
            dagger.kind,
            dagger.amount,
            dagger.raw,
            target=dagger.target,
            banish_name=dagger.banish_name,
            optional=optional,
            condition=condition,
        )

    m = re.search(r"^opt (\d+)", c)
    if m:
        return eff("opt", int(m.group(1)))

    if re.search(r"^amp (\d+)$", c):
        return eff("amp", int(re.search(r"(\d+)", c).group(1)))

    if re.search(r"\breload\b", c):
        return eff("stash_hand", optional=True)

    if re.search(r"\bwager\b", c):
        m = re.search(r"wager an? (.+?) tokens?", c) or re.search(r"wager a (.+?) token", c)
        token = m.group(1).strip() if m else "wager"
        return eff("wager", token_name=token, optional=optional or "you may" in c)

    if re.search(r"lose and can't gain", c) or re.search(r"can't gain or have attack actions granted", c):
        return eff("silence", target="opponent")

    m = re.search(r"gain (\d+)\{g\}", c)
    if m:
        target = "opponent" if re.search(r"they gain \d+\{g\}", c) else "self"
        return eff("gain_gold", int(m.group(1)), target=target)

    m = re.search(r"put (.+?) from your graveyard on top of your deck", c)
    if m:
        return eff("return_gy_to_deck", banish_name=m.group(1).strip())

    m = re.search(r"put (.+?) from your graveyard on (?:the )?bottom of your deck", c)
    if m:
        return eff("return_gy_to_deck", banish_name=m.group(1).strip(), target="bottom")

    m = re.search(r"look at the top (\d+) cards? of your deck", c)
    if m:
        return eff("look_deck", int(m.group(1)))

    m = re.search(r"costs? ((?:\{r\})+|\d+) less to play", c)
    if m:
        return eff("cost_reduction", _resource_amount(m.group(1)))

    if re.search(r"charge your (?:hero'?s )?soul", c):
        return eff("put_soul", optional=True)

    if re.search(r"\boverpower\b", c):
        return eff("overpower")

    if re.search(r"mark them", c):
        return eff("mark")

    if re.search(r"put it into your soul", c) or re.search(r"charged to your soul", c):
        return eff("put_soul")

    if re.search(r"put it into your hero'?s soul", c):
        return eff("put_soul")

    if re.search(r"put a card from your hand into your soul", c):
        return eff("put_soul", optional=optional or "you may" in c)

    m = re.search(r"put a (red|yellow|blue) card from your hand into your soul", c)
    if m:
        return eff("put_soul", optional=optional or "you may" in c, banish_name=m.group(1))

    if re.search(r"put an item with cost 0 or 1 from your hand into the arena", c):
        return eff("put_item_in_arena", 1, optional=optional or "you may" in c)

    m = re.search(r"put an item with cost (\d+) or less from your hand into the arena", c)
    if m:
        return eff("put_item_in_arena", int(m.group(1)), optional=optional or "you may" in c)

    if re.search(r"put an item with cost 0 or 1 from any banished zone into the arena", c):
        return eff("put_item_in_arena", 1, optional=optional or "you may" in c, target="banished")

    if re.search(r"put an arrow card from your hand face-up into your arsenal", c):
        return eff("put_arrow_arsenal", optional=optional or "you may" in c)

    if re.search(r"put an arrow from your hand face-up into your arsenal", c):
        return eff("put_arrow_arsenal", optional=optional or "you may" in c)

    if re.search(r"equip this to another equipment zone", c):
        return eff("modular_equip")

    m = re.search(r"this gets \+(\d+)\{d\} until end of turn", c)
    if m and "destroy an item" in c:
        return eff("galvanize", int(m.group(1)), optional=optional or "you may" in c)

    m = re.search(r"^this gets \+(\d+)\{d\}\.?$", c)
    if m:
        return eff("next_defense_bonus", int(m.group(1)))

    m = re.search(r"the next action card you defend with this turn gets \+(\d+)\{d\}", c)
    if m:
        return eff("next_defense_bonus", int(m.group(1)))

    m = re.search(r"gain (\d+)\{h\}", c)
    if m:
        return eff("gain_life", int(m.group(1)))

    m = re.search(r"your hero gets \+(\d+)(?:\{i\})?(?: until end of turn| this turn)?\.?$", c)
    if m:
        return eff("intellect_mod", int(m.group(1)))

    m = re.search(r"(?:they|that hero) lose(?:s)? (\d+)\{h\}", c)
    if m:
        return eff("lose_life", int(m.group(1)), target="opponent")

    m = re.search(r"(?:they|that hero) lose(?:s)? (\d+)\{g\}", c)
    if m:
        return eff("lose_gold", int(m.group(1)), target="opponent")

    if re.search(r"they lose \{h\} equal to the number of cards in their hand", c):
        return eff("lose_life_per_hand_card", target="opponent")

    m = re.search(r"they lose (?:x\{h\}|\{h\}), where x is the number of times a dagger has hit this combat chain", c)
    if m:
        return eff("lose_life_per_dagger_hit", target="opponent")

    m = re.search(r"it gets \+x\{p\},?\s+where x is the number of defending cards on the combat chain", c)
    if m:
        return eff("power_per_block")

    if re.search(r"they can'?t play or activate more than 1 action", c):
        return eff("limit_actions_next_turn", target="opponent")

    m = re.search(
        r"you may attack with it an additional time this turn",
        c,
    )
    if m:
        return eff("extra_weapon_attack", optional=True)

    if re.search(r"you may attack an additional time with this weapon this turn", c):
        return eff("extra_weapon_attack", optional=True)

    if re.search(
        r"put a card from your hand face-down into your arsenal",
        c,
    ):
        return eff("stash_hand", optional=optional or "you may" in c)

    if re.search(
        r"create runechant tokens equal to the number of non-attack action cards you've played this turn",
        c,
    ):
        return eff(
            "create_token",
            0,
            token_name="runechant",
            banish_name="naa_played_this_turn",
        )

    if re.search(r"gain \{r\} equal to half the number rolled, rounded down", c):
        return eff("gain_resources", 0, banish_name="roll_d6_half")

    m = re.search(r"gain (\d+) action points?", c)
    if m:
        return eff("gain_action_point", int(m.group(1)))

    m = re.search(r"this costs ((?:\{r\})+|\d+) less to play for each (\w+) you control", c)
    if m:
        txt = m.group(1)
        per = txt.count("{r}") if "{r}" in txt else int(re.search(r"\d+", txt).group(0))
        return eff("cost_per_token", per, token_name=m.group(2).strip())

    if re.search(r"the next ability you activate this turn costs \{r\} less", c):
        return eff("next_ability_cost_reduction", 1)

    if re.search(
        r"(?:that hero |they )?loses? all abilities until the end of their next turn",
        c,
    ):
        return eff("lose_all_abilities", target="opponent")

    m = re.search(r"target hero discards (\d+|two|three|a|an|one) cards?", c)
    if m:
        word = m.group(1).lower()
        n = {"two": 2, "three": 3, "a": 1, "an": 1, "one": 1}.get(word)
        if n is None:
            n = int(word)
        return eff("discard", n, target="opponent")

    m = re.search(
        r"the next time you would be dealt \{p\} damage this turn, prevent (\d+) damage that source would deal",
        c,
    )
    if m:
        return eff("prevent_damage", int(m.group(1)), condition="power_damage", target="self")

    m = re.search(
        r"the next time you would be dealt (\d+) or less damage this turn, prevent it",
        c,
    )
    if m:
        return eff("prevent_damage", int(m.group(1)), condition="damage_le", target="self")

    m = re.search(
        r"the next time you would be dealt damage this turn, prevent (\d+) of that damage",
        c,
    )
    if m:
        return eff("prevent_damage", int(m.group(1)), target="self")

    m = re.search(
        r"prevent the next (\d+) damage that would be dealt to you this turn(?: by a shadow source)?",
        c,
    )
    if m:
        cond = "shadow" if "shadow source" in c else ""
        return eff("prevent_damage", int(m.group(1)), condition=cond, target="self")

    m = re.search(
        r"the next (\d+) times you would be dealt damage this turn, prevent (\d+) of that damage",
        c,
    )
    if m:
        return eff(
            "prevent_damage",
            int(m.group(2)),
            condition=f"per_hit:{int(m.group(1))}",
            target="self",
        )

    m = re.search(
        r"the next time a shadow source would deal damage this turn, prevent (\d+) of that damage",
        c,
    )
    if m:
        return eff("prevent_damage", int(m.group(1)), condition="shadow", target="self")

    m = re.search(
        r"they can't play attack action cards with (\d+) or less base \{p\}",
        c,
    )
    if m:
        return eff("play_power_cap", int(m.group(1)), target="opponent")

    if re.search(
        r"if this would be put into your graveyard from anywhere, instead put it on the bottom of your deck",
        c,
    ):
        return eff("gy_to_bottom")

    if re.search(
        r"if an action or instant card you control would deal arcane damage this turn, instead it deals that much plus 1",
        c,
    ):
        return eff("amp", 1)

    m = re.search(
        r"if you have a base (head|chest|arms|legs) equipped, transform it and x hyper drivers you control into this, then equip this",
        c,
    )
    if m:
        return eff("transform_equip", banish_name=f"{m.group(1)}:hyper_drivers")

    m = re.search(
        r"if you have a base (head|chest|arms|legs) equipped, transform it into this, then equip this",
        c,
    )
    if m:
        return eff("transform_equip", banish_name=m.group(1))

    if re.search(
        r"if you do, the next time you would be dealt damage this turn, prevent twice x of that damage",
        c,
    ):
        return eff(
            "prevent_damage",
            0,
            banish_name="twice_x",
            condition="per_hit:1",
            target="self",
        )

    if re.search(r"this counts as a gold", c):
        return eff("counts_as_gold")

    if re.search(
        r"transform (?:up to )?(?:\d+|one|a|an) ash you control into an aether ashwings?",
        c,
    ) or re.search(
        r"transform target ash you control into an aether ashwings?",
        c,
    ):
        return eff("transform_token", banish_name="ash", token_name="Aether Ashwing")

    if re.search(
        r"the next attack action card with crush you play this turn may attack an additional hero",
        c,
    ):
        return eff("extra_attack_targets", 1, banish_name="crush_next_attack")

    if re.search(
        r"runechants you control get spellvoid (\d+) this turn",
        c,
    ):
        m = re.search(r"spellvoid (\d+)", c)
        return eff(
            "arcane_barrier",
            int(m.group(1)) if m else 1,
            banish_name="per_runechant",
        )

    if re.search(
        r"it gets \+x\{p\}, where x is the number of gold you control",
        c,
    ):
        return eff("power", 0, banish_name="gold_count")

    if re.search(r"they reveal a card from their hand", c):
        return eff("reveal_hand", 1, target="opponent")

    if re.search(r"the defending hero reveals their hand", c):
        return eff("reveal_hand", 0, target="opponent", banish_name="store_revealed")

    if re.search(r"you may look at the defending hero'?s hand", c):
        return eff("reveal_hand", 0, target="opponent", optional=True)

    if re.search(
        r"remove all steam counters from an equipment, item, or weapon they control",
        c,
    ):
        return eff(
            "remove_counter",
            0,
            token_name="steam",
            target="opponent",
            banish_name="all_on_one",
        )

    if re.search(
        r"remove all steam counters from up to x equipment, items, and/or weapons they control",
        c,
    ):
        return eff(
            "remove_counter",
            0,
            token_name="steam",
            target="opponent",
            banish_name="evo_count",
        )

    m = re.search(
        r"once per turn, you may play a (\w+) item with cost 0 or 1 from the top of your deck as though it were an instant",
        c,
    )
    if m:
        return eff("play_from_deck_top", 1, banish_name=m.group(1).lower(), optional=True)

    m = re.search(
        r"you may remove a (\w+) counter from an aura you control(?:\.|,)\s*"
        r"if you do,?\s*this gets \+(\d+)\{d\}",
        c,
    )
    if m:
        return eff(
            "remove_counter",
            int(m.group(2)),
            token_name=m.group(1).strip(),
            banish_name="defense",
            optional=optional or "you may" in c,
        )

    m = re.search(r"heave (\d+)", c)
    if m:
        return eff("heave", int(m.group(1)), optional=True)

    if re.search(r"\bscrap\b", c) and "scrapped" not in c:
        return eff("scrap", optional=True)

    m = re.search(r"as an additional cost to play this,?\s*you may pay ((?:\{r\})+|\d+)", c)
    if m:
        txt = m.group(1)
        if "{r}" in txt:
            return eff("pitch_pay", _resource_amount(txt), optional=True)
        return eff("additional_pay", int(txt), optional=True)

    m = re.search(r"whenever an attack hits a hero this turn,?\s*it deals (\d+) damage", c)
    if m:
        return eff("grant_hit_bonus", int(m.group(1)))

    if re.search(r"whenever an attack hits a hero this turn", c):
        m = re.search(r"it deals (\d+) damage", c)
        if m:
            return eff("grant_hit_bonus", int(m.group(1)))

    if re.search(r"create an agility, might, and vigor token", c):
        return eff("create_token_triple")

    if re.search(r"transcend", c):
        return eff("transcend")

    if re.search(r"\bcontract\b", c):
        return eff("contract")

    if re.search(r"blood debt", c):
        return eff("blood_debt")

    if re.search(r"\bboost\b", c):
        return eff("boost")

    m = re.search(r"prevent the next (\d+) (?:arcane )?damage", c)
    if m:
        return eff("prevent_damage", int(m.group(1)))

    m = re.search(r"draw (a|an|one|\d+) cards?", c)
    if m:
        n = 1 if m.group(1) in ("a", "an", "one") else int(m.group(1))
        return eff("draw", n)

    m = re.search(r"ward (\d+)", c)
    if m:
        return eff("ward", int(m.group(1)))

    m = re.search(r"gain (\d+) action points?", c)
    if m:
        return eff("go_again", int(m.group(1)))

    m = re.search(r"gain ((?:\{r\})+|\d+)", c)
    if not m:
        m = re.search(r"gain (\d+)\s*(?:\{r\}|resources?)", c)
    if m:
        txt = m.group(1)
        n = txt.count("{r}") if "{r}" in txt else int(re.search(r"\d+", txt).group(0))
        return eff("gain_resources", n)

    m = re.search(r"they discard (\d+|a|an|one) cards?", c)
    if m:
        n = 1 if m.group(1) in ("a", "an", "one") else int(m.group(1))
        return eff("discard", n, target="opponent")

    m = re.search(r"discard (\d+|a|an|one|a random) cards?", c)
    if m and "they discard" not in c:
        n = 1 if m.group(1) in ("a", "an", "one", "a random") else int(m.group(1))
        target = "self" if "your" in c[:20] else "opponent"
        return eff("discard", n, target=target)

    m = re.search(r"destroy an? (.+?) they control", c)
    if m:
        if "equipment" in m.group(1):
            return eff("reduce_defense", 1, target="opponent")
        if "item" in m.group(1) or "aura" in m.group(1):
            return eff("destroy_arsenal", target="opponent", amount=1)

    m = re.search(r"put a -(\d+)\{d\} counter on an equipment they control", c)
    if m:
        return eff("reduce_defense", int(m.group(1)), target="opponent")

    if re.search(r"put a steam counter on", c):
        return eff("put_counter", token_name="steam")

    if re.search(r"if there are no steam counters on this, put a steam counter on it", c):
        return eff("put_counter", token_name="steam")

    m = re.search(r"arcane barrier (\d+)", c)
    if m:
        return eff("arcane_barrier", int(m.group(1)))

    if re.search(r"equip an equipment they have equipped", c):
        return eff("steal_equipment")

    if re.search(r"gain control of an aura they control", c):
        return eff("steal_aura", target="opponent")

    if re.search(r"gain control of a gold token they control", c):
        return eff("steal_token", token_name="gold", target="opponent")

    if re.search(r"steal an aura token", c):
        return eff("steal_token", token_name="aura", target="opponent")

    if re.search(r"steal a gold token", c):
        return eff("steal_token", token_name="gold", target="opponent")

    if re.search(r"they can't play face-up cards from arsenal", c):
        return eff("block_arsenal_play", target="opponent")

    if re.search(r"opponents can't attack with weapons", c):
        return eff("block_weapon_attacks", target="opponent")

    if re.search(r"put the top card of their deck into their graveyard", c):
        return eff("destroy_top", target="opponent")
    m = re.search(r"put the top (\d+) cards of their deck into their graveyard", c)
    if m:
        return eff("destroy_top", int(m.group(1)), target="opponent")

    if re.search(r"destroy an item you control", c):
        return eff("destroy_item", optional=optional or "you may" in c)

    if re.search(r"the crowd cheers", c):
        return eff("crowd_cheer")

    if re.search(r"the crowd boos", c):
        return eff("crowd_boo")

    if re.search(r"play your next .+ as though it were an instant", c):
        return eff("play_as_instant")

    m = re.search(r"return a (.+?) from your graveyard to your hand", c)
    if m:
        return eff("return_gy_to_hand", banish_name=m.group(1).strip(), optional=optional or "you may" in c)

    m = re.search(
        r"their first attack during their next turn gets -(\d+)\s*(?:\{p\}|power)",
        c,
    )
    if m:
        return eff("next_attack_power", -int(m.group(1)), target="opponent")

    m = re.search(
        r"their first attack during their next turn costs an additional ((?:\{r\})+|\d+) to play or activate",
        c,
    )
    if m:
        return eff(
            "opponent_cost_increase",
            _resource_amount(m.group(1)),
            target="opponent",
            condition="first_attack_next_turn",
        )

    m = re.search(
        r"their first action during their next turn costs an additional ((?:\{r\})+|\d+) to play or activate",
        c,
    )
    if m:
        return eff("opponent_cost_increase", _resource_amount(m.group(1)), target="opponent")

    m = re.search(
        r"the base \{p\} of the first attack action card they play during their next turn is halved, rounded up",
        c,
    )
    if m:
        return eff("halve_base_power", target="opponent", condition="first_attack_next_turn")

    if re.search(r"your next attack this combat chain is draconic", c):
        return eff("grant_draconic", raw=clause.strip())

    m = re.search(r"cards cost (?:\{r\}|(\d+) resource|\d+) more to play this turn", c)
    if m:
        amount = int(m.group(1)) if m.group(1) else 1
        return eff("opponent_cost_increase", amount, target="each_hero")

    if re.search(r"put a card from their arsenal on the bottom of its owner'?s deck", c):
        return eff("put_bottom", target="opponent", banish_name="arsenal")

    m = re.search(
        r"you may put a card from your hand on the bottom of your deck(?:\.|,)\s*if you do,?\s*draw a card",
        c,
    )
    if m:
        return eff("put_bottom", optional=True, banish_name="draw")

    m = re.search(
        r"when this defends,?\s*you may reveal a card with crush from your hand\.?\s*"
        r"if you do,?\s*create an? (.+?) token",
        c,
    )
    if m:
        return eff("create_token", token_name=m.group(1).strip(), optional=True)

    if re.search(r"you may retrieve a dagger from your graveyard", c):
        return eff("retrieve_gy", banish_name="dagger", optional=True, raw=clause.strip())

    if re.search(
        r"each hero who doesn't have a card in their arsenal puts the top card of their deck face-down into their arsenal",
        c,
    ):
        return eff("put_deck_top_arsenal", target="each_hero", banish_name="if_empty_arsenal", raw=clause.strip())

    if re.search(r"equip a dagger from your inventory", c):
        return eff("equip_inventory", banish_name="dagger", raw=clause.strip())

    if re.search(
        r"you may activate abilities of bows you control an additional time this turn and as though they were an instant",
        c,
    ):
        return eff("extra_bow_activations", optional=True, raw=clause.strip())

    if re.search(r"they lose the game", c):
        return eff("lose_game", target="opponent", raw=clause.strip())

    if re.search(r"you lose the game", c):
        return eff("lose_game", target="self", raw=clause.strip())

    m = re.search(
        r"each hero puts the top card of their deck face-down into their arsenal",
        c,
    )
    if m:
        rider = ""
        if re.search(r"if 2 or more cards are put into arsenals this way, this gets go again", c):
            rider = "go_again_if_ge:2"
        return eff("put_deck_top_arsenal", target="each_hero", banish_name=rider, raw=clause.strip())

    m = re.search(
        r"\{u\} an ally they control, then steal it until the end of this action phase",
        c,
    )
    if m:
        return eff("steal_ally", target="opponent", raw=clause.strip())

    m = re.search(
        r"steal an item they control",
        c,
    )
    if m:
        return eff("steal_equipment", target="opponent", raw=clause.strip())

    m = re.search(r"remove (\d+) (\w+) counters? from this", c)
    if m:
        return eff("remove_counter", int(m.group(1)), token_name=m.group(2).strip())

    m = re.search(r"remove a (\w+) counter from this", c)
    if m:
        return eff("remove_counter", 1, token_name=m.group(1).strip())

    if re.search(r"search your deck for an? attack action card with cost (\d+) or less", c):
        m = re.search(r"cost (\d+) or less", c)
        return eff("search", max_cost=int(m.group(1)), banish_name="attack_action")

    if re.search(r"search your deck for", c):
        m = re.search(r"for an? (.+?)(?:\.|$|, and put)", c)
        name = m.group(1).strip() if m else ""
        return eff("search", banish_name=name)

    if re.search(r"look at the top card of your deck", c):
        return eff("reveal_top")

    if re.search(r"put a card from your hand on the bottom of your deck", c):
        return eff("put_bottom", target="self")

    if re.search(r"put a card from their arsenal on the bottom of their deck", c):
        return eff("put_bottom", target="opponent")

    if re.search(r"put it on the bottom of its owner'?s deck", c):
        return eff("put_bottom", target="opponent")

    if re.search(r"put this on the bottom of (?:its owner'?s|your) deck", c):
        return eff("put_bottom", target="self")

    if re.search(r"cards and abilities cost opponents an additional \{r\} to play or activate this turn", c):
        return eff("opponent_cost_increase", 1, target="opponent")

    m = re.search(
        r"put an action card with cost (\d+) or less from their hand on top of their deck",
        c,
    )
    if m:
        return eff("put_hand_top", max_cost=int(m.group(1)), target="opponent", banish_name="action")

    if re.search(
        r"deal arcane damage to that hero equal to the number of frostbites they control",
        c,
    ):
        return eff("arcane_damage", target="opponent", banish_name="token_count:frostbite")

    if re.search(r"put a card from their hand on top of their deck", c):
        return eff("put_hand_top", target="opponent")

    m = re.search(r"discards? a card unless they pay ((?:\{r\})+|\d+)", c)
    if m:
        txt = m.group(1)
        n = txt.count("{r}") if "{r}" in txt else int(txt)
        return eff("unless_pay", n, target="opponent", banish_name="discard")

    if re.search(r"for each", c):
        m = re.search(r"for each (.+?), create an? (.+?) token", c)
        if m:
            return eff("for_each", token_name=m.group(2).strip(), raw=raw)

    m = re.search(r"if you control an? (\w+) token,?\s*create (\d+) more", c)
    if m:
        return eff("create_token", int(m.group(2)), token_name=m.group(1).strip(),
                   condition=f"control_token:{m.group(1).strip()}")

    grant = _parse_grant_may_play(c, raw, optional, condition)
    if grant is not None:
        return grant

    if re.search(r"you may play this from your banished zone", c):
        return eff("enable_banish_play")

    if re.search(r"you may play this from your graveyard", c):
        return eff("enable_gy_play")

    m = re.search(r"you may play an? (.+?) from your graveyard", c)
    if m:
        return eff("enable_gy_play", banish_name=m.group(1).strip())

    if re.search(r"you may play cards with (\w+(?:\s+\w+)*) from your graveyard", c):
        m = re.search(r"you may play cards with (\w+(?:\s+\w+)*) from your graveyard", c)
        return eff("enable_gy_play", banish_name=m.group(1).strip() if m else "")

    if re.search(r"you may play it this turn from your graveyard", c):
        return eff("enable_gy_play", banish_name="this")

    pitch = _parse_pitch_pay(raw, optional=optional or "you may" in c)
    if pitch is not None:
        return pitch

    if re.search(r"you may banish", c) and "graveyard" in c:
        bm = re.search(r"you may banish an? (.+?) from your graveyard", c, re.I)
        banish_name = bm.group(1).strip() if bm else ""
        return eff("banish_combo", optional=True, banish_name=banish_name, raw=raw)

    if re.search(r"you may put a card from your arsenal on the bottom of your deck", c):
        return eff("put_bottom", target="self")

    if re.search(r"put a card from your arsenal on the bottom of your deck", c):
        return eff("put_bottom", target="self")

    m = re.search(r"put a gold counter on (.+)", c)
    if m:
        return eff("put_counter", token_name="gold", banish_name=m.group(1).strip())

    m = re.search(r"put a (.+?) counter on (.+)", c)
    if m:
        return eff("put_counter", token_name=m.group(1).strip(), banish_name=m.group(2).strip())

    if re.search(r"you may put a non-attack action card from your graveyard on (?:top|bottom) of your deck", c):
        return eff("return_gy_to_deck", banish_name="non-attack action", optional=True)

    if re.search(r"you may put .+ from your graveyard on (?:top|bottom) of your deck", c):
        m = re.search(r"put (.+?) from your graveyard", c)
        name = m.group(1).strip() if m else ""
        return eff("return_gy_to_deck", banish_name=name, optional=True)

    if re.search(r"costs? \{r\} less to play for each", c):
        return eff("cost_reduction", 1)

    if re.search(r"the next card you play this turn with an arcane damage effect", c):
        return eff("amp", 2)

    if re.search(r"the next time you .+ this turn", c):
        return eff("next_action_go_again")

    m = re.search(r"it gets -(\d+)\{p\} unless you pay ((?:\{r\})+|\d+)", c)
    if m:
        txt = m.group(2)
        n = txt.count("{r}") if "{r}" in txt else int(txt)
        return eff(
            "unless_pay",
            n,
            target="self",
            banish_name=f"power:{int(m.group(1))}",
        )

    if re.search(r"if you've pitched a blue card this turn, create a crouching tiger in your hand", c):
        return eff("create_in_hand", banish_name="Crouching Tiger", playable_banished=True)

    if re.search(r"create a crouching tiger in your hand", c):
        return eff("create_in_hand", banish_name="Crouching Tiger")

    if re.search(r"they choose and reveal a card from their hand", c):
        return eff("reveal_top")

    if re.search(r"you may \{t\}", c):
        return eff("attack")

    if re.search(r"destroy this", c) and len(c) < 20:
        return eff("destroy_top", target="self")

    return None


def _parse_clause_effects(clause: str, *, allow_this_turn: bool = False) -> tuple[str, tuple[Effect, ...]]:
    """Parse a trigger clause that may contain a condition and multiple effects."""
    condition, remainder = _detect_condition(clause)
    body = remainder.strip()

    trigger = _parse_trigger_clause(body, optional="you may" in body.lower())
    if trigger is not None:
        if condition:
            trigger = Effect(
                trigger.kind,
                trigger.amount,
                trigger.raw,
                max_cost=trigger.max_cost,
                banish_name=trigger.banish_name,
                token_name=trigger.token_name,
                target=trigger.target,
                optional=trigger.optional,
                condition=condition,
            )
        return condition, (trigger,)

    full = _parse_create_banish(body)
    if full is not None and full.implemented:
        if condition:
            full = Effect(
                full.kind,
                full.amount,
                full.raw,
                banish_name=full.banish_name,
                token_name=full.token_name,
                target=full.target,
                playable_banished=full.playable_banished,
                condition=condition,
            )
        return condition, (full,)

    pitch = _parse_pitch_pay(body, optional="you may" in body.lower())
    if pitch is not None:
        if condition:
            pitch = Effect(
                pitch.kind,
                pitch.amount,
                pitch.raw,
                max_cost=pitch.max_cost,
                banish_name=pitch.banish_name,
                token_name=pitch.token_name,
                optional=pitch.optional,
                condition=condition,
            )
        return condition, (pitch,)

    low = body.lower()
    found: list[Effect] = []

    for part in re.split(r"\s+and\s+", body):
        part = part.strip(" ,.")
        if not part:
            continue
        parsed = (
            _parse_create_banish(part)
            or _parse_extended_clause(part, optional=False, condition="")
            or _parse_next_attack_power(part)
            or _parse_effect(part, allow_this_turn=allow_this_turn)
        )
        if parsed.implemented:
            found.append(parsed)

    if not found:
        single = (
            _parse_create_banish(body)
            or _parse_extended_clause(body, optional=False, condition=condition)
            or _parse_next_attack_power(body)
            or _parse_effect(body, allow_this_turn=allow_this_turn)
        )
        if single.implemented:
            found.append(single)
        elif "go again" in low and "create a crouching tiger" in low:
            found.append(Effect("go_again", raw=body))
            found.append(
                Effect(
                    "create_banished",
                    banish_name="Crouching Tiger",
                    playable_banished="you may play it this turn" in low,
                    raw=body,
                )
            )
        elif re.search(r"\bclash\b", low):
            if re.search(r"if you win, destroy the top card of their deck", low):
                found.append(
                    Effect(
                        "clash",
                        banish_name="hit_hero_routine",
                        raw=body.strip(),
                        condition=condition,
                    )
                )
            elif re.search(r"destroy the top card of their deck", low):
                found.append(
                    Effect(
                        "clash",
                        banish_name="destroy_top_loser",
                        raw=body.strip(),
                        condition=condition,
                    )
                )
            else:
                found.append(Effect("clash", raw=body.strip(), condition=condition))
        elif body.strip():
            found.append(Effect("unimplemented", raw=body.strip(), condition=condition))

    if not found:
        return condition, (Effect("unimplemented", raw=clause.strip(), condition=condition),)

    out: list[Effect] = []
    for eff in found:
        if condition and eff.implemented:
            out.append(
                Effect(
                    eff.kind,
                    eff.amount,
                    eff.raw,
                    max_cost=eff.max_cost,
                    banish_name=eff.banish_name,
                    token_name=eff.token_name,
                    target=eff.target,
                    playable_banished=eff.playable_banished,
                    go_again=eff.go_again,
                    optional=eff.optional,
                    condition=condition,
                )
            )
        else:
            out.append(eff)
    return condition, tuple(out)


def _parse_conditional_quoted_hit_triggers(body: str) -> tuple[Trigger, ...]:
    """Parse 'If <gate>, this gets \"When this hits, ...\"' play/hit riders."""
    triggers: list[Trigger] = []

    m = re.search(
        r'if you control (\d+) or more auras,?\s*(?:it |this )?gets \+(\d+)\{p\} and "when this hits a hero, ([^"]+)"',
        body,
        re.I,
    )
    if m:
        cond = f"controls_ge:aura:{int(m.group(1))}"
        inner = re.sub(
            r"^when this hits(?: a hero|,)?,?\s*",
            "",
            m.group(3).strip(),
            flags=re.I,
        )
        hit = _parse_trigger_clause(inner, optional=False)
        triggers.append(
            Trigger(
                "on_play",
                Effect(
                    "power",
                    int(m.group(2)),
                    condition=cond,
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )
        if hit is not None and hit.implemented:
            triggers.append(
                Trigger(
                    "on_hit",
                    Effect(
                        hit.kind,
                        hit.amount,
                        condition=cond,
                        banish_name=hit.banish_name,
                        target=hit.target,
                        raw=m.group(0)[:120],
                    ),
                    m.group(0)[:80],
                )
            )

    for pat, cond in (
        (
            r'if this has an aim counter,?\s*it gets "(when this hits[^"]+)"',
            "has_aim_counter",
        ),
        (
            r'if this has 6 or more \{p\},?\s*it gets "(when this hits[^"]+)"',
            "power_ge_6",
        ),
    ):
        m = re.search(pat, body, re.I)
        if not m:
            continue
        inner = re.sub(
            r"^when this hits(?: a hero|,)?,?\s*",
            "",
            m.group(1).strip(),
            flags=re.I,
        )
        hit = _parse_trigger_clause(inner, optional=False)
        if hit is not None and hit.implemented:
            triggers.append(
                Trigger(
                    "on_hit",
                    Effect(
                        hit.kind,
                        hit.amount,
                        condition=cond,
                        banish_name=hit.banish_name,
                        target=hit.target,
                        raw=m.group(0)[:120],
                    ),
                    m.group(0)[:80],
                )
            )

    m = re.search(
        r"if you(?:'ve| have) played or activated (\d+) or more attack reactions this chain link,?\s*"
        r"(?:it |this )?gets \+(\d+)\{p\} and \"when this hits a hero, ([^\"]+)\"",
        body,
        re.I,
    )
    if m:
        cond = f"attack_reactions_ge:{int(m.group(1))}"
        inner = re.sub(
            r"^when this hits(?: a hero|,)?,?\s*",
            "",
            m.group(3).strip(),
            flags=re.I,
        )
        hit = _parse_trigger_clause(inner, optional=False)
        triggers.append(
            Trigger(
                "on_play",
                Effect(
                    "power",
                    int(m.group(2)),
                    condition=cond,
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )
        if hit is not None and hit.implemented:
            triggers.append(
                Trigger(
                    "on_hit",
                    Effect(
                        hit.kind,
                        hit.amount,
                        condition=cond,
                        banish_name=hit.banish_name,
                        target=hit.target,
                        raw=m.group(0)[:120],
                    ),
                    m.group(0)[:80],
                )
            )

    for pat, cond in (
        (
            r'if you(?:\'ve| have) charged this turn,?\s*(?:it |this )?gets "(when this hits,? ([^"]+))"',
            "charged_this_turn",
        ),
        (
            r'if you(?:\'ve| have) transcended this turn,?\s*(?:it |this )?gets "(when this chain link resolves,? ([^"]+))"',
            "transcended_this_turn",
        ),
    ):
        m = re.search(pat, body, re.I)
        if not m:
            continue
        inner = m.group(2).strip(" .")
        when = "on_chain_resolve" if "chain link resolves" in m.group(1).lower() else "on_hit"
        hit = _parse_trigger_clause(inner, optional=False)
        if hit is not None and hit.implemented:
            triggers.append(
                Trigger(
                    when,
                    Effect(
                        hit.kind,
                        hit.amount,
                        condition=cond,
                        banish_name=hit.banish_name,
                        target=hit.target,
                        raw=m.group(0)[:120],
                    ),
                    m.group(0)[:80],
                )
            )

    m = re.search(
        r"if an attack action card and a non-attack action card were pitched to play this,?\s*"
        r'(?:it |this )?gets "(the first time this deals damage to the defending hero, '
        r'they discard a card and you draw a card\.)"',
        body,
        re.I,
    )
    if m:
        cond = "pitched_attack_and_nonattack:first_damage"
        triggers.append(
            Trigger(
                "on_hit",
                Effect("discard", 1, target="opponent", condition=cond, raw=m.group(0)[:120]),
                m.group(0)[:80],
            )
        )
        triggers.append(
            Trigger(
                "on_hit",
                Effect("draw", 1, condition=cond, raw=m.group(0)[:120]),
                m.group(0)[:80],
            )
        )

    m = re.search(
        r"if (\d+) or more auras of suspense have left the arena this turn,?\s*"
        r'(?:it |this )?gets "(when this hits a hero, ([^"]+))"',
        body,
        re.I,
    )
    if m:
        cond = f"suspense_left_ge:{int(m.group(1))}"
        inner = m.group(2).strip(" .").lower()
        if "extra turn" in inner:
            triggers.append(
                Trigger(
                    "on_hit",
                    Effect("extra_turn", condition=cond, raw=m.group(0)[:120]),
                    m.group(0)[:80],
                )
            )
        if "draw up to their" in inner:
            triggers.append(
                Trigger(
                    "on_hit",
                    Effect(
                        "schedule_end_phase",
                        banish_name="draw_up_to_intellect:opponent",
                        condition=cond,
                        raw=m.group(0)[:120],
                    ),
                    m.group(0)[:80],
                )
            )

    # "If you've dealt arcane damage this turn, this gets 'When this hits a hero, they discard a card.'"
    m = re.search(
        r'if you(?:\'ve| have) dealt arcane damage this turn,?\s*this gets "([^"]+)"',
        body,
        re.I,
    )
    if m:
        inner = re.sub(r"^when this hits(?: a hero)?,?\s*", "", m.group(1).strip(), flags=re.I)
        hit = _parse_trigger_clause(inner, optional=False)
        if hit is not None and hit.implemented:
            triggers.append(
                Trigger(
                    "on_hit",
                    Effect(
                        hit.kind,
                        hit.amount,
                        condition="arcane_dealt_turn",
                        banish_name=hit.banish_name,
                        target=hit.target,
                        raw=m.group(0)[:120],
                    ),
                    m.group(0)[:80],
                )
            )

    # "If you've played or created an aura this turn, this gets 'When this hits a hero, they discard a card.'"
    m = re.search(
        r'if you(?:\'ve| have) played or created an? aura this turn,?\s*this gets "([^"]+)"',
        body,
        re.I,
    )
    if m:
        inner = re.sub(r"^when this hits(?: a hero)?,?\s*", "", m.group(1).strip(), flags=re.I)
        hit = _parse_trigger_clause(inner, optional=False)
        if hit is not None and hit.implemented:
            triggers.append(
                Trigger(
                    "on_hit",
                    Effect(
                        hit.kind,
                        hit.amount,
                        condition="auras_ge:1",
                        banish_name=hit.banish_name,
                        target=hit.target,
                        raw=m.group(0)[:120],
                    ),
                    m.group(0)[:80],
                )
            )

    return tuple(triggers)


def _parse_fused_clause(clause: str) -> Effect | None:
    """Parse fusion riders captured after 'if this was fused,'."""
    c = " ".join(clause.lower().split())
    raw = clause.strip()
    if not c:
        return None

    if re.match(r"^instead create twice that many\.?$", c):
        return Effect(
            "create_token",
            token_name="seismic surge",
            banish_name="life_diff_double:seismic surge",
            condition="life_less",
            raw=raw,
        )

    m = re.match(r"^it gets \+(\d+)\{d\}\.?$", c)
    if m:
        return Effect("next_defense_bonus", int(m.group(1)), raw=raw)

    if re.search(
        r"and deals damage to a hero, freeze them and all equipment they control",
        c,
    ):
        return Effect(
            "freeze",
            target="opponent",
            banish_name="hero_and_equipment",
            condition="dealt_hero_damage",
            raw=raw,
        )

    if re.search(r"and deals damage to a hero, freeze a card in their arsenal", c):
        return Effect(
            "freeze",
            target="opponent",
            banish_name="arsenal",
            condition="dealt_hero_damage",
            raw=raw,
        )

    if re.search(
        r"instead deal x arcane damage, where x is 5 plus the number of frostbites, "
        r"ice afflictions, and frozen cards they control",
        c,
    ):
        return Effect(
            "arcane_damage",
            target="opponent",
            banish_name="ice_markers:5",
            raw=raw,
        )

    if re.search(
        r'it gets "when this hits a hero, deal damage to them equal to the number of equipment they control',
        c,
    ):
        return Effect("hit_rider", banish_name="damage:equipment_count", raw=raw)

    hit = _parse_quoted_hit_rider(
        clause if '"' in clause else f'it gets "{clause}"'
    )
    if hit is not None and hit.implemented:
        return Effect(
            "hit_rider",
            banish_name=f"{hit.kind}:{hit.banish_name or ''}",
            raw=raw,
        )

    return None


def _parse_quoted_hit_rider(clause: str) -> Effect | None:
    """Parse embedded \"When this hits, ...\" grant on next-attack buffs."""
    m = re.search(r'"(when this hits[^"]*)"', clause, re.I)
    if not m:
        return None
    inner = re.sub(
        r"^when this hits(?: a hero or ally| a hero|,)?,?\s*",
        "",
        m.group(1).strip(),
        flags=re.I,
    ).strip()
    return _parse_trigger_clause(inner, optional="you may" in inner.lower())


def _parse_next_attack_power(clause: str) -> Effect | None:
    c = " ".join(clause.lower().split())
    hit_rider = _parse_quoted_hit_rider(clause)
    hit_token = ""
    if hit_rider and hit_rider.implemented:
        hit_token = f"hit:{hit_rider.kind}:{hit_rider.banish_name or ''}"

    m = re.search(
        r"the next (?:attack action card|aura|guardian attack action card|brute attack action card) you play this turn costs ((?:\{r\})+|\d+) less to play",
        c,
    )
    if m:
        return Effect("cost_reduction", _resource_amount(m.group(1)), clause.strip())
    m = re.search(
        r"your next attack with (\d+) or more base \{p\} this turn gets \+(\d+)\s*(?:\{p\}|power)",
        c,
    )
    if m:
        return Effect(
            "next_attack_power",
            int(m.group(2)),
            clause.strip(),
            banish_name=m.group(1),
            condition="min_base_power",
            token_name=hit_token,
        )
    m = re.search(
        r"your next (\w+) attack this turn gets \+(\d+)\s*(?:\{p\}|power)",
        c,
    )
    if m:
        return Effect(
            "next_attack_power",
            int(m.group(2)),
            clause.strip(),
            banish_name=m.group(1),
            condition="weapon_class",
            token_name=hit_token,
        )
    m = re.search(
        r"your next weapon attack this turn gets \+(\d+)\s*(?:\{p\}|power)",
        c,
    )
    if m:
        return Effect("next_attack_power", int(m.group(1)), clause.strip(), condition="weapon_only")
    m = re.search(
        r"(?:your )?(?:the )?next attack(?: action card| action)?(?: this turn)? gets \+(\d+)\s*(?:\{p\}|power)",
        c,
    )
    if m:
        return Effect(
            "next_attack_power",
            int(m.group(1)),
            clause.strip(),
            token_name=hit_token,
        )
    m = re.search(
        r"the next attack action card(?: with cost (\d+) or less)? you play this turn gets \+(\d+)\s*(?:\{p\}|power)",
        c,
    )
    if m:
        max_cost = int(m.group(1)) if m.group(1) else 99
        return Effect(
            "next_attack_power",
            int(m.group(2)),
            clause.strip(),
            max_cost=max_cost,
            token_name=hit_token,
        )
    return None


def _parse_effect(clause: str, *, allow_this_turn: bool = False) -> Effect:
    c = " ".join(clause.lower().split())
    if not c:
        return Effect("unimplemented", raw=clause.strip())

    optional = False
    if c.startswith("you may "):
        optional = True
        c = c[len("you may "):]

    condition, remainder = _detect_condition(c)
    if condition:
        c = " ".join(remainder.lower().split())
        clause = remainder.strip()

    create_banish = _parse_create_banish(c)
    if create_banish is not None:
        return Effect(
            create_banish.kind,
            create_banish.amount,
            clause.strip(),
            banish_name=create_banish.banish_name,
            token_name=create_banish.token_name,
            target=create_banish.target,
            playable_banished=create_banish.playable_banished,
            optional=optional,
            condition=condition,
        )

    extended = _parse_extended_clause(clause, optional=optional, condition=condition)
    if extended is not None:
        return extended

    next_attack = _parse_next_attack_power(c)
    if next_attack is not None:
        return Effect(
            next_attack.kind,
            next_attack.amount,
            clause.strip(),
            max_cost=next_attack.max_cost,
            banish_name=next_attack.banish_name,
            token_name=next_attack.token_name,
            optional=optional,
            condition=condition or next_attack.condition,
        )

    markers = _COMPLEX_MARKERS
    if allow_this_turn:
        markers = tuple(m for m in _COMPLEX_MARKERS if m != "this turn")
    if "discard" in c and re.search(r"draw .+ discard", c):
        # "draw a card, then discard a random card" — take the draw portion.
        dm = re.search(r"draw (a|an|one|\d+) cards?", c)
        if dm:
            n = 1 if dm.group(1) in ("a", "an", "one") else int(dm.group(1))
            return Effect("draw", n, clause.strip(), optional=optional, condition=condition)
    if any(marker in c for marker in markers):
        extended = _parse_extended_clause(clause, optional=optional, condition=condition)
        if extended is not None and extended.implemented:
            return extended
        if "go again" in c and re.search(r"\+(\d+).+go again", c):
            pm = re.search(r"\+(\d+)\s*(?:\{p\}|power)", c)
            if pm:
                return Effect(
                    "power",
                    int(pm.group(1)),
                    clause.strip(),
                    go_again=True,
                    optional=optional,
                    condition=condition,
                )
        return Effect("unimplemented", raw=clause.strip(), condition=condition)

    if re.search(r"\bintimidate\b", c):
        return Effect("intimidate", raw=clause.strip(), optional=optional, condition=condition)
    if re.search(r"\{t\} them", c):
        return Effect("intimidate", raw=clause.strip(), optional=optional, condition=condition, target="opponent")
    if re.search(r"put all cards in all arsenals on the bottom of their owner's deck", c):
        return Effect("put_bottom", banish_name="all_arsenals", raw=clause.strip(), condition=condition)
    if re.search(r"look at the top card of target hero's deck", c):
        return Effect("look_deck", 1, raw=clause.strip(), target="opponent")
    if re.search(r"\bdominate\b", c):
        return Effect("dominate", raw=clause.strip(), optional=optional, condition=condition)
    if re.search(r"gain(?:s)? (?:1|an?) action points?", c):
        return Effect("go_again", raw=clause.strip(), optional=optional, condition=condition)
    if "go again" in c:
        pm = re.search(r"\+(\d+)\s*(?:\{p\}|power)", c)
        if pm:
            return Effect(
                "power",
                int(pm.group(1)),
                clause.strip(),
                go_again=True,
                optional=optional,
                condition=condition,
            )
        return Effect("go_again", raw=clause.strip(), optional=optional, condition=condition)

    m = re.search(r"deal (\d+)\s+(arcane\s+)?damage", c)
    if m:
        kind = "arcane_damage" if m.group(2) else "damage"
        return Effect(kind, int(m.group(1)), clause.strip(), optional=optional, condition=condition)

    m = re.search(r"(?:gets|get|gain|gains)\s*\+(\d+)\s*(?:\{p\}|power)", c)
    if m:
        rider_ga = "go again" in c
        return Effect(
            "power",
            int(m.group(1)),
            clause.strip(),
            go_again=rider_ga,
            optional=optional,
            condition=condition,
        )

    m = re.search(r"draw (a|an|one|\d+) cards?", c)
    if m:
        n = 1 if m.group(1) in ("a", "an", "one") else int(m.group(1))
        return Effect("draw", n, clause.strip(), optional=optional, condition=condition)

    trigger = _parse_trigger_clause(clause, optional=optional)
    if trigger is not None:
        if condition:
            trigger = Effect(
                trigger.kind,
                trigger.amount,
                trigger.raw,
                max_cost=trigger.max_cost,
                banish_name=trigger.banish_name,
                token_name=trigger.token_name,
                target=trigger.target,
                playable_banished=trigger.playable_banished,
                go_again=trigger.go_again,
                optional=trigger.optional or optional,
                condition=condition,
            )
        return trigger

    from .effect_coverage import _parse_effect_aggressive

    agg = _parse_effect_aggressive(clause)
    if agg is not None:
        if condition and agg.implemented:
            return Effect(
                agg.kind, agg.amount, agg.raw,
                max_cost=agg.max_cost, banish_name=agg.banish_name,
                token_name=agg.token_name, target=agg.target,
                playable_banished=agg.playable_banished, go_again=agg.go_again,
                optional=optional, condition=condition,
            )
        if agg.implemented:
            return Effect(
                agg.kind, agg.amount, agg.raw,
                max_cost=agg.max_cost, banish_name=agg.banish_name,
                token_name=agg.token_name, target=agg.target,
                playable_banished=agg.playable_banished, go_again=agg.go_again,
                optional=optional, condition=condition,
            )

    return Effect("unimplemented", raw=clause.strip(), condition=condition)


def _split_on_play_clause(clause: str) -> tuple[str, str]:
    """Split 'if <cond>, <effect>' from an on-play remainder."""
    low = clause.lower()
    for key, patterns in CONDITION_ALIASES.items():
        for pat in patterns:
            m = re.search(pat, low)
            if not m:
                continue
            effect_part = clause[m.end():].lstrip(" ,.")
            return key, effect_part
    return "", clause


@functools.lru_cache(maxsize=8192)
def parse_play_modifiers(text: str) -> tuple[PlayModifier, ...]:
    """Inline bonuses on play such as 'If you've played another red card, this gets go again'."""
    body = _clean(text)
    if not body.strip():
        return ()

    mods: list[PlayModifier] = []
    if re.search(r"this can only be played from arsenal", body, re.I):
        mods.append(PlayModifier(condition="play_zone:arsenal_only", raw="arsenal only"))
    for pat, kind in PLAY_GATE_PATTERNS:
        m = re.search(pat, body, re.I)
        if not m:
            continue
        arg = m.group(1) if m.lastindex else ""
        mods.append(
            PlayModifier(
                condition=f"play_gate:{kind}:{arg}",
                raw=m.group(0)[:80],
            )
        )
        break
    patterns = (
        (
            r"if you(?:'ve| have) played another red card this turn,?\s*this gets\s*\+?(\d+)?\s*(?:\{p\}|power)?\s*(?:and\s*)?(?:\*\*)?go again",
            "played_red_other",
        ),
        (
            r"if you(?:'ve| have) played another blue card this turn,?\s*this gets\s*\+?(\d+)?\s*(?:\{p\}|power)?\s*(?:and\s*)?(?:\*\*)?go again",
            "played_blue_other",
        ),
        (
            r"if there is a card with 6 or more \{p\} in your pitch zone,?\s*this gets\s*(?:\*\*)?go again",
            "pitched_power_6",
        ),
    )
    for pat, kind in patterns:
        m = re.search(pat, body, re.I)
        if not m:
            continue
        if kind == "played_red_other":
            mods.append(PlayModifier(go_again=True, condition="played_red_other", raw=m.group(0)[:80]))
        elif kind == "played_blue_other":
            mods.append(PlayModifier(go_again=True, condition="played_blue_other", raw=m.group(0)[:80]))
        elif kind == "pitched_power_6":
            mods.append(PlayModifier(go_again=True, condition="pitched_power_6", raw=m.group(0)[:80]))

    m = re.search(
        r"if the last action card you played this turn was lightning,?\s*"
        r"this and the next action card you play this turn get\s*(?:\*\*)?go again",
        body,
        re.I,
    )
    if m:
        mods.append(
            PlayModifier(
                go_again=True,
                next_action_go_again=1,
                condition="last_action_lightning",
                raw=m.group(0)[:80],
            )
        )

    for m in re.finditer(
        r"the next action card you play this turn gets\s*(?:\*\*)?go again",
        body,
        re.I,
    ):
        mods.append(PlayModifier(next_action_go_again=1, raw=m.group(0)[:80]))

    # Standalone next-attack buff lines (e.g. Awakening Bellow, Nimblism variants).
    for m in re.finditer(
        r"the next (?:\w+\s+){0,3}attack(?: action card| action)? you play this turn gets \+(\d+)\s*(?:\{p\}|power)",
        body,
        re.I,
    ):
        mods.append(
            PlayModifier(
                next_attack_power=int(m.group(1)),
                raw=m.group(0)[:80],
            )
        )

    for m in re.finditer(
        r"your next weapon attack this turn gets \+(\d+)\s*(?:\{p\}|power)",
        body,
        re.I,
    ):
        mods.append(
            PlayModifier(
                next_attack_power=int(m.group(1)),
                condition="weapon_only",
                raw=m.group(0)[:80],
            )
        )

    play_mod_patterns: tuple[tuple[str, str], ...] = (
        (
            r"if this has 6 or more \{p\},?\s*(?:it |this )?gets \+(\d+)\s*(?:\{p\}|power)",
            "power_ge_6",
        ),
        (
            r"if this was played as chain link (\d+) or higher,?\s*(?:it |this )?gets \+(\d+)\s*(?:\{p\}|power)",
            "chain_link_ge",
        ),
        (
            r"if this is defended by an action card,?\s*(?:it |this )?gets \+(\d+)\s*(?:\{p\}|power)",
            "defended_by_action",
        ),
        (
            r"if this is defended by fewer than (\d+) non-equipment cards,?\s*(?:it |this )?gets \+(\d+)\s*(?:\{p\}|power)",
            "defended_by_fewer",
        ),
        (
            r"this gets \+(\d+)\{d\} for each blue card you've pitched this turn",
            "blue_pitched_each",
        ),
        (
            r"if you control an (\w+) token,?\s*(?:it |this )?gets \+(\d+)\{d\}",
            "has_token",
        ),
        (
            r"if this was played from arsenal,?\s*(?:it |this )?gets \+(\d+)\s*(?:\{p\}|power)",
            "played_from_arsenal",
        ),
        (
            r"if this was played from arsenal,?\s*(?:it |this )?gets \+(\d+)\{d\}",
            "played_from_arsenal_defense",
        ),
        (
            r"if this was played from arsenal,?\s*(?:it |this )?gets(?:\*\*)?go again",
            "played_from_arsenal_go_again",
        ),
        (
            r"if you have a yellow card in your pitch zone,?\s*(?:it |this )?gets \+(\d+)\s*(?:\{p\}|power)",
            "yellow_in_pitch",
        ),
        (
            r"if an attack reaction has been played or activated this chain link,?\s*(?:it |this )?gets \+(\d+)\{d\}",
            "attack_reaction_chain",
        ),
        (
            r"if this is defending on chain link (\d+) or higher,?\s*(?:it |this )?gets \+(\d+)\{d\}",
            "defending_chain_link_ge",
        ),
        (
            r"if a card has been put into your banished zone this turn,?\s*(?:it |this )?gets \+(\d+)\s*(?:\{p\}|power)",
            "banished_this_turn",
        ),
        (
            r"if you have less \{h\} than your opponent,?\s*(?:it |this )?gets \+(\d+)\{d\}",
            "life_less",
        ),
        (
            r"if a yellow card is charged this way,?\s*(?:(?:it|this) )?gets \+(\d+)\s*(?:\{p\}|power)",
            "charged_yellow",
        ),
        (
            r"if this has an aim counter,?\s*(?:it |this )?gets \+(\d+)\s*(?:\{p\}|power)",
            "has_aim_counter",
        ),
        (
            r"if you(?:'ve| have) hit this turn with an attack with stealth, and this is attacking a hero with marked,?\s*"
            r"(?:it |this )?gets \+(\d+)\s*(?:\{p\}|power)",
            "stealth_hit_marked",
        ),
        (
            r"if this is attacking a marked hero,?\s*(?:it |this )?gets \+(\d+)\s*(?:\{p\}|power)",
            "attacking_marked",
        ),
        (
            r"if you control a guardian off-hand,?\s*(?:it |this )?gets \+(\d+)\{d\}",
            "has_guardian_offhand",
        ),
        (
            r"if this is defended by an action card,?\s*(?:it |this )?gets -(\d+)\s*(?:\{p\}|power)",
            "defended_by_action_penalty",
        ),
        (
            r"if you've dealt arcane damage this turn,?\s*(?:it |this )?gets \+(\d+)\{d\}",
            "arcane_dealt_turn",
        ),
        (
            r"when this is defended by (\d+) or more cards,?\s*you may pay \{r\}\.\s*if you do,?\s*"
            r"(?:it |this )?gets \+(\d+)\{p\}",
            "defended_by_ge_pay",
        ),
        (
            r"if an opponent declares an attack,?\s*they must choose this as the target of that attack if able",
            "must_attack_this",
        ),
        (
            r"if the next card you defend with this turn is a card with combo,?\s*(?:it )?gets \+(\d+)\{d\}",
            "next_defend_combo",
        ),
    )
    for pat, kind in play_mod_patterns:
        m = re.search(pat, body, re.I)
        if not m:
            continue
        if kind == "power_ge_6":
            mods.append(PlayModifier(power=int(m.group(1)), condition=kind, raw=m.group(0)[:80]))
        elif kind == "chain_link_ge":
            mods.append(
                PlayModifier(
                    power=int(m.group(2)),
                    condition=kind,
                    condition_arg=m.group(1),
                    raw=m.group(0)[:80],
                )
            )
        elif kind == "defended_by_action":
            mods.append(PlayModifier(power=int(m.group(1)), condition=kind, raw=m.group(0)[:80]))
        elif kind == "defended_by_fewer":
            mods.append(
                PlayModifier(
                    power=int(m.group(2)),
                    condition=kind,
                    condition_arg=m.group(1),
                    raw=m.group(0)[:80],
                )
            )
        elif kind == "blue_pitched_each":
            mods.append(PlayModifier(defense=int(m.group(1)), condition=kind, raw=m.group(0)[:80]))
        elif kind == "has_token":
            mods.append(
                PlayModifier(
                    defense=int(m.group(2)),
                    condition=kind,
                    condition_arg=m.group(1).lower(),
                    raw=m.group(0)[:80],
                )
            )
        elif kind == "played_from_arsenal":
            mods.append(PlayModifier(power=int(m.group(1)), condition=kind, raw=m.group(0)[:80]))
        elif kind == "played_from_arsenal_defense":
            mods.append(PlayModifier(defense=int(m.group(1)), condition=kind, raw=m.group(0)[:80]))
        elif kind == "played_from_arsenal_go_again":
            mods.append(PlayModifier(go_again=True, condition="played_from_arsenal", raw=m.group(0)[:80]))
        elif kind == "yellow_in_pitch":
            mods.append(PlayModifier(power=int(m.group(1)), condition=kind, raw=m.group(0)[:80]))
        elif kind == "attack_reaction_chain":
            mods.append(PlayModifier(defense=int(m.group(1)), condition=kind, raw=m.group(0)[:80]))
        elif kind == "defending_chain_link_ge":
            mods.append(
                PlayModifier(
                    defense=int(m.group(2)),
                    condition=kind,
                    condition_arg=m.group(1),
                    raw=m.group(0)[:80],
                )
            )
        elif kind == "banished_this_turn":
            mods.append(PlayModifier(power=int(m.group(1)), condition=kind, raw=m.group(0)[:80]))
        elif kind == "life_less":
            mods.append(PlayModifier(defense=int(m.group(1)), condition=kind, raw=m.group(0)[:80]))
        elif kind == "charged_yellow":
            mods.append(PlayModifier(power=int(m.group(1)), condition="charged_yellow", raw=m.group(0)[:80]))
        elif kind == "has_aim_counter":
            mods.append(PlayModifier(power=int(m.group(1)), condition=kind, raw=m.group(0)[:80]))
        elif kind == "stealth_hit_marked":
            mods.append(PlayModifier(power=int(m.group(1)), condition=kind, raw=m.group(0)[:80]))
        elif kind == "attacking_marked":
            mods.append(PlayModifier(power=int(m.group(1)), condition=kind, raw=m.group(0)[:80]))
        elif kind == "has_guardian_offhand":
            mods.append(PlayModifier(defense=int(m.group(1)), condition=kind, raw=m.group(0)[:80]))
        elif kind == "defended_by_action_penalty":
            mods.append(PlayModifier(power=-int(m.group(1)), condition=kind, raw=m.group(0)[:80]))
        elif kind == "arcane_dealt_turn":
            mods.append(PlayModifier(defense=int(m.group(1)), condition=kind, raw=m.group(0)[:80]))
        elif kind == "defended_by_ge_pay":
            mods.append(
                PlayModifier(
                    power=int(m.group(2)),
                    condition=kind,
                    condition_arg=m.group(1),
                    raw=m.group(0)[:80],
                )
            )
        elif kind == "next_defend_combo":
            mods.append(PlayModifier(defense=int(m.group(1)), condition=kind, raw=m.group(0)[:80]))

    # Token-pair defense bonus (e.g. Laughing Knee-Slappers, Plate of Tough Love).
    for m in re.finditer(
        r"if you control an? ([\w][\w ]+?) and an? ([\w][\w ]+?) tokens?,?\s*this gets \+(\d+)\{d\}",
        body,
        re.I,
    ):
        tok1 = m.group(1).strip().lower()
        tok2 = m.group(2).strip().lower()
        mods.append(
            PlayModifier(
                defense=int(m.group(3)),
                condition=f"control_tokens:{tok1},{tok2}",
                raw=m.group(0)[:80],
            )
        )

    # Single token defense bonus (e.g. Tremor of Resistance, Stand Ground).
    # Only match if NOT already covered by the pair pattern above.
    _pair_spans = {
        m2.start()
        for m2 in re.finditer(
            r"if you control an? ([\w][\w ]+?) and an? ([\w][\w ]+?) tokens?,?\s*this gets \+(\d+)\{d\}",
            body,
            re.I,
        )
    }
    for m in re.finditer(
        r"if you control an? ([\w][\w ]+?) tokens?,?\s*this gets \+(\d+)\{d\}",
        body,
        re.I,
    ):
        if m.start() in _pair_spans:
            continue
        tok = m.group(1).strip().lower()
        mods.append(
            PlayModifier(
                defense=int(m.group(2)),
                condition=f"control_token:{tok}",
                raw=m.group(0)[:80],
            )
        )

    # "If you control less Gold than an opponent, this gets +N{p}."
    m = re.search(
        r"if you control less gold than an opponent,?\s*this gets \+(\d+)\s*(?:\{p\}|power)",
        body,
        re.I,
    )
    if m:
        mods.append(PlayModifier(power=int(m.group(1)), condition="less_gold_than_opponent", raw=m.group(0)[:80]))

    # "If the defending hero controls an aura token, this gets +N{p}."
    m = re.search(
        r"if the defending hero controls an? aura tokens?,?\s*this gets \+(\d+)\s*(?:\{p\}|power)",
        body,
        re.I,
    )
    if m:
        mods.append(
            PlayModifier(
                power=int(m.group(1)),
                condition="defender_controls_token:aura",
                raw=m.group(0)[:80],
            )
        )

    return tuple(mods)


_MODE_MENU_SPECS: tuple[tuple[str, str, str, int], ...] = (
    (
        r"choose 2",
        "on_play",
        "pick:fixed:2",
        2,
    ),
    (
        r"when (?:this|it) hits(?: a hero)?,?\s*choose 1 at random",
        "on_hit",
        "pick:random:1",
        1,
    ),
    (
        r"when (?:this|it) attacks,?\s*choose 1 for each card you(?:'ve| have) banished from your soul this combat chain",
        "on_attack",
        "pick:soul_banish_chain",
        3,
    ),
    (
        r"when (?:this|it) attacks,?\s*choose 1 for each blue card you(?:'ve| have) pitched this turn",
        "on_attack",
        "pick:blue_pitched",
        1,
    ),
    (
        r"when(?:ever)? (?:this|it) hits,?\s*choose any number",
        "on_hit",
        "pick:any",
        1,
    ),
    (
        r"if you(?:'ve| have) played another blue card this turn,?\s*choose 3\.?\s*otherwise,?\s*choose 1",
        "on_play",
        "pick:played_blue_other:3:1",
        1,
    ),
)


def _parse_mode_bullet(clause: str) -> Effect | None:
    """Parse one * mode bullet into a single resolved effect."""
    c = clause.strip().lstrip("*").strip(" .")
    if not c:
        return None
    low = c.lower()

    if re.search(r"^transcend", low):
        return Effect("transcend", target="self", raw=clause)

    m = re.search(r"each hero creates a (.+?) token", low)
    if m:
        return Effect(
            "create_token",
            token_name=m.group(1).strip(),
            target="each_hero",
            raw=clause,
        )

    if re.search(r"each hero draws a card", low):
        return Effect("draw", 1, target="each_hero", raw=clause)

    if re.search(r"each hero gains 1\{h\}", low):
        return Effect("gain_life", 1, target="each_hero", raw=clause)

    m = re.search(r"create (\d+) (.+?) tokens?", low)
    if m:
        return Effect(
            "create_token",
            int(m.group(1)),
            token_name=m.group(2).strip(),
            target="self",
            raw=clause,
        )

    m = re.search(r"create a (.+?) token", low)
    if m:
        return Effect(
            "create_token",
            token_name=m.group(1).strip(),
            target="self",
            raw=clause,
        )

    if re.search(r"draw a card", low):
        return Effect("draw", 1, target="self", raw=clause)

    m = re.search(r"this gets \+(\d+)\{p\}", low)
    if m:
        return Effect("power", int(m.group(1)), target="self", raw=clause)

    if "go again" in low and "this gets" in low:
        return Effect("go_again", target="self", raw=clause)

    m = re.search(r"put a \+(\d+)\{p\} counter on each aura with ward", low)
    if m:
        return Effect(
            "put_counter",
            int(m.group(1)),
            token_name="power",
            banish_name="ward_auras",
            raw=clause,
        )

    hand = _parse_create_in_hand(c)
    if hand is not None:
        return hand

    m = re.search(r"your (.+?) get \+(\d+)\{p\} this turn", low)
    if m:
        return Effect(
            "named_power_bonus",
            int(m.group(2)),
            token_name=_singular_card_name(m.group(1).strip()),
            target="self",
            raw=clause,
        )

    m = re.search(r"banish up to (\d+) cards in an opposing hero'?s? graveyard", low)
    if m:
        return Effect(
            "banish_graveyard",
            int(m.group(1)),
            target="opponent",
            raw=clause,
        )

    if re.search(r"equip a base equipment with proto in its name from your inventory", low):
        return Effect("equip_inventory", banish_name="proto base", raw=clause)

    if re.search(r"evo permanents you control get \+(\d+)\{d\} this turn", low):
        m = re.search(r"evo permanents you control get \+(\d+)\{d\} this turn", low)
        return Effect(
            "put_counter",
            int(m.group(1)),
            token_name="defense",
            banish_name="evo_permanents",
            raw=clause,
        )

    if re.search(r"put this under an evo permanent you control", low):
        return Effect("modular_equip", raw=clause)

    if re.search(r"you may banish an evo from your hand", low):
        rider = "draw" if re.search(r"if you do, draw a card", low) else ""
        return Effect(
            "destroy_hand",
            1,
            optional=True,
            banish_name=f"evo:{rider}" if rider else "evo",
            raw=clause,
        )

    if re.search(r"they choose a card in their hand", low):
        return Effect("destroy_hand", 1, target="opponent", raw=clause)

    if re.search(r"they choose a card in their arsenal", low):
        return Effect("banish_arsenal", target="opponent", raw=clause)

    if re.search(r"banish the top card of their deck", low):
        return Effect("banish_top", target="opponent", raw=clause)

    return None


def _encode_mode_spec(eff: Effect) -> str:
    if eff.kind == "create_token":
        return (
            f"create_token:{max(1, eff.amount or 1)}:{eff.token_name or ''}:"
            f"{eff.target or 'self'}"
        )
    if eff.kind == "draw":
        return f"draw:{max(1, eff.amount or 1)}:{eff.target or 'self'}"
    if eff.kind == "gain_life":
        return f"gain_life:{max(1, eff.amount or 1)}:{eff.target or 'self'}"
    if eff.kind == "power":
        return f"power:{eff.amount or 0}:self"
    if eff.kind == "go_again":
        return "go_again:0:self"
    if eff.kind == "put_counter":
        return (
            f"put_counter:{max(1, eff.amount or 1)}:{eff.token_name or 'power'}:"
            f"{eff.banish_name or ''}"
        )
    if eff.kind == "create_in_hand":
        return (
            f"create_in_hand:{max(1, eff.amount or 1)}:{eff.banish_name or ''}:"
            f"{eff.target or 'self'}"
        )
    if eff.kind == "named_power_bonus":
        return (
            f"named_power_bonus:{max(1, eff.amount or 1)}:{eff.token_name or ''}:"
            f"{eff.target or 'self'}"
        )
    if eff.kind == "banish_graveyard":
        return (
            f"banish_graveyard:{max(1, eff.amount or 1)}:{eff.target or 'opponent'}"
        )
    if eff.kind == "equip_inventory":
        return f"equip_inventory:0:{eff.banish_name or ''}"
    if eff.kind == "modular_equip":
        return "modular_equip:0:self"
    if eff.kind == "destroy_hand":
        return (
            f"destroy_hand:{max(1, eff.amount or 1)}:"
            f"{eff.banish_name or ''}:{eff.target or 'self'}"
        )
    if eff.kind == "banish_arsenal":
        return f"banish_arsenal:0:{eff.target or 'opponent'}"
    if eff.kind == "banish_top":
        return f"banish_top:{max(1, eff.amount or 1)}:{eff.target or 'opponent'}"
    if eff.kind == "transcend":
        return "transcend:0:self"
    return ""


def _extract_mode_bullets(body: str) -> list[str]:
    return [
        m.group(1).strip(" .")
        for m in re.finditer(r"\*\s*([^*]+?)(?:\.\s|\.$|$)", body)
        if m.group(1).strip(" .")
    ]


def parse_mode_menu_triggers(body: str) -> tuple[Trigger, ...]:
    """Parse multi-mode choose menus (* bullet modes) into choose_mode triggers."""
    low = body.lower()
    bullets = _extract_mode_bullets(body)
    if len(bullets) < 2:
        return ()

    parsed: list[Effect] = []
    for bullet in bullets:
        eff = _parse_mode_bullet(f"* {bullet}")
        if eff is not None and eff.implemented and _encode_mode_spec(eff):
            parsed.append(eff)
    if len(parsed) < 2:
        return ()

    specs = "|".join(_encode_mode_spec(e) for e in parsed)
    labels = "|".join(e.raw.lstrip("*").strip(" .")[:80] for e in parsed)
    out: list[Trigger] = []

    for pattern, when, pick_spec, max_repeat in _MODE_MENU_SPECS:
        m = re.search(pattern, low)
        if not m:
            continue
        header = m.group(0).strip()
        out.append(
            Trigger(
                when,
                Effect(
                    "choose_mode",
                    max_repeat,
                    header,
                    banish_name=specs,
                    token_name=labels,
                    condition=pick_spec,
                ),
                header,
            )
        )
    return tuple(out)


def card_has_mode_menu_bullets(text: str) -> bool:
    """True when card text contains a parsed multi-mode choose menu."""
    body = _clean(text)
    return bool(parse_mode_menu_triggers(body))


@functools.lru_cache(maxsize=8192)
def parse_triggers(text: str) -> tuple[Trigger, ...]:
    """Extract triggered effects from a card's rules text."""
    body = _clean(text)
    if not body.strip():
        return ()

    triggers: list[Trigger] = []
    mode_menus = parse_mode_menu_triggers(body)
    if mode_menus:
        triggers.extend(mode_menus)

    # Pending buff: "the next attack action card [with cost N or less] you play
    # this turn gets +M power" (e.g. Nimblism).
    m = re.search(
        r"the next attack action card(?: with cost (\d+) or less)? you play this turn gets \+(\d+)\s*(?:\{p\}|power)",
        body,
        re.I,
    )
    if m:
        max_cost = int(m.group(1)) if m.group(1) else 99
        triggers.append(
            Trigger(
                "on_play",
                Effect("next_attack_power", amount=int(m.group(2)), max_cost=max_cost, raw="next attack +power"),
                "next attack power buff",
            )
        )

    # Pending buff: "the next [non-attack] action you play this turn gains go again".
    if re.search(
        r"the next (?:non-attack action|attack action|action)[^.]*?gains? go again",
        body,
        re.I,
    ):
        triggers.append(
            Trigger(
                "on_play",
                Effect("next_naa_go_again", raw="next action gains go again"),
                "next action gains go again",
            )
        )

    # Pending buff: "the next action card you play this turn gets go again".
    if re.search(r"the next action card you play this turn gets\s*(?:\*\*)?go again", body, re.I):
        triggers.append(
            Trigger(
                "on_play",
                Effect("next_action_go_again", raw="next action go again"),
                "next action go again",
            )
        )

    # On-attack graveyard combo.
    m = re.search(
        r"you may banish an? (.+?) from your graveyard\.?\s*if you do,?\s*(.+?)(?:\.|$)",
        body,
        re.I,
    )
    if m:
        rider = m.group(2)
        pm = re.search(r"\+(\d+)\s*(?:\{p\}|power)", rider)
        triggers.append(
            Trigger(
                "on_attack",
                Effect(
                    "banish_combo",
                    amount=int(pm.group(1)) if pm else 0,
                    banish_name=m.group(1).strip(),
                    go_again="go again" in rider.lower(),
                    raw=m.group(0)[:70],
                    optional=True,
                ),
                m.group(0)[:70],
            )
        )

    # On-leave: equipment that buffs your next attack when it leaves (Act of Glory).
    m = re.search(
        r"when (?:this|it) leaves the arena,?\s*your next attack this turn gets \+(\d+)\s*(?:\{p\}|power)",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_leave",
                Effect("next_attack_power", amount=int(m.group(1)), raw=m.group(0)[:70]),
                m.group(0)[:70],
            )
        )

    # On-play: "when (this|it) is played[, if ...], <effect>".
    for m in re.finditer(
        r"when (?:this|it) is played(?: from your banished zone)?,?\s*([^.]*)\.?",
        body,
        re.I,
    ):
        clause = m.group(1).strip()
        condition, effect_text = _split_on_play_clause(clause)
        eff = _parse_effect(effect_text)
        if condition and eff.implemented:
            eff = Effect(
                eff.kind,
                eff.amount,
                eff.raw,
                max_cost=eff.max_cost,
                banish_name=eff.banish_name,
                go_again=eff.go_again,
                optional=eff.optional,
                condition=condition,
            )
        elif condition and not eff.implemented:
            eff = Effect("unimplemented", raw=clause, condition=condition)
        if eff.implemented or condition:
            triggers.append(Trigger("on_play", eff, clause))

    # "... if (this|it) was fused, <effect>"
    for m in re.finditer(r"if (?:this|it) (?:was|is) fused,?\s*([^.]*)\.?", body, re.I):
        clause = m.group(1).strip()
        if re.search(
            r"when this attacks, if this was fused, cards and abilities cost opponents",
            body,
            re.I,
        ):
            continue
        if clause.startswith("until end of turn if an attack would deal damage") and re.search(
            r"when this attacks,?\s*if this was fused",
            body,
            re.I,
        ):
            continue
        fused_eff = _parse_fused_clause(clause)
        eff = fused_eff if fused_eff is not None else _parse_effect(clause)
        if eff.implemented:
            triggers.append(Trigger("when_fused", eff, clause))
        elif clause:
            triggers.append(Trigger("when_fused", eff, clause))

    # On-attack: Cold Wave — fused tax on opponents for this turn.
    m = re.search(
        r"when this attacks, if this was fused, cards and abilities cost opponents an additional \{r\} to play or activate this turn",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_attack",
                Effect(
                    "opponent_cost_increase",
                    1,
                    target="opponent",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    # On-attack: Sound the Alarm — reveal hand, optional defense-reaction search to top.
    m = re.search(
        r"when this attacks a hero, they reveal their hand\.?\s*"
        r"if an attack reaction card is revealed this way, you may search your deck for a defense reaction card, reveal it, then shuffle and put it on top",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_attack",
                Effect(
                    "reveal_hand",
                    0,
                    target="opponent",
                    banish_name="store_revealed",
                    raw="reveal hand on attack",
                ),
                m.group(0)[:80],
            )
        )
        triggers.append(
            Trigger(
                "on_attack",
                Effect(
                    "search",
                    optional=True,
                    banish_name="defense_reaction:deck_top",
                    condition="revealed_attack_reaction",
                    raw="search defense reaction to top",
                ),
                m.group(0)[:80],
            )
        )

    # On-attack: reveal defender's deck top (Crash Down the Gates, etc.).
    m = re.search(r"when this attacks a hero, they reveal the top card of their deck", body, re.I)
    if m:
        triggers.append(
            Trigger(
                "on_attack",
                Effect(
                    "reveal_top",
                    target="opponent",
                    banish_name="store_revealed",
                    raw=m.group(0)[:80],
                ),
                m.group(0)[:80],
            )
        )

    # Regicide — chain close and royal hit alternate win conditions.
    if re.search(r"when the combat chain closes, you lose the game", body, re.I):
        triggers.append(
            Trigger(
                "on_chain_close",
                Effect("lose_game", target="self", raw="you lose the game"),
                "you lose the game",
            )
        )
    m = re.search(r"when this hits a royal hero, they lose the game", body, re.I)
    if m:
        triggers.append(
            Trigger(
                "on_hit",
                Effect(
                    "lose_game",
                    target="opponent",
                    condition="royal_hero",
                    raw=m.group(0)[:80],
                ),
                m.group(0)[:80],
            )
        )

    # On-attack: Concoct Disorder — multi-sentence deck-top-to-arsenal + conditional go again.
    m = re.search(
        r"when (?:this|it) attacks,?\s*"
        r"each hero puts the top card of their deck face-down into their arsenal\.?\s*"
        r"if 2 or more cards are put into arsenals this way, this gets go again",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_attack",
                Effect(
                    "put_deck_top_arsenal",
                    target="each_hero",
                    banish_name="go_again_if_ge:2",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    # On-attack: Demolition Protocol — remove steam from opponent Evos-scaled.
    m = re.search(
        r"when this attacks a hero, remove all steam counters from up to x equipment, items, and/or weapons they control",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_attack",
                Effect(
                    "remove_counter",
                    0,
                    token_name="steam",
                    target="opponent",
                    banish_name="evo_count",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    # On-attack: Blood Runs Deep — each dagger deals damage to attacked hero.
    m = re.search(
        r"when this attacks a hero, (each dagger you control deals \d+ damage to them\.?\s*"
        r"if damage is dealt this way, the dagger has hit\.?\s*destroy the daggers)",
        body,
        re.I,
    )
    if m:
        dd = _parse_dagger_damage_text(m.group(1), full_context=m.group(0))
        if dd:
            triggers.append(Trigger("on_attack", dd, m.group(0)[:80]))

    # On-attack: Gore Belching — reveal attack from deck top, banish or -7 power.
    m = re.search(
        r"when this attacks, reveal cards from the top of your deck until you reveal an attack action card\.?\s*"
        r"if you do, banish it and this gets -x\{p\}, where x is the \{p\} of the card banished this way\.?\s*"
        r"otherwise, this gets -7\{p\}\.?\s*shuffle",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_attack",
                Effect(
                    "search",
                    banish_name="attack_action_reveal:-7",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    # On-attack: Hurl — optional pay grants quoted dagger damage on attack.
    m = re.search(
        r'if the additional cost is paid, this gets "(when this attacks, target dagger you control deals \d+ damage to target hero[^"]*)"',
        body,
        re.I,
    )
    if m:
        dd = _parse_dagger_damage_text(m.group(1), full_context=m.group(1))
        if dd:
            triggers.append(
                Trigger(
                    "on_attack",
                    Effect(
                        dd.kind,
                        dd.amount,
                        dd.raw,
                        target=dd.target,
                        banish_name=dd.banish_name,
                        condition="additional_cost_paid",
                    ),
                    m.group(1)[:80],
                )
            )

    # Special-case: Gone in a Flash — on_attack, set up next-instant return self.
    if re.search(
        r"the next time you play an instant card this chain link,?\s*you may return this to its owner'?s hand",
        body,
        re.I,
    ):
        triggers.append(
            Trigger(
                "on_attack",
                Effect("set_next_instant_return_self", optional=True, raw="set next instant: return self to hand"),
                "next instant: return self",
            )
        )

    # Special-case: Blast to Oblivion — on_attack, set up next-instant return aura.
    if re.search(
        r"the next time you play an instant card this chain link,?\s*you may return target aura",
        body,
        re.I,
    ):
        triggers.append(
            Trigger(
                "on_attack",
                Effect("set_next_instant_return_aura", optional=True, raw="set next instant: return target aura"),
                "next instant: return aura",
            )
        )

    # Special-case: Attune with Cosmic Vibrations — on_attack + on_defend, reveal top for blue bonus.
    if re.search(
        r"(?:attacks|defends) a hero'?s? attack,?\s*reveal the top card of their deck",
        body,
        re.I,
    ):
        for when in ("on_attack", "on_defend"):
            triggers.append(
                Trigger(
                    when,
                    Effect("reveal_for_blue_bonus", raw="reveal top card for blue +3 bonus"),
                    "attune: reveal top for blue bonus",
                )
            )

    # Special-case: Red Hot — on_attack, reveal X draconic chain link cards for +power.
    if re.search(
        r"reveal the top x cards of your deck,?\s*where x is the number of draconic chain links you control",
        body,
        re.I,
    ):
        triggers.append(
            Trigger(
                "on_attack",
                Effect("reveal_top_draconic_power", raw="reveal top X draconic links, +power per cost-3+"),
                "red hot: reveal top draconic links",
            )
        )

    # Special-case: Whittle from Bone — on_attack against marked hero, equip Graphene Chelicera token.
    m = re.search(
        r"when this attacks a marked hero,?\s*equip an? (.+?) token",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_attack",
                Effect(
                    "create_token",
                    token_name=m.group(1).strip().lower(),
                    condition="target_marked",
                    raw=m.group(0)[:80],
                ),
                m.group(0)[:80],
            )
        )
        # Skip generic loop catching "attacks a marked hero" clause
        body = body  # no mutation needed; generic loop will skip "^a hero," prefix for "a marked hero,"

    # On-attack: "when (this|it) attacks, <effect>".
    _douse_runeblood = re.search(
        r"when this attacks, create runechant tokens equal to the number of non-attack action cards you've played this turn\.?\s*"
        r"if 3 or more runechants are created this way, this gets go again",
        body,
        re.I,
    )
    for m in re.finditer(r"when (?:this|it) attacks,?\s*([^.]*)\.?", body, re.I):
        clause = m.group(1)
        if _douse_runeblood:
            continue
        if re.search(
            r"when this attacks a hero, remove all steam counters from up to x equipment",
            body,
            re.I,
        ):
            continue
        if re.search(
            r"when this attacks, transform up to 1 ash you control into an aether ashwings?",
            body,
            re.I,
        ):
            continue
        if re.search(r"you may look at the defending hero'?s hand", clause, re.I):
            continue
        # Skip special-cased on_attack triggers already handled above
        if re.search(r"^a marked hero,", clause, re.I):
            continue
        if re.search(r"the next time you play an instant card this chain link", clause, re.I):
            continue
        if re.search(
            r"reveal the top x cards of your deck.*draconic chain links", clause, re.I
        ):
            continue
        if re.search(
            r"when this attacks, it gets \+x\{p\}, where x is the number of gold you control",
            body,
            re.I,
        ):
            continue
        if re.search(
            r"when this attacks, the defending hero reveals their hand",
            body,
            re.I,
        ):
            continue
        if re.search(
            r"when this attacks a hero, they reveal the top card of their deck",
            body,
            re.I,
        ):
            continue
        if re.search(
            r"when this attacks, if this was fused, cards and abilities cost opponents",
            body,
            re.I,
        ):
            continue
        if re.search(
            r"if 2 or more cards are put into arsenals this way, this gets go again",
            body,
            re.I,
        ):
            continue
        if re.search(r"target dagger you control deals \d+ damage", clause, re.I):
            if re.search(r'if the additional cost is paid, this gets "', body, re.I):
                continue
        if re.search(
            r"when this attacks, reveal cards from the top of your deck until you reveal an attack action card",
            body,
            re.I,
        ):
            continue
        if re.search(r"^a hero[,\s]", clause, re.I):
            continue
        if re.search(r"for each hyper driver destroyed this way", clause, re.I):
            m = re.search(
                r"it gets \+(\d+)\{p\} for each hyper driver destroyed this way",
                clause,
                re.I,
            )
            if m:
                triggers.append(
                    Trigger(
                        "on_attack",
                        Effect(
                            "power",
                            int(m.group(1)),
                            banish_name="hyper_drivers_destroyed",
                            raw=clause.strip(),
                        ),
                        clause.strip(),
                    )
                )
            continue
        if re.search(r"^name another card", clause, re.I):
            _, name_effs = _parse_clause_effects(clause, allow_this_turn=True)
            for eff in name_effs:
                if eff.implemented:
                    triggers.append(Trigger("on_attack", eff, clause))
            pm = re.search(
                r"attack action cards with that name get \+(\d+)\{p\} this combat chain",
                body,
                re.I,
            )
            if pm:
                triggers.append(
                    Trigger(
                        "on_attack",
                        Effect(
                            "named_power_bonus",
                            int(pm.group(1)),
                            banish_name="this_chain",
                            raw=pm.group(0)[:80],
                        ),
                        pm.group(0)[:80],
                    )
                )
            continue
        if re.search(r"you may reveal any number of crouching tigers from your hand", clause, re.I):
            reveal = _parse_trigger_clause(clause, optional=True)
            if reveal is not None:
                triggers.append(Trigger("on_attack", reveal, clause))
            continue
        if re.search(
            r"each dagger you control deals \d+ damage",
            clause,
            re.I,
        ) and re.search(r"when this attacks a hero", body, re.I):
            continue
        if re.search(r"you may banish .+ from your graveyard", clause, re.I):
            continue
        if re.search(r"^or defends,", clause, re.I):
            continue
        if re.search(
            r"transform up to 1 ash you control into an aether ashwings?",
            clause,
            re.I,
        ):
            continue
        if re.search(r"you may look at the defending hero'?s hand", clause, re.I):
            continue
        if re.search(
            r"look at the defending hero'?s hand and choose a card",
            clause,
            re.I,
        ):
            continue
        low = clause.lower()
        if "fused" in low and "create" not in low and "banish" not in low:
            continue
        if mode_menus and re.search(r"choose (?:\d|any|1 for each)", low):
            continue
        clause = _extend_trigger_clause(body, clause)
        _, effects = _parse_clause_effects(clause)
        for eff in effects:
            if mode_menus and eff.kind == "unimplemented":
                continue
            if (
                eff.kind == "banish_combo"
                and not (eff.banish_name or "").strip()
                and re.search(r"you may banish .+ from your graveyard", body, re.I)
            ):
                continue
            triggers.append(Trigger("on_attack", eff, clause.strip()))

    # On-hit: Miller's Grindstone-style multi-sentence clash (periods inside the effect).
    m = re.search(
        r"when(?:ever)? (?:this|it) hits(?: a hero)?,?\s*"
        r"clash with them\.?\s*if you win, destroy the top card of their deck\.?\s*"
        r"if they win, put a -1\{p\} counter on this",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_hit",
                Effect("clash", banish_name="hit_hero_routine", raw=m.group(0)[:80]),
                m.group(0)[:80],
            )
        )

    # On-hit: Heat Seeker — delayed end-phase deck top to arsenal.
    m = re.search(
        r"when this hits, at the beginning of your end phase, put the top card of your deck face-up into your arsenal",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_hit",
                Effect(
                    "schedule_end_phase",
                    banish_name="deck_top_arsenal_face_up",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    # On-hit: Promise of Plenty — empty-arsenal heroes stash deck top.
    m = re.search(
        r"when this hits, each hero who doesn't have a card in their arsenal puts the top card of their deck face-down into their arsenal",
        body,
        re.I,
    )
    if m:
        rider = ""
        if re.search(r"if this was played from arsenal, this gets go again", body, re.I):
            rider = "go_again_if_played_from_arsenal"
        triggers.append(
            Trigger(
                "on_hit",
                Effect(
                    "put_deck_top_arsenal",
                    target="each_hero",
                    banish_name="if_empty_arsenal:" + rider if rider else "if_empty_arsenal",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    # On-hit: Pathing Helix — empty arsenal stash.
    m = re.search(
        r"when this hits and you have no cards in your arsenal,?\s*"
        r"you may put a card from your hand face-down into your arsenal",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_hit",
                Effect(
                    "stash_hand",
                    optional=True,
                    condition="empty_arsenal",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    # On-hit: Bonds of Agony — choose hand card, banish same name (quoted ability).
    m = re.search(
        r"when this hits a hero, look at their hand and choose a card\.\s*"
        r"search their hand, deck, and graveyard and banish up to (\d+) cards "
        r"with the same name as the chosen card",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_hit",
                Effect(
                    "choose_card",
                    target="opponent",
                    banish_name=f"banish_same_name:{m.group(1)}",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    # On-hit: Spring a Leak — strip opponent steam counters.
    m = re.search(
        r"when this hits a hero, remove all steam counters from an equipment, item, or weapon they control",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_hit",
                Effect(
                    "remove_counter",
                    0,
                    token_name="steam",
                    target="opponent",
                    banish_name="all_on_one",
                    raw=m.group(0)[:80],
                ),
                m.group(0)[:80],
            )
        )

    # On-hit: Be Like Water — pay {r}, choose name (multi-sentence).
    m = re.search(
        r"when(?:ever)? (?:this|it) hits(?: a hero)?,?\s*"
        r"you may pay \{r\}\.?\s*if you do,?\s*choose (.+?)\.?\s*this gets the chosen name",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_hit",
                Effect(
                    "pitch_pay",
                    1,
                    banish_name="choose_name:" + m.group(1).strip(),
                    optional=True,
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    # Special-case: Dishonor — on_hit, lose all abilities if trio of named cards controlled.
    if re.search(r"surging strike.*descendent gustwave.*bonds of ancestry", body, re.I) or re.search(
        r"bonds of ancestry.*surging strike.*descendent gustwave", body, re.I
    ):
        triggers.append(
            Trigger(
                "on_hit",
                Effect(
                    "lose_all_abilities",
                    condition="control:Surging Strike,Descendent Gustwave,Bonds of Ancestry",
                    raw="if you control Surging Strike, Descendent Gustwave, and Bonds of Ancestry, lose all abilities",
                ),
                "dishonor: on-hit lose all abilities",
            )
        )

    # On-hit: "when(ever) (this|it) hits[ ...], <effect>".
    for m in re.finditer(r"when(?:ever)? (?:this|it) hits[^,]*,\s*([^.]*)\.?", body, re.I):
        if m.start() > 0 and body[m.start() - 1] == '"':
            continue
        clause = m.group(1).strip()
        # Skip Dishonor's complex conditional — handled by special case above
        if re.search(r"surging strike.*descendent gustwave", clause, re.I) or re.search(
            r"if you control surging strike", clause, re.I
        ):
            continue
        if re.search(r"this gets the chosen name", body, re.I):
            continue
        if re.search(
            r"each hero who doesn't have a card in their arsenal puts the top card of their deck face-down into their arsenal",
            body,
            re.I,
        ):
            continue
        if re.search(
            r"when this hits and you have no cards in your arsenal,?\s*"
            r"you may put a card from your hand face-down into your arsenal",
            body,
            re.I,
        ):
            continue
        if re.search(
            r"when this hits, they reveal a card from their hand",
            body,
            re.I,
        ):
            continue
        if re.search(
            r"when this hits a hero, look at their hand and choose a card\.\s*"
            r"search their hand, deck, and graveyard and banish",
            body,
            re.I,
        ):
            continue
        if re.search(
            r"remove all steam counters from an equipment, item, or weapon they control",
            clause,
            re.I,
        ):
            continue
        if re.search(
            r"you may attack an additional time with this weapon this turn",
            clause,
            re.I,
        ):
            continue
        if re.search(
            r"look at their hand and choose a card",
            clause,
            re.I,
        ):
            continue
        if re.search(
            r"you may put a card from your hand face-down into your arsenal",
            clause,
            re.I,
        ):
            continue
        if re.search(r"you may turn a card in their banished zone face-down", clause, re.I):
            continue
        if re.search(
            r"you may turn a card in their graveyard face-down",
            clause,
            re.I,
        ):
            continue
        if re.search(
            r"at the beginning of your end phase, put the top card of your deck face-up into your arsenal",
            body,
            re.I,
        ):
            continue
        if re.search(
            r"clash with them\.?\s*if you win, destroy the top card of their deck",
            body,
            re.I,
        ):
            continue
        if mode_menus and re.search(r"choose (?:\d|any|1 for each)", clause, re.I):
            continue
        if re.search(
            r"create x runechant tokens,?\s*where x is the damage dealt this way",
            clause,
            re.I,
        ):
            triggers.append(
                Trigger(
                    "on_hit",
                    Effect(
                        "create_token",
                        0,
                        clause,
                        token_name="runechant",
                        banish_name="damage_dealt",
                    ),
                    clause,
                )
            )
            continue
        if re.search(r"\bclash with them\b", clause, re.I):
            continue
        clause = _extend_trigger_clause(body, clause)
        _, effects = _parse_clause_effects(clause, allow_this_turn=True)
        if len(effects) == 1:
            triggers.append(Trigger("on_hit", effects[0], clause))
        else:
            for eff in effects:
                triggers.append(Trigger("on_hit", eff, clause))

    if re.search(
        r"your next .+ attack this turn gets .+(?:attack with it an additional time|attack an additional time with this weapon) this turn",
        body,
        re.I,
    ):
        triggers.append(
            Trigger(
                "on_play",
                Effect("extra_weapon_attack", condition="next_attack_on_hit", raw="next attack extra weapon on hit"),
                "next attack extra weapon on hit",
            )
        )

    # On-attack: Douse in Runeblood — NAA-count Runechants + conditional go again.
    m = re.search(
        r"when this attacks, create runechant tokens equal to the number of non-attack action cards you've played this turn\.?\s*"
        r"if 3 or more runechants are created this way, this gets go again",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_attack",
                Effect(
                    "create_token",
                    0,
                    token_name="runechant",
                    banish_name="naa_played:go_again_if_ge:3",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    # On-play: Sonata Fantasmia — X Runechants (X = pitch) + discard if X >= 6.
    if re.search(r"create x runechant tokens", body, re.I):
        triggers.append(
            Trigger(
                "on_play",
                Effect(
                    "create_token",
                    0,
                    token_name="runechant",
                    banish_name="pitch_value",
                    raw="Create X Runechant tokens",
                ),
                "Create X Runechant tokens",
            )
        )
        if re.search(r"if x is 6 or greater, target hero discards 3 random cards", body, re.I):
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "discard",
                        3,
                        target="opponent",
                        condition="pitch_ge:6",
                        raw="If X is 6 or greater, target hero discards 3 random cards",
                    ),
                    "discard if X >= 6",
                )
            )

    # On-attack: Frontline Scout — optional look at defender's hand.
    m = re.search(
        r"when this attacks, you may look at the defending hero'?s hand",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_attack",
                Effect(
                    "reveal_hand",
                    optional=True,
                    target="opponent",
                    raw=m.group(0)[:80],
                ),
                m.group(0)[:80],
            )
        )

    # On-attack: Alluring Inducement — defender reveals hand.
    m = re.search(r"when this attacks, the defending hero reveals their hand", body, re.I)
    if m:
        triggers.append(
            Trigger(
                "on_attack",
                Effect(
                    "reveal_hand",
                    target="opponent",
                    banish_name="store_revealed",
                    raw=m.group(0)[:80],
                ),
                m.group(0)[:80],
            )
        )

    # On-hit: Bingo — reveal a random card from hand.
    m = re.search(r"when this hits, they reveal a card from their hand", body, re.I)
    if m:
        triggers.append(
            Trigger(
                "on_hit",
                Effect("reveal_hand", 1, target="opponent", raw=m.group(0)[:80]),
                m.group(0)[:80],
            )
        )

    # On-attack: Benefactor — +X power where X is Gold controlled.
    m = re.search(
        r"when this attacks, it gets \+x\{p\}, where x is the number of gold you control",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_attack",
                Effect("power", 0, banish_name="gold_count", raw=m.group(0)[:80]),
                m.group(0)[:80],
            )
        )

    # On-attack: Billowing Mirage — transform ash into Aether Ashwing.
    m = re.search(
        r"when this attacks, transform up to 1 ash you control into an aether ashwings?",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_attack",
                Effect(
                    "transform_token",
                    banish_name="ash",
                    token_name="Aether Ashwing",
                    raw=m.group(0)[:80],
                ),
                m.group(0)[:80],
            )
        )

    # Attacks or defends: optional pitch pay for +power (Flex).
    m = re.search(
        r"when this attacks or defends,?\s*you may pay ((?:\{r\})+|\d+)\.?\s*"
        r"if you do,?\s*(?:it|this) gets \+(\d+)\{p\}",
        body,
        re.I,
    )
    if m:
        eff = Effect(
            "pitch_pay",
            _resource_amount(m.group(1)),
            m.group(0)[:120],
            max_cost=int(m.group(2)),
            banish_name="power",
            optional=True,
        )
        triggers.append(Trigger("on_attack", eff, m.group(0)[:80]))
        triggers.append(Trigger("on_defend", eff, m.group(0)[:80]))

    m = re.search(
        r"when this defends,?\s*you may remove a suspense counter from an aura you control[,.\s]+"
        r"if you do,?\s*gain ((?:\{r\})+|\d+)",
        body,
        re.I,
    )
    if m:
        gain = _resource_amount(m.group(1))
        triggers.append(
            Trigger(
                "on_defend",
                Effect(
                    "remove_counter",
                    1,
                    token_name="suspense",
                    banish_name=f"gain_resources:{gain}",
                    optional=True,
                    raw=m.group(0)[:80],
                ),
                m.group(0)[:80],
            )
        )

    # Attacks or defends: intellect debuff next end phase (Ten Foot Tall and Bulletproof).
    m = re.search(
        r"when this attacks or defends, your hero gets -(\d+)(?:\{i\})? during your next end phase",
        body,
        re.I,
    )
    if m:
        debuff = Effect(
            "intellect_mod",
            -int(m.group(1)),
            target="self",
            condition="next_end_phase",
            raw=m.group(0)[:80],
        )
        triggers.append(Trigger("on_attack", debuff, m.group(0)[:80]))
        triggers.append(Trigger("on_defend", debuff, m.group(0)[:80]))

    m = re.search(
        r"if an earth card was pitched to attack with this,?\s*the attack gets \+(\d+)\{p\}",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_attack",
                Effect(
                    "power",
                    int(m.group(1)),
                    condition="earth_pitched_weapon",
                    raw=m.group(0)[:80],
                ),
                m.group(0)[:80],
            )
        )

    m = re.search(
        r"if a card has been put into your soul this turn,?\s*if you pitch a light card,?\s*"
        r"instead gain that many \{r\} plus 1",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_enters",
                Effect(
                    "pitch_bonus",
                    1,
                    banish_name="light_plus_one",
                    condition="soul_this_turn",
                    raw=m.group(0)[:80],
                ),
                m.group(0)[:80],
            )
        )

    for m in re.finditer(
        r"when this is pitched,?\s*defense reaction cards cost opponents an additional ((?:\{r\})+|\d+)",
        body,
        re.I,
    ):
        triggers.append(
            Trigger(
                "on_pitched",
                Effect(
                    "opponent_cost_increase",
                    _resource_amount(m.group(1)),
                    condition="defense_reaction",
                    target="opponent",
                    raw=m.group(0)[:80],
                ),
                m.group(0)[:80],
            )
        )

    # Tiered aura thresholds (Swarming Gloomveil).
    m = re.search(
        r"if you've played or created (\d+) or more auras this turn, this gets \*\*go again\*\*",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_play",
                Effect(
                    "go_again",
                    condition=f"auras_ge:{int(m.group(1))}",
                    raw=m.group(0)[:80],
                ),
                m.group(0)[:80],
            )
        )
    m = re.search(r"(\d+) or more, this gets \+(\d+)\{p\}", body, re.I)
    if m and re.search(r"played or created \d+ or more auras", body, re.I):
        triggers.append(
            Trigger(
                "on_play",
                Effect(
                    "power",
                    int(m.group(2)),
                    condition=f"auras_ge:{int(m.group(1))}",
                    raw=m.group(0)[:80],
                ),
                m.group(0)[:80],
            )
        )
    m = re.search(
        r'(\d+) or more, this gets "when this hits a hero, ([^"]+)"',
        body,
        re.I,
    )
    if m:
        hit = _parse_trigger_clause(m.group(2).strip(), optional=False)
        if hit is not None and hit.implemented:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        hit.kind,
                        hit.amount,
                        condition=f"auras_ge:{int(m.group(1))}",
                        banish_name=hit.banish_name,
                        target=hit.target,
                        raw=m.group(0)[:120],
                    ),
                    m.group(0)[:80],
                )
            )

    # Combo — If [named card] was the last attack ...
    m = re.search(
        r"combo\s*[-—–]?\s*if (.+?) was the last attack this combat chain,?\s*(.+)",
        body,
        re.I,
    )
    if m:
        card_name = m.group(1).strip()
        clause = m.group(2).strip(" .")
        cond = f"combo_named:{card_name}"
        qm = re.search(r'and "([^"]+)"', clause, re.I)
        if qm:
            inner = re.sub(
                r"^when this hits,?\s*",
                "",
                qm.group(1).strip(),
                flags=re.I,
            )
            hit = _parse_trigger_clause(inner, optional=False)
            if hit is not None and hit.implemented:
                triggers.append(
                    Trigger(
                        "on_hit",
                        Effect(
                            hit.kind,
                            hit.amount,
                            condition=cond,
                            banish_name=hit.banish_name,
                            raw=inner[:80],
                        ),
                        inner[:80],
                    )
                )
            clause = clause[: qm.start()].strip(" ,.")
        for part in re.split(r",?\s+and\s+", clause):
            part = part.strip(" ,.")
            if not part:
                continue
            if re.search(r"go again", part, re.I):
                triggers.append(
                    Trigger(
                        "on_play",
                        Effect("go_again", condition=cond, raw=part[:80]),
                        part[:80],
                    )
                )
                continue
            pm = re.search(r"gets \+(\d+)\{p\}", part, re.I)
            if pm:
                triggers.append(
                    Trigger(
                        "on_play",
                        Effect(
                            "power",
                            int(pm.group(1)),
                            condition=cond,
                            raw=part[:80],
                        ),
                        part[:80],
                    )
                )

    m = re.search(
        r"if this has 10 or more \{p\},?\s*(?:it |this )?gets \*\*overpower\*\*",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_play",
                Effect(
                    "overpower",
                    condition="power_ge_10",
                    raw=m.group(0)[:80],
                ),
                m.group(0)[:80],
            )
        )

    triggers.extend(_parse_conditional_quoted_hit_triggers(body))

    from .effect_parsers import parse_extended_triggers
    from .effect_coverage import parse_passive_triggers

    extended = parse_extended_triggers(text)
    passive = parse_passive_triggers(text)
    extra = (*extended, *passive)
    if mode_menus:
        extra = tuple(
            t
            for t in extra
            if not (t.when == "on_play" and str(t.effect.raw or "").strip().startswith("*"))
        )
    if not extra:
        return tuple(triggers)

    def _trigger_key(t: Trigger) -> tuple:
        return (
            t.when,
            t.effect.kind,
            t.effect.condition,
            t.effect.banish_name,
            t.effect.amount,
            t.effect.token_name,
        )

    seen = {_trigger_key(t) for t in triggers}
    for t in extra:
        key = _trigger_key(t)
        if key not in seen:
            triggers.append(t)
            seen.add(key)
    return tuple(triggers)


def _parse_ability_cost(cost_area: str) -> tuple[int, bool, bool, bool, str, int]:
    """Return (resource_cost, destroy, discard, unsupported_extra, counter_name, counter_cost)."""
    area = cost_area or ""
    destroy = bool(re.search(r"destroy this", area, re.I))
    discard = bool(re.search(r"discard this", area, re.I))
    counter_name = ""
    counter_cost = 0
    cm = re.search(
        r"remove (?:x|(\d+|a|an)) (\w+) counters? from (?:this|(\w+))",
        area,
        re.I,
    )
    if cm:
        qty = cm.group(1)
        counter_cost = 0 if not qty or qty.lower() == "x" else (
            1 if qty in ("a", "an") else int(qty)
        )
        counter_name = cm.group(2).lower()
    residue = re.sub(r"\{r\}", "", area)
    residue = re.sub(r"destroy this", "", residue, flags=re.I)
    residue = re.sub(r"discard this", "", residue, flags=re.I)
    residue = re.sub(r"\{t\}", "", residue)
    residue = re.sub(r"\{g\}", "", residue)
    residue = re.sub(r"turn this face-down", "", residue, flags=re.I)
    residue = re.sub(r"banish this", "", residue, flags=re.I)
    if cm:
        residue = re.sub(re.escape(cm.group(0)), "", residue, flags=re.I)
    residue = residue.strip(" -—–,")
    if re.fullmatch(r"\d+", residue.strip()):
        residue = ""
    unsupported = bool(residue.strip())
    resource_cost = area.count("{r}")
    if resource_cost == 0:
        nm = re.search(r"\b(\d+)\b", area)
        if nm and not counter_name:
            resource_cost = int(nm.group(1))
    return resource_cost, destroy, discard, unsupported, counter_name, counter_cost


def _strip_granted_activated_text(body: str) -> str:
    """Remove activated-ability text granted inside quotes to other objects."""
    return re.sub(
        r'"[^"]*(?:once per turn\s+)?(?:action|instant|attack reaction)\s*[-—–][^"]*"',
        "",
        body,
        flags=re.I,
    )


def _parse_activated_effect(clause: str) -> Effect | None:
    """Parse the effect clause of an activated ability (after the colon)."""
    c = " ".join(clause.lower().split())
    raw = clause.strip()
    if not c:
        return None

    def eff(kind: str, amount: int = 0, **kw) -> Effect:
        text = kw.pop("raw", raw)
        return Effect(kind, amount, text, **kw)

    if re.match(r"^attack\"?\.?$", c):
        return eff("attack")

    if re.search(r"mark target (?:opposing hero|arakni)", c):
        return eff("mark", target="opponent")

    m = re.search(r"this gets \+(\d+)\{d\}", c)
    if m:
        return eff("put_counter", int(m.group(1)), token_name="defense")

    if re.search(r"roll a 6 sided die", c):
        return eff("gain_resources", 0, banish_name="roll_d6_half")

    m = re.search(
        r"target attack action card you control has (\d+) base \{p\}",
        c,
    )
    if m:
        return eff("modify_attack_power", int(m.group(1)), banish_name="set_base")

    if re.search(r"awaken target figment you control", c):
        return eff("create_token", 1, token_name="figment", banish_name="awaken")

    if re.search(r"equip a graphene chelicera token", c):
        return eff("create_banished", banish_name="Graphene Chelicera", playable_banished=True)

    if re.search(
        r"you may turn a face-down arrow in your arsenal face-up|"
        r"turn a face-down card in your arsenal face-up",
        c,
    ):
        return eff("turn_arsenal_face_up", optional="you may" in c)

    if re.search(r"put an arrow card from your hand face-up into your arsenal", c):
        return eff("put_arrow_arsenal", optional="you may" in c)

    if re.search(
        r"until end of turn, opponents must choose this as the target of attacks if able",
        c,
    ):
        return eff("taunt")

    if re.search(r"equip up to 2 draconic daggers from your graveyard", c):
        return eff("return_gy_to_hand", 2, banish_name="draconic dagger")

    m = re.search(
        r"your (?:sword attacks cost|next sword attack costs?) ((?:\{r\})+|\d+) less to activate this turn",
        c,
    )
    if m:
        return eff("next_ability_cost_reduction", _resource_amount(m.group(1)))

    m = re.search(
        r"your next weapon attack this turn costs ((?:\{r\})+|\d+) less to activate",
        c,
    )
    if m:
        return eff("weapon_swing_cost_reduction", _resource_amount(m.group(1)))

    if re.search(r"heroes can't gain \{g\} this turn", c):
        return eff("block_gold_gain")

    if re.search(
        r"the next illusionist attack action card you play this turn loses and can't gain phantasm",
        c,
    ):
        return eff("lose_phantasm")

    if re.search(r"you may attack an additional time with (?:target weapon|this weapon)", c):
        return eff("extra_weapon_attack", optional=True)

    if re.search(r"equip an equipment from a graveyard", c):
        return eff("retrieve_gy", banish_name="equipment")

    if re.search(
        r"until end of turn, effects controlled by opponents don't trigger when their attacks hit",
        c,
    ):
        return eff("block_opponent_hit_effects")

    if re.search(r"each other hero may put a card from their hand on the bottom of their deck", c):
        return eff("put_bottom", target="each_other", optional=True)

    if re.search(r"choose (?:a hero|an opponent|an equipment|any number of heroes)", c):
        return eff("choose_card", banish_name="hero")

    m = re.search(r"equip this with a \+(\d+)\{p\} counter", c)
    if m:
        return eff("put_counter", int(m.group(1)), token_name="power")

    if re.search(r"target hero destroys the top card of their deck", c):
        return eff("destroy_top", 1, target="opponent")

    if re.search(r"return this to your hand", c):
        return eff("return_self_hand")

    if re.search(r"sharpen target sword you control", c):
        return eff("put_counter", 1, token_name="power", banish_name="sword")

    if re.search(r"banish a card from your hand", c):
        rider = ""
        if re.search(r"if it'?s a shadow card, draw a card", c):
            rider = "shadow_draw"
        return eff("destroy_hand", 1, target="self", banish_name=f"banish:{rider}" if rider else "banish")

    if re.search(r"your hero gets \+1(?: until end of turn| this turn)?", c):
        return eff("gain_life", 1, target="self")

    if re.search(r"defense reactions can't be played this turn", c):
        return eff("block_defense_reactions", condition="turn")

    if re.search(r"\{t\} target hero or ally", c):
        return eff("destroy_item", target="opponent")

    if re.search(
        r"reveal cards from the top of your deck until you reveal an attack action card",
        c,
    ):
        penalty = "-7"
        pm = re.search(r"otherwise, this gets -(\d+)\{p\}", c)
        if pm:
            penalty = f"-{pm.group(1)}"
        return eff("search", banish_name=f"attack_action_reveal:{penalty}")

    if re.search(
        r"banish a wizard non-attack action card from your hand with an effect that deals arcane damage equal to x",
        c,
    ):
        return eff(
            "banish_hand_play",
            banish_name="wizard:non-attack:arcane_x",
            playable_banished=True,
            optional=True,
        )

    if re.search(r"target defending hero banishes a card from their hand", c):
        return eff("banish_graveyard", 1, target="opponent", banish_name="hand")

    if re.search(r"each hero draws a card", c):
        return eff("draw", 1, target="each_hero")

    if re.search(r"look at the top card of an opposing hero's deck", c):
        return eff("look_deck", 1, target="opponent")

    if re.search(r"\{t\} all heroes and allies", c):
        return eff("destroy_item", target="each_hero")

    if re.search(r"return a card from your arsenal to your hand", c):
        return eff("return_arsenal_hand", optional="you may" in c)

    if re.search(r"\{u\} target permanent", c):
        return eff("destroy_item", target="opponent")

    if re.search(
        r"return target earth action card or earth instant card from your graveyard to your hand",
        c,
    ):
        return eff("return_gy_to_hand", banish_name="earth")

    if re.search(r"put all cards from your pitch zone on top of your deck", c):
        return eff("put_hand_top", banish_name="pitch_to_deck")

    if re.search(r"look at target hero's hand", c):
        return eff("reveal_hand", target="opponent")

    if re.search(r"target weapon attack you control wagers a gold token", c):
        return eff("wager", token_name="gold")

    if re.search(r"put a mechanologist item with cost 0 or 1 from your hand into the arena", c):
        return eff("put_item_in_arena", 1, optional="you may" in c)

    if re.search(
        r"shuffle up to 3 arrows with different names from your graveyard into your deck",
        c,
    ):
        return eff("return_gy_to_deck", 3, banish_name="arrow")

    if re.search(
        r"the next time an attack action card you control hits a hero this turn, it deals 1 damage",
        c,
    ):
        return eff("grant_hit_bonus", 1)

    if re.search(r"target attack action card defending an assassin attack gets -1\{d\}", c):
        return eff("reduce_defense", 1, target="opponent")

    if re.search(r"pitch the top card of your deck", c):
        return eff("pitch_deck_top")

    if re.search(r"equip an off-hand with proclamation in its name from your inventory", c):
        return eff("equip_inventory", banish_name="proclamation off-hand")

    if re.search(
        r"you may add an action card from your arsenal to the active chain link as a defending card",
        c,
    ):
        return eff("chain_defend", banish_name="arsenal", optional=True)

    if re.search(r"add this to the active chain link as a defending card", c):
        base_def = 0
        m_def = re.search(r"it has (\d+) base \{d\}", c)
        if m_def:
            base_def = int(m_def.group(1))
        return eff("chain_defend", base_def, banish_name="self")

    if re.search(r"turn target face-down equipment you have equipped face-up", c):
        return eff("turn_equipment_face_up")

    if re.search(r"look at the top 3 cards of the event deck", c):
        return eff("look_deck", 3, banish_name="event_deck")

    if re.match(r"^name a card\.?$", c):
        return eff("name_card")

    if re.search(r"transform into levia, redeemed", c):
        return eff("transform_hero", banish_name="Levia, Redeemed")

    if re.search(r"the next card you reveal this turn has its grade increased by 1", c):
        return eff("grade_increase", 1)

    if re.search(
        r"return this to the arena under its owner's control, unequipped, tapped, and with a steam counter",
        c,
    ):
        return eff("return_arena_tapped", token_name="steam")

    if re.search(r"each hero draws up to their", c):
        return eff("draw", banish_name="up_to_intellect", target="each_hero")

    if re.search(
        r"until the start of your next turn, the only actions heroes may play or activate are weapon and attacks actions",
        c,
    ):
        return eff("play_restriction", banish_name="weapon_attack_only")

    if re.search(r"each opponent chooses an item or landmark they control", c):
        return eff("choose_card", target="opponent", banish_name="destroy_item_landmark")

    if re.search(
        r"target 3 action cards with different names in your banished zone and choose one at random",
        c,
    ):
        return eff("random_banished_pick", 3, banish_name="action")

    if re.search(r"turn the banished card face-up", c):
        return eff("turn_banished_face", 1, banish_name="face_up")

    if re.search(r"turn x target cards in a banished zone face-down", c):
        return eff("turn_banished_face", banish_name="face_down:variable")

    if re.search(
        r"the next time target weapon deals x or less damage to you this turn, deal that much damage to its controller",
        c,
    ):
        return eff("damage_redirect", banish_name="weapon_reflect")

    if re.search(
        r"the next time another target hero would be dealt damage this turn, instead that damage is dealt to yoji and prevent 1",
        c,
    ):
        return eff("damage_redirect", 1, banish_name="yoji_shield")

    if re.search(
        r"the next prevention effect that prevents \{p\} damage this turn, prevents 1 less",
        c,
    ):
        return eff("prevention_reduction", 1)

    if re.search(
        r"you may reveal a reviled attack action card from your inventory and put it into your hand",
        c,
    ):
        return eff("inventory_to_hand", banish_name="reviled attack", optional=True)

    ext = _parse_extended_clause(clause, optional=False, condition="")
    if ext is not None and ext.implemented:
        return ext
    return None


@functools.lru_cache(maxsize=8192)
def parse_activated_abilities(text: str) -> tuple[ActivatedAbility, ...]:
    """Parse "[Once per Turn] Action|Instant -- {cost}: <effect>" abilities."""
    body = _clean(text)
    if not body.strip():
        return ()

    body = _strip_granted_activated_text(body)

    abilities: list[ActivatedAbility] = []
    pattern = re.compile(
        r"(once per turn\s+)?(?:action|instant|attack reaction)\s*(?:[-—–]+\s*)?([^:]*?):\s*([^.]+?)(?:\.|$)",
        re.I,
    )
    for m in pattern.finditer(body):
        once = bool(m.group(1))
        cost_area = m.group(2) or ""
        eff_text = (m.group(3) or "").strip()
        full_ability = m.group(0) or ""
        kind_match = re.search(
            r"(?:once per turn\s+)?(action|instant|attack reaction)\s*[-—–]",
            full_ability,
            re.I,
        )
        ability_kind = (kind_match.group(1) if kind_match else "action").lower()
        uses_ap = ability_kind in ("action", "attack reaction")

        cost, destroy, discard, unsupported, counter_name, counter_cost = _parse_ability_cost(cost_area)
        if destroy and re.search(r"roll a 6 sided die", eff_text, re.I):
            m_roll = re.search(
                r"roll a 6 sided die\.?\s*gain \{r\} equal to half the number rolled, rounded down",
                body,
                re.I,
            )
            if m_roll:
                eff_text = m_roll.group(0)
        effect = _parse_dagger_damage_text(eff_text, full_context=full_ability)
        if effect is None:
            effect = (
                _parse_activated_effect(eff_text)
                or _parse_next_attack_power(eff_text)
                or _parse_extended_clause(eff_text, optional=False, condition="")
                or _parse_effect(eff_text, allow_this_turn=True)
            )
        if effect.kind == "attack" and unsupported:
            unsupported = False
        if counter_name and effect.kind == "unimplemented":
            effect = Effect("remove_counter", counter_cost, eff_text, token_name=counter_name)
            unsupported = False
        if unsupported and effect.kind not in SUPPORTED_EFFECTS:
            effect = Effect("unimplemented", raw=eff_text)

        grants_go_again = "go again" in (m.group(0) or "").lower()

        abilities.append(
            ActivatedAbility(
                effect=effect,
                cost=cost,
                uses_action_point=uses_ap,
                once_per_turn=once,
                destroy_source=destroy,
                discard_source=discard,
                counter_name=counter_name,
                counter_cost=counter_cost,
                grants_go_again=grants_go_again,
                raw=" ".join(m.group(0).split())[:90],
            )
        )
    return tuple(abilities)


@functools.lru_cache(maxsize=8192)
def parse_play_costs(text: str) -> tuple[Effect, ...]:
    """Optional additional costs to play a card (Heave, Scrap, pay {r})."""
    body = _clean(text)
    if not body.strip():
        return ()

    costs: list[Effect] = []
    m = re.search(r"\bheave (\d+)\b", body, re.I)
    if m:
        costs.append(Effect("heave", int(m.group(1)), optional=True, raw=f"heave {m.group(1)}"))

    if re.search(r"(?:^|\.\s+)scrap(?:\s|\.|$)", body, re.I):
        costs.append(Effect("scrap", optional=True, raw="scrap"))

    if re.search(r"as an additional cost to play this,?\s*you may charge your soul", body, re.I):
        costs.append(Effect("put_soul", optional=True, raw="charge soul"))

    if re.search(
        r"as an additional cost to play this,?\s*you may (?:\*\*)?charge(?:\*\*)? your hero'?s soul",
        body,
        re.I,
    ):
        costs.append(Effect("put_soul", optional=True, raw="charge hero soul"))

    if re.search(r"as an additional cost to play this,?\s*banish x items from your graveyard", body, re.I):
        costs.append(
            Effect(
                "banish_gy_variable",
                banish_name="item",
                optional=False,
                raw="banish X items from your graveyard",
            )
        )

    if re.search(
        r"as an additional cost to play this,?\s*banish x cards from your hero'?s soul",
        body,
        re.I,
    ):
        costs.append(
            Effect(
                "destroy_hand",
                banish_name="soul:variable",
                optional=False,
                raw="banish X cards from soul",
            )
        )

    if re.search(
        r"as an additional cost to play this,?\s*banish a random card from your hand",
        body,
        re.I,
    ):
        costs.append(
            Effect(
                "destroy_hand",
                1,
                banish_name="random",
                target="self",
                optional=False,
                raw="banish random card from hand",
            )
        )

    if re.search(
        r"as an additional cost to play this,?\s*discard a random card",
        body,
        re.I,
    ):
        costs.append(
            Effect(
                "destroy_hand",
                1,
                banish_name="random",
                target="self",
                optional=False,
                raw="discard random card",
            )
        )

    if re.search(
        r"as an additional cost to play this,?\s*banish any number of cards with 6 or more \{p\} from your hand",
        body,
        re.I,
    ):
        costs.append(
            Effect(
                "destroy_hand",
                banish_name="power6:variable",
                optional=False,
                raw="banish 6+ power cards from hand",
            )
        )

    if re.search(r"as an additional cost to play this,?\s*you may pay ((?:\{r\})+|\d+)", body, re.I):
        m = re.search(r"as an additional cost to play this,?\s*you may pay ((?:\{r\})+|\d+)", body, re.I)
        txt = m.group(1)
        rider = body[m.end():]
        if "{r}" in txt:
            amount = _resource_amount(txt)
            extra = Effect(
                "pitch_pay",
                amount,
                optional=True,
                raw=m.group(0)[:120],
            )
            if re.search(r"attacks an additional target", rider, re.I):
                extra = Effect(
                    "pitch_pay",
                    amount,
                    optional=True,
                    banish_name="extra_target",
                    raw=m.group(0)[:120],
                )
            costs.append(extra)
        else:
            amount = int(txt)
            extra = Effect(
                "additional_pay",
                amount,
                optional=True,
                raw=m.group(0)[:60],
            )
            if re.search(r"attacks an additional target", rider, re.I):
                extra = Effect(
                    "additional_pay",
                    amount,
                    optional=True,
                    banish_name="extra_target",
                    raw=m.group(0)[:60],
                )
            costs.append(extra)

    m = re.search(
        r"additional cost:?\s*banish a card from your hand",
        body,
        re.I,
    )
    if m:
        extra = Effect(
            "additional_pay",
            0,
            optional=False,
            raw=m.group(0)[:60],
        )
        if re.search(r"target an additional hero|additional target", body, re.I):
            extra = Effect(
                "additional_pay",
                0,
                optional=False,
                banish_name="extra_target",
                raw=m.group(0)[:60],
            )
        costs.append(extra)

    m = re.search(
        r"as an additional cost to play this,?\s*destroy x hyper drivers you control",
        body,
        re.I,
    )
    if m:
        costs.append(
            Effect(
                "destroy_item",
                banish_name="hyper_driver",
                optional=False,
                raw=m.group(0)[:80],
            )
        )

    return tuple(costs)


def parse_all_interactions(
    text: str,
    keywords: tuple[str, ...] | list[str] = (),
    card_types: tuple[str, ...] | list[str] = (),
) -> dict[str, tuple]:
    """Full parse used by coverage scanner: triggers, abilities, modifiers, keywords, base damage."""
    from .effect_coverage import parse_action_damage, parse_keyword_triggers

    trigs = parse_triggers(text)
    abilities = parse_activated_abilities(text)
    mods = parse_play_modifiers(text)
    kw_trigs = parse_keyword_triggers(keywords)
    play_costs = parse_play_costs(text)
    is_attack = "attack_action" in card_types
    is_utility = "utility_action" in card_types
    damage, arcane = parse_action_damage(text, 0, is_attack) if (is_attack or is_utility) else (0, False)
    return {
        "triggers": trigs,
        "abilities": abilities,
        "modifiers": mods,
        "keyword_triggers": kw_trigs,
        "play_costs": play_costs,
        "action_damage": (damage, arcane),
    }
