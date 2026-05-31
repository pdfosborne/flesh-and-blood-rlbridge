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
    "enable_gy_play",
    "fusion",
    "grant_hit_bonus",
    "hit_bonus_damage",
    "create_token_triple",
    "upkeep_or_destroy",
    "play_power_cap",
    "gy_to_bottom",
    "grant_light_block",
    "transform_equip",
    "counts_as_gold",
    "play_from_deck_top",
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
        re.search(r"(?:once per turn\s+)?(?:action|instant)\s*[-—–]", clause, re.I)
    )


def _resource_amount(text: str) -> int:
    """Count {r} symbols or parse a numeric resource cost."""
    txt = str(text or "")
    if "{r}" in txt:
        return max(1, txt.count("{r}"))
    m = re.search(r"\d+", txt)
    return max(1, int(m.group(0))) if m else 1


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

    m = re.search(r"you may pay ((?:\{r\})+|\d+)\.?\s*if you do", c)
    if m and "choose" not in c:
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

    if c == "attack" or c.startswith("attack "):
        return eff("attack")

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
        return eff("gain_gold", int(m.group(1)))

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

    m = re.search(r"the next action card you defend with this turn gets \+(\d+)\{d\}", c)
    if m:
        return eff("next_defense_bonus", int(m.group(1)))

    m = re.search(r"gain (\d+)\{h\}", c)
    if m:
        return eff("gain_life", int(m.group(1)))

    m = re.search(r"(?:they|that hero) lose(?:s)? (\d+)\{h\}", c)
    if m:
        return eff("lose_life", int(m.group(1)), target="opponent")

    m = re.search(
        r"you may attack with it an additional time this turn",
        c,
    )
    if m:
        return eff("extra_weapon_attack", optional=True)

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
        return eff("prevent_damage", int(m.group(1)), condition="power_damage")

    m = re.search(
        r"the next time you would be dealt (\d+) or less damage this turn, prevent it",
        c,
    )
    if m:
        return eff("prevent_damage", int(m.group(1)), condition="damage_le")

    m = re.search(
        r"the next time you would be dealt damage this turn, prevent (\d+) of that damage",
        c,
    )
    if m:
        return eff("prevent_damage", int(m.group(1)))

    m = re.search(
        r"prevent the next (\d+) damage that would be dealt to you this turn(?: by a shadow source)?",
        c,
    )
    if m:
        cond = "shadow" if "shadow source" in c else ""
        return eff("prevent_damage", int(m.group(1)), condition=cond)

    m = re.search(
        r"the next (\d+) times you would be dealt damage this turn, prevent (\d+) of that damage",
        c,
    )
    if m:
        return eff(
            "prevent_damage",
            int(m.group(2)),
            condition=f"per_hit:{int(m.group(1))}",
        )

    m = re.search(
        r"the next time a shadow source would deal damage this turn, prevent (\d+) of that damage",
        c,
    )
    if m:
        return eff("prevent_damage", int(m.group(1)), condition="shadow")

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
        r"if you have a base (head|chest|arms|legs) equipped, transform it into this, then equip this",
        c,
    )
    if m:
        return eff("transform_equip", banish_name=m.group(1))

    if re.search(r"this counts as a gold", c):
        return eff("counts_as_gold")

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
        r"their first action during their next turn costs an additional ((?:\{r\})+|\d+) to play or activate",
        c,
    )
    if m:
        return eff("opponent_cost_increase", _resource_amount(m.group(1)), target="opponent")

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

    if re.search(r"cards and abilities cost opponents an additional \{r\}", c):
        return eff("opponent_cost_increase", 1)

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

    if re.search(r"put a card from their hand on top of their deck", c):
        return eff("put_bottom", target="opponent")

    m = re.search(r"discards? a card unless they pay ((?:\{r\})+|\d+)", c)
    if m:
        txt = m.group(1)
        n = txt.count("{r}") if "{r}" in txt else int(txt)
        return eff("unless_pay", n, target="opponent", banish_name="discard")

    if re.search(r"for each", c):
        m = re.search(r"for each (.+?), create an? (.+?) token", c)
        if m:
            return eff("for_each", token_name=m.group(2).strip(), raw=raw)

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
        return eff("banish_combo", optional=True, raw=raw)

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


def _parse_next_attack_power(clause: str) -> Effect | None:
    c = " ".join(clause.lower().split())
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
        return Effect("next_attack_power", int(m.group(1)), clause.strip())
    m = re.search(
        r"the next attack action card(?: with cost (\d+) or less)? you play this turn gets \+(\d+)\s*(?:\{p\}|power)",
        c,
    )
    if m:
        max_cost = int(m.group(1)) if m.group(1) else 99
        return Effect("next_attack_power", int(m.group(2)), clause.strip(), max_cost=max_cost)
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
            optional=optional,
            condition=condition,
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
    return tuple(mods)


_MODE_MENU_SPECS: tuple[tuple[str, str, str, int], ...] = (
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
        triggers.append(Trigger("when_fused", _parse_effect(m.group(1)), m.group(1).strip()))

    # On-attack: "when (this|it) attacks, <effect>".
    for m in re.finditer(r"when (?:this|it) attacks,?\s*([^.]*)\.?", body, re.I):
        clause = m.group(1)
        low = clause.lower()
        if "fused" in low and "create" not in low and "banish" not in low:
            continue
        if mode_menus and re.search(r"choose (?:\d|any|1 for each)", low):
            continue
        _, effects = _parse_clause_effects(clause)
        for eff in effects:
            if mode_menus and eff.kind == "unimplemented":
                continue
            triggers.append(Trigger("on_attack", eff, clause.strip()))

    # On-hit: "when(ever) (this|it) hits[ ...], <effect>".
    for m in re.finditer(r"when(?:ever)? (?:this|it) hits[^,]*,\s*([^.]*)\.?", body, re.I):
        clause = m.group(1).strip()
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
        _, effects = _parse_clause_effects(clause, allow_this_turn=True)
        if len(effects) == 1:
            triggers.append(Trigger("on_hit", effects[0], clause))
        else:
            for eff in effects:
                triggers.append(Trigger("on_hit", eff, clause))

    if re.search(
        r"your next .+ attack this turn gets .+attack with it an additional time this turn",
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
    seen = {(t.when, t.effect.kind, t.effect.raw, t.threshold) for t in triggers}
    for t in extra:
        key = (t.when, t.effect.kind, t.effect.raw, t.threshold)
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
    cm = re.search(r"remove (\d+|a|an) (\w+) counters? from this", area, re.I)
    if cm:
        counter_cost = 1 if cm.group(1) in ("a", "an") else int(cm.group(1))
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


@functools.lru_cache(maxsize=8192)
def parse_activated_abilities(text: str) -> tuple[ActivatedAbility, ...]:
    """Parse "[Once per Turn] Action|Instant -- {cost}: <effect>" abilities."""
    body = _clean(text)
    if not body.strip():
        return ()

    abilities: list[ActivatedAbility] = []
    pattern = re.compile(
        r"(once per turn\s+)?(action|instant)\s*[-—–]+\s*([^:]*?):\s*([^.]+?)(?:\.|$)",
        re.I,
    )
    for m in pattern.finditer(body):
        once = bool(m.group(1))
        uses_ap = m.group(2).lower() == "action"
        cost_area = m.group(3) or ""
        eff_text = (m.group(4) or "").strip()

        cost, destroy, discard, unsupported, counter_name, counter_cost = _parse_ability_cost(cost_area)
        effect = (
            _parse_next_attack_power(eff_text)
            or _parse_extended_clause(eff_text, optional=False, condition="")
            or _parse_effect(eff_text, allow_this_turn=True)
        )
        if effect.kind == "attack" and unsupported:
            unsupported = False
        if counter_name and counter_cost and effect.kind == "unimplemented":
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

    if re.search(r"as an additional cost to play this,?\s*you may pay ((?:\{r\})+|\d+)", body, re.I):
        m = re.search(r"as an additional cost to play this,?\s*you may pay ((?:\{r\})+|\d+)", body, re.I)
        txt = m.group(1)
        amount = _resource_amount(txt) if "{r}" in txt else int(txt)
        rider = body[m.end():]
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
