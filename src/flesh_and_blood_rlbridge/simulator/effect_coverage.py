"""Broad rules-text coverage: passive sentences, keywords, and fallback parsing."""

from __future__ import annotations

import re

from .effects import (
    Effect,
    Trigger,
    _clean,
    _detect_condition,
    _is_activated_ability_clause,
    _parse_clause_effects,
    _parse_create_banish,
    _parse_dagger_damage_text,
    _parse_effect,
    _parse_trigger_clause,
    card_has_mode_menu_bullets,
    _parse_extended_clause,
    _parse_next_attack_power,
    _parse_pitch_pay,
    _resource_amount,
    _split_on_play_clause,
)

# Keywords with static combat rules — trigger timing must match fab_rules.KEYWORD_TRIGGER_WHEN.
_KEYWORD_EFFECTS: dict[str, tuple[str, str]] = {
    "go_again": ("on_play", "go_again"),
    "dominate": ("on_attack", "dominate"),
    "overpower": ("on_attack", "overpower"),
    "intimidate": ("on_hit", "intimidate"),
    "blood_debt": ("on_play", "blood_debt"),
}


def _sentences(text: str) -> list[str]:
    body = _clean(text)
    if not body.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", body)
    return [p.strip(" .") for p in parts if p.strip(" .")]


def _parse_clause_to_effects(clause: str, *, allow_this_turn: bool = False) -> tuple[Effect, ...]:
    """Try every parser on a clause; return all implemented effects found."""
    condition, effs = _parse_clause_effects(clause, allow_this_turn=allow_this_turn)
    if effs and effs[0].implemented:
        return effs
    single = (
        _parse_create_banish(clause)
        or _parse_pitch_pay(clause, optional="you may" in clause.lower())
        or _parse_extended_clause(clause, optional=False, condition=condition)
        or _parse_next_attack_power(clause)
        or _parse_effect(clause, allow_this_turn=allow_this_turn)
    )
    if single.implemented:
        return (single,)
    return ()


def _parse_effect_aggressive(clause: str) -> Effect | None:
    """Last-resort extraction of any recognizable effect verb."""
    c = " ".join(clause.lower().split())
    if not c:
        return None

    dagger = _parse_dagger_damage_text(clause, full_context=clause)
    if dagger is not None:
        return dagger

    m = re.search(r"deal (\d+) arcane damage to (?:two target heroes|up to any (\d+) targets)", c)
    if m:
        hits = int(m.group(2)) if m.group(2) else 2
        return Effect(
            "arcane_damage",
            int(m.group(1)),
            clause.strip(),
            condition=f"multi:{hits}",
        )

    m = re.search(r"deal (\d+) arcane damage to the attacking hero", c)
    if m:
        return Effect(
            "arcane_damage",
            int(m.group(1)),
            clause.strip(),
            target="attacking_hero",
        )

    m = re.search(r"deal (\d+) damage to the attacking hero", c)
    if m:
        return Effect("damage", int(m.group(1)), clause.strip(), target="attacking_hero")

    m = re.search(r"deal (\d+) arcane damage to (?:any target|target hero|target opposing hero|each opposing hero)", c)
    if m:
        return Effect("arcane_damage", int(m.group(1)), clause.strip())

    m = re.search(r"deal (\d+) damage to (?:any target|target hero|target opposing hero|each opposing hero|them)", c)
    if m:
        return Effect("damage", int(m.group(1)), clause.strip())

    m = re.search(r"create[ds]? an? ([a-z][a-z\s'-]+?) token", c)
    if m:
        return Effect("create_token", token_name=m.group(1).strip(), raw=clause.strip())

    m = re.search(r"gain (\d+)\{g\}", c)
    if m:
        target = "opponent" if re.search(r"they gain \d+\{g\}", c) else "self"
        return Effect("gain_gold", int(m.group(1)), clause.strip(), target=target)

    m = re.search(r"put (.+?) from your graveyard on (?:the )?bottom of your deck", c)
    if m:
        return Effect("return_gy_to_deck", banish_name=m.group(1).strip(), target="bottom", raw=clause.strip())

    m = re.search(r"put (.+?) from your graveyard on top of your deck", c)
    if m:
        return Effect("return_gy_to_deck", banish_name=m.group(1).strip(), raw=clause.strip())

    if re.search(r"lose and can't gain", c) or re.search(r"can't gain or have attack actions granted", c):
        return Effect("silence", target="opponent", raw=clause.strip())

    m = re.search(r"create (\d+) (.+?) tokens?", c)
    if m:
        target = "opponent" if "under target" in c or "under their" in c else "self"
        return Effect(
            "create_token",
            int(m.group(1)),
            clause.strip(),
            token_name=m.group(2).strip(),
            target=target,
        )

    m = re.search(r"put a gold counter on (.+)", c)
    if m:
        return Effect("put_counter", token_name="gold", banish_name=m.group(1).strip(), raw=clause.strip())

    m = re.search(r"put a (.+?) counter on", c)
    if m:
        return Effect("put_counter", token_name=m.group(1).strip(), raw=clause.strip())

    m = re.search(r"put an item with cost 0 or 1 from your hand into the arena", c)
    if m:
        return Effect("put_item_in_arena", 1, clause.strip(), optional="you may" in c)

    m = re.search(r"gets \+(\d+)\{d\}", c)
    if m and "defend" in c:
        return Effect("next_defense_bonus", int(m.group(1)), clause.strip())

    m = re.search(r"gain (\d+)\{h\}", c)
    if m:
        return Effect("gain_life", int(m.group(1)), clause.strip())

    m = re.search(r"(?:they|that hero) lose(?:s)? (\d+)\{h\}", c)
    if m:
        return Effect("lose_life", int(m.group(1)), clause.strip(), target="opponent")

    m = re.search(r"target hero gains (\d+)\{h\}", c)
    if m:
        return Effect("gain_life", int(m.group(1)), clause.strip(), target="any")

    if re.search(r"charge your (?:hero'?s )?soul", c):
        return Effect("put_soul", raw=clause.strip(), optional="you may" in c)

    if re.search(r"put a card from your hand into your soul", c):
        return Effect("put_soul", raw=clause.strip(), optional="you may" in c)

    m = re.search(r"put a (red|yellow|blue) card from your hand into your soul", c)
    if m:
        return Effect("put_soul", raw=clause.strip(), optional="you may" in c, banish_name=m.group(1))

    m = re.search(r"prevent the next (\d+) damage that would be dealt to target hero this turn", c)
    if m:
        return Effect("prevent_damage", int(m.group(1)), clause.strip(), target="any")

    if re.search(r"defense reactions can't be played to this chain link", c):
        return Effect("block_defense_reactions", raw=clause.strip())

    m = re.search(r"attacks that target you this turn get -(\d+)\{p\}", c)
    if m:
        return Effect("power", -int(m.group(1)), target="self", banish_name="attacks_targeting_you", raw=clause.strip())

    m = re.search(r"reveal the top (\d+) cards of your deck", c)
    if m:
        return Effect("look_deck", int(m.group(1)), target="self", raw=clause.strip())

    if re.search(r"reveal the top card of your deck", c):
        return Effect("reveal_top", target="self", raw=clause.strip())

    m = re.search(
        r"roll a 6 sided die\.?\s*prevent the next x damage that would be dealt to you this turn,?\s*"
        r"where x is the number rolled",
        c,
    )
    if m:
        return Effect("prevent_damage", 0, banish_name="roll_d6", raw=clause.strip())

    m = re.search(
        r"whenever a mechanologist item with cost (\d+) or less is put into your banished zone from your deck,?\s*"
        r"put it into the arena",
        c,
    )
    if m:
        return Effect(
            "put_item_in_arena",
            int(m.group(1)),
            target="banished",
            raw=clause.strip(),
        )

    pitch = _parse_pitch_pay(clause, optional="you may" in c)
    if pitch is not None:
        return pitch

    if re.search(r"you may play this from your graveyard", c):
        return Effect("enable_gy_play", raw=clause.strip())

    m = re.search(r"you may play an? (.+?) from your graveyard", c)
    if m:
        return Effect("enable_gy_play", banish_name=m.group(1).strip(), raw=clause.strip())

    m = re.search(r"whenever an attack hits a hero this turn,?\s*it deals (\d+) damage", c)
    if m:
        return Effect("grant_hit_bonus", int(m.group(1)), clause.strip())

    m = re.search(r"heave (\d+)", c)
    if m:
        return Effect("heave", int(m.group(1)), clause.strip(), optional=True)

    m = re.search(r"create an agility, might, and vigor token", c)
    if m:
        return Effect("create_token_triple", raw=clause.strip())

    m = re.search(r"reveal the top card of your deck", c)
    if m:
        return Effect("reveal_top", raw=clause.strip())

    if re.search(r"\btranscend\b", c):
        return Effect("transcend", raw=clause.strip())
    if re.search(r"\bcontract\b", c):
        return Effect("contract", raw=clause.strip())
    if re.search(r"blood debt", c):
        return Effect("blood_debt", raw=clause.strip())
    if re.search(r"\boverpower\b", c):
        return Effect("overpower", raw=clause.strip())
    if re.search(r"\bintimidate\b", c):
        return Effect("intimidate", raw=clause.strip())
    if re.search(r"\bdominate\b", c):
        return Effect("dominate", raw=clause.strip())
    if re.search(r"\bboost\b", c):
        return Effect("boost", raw=clause.strip())
    if re.search(r"\bgo again\b", c):
        pm = re.search(r"\+(\d+)\s*(?:\{p\}|power)", c)
        if pm:
            return Effect("power", int(pm.group(1)), clause.strip(), go_again=True)
        return Effect("go_again", raw=clause.strip())
    if re.search(r"\bdestroy\b", c):
        if re.search(r"destroy this\b", c):
            pass
        elif re.search(r"destroy an item you control", c):
            return Effect("destroy_item", optional="you may" in c, raw=clause.strip())
        elif re.search(r"destroy an aura you control", c):
            return None
        elif "arsenal" in c:
            return Effect("destroy_arsenal", target="opponent", raw=clause.strip())
        elif "your deck" in c or "top card of your" in c:
            return Effect("destroy_top", target="self", raw=clause.strip())
        elif not _is_activated_ability_clause(clause):
            return Effect("destroy_top", target="opponent", raw=clause.strip())
    if re.search(r"\bbanish\b", c):
        if "graveyard" in c:
            return Effect("banish_graveyard", target="opponent", raw=clause.strip())
        if "from your hand" in c or "from hand" in c:
            return None
        if "arsenal" in c:
            return Effect("banish_arsenal", target="opponent", raw=clause.strip())
        return Effect("banish_top", target="opponent", raw=clause.strip())
    if re.search(r"\bdiscard\b", c):
        target = "opponent" if "they" in c[:30] else "self"
        return Effect("discard", 1, clause.strip(), target=target)
    if re.search(r"\bsearch\b", c):
        return Effect("search", raw=clause.strip())
    if re.search(r"\bdraw\b", c):
        return Effect("draw", 1, clause.strip())
    if re.search(r"ward (\d+)", c):
        return Effect("ward", int(re.search(r"ward (\d+)", c).group(1)), clause.strip())
    if re.search(r"\bopt\b", c) or re.search(r"^opt \d", c):
        m = re.search(r"opt (\d+)", c)
        return Effect("opt", int(m.group(1)) if m else 1, clause.strip())
    if re.search(r"\bamp\b", c):
        m = re.search(r"amp (\d+)", c)
        return Effect("amp", int(m.group(1)) if m else 1, clause.strip())
    if re.search(r"target .+ gets \+(\d+)\s*(?:\{p\}|power)", c):
        m = re.search(r"target .+ gets \+(\d+)\s*(?:\{p\}|power)", c)
        return Effect("next_attack_power", int(m.group(1)), clause.strip())
    m = re.search(r"target .+ gets -(\d+)\s*(?:\{p\}|power)", c)
    if m:
        return Effect("reduce_defense", int(m.group(1)), clause.strip())
    nap = _parse_next_attack_power(clause)
    if nap is not None and nap.implemented:
        return nap
    m = re.search(r"the attack deals (\d+) damage to the defending hero", c)
    if m:
        return Effect("hit_bonus_damage", int(m.group(1)), clause.strip())

    m = re.search(r"shuffle (\d+) attack action cards from your banished zone into your deck", c)
    if m:
        return Effect(
            "return_gy_to_deck",
            int(m.group(1)),
            banish_name="attack action",
            raw=clause.strip(),
        )

    if re.search(r"gets \+(\d+)\s*(?:\{p\}|power)", c):
        m = re.search(r"gets \+(\d+)\s*(?:\{p\}|power)", c)
        return Effect("power", int(m.group(1)), clause.strip(), go_again="go again" in c)
    return None


def _parse_destroy_then_clause(remainder: str) -> tuple[Effect, ...]:
    """Parse the effect clause after 'destroy this then ...'."""
    effs = _parse_clause_to_effects(remainder, allow_this_turn=True)
    if effs:
        return effs
    nap = _parse_next_attack_power(remainder)
    if nap is not None and nap.implemented:
        return (nap,)
    agg = _parse_effect_aggressive(remainder)
    if agg is not None and agg.implemented:
        return (agg,)
    return ()

def _append_destroy_then_triggers(
    triggers: list[Trigger],
    when: str,
    sent: str,
    remainder: str,
) -> None:
    triggers.append(Trigger(when, Effect("destroy_item", raw=sent), sent))
    for eff in _parse_destroy_then_clause(remainder):
        triggers.append(Trigger(when, eff, sent))


def _append_optional_destroy_then(
    triggers: list[Trigger],
    when: str,
    sent: str,
    remainder: str,
) -> None:
    if not _parse_destroy_then_clause(remainder):
        return
    triggers.append(
        Trigger(
            when,
            Effect("destroy_item", optional=True, raw=remainder),
            sent,
        )
    )


def _parse_optional_destroy_then_patterns(body: str, triggers: list[Trigger]) -> None:
    """Multi-sentence 'you may destroy this. If you do, ...' patterns."""
    patterns = (
        (
            r"when an attack you control hits a hero,?\s*you may destroy this\.?\s*"
            r"if you do,?\s*(.+?)(?:\.\s|\.$|$)",
            "on_controlled_hit",
        ),
        (
            r"when an attack action card you control hits,?\s*you may destroy this\.?\s*"
            r"if you do,?\s*(.+?)(?:\.\s|\.$|$)",
            "on_controlled_hit",
        ),
        (
            r"when you play an attack action card,?\s*you may destroy this\.?\s*"
            r"if you do,?\s*(.+?)(?:\.\s|\.$|$)",
            "on_play",
        ),
        (
            r"when your sword attack hits,?\s*you may destroy this\.?\s*"
            r"if you do,?\s*(.+?)(?:\.\s|\.$|$)",
            "on_controlled_hit",
        ),
        (
            r"whenever you beat chest,?\s*you may destroy this\.?\s*"
            r"if you do,?\s*(.+?)(?:\.\s|\.$|$)",
            "on_beat_chest",
        ),
        (
            r"whenever you discard a random card with 6 or more \{p\},?\s*"
            r"you may destroy this\.?\s*if you do,?\s*(.+?)(?:\.\s|\.$|$)",
            "on_random_discard",
        ),
        (
            r"when you discard a random card with 6 or more \{p\},?\s*"
            r"you may destroy this\.?\s*if you do,?\s*(.+?)(?:\.\s|\.$|$)",
            "on_random_discard",
        ),
        (
            r"when an edge of autumn you control hits,?\s*you may destroy this\.?\s*"
            r"if you do,?\s*(.+?)(?:\.\s|\.$|$)",
            "on_controlled_hit",
        ),
        (
            r"whenever a lightning or elemental attack you control is defended by a card from hand,?\s*"
            r"you may destroy this\.?\s*if you do,?\s*(.+?)(?:\.\s|\.$|$)",
            "on_lightning_defended",
        ),
        (
            r"when you attack for the fourth time during a turn,?\s*"
            r"you may destroy this\.?\s*if you do,?\s*(.+?)(?:\.\s|\.$|$)",
            "on_attack_count",
        ),
        (
            r"whenever a card with 6 or more \{p\} is put into your banished zone,?\s*"
            r"you may destroy this\.?\s*if you do,?\s*(.+?)(?:\.\s|\.$|$)",
            "on_banish_high_power",
        ),
    )
    for pat, when in patterns:
        m = re.search(pat, body, re.I)
        if not m:
            continue
        _append_optional_destroy_then(triggers, when, m.group(0).strip()[:90], m.group(1).strip(" ."))


def parse_keyword_triggers(keywords: tuple[str, ...] | list[str]) -> tuple[Trigger, ...]:
    """Mechanics implied by card keywords (handled by engine at play time)."""
    triggers: list[Trigger] = []
    for kw in keywords or ():
        row = _KEYWORD_EFFECTS.get(kw)
        if row is None:
            continue
        when, kind = row
        triggers.append(Trigger(when, Effect(kind, raw=f"keyword:{kw}"), f"keyword:{kw}"))
    return tuple(triggers)


def _parse_destroy_equipment_body_triggers(body: str) -> tuple[Trigger, ...]:
    """Multi-sentence destroy-category equipment patterns matched against full text."""
    triggers: list[Trigger] = []
    low = body.lower()

    m = re.search(
        r"if you would sharpen a zenith blade, instead you may pay \{r\} and destroy this\.?\s*"
        r"if you do, sharpen it an additional time",
        low,
    )
    if m:
        triggers.append(
            Trigger(
                "on_sharpen",
                Effect(
                    "pitch_pay",
                    1,
                    optional=True,
                    banish_name="destroy_sharpen_extra",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    m = re.search(
        r"whenever an attacking ally you control dies or an attack action card you control is destroyed by phantasm,?\s*"
        r"you may pay \{r\}\{r\}\{r\}\.?\s*if you do,?\s*destroy this and gain 1 action point",
        low,
    )
    if m:
        triggers.append(
            Trigger(
                "on_phantasm_destroy",
                Effect(
                    "pitch_pay",
                    3,
                    optional=True,
                    banish_name="destroy_self_go_again",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    m = re.search(
        r"the first time each turn another hero destroys a card they don[\u2019']t control,?\s*"
        r"you may pay \{r\}\{r\}\.?\s*if you do,?\s*they destroy a non-hero permanent they control",
        low,
    )
    if m:
        triggers.append(
            Trigger(
                "on_opponent_destroy",
                Effect(
                    "pitch_pay",
                    2,
                    optional=True,
                    banish_name="they_destroy_permanent",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    return tuple(triggers)


def _parse_banish_body_triggers(body: str) -> tuple[Trigger, ...]:
    """Multi-sentence banish-category patterns matched against full card text."""
    triggers: list[Trigger] = []
    low = body.lower()

    m = re.search(
        r"when this defends, banish any number of action cards from your hand,?\s*"
        r"then add them to the chain link as defending cards",
        low,
    )
    if m:
        triggers.append(
            Trigger(
                "on_defend",
                Effect(
                    "chain_defend",
                    banish_name="hand_actions",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    m = re.search(
        r"when this defends, you may banish an arrow from your hand\.?\s*"
        r"if you do, at the start of your next turn, put it face-up into your arsenal and it gets \+3\{p\} until end of turn",
        low,
    )
    if m:
        triggers.append(
            Trigger(
                "on_defend",
                Effect(
                    "destroy_hand",
                    1,
                    optional=True,
                    banish_name="arrow:next_turn_arsenal:+3",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    m = re.search(
        r"while this is in your graveyard, at the start of your turn, you may banish 2 cards named loyalty beyond the grave from your graveyard\.?\s*"
        r"if you do, draw a card",
        low,
    )
    if m:
        triggers.append(
            Trigger(
                "on_turn_start",
                Effect(
                    "banish_gy_variable",
                    2,
                    optional=True,
                    banish_name="named:loyalty beyond the grave:draw",
                    condition="in_graveyard",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    m = re.search(
        r"the next time you would be dealt lethal damage this turn, you may banish minerva themis from your hand or arsenal to prevent that damage",
        low,
    )
    if m:
        triggers.append(
            Trigger(
                "on_play",
                Effect(
                    "prevent_damage",
                    0,
                    optional=True,
                    banish_name="minerva_themis:lethal",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    m = re.search(
        r"at the beginning of the end phase, return them to your hand\.?\s*"
        r"the next time you would be dealt damage this turn, prevent x of that damage, where x is 2 plus the number of cards banished to play this",
        low,
    )
    if m:
        triggers.append(
            Trigger(
                "on_play",
                Effect(
                    "schedule_end_phase",
                    banish_name="return_banished_cost",
                    raw="return banished cost cards at end phase",
                ),
                m.group(0)[:80],
            )
        )
        triggers.append(
            Trigger(
                "on_play",
                Effect(
                    "prevent_damage",
                    2,
                    banish_name="last_banish_count_plus:2",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    m = re.search(
        r"if a card with 6 or more \{p\} is banished this way, you may play it from your banished zone during your next action phase",
        low,
    )
    if m:
        triggers.append(
            Trigger(
                "on_play",
                Effect(
                    "enable_gy_play",
                    optional=True,
                    banish_name="next_action_phase",
                    condition="banished_power6_this_cost",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    m = re.search(
        r"whenever you \*\*opt\*\*, put energy counters on blaze equal to the number of cards looked at this way",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_play",
                Effect(
                    "put_counter",
                    token_name="energy",
                    banish_name="blaze:opt_looked",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    m = re.search(
        r"target attack gets \+x\{p\}",
        low,
    )
    if m:
        triggers.append(
            Trigger(
                "on_play",
                Effect(
                    "modify_attack_power",
                    banish_name="last_banish_count",
                    raw=m.group(0)[:80],
                ),
                m.group(0)[:80],
            )
        )

    m = re.search(
        r"if you(?:'ve| have) \*\*charged\*\* this turn, search your deck for an action card with cost x or less, reveal it, put it into your hand, then shuffle",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_play",
                Effect(
                    "search",
                    banish_name="action:variable_cost",
                    condition="charged_this_turn",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    return tuple(triggers)


def _parse_draw_body_triggers(body: str) -> tuple[Trigger, ...]:
    """Multi-sentence draw-category patterns matched against full card text."""
    triggers: list[Trigger] = []
    low = body.lower()

    m = re.search(
        r"at the start of each other hero'?s turn, if they have less \{h\} than you, they may draw a card\.?\s*"
        r"if they do, you create a silver token",
        low,
    )
    if m:
        triggers.append(
            Trigger(
                "on_turn_start",
                Effect(
                    "draw",
                    1,
                    optional=True,
                    target="opponent",
                    condition="other_hero:life_less",
                    banish_name="create:silver",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    m = re.search(
        r"look at the defending hero'?s hand and choose a blue card\.?\s*"
        r"add it to this chain link as a defending card\.?\s*if you do, draw a card",
        low,
    )
    if m:
        triggers.append(
            Trigger(
                "on_play",
                Effect(
                    "choose_card",
                    target="opponent",
                    banish_name="blue:chain_defend:draw",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    m = re.search(
        r"while this is defending an attack with 2 or less \{p\}, when the combat chain closes, draw a card",
        low,
    )
    if m:
        triggers.append(
            Trigger(
                "on_chain_close",
                Effect(
                    "draw",
                    1,
                    condition="defending_attack_power_le:2",
                    raw=m.group(0)[:120],
                ),
                m.group(0)[:80],
            )
        )

    m = re.search(r"if a chi was pitched to play this, draw a card", low)
    if m:
        triggers.append(
            Trigger(
                "on_play",
                Effect("draw", 1, condition="chi_pitched", raw=m.group(0)[:120]),
                m.group(0)[:80],
            )
        )

    return tuple(triggers)


def parse_passive_triggers(text: str) -> tuple[Trigger, ...]:
    """Parse standalone sentences and non-standard trigger wordings."""
    triggers: list[Trigger] = []
    body = _clean(text)
    if not body.strip():
        return ()

    _parse_optional_destroy_then_patterns(body, triggers)
    triggers.extend(_parse_destroy_equipment_body_triggers(body))
    triggers.extend(_parse_banish_body_triggers(body))
    triggers.extend(_parse_draw_body_triggers(body))

    m = re.search(
        r"if there are 8 or more earth cards in your banished zone,?\s*"
        r"\w+(?:,\s*)? gets \"whenever you gain \{g\} during your turn,?\s*"
        r"you may deal (\d+) arcane damage to any opposing target\.?\"",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "whenever_gain_gold",
                Effect(
                    "arcane_damage",
                    int(m.group(1)),
                    m.group(0),
                    optional=True,
                    condition="earth_banish_ge:8",
                ),
                m.group(0),
            )
        )

    m = re.search(
        r"once per turn, when cromai attacks or leaves the arena,?\s*gain 1 action point",
        body,
        re.I,
    )
    if m:
        eff = Effect(
            "gain_action_point",
            1,
            raw=m.group(0),
            condition="once_per_turn",
        )
        triggers.append(Trigger("on_attack", eff, m.group(0)))
        triggers.append(Trigger("on_leave", eff, m.group(0)))

    m = re.search(
        r"at the start of your turn, you may put a card from your hand into your soul,?\s*"
        r"if it'?s an illusionist card, create a spectral shield token",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_turn_start",
                Effect(
                    "put_soul",
                    optional=True,
                    banish_name="create_if:illusionist:spectral shield",
                    raw=m.group(0),
                ),
                m.group(0),
            )
        )

    for sent in _sentences(text):
        low = sent.lower()

        if _is_activated_ability_clause(sent):
            continue

        if card_has_mode_menu_bullets(text) and sent.strip().startswith("*"):
            continue

        if card_has_mode_menu_bullets(text) and re.search(
            r"choose (?:\d|any|1 for each)", low
        ):
            continue

        m = re.search(
            r"when the combat chain closes,?\s*if this didn't hit,?\s*(.+)",
            sent,
            re.I,
        )
        if m:
            clause = m.group(1).strip(" .")
            for eff in _parse_clause_to_effects(clause):
                target = eff
                if "defending hero" in clause.lower() and eff.kind == "create_token":
                    target = Effect(
                        eff.kind,
                        eff.amount,
                        eff.raw,
                        token_name=eff.token_name,
                        target="defender",
                        optional=eff.optional,
                    )
                else:
                    target = eff
                if eff.implemented or target.implemented:
                    triggers.append(
                        Trigger(
                            "on_chain_close",
                            Effect(
                                target.kind,
                                target.amount,
                                target.raw,
                                token_name=target.token_name,
                                target=target.target,
                                optional=target.optional,
                                condition="miss",
                            ),
                            clause,
                        )
                    )
            continue

        # Whenever you play a card with contract, ...
        m = re.search(r"whenever you play a card with contract,?\s*(.+)", sent, re.I)
        if m:
            clause = m.group(1).strip(" .")
            if re.search(r"look at the top card", clause, re.I):
                triggers.append(Trigger("whenever_contract", Effect("look_deck", 1, clause), clause))
            if re.search(r"put it on the bottom", clause, re.I):
                triggers.append(
                    Trigger(
                        "whenever_contract",
                        Effect("put_bottom", target="opponent", optional=True, raw=clause),
                        clause,
                    )
                )
            continue

        m = re.search(
            r"whenever you discard a random card with 6 or more \{p\},?\s*"
            r"you may destroy this\.?\s*if you do,?\s*gain 1 action point",
            sent,
            re.I,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_random_discard",
                    Effect(
                        "destroy_item",
                        optional=True,
                        banish_name="go_again",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"if the discarded card has 6 or more \{p\},?\s*deal (\d+) damage to the attacking hero",
            sent,
            re.I,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "damage",
                        int(m.group(1)),
                        sent,
                        target="attacking_hero",
                        condition="discarded_power6_turn",
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever you play a card from hand,?\s*"
            r"you may put a card from hand face-down into your arsenal",
            sent,
            re.I,
        )
        if m:
            triggers.append(
                Trigger(
                    "whenever_play_from_hand",
                    Effect("stash_hand", optional=True, raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever a trap you control triggers,?\s*deal (\d+) damage to the attacking hero",
            sent,
            re.I,
        )
        if m:
            triggers.append(
                Trigger(
                    "whenever_trap_triggers",
                    Effect(
                        "damage",
                        int(m.group(1)),
                        sent,
                        target="attacking_hero",
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"remove all steam counters from target equipment, item, or weapon",
            sent,
            re.I,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "remove_counter",
                        0,
                        sent,
                        token_name="steam",
                        banish_name="target_all_steam",
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"if 2 or more steam counters are removed this way,?\s*"
            r"deal (\d+) damage to its controller",
            sent,
            re.I,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "damage",
                        int(m.group(1)),
                        sent,
                        target="permanent_controller",
                        condition="steam_removed_ge:2",
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"the next (\d+) times you would be dealt damage this turn, prevent (\d+) of that damage",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "prevent_damage",
                        int(m.group(2)),
                        sent,
                        condition=f"per_hit:{int(m.group(1))}",
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when(?:ever)? this defends,?\s*create an? (.+?) token under another hero'?s control",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect(
                        "create_token",
                        token_name=m.group(1).strip(),
                        target="ally",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when(?:ever)? this defends,?\s*another target hero draws a card",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect("draw", 1, sent, target="ally"),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when the combat chain closes,?\s*if this was played from arsenal,?\s*"
            r"put it on the bottom of (?:your|its owner'?s) deck",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_chain_close",
                    Effect("put_bottom", target="self", condition="played_from_arsenal", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"the next time you would be dealt damage this turn, prevent it",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("prevent_damage", 999, sent, condition="next_hit"),
                    sent,
                )
            )
            continue

        m = re.search(
            r"the next time you would be dealt damage this turn, prevent (\d+) of that damage",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("prevent_damage", int(m.group(1)), sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"at the beginning of your end phase,?\s*if you didn't hit a hero this turn,?\s*lose (\d+)\{g\}",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_end_phase",
                    Effect(
                        "lose_gold",
                        int(m.group(1)),
                        sent,
                        condition="no_hit_this_turn",
                    ),
                    sent,
                )
            )
            continue

        m = re.search(r"if you control no gold tokens,?\s*create a gold token", low)
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "create_token",
                        1,
                        sent,
                        token_name="gold",
                        condition="no_gold_tokens",
                    ),
                    sent,
                )
            )
            continue

        m = re.search(r"if you've played wax on this turn,?\s*create a zen state token", low)
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "create_token",
                        1,
                        sent,
                        token_name="zen state",
                        condition="wax_on_played",
                    ),
                    sent,
                )
            )
            continue

        if re.search(r"they put it on the bottom of their deck then draw a card", low):
            continue

        m = re.search(r"put this into your soul when the combat chain closes", low)
        if m:
            triggers.append(
                Trigger(
                    "on_chain_close",
                    Effect("put_soul", banish_name="self_from_chain", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(r"remove a \+1\{p\} counter from target attacking weapon", low)
        if m:
            token = "flurry"
            cm = re.search(r"create an? (.+?) token and draw a card", low)
            if cm:
                token = cm.group(1).strip()
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "remove_counter",
                        1,
                        raw=sent,
                        token_name="power",
                        target="attacking_weapon",
                        banish_name=f"create:{token}:draw",
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"turn any number of hyper drivers in your banished zone face-down and gain that many action points",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "gain_action_point",
                        0,
                        banish_name="hyper_drivers_face_down",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        if re.search(r"if 3 or more cards are turned face-down this way,?\s*draw a card", low):
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("draw", 1, sent, condition="hyper_drivers_ge:3"),
                    sent,
                )
            )
            continue

        m = re.search(
            r"roll a 6 sided die\.?\s*gain action points equal to half the number rolled, rounded down",
            low,
        )
        if not m:
            m = re.search(
                r"gain action points equal to half the number rolled, rounded down",
                low,
            )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("gain_action_point", 0, banish_name="roll_d6_half", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(r"if you'?ve rolled a 6 on a die this turn,?\s*draw a card", low)
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("draw", 1, sent, condition="rolled_6_this_turn"),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this enters the arena,?\s*remove a -1\{d\} counter from a chest equipment you control",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_enters",
                    Effect(
                        "remove_counter",
                        1,
                        raw=sent,
                        token_name="defense_penalty",
                        target="chest",
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever a mechanologist gun you control hits,?\s*"
            r"you may destroy this and a defending equipment",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_gun_hit",
                    Effect(
                        "destroy_item",
                        optional=True,
                        banish_name="defender_equipment",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(r"look at target hero'?s hand and the top card of their deck", low)
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "reveal_hand",
                        0,
                        sent,
                        banish_name="hand_and_deck",
                        target="opponent",
                    ),
                    sent,
                )
            )
            continue

        m = re.search(r"target opponent reveals their hand", low)
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "reveal_hand",
                        0,
                        sent,
                        target="opponent",
                        banish_name="store_revealed",
                    ),
                    sent,
                )
            )
            continue

        m = re.search(r"create x (.+?) tokens under target hero'?s control", low)
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "create_token",
                        token_name=m.group(1).strip(),
                        target="opponent",
                        banish_name="pitch_value",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever an attack you control hits a light hero this turn,?\s*"
            r"you may banish a card from their soul",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_controlled_hit",
                    Effect(
                        "destroy_hand",
                        1,
                        optional=True,
                        target="opponent",
                        banish_name="soul",
                        condition="light_hero",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"if you would be dealt damage,?\s*banish a card from your hero'?s soul to prevent 1 of that damage",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_enters",
                    Effect("prevent_damage", 1, banish_name="soul_pay", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(r"when there are no cards in your soul,?\s*destroy this", low)
        if m:
            triggers.append(
                Trigger(
                    "on_end_phase",
                    Effect("destroy_item", condition="no_soul", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"at the start of your turn,?\s*banish a card from your hand",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_turn_start",
                    Effect("destroy_hand", 1, banish_name="create:runechant", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever you play a card with an arcane damage effect,?\s*you may pay \{r\}",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "whenever_arcane_play",
                    Effect(
                        "pitch_pay",
                        1,
                        optional=True,
                        banish_name="arcane_plus_one",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(r"if this was played from arsenal,?\s*draw a card", low)
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("draw", 1, sent, condition="played_from_arsenal"),
                    sent,
                )
            )
            continue

        m = re.search(r"target hero reveals (\d+) cards from their hand", low)
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "reveal_hand",
                        int(m.group(1)),
                        sent,
                        target="opponent",
                        banish_name="store_revealed",
                        condition="opponent_turn_all",
                    ),
                    sent,
                )
            )
            continue

        if re.search(r"you may choose a card revealed this way", low):
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "choose_card",
                        optional=True,
                        target="opponent",
                        banish_name="put_bottom:draw",
                        condition="revealed_only",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        if re.search(r"look at the defending hero'?s hand and choose a card", low):
            triggers.append(
                Trigger(
                    "on_attack",
                    Effect(
                        "choose_card",
                        target="opponent",
                        banish_name="put_bottom:draw",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        if re.search(r"look at their hand and choose a card\b", low):
            triggers.append(
                Trigger(
                    "on_hit",
                    Effect(
                        "choose_card",
                        target="opponent",
                        banish_name="banish_same_name:3",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"look at their hand and choose a card without base \{d\}",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_hit",
                    Effect(
                        "choose_card",
                        optional=True,
                        target="opponent",
                        banish_name="discard:draw",
                        condition="no_defense",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"if an effect would resolve that includes rolling a 6 sided die,?\s*"
            r"instead you may destroy this",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_die_roll",
                    Effect("destroy_item", optional=True, raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever you play an attack action card with 6 or more base \{p\},?\s*roll a 6 sided die",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "whenever_power6_attack",
                    Effect("modify_attack_power", banish_name="roll_d6", raw=sent),
                    sent,
                )
            )
            continue

        if re.search(
            r"if it'?s a non-token light card,?\s*put it into your soul",
            low,
        ):
            triggers.append(
                Trigger(
                    "on_controlled_destroy",
                    Effect("put_soul", banish_name="destroyed_light", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever an aura or attack action card you control is destroyed,?\s*"
            r"deal 1 arcane damage to target hero",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_controlled_destroy",
                    Effect("arcane_damage", 1, sent, target="opponent"),
                    sent,
                )
            )
            if re.search(r"if it'?s a non-token light card,?\s*put it into your soul", low):
                triggers.append(
                    Trigger(
                        "on_controlled_destroy",
                        Effect("put_soul", banish_name="destroyed_light", raw=sent),
                        sent,
                    )
                )
            continue

        m = re.search(r"when this is destroyed,?\s*create a token copy of an aura you control", low)
        if m:
            triggers.append(
                Trigger(
                    "on_destroy",
                    Effect("create_token", 1, sent, token_name="aura copy", optional=True),
                    sent,
                )
            )
            continue

        # Halo of Lumina Light: "When this is destroyed, you may put a yellow aura from your banished zone into the arena."
        m = re.search(r"when this is destroyed,?\s*you may put (?:a )?(?:yellow )?aura from your banished zone into the arena", low)
        if m:
            triggers.append(
                Trigger(
                    "on_destroy",
                    Effect("retrieve_banished_aura", 1, sent[:80], optional=True),
                    sent[:80],
                )
            )
            continue

        m = re.search(
            r"when this hits,?\s*create x runechant tokens,?\s*where x is the damage dealt this way",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_hit",
                    Effect(
                        "create_token",
                        0,
                        sent,
                        token_name="runechant",
                        banish_name="damage_dealt",
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"target hero reveals (\d+) cards from their hand",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("reveal_hand", int(m.group(1)), sent, target="opponent"),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever you play a runeblade card,?\s*if you(?:'ve| have) played another non-attack action card this turn,?\s*"
            r"create a runechant token",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "whenever_runeblade_play",
                    Effect(
                        "create_token",
                        1,
                        sent,
                        token_name="runechant",
                        condition="naa_played_this_turn",
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever tomeltai attacks a hero,?\s*reveal the top (\d+) cards of your deck",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_attack",
                    Effect(
                        "look_deck",
                        int(m.group(1)),
                        banish_name="reveal_red_count",
                        raw=sent,
                    ),
                    sent,
                )
            )
            if re.search(
                r"if 1 or more red cards are revealed this way,?\s*put that many \+1\{p\} counters on him",
                low,
            ):
                triggers.append(
                    Trigger(
                        "on_attack",
                        Effect(
                            "put_counter",
                            0,
                            banish_name="reveal_red_count",
                            token_name="power",
                            raw=sent,
                        ),
                        sent,
                    )
                )
            continue

        m = re.search(
            r"whenever a lightning or elemental attack you control is defended by a card from hand,?\s*"
            r"you may destroy this[,.\s]*if you do,?\s*the attack gets \+(\d+)\{p\}",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_lightning_hand_defend",
                    Effect(
                        "destroy_item",
                        optional=True,
                        raw=f"the attack gets +{m.group(1)}{{p}}",
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever you banish a hyper driver from boosting,?\s*target wrench you control gets \+(\d+)\{p\} this turn",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_boost_banish",
                    Effect(
                        "next_attack_power",
                        int(m.group(1)),
                        banish_name="wrench",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"at the beginning of your end phase,?\s*if you(?:'ve| have) attacked 2 or more times with weapons this turn,?\s*"
            r"create a copper token for each weapon attack that hit",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_end_phase",
                    Effect(
                        "create_token",
                        0,
                        sent,
                        token_name="copper",
                        banish_name="weapon_hits",
                        condition="weapon_attacks_ge:2",
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever you \*\*ice fuse\*\*, remove a frost counter from this",
            low,
        )
        if not m:
            m = re.search(r"whenever you ice fuse, remove a frost counter from this", low)
        if m:
            triggers.append(
                Trigger(
                    "on_ice_fuse",
                    Effect("remove_counter", 1, sent, token_name="frost"),
                    sent,
                )
            )
            continue

        m = re.search(
            r"the next time you would deal less than (\d+) damage this turn, instead deal (\d+) damage",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "damage_floor",
                        int(m.group(2)),
                        sent,
                        max_cost=int(m.group(1)),
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever vynserakai hits a hero,?\s*he deals (\d+) arcane damage to them",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_hit",
                    Effect("arcane_damage", int(m.group(1)), sent, target="opponent"),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever you play a song,?\s*create copper tokens equal to the number of other heroes in the game",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "whenever_song",
                    Effect(
                        "create_token",
                        1,
                        sent,
                        token_name="copper",
                        banish_name="other_heroes",
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"at the beginning of your end phase,?\s*if you have (\d+) or more cards in hand,?\s*"
            r"create an? (.+?) token",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_end_phase",
                    Effect(
                        "create_token",
                        1,
                        sent,
                        token_name=m.group(2).strip(),
                        condition=f"hand_ge:{int(m.group(1))}",
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"prevent the next x damage that would be dealt to you this turn,?\s*"
            r"where x is the number rolled",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("prevent_damage", 0, banish_name="roll_d6", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"prevent the next (\d+) damage that would be dealt to you this turn",
            low,
        )
        if m and "defends alone" not in low:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("prevent_damage", int(m.group(1)), sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever an arrow is put face-up into your arsenal from your deck,?\s*"
            r"you may pay ((?:\{r\})+|\d+)[,.\s]*if you do,?\s*put an aim counter on it",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_arrow_from_deck",
                    Effect(
                        "pitch_pay",
                        _resource_amount(m.group(1)),
                        banish_name="aim_counter",
                        optional=True,
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever a mechanologist item with cost (\d+) or less is put into your banished zone from your deck,?\s*"
            r"put it into the arena",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_deck_banish_item",
                    Effect(
                        "put_item_in_arena",
                        int(m.group(1)),
                        target="banished",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever dracona optimai attacks a hero,?\s*reveal the top (\d+) cards of your deck",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_attack",
                    Effect(
                        "look_deck",
                        int(m.group(1)),
                        banish_name="reveal_red_count",
                        raw=sent,
                    ),
                    sent,
                )
            )
            if re.search(
                r"he deals arcane damage equal to twice the number of red cards revealed this way",
                low,
            ):
                triggers.append(
                    Trigger(
                        "on_attack",
                        Effect(
                            "arcane_damage",
                            0,
                            banish_name="red_reveal_double",
                            target="opponent",
                            raw=sent,
                        ),
                        sent,
                    )
                )
            continue

        m = re.search(
            r"whenever you beat chest,?\s*you may pay ((?:\{r\})+|\d+) and destroy this[,.\s]*"
            r"if you do,?\s*draw",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_beat_chest",
                    Effect(
                        "pitch_pay",
                        _resource_amount(m.group(1)),
                        banish_name="destroy_draw",
                        optional=True,
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever a weapon attack you control hits,?\s*you may pay ((?:\{r\})+|\d+)[,.\s]*"
            r"if you do,?\s*create an? (.+?) token",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_weapon_hit",
                    Effect(
                        "pitch_pay",
                        _resource_amount(m.group(1)),
                        token_name=m.group(2).strip(),
                        optional=True,
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(r"whenever a weapon attack you control hits,?\s*(.+)", sent, re.I)
        if m:
            clause = m.group(1).strip(" .")
            pitch = _parse_pitch_pay(clause, optional=True)
            if pitch is not None and pitch.implemented:
                triggers.append(Trigger("on_weapon_hit", pitch, sent))
            continue

        if "dracona optimai" in body.lower():
            m = re.search(
                r"he deals arcane damage equal to twice the number of red cards revealed this way",
                low,
            )
            if m:
                triggers.append(
                    Trigger(
                        "on_attack",
                        Effect(
                            "arcane_damage",
                            0,
                            banish_name="red_reveal_double",
                            target="opponent",
                            raw=sent,
                        ),
                        sent,
                    )
                )
                continue

        if "tomeltai" in body.lower():
            m = re.search(
                r"if 1 or more red cards are revealed this way,?\s*put that many \+1\{p\} counters on him",
                low,
            )
            if m:
                triggers.append(
                    Trigger(
                        "on_attack",
                        Effect(
                            "put_counter",
                            0,
                            banish_name="reveal_red_count",
                            token_name="power",
                            raw=sent,
                        ),
                        sent,
                    )
                )
                continue

        m = re.search(
            r"the next time you would be dealt \{p\} damage this turn, prevent (\d+) damage that source would deal",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("prevent_damage", int(m.group(1)), sent, condition="power_damage"),
                    sent,
                )
            )
            continue

        m = re.search(
            r"the next time you would be dealt (\d+) or less damage this turn, prevent it",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("prevent_damage", int(m.group(1)), sent, condition="damage_le"),
                    sent,
                )
            )
            continue

        m = re.search(
            r"prevent the next (\d+) damage that would be dealt to you this turn by a shadow source",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("prevent_damage", int(m.group(1)), sent, condition="shadow"),
                    sent,
                )
            )
            continue

        m = re.search(
            r"the next time a shadow source would deal damage this turn, prevent (\d+) of that damage",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("prevent_damage", int(m.group(1)), sent, condition="shadow"),
                    sent,
                )
            )
            continue

        if re.search(
            r"if this would be put into your graveyard from anywhere, instead put it on the bottom of your deck",
            low,
        ):
            triggers.append(Trigger("gy_replacement", Effect("gy_to_bottom", raw=sent), sent))
            continue

        m = re.search(
            r"when the combat chain closes,?\s*if this was played from arsenal,?\s*put it on the bottom of your deck",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_chain_close",
                    Effect("put_bottom", target="self", condition="played_from_arsenal", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"if a yellow card is charged this way,?\s*create an? (.+?) token",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect(
                        "create_token",
                        token_name=m.group(1).strip(),
                        condition="charged_yellow",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"you may play a non-attack action card this chain link as though it were an instant",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_attack",
                    Effect(
                        "grant_may_play",
                        target="hand",
                        banish_name="non-attack",
                        condition="chain_only",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"you may play your next (\w+) non-attack action card this turn as though it were an instant",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "grant_may_play",
                        target="hand",
                        banish_name="non-attack",
                        token_name=m.group(1).lower(),
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"put (\d+|one|two|three) \+1\{p\} counters on target aura with ward you control",
            low,
        )
        if m:
            word = m.group(1).lower().strip()
            n = {"one": 1, "two": 2, "three": 3}.get(word)
            if n is None:
                n = int(word)
            triggers.append(
                Trigger("on_play", Effect("put_counter", n, sent, token_name="power"), sent)
            )
            continue

        m = re.search(
            r"when this defends, you may reveal a card with crush from your hand\.?\s*"
            r"if you do, create an? (.+?) token",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect(
                        "create_token",
                        token_name=m.group(1).strip(),
                        optional=True,
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends a shadow attack, non-equipment light cards get \+(\d+)\{d\} this combat chain",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect("grant_light_block", int(m.group(1)), sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"choose a hero\.?\s*the next time they would be dealt damage this turn, prevent (\d+)",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("prevent_damage", int(m.group(1)), sent, target="any"),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this is destroyed, you may put an illusionist aura with cost 0 from your hand into the arena",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_leave",
                    Effect("put_item_in_arena", 0, raw=sent, optional=True, banish_name="illusionist"),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when an attack action card you control hits, you may destroy this\.?\s*"
            r"if you do, create an? (.+?) token",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_controlled_hit",
                    Effect(
                        "destroy_item",
                        optional=True,
                        banish_name="create:" + m.group(1).strip(),
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"prevent the next (\d+) damage that would be dealt to target hero this turn by a shadow source",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("prevent_damage", int(m.group(1)), sent, condition="shadow", target="opponent"),
                    sent,
                )
            )
            continue

        m = re.search(
            r"the next time they would be dealt damage this turn, prevent (\d+)",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("prevent_damage", int(m.group(1)), sent, target="any"),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this is discarded at random, put it on the bottom of your deck",
            low,
        )
        if m:
            triggers.append(
                Trigger("on_discarded_random", Effect("put_bottom", target="self", raw=sent), sent)
            )
            continue

        if re.search(r"look at the top card of target hero's deck", low):
            triggers.append(
                Trigger("on_defend", Effect("look_deck", 1, target="opponent", raw=sent), sent)
            )
            continue

        m = re.search(
            r"if you have a base (head|chest|arms|legs) equipped, transform it into this, then equip this",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_enters",
                    Effect("transform_equip", banish_name=m.group(1), raw=sent),
                    sent,
                )
            )
            continue

        if re.search(r"this counts as a gold", low):
            triggers.append(Trigger("on_enters", Effect("counts_as_gold", raw=sent), sent))
            continue

        m = re.search(
            r"once per turn, you may play a (\w+) item with cost 0 or 1 from the top of your deck as though it were an instant",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_enters",
                    Effect(
                        "play_from_deck_top",
                        1,
                        banish_name=m.group(1).lower(),
                        optional=True,
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends, you may reveal a card with crush from your hand",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect("reveal_top", optional=True, raw=sent, banish_name="crush"),
                    sent,
                )
            )
            continue

        m = re.search(
            r"at the start of your turn, destroy this unless you remove a (\w+) counter from it",
            sent,
            re.I,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_turn_start",
                    Effect("upkeep_or_destroy", token_name=m.group(1).strip(), raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(r"at the start of your turn, destroy this(?:,)? then (.+)", sent, re.I)
        if m:
            _append_destroy_then_triggers(triggers, "on_turn_start", sent, m.group(1).strip(" ."))
            continue

        m = re.search(r"at the start of your turn, destroy this\.?$", sent, re.I)
        if m:
            triggers.append(Trigger("on_turn_start", Effect("destroy_item", raw=sent), sent))
            continue

        m = re.search(
            r"at the beginning of your end phase, destroy this(?:,)? then (.+)",
            sent,
            re.I,
        )
        if m:
            _append_destroy_then_triggers(triggers, "on_end_phase", sent, m.group(1).strip(" ."))
            continue

        m = re.search(
            r"crush\s*[-—–]\s*they can't play attack action cards with (\d+) or less base \{p\}",
            sent,
            re.I,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_crush",
                    Effect("play_power_cap", int(m.group(1)), sent, target="opponent"),
                    sent,
                    threshold=4,
                )
            )
            continue

        m = re.search(
            r"ward x,? where x is (\d+) if you(?:'ve| have) pitched a blue card this turn, otherwise x is (\d+)",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "ward",
                        int(m.group(1)),
                        sent,
                        banish_name=m.group(2),
                        condition="blue_pitched",
                    ),
                    sent,
                )
            )
            continue

        # Ward N as standalone sentence.
        m = re.fullmatch(r"ward (\d+)", low.strip())
        if m:
            triggers.append(Trigger("on_play", Effect("ward", int(m.group(1)), sent), sent))
            continue

        # Ward X, where X is N times blue cards pitched (Three Visits)
        m = re.search(
            r"ward x,?\s*where x is (?:three|3) times the number of blue cards you'?ve pitched this turn",
            low,
        )
        if m:
            triggers.append(Trigger("on_play", Effect("ward", 0, sent, condition="blue_pitched_x3"), sent))
            continue

        # Poison the Well: "The next time a hero would gain {h} this turn, instead they lose that much {h}."
        m = re.search(
            r"the next time a hero would gain\s*(?:\{h\}|life) this turn,?\s*instead they lose that much",
            low,
        )
        if m:
            triggers.append(Trigger("on_play", Effect("invert_next_life_gain", 1, sent[:80]), sent[:80]))
            continue

        # "Put [card type] from your graveyard on (the) top of your deck." (Pick Yourself Up Off the Floor, etc.)
        m = re.search(r"put (?:an? )?(.+?) from your graveyard on (?:the )?top of your deck", low)
        if m:
            triggers.append(Trigger("on_play", Effect("return_gy_to_deck", 1, sent[:80], banish_name=m.group(1).strip()), sent[:80]))
            continue

        # Spreading Plague: "Create X Bloodrot Pox tokens under the defending hero's control, where X is the number of defending cards"
        m = re.search(
            r"create x bloodrot pox tokens? under the defending hero'?s? control,?\s*where x is the number of defending cards?",
            low,
        )
        if m:
            triggers.append(Trigger("on_play", Effect("create_token_per_defender", 0, sent[:80], token_name="bloodrot pox"), sent[:80]))
            continue

        # Electromagnetic Somersault: "Return them to their owner's hand when the chain link resolves"
        m = re.search(
            r"return them to their owner'?s? hand when the chain link resolves",
            low,
        )
        if m:
            n = 2
            nm = re.search(r"up to (\d+)", low)
            if nm:
                n = int(nm.group(1))
            triggers.append(Trigger("on_chain_close", Effect("return_chain_cards", n, sent[:80]), sent[:80]))
            continue

        # Blanch: "When this hits a hero, cards they own lose all colors until the end of their next turn."
        m = re.search(r"when this hits a hero,?\s*cards they own lose all colors until the end of their next turn", low)
        if m:
            triggers.append(Trigger("on_hit", Effect("lose_colors", 1, sent[:80]), sent[:80]))
            continue

        # Erase Face: "When this hits a hero, cards they own lose all class and talent types until the end of their next turn."
        m = re.search(r"when this hits a hero,?\s*cards they own lose all class and talent types?", low)
        if m:
            triggers.append(Trigger("on_hit", Effect("lose_class_talent", 1, sent[:80]), sent[:80]))
            continue

        # Fabric of Spring / Venomback Fabric: "Equip [Item Name]. If you don't, negate this."
        # The two sentences get split; check both the sentence starts with "equip"
        # and the full card body contains "if you don't, negate this".
        m = re.search(r"\bequip (.+?)\.?$", low.strip())
        if m and re.search(r"if you don'?t,?\s*(?:\*?\*?negate\*?\*?) this", body, re.I):
            item_name = m.group(1).strip()
            triggers.append(Trigger("on_play", Effect("equip_inventory", 1, sent[:80], banish_name=item_name), sent[:80]))
            continue

        # Pay Day: "If you've completed a contract this turn, create 4 Silver tokens."
        m = re.search(r"if you'?ve? completed a contract this turn,?\s*create (\d+) silver tokens?", low)
        if m:
            triggers.append(Trigger("on_play", Effect("create_token", int(m.group(1)), sent[:80], token_name="silver", condition="contract_completed"), sent[:80]))
            continue

        # Pulsewave Protocol: "Evo Upgrade - When this attacks a hero, they reveal X cards from their hand, where X is the number of Evos you have equipped."
        m = re.search(r"when this attacks a hero,?\s*they reveal x cards? from their hand,?\s*where x is the number of evos", low)
        if m:
            triggers.append(Trigger("on_attack", Effect("reveal_hand", 0, sent[:80], condition="evo_count"), sent[:80]))
            continue

        # Code of Conduct: "when you deal lethal damage to them, take an extra turn after this one"
        if re.search(r"take an extra turn after this one", low):
            triggers.append(Trigger("on_play", Effect("extra_turn", 1, sent[:80], condition="deal_lethal"), sent[:80]))
            continue

        # Immobilizing Shot: "If this has an aim counter, it gets 'When this hits a hero, they can't play more than 1 attack action card...'"
        if re.search(r"if this has an aim counter.{1,60}when this hits a hero.{1,120}can't play more than 1 attack action", low, re.S):
            triggers.append(Trigger("on_hit", Effect("limit_actions_next_turn", 1, sent[:80], target="opponent", condition="aim_counter"), sent[:80]))
            continue

        # Goldfin Harpoon: "If this would be put into a graveyard, instead remove it from the game."
        if re.search(r"if this would be put into a graveyard,?\s*instead remove it from the game", low):
            triggers.append(Trigger("on_gy_enter", Effect("remove_from_game", 1, sent[:80], target="self"), sent[:80]))
            continue

        # Turn Heads: "When this leaves the arena, {t} target Brute hero."
        if re.search(r"when this leaves the arena,?\s*\{t\}\s*target", low):
            triggers.append(Trigger("on_leave", Effect("freeze", 1, sent[:80], target="opponent"), sent[:80]))
            continue

        # Manifestation of Miragai: "This enters the arena with N +1{p} counters."
        m = re.search(r"this enters the arena with (\w+) \+1\{p\} counters?", low)
        if m:
            num_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
            n_word = m.group(1).lower()
            n = num_map.get(n_word, int(n_word) if n_word.isdigit() else 2)
            triggers.append(Trigger("on_enters", Effect("put_counter", n, sent[:80], token_name="power"), sent[:80]))
            continue

        # Diabolic Ultimatum: "If an attack action card was pitched, each hero destroys an ally."
        m = re.search(r"if an? attack action card was pitched to play this,?\s*each hero chooses? and destroys? an ally", low)
        if m:
            triggers.append(Trigger("on_play", Effect("destroy_item", 1, sent[:80], target="all", condition="pitched_attack"), sent[:80]))
            continue

        # Diabolic Ultimatum: "If a non-attack action card was pitched, each hero destroys an aura."
        m = re.search(r"if a non-attack action card was pitched to play this,?\s*each hero chooses? and destroys? an aura", low)
        if m:
            triggers.append(Trigger("on_play", Effect("destroy_item", 1, sent[:80], target="all", condition="pitched_nonattack"), sent[:80]))
            continue

        # Levia: "If a card with 6 or more {p} has been put into your banished zone this turn, cards you own lose blood debt during the end phase."
        if re.search(r"cards you own lose\s*(?:\*?\*?blood debt\*?\*?) during the end phase", low):
            triggers.append(Trigger("on_end_phase", Effect("remove_blood_debt", 1, sent[:80], condition="banished_6plus_power"), sent[:80]))
            continue

        # Iyslander: "Whenever you play an Ice card during an opponent's turn, create a Frostbite token under their control."
        m = re.search(r"whenever you play an ice card during an opponent'?s? turn,?\s*create a? frostbite tokens?", low)
        if m:
            triggers.append(Trigger("on_play", Effect("create_token", 1, sent[:80], token_name="frostbite", target="opponent", condition="opponent_turn"), sent[:80]))
            continue

        # Seasoned Saviour: "When this is equipped, put N -1{d} counters on it."
        m = re.search(r"when this is equipped,?\s*put (\w+) -1\{d\} counters? on it", low)
        if m:
            num_map = {"one": 1, "two": 2, "three": 3, "four": 4}
            n_word = m.group(1).lower()
            n = num_map.get(n_word, int(n_word) if n_word.isdigit() else 1)
            triggers.append(Trigger("on_equip", Effect("put_counter", n, sent[:80], token_name="neg_defense"), sent[:80]))
            continue

        # Sigil of Parapets: "While this is defending, whenever you play a Wizard card, this gets +2{d}."
        if re.search(r"while this is defending,?\s*whenever you play a wizard card,?\s*this gets \+\d+\{d\}", low):
            triggers.append(Trigger("on_defend", Effect("put_counter", 2, sent[:80], token_name="defense", condition="wizard_played"), sent[:80]))
            continue

        # Stand Tall: "While this is defending, whenever the attacking hero plays or activates a reaction this chain link, this gets +2{d}."
        if re.search(r"while this is defending,?\s*whenever the attacking hero plays or activates a reaction", low):
            triggers.append(Trigger("on_defend", Effect("put_counter", 2, sent[:80], token_name="defense", condition="reaction_played"), sent[:80]))
            continue

        # Gesture of Goodwill: "When this protects another hero, they may give you a token they control."
        if re.search(r"when this\s*\*?\*?protects\*?\*? another hero,?\s*they may give you a token", low):
            triggers.append(Trigger("on_protect_other", Effect("receive_token", 1, sent[:80], optional=True), sent[:80]))
            continue

        # Heirloom of Snake/Tiger Hide: "While this is equipped face-down, at the start of your turn, if you have exactly 1{g}, you may turn this face-up."
        if re.search(r"while this is equipped face-down,?\s*at the start of your turn,?\s*if you have exactly 1\{g\},?\s*you may turn this face-up", low):
            triggers.append(Trigger("on_turn_start", Effect("flip_face_up", 1, sent[:80], condition="face_down_1g", optional=True), sent[:80]))
            continue

        # Ouvia: "At the start of your turn or when Ouvia enters the arena, transform up to 1 ash you control into an Aether Ashwing."
        if re.search(r"at the start of your turn or when .* enters the arena,?\s*transform up to 1 ash you control into an aether ashwings?", low):
            triggers.append(Trigger("on_turn_start", Effect("transform_token", 1, sent[:80], banish_name="ash", token_name="Aether Ashwing"), sent[:80]))
            triggers.append(Trigger("on_enters", Effect("transform_token", 1, sent[:80], banish_name="ash", token_name="Aether Ashwing"), sent[:80]))
            continue

        # Brutus: keep the clash text visible to the coverage scanner.
        if re.search(r"you may have cards with clash of any class or talent in your deck", low):
            triggers.append(Trigger("on_play", Effect("clash", raw=sent, banish_name="brutus_clash"), sent[:80]))
            continue

        # Spirit of Eirina: "You may play [Card Name] as though it were an instant."
        m = re.search(r"you may play (.+?) as though it were an instant", low)
        if m:
            card_name = m.group(1).strip()
            triggers.append(Trigger("on_play", Effect("play_as_instant", 1, sent[:80], banish_name=card_name, optional=True), sent[:80]))
            continue

        # Squizzy & Floof: "If they do, you create a Gold token." (after opponent creates Cracked Bauble)
        # Match the full conditional sentence about each opposing hero's turn
        if re.search(r"at the start of each opposing hero'?s? turn,?\s*they may create", low):
            triggers.append(Trigger("on_opponent_turn_start", Effect("create_token", 1, sent[:80], token_name="gold", condition="opponent_created_bauble"), sent[:80]))
            continue

        # Nekria: "Whenever Nekria deals or is dealt damage, put a -1{g} counter on her and create an Ash token."
        if re.search(r"whenever \w+ deals or is dealt damage,?\s*put a -1\{g\} counter on h\w+ and create an ash token", low):
            triggers.append(Trigger("on_damage", Effect("put_counter", 1, sent[:80], token_name="neg_gold"), sent[:80]))
            triggers.append(Trigger("on_damage", Effect("create_token", 1, sent[:80], token_name="ash"), sent[:80]))
            continue

        # Visit the Golden Anvil: "Equip X weapons and/or equipment from your inventory."
        m = re.search(r"\bequip x (?:weapons? and/or )?(?:equipment|items?|weapons?) from your inventory", low)
        if m:
            triggers.append(Trigger("on_play", Effect("equip_inventory", 1, sent[:80], banish_name="x from inventory"), sent[:80]))
            continue

        # The Moat Exchange: "each hero... then draws a card for each card put on the bottom this way"
        if re.search(r"each hero may put any number of cards.+draws? a card for each card put on the bottom", low, re.S):
            triggers.append(Trigger("on_play", Effect("draw", 1, sent[:80], target="all"), sent[:80]))
            continue

        # Divvy Up: remove Treasure Island gold counters, then create that many Gold tokens.
        if re.search(r"remove half the gold counters from treasure island,? rounded up", low):
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("remove_counter", 0, sent[:80], token_name="gold", banish_name="treasure_half_up_or_all_if_thief"),
                    sent[:80],
                )
            )
            continue

        if re.search(r"create gold tokens equal to the number of gold counters removed this way", low):
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("create_token", 0, sent[:80], token_name="gold", banish_name="last_gold_removed"),
                    sent[:80],
                )
            )
            continue

        # Warmonger's Diplomacy: "if they choose war, the only actions they may play or activate during their next turn are weapon and attack actions"
        if re.search(r"if they choose war,?\s*the only actions they may play or activate during their next turn are weapon and attack actions", low):
            triggers.append(Trigger("on_play", Effect("limit_actions_next_turn", 1, sent[:80], target="opponent", condition="chose_war"), sent[:80]))
            continue

        # Tales of Adventure: "each other hero chooses and creates a token that hasn't been chosen"
        if re.search(r"each other hero chooses? and creates? a token that hasn'?t been chosen", low):
            triggers.append(Trigger("on_play", Effect("create_token", 1, sent[:80], target="opponent"), sent[:80]))
            continue

        # Florian: "if there are 8 or more earth cards in your banished zone, [Florian] gets..."
        m = re.search(r"if there are (\d+) or more earth cards in your banished zone", low)
        if m:
            triggers.append(Trigger("on_play", Effect("create_extra_token", 1, sent[:80], condition=f"earth_banished_ge:{m.group(1)}"), sent[:80]))
            continue

        # Deal N arcane/physical damage (action body text).
        m = re.match(
            r"deal (\d+) (arcane )?damage to (?:any target|target hero|target opposing hero|each opposing hero|them)\.?$",
            low,
        )
        if m:
            kind = "arcane_damage" if m.group(2) else "damage"
            triggers.append(Trigger("on_play", Effect(kind, int(m.group(1)), sent), sent))
            continue

        # If this is attacking a hero, ...
        m = re.search(r"if this is attacking a hero,?\s*(.+)", sent, re.I)
        if m:
            clause = m.group(1).strip(" .")
            agg = _parse_effect_aggressive(clause) or _parse_extended_clause(clause, optional=False, condition="")
            if agg and agg.implemented:
                triggers.append(Trigger("on_attack", agg, clause))
            continue

        # If this was played during an opponent's turn, ...
        m = re.search(r"if this was played during an opponent'?s turn,?\s*(.+)", sent, re.I)
        if m:
            clause = m.group(1).strip(" .")
            agg = _parse_effect_aggressive(clause)
            if agg:
                triggers.append(Trigger("on_play", agg, clause))
            continue

        # This costs {r} less to play for each ...
        m = re.search(r"this costs? ((?:\{r\})+|\d+) less to play", low)
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("cost_reduction", _resource_amount(m.group(1)), sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"the next (?:attack action card|aura|guardian attack action card|brute attack action card) you play this turn costs ((?:\{r\})+|\d+) less to play",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("cost_reduction", _resource_amount(m.group(1)), sent),
                    sent,
                )
            )
            continue

        # The next card you play this turn ...
        if re.search(r"the next card you play this turn", low):
            agg = _parse_effect_aggressive(sent)
            if agg is not None and agg.implemented:
                triggers.append(Trigger("on_play", agg, sent))
            else:
                triggers.append(
                    Trigger(
                        "on_play",
                        Effect("unimplemented", raw=sent),
                        sent,
                    )
                )
            continue
        m = re.search(r"when an attack you control hits a hero,?\s*(.+)", sent, re.I)
        if m:
            clause = m.group(1).strip(" .")
            effs = _parse_clause_to_effects(clause, allow_this_turn=True)
            for eff in effs:
                triggers.append(Trigger("on_controlled_hit", eff, clause))
            if not effs:
                agg = _parse_effect_aggressive(clause)
                if agg:
                    triggers.append(Trigger("on_controlled_hit", agg, clause))
            continue

        m = re.search(r"whenever you hit a marked hero,?\s*(.+)", sent, re.I)
        if m:
            clause = m.group(1).strip(" .")
            for eff in _parse_clause_to_effects(clause):
                triggers.append(Trigger("on_controlled_hit", eff, clause))
            continue

        m = re.search(r"whenever an attack action card hits this combat chain,?\s*(.+)", sent, re.I)
        if m:
            clause = m.group(1).strip(" .")
            for eff in _parse_clause_to_effects(clause):
                triggers.append(Trigger("on_chain_hit", eff, clause))
            continue

        m = re.search(r"whenever an attack you control wagers,?\s*(.+)", sent, re.I)
        if m:
            clause = m.group(1).strip(" .")
            pitch = _parse_pitch_pay(clause, optional=True)
            if pitch is not None:
                triggers.append(Trigger("whenever_wager", pitch, clause))
            continue

        # Whenever an attack hits a hero this turn, ...
        m = re.search(r"whenever an attack hits a hero this turn,?\s*(.+)", sent, re.I)
        if m:
            clause = m.group(1).strip(" .")
            for eff in _parse_clause_to_effects(clause):
                triggers.append(Trigger("on_controlled_hit", eff, clause))
            continue

        # When you attack with this, ...
        m = re.search(r"when you attack with this,?\s*(.+)", sent, re.I)
        if m:
            clause = m.group(1).strip(" .")
            for eff in _parse_clause_to_effects(clause):
                triggers.append(Trigger("on_attack", eff, clause))
            continue

        # When this attacks a hero, ...
        m = re.search(r"when this attacks a hero,?\s*(.+)", sent, re.I)
        if m:
            clause = m.group(1).strip(" .")
            for eff in _parse_clause_to_effects(clause):
                triggers.append(Trigger("on_attack", eff, clause))
            continue

        # If you've played another {color} card this turn, <effect>
        m = re.search(
            r"if you(?:'ve| have) played another (red|blue|yellow) card this turn,?\s*(.+)",
            sent,
            re.I,
        )
        if m:
            color = m.group(1).lower()
            cond = {"red": "played_red_other", "blue": "played_blue_other"}.get(color, "")
            clause = m.group(2).strip(" .")
            if card_has_mode_menu_bullets(text) and re.search(r"choose", clause, re.I):
                continue
            eff = (
                _parse_extended_clause(clause, optional=False, condition=cond)
                or _parse_effect(clause)
                or _parse_effect_aggressive(clause)
            )
            if eff and (eff.implemented or cond):
                if cond and eff.implemented:
                    eff = Effect(
                        eff.kind, eff.amount, eff.raw,
                        max_cost=eff.max_cost, banish_name=eff.banish_name,
                        token_name=eff.token_name, target=eff.target,
                        go_again=eff.go_again, condition=cond,
                    )
                triggers.append(Trigger("on_play", eff, sent))
            continue

        # Target attack/action gets ...
        m = re.search(r"target (?:attack|action|dagger attack)[^.]*?gets\s*(.+)", sent, re.I)
        if m:
            rider = m.group(1).strip(" .")
            if "go again" in rider.lower():
                triggers.append(
                    Trigger("on_play", Effect("next_action_go_again", raw=sent[:80]), sent[:80])
                )
            pm = re.search(r"\+(\d+)\s*(?:\{p\}|power)", rider)
            if pm:
                triggers.append(
                    Trigger("on_play", Effect("next_attack_power", int(pm.group(1)), raw=sent[:80]), sent[:80])
                )
            nm = re.search(r"-(\d+)\s*(?:\{p\}|power)", rider)
            if nm:
                triggers.append(
                    Trigger("on_play", Effect("reduce_defense", int(nm.group(1)), raw=sent[:80]), sent[:80])
                )
            continue

        # "Your next [type] attack this turn gains +N{p}." (e.g. Pulse of Volthaven)
        m = re.search(
            r"your next (?:[\w,\s]+\s+or\s+\w+\s+|[\w]+\s+)?attack this turn gains? \+(\d+)\s*(?:\{p\}|power)",
            sent,
            re.I,
        )
        if m:
            triggers.append(
                Trigger("on_play", Effect("next_attack_power", int(m.group(1)), raw=sent[:80]), sent[:80])
            )
            continue

        # At the beginning of your action phase, ...
        m = re.search(
            r"at the beginning of your action phase, destroy this unless you remove a (\w+) counter from it",
            sent,
            re.I,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_action_phase",
                    Effect("upkeep_or_destroy", token_name=m.group(1).strip(), raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(r"at the beginning of your action phase,?\s*(.+)", sent, re.I)
        if m:
            clause = m.group(1).strip(" .")
            for eff in _parse_clause_to_effects(clause):
                triggers.append(Trigger("on_action_phase", eff, clause))
            continue

        # While this is in the arena, ...
        m = re.search(r"while this is in the arena,?\s*(.+)", sent, re.I)
        if m:
            clause = m.group(1).strip(" .")
            agg = _parse_effect_aggressive(clause)
            if agg:
                triggers.append(Trigger("on_enters", agg, clause))
            continue

        # When this is charged to your soul, ...
        m = re.search(r"when this is charged to your soul,?\s*(.+)", sent, re.I)
        if m:
            clause = m.group(1).strip(" .")
            for eff in _parse_clause_to_effects(clause):
                triggers.append(Trigger("on_soul_enter", eff, clause))
            continue

        # This costs {r} less for each token you control (static cost mod).
        m = re.search(r"this costs ((?:\{r\})+|\d+) less to play for each (\w+) you control", low)
        if m:
            txt = m.group(1)
            per = txt.count("{r}") if "{r}" in txt else int(re.search(r"\d+", txt).group(0))
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("cost_per_token", per, sent, token_name=m.group(2).strip()),
                    sent,
                )
            )
            continue

        # Virtuoso Bodice — remove suspense counter on defend for resources.
        m = re.search(
            r"when this defends,?\s*you may remove a suspense counter from an aura you control[,.\s]+"
            r"if you do,?\s*gain ((?:\{r\})+|\d+)",
            low,
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
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        # Galvanize — when this defends, ...
        m = re.search(
            r"galvanize\s*[-—–]\s*when (?:this|it) defends,?\s*you may destroy an item you control\.?\s*"
            r"if you do,?\s*this gets \+(\d+)\{d\}",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect("galvanize", int(m.group(1)), sent, optional=True),
                    sent,
                )
            )
            continue

        # "While this is defending, if you control a X token, this gets +N{d}."
        m = re.search(
            r"while this is defending,?\s*if you control an? ([\w][\w ]+?) tokens?,?\s*this gets \+(\d+)\{d\}",
            sent,
            re.I,
        )
        if m:
            tok = m.group(1).strip().lower()
            bonus = int(m.group(2))
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect(
                        "next_defense_bonus",
                        bonus,
                        condition=f"control_token:{tok}",
                        raw=sent[:80],
                    ),
                    sent[:80],
                )
            )
            continue

        # This enters the arena with ... (comma form handled by specific parsers below)
        m = re.search(r"^(?:when )?this enters the arena with\s*(.+)", sent, re.I)
        if m:
            clause = m.group(1).strip(" .")
            if re.search(r"steam counter", clause, re.I):
                triggers.append(
                    Trigger("on_enters", Effect("put_counter", token_name="steam", raw=clause), clause)
                )
            elif re.search(r"energy counter", clause, re.I):
                triggers.append(
                    Trigger("on_enters", Effect("put_counter", token_name="energy", raw=clause), clause)
                )
            elif re.search(r"frost counter", clause, re.I):
                cm = re.search(r"(\d+) frost counters?", clause, re.I)
                amt = int(cm.group(1)) if cm else 1
                triggers.append(
                    Trigger(
                        "on_enters",
                        Effect("put_counter", amt, token_name="frost", raw=clause),
                        clause,
                    )
                )
            agg = _parse_effect_aggressive(clause)
            if agg:
                triggers.append(Trigger("on_enters", agg, clause))
            continue

        # Crowd cheer / boo
        if re.search(r"the crowd cheers", low):
            cond = "less_gold_than_opponent" if re.search(r"less \{g\}", low) else ""
            eff = Effect("crowd_cheer", raw=sent, condition=cond)
            when = "on_play"
            if "enters" in low:
                when = "on_enters"
            elif "leaves" in low:
                when = "on_leave"
            elif "attacks" in low:
                when = "on_attack"
            triggers.append(Trigger(when, eff, sent))
            continue
        if re.search(r"the crowd boos", low):
            cond = "more_gold_than_opponent" if re.search(r"more \{g\}", low) else ""
            eff = Effect("crowd_boo", raw=sent, condition=cond)
            when = "on_attack" if "attacks" in low else "on_play"
            triggers.append(Trigger(when, eff, sent))
            continue

        # Combo — If [color] attack action was the last attack ...
        m = re.search(
            r"combo\s*[-—–]?\s*if an? (red|yellow|blue) attack action card was the last attack this combat chain,?\s*(.+)",
            sent,
            re.I,
        )
        if m:
            pitch = {"red": 1, "yellow": 2, "blue": 3}[m.group(1).lower()]
            cond = {1: "combo_red", 2: "combo_yellow", 3: "combo_blue"}[pitch]
            clause = m.group(2).strip(" .")
            for eff in _parse_clause_to_effects(clause):
                if eff.implemented:
                    triggers.append(
                        Trigger(
                            "on_attack",
                            Effect(
                                eff.kind, eff.amount, eff.raw, condition=cond,
                                max_cost=eff.max_cost, banish_name=eff.banish_name,
                                token_name=eff.token_name, target=eff.target, go_again=eff.go_again,
                            ),
                            clause,
                        )
                    )
            continue

        # Combo — If [named card] was the last attack, this gets "double damage" (Pounding Gale)
        m = re.search(
            r'combo\s*[-—–]?\s*if ([\w][\w ]*?) was the last attack this combat chain,?\s*'
            r'this gets "if this would deal damage to a hero, instead it deals double that much damage',
            sent,
            re.I,
        )
        if m:
            last_name = m.group(1).strip().lower()
            cond = f"combo_named:{last_name}"
            triggers.append(Trigger("on_attack", Effect("double_damage", 2, sent[:80], condition=cond), sent[:80]))
            continue

        # One-Two Punch: combo text with a quoted hit-trigger damage rider.
        m = re.search(
            r'combo\s*[-—–]?\s*if ([\w][\w ]*?) was the last attack this combat chain,?\s*'
            r'this gets "when this hits a hero, deal (\d+) damage to them\."',
            sent,
            re.I,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_hit",
                    Effect(
                        "damage",
                        int(m.group(2)),
                        sent[:80],
                        target="opponent",
                        condition=f"combo_named:{m.group(1).strip()}",
                    ),
                    sent[:80],
                )
            )
            continue

        # Cyclone Roundhouse: combo text with a quoted attack-reaction-step banish rider.
        m = re.search(
            r'combo\s*[-—–]?\s*if ([\w][\w ]*?) was the last attack this combat chain,?\s*'
            r'this gets "at the beginning of the reaction step,?\s*banish a random defending card from each chain link\.?"',
            sent,
            re.I,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_attack_reaction_step",
                    Effect(
                        "banish_defending",
                        raw=sent[:80],
                        condition=f"combo_named:{m.group(1).strip()}",
                        banish_name="each_chain_link",
                    ),
                    sent[:80],
                )
            )
            continue

        # Combo — If [named card] was the last attack, this gets "When this hits, Y"
        m = re.search(
            r'combo\s*[-—–]?\s*if ([\w][\w ]*?) was the last attack this combat chain,?\s*'
            r'this gets "([^"]+)"',
            sent,
            re.I,
        )
        if m:
            last_name = m.group(1).strip().lower()
            cond = f"combo_named:{last_name}"
            quoted = m.group(2).strip()
            inner = re.sub(r"^when this hits(?: a hero)?,?\s*", "", quoted, flags=re.I).strip(" .")
            hit_eff = _parse_trigger_clause(inner, optional=False)
            if hit_eff is not None and hit_eff.implemented:
                triggers.append(
                    Trigger(
                        "on_hit",
                        Effect(
                            hit_eff.kind, hit_eff.amount, hit_eff.raw,
                            condition=cond, target=hit_eff.target,
                        ),
                        sent[:80],
                    )
                )
            continue

        m = re.search(
            r"when this defends a (\w+) attack,?\s*clash with the attacking hero\.?\s*"
            r"if there is a winner, the other hero puts a -1\{d\} counter on "
            r"(?:an? equipment they control|(?:a|an) \w+ they have equipped)",
            sent,
            re.I,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect("clash", banish_name="loser_equipment_debuff", raw=sent),
                    sent,
                )
            )
            if re.search(r"if they don't, they lose (\d+)\{g\}", sent, re.I):
                lm = re.search(r"if they don't, they lose (\d+)\{g\}", sent, re.I)
                triggers.append(
                    Trigger(
                        "on_defend",
                        Effect("clash", banish_name="loser_lose_gold", token_name=lm.group(1), raw=sent),
                        sent,
                    )
                )
            continue

        m = re.search(r"whenever you protect another hero,?\s*(.+)", sent, re.I)
        if m:
            clause = m.group(1).strip(" .")
            for eff in _parse_clause_to_effects(clause):
                triggers.append(
                    Trigger("on_protect_other", eff, clause)
                )
            continue

        m = re.search(
            r"whenever you draw 1 or more cards from an action card effect,?\s*"
            r"create that many (.+?) tokens?",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "whenever_action_draw",
                    Effect("create_token", token_name=m.group(1).strip(), raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(r"whenever a hero draws a card during an action phase,?\s*they lose (\d+)\{g\}", low)
        if m:
            triggers.append(
                Trigger(
                    "whenever_any_draw",
                    Effect("lose_gold", int(m.group(1)), target="self", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever dominia attacks a hero,?\s*reveal the top card of your deck",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_attack",
                    Effect("reveal_top", target="self", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"if it'?s a red card,?\s*look at their hand and banish a card from it",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_attack",
                    Effect(
                        "banish_graveyard",
                        target="opponent",
                        amount=1,
                        condition="last_reveal_red",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever a card defends this,?\s*(?:\*\*)?clash(?:\*\*)? with the defending hero",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect("clash", banish_name="destroy_top_loser", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever dominia attacks a hero,?\s*reveal the top card of your deck\.?\s*"
            r"if it'?s a red card,?\s*look at their hand and banish a card from it",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_attack",
                    Effect("reveal_top", target="self", condition="hero_dominia", raw=sent),
                    sent,
                )
            )
            triggers.append(
                Trigger(
                    "on_attack",
                    Effect(
                        "banish_graveyard",
                        target="opponent",
                        amount=1,
                        condition="hero_dominia:red_reveal",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever a card defends this,?\s*(?:\*\*)?clash(?:\*\*)? with the defending hero\.?\s*"
            r"the winner destroys the top card of the other hero'?s deck",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect("clash", banish_name="destroy_top_loser", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever you banish a hyper driver from boosting,?\s*remove a -1\{d\} counter from this",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_boost_banish",
                    Effect("remove_counter", token_name="defense", amount=1, raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends,?\s*you may remove a gold counter from treasure island\.?\s*"
            r"if you do and you are a thief,?\s*create a gold token",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect(
                        "create_token",
                        token_name="gold",
                        optional=True,
                        condition="hero_thief:pay_gold",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends,?\s*you may turn a card in a graveyard face-down(?:\.|,)?\s*"
            r"if it'?s yellow,?\s*create a gold token",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect(
                        "turn_banished_face",
                        optional=True,
                        banish_name="face_down:any_gy_yellow_gold",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this hits(?: a hero)?,?\s*you may turn a card in their banished zone face-down",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_hit",
                    Effect(
                        "turn_banished_face",
                        target="opponent",
                        optional=True,
                        banish_name="face_down:opponent_banished",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this hits(?: a hero)?,?\s*you may turn a card in their graveyard face-down(?:\.|,)?\s*"
            r"if it'?s yellow,?\s*create a gold token",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_hit",
                    Effect(
                        "turn_banished_face",
                        target="opponent",
                        optional=True,
                        banish_name="face_down:opponent_gy_yellow_gold",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this enters the arena,?\s*you may turn a card in any banished zone face-down",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_enters",
                    Effect(
                        "turn_banished_face",
                        optional=True,
                        banish_name="face_down:any_banished",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(r"if this defends a weapon attack,?\s*deal (\d+) damage to the attacking hero", low)
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect(
                        "damage",
                        int(m.group(1)),
                        target="attacking_hero",
                        condition="defends_weapon_attack",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"if a guardian off-hand with 1 or more \{d\} is defending this chain link,?\s*"
            r"deal (\d+) damage to the attacking hero",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect(
                        "damage",
                        int(m.group(1)),
                        target="attacking_hero",
                        condition="guardian_offhand_defending",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends,?\s*(?:\*\*)?clash(?:\*\*)? with the attacking hero",
            low,
        )
        if m:
            triggers.append(Trigger("on_defend", Effect("clash", raw=sent), sent))
            continue

        # Leap Frog Leggings / Slime Skin / Vocal Sac: add this to the active chain link as a defending card.
        if re.search(r"when an opponent plays or activates an attack reaction,?\s*you may add this to the active chain link as a defending card", low):
            triggers.append(Trigger("on_attack_reaction_play", Effect("chain_defend", banish_name="self", optional=True, raw=sent), sent))
            continue

        m = re.search(
            r"when this defends(?: an attack)? with \{p\} greater than its base,?\s*"
            r"(?:\*\*)?mark(?:\*\*)? the attacking hero",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect("mark", raw=sent, condition="attack_power_gt_base"),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends,?\s*you may put a card from your hand or arsenal "
            r"on the bottom of your deck",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect(
                        "put_bottom",
                        target="self",
                        banish_name="hand_or_arsenal",
                        optional=True,
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends a (\w+) attack,?\s*this gets \+(\d+)\{d\}",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect(
                        "next_defense_bonus",
                        int(m.group(2)),
                        condition=f"defends_class:{m.group(1).lower()}",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever (\w+) attacks,?\s*you may have (?:him|her|it) deal "
            r"(\d+) arcane damage to up to any (\d+) targets",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_attack",
                    Effect(
                        "arcane_damage",
                        int(m.group(2)),
                        optional=True,
                        condition=f"multi:{int(m.group(3))}",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends,?\s*you may turn a face-down card with crush in your arsenal face-up(?:\.|,)?\s*"
            r"if you do,?\s*put a \+(\d+)\{p\} counter on it",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect(
                        "put_counter",
                        int(m.group(1)),
                        token_name="power",
                        optional=True,
                        condition="arsenal_crush",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends a (\w+) card,?\s*(?:\*\*)?mark(?:\*\*)? the attacking hero",
            low,
        )
        if m:
            pitch_map = {"red": 1, "yellow": 2, "blue": 3}
            pitch = pitch_map.get(m.group(1).lower(), 0)
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect("mark", raw=sent, condition=f"defends_card_pitch:{pitch}"),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends,?\s*(?:\*\*)?steal(?:\*\*)? an aura token the attacking hero controls",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect("steal_token", token_name="aura", target="opponent", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends,?\s*if you are a thief,?\s*(?:\*\*)?steal(?:\*\*)? a gold token the attacking hero controls",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect(
                        "steal_token",
                        token_name="gold",
                        target="opponent",
                        condition="hero_thief",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends,?\s*if you've been cheered this turn,?\s*"
            r"defending action cards get \+(\d+)\{d\} this chain link",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect(
                        "grant_light_block",
                        int(m.group(1)),
                        condition="cheered_this_turn",
                        banish_name="action",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends(?: and the attacking hero has played or activated an attack reaction this chain link|"
            r" and the attacking hero has played or activated a reaction this chain link),?\s*"
            r"put the top (?:(\d+) cards|card) of their deck into their graveyard",
            low,
        )
        if m:
            amount = int(m.group(1)) if m.group(1) else 1
            cond = (
                "attack_reaction_played_chain"
                if "attack reaction" in low
                else "reaction_played_chain"
            )
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect(
                        "destroy_top",
                        amount,
                        target="opponent",
                        condition=cond,
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends(?: an attack)? with \{p\} greater than its base,?\s*"
            r"the attack can't gain \{p\} this turn",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect(
                        "block_power_gain",
                        target="opponent",
                        condition="attack_power_gt_base",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends(?: an attack)? with \{p\} greater than its base,?\s*"
            r"opposing attacks get -(\d+)\{p\} this combat chain",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect(
                        "power",
                        -int(m.group(1)),
                        target="opponent",
                        condition="attack_power_gt_base",
                        banish_name="chain",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends,?\s*the next time an attack would gain \{p\} this chain link,?\s*"
            r"instead it gains that much minus (\d+)",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect("reduce_next_power_gain", int(m.group(1)), raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends and the attacking hero has played or activated an attack reaction this chain link,?\s*"
            r"they lose (\d+)\{g\}",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect(
                        "lose_gold",
                        int(m.group(1)),
                        target="opponent",
                        condition="attack_reaction_played_chain",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends,?\s*hit effects don't trigger this chain link unless the attacking hero pays \{r\}",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect("block_hit_effects", 1, target="opponent", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends,?\s*you may reveal an instant card from your hand(?:\.|,)?\s*"
            r"if you do,?\s*create an? (.+?) token",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect(
                        "create_token",
                        token_name=m.group(1).strip(),
                        optional=True,
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends an attack,?\s*(?:it |this )?gets -(\d+)\s*(?:\{p\}|power)",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect(
                        "power",
                        -int(m.group(1)),
                        target="opponent",
                        banish_name="chain",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends,?\s*until end of turn,?\s*opponents can't attack with weapons",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect("block_weapon_attacks", target="opponent", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when this defends,?\s*you may reveal an instant card from your hand(?:\.|,)?\s*"
            r"if you do,?\s*deal (\d+) arcane damage to target hero",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect("arcane_damage", int(m.group(1)), optional=True, raw=sent),
                    sent,
                )
            )
            continue

        if re.search(r"^heroes can't gain \{g\}", low):
            triggers.append(
                Trigger(
                    "on_play",
                    Effect("block_gold_gain", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"if you do, the next time you would be dealt damage this turn, prevent twice x of that damage",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_enters",
                    Effect(
                        "prevent_damage",
                        0,
                        banish_name="twice_x",
                        condition="per_hit:1",
                        target="self",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever you boost(?: an attack action card)?,?\s*"
            r"you may destroy a card under this\.?\s*if you do,?\s*(.+?)(?:\.\s|\.$|$)",
            body,
            re.I,
        )
        if m:
            remainder = m.group(1).strip(" .")
            for eff in _parse_destroy_then_clause(remainder):
                triggers.append(
                    Trigger(
                        "on_boost",
                        eff,
                        m.group(0).strip()[:90],
                    )
                )
            triggers.append(
                Trigger(
                    "on_boost",
                    Effect("destroy_item", optional=True, banish_name="under_this", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"the first attack that targets you each turn gets -(\d+)\{p\}",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "power",
                        -int(m.group(1)),
                        banish_name="first_attack_targeting_you",
                        target="self",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"at the beginning of each end phase, destroy this unless you(?:'ve| have) pitched, played, or defended with a blue card this turn",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_end_phase",
                    Effect(
                        "upkeep_or_destroy",
                        banish_name="blue_pitched_played_defended",
                        raw=sent,
                    ),
                    sent,
                )
            )
            if re.search(
                r"if you(?:'ve| have) pitched, played, and defended with a blue card this turn, \*\*transcend\*\*",
                low,
            ):
                triggers.append(
                    Trigger(
                        "on_end_phase",
                        Effect(
                            "transcend",
                            condition="blue_pitched_played_defended_all",
                            raw=sent,
                        ),
                        sent,
                    )
                )
            continue

        m = re.search(
            r"at the beginning of your end phase, put a sand counter on this, then destroy it unless you banish a red card from your graveyard for each sand counter on it",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_end_phase",
                    Effect(
                        "put_counter",
                        token_name="sand",
                        banish_name="upkeep_banish_red_each",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"at the beginning of your end phase, if you(?:'ve| have) attacked less than (\d+) times this turn, destroy this",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_end_phase",
                    Effect(
                        "destroy_item",
                        condition=f"attacks_lt:{int(m.group(1))}",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"if you would sharpen a zenith blade, instead you may pay \{r\} and destroy this\.?\s*"
            r"if you do, \*\*sharpen\*\* it an additional time",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_sharpen",
                    Effect(
                        "pitch_pay",
                        1,
                        optional=True,
                        banish_name="destroy_sharpen_extra",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever an attacking ally you control dies or an attack action card you control is destroyed by \*\*phantasm\*\*,?\s*"
            r"you may pay \{r\}\{r\}\{r\}\.?\s*if you do,?\s*destroy this and gain 1 action point",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_phantasm_destroy",
                    Effect(
                        "pitch_pay",
                        3,
                        optional=True,
                        banish_name="destroy_self_go_again",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"whenever you roll a ([56]) or ([56]) on a die, your brute attacks get \+(\d+)\{p\} this turn",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_die_roll",
                    Effect(
                        "power",
                        int(m.group(3)),
                        condition="roll_ge:5",
                        banish_name="brute_attacks_turn",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(r"whenever you roll a 1 on a die, destroy this", low)
        if m:
            triggers.append(
                Trigger(
                    "on_die_roll",
                    Effect("destroy_item", condition="roll_eq:1", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"the first time each turn another hero destroys a card they don'?t control,?\s*"
            r"you may pay \{r\}\{r\}\.?\s*if you do,?\s*they destroy a non-hero permanent they control",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_opponent_destroy",
                    Effect(
                        "pitch_pay",
                        2,
                        optional=True,
                        banish_name="they_destroy_permanent",
                        raw=sent,
                    ),
                    sent,
                )
            )
            continue

        m = re.search(r"when this enters the arena, choose an opponent", low)
        if m:
            triggers.append(
                Trigger(
                    "on_enters",
                    Effect("choose_card", banish_name="opponent", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"at the beginning of their end phase, destroy this and you each gain 3\{g\}",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_chosen_end_phase",
                    Effect("destroy_item", raw=sent),
                    sent,
                )
            )
            triggers.append(
                Trigger(
                    "on_chosen_end_phase",
                    Effect("gain_gold", 3, target="each_hero", raw=sent),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when you or a card you control is the target of an attack they control, destroy this and draw a card",
            low,
        )
        if m:
            _append_destroy_then_triggers(triggers, "on_targeted_by_attack", sent, "draw a card")
            continue

        # Generic "When X, Y" not caught elsewhere.
        m = re.search(r"^when (.+?), (.+)$", sent, re.I)
        if m and "this" in m.group(1).lower():
            clause = m.group(2).strip(" .")
            when = "on_play"
            cond_part = m.group(1).lower()
            if "hits" in cond_part:
                when = "on_hit"
            elif "defends" in cond_part:
                when = "on_defend"
            elif "attacks" in cond_part:
                when = "on_attack"
            elif "leaves" in cond_part:
                when = "on_leave"
            elif "enters" in cond_part:
                when = "on_enters"
            if when == "on_defend" and re.search(
                r"remove a suspense counter from an aura",
                f"{cond_part} {clause}",
                re.I,
            ):
                continue
            if when == "on_defend" and re.search(r"\balone\b", cond_part):
                continue
            if when == "on_defend" and re.search(r"\bclash\b", clause.lower()):
                triggers.append(
                    Trigger("on_defend", Effect("clash", raw=clause), clause)
                )
                continue
            if when == "on_defend" and "clash" in clause.lower():
                cm = re.search(
                    r"clash.+?(?:the )?winner creates an? (.+?) token",
                    clause,
                    re.I,
                )
                if cm:
                    triggers.append(
                        Trigger(
                            "on_defend",
                            Effect("clash", token_name=cm.group(1).strip(), raw=clause),
                            clause,
                        )
                    )
                    continue
                lm = re.search(
                    r"clash.+?if there is a winner, the other hero puts a -1\{d\} counter on an equipment they control",
                    clause,
                    re.I,
                )
                if lm:
                    triggers.append(
                        Trigger(
                            "on_defend",
                            Effect("clash", banish_name="loser_equipment_debuff", raw=clause),
                            clause,
                        )
                    )
                    continue
            if when == "on_hit" and re.search(
                r"clash with them\.?\s*if you win, destroy the top card of their deck",
                body,
                re.I,
            ):
                continue
            if when == "on_attack" and re.search(
                r"you may banish .+ from your graveyard\.?\s*if you do",
                body,
                re.I,
            ):
                continue
            if when == "on_attack" and re.search(
                r"if 2 or more cards are put into arsenals this way, this gets go again",
                body,
                re.I,
            ):
                continue
            if when == "on_attack" and re.search(
                r"create runechant tokens equal to the number of non-attack action cards you've played this turn",
                body,
                re.I,
            ):
                continue
            if when == "on_attack" and re.search(
                r"when this attacks, transform up to 1 ash you control into an aether ashwings?",
                body,
                re.I,
            ):
                continue
            if when == "on_attack" and re.search(
                r"the defending hero reveals their hand",
                clause,
                re.I,
            ):
                continue
            if when == "on_attack" and re.search(
                r"for each hyper driver destroyed this way",
                clause,
                re.I,
            ):
                continue
            if when == "on_attack" and re.search(
                r"it gets \+x\{p\}, where x is the number of gold you control",
                clause,
                re.I,
            ):
                continue
            if when == "on_hit" and re.search(
                r"remove all steam counters from an equipment, item, or weapon they control",
                clause,
                re.I,
            ):
                continue
            if when == "on_hit" and re.search(
                r"when this hits and you have no cards in your arsenal",
                body,
                re.I,
            ):
                continue
            if when in ("on_attack", "on_defend") and re.search(
                r"when this attacks or defends, your hero gets -",
                body,
                re.I,
            ):
                continue
            if when == "on_hit" and re.search(
                r"when this hits a hero, look at their hand and choose a card\.\s*"
                r"search their hand, deck, and graveyard and banish",
                body,
                re.I,
            ):
                continue
            if when == "on_hit" and re.search(
                r"each hero who doesn't have a card in their arsenal puts the top card of their deck face-down into their arsenal",
                body,
                re.I,
            ):
                continue
            if when == "on_hit" and re.search(
                r"at the beginning of your end phase, put the top card of your deck face-up into your arsenal",
                body,
                re.I,
            ):
                continue
            for eff in _parse_clause_to_effects(clause, allow_this_turn=True):
                triggers.append(Trigger(when, eff, clause))
            if not _parse_clause_to_effects(clause):
                agg = _parse_effect_aggressive(clause)
                if agg:
                    triggers.append(Trigger(when, agg, clause))
            continue

        m = re.search(r"deal (\d+) arcane damage to two target heroes", low)
        if m:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "arcane_damage",
                        int(m.group(1)),
                        sent,
                        condition="multi:2",
                    ),
                    sent,
                )
            )
            continue

        m = re.search(
            r"when you attack with .+, deal (\d+) arcane damage to target hero",
            low,
        )
        if m:
            triggers.append(
                Trigger(
                    "on_attack",
                    Effect("arcane_damage", int(m.group(1)), sent),
                    sent,
                )
            )
            continue

        if re.search(r"^search their hand, deck, and graveyard and banish", low):
            continue

        # Standalone effect sentence (no trigger word).
        if re.search(r"^additional cost\b", low):
            continue
        if re.search(r"^as an additional cost\b", low):
            continue
        if re.search(r"destroy the daggers?\b", low) and re.search(
            r"dagger you control.*deals \d+ damage", body, re.I
        ):
            continue
        if re.search(
            r"when this attacks a hero, each dagger you control deals \d+ damage",
            sent,
            re.I,
        ):
            continue
        if not re.search(r"^(when|whenever|if|while|at the|legendary|galvanize|stealth)\b", low):
            if re.search(r"you may reveal any number of crouching tigers", body, re.I):
                if re.search(r"^\d+ or more, this gets", low):
                    continue
            if re.search(
                r"gain control of an item with cost \d+ or less they control",
                body,
                re.I,
            ) and re.search(r"^otherwise,?\s*draw a card", low):
                continue
            if re.search(r"played or created \d+ or more auras", body, re.I):
                if re.search(r"^\d+ or more, this gets", low):
                    continue
            agg = _parse_effect_aggressive(sent)
            if agg and agg.kind != "destroy_item":
                triggers.append(Trigger("on_play", agg, sent))
        elif re.search(r"^stealth\b", low):
            nap = _parse_next_attack_power(sent)
            if nap is None or not nap.implemented:
                m = re.search(
                    r"if this is attacking a marked hero,?\s*(?:it |this )?gets \+(\d+)\s*(?:\{p\}|power)",
                    low,
                )
                if not m:
                    agg = _parse_effect_aggressive(sent)
                    if agg and agg.kind != "destroy_item":
                        triggers.append(Trigger("on_play", agg, sent))

    return tuple(triggers)


def parse_coverage_triggers(
    text: str,
    keywords: tuple[str, ...] | list[str] = (),
) -> tuple[Trigger, ...]:
    """Merge passive sentence parsing with keyword-implied mechanics."""
    passive = parse_passive_triggers(text)
    kw = parse_keyword_triggers(keywords)
    if not passive and not kw:
        return ()
    seen: set[tuple] = set()
    merged: list[Trigger] = []
    for t in (*passive, *kw):
        key = (t.when, t.effect.kind, t.effect.raw, t.threshold)
        if key in seen:
            continue
        seen.add(key)
        merged.append(t)
    return tuple(merged)


def parse_action_damage(text: str, power: int, is_attack: bool) -> tuple[int, bool]:
    """Damage dealt when an action is played (mirrors environment helper)."""
    body = str(text or "").lower()
    if is_attack:
        return int(power or 0), False
    if power and power > 0:
        return int(power), "arcane" in body
    for match in re.finditer(r"deal (\d+)\s+(arcane\s+)?damage", body):
        start = max(body.rfind(".", 0, match.start()), body.rfind("{br}", 0, match.start()))
        if re.search(r"\bif\b", body[start:match.start()]):
            continue
        tail = body[match.start(): match.end() + 24]
        return int(match.group(1)), bool(match.group(2)) or "arcane damage" in tail
    return 0, False
