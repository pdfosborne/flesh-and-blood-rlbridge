"""Extended rules-text parsers for FAB card effect categories."""

from __future__ import annotations

import re

from .effects import (
    Effect,
    Trigger,
    _clean,
    _parse_clause_effects,
    _parse_effect,
    _parse_trigger_clause,
    card_has_mode_menu_bullets,
)


def parse_clause(clause: str, *, allow_this_turn: bool = False) -> tuple[str, tuple[Effect, ...]]:
    return _parse_clause_effects(clause, allow_this_turn=allow_this_turn)


def parse_single(clause: str, *, allow_this_turn: bool = False) -> Effect:
    _, effects = _parse_clause_effects(clause, allow_this_turn=allow_this_turn)
    if effects and effects[0].implemented:
        return effects[0]
    return _parse_effect(clause, allow_this_turn=allow_this_turn)


def parse_extended_triggers(text: str) -> tuple[Trigger, ...]:
    """Parse trigger patterns not covered by the core parser."""
    body = _clean(text)
    if not body.strip():
        return ()

    triggers: list[Trigger] = []

    # Multi-sentence patterns (body-level).
    m = re.search(
        r"when this defends, you may reveal a card with crush from your hand\.?\s*"
        r"if you do, create an? (.+?) token",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_defend",
                Effect(
                    "create_token",
                    token_name=m.group(1).strip(),
                    optional=True,
                    raw=m.group(0)[:80],
                ),
                m.group(0)[:80],
            )
        )

    m = re.search(
        r"if you have a base (head|chest|arms|legs) equipped, transform it and x hyper drivers you control into this, then equip this",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_enters",
                Effect(
                    "transform_equip",
                    banish_name=f"{m.group(1)}:hyper_drivers",
                    raw=m.group(0)[:80],
                ),
                m.group(0)[:80],
            )
        )

    m = re.search(
        r"if you have a base (head|chest|arms|legs) equipped, transform it into this, then equip this",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_enters",
                Effect("transform_equip", banish_name=m.group(1), raw=m.group(0)[:80]),
                m.group(0)[:80],
            )
        )

    if re.search(r"this counts as a gold", body, re.I):
        triggers.append(
            Trigger("on_enters", Effect("counts_as_gold", raw="counts as gold"), "counts as gold")
        )

    # When this attacks, if this was fused, ...
    for m in re.finditer(
        r"when (?:this|it) attacks,?\s*if this was fused,?\s*(.+?)(?:\.|$)",
        body,
        re.I,
    ):
        clause = m.group(1).strip(" .")
        agg = parse_single(clause, allow_this_turn=True)
        if agg and agg.implemented:
            triggers.append(Trigger("on_attack", agg, clause))
        elif "whenever" in clause.lower():
            from .effect_coverage import _parse_effect_aggressive

            agg2 = _parse_effect_aggressive(clause) or parse_single(clause, allow_this_turn=True)
            if agg2 and agg2.implemented:
                triggers.append(Trigger("on_attack", agg2, clause))

    # Galvanize without full sentence match
    for m in re.finditer(
        r"when (?:this|it) defends,?\s*you may destroy an item you control\.?\s*"
        r"if you do,?\s*this gets \+(\d+)\{d\}(?: until end of turn)?",
        body,
        re.I,
    ):
        clause = m.group(0).strip()
        if re.search(r"galvanize\s*[-—–]", body[: m.start()], re.I):
            triggers.append(
                Trigger(
                    "on_defend",
                    Effect("galvanize", int(m.group(1)), clause, optional=True),
                    clause,
                )
            )

    # Crush — When this deals N+ damage to a [Class] hero, ...
    for m in re.finditer(
        r"crush\s*[-—–]\s*when this deals (\d+) or more damage to a (\w+) hero,?\s*([^.]+)\.?",
        body,
        re.I,
    ):
        threshold = int(m.group(1))
        hero_class = m.group(2).lower()
        clause = m.group(3).strip()
        condition = f"defender_{hero_class}"
        _, effs = parse_clause(clause)
        for eff in effs:
            if eff.implemented:
                triggers.append(
                    Trigger(
                        "on_crush",
                        Effect(
                            eff.kind, eff.amount, eff.raw, condition=condition,
                            max_cost=eff.max_cost, banish_name=eff.banish_name,
                            token_name=eff.token_name, target=eff.target, go_again=eff.go_again,
                        ),
                        clause,
                        threshold=threshold,
                    )
                )
            elif eff.kind == "unimplemented":
                agg = parse_single(clause)
                if agg.implemented:
                    triggers.append(
                        Trigger(
                            "on_crush",
                            Effect(
                                agg.kind, agg.amount, agg.raw, condition=condition,
                                max_cost=agg.max_cost, banish_name=agg.banish_name,
                                token_name=agg.token_name, target=agg.target,
                            ),
                            clause,
                            threshold=threshold,
                        )
                    )

    # Crush — When this deals N+ damage to a hero, ...
    for m in re.finditer(
        r"crush\s*[-—–]\s*when this deals (\d+) or more damage to a hero,?\s*([^.]+)\.?",
        body,
        re.I,
    ):
        threshold = int(m.group(1))
        clause = m.group(2).strip()
        _, effs = parse_clause(clause)
        found = False
        for eff in effs:
            if eff.implemented:
                found = True
                triggers.append(Trigger("on_crush", eff, clause, threshold=threshold))
        if not found:
            cap = re.search(
                r"they can't play attack action cards with (\d+) or less base \{p\}",
                clause,
                re.I,
            )
            if cap:
                triggers.append(
                    Trigger(
                        "on_crush",
                        Effect("play_power_cap", int(cap.group(1)), clause, target="opponent"),
                        clause,
                        threshold=threshold,
                    )
                )

    # Surge — If this deals more than N damage, ...
    for m in re.finditer(
        r"surge\s*[-—–]\s*if this deals more than (\d+) damage,?\s*([^.]+)\.?",
        body,
        re.I,
    ):
        threshold = int(m.group(1))
        clause = m.group(2).strip()
        _, effs = parse_clause(clause)
        for eff in effs:
            if eff.implemented:
                triggers.append(Trigger("on_surge", eff, clause, threshold=threshold))

    # When this deals N+ damage (non-crush header)
    for m in re.finditer(
        r"when this deals (\d+) or more damage to a hero,?\s*([^.]+)\.?",
        body,
        re.I,
    ):
        if re.search(r"crush\s*[-—–]", body[: m.start()], re.I):
            continue
        threshold = int(m.group(1))
        clause = m.group(2).strip()
        _, effs = parse_clause(clause)
        for eff in effs:
            if eff.implemented:
                triggers.append(Trigger("on_crush", eff, clause, threshold=threshold))

    for m in re.finditer(
        r"when (?:this|it) defends a \w+ attack,?\s*clash with the attacking hero\.?\s*"
        r"if there is a winner, the other hero puts a -1\{d\} counter on "
        r"(?:an? equipment they control|(?:a|an) \w+ they have equipped)",
        body,
        re.I,
    ):
        triggers.append(
            Trigger(
                "on_defend",
                Effect("clash", banish_name="loser_equipment_debuff", raw=m.group(0)[:80]),
                m.group(0)[:80],
            )
        )

    # When this defends alone, ...
    for m in re.finditer(r"when (?:this|it) defends alone,?\s*([^.]+)\.?", body, re.I):
        clause = m.group(1).strip()
        _, effs = parse_clause(clause)
        for eff in effs:
            if not eff.implemented:
                continue
            merged = eff
            if eff.condition != "defends_alone":
                merged = Effect(
                    eff.kind,
                    eff.amount,
                    eff.raw,
                    max_cost=eff.max_cost,
                    banish_name=eff.banish_name,
                    token_name=eff.token_name,
                    target=eff.target if eff.target != "opponent" else "self"
                    if eff.kind == "prevent_damage"
                    else eff.target,
                    playable_banished=eff.playable_banished,
                    go_again=eff.go_again,
                    optional=eff.optional,
                    condition="defends_alone",
                )
            triggers.append(Trigger("on_defend", merged, clause))

    # When this defends, ...
    for m in re.finditer(
        r"when (?:this|it) defends(?! alone)(?: a \w+ attack)?,?\s*([^.]+)\.?",
        body,
        re.I,
    ):
        clause = m.group(1).strip()
        if any(
            re.search(pat, clause, re.I)
            for pat in (
                r"you may turn a face-down card with crush in your arsenal",
                r"you may reveal an instant card from your hand",
                r"steal an aura token",
                r"if you've been cheered",
                r"opponents can't attack with weapons",
                r"put the top card of their deck into their graveyard",
                r"gets -\d+\s*(?:\{p\}|power)",
                r"the attack can't gain",
                r"opposing attacks get -",
                r"the next time an attack would gain",
                r"they lose \d+\{g\}",
                r"hit effects don't trigger",
                r"create an? .+ token",
                r"put the top \d+ cards",
            )
        ):
            continue
        if re.search(r"\bclash\b", clause, re.I):
            cm = re.search(
                r"clash.+?(?:the )?winner creates an? (.+?) token",
                clause,
                re.I,
            )
            if cm:
                triggers.append(
                    Trigger(
                        "on_defend",
                        Effect("clash", token_name=cm.group(1).strip(), raw=clause[:80]),
                        clause[:80],
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
                        Effect("clash", banish_name="loser_equipment_debuff", raw=clause[:80]),
                        clause[:80],
                    )
                )
                continue
        _, effs = parse_clause(clause)
        for eff in effs:
            if eff.implemented:
                triggers.append(Trigger("on_defend", eff, clause))

    # When this enters the arena, ...
    for m in re.finditer(r"when (?:this|it) enters the arena,?\s*([^.]+)\.?", body, re.I):
        clause = m.group(1).strip()
        _, effs = parse_clause(clause)
        for eff in effs:
            if eff.implemented:
                triggers.append(Trigger("on_enters", eff, clause))

    # When this is put into a graveyard, ...
    for m in re.finditer(
        r"when (?:this|it) is put into a graveyard[^,]*,?\s*([^.]+)\.?",
        body,
        re.I,
    ):
        clause = m.group(1).strip()
        _, effs = parse_clause(clause)
        for eff in effs:
            if eff.implemented:
                triggers.append(Trigger("on_gy_enter", eff, clause))

    # Whenever you go again this turn, ...
    for m in re.finditer(r"whenever you go again this turn,?\s*([^.]+)\.?", body, re.I):
        clause = m.group(1).strip()
        _, effs = parse_clause(clause)
        for eff in effs:
            if eff.implemented:
                triggers.append(Trigger("whenever_go_again", eff, clause))

    # Target X gets go again / +power (on play auras)
    for m in re.finditer(
        r"target (?:attack|action|hero|dagger attack)[^.]*? gets\s*(?:\*\*)?go again",
        body,
        re.I,
    ):
        triggers.append(
            Trigger("on_play", Effect("next_action_go_again", raw=m.group(0)[:80]), m.group(0)[:80])
        )

    for m in re.finditer(
        r"target attack gets\s*\+(\d+)\s*(?:\{p\}|power)",
        body,
        re.I,
    ):
        triggers.append(
            Trigger(
                "on_play",
                Effect("next_attack_power", int(m.group(1)), raw=m.group(0)[:80]),
                m.group(0)[:80],
            )
        )

    if re.search(r"target attack gets \+x\{p\}", body, re.I):
        triggers.append(
            Trigger(
                "on_play",
                Effect(
                    "modify_attack_power",
                    banish_name="last_banish_count",
                    raw="target attack gets +X power",
                ),
                "target attack gets +X power",
            )
        )

    # Quoted on-hit grants: gets "When this hits, ..."
    for m in re.finditer(r'gets?\s+"when this hits[^"]*"', body, re.I):
        inner = m.group(0)
        qm = re.search(r'"(when this hits[^"]*)"', inner, re.I)
        if not qm:
            continue
        hit_text = re.sub(
            r"^when this hits(?: a hero or ally| a hero|,)?,?\s*",
            "",
            qm.group(1).strip(),
            flags=re.I,
        )
        hit = _parse_trigger_clause(hit_text, optional="you may" in hit_text.lower())
        if hit is not None and hit.implemented:
            triggers.append(
                Trigger(
                    "on_play",
                    Effect(
                        "next_attack_power",
                        0,
                        token_name=f"hit:{hit.kind}:{hit.banish_name or ''}",
                        raw=inner[:80],
                    ),
                    inner[:80],
                )
            )
            continue
        if "go again" in inner.lower():
            triggers.append(
                Trigger("on_play", Effect("next_action_go_again", raw=inner[:80]), inner[:80])
            )
        if "banish" in inner.lower() and not card_has_mode_menu_bullets(text):
            triggers.append(
                Trigger("on_play", Effect("banish_top", target="self", raw=inner[:80]), inner[:80])
            )
        if "play" in inner.lower() and "graveyard" in inner.lower():
            m = re.search(r"you may play an? (.+?) from your graveyard", inner, re.I)
            name = m.group(1).strip() if m else ""
            triggers.append(
                Trigger(
                    "on_hit",
                    Effect("enable_gy_play", banish_name=name, raw=inner[:80]),
                    inner[:80],
                )
            )

    m = re.search(
        r"when this hits, create x (.+?)s in your banished zone, where x is the number of (.+?) you control\. you may play them this turn",
        body,
        re.I,
    )
    if m:
        triggers.append(
            Trigger(
                "on_hit",
                Effect(
                    "create_banished",
                    banish_name=m.group(1).strip(),
                    token_name=m.group(2).strip(),
                    playable_banished=True,
                    raw=m.group(0)[:90],
                ),
                m.group(0)[:90],
            )
        )

    # Static may-play from banished zone.
    if re.search(r"you may play evos from your banished zone", body, re.I):
        triggers.append(
            Trigger(
                "on_enters",
                Effect("grant_may_play", target="banished", banish_name="evo", condition="static", raw="play evos from banish"),
                "play evos from banish",
            )
        )

    if re.search(r"you may play this from your banished zone", body, re.I):
        triggers.append(
            Trigger("on_play", Effect("enable_banish_play", raw="play from banished zone"), "banish play")
        )

    if re.search(r"you may play this from your graveyard", body, re.I):
        triggers.append(
            Trigger("on_play", Effect("enable_gy_play", raw="play from graveyard"), "graveyard play")
        )

    # Clash — When this defends, clash ... create a X token
    for m in re.finditer(
        r"when (?:this|it) defends(?: a \w+ attack)?,?\s*clash[^,]*?(?:the )?winner creates an? (.+?) token",
        body,
        re.I,
    ):
        triggers.append(
            Trigger(
                "on_defend",
                Effect("clash", token_name=m.group(1).strip(), raw=m.group(0)[:80]),
                m.group(0)[:80],
            )
        )

    return tuple(triggers)
