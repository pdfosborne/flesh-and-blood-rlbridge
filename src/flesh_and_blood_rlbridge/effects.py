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

# Effect kinds the engine knows how to resolve faithfully.
SUPPORTED_EFFECTS = {
    "go_again",
    "damage",
    "arcane_damage",
    "power",
    "draw",
    "next_naa_go_again",
    "next_attack_power",
    "banish_combo",
}

# Phrases that signal a stateful / optional / open-ended effect we will not
# fake. Their presence in a trigger clause forces an ``unimplemented`` effect.
_COMPLEX_MARKERS = (
    "whenever",
    "this turn",
    'gains "',
    "gains \u201c",
    "for each",
    "instead",
    "unless",
    "as though",
    "may play",
    "search",
    "create",
    "destroy",
    "banish",
    "if you do",
    "put it",
    "discard",
    "return",
)


@dataclass(frozen=True)
class Effect:
    kind: str          # one of SUPPORTED_EFFECTS or "unimplemented"
    amount: int = 0
    raw: str = ""
    max_cost: int = -1     # cost cap for next_attack_power (-1 = no cap)
    banish_name: str = ""  # graveyard card to banish for banish_combo
    go_again: bool = False  # rider grants go again (banish_combo)
    optional: bool = False  # "you may ..." -> player chooses whether to apply

    @property
    def implemented(self) -> bool:
        return self.kind in SUPPORTED_EFFECTS


@dataclass(frozen=True)
class Trigger:
    when: str          # "on_play" | "on_attack" | "on_hit" | "when_fused"
    effect: Effect
    raw: str


@dataclass(frozen=True)
class ActivatedAbility:
    """An activated ability on a permanent (equipment / weapon / hero).

    Modeled after Talishar's playable arena abilities: an optional cost is paid
    (resources, and an action point for "Action" abilities) to apply an effect.
    Abilities whose cost or effect we cannot model faithfully are kept with an
    ``unimplemented`` effect so the engine can show them without offering them.
    """

    effect: Effect
    cost: int = 0                  # resource ({r}) cost
    uses_action_point: bool = True  # "Action" costs 1 AP; "Instant" costs 0
    once_per_turn: bool = False
    raw: str = ""

    @property
    def implemented(self) -> bool:
        return self.effect.implemented


def _clean(text: str) -> str:
    body = str(text or "")
    body = body.replace("{br}", ". ").replace("**", "")
    body = body.replace("{i}", "").replace("{/i}", "")
    return body


def _parse_effect(clause: str) -> Effect:
    c = " ".join(clause.lower().split())
    if not c:
        return Effect("unimplemented", raw=clause.strip())

    # "you may <effect>" is an optional effect the player chooses to apply.
    optional = False
    if c.startswith("you may "):
        optional = True
        c = c[len("you may "):]

    if any(marker in c for marker in _COMPLEX_MARKERS):
        return Effect("unimplemented", raw=clause.strip())

    if "go again" in c:
        return Effect("go_again", raw=clause.strip(), optional=optional)

    m = re.search(r"deal (\d+)\s+(arcane\s+)?damage", c)
    if m:
        kind = "arcane_damage" if m.group(2) else "damage"
        return Effect(kind, int(m.group(1)), clause.strip(), optional=optional)

    m = re.search(r"(?:gets|gain|gains)\s*\+(\d+)\s*(?:\{p\}|power)", c)
    if m:
        return Effect("power", int(m.group(1)), clause.strip(), optional=optional)

    m = re.search(r"draw (a|an|one|\d+) cards?", c)
    if m:
        n = 1 if m.group(1) in ("a", "an", "one") else int(m.group(1))
        return Effect("draw", n, clause.strip(), optional=optional)

    return Effect("unimplemented", raw=clause.strip())


@functools.lru_cache(maxsize=8192)
def parse_triggers(text: str) -> tuple[Trigger, ...]:
    """Extract triggered effects from a card's rules text."""
    body = _clean(text)
    if not body.strip():
        return ()

    triggers: list[Trigger] = []

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

    # Pending buff: "the next [non-attack] action you play this turn gains go
    # again" -> recognized explicitly (despite the "this turn" marker).
    if re.search(
        r"the next (?:non-attack action|attack action|action)[^.]*?gains? go again",
        body,
        re.I,
    ):
        triggers.append(Trigger("on_play", Effect("next_naa_go_again", raw="next action gains go again"), "next action gains go again"))

    # On-attack graveyard combo: "you may banish a <X> from your graveyard. If
    # you do, this gets +N power [and go again]" (e.g. Jack Be Quick).
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

    # "... if (this|it) was fused, <effect>"
    for m in re.finditer(r"if (?:this|it) (?:was|is) fused,?\s*([^.]*)\.?", body, re.I):
        triggers.append(Trigger("when_fused", _parse_effect(m.group(1)), m.group(1).strip()))

    # On-attack: "when (this|it) attacks, <effect>" (fused/banish handled above).
    for m in re.finditer(r"when (?:this|it) attacks,?\s*([^.]*)\.?", body, re.I):
        clause = m.group(1)
        low = clause.lower()
        if "fused" in low or "banish" in low:
            continue
        triggers.append(Trigger("on_attack", _parse_effect(clause), clause.strip()))

    # On-hit: "when(ever) (this|it) hits[ ...], <effect>"
    for m in re.finditer(r"when(?:ever)? (?:this|it) hits[^,]*,\s*([^.]*)\.?", body, re.I):
        triggers.append(Trigger("on_hit", _parse_effect(m.group(1)), m.group(1).strip()))

    return tuple(triggers)


@functools.lru_cache(maxsize=8192)
def parse_activated_abilities(text: str) -> tuple[ActivatedAbility, ...]:
    """Parse "[Once per Turn] Action|Instant -- {cost}: <effect>" abilities.

    Only pure resource ({r}) costs are modeled; abilities with non-resource
    costs (destroy this, banish, turn face-up, ...) or effects we cannot resolve
    are returned with an ``unimplemented`` effect.
    """
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

        residue = re.sub(r"\{r\}", "", cost_area).strip(" -—–")
        if residue:  # non-resource cost we cannot pay faithfully
            effect = Effect("unimplemented", raw=eff_text)
        else:
            effect = _parse_effect(eff_text)

        abilities.append(
            ActivatedAbility(
                effect=effect,
                cost=cost_area.count("{r}"),
                uses_action_point=uses_ap,
                once_per_turn=once,
                raw=" ".join(m.group(0).split())[:90],
            )
        )
    return tuple(abilities)
