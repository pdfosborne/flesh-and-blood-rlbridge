#!/usr/bin/env python3
"""Scan cards.json and report effects the rlbridge simulator does not implement.

Outputs:
  - unimplemented_effects.json  (full structured report)
  - unimplemented_effects.md    (human-readable summary)

Run from repo root:
  python src/flesh_and_blood_rlbridge/card_db/scan_unimplemented_effects.py
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from flesh_and_blood_rlbridge import effects

CARDS_PATH = Path(__file__).with_name("cards.json")
OUT_JSON = Path(__file__).with_name("unimplemented_effects.json")
OUT_MD = Path(__file__).with_name("unimplemented_effects.md")

# Mechanics we model in environment.py (partial coverage noted in README section).
IMPLEMENTED_KEYWORDS = {
    "go_again",
    "dominate",
    "intimidate",
    "overpower",
    "fusion",
    "battleworn",
    "blade_break",
    "ward",
    "blood_debt",
    "transcend",
    "contract",
    "boost",
    "reload",
    "phalanx",
    "stealth",
}

# Regex buckets for grouping unimplemented / unparsed rules text.
PATTERN_RULES: tuple[tuple[str, str], ...] = (
    ("crush", r"\*\*crush\*\*|\bcrush\b"),
    ("surge", r"\*\*surge\*\*|\bsurge\b"),
    ("opt", r"\*\*opt\b|\bopt \d"),
    ("transcend", r"\btranscend\b"),
    ("contract", r"\bcontract\b"),
    ("boost", r"\*\*boost\*\*|\bboost\b"),
    ("clash", r"\bclash\b"),
    ("ward", r"\*\*ward\b|\bward \d"),
    ("overpower", r"\boverpower\b"),
    ("phalanx", r"\bphalanx\b"),
    ("stealth", r"\bstealth\b"),
    ("blood_debt", r"\bblood debt\b"),
    ("fusion", r"\b[a-z]+ fusion\b|\bfused\b"),
    ("dominate", r"\bdominate\b"),
    ("go_again", r"\bgo again\b"),
    ("intimidate", r"\bintimidate\b"),
    ("destroy", r"\bdestroy\b"),
    ("search", r"\bsearch\b"),
    ("discard", r"\bdiscard\b"),
    ("banish", r"\bbanish\b"),
    ("create_token", r"\bcreate a .+ token\b"),
    ("create_banished", r"\bcreate a .+ in your banished zone\b"),
    ("prevent_damage", r"\bprevent (?:the )?\d+ damage\b"),
    ("mark", r"\bmark them\b"),
    ("reload", r"\breload\b"),
    ("gains_quoted_ability", r'gains?\s+"|gains?\s+\u201c'),
    ("whenever", r"\bwhenever\b"),
    ("when_defends", r"\bwhen (?:this|it) defends\b"),
    ("when_leaves", r"\bwhen (?:this|it) leaves\b"),
    ("when_enters", r"\bwhen (?:this|it) enters\b"),
    ("when_deals_damage", r"\bwhen (?:this|it) deals\b"),
    ("when_put_into_graveyard", r"\bwhen (?:this|it) is put into a graveyard\b"),
    ("deal_damage", r"\bdeal \d+ (?:arcane )?damage\b"),
    ("play_from_banish", r"\bplay this from your banished zone\b"),
    ("target_gets", r"\btarget .+ gets\b"),
    ("for_each", r"\bfor each\b"),
    ("unless", r"\bunless\b"),
    ("may_play", r"\bmay play\b"),
    ("put_counter", r"\bput .+ counter\b"),
    ("amp", r"\bamp\b"),
    ("combo", r"\bcombo\b"),
    ("draw", r"\bdraw\b"),
    ("power_buff", r"\bgets \+\d+\s*(?:\{p\}|power)\b"),
    ("activated_ability", r"\b(?:once per turn\s+)?(?:action|instant)\s*[-—–]"),
    ("pitch_pay", r"\bpitch\b|\{r\}"),
    ("put_into_soul", r"\bput it into your soul\b|\bsoul\b"),
    ("equipment_mod", r"\bequipment\b|\{d\} counter"),
)

INTERACTION_HINTS = re.compile(
    r"\b(when |whenever |if this |if you |if an? |action --|instant --|"
    r"the next |go again|deal \d|draw |create |banish |destroy |search |"
    r"crush|surge|opt\b|transcend|contract|boost|clash|ward\b|fusion|"
    r"dominate|intimidate|blood debt|legendary|may play|unless|for each)\b",
    re.I,
)


def _classify(text: str) -> list[str]:
    clean = effects._clean(text).lower()
    found = [name for name, pat in PATTERN_RULES if re.search(pat, clean, re.I)]
    return found or ["other"]


def _unique_cards(raw: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for card in raw:
        key = (card.get("name"), card.get("pitch"))
        if key in seen:
            continue
        seen.add(key)
        out.append(card)
    return out


def scan() -> dict:
    cards = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
    unique = _unique_cards(cards)

    pattern_index: dict[str, dict] = {}
    keyword_gaps: Counter[str] = Counter()
    card_rows: list[dict] = []

    stats = Counter()
    parsed_unimpl_patterns: Counter[tuple[str, str]] = Counter()

    for card in unique:
        text = card.get("text") or ""
        keywords = list(card.get("keywords") or [])
        if not text.strip() or not INTERACTION_HINTS.search(text):
            stats["no_interaction_text"] += 1
            continue

        stats["has_interaction_text"] += 1
        categories = _classify(text)

        parsed = effects.parse_all_interactions(
            text,
            keywords=keywords,
            card_types=tuple(card.get("card_types") or ()),
        )
        trigs = (*parsed["triggers"], *parsed["keyword_triggers"])
        abilities = parsed["abilities"]
        mods = parsed["modifiers"]
        play_costs = parsed.get("play_costs") or ()
        action_damage, action_arcane = parsed["action_damage"]

        impl: list[dict] = []
        unimpl: list[dict] = []

        if action_damage > 0:
            kind = "arcane_damage" if action_arcane else "damage"
            impl.append({"source": "action", "kind": kind, "raw": f"deal {action_damage} damage"})

        for pc in play_costs:
            row = {"source": "play_cost", "kind": pc.kind, "raw": pc.raw}
            if pc.implemented:
                impl.append(row)
            else:
                unimpl.append(row)
                for cat in _classify(pc.raw or ""):
                    parsed_unimpl_patterns[(cat, (pc.raw or "")[:120])] += 1

        for t in trigs:
            row = {"source": "trigger", "when": t.when, "kind": t.effect.kind, "raw": t.effect.raw}
            if t.effect.implemented:
                impl.append(row)
            else:
                unimpl.append(row)
                for cat in _classify(t.effect.raw or t.raw):
                    parsed_unimpl_patterns[(cat, (t.effect.raw or t.raw)[:120])] += 1

        for a in abilities:
            row = {"source": "activated", "kind": a.effect.kind, "raw": a.effect.raw, "ability": a.raw[:120]}
            if a.implemented:
                impl.append(row)
            else:
                unimpl.append(row)
                for cat in _classify(a.effect.raw or a.raw):
                    parsed_unimpl_patterns[(cat, (a.effect.raw or a.raw)[:120])] += 1

        has_any_parse = bool(trigs or abilities or mods or play_costs)
        if unimpl and not impl and not mods:
            status = "unparsed" if not has_any_parse else "unimplemented_only"
        elif unimpl:
            status = "partial"
        elif has_any_parse or mods:
            status = "implemented"
        else:
            status = "unparsed"

        stats[status] += 1

        for kw in keywords:
            if kw not in IMPLEMENTED_KEYWORDS and kw not in {
                "temper",
                "arcane_barrier",
                "galvanize",
                "modular",
            }:
                keyword_gaps[kw] += 1

        if status == "implemented":
            continue

        for cat in categories:
            bucket = pattern_index.setdefault(
                cat,
                {"category": cat, "card_count": 0, "example_cards": [], "example_text": []},
            )
            bucket["card_count"] += 1
            name = card.get("name", "")
            if len(bucket["example_cards"]) < 8 and name not in bucket["example_cards"]:
                bucket["example_cards"].append(name)
            if len(bucket["example_text"]) < 3:
                bucket["example_text"].append(text[:200].replace("{br}", " "))

        card_rows.append(
            {
                "id": card.get("id"),
                "name": card.get("name"),
                "pitch": card.get("pitch"),
                "class": card.get("class"),
                "keywords": keywords,
                "status": status,
                "categories": categories,
                "unimplemented": unimpl,
                "implemented": impl,
                "text": text,
            }
        )

    parsed_patterns = [
        {
            "category": cat,
            "effect_text": raw,
            "occurrences": count,
        }
        for (cat, raw), count in parsed_unimpl_patterns.most_common()
    ]

    pattern_summary = sorted(
        pattern_index.values(),
        key=lambda x: (-x["card_count"], x["category"]),
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "supported_effects": sorted(effects.SUPPORTED_EFFECTS),
        "implemented_keywords": sorted(IMPLEMENTED_KEYWORDS),
        "summary": {
            "total_card_records": len(cards),
            "unique_name_pitch_cards": len(unique),
            **dict(stats),
            "distinct_unimplemented_categories": len(pattern_summary),
            "distinct_parsed_unimplemented_patterns": len(parsed_patterns),
            "cards_with_gaps": len(card_rows),
        },
        "unimplemented_categories": pattern_summary,
        "parsed_unimplemented_patterns": parsed_patterns,
        "unimplemented_keywords": [
            {"keyword": kw, "card_count": n}
            for kw, n in keyword_gaps.most_common()
        ],
        "cards": card_rows,
    }


def _write_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        "# Unimplemented Card Effects",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "This report lists Flesh and Blood card interactions **not faithfully modeled**",
        "by `flesh_and_blood_rlbridge` as of the scan date. Cards with at least one",
        "implemented effect on the same card are marked **partial**.",
        "",
        "## Summary",
        "",
        f"| Metric | Count |",
        f"|--------|------:|",
        f"| Unique cards (name+pitch) | {s['unique_name_pitch_cards']} |",
        f"| Cards with interaction text | {s.get('has_interaction_text', 0)} |",
        f"| Fully implemented | {s.get('implemented', 0)} |",
        f"| Partial (some gaps) | {s.get('partial', 0)} |",
        f"| Parsed but nothing works | {s.get('unimplemented_only', 0)} |",
        f"| Nothing parsed | {s.get('unparsed', 0)} |",
        f"| Distinct gap categories | {s['distinct_unimplemented_categories']} |",
        "",
        "## Supported effect kinds (implemented)",
        "",
        ", ".join(f"`{k}`" for k in report["supported_effects"]),
        "",
        "## Unimplemented categories (by rules-text pattern)",
        "",
        "Cards are counted in every category their text matches.",
        "",
        "| Category | Cards | Examples |",
        "|----------|------:|---------|",
    ]

    for bucket in report["unimplemented_categories"]:
        examples = ", ".join(bucket["example_cards"][:5])
        lines.append(f"| {bucket['category']} | {bucket['card_count']} | {examples} |")

    lines.extend(
        [
            "",
            "## Parsed but unimplemented effect clauses",
            "",
            "Text the parser recognizes as a trigger/ability but cannot resolve.",
            "",
            "| Category | Occurrences | Effect text (sample) |",
            "|----------|------------:|---------------------|",
        ]
    )
    for row in report["parsed_unimplemented_patterns"][:80]:
        text = row["effect_text"].replace("|", "\\|")[:90]
        lines.append(f"| {row['category']} | {row['occurrences']} | {text} |")
    if len(report["parsed_unimplemented_patterns"]) > 80:
        lines.append(f"| … | … | *{len(report['parsed_unimplemented_patterns']) - 80} more in JSON* |")

    lines.extend(
        [
            "",
            "## Keywords on cards without full mechanic support",
            "",
            "| Keyword | Cards |",
            "|---------|------:|",
        ]
    )
    for row in report["unimplemented_keywords"][:40]:
        lines.append(f"| {row['keyword']} | {row['card_count']} |")

    lines.extend(
        [
            "",
            "## Full per-card detail",
            "",
            "See `unimplemented_effects.json` → `cards` for the complete list.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    report = scan()
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(_write_markdown(report), encoding="utf-8")
    s = report["summary"]
    print(f"Wrote {OUT_JSON} ({s['cards_with_gaps']} cards with gaps)")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
