#!/usr/bin/env python3
"""Audit silver_age / sage decks used by unified random matchup training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAB_PATH = REPO / "src" / "flesh_and_blood_rlbridge" / "card_db" / "fabrary_decks.json"
ASSETS_DIRS = [
    REPO / "Talishar" / "Assets",
    REPO / "assets" / "talishar_decks",
]


def analyze_file(path: Path) -> dict | None:
    if not path.is_file():
        return None
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if not lines:
        return {"exists": True, "header": "", "cards": 0, "header_tokens": 0}
    header = lines[0]
    cards = " ".join(lines[1:]).split() if len(lines) > 1 else []
    return {
        "exists": True,
        "header": header,
        "header_tokens": len(header.split()),
        "cards": len(cards),
        "lines": len(lines),
    }


def json_card_count(entry: dict) -> int:
    if entry.get("card_ids"):
        return sum(int(c.get("count", 1)) for c in entry["card_ids"])
    return sum(int(c.get("count", 1)) for c in entry.get("cards", []))


def main() -> int:
    data = json.loads(FAB_PATH.read_text(encoding="utf-8"))
    decks = [d for d in data["decks"] if d.get("format") == "silver_age"]
    print(f"Silver age decks in fabrary_decks.json: {len(decks)}\n")

    issues: list[tuple[str, str, list[str], dict]] = []
    for entry in sorted(decks, key=lambda x: x["id"]):
        deck_id = entry["id"]
        json_cards = json_card_count(entry)
        found = False
        for assets_dir in ASSETS_DIRS:
            info = analyze_file(assets_dir / f"{deck_id}.txt")
            if info is None:
                continue
            found = True
            problems: list[str] = []
            if info["header_tokens"] <= 1:
                problems.append(
                    f"header has only {info['header_tokens']} token(s): {info['header']!r}"
                )
            if info["cards"] < 40:
                problems.append(f"only {info['cards']} cards (expected 40)")
            if json_cards and info["cards"] != json_cards:
                problems.append(
                    f"json resolves to {json_cards} cards but file has {info['cards']}"
                )
            if problems:
                issues.append((deck_id, assets_dir.name, problems, info))
            break
        if not found:
            issues.append(
                (deck_id, "MISSING", ["no asset file found"], {"json_cards": json_cards})
            )

    if issues:
        print("ISSUES FOUND:")
        for deck_id, loc, problems, info in issues:
            print(f"  {deck_id} [{loc}]")
            for problem in problems:
                print(f"    - {problem}")
            if "header" in info:
                print(f"    header: {info['header'][:140]}")
        print()
    else:
        print("All fabrary silver_age decks look OK in on-disk asset files.\n")

    print("=== All fab_*sage* asset files ===")
    for assets_dir in ASSETS_DIRS:
        if not assets_dir.is_dir():
            print(f"{assets_dir}: missing")
            continue
        files = sorted(assets_dir.glob("fab_*sage*.txt"))
        bad: list[tuple[str, dict]] = []
        for path in files:
            info = analyze_file(path)
            assert info is not None
            if info["header_tokens"] <= 1 or info["cards"] < 40:
                bad.append((path.name, info))
        print(f"{assets_dir}: {len(files)} sage files, {len(bad)} problematic")
        for name, info in bad:
            print(
                f"  {name}: header_tokens={info['header_tokens']}, "
                f"cards={info['cards']}, header={info['header']!r}"
            )

    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
