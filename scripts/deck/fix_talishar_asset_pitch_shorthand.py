#!/usr/bin/env python3
"""Rewrite Talishar Assets deck lines that stack >2 copies on one pitch id.

Talishar precon files sometimes list ``warriors_valor_red`` six times instead of
two of each pitch. This script rewrites those asset ``.txt`` files in place so
each pitch variant appears at most twice.

Run once after updating precons, or when adding new shorthand asset decks:

    python scripts/deck/fix_talishar_asset_pitch_shorthand.py Talishar/Assets/*SAGE*.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from fab_tui.deck_cards import assign_pitch_variants  # noqa: E402

_CARDS_PATH = _REPO / "src" / "flesh_and_blood_rlbridge" / "card_db" / "cards.json"
_PITCH_SUFFIX = re.compile(r"_(red|yellow|blue|purple)$", re.IGNORECASE)
_TALISHAR_ID = re.compile(r"^[a-z0-9_]+$")


def _load_name_map() -> dict[str, list[dict]]:
    records = json.loads(_CARDS_PATH.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = {}
    for rec in records:
        cid = str(rec.get("id") or "").strip()
        if not cid or not _TALISHAR_ID.match(cid) or not _PITCH_SUFFIX.search(cid):
            continue
        pitch = rec.get("pitch")
        if pitch not in (1, 2, 3):
            continue
        grouped.setdefault(str(rec.get("name") or "").lower(), []).append(rec)
    for rows in grouped.values():
        rows.sort(key=lambda rec: int(rec.get("pitch") or 99))
    return grouped


def _resolve_card_id(raw: str, resolver) -> str:
    token = str(raw or "").strip()
    if not token:
        return token
    if resolver is not None:
        resolved = resolver.resolve(token)
        if resolved:
            return resolved
    return token


def _expand_deck_counts(counts: Counter[str], name_map: dict[str, list[dict]]) -> list[str]:
    by_name: dict[str, Counter[str]] = {}
    passthrough: Counter[str] = Counter()

    for card_id, qty in counts.items():
        if qty <= 0:
            continue
        match = _PITCH_SUFFIX.search(card_id)
        if not match:
            passthrough[card_id] += qty
            continue
        base = _PITCH_SUFFIX.sub("", card_id)
        name = base.replace("_", " ").lower()
        for deck_name, rows in name_map.items():
            if rows and str(rows[0]["id"]).startswith(base):
                name = deck_name
                break
        by_name.setdefault(name, Counter())[card_id] += qty

    out: list[str] = []
    for card_id, qty in sorted(passthrough.items()):
        out.extend([card_id] * qty)

    for name, pitch_counts in by_name.items():
        if all(qty <= 2 for qty in pitch_counts.values()):
            for card_id, qty in sorted(pitch_counts.items()):
                out.extend([card_id] * qty)
            continue

        candidates = name_map.get(name, [])
        if not candidates:
            for card_id, qty in sorted(pitch_counts.items()):
                out.extend([card_id] * qty)
            continue

        total = sum(pitch_counts.values())
        out.extend(assign_pitch_variants(candidates, total))

    return out


def fix_asset_file(path: Path, *, dry_run: bool = False) -> bool:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if not lines:
        return False

    resolver = None
    try:
        from flesh_and_blood_rlbridge.card_db.talishar_card_ids import (  # noqa: PLC0415
            TalisharCardIdResolver,
        )

        php = _REPO / "Talishar" / "GeneratedCode" / "GeneratedCardDictionaries.php"
        resolver = TalisharCardIdResolver(talishar_php_path=php, cards_path=_CARDS_PATH)
    except ImportError:
        resolver = None

    name_map = _load_name_map()
    setup = [_resolve_card_id(token, resolver) for token in lines[0].split()]
    raw_counts = Counter(
        _resolve_card_id(token, resolver)
        for token in " ".join(lines[1:]).split()
        if token.strip()
    )
    expanded = _expand_deck_counts(raw_counts, name_map)
    if Counter(expanded) == raw_counts:
        return False

    deck_line = " ".join(expanded)
    new_text = f"{' '.join(setup)}\n{deck_line}\n"
    if dry_run:
        print(f"Would update {path.name}: {dict(raw_counts)} -> {dict(Counter(expanded))}")
    else:
        path.write_text(new_text, encoding="utf-8")
        print(f"Updated {path.name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="Asset .txt files to rewrite")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    changed = 0
    for raw in args.files:
        path = Path(raw)
        if not path.is_file():
            print(f"Skip missing file: {path}", file=sys.stderr)
            continue
        if fix_asset_file(path, dry_run=args.dry_run):
            changed += 1
    print(f"Done — {changed} file(s) {'would change' if args.dry_run else 'updated'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
