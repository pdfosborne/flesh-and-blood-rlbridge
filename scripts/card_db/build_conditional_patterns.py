#!/usr/bin/env python3
"""Generate conditional_patterns.json from cards.json rules text."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from flesh_and_blood_rlbridge.card_conditionals import classify_card_patterns  # noqa: E402

_CARDS_PATH = _REPO_ROOT / "src" / "flesh_and_blood_rlbridge" / "card_db" / "cards.json"
_OUT_PATH = _CARDS_PATH.parent / "conditional_patterns.json"


def main() -> None:
    records = json.loads(_CARDS_PATH.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for rec in records if isinstance(records, list) else []:
        if not isinstance(rec, dict):
            continue
        cid = str(rec.get("id", "") or "").strip().lower()
        if not cid:
            continue
        out[cid] = classify_card_patterns(
            cid,
            str(rec.get("text", "") or ""),
            list(rec.get("keywords") or []),
        )
    _OUT_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(out)} entries to {_OUT_PATH}")


if __name__ == "__main__":
    main()
