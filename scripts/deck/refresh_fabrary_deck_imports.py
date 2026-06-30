#!/usr/bin/env python3
"""Re-fetch fabrary decks that already exist in fabrary_decks.json."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_fabrary_deck import (  # noqa: E402
    _extract_slug,
    append_to_fabrary_decks_json,
    fetch_raw_fabrary,
    parse_fabrary_deck,
    resolve_api_key,
)

FAB_PATH = REPO / "src" / "flesh_and_blood_rlbridge" / "card_db" / "fabrary_decks.json"

REFRESH = [
    ("fab_ira_crimson_haze_sage_aggro", "01KR40W4Z2ZS9EQPT6VT6CDSPE"),
    ("fab_oscilio_omn", "01KVY39BSCSNC9A0ZAGMQBCW6N"),
]


def main() -> int:
    api_key = resolve_api_key(None)
    failed = 0
    for deck_id, slug in REFRESH:
        print(f"Refreshing {deck_id} from {slug} ...", file=sys.stderr)
        try:
            raw = fetch_raw_fabrary(slug, api_key)
        except RuntimeError as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            failed += 1
            continue
        deck_info = parse_fabrary_deck(raw)
        deck_info["deck_id"] = slug
        append_to_fabrary_decks_json(
            deck_info,
            slug,
            deck_id,
            FAB_PATH,
            replace_existing=True,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
