#!/usr/bin/env python3
"""Fetch FaBrary decks and append them to fabrary_decks.json for unified training.

By default reads ``META.unified_random_matchups.custom_deck_links`` from
``runtime_defaults.py``. Pass extra slugs or full fabrary.net URLs on the CLI.

Usage::

    python scripts/deck/add_custom_decks_to_pool.py
    python scripts/deck/add_custom_decks_to_pool.py 01KR40W4Z2ZS9EQPT6VT6CDSPE
    python scripts/deck/add_custom_decks_to_pool.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_fabrary_deck import (  # noqa: E402
    _extract_slug,
    append_to_fabrary_decks_json,
    fetch_raw_fabrary,
    parse_fabrary_deck,
    resolve_api_key,
)
from runtime_defaults import META  # noqa: E402

DEFAULT_FABRARY_DECKS = (
    REPO_ROOT / "src" / "flesh_and_blood_rlbridge" / "card_db" / "fabrary_decks.json"
)


def _slugify_deck_id(name: str, slug: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not token:
        token = f"imported_{slug[:12].lower()}"
    deck_id = f"fab_{token}"
    if len(deck_id) > 48:
        deck_id = f"fab_imported_{slug[:12].lower()}"
    return deck_id


def append_deck_link(
    link: str,
    *,
    json_path: Path,
    api_key: str | None,
    deck_id: str | None,
    dry_run: bool,
) -> bool:
    slug = _extract_slug(link.strip())
    if not slug:
        print(f"  Skipping empty link: {link!r}", file=sys.stderr)
        return False

    print(f"Fetching {slug} ...", file=sys.stderr)
    if dry_run:
        print(f"  [dry-run] would append deck from slug {slug}", file=sys.stderr)
        return True

    try:
        raw = fetch_raw_fabrary(slug, api_key)
    except RuntimeError as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        return False

    deck_info = parse_fabrary_deck(raw)
    deck_info["deck_id"] = slug
    resolved_id = deck_id or _slugify_deck_id(str(deck_info.get("name", slug)), slug)
    append_to_fabrary_decks_json(deck_info, slug, resolved_id, json_path)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch FaBrary decks listed in runtime_defaults custom_deck_links "
            "and append them to fabrary_decks.json."
        ),
    )
    parser.add_argument(
        "links",
        nargs="*",
        help="Extra FaBrary slugs or URLs (merged with runtime_defaults links)",
    )
    parser.add_argument(
        "--fabrary-decks",
        type=Path,
        default=DEFAULT_FABRARY_DECKS,
        help="fabrary_decks.json path to update",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="FaBrary API key (overrides FABRARY_API_KEY / APIKeys.php)",
    )
    parser.add_argument(
        "--deck-id",
        default=None,
        help="Force deck id for a single link (only valid with one link)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print links that would be fetched without calling the API",
    )
    args = parser.parse_args(argv)

    runtime_links = list(META.unified_random_matchups.custom_deck_links)
    all_links = runtime_links + [link for link in args.links if link not in runtime_links]
    if not all_links:
        print(
            "No deck links configured. Add slugs to "
            "META.unified_random_matchups.custom_deck_links in runtime_defaults.py "
            "or pass links on the command line.",
            file=sys.stderr,
        )
        return 1

    if args.deck_id and len(all_links) != 1:
        print("--deck-id requires exactly one link", file=sys.stderr)
        return 1

    api_key = resolve_api_key(args.api_key)
    if not api_key and not args.dry_run:
        print(
            "  INFO: No FaBrary API key found; unauthenticated fetch may fail.",
            file=sys.stderr,
        )

    ok = 0
    failed = 0
    for link in all_links:
        if append_deck_link(
            link,
            json_path=args.fabrary_decks,
            api_key=api_key,
            deck_id=args.deck_id,
            dry_run=args.dry_run,
        ):
            ok += 1
        else:
            failed += 1

    print(f"Done: {ok} succeeded, {failed} failed.", file=sys.stderr)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
