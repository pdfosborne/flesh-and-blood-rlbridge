#!/usr/bin/env python3
"""Fetch Fabrary Talishar meta matchup matrices into a local lookup table.

Fabrary serves pre-aggregated hero-vs-hero stats from their CDN (not the
meta-results HTML page).  See ``fabrary_meta.py`` for URL mapping and lookup.

Usage
-----
    python scripts/deck/fetch_fabrary_meta.py

    python scripts/deck/fetch_fabrary_meta.py \\
        --format silver_age classic_constructed \\
        --period last-30-days \\
        --out src/flesh_and_blood_rlbridge/card_db/fabrary_meta_matchups.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from flesh_and_blood_rlbridge.card_db.fabrary_meta import (  # noqa: E402
    DEFAULT_FORMATS,
    DEFAULT_GAMES,
    DEFAULT_META_PATH,
    DEFAULT_PERIOD,
    DEFAULT_USER_AGENT,
    build_lookup,
    write_lookup,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Fabrary meta matchup data into fabrary_meta_matchups.json",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        choices=["silver_age", "classic_constructed"],
        default=list(DEFAULT_FORMATS),
        help="Formats to fetch (default: both)",
    )
    parser.add_argument(
        "--games",
        nargs="+",
        choices=["all", "competitive", "standard"],
        default=list(DEFAULT_GAMES),
        help="Talishar queue filters (default: all three)",
    )
    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
        help="Time window slug, e.g. last-30-days, last-7-days, 2026-06",
    )
    parser.add_argument(
        "--min-plays",
        type=int,
        default=0,
        help="Drop matchups with fewer than N games (default: keep all)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_META_PATH,
        help="Output lookup JSON path",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="HTTP User-Agent (required by Fabrary CDN)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout per dataset in seconds",
    )
    args = parser.parse_args(argv)

    payload = build_lookup(
        formats=tuple(args.format),
        games_filters=tuple(args.games),
        period=args.period,
        user_agent=args.user_agent,
        min_plays=max(0, int(args.min_plays)),
        timeout=float(args.timeout),
    )
    out_path = args.out.expanduser().resolve()
    write_lookup(out_path, payload)
    n_datasets = len(payload.get("datasets") or {})
    n_matchups = sum(
        len(ds.get("matchups") or {})
        for ds in (payload.get("datasets") or {}).values()
        if isinstance(ds, dict)
    )
    print(f"Wrote {out_path} ({n_datasets} datasets, {n_matchups} matchup rows total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
