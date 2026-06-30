"""Fabrary Talishar meta matchup lookup (hero-level win rates from fabrary.net)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

CARD_DB_DIR = Path(__file__).resolve().parent
DEFAULT_META_PATH = CARD_DB_DIR / "fabrary_meta_matchups.json"
DEFAULT_DECKS_PATH = CARD_DB_DIR / "fabrary_decks.json"
HEROES_PATH = CARD_DB_DIR / "heroes.json"

CDN_BASE = "https://content.fabrary.net/results"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; flesh-and-blood-rlbridge/1.0; +fabrary-meta-fetch)"
)
DEFAULT_PERIOD = "last-30-days"
DEFAULT_GAMES = ("all", "competitive", "standard")
DEFAULT_FORMATS = ("silver_age", "classic_constructed")

# Fabrary format + queue → CDN filename prefix (before -{period}.json)
FORMAT_GAMES_SLUG: dict[tuple[str, str], str] = {
    ("silver_age", "competitive"): "silver-age-competitive",
    ("silver_age", "standard"): "silver-age",
    ("silver_age", "all"): "all-silver-age",
    ("classic_constructed", "competitive"): "classic-constructed-competitive",
    ("classic_constructed", "standard"): "classic-constructed",
    ("classic_constructed", "all"): "all-classic-constructed",
}

# Optional overrides when hero_id slug does not match Fabrary's identifier.
HERO_ID_TO_FABRARY_SLUG: dict[str, str] = {}
FABRARY_SLUG_TO_HERO_ID: dict[str, str] = {}


def _card_db_path(path: Path | None) -> Path:
    return path if path is not None else DEFAULT_META_PATH


def hero_id_to_fabrary_slug(hero_id: str) -> str:
    """Map rlbridge ``hero_*`` id to Fabrary CDN hero slug."""
    token = (hero_id or "").strip()
    if not token:
        return ""
    if token in HERO_ID_TO_FABRARY_SLUG:
        return HERO_ID_TO_FABRARY_SLUG[token]
    return token.removeprefix("hero_").replace("_", "-")


def fabrary_slug_to_hero_id(slug: str, *, heroes_path: Path | None = None) -> str:
    """Best-effort map Fabrary slug → ``hero_*`` id using heroes.json."""
    slug = (slug or "").strip()
    if not slug:
        return ""
    if slug in FABRARY_SLUG_TO_HERO_ID:
        return FABRARY_SLUG_TO_HERO_ID[slug]
    path = heroes_path or HEROES_PATH
    if path.is_file():
        heroes = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(heroes, list):
            exact = f"hero_{slug.replace('-', '_')}"
            for entry in heroes:
                if not isinstance(entry, dict):
                    continue
                hid = str(entry.get("id") or "")
                if hid == exact:
                    return hid
            # Prefer shortest matching hero id (base hero over variants).
            candidates = [
                str(entry.get("id") or "")
                for entry in heroes
                if isinstance(entry, dict)
                and str(entry.get("id") or "").removeprefix("hero_").replace("_", "-") == slug
            ]
            if candidates:
                return sorted(candidates, key=len)[0]
    return f"hero_{slug.replace('-', '_')}"


def matchup_key(hero_a_slug: str, hero_b_slug: str) -> str:
    """Sorted Fabrary slug pair key."""
    a = (hero_a_slug or "").strip()
    b = (hero_b_slug or "").strip()
    if a > b:
        a, b = b, a
    return f"{a}|{b}"


def dataset_key(format_name: str, games: str, period: str) -> str:
    return f"{format_name}|{games}|{period}"


def build_cdn_url(
    format_name: str,
    games: str,
    period: str,
) -> str:
    """Return full CDN URL for a Fabrary meta dataset."""
    prefix = FORMAT_GAMES_SLUG.get((format_name, games))
    if not prefix:
        raise ValueError(f"Unknown format/games pair: {format_name!r}, {games!r}")
    period = (period or DEFAULT_PERIOD).strip()
    return f"{CDN_BASE}/{prefix}-{period}.json"


def _win_rates(wins: int, plays: int) -> tuple[float, float]:
    if plays <= 0:
        return 0.0, 0.0
    rate_a = wins / plays
    return rate_a, 1.0 - rate_a


def parse_hero_results(
    raw: dict[str, Any],
    *,
    min_plays: int = 0,
) -> dict[str, dict[str, Any]]:
    """Normalize Fabrary ``heroResults`` into sorted slug-keyed matchups."""
    hero_results = raw.get("heroResults")
    if not isinstance(hero_results, list):
        return {}

    matchups: dict[str, dict[str, Any]] = {}
    for hero_entry in hero_results:
        if not isinstance(hero_entry, dict):
            continue
        hero_slug = str(hero_entry.get("heroIdentifier") or "").strip()
        if not hero_slug:
            continue
        results = hero_entry.get("results")
        if not isinstance(results, list):
            continue
        for row in results:
            if not isinstance(row, dict):
                continue
            opp_slug = str(row.get("opposingHeroIdentifier") or "").strip()
            if not opp_slug or opp_slug == hero_slug:
                continue
            plays = int(row.get("plays") or 0)
            if plays < min_plays:
                continue
            wins = int(row.get("wins") or 0)
            key = matchup_key(hero_slug, opp_slug)
            if key in matchups:
                continue
            # Orient wins to alphabetically first hero in key.
            first, second = key.split("|", 1)
            wins_a = wins if hero_slug == first else (plays - wins)
            rate_a, rate_b = _win_rates(wins_a, plays)
            matchups[key] = {
                "hero_a": first,
                "hero_b": second,
                "hero_a_id": fabrary_slug_to_hero_id(first),
                "hero_b_id": fabrary_slug_to_hero_id(second),
                "plays": plays,
                "wins_a": wins_a,
                "win_rate_a": rate_a,
                "win_rate_b": rate_b,
            }
    return matchups


def fetch_cdn_json(
    url: str,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Fetch and parse a Fabrary CDN JSON file."""
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} fetching {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error fetching {url}: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected JSON type from {url}")
    return payload


def build_dataset(
    format_name: str,
    games: str,
    period: str,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    min_plays: int = 0,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Fetch one Fabrary meta dataset and return a lookup entry."""
    source_url = build_cdn_url(format_name, games, period)
    raw = fetch_cdn_json(source_url, user_agent=user_agent, timeout=timeout)
    return {
        "format": format_name,
        "games": games,
        "period": period,
        "period_label": str(raw.get("period") or period),
        "queue_label": str(raw.get("queue") or ""),
        "source_url": source_url,
        "timestamp": str(raw.get("timestamp") or ""),
        "matchups": parse_hero_results(raw, min_plays=min_plays),
    }


def build_lookup(
    *,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    games_filters: tuple[str, ...] = DEFAULT_GAMES,
    period: str = DEFAULT_PERIOD,
    user_agent: str = DEFAULT_USER_AGENT,
    min_plays: int = 0,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Fetch all requested datasets and assemble the lookup document."""
    datasets: dict[str, dict[str, Any]] = {}
    for format_name in formats:
        for games in games_filters:
            key = dataset_key(format_name, games, period)
            datasets[key] = build_dataset(
                format_name,
                games,
                period,
                user_agent=user_agent,
                min_plays=min_plays,
                timeout=timeout,
            )
    return {
        "version": 1,
        "source": "https://fabrary.net/meta-results",
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "datasets": datasets,
    }


def write_lookup(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_fabrary_meta(path: Path | None = None) -> dict[str, Any]:
    """Load the committed Fabrary meta lookup table."""
    meta_path = _card_db_path(path)
    if not meta_path.is_file():
        return {"version": 1, "datasets": {}}
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {"version": 1, "datasets": {}}


def _load_deck_hero_ids(
    decks_path: Path | None = None,
) -> dict[str, str]:
    path = decks_path or DEFAULT_DECKS_PATH
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    decks = raw.get("decks") if isinstance(raw, dict) else raw
    if not isinstance(decks, list):
        return {}
    result: dict[str, str] = {}
    for entry in decks:
        if not isinstance(entry, dict):
            continue
        deck_id = str(entry.get("id") or "").strip()
        hero_id = str(entry.get("hero_id") or "").strip()
        if deck_id and hero_id:
            result[deck_id] = hero_id
    return result


_deck_hero_cache: dict[str, dict[str, str]] = {}


def deck_hero_id(deck_id: str, *, decks_path: Path | None = None) -> str:
    cache_key = str(decks_path or DEFAULT_DECKS_PATH)
    if cache_key not in _deck_hero_cache:
        _deck_hero_cache[cache_key] = _load_deck_hero_ids(decks_path)
    return _deck_hero_cache[cache_key].get((deck_id or "").strip(), "")


def lookup_hero_matchup(
    hero_a_id: str,
    hero_b_id: str,
    *,
    format_name: str,
    games: str = "competitive",
    period: str = DEFAULT_PERIOD,
    meta: dict[str, Any] | None = None,
    meta_path: Path | None = None,
) -> Optional[dict[str, Any]]:
    """Return Fabrary reference stats for a hero pair from P1's perspective."""
    slug_a = hero_id_to_fabrary_slug(hero_a_id)
    slug_b = hero_id_to_fabrary_slug(hero_b_id)
    if not slug_a or not slug_b:
        return None
    lookup = meta if meta is not None else load_fabrary_meta(meta_path)
    datasets = lookup.get("datasets")
    if not isinstance(datasets, dict):
        return None
    ds_key = dataset_key(format_name, games, period)
    dataset = datasets.get(ds_key)
    if not isinstance(dataset, dict):
        return None
    matchups = dataset.get("matchups")
    if not isinstance(matchups, dict):
        return None
    key = matchup_key(slug_a, slug_b)
    row = matchups.get(key)
    if not isinstance(row, dict):
        return None
    first = key.split("|", 1)[0]
    if slug_a == first:
        p1_win_rate = float(row.get("win_rate_a") or 0.0)
        p2_win_rate = float(row.get("win_rate_b") or 0.0)
    else:
        p2_win_rate = float(row.get("win_rate_a") or 0.0)
        p1_win_rate = float(row.get("win_rate_b") or 0.0)
    return {
        "hero_a": slug_a,
        "hero_b": slug_b,
        "hero_a_id": hero_a_id,
        "hero_b_id": hero_b_id,
        "plays": int(row.get("plays") or 0),
        "p1_win_rate": p1_win_rate,
        "p2_win_rate": p2_win_rate,
        "dataset_key": ds_key,
        "period_label": str(dataset.get("period_label") or period),
        "queue_label": str(dataset.get("queue_label") or ""),
    }


def lookup_deck_matchup(
    p1_deck_id: str,
    p2_deck_id: str,
    *,
    format_name: str,
    games: str = "competitive",
    period: str = DEFAULT_PERIOD,
    meta: dict[str, Any] | None = None,
    meta_path: Path | None = None,
    decks_path: Path | None = None,
) -> Optional[dict[str, Any]]:
    """Resolve deck ids to heroes and look up Fabrary meta."""
    hero_a = deck_hero_id(p1_deck_id, decks_path=decks_path)
    hero_b = deck_hero_id(p2_deck_id, decks_path=decks_path)
    if not hero_a or not hero_b:
        return None
    result = lookup_hero_matchup(
        hero_a,
        hero_b,
        format_name=format_name,
        games=games,
        period=period,
        meta=meta,
        meta_path=meta_path,
    )
    if result is None:
        return None
    result["p1_deck"] = p1_deck_id
    result["p2_deck"] = p2_deck_id
    return result


def lookup_matchup_dir(
    matchup_dir: Path,
    *,
    format_name: str,
    games: str = "competitive",
    period: str = DEFAULT_PERIOD,
    meta: dict[str, Any] | None = None,
    meta_path: Path | None = None,
    decks_path: Path | None = None,
) -> Optional[dict[str, Any]]:
    """Read ``matchup_label.json`` and look up Fabrary meta for that deck pair."""
    label_path = Path(matchup_dir) / "matchup_label.json"
    if not label_path.is_file():
        return None
    raw = json.loads(label_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    p1_deck = str(raw.get("p1_deck") or "").strip()
    p2_deck = str(raw.get("p2_deck") or "").strip()
    if not p1_deck or not p2_deck:
        return None
    return lookup_deck_matchup(
        p1_deck,
        p2_deck,
        format_name=format_name,
        games=games,
        period=period,
        meta=meta,
        meta_path=meta_path,
        decks_path=decks_path,
    )


def lookup_all_queues_for_matchup_dir(
    matchup_dir: Path,
    *,
    format_name: str,
    period: str = DEFAULT_PERIOD,
    meta: dict[str, Any] | None = None,
    meta_path: Path | None = None,
    decks_path: Path | None = None,
) -> dict[str, Optional[dict[str, Any]]]:
    """Look up competitive, all, and standard queue references."""
    return {
        games: lookup_matchup_dir(
            matchup_dir,
            format_name=format_name,
            games=games,
            period=period,
            meta=meta,
            meta_path=meta_path,
            decks_path=decks_path,
        )
        for games in DEFAULT_GAMES
    }
