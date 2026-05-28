"""
Update local cards.json from the official FAB Card Vault API.

Standalone usage:

    python update_cards_db_from_fabtcg.py
    python update_cards_db_from_fabtcg.py --dry-run
    python update_cards_db_from_fabtcg.py --legality-scope decks
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_CARDS_PATH = Path(__file__).with_name("cards.json")
DEFAULT_DECKS_PATH = Path(__file__).with_name("fabrary_decks.json")
API_FIND_CARDS_URL = "https://api.cardvault.fabtcg.com/api/gem/v1/find_cards/find/"
API_CARD_ID_URL_TMPL = "https://api.cardvault.fabtcg.com/carddb/api/v1/card_id/{card_id}/"


class _ProgressTracker:
    def __init__(self) -> None:
        self._start = time.monotonic()

    def _elapsed(self) -> str:
        sec = int(time.monotonic() - self._start)
        m, s = divmod(sec, 60)
        return f"{m:02d}:{s:02d}"

    def stage(self, label: str) -> None:
        print(f"[{self._elapsed()}] {label}")

    def tick(self, label: str, index: int, total: int, *, every: int = 25) -> None:
        if total <= 0:
            return
        if index == 1 or index == total or (index % max(1, every) == 0):
            pct = (index / total) * 100.0
            print(f"[{self._elapsed()}] {label}: {index}/{total} ({pct:.1f}%)")


def _card_name_key(name: str) -> str:
    text = " ".join(str(name or "").strip().split())
    text = text.replace("||", "//")
    text = text.replace(" // ", "//")
    return text.lower()


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _http_json_get(url: str) -> Any:
    last_exc: Optional[Exception] = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                break
        except urllib.error.URLError as exc:
            last_exc = exc
        except TimeoutError as exc:
            last_exc = exc
        if attempt < 3:
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"GET failed for {url}: {last_exc}")


def _http_json_post(url: str, payload: Any) -> Any:
    last_exc: Optional[Exception] = None
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                break
        except urllib.error.URLError as exc:
            last_exc = exc
        except TimeoutError as exc:
            last_exc = exc
        if attempt < 3:
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"POST failed for {url}: {last_exc}")


def _read_deck_card_names(decks_path: Path) -> set[str]:
    data = _load_json(decks_path)
    if not isinstance(data, dict):
        return set()
    names: set[str] = set()
    for deck in data.get("decks", []):
        if not isinstance(deck, dict):
            continue
        for card in deck.get("cards", []):
            if not isinstance(card, dict):
                continue
            name = str(card.get("name", "")).strip()
            if name:
                names.add(name)
    return names


def _card_match_key(card: dict[str, Any]) -> tuple[str, int]:
    return (_card_name_key(card.get("name", "")), int(card.get("pitch", 0) or 0))


def _pitch_to_color(pitch: int) -> str:
    return {1: "red", 2: "yellow", 3: "blue"}.get(int(pitch), "")


def _normalize_card_types_from_typebox(typebox: str) -> list[str]:
    text = str(typebox or "").lower()
    out: list[str] = []
    if "hero" in text:
        out.append("hero")
    if "attack reaction" in text:
        out.append("attack_reaction")
    if "defense reaction" in text:
        out.append("defense_reaction")
    if "action" in text and "attack" in text:
        out.append("attack_action")
    elif "action" in text:
        out.append("utility_action")
    if "equipment" in text or "weapon" in text:
        out.append("utility_item")
    return out


def _normalize_legality(card_legality: Any) -> dict[str, str]:
    if not isinstance(card_legality, dict):
        return {}
    mapped: dict[str, str] = {}
    key_map = {
        "classic constructed": "classic_constructed",
        "silver age": "silver_age",
        "blitz": "blitz",
        "living legend": "living_legend",
        "commoner": "commoner",
        "ultimate pit fight": "ultimate_pit_fight",
    }
    for raw_fmt, raw_rec in card_legality.items():
        fmt = key_map.get(str(raw_fmt).strip().lower())
        if not fmt:
            continue
        legality_val = ""
        if isinstance(raw_rec, dict):
            legality_val = str(raw_rec.get("legality", "")).strip().lower()
        if legality_val in {"legal", "banned", "suspended", "restricted"}:
            mapped[fmt] = legality_val
    return mapped


def _extract_set_code_from_print_id(print_id: str) -> str:
    text = str(print_id or "").strip()
    if not text:
        return "SIM"
    token = text.split("_", 1)[-1] if "_" in text else text
    match = re.match(r"([A-Z]{2,4})", token)
    return match.group(1) if match else "SIM"


def _build_local_record_from_card_detail(detail_entry: dict[str, Any]) -> Optional[dict[str, Any]]:
    cores = detail_entry.get("cores")
    if not isinstance(cores, list) or not cores:
        return None
    core = cores[0] if isinstance(cores[0], dict) else {}

    card_prints = detail_entry.get("card_prints")
    card_print = card_prints[0] if isinstance(card_prints, list) and card_prints and isinstance(card_prints[0], dict) else {}
    rarity_obj = card_print.get("rarity", {}) if isinstance(card_print, dict) else {}
    rarity = rarity_obj.get("name") if isinstance(rarity_obj, dict) else rarity_obj

    core_classes = core.get("core_classes")
    core_talents = core.get("core_talents")
    card_class = "Generic"
    talent = None
    if isinstance(core_classes, list) and core_classes:
        c0 = core_classes[0]
        if isinstance(c0, dict):
            card_class = str(c0.get("name", "Generic"))
    if isinstance(core_talents, list) and core_talents:
        t0 = core_talents[0]
        if isinstance(t0, dict):
            talent = str(t0.get("name", "")).strip() or None

    name = str(core.get("name", detail_entry.get("card_id", ""))).strip()
    pitch = _as_int(core.get("pitch_value", core.get("pitch", 0)), default=0)
    cost = _as_int(core.get("cost_value", core.get("cost", 0)), default=0)
    power = _as_int(core.get("power_value", core.get("power", 0)), default=0)
    defense = _as_int(core.get("defense_value", core.get("defense", 0)), default=0)
    type_line = str(core.get("typebox", "")).strip()
    text = str(core.get("textbox", "")).strip()
    card_id = str(detail_entry.get("card_id", "")).strip()
    if not card_id or not name:
        return None

    print_id = ""
    if isinstance(card_print, dict):
        print_id = str(card_print.get("print_id", "")).strip()

    return {
        "id": card_id,
        "name": name,
        "pitch": pitch,
        "cost": cost,
        "power": power,
        "defense": defense,
        "type_line": type_line,
        "card_types": _normalize_card_types_from_typebox(type_line),
        "class": card_class,
        "talent": talent,
        "rarity": str(rarity or "Common"),
        "set": _extract_set_code_from_print_id(print_id),
        "keywords": [],
        "text": text,
        "legality": _normalize_legality(detail_entry.get("card_legality")),
    }


def _batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _fetch_card_detail_record(card_id: str) -> Optional[tuple[tuple[str, int], dict[str, Any]]]:
    detail_url = API_CARD_ID_URL_TMPL.format(card_id=card_id)
    detail = _http_json_get(detail_url)
    results = detail.get("results", []) if isinstance(detail, dict) else []
    if not isinstance(results, list) or not results:
        return None
    local_record = _build_local_record_from_card_detail(results[0])
    if local_record is None:
        return None
    return (_card_match_key(local_record), local_record)


def main() -> int:
    parser = argparse.ArgumentParser(description="Update cards.json missing cards and legality from official FAB Card Vault API")
    parser.add_argument("--cards", default=str(DEFAULT_CARDS_PATH), help="Path to local cards.json")
    parser.add_argument("--decks", default=str(DEFAULT_DECKS_PATH), help="Path to fabrary_decks.json")
    parser.add_argument(
        "--legality-scope",
        choices=("all", "decks"),
        default="all",
        help="Choose which existing cards get legality refresh. 'all' refreshes all local cards; 'decks' refreshes only deck-referenced cards.",
    )
    parser.add_argument(
        "--detail-workers",
        type=int,
        default=8,
        help="Number of concurrent workers for card detail fetches.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing cards.json")
    args = parser.parse_args()
    progress = _ProgressTracker()

    cards_path = Path(args.cards)
    decks_path = Path(args.decks)
    if not cards_path.exists():
        raise FileNotFoundError(f"cards file not found: {cards_path}")
    if not decks_path.exists():
        raise FileNotFoundError(f"decks file not found: {decks_path}")

    progress.stage("Loading local cards and deck files")
    local_cards_raw = _load_json(cards_path)
    if not isinstance(local_cards_raw, list):
        raise ValueError(f"Expected list in cards file: {cards_path}")
    local_cards: list[dict[str, Any]] = [rec for rec in local_cards_raw if isinstance(rec, dict)]

    deck_names = _read_deck_card_names(decks_path)
    local_name_keys = {_card_name_key(rec.get("name", "")) for rec in local_cards if isinstance(rec, dict)}
    missing_name_keys = {_card_name_key(name) for name in deck_names if _card_name_key(name) not in local_name_keys}
    deck_name_keys = {_card_name_key(name) for name in deck_names}
    progress.stage(f"Loaded {len(local_cards)} local cards and {len(deck_names)} unique deck card names")

    lookup_payload: list[dict[str, str]] = []
    key_to_local_cards: dict[tuple[str, int], list[dict[str, Any]]] = {}
    local_keys_to_refresh: set[tuple[str, int]] = set()

    for rec in local_cards:
        key = _card_match_key(rec)
        key_to_local_cards.setdefault(key, []).append(rec)
        if args.legality_scope == "all":
            local_keys_to_refresh.add(key)
        elif key[0] in deck_name_keys:
            local_keys_to_refresh.add(key)

    for name_key, pitch in sorted(local_keys_to_refresh):
        if not name_key:
            continue
        display_name = next((str(c.get("name", "")).strip() for c in key_to_local_cards.get((name_key, pitch), []) if str(c.get("name", "")).strip()), "")
        if not display_name:
            continue
        lookup_payload.append({"name": display_name, "color": _pitch_to_color(pitch)})

    # For missing names, probe common pitch colors plus blank color.
    for name_key in sorted(missing_name_keys):
        display_name = next((name for name in sorted(deck_names) if _card_name_key(name) == name_key), "")
        if not display_name:
            continue
        for color in ("red", "yellow", "blue", ""):
            lookup_payload.append({"name": display_name, "color": color})
    progress.stage(
        f"Prepared {len(lookup_payload)} lookup requests ({len(local_keys_to_refresh)} for legality refresh, {len(missing_name_keys)} missing names)"
    )

    card_id_by_key: dict[tuple[str, str], str] = {}
    lookup_batches = list(_batched(lookup_payload, size=200))
    progress.stage(f"Resolving card ids via Card Vault find API in {len(lookup_batches)} batches")
    for batch_idx, batch in enumerate(lookup_batches, start=1):
        progress.tick("Lookup batch", batch_idx, len(lookup_batches), every=1)
        response = _http_json_post(API_FIND_CARDS_URL, batch)
        if not isinstance(response, list):
            continue
        for row in response:
            if not isinstance(row, dict):
                continue
            raw_name = str(row.get("name", "")).strip()
            raw_color = str(row.get("color", "")).strip().lower()
            raw_card_id = str(row.get("card_id", "")).strip()
            if raw_name and raw_card_id:
                card_id_by_key[(_card_name_key(raw_name), raw_color)] = raw_card_id

    needed_card_ids: set[str] = set(card_id_by_key.values())
    detail_records_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    unresolved_missing: list[str] = []
    detail_workers = max(1, int(args.detail_workers))
    progress.stage(f"Fetching detailed card records for {len(needed_card_ids)} card ids (workers={detail_workers})")

    needed_ids_sorted = sorted(needed_card_ids)
    with ThreadPoolExecutor(max_workers=detail_workers) as executor:
        future_to_card_id = {executor.submit(_fetch_card_detail_record, card_id): card_id for card_id in needed_ids_sorted}
        for idx, future in enumerate(as_completed(future_to_card_id), start=1):
            progress.tick("Card detail fetch", idx, len(needed_ids_sorted), every=50)
            card_id = future_to_card_id[future]
            try:
                record = future.result()
            except Exception as exc:
                print(f"Warning: detail fetch failed for {card_id}: {exc}", file=sys.stderr)
                continue
            if record is None:
                continue
            key, local_record = record
            detail_records_by_key[key] = local_record

    for name_key in sorted(missing_name_keys):
        found = any(key[0] == name_key for key in detail_records_by_key.keys())
        if not found:
            unresolved_missing.append(name_key)
    progress.stage(f"Matched {len(detail_records_by_key)} card records; unresolved missing names: {len(unresolved_missing)}")

    added_cards: list[dict[str, Any]] = []
    legality_updates = 0
    existing_ids = {str(rec.get("id", "")) for rec in local_cards if str(rec.get("id", ""))}
    existing_keys = {_card_match_key(rec) for rec in local_cards}

    # Update legality on existing cards using matched official entries.
    progress.stage("Refreshing legality for existing local cards")
    for idx, rec in enumerate(local_cards, start=1):
        progress.tick("Legality refresh", idx, len(local_cards), every=200)
        matched = detail_records_by_key.get(_card_match_key(rec))
        if not matched:
            continue
        old_legality = rec.get("legality", {})
        new_legality = matched.get("legality", {})
        if isinstance(new_legality, dict) and new_legality and old_legality != new_legality:
            rec["legality"] = new_legality
            legality_updates += 1

    # Add missing cards discovered from official API.
    progress.stage("Collecting newly discovered cards")
    detail_items = sorted(detail_records_by_key.items())
    for idx, (key, rec) in enumerate(detail_items, start=1):
        progress.tick("New-card scan", idx, len(detail_items), every=200)
        if key in existing_keys:
            continue
        cid = str(rec.get("id", ""))
        if not cid or cid in existing_ids:
            continue
        existing_ids.add(cid)
        added_cards.append(rec)

    merged_cards = local_cards + added_cards
    merged_cards.sort(key=lambda rec: (_card_name_key(rec.get("name", "")), int(rec.get("pitch", 0) or 0), str(rec.get("id", ""))))

    print(f"Deck card names scanned: {len(deck_names)}")
    print(f"Missing names detected: {len(missing_name_keys)}")
    print(f"Official card records matched: {len(detail_records_by_key)}")
    print(f"Cards added: {len(added_cards)}")
    print(f"Legality rows updated: {legality_updates}")
    if unresolved_missing:
        preview = ", ".join(unresolved_missing[:10])
        extra = f" (+{len(unresolved_missing)-10} more)" if len(unresolved_missing) > 10 else ""
        print(f"Unresolved missing names: {preview}{extra}")

    if args.dry_run:
        progress.stage("Dry-run complete")
        print("Dry run complete; no files were written.")
        return 0

    cards_path.write_text(json.dumps(merged_cards, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    progress.stage(f"Write complete: {len(merged_cards)} cards")
    print(f"Wrote {len(merged_cards)} cards to {cards_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
