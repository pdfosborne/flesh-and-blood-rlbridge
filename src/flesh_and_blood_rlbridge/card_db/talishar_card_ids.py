"""Map rlbridge ``cards.json`` IDs to Talishar engine card IDs."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

_PITCH_SUFFIX = re.compile(r"_(red|yellow|blue|purple)$")
_TALISHAR_ID_RE = re.compile(r"^[a-z][a-z0-9_]+$")
_PHP_CARD_ID_RE = re.compile(r'"([a-z0-9][a-z0-9_]*)"\s*=>')

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CARDS_PATH = Path(__file__).with_name("cards.json")
_DEFAULT_TALISHAR_PHP = (
    _REPO_ROOT / "Talishar" / "GeneratedCode" / "GeneratedCardDictionaries.php"
)


def _collapse_numeric_underscores(card_id: str) -> str:
    """``10_000_year_reunion_red`` → ``10000_year_reunion_red``."""
    parts = card_id.split("_")
    out: list[str] = []
    i = 0
    while i < len(parts):
        if parts[i].isdigit():
            digits = parts[i]
            j = i + 1
            while j < len(parts) and parts[j].isdigit():
                digits += parts[j]
                j += 1
            out.append(digits)
            i = j
        else:
            out.append(parts[i])
            i += 1
    return "_".join(out)


def _apostrophe_slug_variants(card_id: str) -> list[str]:
    """Generate Talishar-style apostrophe slug variants."""
    variants = [card_id]
    current = card_id
    while True:
        updated = re.sub(r"(\w)_s_(\w)", r"\1s_\2", current)
        if updated == current:
            break
        variants.append(updated)
        current = updated
    collapsed = _collapse_numeric_underscores(card_id)
    if collapsed not in variants:
        variants.append(collapsed)
    for base in list(variants):
        collapsed_base = _collapse_numeric_underscores(base)
        if collapsed_base not in variants:
            variants.append(collapsed_base)
    return variants


def _similarity_key(card_id: str) -> str:
    """Normalise punctuation differences while preserving word order."""
    return re.sub(r"_+", "_", card_id.strip().lower()).strip("_")


def _best_talishar_id_match(card_id: str, candidates: frozenset[str]) -> Optional[str]:
    """Return a high-confidence canonical Talishar ID for a near-miss slug."""
    token = card_id.strip().lower()
    if not token or not candidates:
        return None

    token_key = _similarity_key(token)
    exact_key_matches = [cid for cid in candidates if _similarity_key(cid) == token_key]
    if len(exact_key_matches) == 1:
        return exact_key_matches[0]

    token_parts = token_key.split("_")
    first_part = token_parts[0] if token_parts else ""
    pitch_suffix = _PITCH_SUFFIX.search(token)
    suffix = pitch_suffix.group(0) if pitch_suffix else ""

    scored: list[tuple[float, str]] = []
    for candidate in candidates:
        candidate_key = _similarity_key(candidate)
        if first_part and not candidate_key.startswith(first_part):
            continue
        if suffix and not candidate.endswith(suffix):
            continue
        score = SequenceMatcher(None, token_key, candidate_key).ratio()
        if score >= 0.94:
            scored.append((score, candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    best_score = scored[0][0]
    best = [candidate for score, candidate in scored if score == best_score]
    return best[0] if len(best) == 1 else None


@lru_cache(maxsize=1)
def load_talishar_card_ids(
    php_path: str = str(_DEFAULT_TALISHAR_PHP),
) -> frozenset[str]:
    path = Path(php_path)
    if not path.is_file():
        return frozenset()
    text = path.read_text(encoding="utf-8", errors="replace")
    start = text.find("function GeneratedCardName")
    if start < 0:
        return frozenset(_PHP_CARD_ID_RE.findall(text))
    end = text.find("\nfunction ", start + 1)
    block = text[start:end] if end > start else text[start:]
    return frozenset(_PHP_CARD_ID_RE.findall(block))


@lru_cache(maxsize=1)
def _talishar_name_to_ids(
    php_path: str = str(_DEFAULT_TALISHAR_PHP),
) -> dict[str, list[str]]:
    path = Path(php_path)
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    start = text.find("function GeneratedCardName")
    if start < 0:
        return {}
    end = text.find("\nfunction ", start + 1)
    block = text[start:end] if end > start else text[start:]
    mapping: dict[str, list[str]] = {}
    for match in re.finditer(r'"([a-z0-9][a-z0-9_]*)"\s*=>\s*"([^"]*)"', block):
        cid, name = match.group(1), match.group(2)
        key = name.strip().lower()
        mapping.setdefault(key, []).append(cid)
    return mapping


@lru_cache(maxsize=2)
def _cards_db_by_id(cards_path: str = str(_DEFAULT_CARDS_PATH)) -> dict[str, dict[str, Any]]:
    path = Path(cards_path)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, dict[str, Any]] = {}
    if isinstance(raw, list):
        for rec in raw:
            if isinstance(rec, dict):
                cid = str(rec.get("id") or "").strip()
                if cid:
                    out[cid] = rec
    return out


@lru_cache(maxsize=1)
def load_talishar_card_subtypes(
    php_path: str = str(_DEFAULT_TALISHAR_PHP),
) -> dict[str, str]:
    """Parse ``GeneratedCardSubtype`` from Talishar PHP (card id → subtype string)."""
    path = Path(php_path)
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    start = text.find("function GeneratedCardSubtype")
    if start < 0:
        return {}
    end = text.find("\nfunction ", start + 1)
    block = text[start:end] if end > start else text[start:]
    mapping: dict[str, str] = {}
    for match in re.finditer(r'"([a-z0-9][a-z0-9_]*)"\s*=>\s*"([^"]*)"', block):
        mapping[match.group(1)] = match.group(2)
    return mapping


@lru_cache(maxsize=1)
def load_talishar_one_handed_weapons(
    php_path: str = str(_DEFAULT_TALISHAR_PHP),
) -> frozenset[str]:
    """Parse Talishar ``GeneratedIs1H`` weapon ids (true = one-handed)."""
    path = Path(php_path)
    if not path.is_file():
        return frozenset()
    text = path.read_text(encoding="utf-8", errors="replace")
    start = text.find("function GeneratedIs1H")
    if start < 0:
        return frozenset()
    end = text.find("\nfunction ", start + 1)
    block = text[start:end] if end > start else text[start:]
    one_handed: set[str] = set()
    for match in re.finditer(r'"([a-z0-9][a-z0-9_]*)"\s*=>\s*true', block):
        one_handed.add(match.group(1))
    return frozenset(one_handed)


def clear_talishar_card_id_caches() -> None:
    load_talishar_card_ids.cache_clear()
    load_talishar_card_subtypes.cache_clear()
    load_talishar_one_handed_weapons.cache_clear()
    _talishar_name_to_ids.cache_clear()
    _cards_db_by_id.cache_clear()


class TalisharCardIdResolver:
    """Resolve local card IDs to IDs understood by the Talishar PHP engine."""

    def __init__(
        self,
        *,
        talishar_php_path: str | Path | None = None,
        cards_path: str | Path | None = None,
    ) -> None:
        php = str(talishar_php_path or _DEFAULT_TALISHAR_PHP)
        cards = str(cards_path or _DEFAULT_CARDS_PATH)
        self._talishar_ids = load_talishar_card_ids(php)
        self._name_to_ids = _talishar_name_to_ids(php)
        self._cards_db = _cards_db_by_id(cards)

    def resolve(self, card_id: str) -> Optional[str]:
        token = str(card_id or "").strip().lower()
        if not token:
            return None
        if token in self._talishar_ids:
            return token
        for variant in _apostrophe_slug_variants(token):
            if variant in self._talishar_ids:
                return variant

        rec = self._cards_db.get(token) or self._cards_db.get(card_id)
        name = str(rec.get("name") or "").strip().lower() if rec else ""
        if name:
            candidates = self._name_to_ids.get(name, [])
            if len(candidates) == 1:
                return candidates[0]
            pitch_suffix = _PITCH_SUFFIX.search(token)
            if pitch_suffix:
                suffix = pitch_suffix.group(0)
                for cid in candidates:
                    if cid.endswith(suffix):
                        return cid
            if candidates:
                return candidates[0]
        best = _best_talishar_id_match(token, self._talishar_ids)
        if best is not None:
            return best
        return None

    def sanitize_deck(
        self,
        deck: dict[str, int],
    ) -> tuple[dict[str, int], list[str]]:
        """Return *(sanitized_deck, warnings)* dropping unresolvable cards."""
        cleaned: dict[str, int] = {}
        warnings: list[str] = []
        for card_id, count in deck.items():
            qty = int(count)
            if qty <= 0:
                continue
            resolved = self.resolve(card_id)
            if resolved is None:
                warnings.append(f"Dropped unknown Talishar card id: {card_id} x{qty}")
                continue
            if resolved != card_id:
                warnings.append(f"Mapped {card_id} -> {resolved} x{qty}")
            cleaned[resolved] = cleaned.get(resolved, 0) + qty
        return cleaned, warnings


def sanitize_deck_for_talishar(
    deck: dict[str, int],
    *,
    talishar_php_path: str | Path | None = None,
    cards_path: str | Path | None = None,
) -> tuple[dict[str, int], list[str]]:
    resolver = TalisharCardIdResolver(
        talishar_php_path=talishar_php_path,
        cards_path=cards_path,
    )
    return resolver.sanitize_deck(deck)


def normalize_cards_json_file(
    cards_path: str | Path | None = None,
    *,
    talishar_php_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Rewrite duplicate card records to canonical Talishar IDs where possible."""
    path = Path(cards_path or _DEFAULT_CARDS_PATH)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected card list in {path}")

    resolver = TalisharCardIdResolver(
        talishar_php_path=talishar_php_path,
        cards_path=path,
    )

    merged: dict[str, dict[str, Any]] = {}
    remapped = 0
    dropped = 0
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        cid = str(rec.get("id") or "").strip()
        if not cid:
            continue
        resolved = resolver.resolve(cid)
        if resolved is None:
            dropped += 1
            continue
        if resolved != cid:
            remapped += 1
        target = dict(rec)
        target["id"] = resolved
        existing = merged.get(resolved)
        if existing is None:
            merged[resolved] = target
            continue
        # Prefer the record with richer metadata.
        def _score(item: dict[str, Any]) -> int:
            return int(bool(item.get("type_line"))) + int(bool(item.get("text"))) + int(
                bool(item.get("card_types"))
            )

        if _score(target) > _score(existing):
            merged[resolved] = target

    normalized = list(merged.values())
    normalized.sort(key=lambda item: str(item.get("id", "")))

    if not dry_run:
        path.write_text(
            json.dumps(normalized, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        clear_talishar_card_id_caches()

    return {
        "cards_path": str(path),
        "before": len(raw),
        "after": len(normalized),
        "remapped": remapped,
        "dropped": dropped,
        "dry_run": dry_run,
    }
