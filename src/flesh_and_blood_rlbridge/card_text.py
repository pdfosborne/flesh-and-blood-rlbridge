"""Frozen card text embeddings for attention_v2_text policy."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .card_vocab import card_index, vocab_size

TEXT_EMBED_VERSION = "v1"
TEXT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMBED_FILENAME = f"card_text_embeddings_{TEXT_EMBED_VERSION}.npz"
TEXT_EMBED_META_FILENAME = f"card_text_embeddings_{TEXT_EMBED_VERSION}.meta.json"

_CARD_DB_DIR = Path(__file__).parent / "card_db"
_PACKAGE_EMBED_PATH = _CARD_DB_DIR / TEXT_EMBED_FILENAME
_PACKAGE_META_PATH = _CARD_DB_DIR / TEXT_EMBED_META_FILENAME


def _agent_cache_dir() -> Path:
    override = os.environ.get("FAB_AGENT_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "results" / "agent_cache"


def shared_text_embed_cache_dir(cache_dir: Path | None = None) -> Path:
    return Path(cache_dir or _agent_cache_dir()) / "shared"


def shared_text_embed_path(cache_dir: Path | None = None) -> Path:
    return shared_text_embed_cache_dir(cache_dir) / TEXT_EMBED_FILENAME


def shared_text_embed_meta_path(cache_dir: Path | None = None) -> Path:
    return shared_text_embed_cache_dir(cache_dir) / TEXT_EMBED_META_FILENAME


def cards_json_sha256() -> str:
    data = (_CARD_DB_DIR / "cards.json").read_bytes()
    return hashlib.sha256(data).hexdigest()


def _read_meta(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def resolve_text_embed_paths(
    *,
    cache_dir: Path | None = None,
    explicit_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return (npz_path, meta_path) using package/cache/env resolution order."""
    if explicit_path is not None:
        npz = Path(explicit_path).expanduser().resolve()
        meta = npz.with_name(TEXT_EMBED_META_FILENAME)
        return npz, meta

    env_path = os.environ.get("FAB_TEXT_EMBED_PATH", "").strip()
    if env_path:
        npz = Path(env_path).expanduser().resolve()
        meta = npz.with_name(TEXT_EMBED_META_FILENAME)
        return npz, meta

    cached = shared_text_embed_path(cache_dir)
    if cached.is_file():
        return cached, shared_text_embed_meta_path(cache_dir)

    return _PACKAGE_EMBED_PATH, _PACKAGE_META_PATH


def validate_text_embed_meta(meta: dict[str, Any], *, table_rows: int) -> None:
    expected_version = str(meta.get("text_embed_version", ""))
    if expected_version and expected_version != TEXT_EMBED_VERSION:
        raise ValueError(
            f"Text embedding version mismatch: expected {TEXT_EMBED_VERSION!r}, "
            f"got {expected_version!r}. Run: fab-bridge agents sync"
        )
    expected_vocab = int(meta.get("vocab_size", 0))
    if expected_vocab > 0 and expected_vocab != table_rows:
        raise ValueError(
            f"Text embedding vocab_size mismatch: meta={expected_vocab}, table={table_rows}"
        )
    expected_sha = str(meta.get("cards_json_sha256", ""))
    if expected_sha and expected_sha != cards_json_sha256():
        raise ValueError(
            "Text embeddings were built from a different cards.json; "
            "regenerate embeddings or run: fab-bridge agents sync"
        )


@lru_cache(maxsize=1)
def load_text_embedding_table(
    *,
    cache_dir: str | None = None,
    explicit_path: str | None = None,
) -> np.ndarray:
    """Load frozen embedding matrix shaped (vocab_size+1, embed_dim); row 0 is zeros."""
    npz_path, meta_path = resolve_text_embed_paths(
        cache_dir=Path(cache_dir) if cache_dir else None,
        explicit_path=explicit_path,
    )
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"Card text embeddings not found at {npz_path}. "
            "Run scripts/card_db/build_card_text_embeddings.py or fab-bridge agents sync."
        )
    data = np.load(npz_path)
    table = np.asarray(data["embeddings"], dtype=np.float32)
    meta = _read_meta(meta_path)
    validate_text_embed_meta(meta, table_rows=int(table.shape[0]))
    return table


def text_embed_dim() -> int:
    try:
        return int(load_text_embedding_table().shape[1])
    except FileNotFoundError:
        meta = _read_meta(_PACKAGE_META_PATH)
        return int(meta.get("embed_dim", 384))


def text_embed_for_card_index(card_idx: int) -> np.ndarray:
    table = load_text_embedding_table()
    idx = int(card_idx)
    if idx < 0 or idx >= table.shape[0]:
        return np.zeros(table.shape[1], dtype=np.float32)
    return table[idx]


def copy_package_embeddings_to_cache(cache_dir: Path | None = None) -> Path:
    """Copy bundled embeddings into agent cache shared/ (bootstrap/offline)."""
    dest_dir = shared_text_embed_cache_dir(cache_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_npz = dest_dir / TEXT_EMBED_FILENAME
    dest_meta = dest_dir / TEXT_EMBED_META_FILENAME
    if not _PACKAGE_EMBED_PATH.is_file():
        raise FileNotFoundError(f"Package embeddings missing at {_PACKAGE_EMBED_PATH}")
    dest_npz.write_bytes(_PACKAGE_EMBED_PATH.read_bytes())
    if _PACKAGE_META_PATH.is_file():
        dest_meta.write_bytes(_PACKAGE_META_PATH.read_bytes())
    load_text_embedding_table.cache_clear()
    return dest_npz


def embedding_status(cache_dir: Path | None = None) -> dict[str, Any]:
    npz_path, meta_path = resolve_text_embed_paths(cache_dir=cache_dir)
    meta = _read_meta(meta_path)
    present = npz_path.is_file()
    sha256 = ""
    if present:
        sha256 = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    return {
        "text_embed_present": present,
        "present": present,
        "text_embed_path": str(npz_path),
        "text_embed_meta_path": str(meta_path),
        "text_embed_version": str(meta.get("text_embed_version", TEXT_EMBED_VERSION)),
        "embed_dim": int(meta.get("embed_dim", 0)) if meta else 0,
        "vocab_size": int(meta.get("vocab_size", vocab_size())),
        "sha256": sha256,
        "model": str(meta.get("model", TEXT_EMBED_MODEL)),
    }
