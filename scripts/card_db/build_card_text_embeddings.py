#!/usr/bin/env python3
"""Build frozen MiniLM embeddings for all cards in cards.json."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from flesh_and_blood_rlbridge.card_text import (  # noqa: E402
    TEXT_EMBED_FILENAME,
    TEXT_EMBED_META_FILENAME,
    TEXT_EMBED_MODEL,
    TEXT_EMBED_VERSION,
    cards_json_sha256,
)
from flesh_and_blood_rlbridge.card_vocab import _build_card_index, card_index  # noqa: E402
from flesh_and_blood_rlbridge.fab_rules import derive_keywords_from_text  # noqa: E402

_CARDS_PATH = _REPO_ROOT / "src" / "flesh_and_blood_rlbridge" / "card_db" / "cards.json"
_OUT_DIR = _CARDS_PATH.parent

_RESOURCE_RE = re.compile(r"\{[a-z]+\}", re.I)


def _normalize_card_text(text: str) -> str:
    body = str(text or "").replace("{br}", " ")
    body = re.sub(r"\*\*([^*]+)\*\*", r"\1", body)
    body = _RESOURCE_RE.sub(" ", body)
    return " ".join(body.split())


def _embedding_input(rec: dict) -> str:
    name = str(rec.get("name", "") or "")
    type_line = str(rec.get("type_line", "") or "")
    clazz = str(rec.get("class", "") or "")
    talent = str(rec.get("talent", "") or "")
    keywords = rec.get("keywords") or derive_keywords_from_text(str(rec.get("text", "") or ""))
    kw = ", ".join(str(k) for k in keywords if k)
    text = _normalize_card_text(str(rec.get("text", "") or ""))
    parts = [name, type_line, f"{clazz} {talent}".strip(), f"keywords: {kw}", text]
    return " | ".join(p for p in parts if p.strip())


def main() -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers required: pip install 'flesh-and-blood-rlbridge[train]' "
            "or pip install sentence-transformers"
        ) from exc

    records = json.loads(_CARDS_PATH.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("cards.json must be a list")

    vocab = _build_card_index()
    size = max(len(vocab) + 1, 1)
    texts_by_index: dict[int, str] = {}
    rec_by_id = {str(r.get("id", "")): r for r in records if isinstance(r, dict)}

    for cid in vocab:
        idx = card_index(cid)
        if idx <= 0:
            continue
        rec = rec_by_id.get(cid) or rec_by_id.get(cid.replace("-", "_"))
        if rec is None:
            texts_by_index[idx] = cid.replace("_", " ")
        else:
            texts_by_index[idx] = _embedding_input(rec)

    ordered_indices = sorted(texts_by_index)
    ordered_texts = [texts_by_index[i] for i in ordered_indices]

    model = SentenceTransformer(TEXT_EMBED_MODEL)
    vectors = model.encode(ordered_texts, show_progress_bar=True, normalize_embeddings=True)
    embed_dim = int(vectors.shape[1])

    table = np.zeros((size, embed_dim), dtype=np.float32)

    for row_pos, card_idx in enumerate(ordered_indices):
        table[card_idx] = np.asarray(vectors[row_pos], dtype=np.float32)

    out_npz = _OUT_DIR / TEXT_EMBED_FILENAME
    out_meta = _OUT_DIR / TEXT_EMBED_META_FILENAME
    np.savez_compressed(out_npz, embeddings=table)

    meta = {
        "text_embed_version": TEXT_EMBED_VERSION,
        "model": TEXT_EMBED_MODEL,
        "embed_dim": embed_dim,
        "vocab_size": size,
        "cards_json_sha256": cards_json_sha256(),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "card_count": len(ordered_indices) - (1 if 0 in ordered_indices else 0),
    }
    out_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_npz} ({table.shape})")
    print(f"Wrote {out_meta}")


if __name__ == "__main__":
    main()
