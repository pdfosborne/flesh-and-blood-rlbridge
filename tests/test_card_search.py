"""Tests for fuzzy card search."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from fab_tui.card_search import CardSearchIndex


def test_search_finds_card_by_partial_name() -> None:
    index = CardSearchIndex("silver_age")
    hits = index.search("Flying Kick", limit=5)
    assert hits
    assert any("flying" in hit.name.lower() for hit in hits)


def test_search_prefers_talishar_card_ids() -> None:
    index = CardSearchIndex("silver_age")
    hits = index.search("Surging Strike", limit=3)
    assert hits
    assert all("_" in hit.card_id for hit in hits)


def test_lookup_display_name() -> None:
    index = CardSearchIndex("silver_age")
    hits = index.search("enlightened strike", limit=1)
    assert hits
    assert index.display_name(hits[0].card_id) == hits[0].name


def test_lookup_respects_pitch_suffix() -> None:
    index = CardSearchIndex("silver_age")
    red = index.lookup("snatch_red")
    yellow = index.lookup("snatch_yellow")
    assert red is not None and yellow is not None
    assert red.pitch == 1
    assert yellow.pitch == 2
    assert red.card_id != yellow.card_id


def test_lookup_matches_hyphen_and_underscore_ids() -> None:
    index = CardSearchIndex("silver_age")
    underscore = index.lookup("surging_strike_yellow")
    hyphen = index.lookup("surging-strike-yellow")
    assert underscore is not None and hyphen is not None
    assert underscore.pitch == 2
    assert hyphen.pitch == 2
