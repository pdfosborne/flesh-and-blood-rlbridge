"""Tests for deck card id helpers."""

from __future__ import annotations

from fab_tui.deck_cards import assign_pitch_variants


def _variants(*ids: str) -> list[dict]:
    pitch = {"red": 1, "yellow": 2, "blue": 3}
    out = []
    for cid in ids:
        suffix = cid.rsplit("_", 1)[-1]
        out.append({"id": cid, "pitch": pitch.get(suffix, 99)})
    return out


def test_assign_six_copies_across_three_pitches() -> None:
    cands = _variants("warriors_valor_red", "warriors_valor_yellow", "warriors_valor_blue")
    assert assign_pitch_variants(cands, 6) == [
        "warriors_valor_red",
        "warriors_valor_red",
        "warriors_valor_yellow",
        "warriors_valor_yellow",
        "warriors_valor_blue",
        "warriors_valor_blue",
    ]


def test_assign_four_copies_uses_red_then_yellow() -> None:
    cands = _variants("snatch_red", "snatch_yellow", "snatch_blue")
    assert assign_pitch_variants(cands, 4) == [
        "snatch_red",
        "snatch_red",
        "snatch_yellow",
        "snatch_yellow",
    ]


def test_assign_two_copies_stays_on_first_pitch() -> None:
    cands = _variants("snatch_red", "snatch_yellow", "snatch_blue")
    assert assign_pitch_variants(cands, 2) == ["snatch_red", "snatch_red"]


def test_expand_preserves_explicit_red_and_blue_split() -> None:
    from collections import Counter

    from scripts.deck.fix_talishar_asset_pitch_shorthand import _expand_deck_counts, _load_name_map

    src = Counter({"warriors_valor_red": 2, "warriors_valor_blue": 2})
    expanded = Counter(_expand_deck_counts(src, _load_name_map()))
    assert expanded == src
