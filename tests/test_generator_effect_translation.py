"""Tests for PHP effect auto-translation in the C++ generator."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "cpp"))

from generate_cpp_engine import CardMeta, _render_card_effect_body  # noqa: E402


def test_render_draw_effect() -> None:
    meta = CardMeta(card_id="test_draw", php_snippet="DrawCard($playerID, 2);")
    body, status = _render_card_effect_body(meta)
    assert status == "auto"
    assert "draw_cards" in body
    assert "2" in body


def test_render_damage_effect() -> None:
    meta = CardMeta(
        card_id="test_dmg",
        php_snippet="$gamestate->playerHealth -= 3;",
        power=3,
    )
    body, status = _render_card_effect_body(meta)
    assert status == "auto"
    assert "health" in body


def test_render_stub_has_parity_todo() -> None:
    meta = CardMeta(card_id="unknown_card", php_snippet="", power=4)
    body, status = _render_card_effect_body(meta)
    assert status == "stub"
    assert "PARITY_TODO" in body
