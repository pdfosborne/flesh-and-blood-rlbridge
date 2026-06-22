"""Tests for FAB rules-text helpers."""

from __future__ import annotations

from flesh_and_blood_rlbridge import fab_rules


def test_derive_keywords_from_text_finds_common_mechanics() -> None:
    text = "**Dominate** When this hits, **go again**.{br}**Ward 3**"
    keywords = fab_rules.derive_keywords_from_text(text)
    assert "dominate" in keywords
    assert "go_again" in keywords
    assert "ward" in keywords


def test_parse_ward_value_reads_numeric_ward() -> None:
    assert fab_rules.parse_ward_value("**Ward 10**", []) == 10
    assert fab_rules.parse_ward_value("Ward 2", ["ward"]) == 2
    assert fab_rules.parse_ward_value("No ward here", []) == 0
