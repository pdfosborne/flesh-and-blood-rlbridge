"""Tests for Fabrary meta matchup lookup."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from flesh_and_blood_rlbridge.card_db.fabrary_meta import (  # noqa: E402
    FORMAT_GAMES_SLUG,
    build_cdn_url,
    dataset_key,
    deck_hero_play_weight,
    hero_id_to_fabrary_slug,
    hero_play_counts,
    load_fabrary_meta,
    lookup_deck_matchup,
    lookup_hero_matchup,
    matchup_key,
    parse_hero_results,
)

_FIXTURE = _REPO / "tests" / "fixtures" / "fabrary_meta_sample.json"


def test_format_games_slug_mapping_has_six_entries() -> None:
    assert len(FORMAT_GAMES_SLUG) == 6
    assert FORMAT_GAMES_SLUG[("silver_age", "competitive")] == "silver-age-competitive"
    assert FORMAT_GAMES_SLUG[("classic_constructed", "all")] == "all-classic-constructed"


@pytest.mark.parametrize(
    ("format_name", "games", "period", "expected"),
    [
        (
            "silver_age",
            "competitive",
            "last-30-days",
            "https://content.fabrary.net/results/silver-age-competitive-last-30-days.json",
        ),
        (
            "classic_constructed",
            "standard",
            "2026-06",
            "https://content.fabrary.net/results/classic-constructed-2026-06.json",
        ),
        (
            "silver_age",
            "all",
            "last-7-days",
            "https://content.fabrary.net/results/all-silver-age-last-7-days.json",
        ),
    ],
)
def test_build_cdn_url(format_name: str, games: str, period: str, expected: str) -> None:
    assert build_cdn_url(format_name, games, period) == expected


def test_hero_id_to_fabrary_slug() -> None:
    assert hero_id_to_fabrary_slug("hero_arakni_web_of_deceit") == "arakni-web-of-deceit"
    assert hero_id_to_fabrary_slug("hero_dash") == "dash"
    assert hero_id_to_fabrary_slug("hero_gravy_bones") == "gravy-bones"


def test_matchup_key_is_sorted() -> None:
    assert matchup_key("gravy-bones", "dash") == "dash|gravy-bones"
    assert matchup_key("dash", "gravy-bones") == "dash|gravy-bones"


def test_parse_hero_results_from_fixture() -> None:
    raw = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    matchups = parse_hero_results(raw)
    assert "dash|gravy-bones" in matchups
    row = matchups["dash|gravy-bones"]
    assert row["hero_a"] == "dash"
    assert row["hero_b"] == "gravy-bones"
    assert row["plays"] == 42
    assert row["wins_a"] == 25
    assert row["win_rate_a"] == pytest.approx(25 / 42)
    assert row["win_rate_b"] == pytest.approx(17 / 42)


def test_parse_hero_results_min_plays_filter() -> None:
    raw = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    matchups = parse_hero_results(raw, min_plays=20)
    assert "dash|gravy-bones" in matchups
    assert "arakni-web-of-deceit|azalea" not in matchups


def test_lookup_hero_matchup_from_inline_meta() -> None:
    meta = {
        "datasets": {
            dataset_key("silver_age", "competitive", "last-30-days"): {
                "matchups": parse_hero_results(json.loads(_FIXTURE.read_text(encoding="utf-8"))),
            }
        }
    }
    ref = lookup_hero_matchup(
        "hero_dash",
        "hero_gravy_bones",
        format_name="silver_age",
        games="competitive",
        meta=meta,
    )
    assert ref is not None
    assert ref["p1_win_rate"] == pytest.approx(25 / 42)
    assert ref["p2_win_rate"] == pytest.approx(17 / 42)
    assert ref["plays"] == 42

    swapped = lookup_hero_matchup(
        "hero_gravy_bones",
        "hero_dash",
        format_name="silver_age",
        games="competitive",
        meta=meta,
    )
    assert swapped is not None
    assert swapped["p1_win_rate"] == pytest.approx(17 / 42)
    assert swapped["p2_win_rate"] == pytest.approx(25 / 42)


def test_lookup_deck_matchup_with_mock_decks(tmp_path: Path) -> None:
    decks_path = tmp_path / "decks.json"
    decks_path.write_text(
        json.dumps(
            {
                "decks": [
                    {"id": "fab_dash_sage_combo", "hero_id": "hero_dash"},
                    {"id": "fab_precon_sage_ch3_gravy_bones", "hero_id": "hero_gravy_bones"},
                ]
            }
        ),
        encoding="utf-8",
    )
    meta = {
        "datasets": {
            dataset_key("silver_age", "competitive", "last-30-days"): {
                "matchups": parse_hero_results(json.loads(_FIXTURE.read_text(encoding="utf-8"))),
            }
        }
    }
    ref = lookup_deck_matchup(
        "fab_dash_sage_combo",
        "fab_precon_sage_ch3_gravy_bones",
        format_name="silver_age",
        games="competitive",
        meta=meta,
        decks_path=decks_path,
    )
    assert ref is not None
    assert ref["p1_deck"] == "fab_dash_sage_combo"
    assert ref["p2_deck"] == "fab_precon_sage_ch3_gravy_bones"
    assert ref["p1_win_rate"] == pytest.approx(25 / 42)


def test_load_fabrary_meta_committed_file() -> None:
    meta = load_fabrary_meta()
    datasets = meta.get("datasets")
    assert isinstance(datasets, dict)
    assert len(datasets) >= 1
    key = dataset_key("silver_age", "competitive", "last-30-days")
    assert key in datasets


def test_hero_play_counts_sums_both_heroes_per_matchup() -> None:
    meta = {
        "datasets": {
            dataset_key("silver_age", "all", "last-30-days"): {
                "matchups": {
                    "dash|gravy-bones": {
                        "hero_a": "dash",
                        "hero_b": "gravy-bones",
                        "plays": 10,
                    },
                    "dash|kayo": {
                        "hero_a": "dash",
                        "hero_b": "kayo",
                        "plays": 5,
                    },
                }
            }
        }
    }
    counts = hero_play_counts("silver_age", games="all", meta=meta)
    assert counts["dash"] == 15
    assert counts["gravy-bones"] == 10
    assert counts["kayo"] == 5


def test_deck_hero_play_weight_uses_min_weight_for_unknown_hero() -> None:
    weight = deck_hero_play_weight(
        {"hero_id": "hero_unknown_hero"},
        {"dash": 100},
        min_weight=1,
    )
    assert weight == 1
    assert deck_hero_play_weight({"hero_id": "hero_dash"}, {"dash": 100}) == 100
