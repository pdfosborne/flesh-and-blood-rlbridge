"""Tests for random fabrary matchup sampling and checkpoint meta fields."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts" / "training"))

from agent_cache import AgentCacheStore  # noqa: E402
from flesh_and_blood_rlbridge.player_observation import PLAYER_OBS_DIM  # noqa: E402
from rl_agents.ppo import PPOAgent  # noqa: E402
from train_dual_agent_common import (  # noqa: E402
    _MAX_WIN_PATH_LEN,
    build_fabrary_matchup,
    resolve_checkpoint_interval,
    resolve_fabrary_deck_cards,
    sample_random_fabrary_matchups,
    shorten_matchup_dir_name,
)


def _fake_deck(slug: str, hero: str) -> tuple[str, str, dict]:
    return (
        slug,
        f"fab_{slug}",
        {"id": f"fab_{slug}", "name": slug, "hero_id": hero, "format": "silver_age"},
    )


def test_build_fabrary_matchup_carries_fabrary_entries() -> None:
    entry1 = {"id": "fab_a", "hero_id": "hero_a", "name": "A"}
    entry2 = {"id": "fab_b", "hero_id": "hero_b", "name": "B"}
    matchup = build_fabrary_matchup("a", "fab_a", entry1, "b", "fab_b", entry2, "silver_age")
    assert matchup.p1_fabrary_entry is entry1
    assert matchup.p2_fabrary_entry is entry2


def test_sample_random_fabrary_matchups_unique_pairs() -> None:
    decks = [_fake_deck(f"deck_{i}", f"hero_{i}") for i in range(6)]
    rng = random.Random(7)
    matchups = sample_random_fabrary_matchups(decks, 4, rng, "silver_age")
    assert len(matchups) == 4
    names = {m.name for m in matchups}
    assert len(names) == 4
    for m in matchups:
        assert m.p1_deck != m.p2_deck
        assert len(m.dir_name) <= 48
        assert m.p1_fabrary_entry is not None
        assert m.p2_fabrary_entry is not None


def test_sample_random_fabrary_matchups_fabrary_weighted_prefers_heavy_hero() -> None:
    decks = [
        _fake_deck("heavy", "hero_dash"),
        _fake_deck("light_a", "hero_gravy_bones"),
        _fake_deck("light_b", "hero_kayo"),
    ]
    hero_counts = {"dash": 1000, "gravy-bones": 1, "kayo": 1}

    def _counts(*_args, **_kwargs):
        return hero_counts

    rng = random.Random(0)
    with patch(
        "flesh_and_blood_rlbridge.card_db.fabrary_meta.hero_play_counts",
        side_effect=_counts,
    ):
        matchups = sample_random_fabrary_matchups(
            decks,
            200,
            rng,
            "silver_age",
            unique_pairs=False,
            fabrary_weighted_heroes=True,
        )
    heavy_in_matchup = sum(
        1
        for m in matchups
        if m.p1_deck == "fab_heavy" or m.p2_deck == "fab_heavy"
    )
    assert heavy_in_matchup > 150


def test_resolve_checkpoint_interval_pct() -> None:
    assert resolve_checkpoint_interval(100, checkpoint_interval_pct=10.0) == 10
    assert resolve_checkpoint_interval(37, checkpoint_interval_pct=10.0) == 4


def test_long_fabrary_matchup_dir_stays_within_windows_path_limit() -> None:
    long_name = (
        "dorinthea_sage_aggro-vs-precon_sage_ch2_arakni_web_of_deceit"
    )
    p1 = "fab_dorinthea_sage_aggro"
    p2 = "fab_precon_sage_ch2_arakni_web_of_deceit"
    short_dir = shorten_matchup_dir_name(long_name, p1, p2)
    assert len(short_dir) <= 48
    assert short_dir != long_name

    run_dir = (
        Path("C:/Users/phili/Documents/RL/flesh-and-blood-rlbridge")
        / "results/unified_random_matchups/silver_age/20260626_154553"
    )
    package = (
        run_dir
        / short_dir
        / "ppo_p1-12345678"
        / "weights"
        / "agent_weights.json"
    )
    assert len(str(package)) <= _MAX_WIN_PATH_LEN


def test_training_history_stores_checkpoint_win_rates(tmp_path: Path) -> None:
    store = AgentCacheStore(tmp_path / "cache", "silver_age")
    agent = PPOAgent()
    agent.obs_dim = PLAYER_OBS_DIM
    agent.n_actions = 128
    agent._init_nets(PLAYER_OBS_DIM)

    store.persist(
        agent,
        training_summary={
            "matchup_name": "a-vs-b",
            "p1_fingerprint": "aaa",
            "p2_fingerprint": "bbb",
            "p1_hero": "a",
            "p2_hero": "b",
            "episodes_completed": 100,
            "target_episodes": 100,
            "p1_win_rate": 0.52,
            "first_checkpoint_win_rate": 0.41,
            "final_checkpoint_win_rate": 0.55,
        },
    )

    meta = json.loads((store.cache_root / "unified_agent.meta.json").read_text())
    row = meta["training_history"][0]
    assert row["first_checkpoint_win_rate"] == 0.41
    assert row["final_checkpoint_win_rate"] == 0.55
    assert row["checkpoint_eval_win_rate"] == 0.55

    record = store.training_history()[0]
    assert record.first_checkpoint_win_rate == 0.41
    assert record.final_checkpoint_win_rate == 0.55


def test_build_assets_hero_map_skips_empty_txt_files(tmp_path: Path) -> None:
    from train_dual_agent_common import _build_assets_hero_map  # noqa: PLC0415

    assets = tmp_path / "Assets"
    assets.mkdir()
    (assets / "empty.txt").write_text("", encoding="utf-8")
    (assets / "valid.txt").write_text("aurora AuroraText\naurora_red", encoding="utf-8")

    hero_map = _build_assets_hero_map(assets)
    assert "aurora" in hero_map
    assert hero_map["aurora"] == "aurora AuroraText"


def test_clone_agent_weights_requires_initialized_source() -> None:
    from agent_cache import clone_agent_weights  # noqa: E402

    src = PPOAgent()
    dst = PPOAgent()
    try:
        clone_agent_weights(src, dst)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "no initialized networks" in str(exc).lower()


def test_sample_policy_action_index_respects_legal_mask() -> None:
    import numpy as np

    from train_dual_agent_common import _sample_policy_action_index  # noqa: E402

    policy = PPOAgent(n_actions=8, obs_dim=PLAYER_OBS_DIM)
    policy._init_nets(PLAYER_OBS_DIM)
    rng = np.random.default_rng(0)
    obs = np.zeros(PLAYER_OBS_DIM, dtype=np.float64)
    action = _sample_policy_action_index(policy, obs, n_legal=3, rng=rng)
    assert 0 <= action < 3


def test_resolve_fabrary_deck_cards_uses_card_ids_field() -> None:
    deck_entry = {
        "id": "fab_test_import",
        "card_ids": [
            {"id": "wtr001", "count": 2},
            {"id": "wtr002", "count": 1},
        ],
        "cards": [],
    }
    result = resolve_fabrary_deck_cards(deck_entry, "silver_age")
    assert result == ["wtr001", "wtr001", "wtr002"]
