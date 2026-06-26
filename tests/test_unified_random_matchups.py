"""Tests for random fabrary matchup sampling and checkpoint meta fields."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts" / "training"))

from agent_cache import AgentCacheStore  # noqa: E402
from flesh_and_blood_rlbridge.player_observation import PLAYER_OBS_DIM  # noqa: E402
from rl_agents.ppo import PPOAgent  # noqa: E402
from train_dual_agent_common import (  # noqa: E402
    resolve_checkpoint_interval,
    sample_random_fabrary_matchups,
)


def _fake_deck(slug: str, hero: str) -> tuple[str, str, dict]:
    return (
        slug,
        f"fab_{slug}",
        {"id": f"fab_{slug}", "name": slug, "hero_id": hero, "format": "silver_age"},
    )


def test_sample_random_fabrary_matchups_unique_pairs() -> None:
    decks = [_fake_deck(f"deck_{i}", f"hero_{i}") for i in range(6)]
    rng = random.Random(7)
    matchups = sample_random_fabrary_matchups(decks, 4, rng, "silver_age")
    assert len(matchups) == 4
    names = {m.name for m in matchups}
    assert len(names) == 4
    for m in matchups:
        assert m.p1_deck != m.p2_deck


def test_resolve_checkpoint_interval_pct() -> None:
    assert resolve_checkpoint_interval(100, checkpoint_interval_pct=10.0) == 10
    assert resolve_checkpoint_interval(37, checkpoint_interval_pct=10.0) == 4


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
