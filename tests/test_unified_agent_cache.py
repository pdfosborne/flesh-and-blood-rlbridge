"""Tests for unified AgentCacheStore."""

from __future__ import annotations

import sys
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts" / "training"))
sys.path.insert(0, str(_REPO / "src"))

from agent_cache import AgentCacheStore, deck_content_fingerprint  # noqa: E402
from flesh_and_blood_rlbridge.player_observation import PLAYER_OBS_DIM  # noqa: E402
from rl_agents.ppo import ARCHITECTURE, PPOAgent, UNIFIED_AGENT_WEIGHT_VERSION  # noqa: E402


def _make_agent() -> PPOAgent:
    return PPOAgent()


def test_unified_store_save_load_round_trip(tmp_path: Path) -> None:
    store = AgentCacheStore(tmp_path / "cache", "silver_age")
    agent, src = store.load_or_create(
        _make_agent,
        obs_dim=PLAYER_OBS_DIM,
        n_actions=128,
        mask_actions=True,
    )
    assert src == "fresh"
    agent._init_nets(PLAYER_OBS_DIM)

    store.persist(agent, episodes_delta=10, training_summary={
        "matchup_name": "a_vs_b",
        "p1_fingerprint": "aaa",
        "p2_fingerprint": "bbb",
        "p1_hero": "a",
        "p2_hero": "b",
        "episodes_completed": 10,
        "target_episodes": 10,
    })
    reloaded = store.load_if_exists()
    assert reloaded is not None
    assert reloaded.obs_dim == PLAYER_OBS_DIM

    meta = store.load_meta()
    assert meta["obs_schema_version"] == 2
    assert meta["weight_version"] == UNIFIED_AGENT_WEIGHT_VERSION
    assert meta["architecture"] == ARCHITECTURE
    assert meta["total_episodes_trained"] == 10
    assert meta["last_matchup"] == "a_vs_b"
    assert len(meta["training_history"]) == 1


def test_unified_store_rejects_schema_mismatch(tmp_path: Path) -> None:
    store = AgentCacheStore(tmp_path / "cache", "silver_age", obs_schema_version=1)
    agent, _ = store.load_or_create(
        _make_agent,
        obs_dim=PLAYER_OBS_DIM,
        n_actions=128,
        mask_actions=True,
    )
    agent._init_nets(PLAYER_OBS_DIM)
    store.persist(agent)

    store2 = AgentCacheStore(tmp_path / "cache", "silver_age", obs_schema_version=99)
    agent2, src = store2.load_or_create(
        _make_agent,
        obs_dim=PLAYER_OBS_DIM,
        n_actions=128,
        mask_actions=True,
    )
    assert src == "fresh"
    assert agent2._actor is not None


def test_coverage_lookup_requires_weights(tmp_path: Path) -> None:
    store = AgentCacheStore(tmp_path / "cache", "silver_age")
    p1_fp = deck_content_fingerprint({"strike_red": 6})
    p2_fp = "opponent_fp"

    store.record_matchup_training(
        p1_fingerprint=p1_fp,
        p2_fingerprint=p2_fp,
        p1_hero="briar",
        p2_hero="kayo",
        episodes_completed=100,
        target_episodes=100,
        p1_win_rate=0.52,
    )
    assert store.should_skip_training(
        p1_fingerprint=p1_fp,
        p2_fingerprint=p2_fp,
        target_episodes=100,
    ) is None

    agent, _ = store.load_or_create(
        _make_agent,
        obs_dim=PLAYER_OBS_DIM,
        n_actions=128,
        mask_actions=True,
    )
    agent._init_nets(PLAYER_OBS_DIM)
    store.persist(agent)

    assert store.should_skip_training(
        p1_fingerprint=p1_fp,
        p2_fingerprint=p2_fp,
        target_episodes=100,
    ) is not None


def test_load_if_exists_rejects_legacy_mlp_weights(tmp_path: Path) -> None:
    store = AgentCacheStore(tmp_path / "cache", "silver_age")
    legacy = {
        "agent": "ppo",
        "n_actions": 128,
        "obs_dim": PLAYER_OBS_DIM,
        "hidden_size": 64,
        "actor_weights": {"W1": [[0.0]]},
        "critic_weights": {"W1": [[0.0]]},
    }
    path = store.cache_root / f"unified_agent_v{UNIFIED_AGENT_WEIGHT_VERSION}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert store.load_if_exists() is None
