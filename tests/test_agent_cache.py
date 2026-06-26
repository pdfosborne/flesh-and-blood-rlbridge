"""Tests for unified agent cache fingerprints and training history."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts" / "training"))

from agent_cache import (  # noqa: E402
    AgentCacheStore,
    MatchupTrainingRecord,
    deck_content_fingerprint,
    deck_matchup_key,
)
from flesh_and_blood_rlbridge.player_observation import PLAYER_OBS_DIM  # noqa: E402
from rl_agents.ppo import PPOAgent  # noqa: E402


def test_deck_content_fingerprint_is_stable() -> None:
    deck = {"a_red": 2, "b_red": 1}
    fp1 = deck_content_fingerprint(deck, equipment_header="hero sword")
    fp2 = deck_content_fingerprint(
        {"b_red": 1, "a_red": 2},
        equipment_header="hero sword",
    )
    assert fp1 == fp2
    assert len(fp1) == 16


def test_deck_matchup_key_orders_p1_then_p2() -> None:
    assert deck_matchup_key("aaa", "bbb") == "aaa__vs__bbb"


def test_training_history_in_meta(tmp_path: Path) -> None:
    cache_root = tmp_path / "agent_cache"
    store = AgentCacheStore(cache_root, "silver_age")

    p1_fp = deck_content_fingerprint({"strike_red": 6})
    p2_fp = "opponent_precon_fp"
    matchup_key = deck_matchup_key(p1_fp, p2_fp)

    agent = PPOAgent()
    agent.obs_dim = PLAYER_OBS_DIM
    agent.n_actions = 128
    agent._init_nets(PLAYER_OBS_DIM)

    store.persist(
        agent,
        episodes_delta=1000,
        training_summary={
            "matchup_name": "briar_vs_kayo",
            "p1_fingerprint": p1_fp,
            "p2_fingerprint": p2_fp,
            "p1_hero": "briar",
            "p2_hero": "kayo",
            "episodes_completed": 1000,
            "target_episodes": 1000,
            "p1_win_rate": 0.55,
            "p2_win_rate": 0.45,
            "training_stats": {"timeouts": 2, "episodes": 1000},
        },
    )

    record = store.should_skip_training(
        p1_fingerprint=p1_fp,
        p2_fingerprint=p2_fp,
        target_episodes=1000,
    )
    assert record is not None
    assert record.matchup_key == matchup_key
    assert record.converged is True
    assert record.p1_win_rate == 0.55
    assert record.training_stats == {"timeouts": 2, "episodes": 1000}

    assert store.should_skip_training(
        p1_fingerprint=p1_fp,
        p2_fingerprint=p2_fp,
        target_episodes=2000,
    ) is None

    meta_path = cache_root / "silver_age" / "unified_agent.meta.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert len(meta["training_history"]) == 1
    assert meta["training_history"][0]["matchup_name"] == "briar_vs_kayo"
    assert MatchupTrainingRecord.from_dict(meta["training_history"][0]).p1_win_rate == 0.55


def test_load_required_reads_unified_weights(tmp_path: Path) -> None:
    cache_root = tmp_path / "agent_cache"
    store = AgentCacheStore(cache_root, "silver_age")

    agent = PPOAgent()
    agent.obs_dim = PLAYER_OBS_DIM
    agent.n_actions = 128
    agent._init_nets(PLAYER_OBS_DIM)
    store.persist(agent, episodes_delta=1)

    loaded = store.load_required(obs_dim=PLAYER_OBS_DIM, n_actions=128)
    assert loaded._actor is not None
    assert loaded.obs_dim == PLAYER_OBS_DIM
    assert loaded.n_actions == 128


def test_load_required_raises_when_missing(tmp_path: Path) -> None:
    store = AgentCacheStore(tmp_path / "agent_cache", "silver_age")
    try:
        store.load_required(obs_dim=PLAYER_OBS_DIM, n_actions=128)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as exc:
        assert "Unified agent weights not found" in str(exc)
