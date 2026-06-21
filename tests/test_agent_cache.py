"""Tests for agent cache fingerprints and convergence registry."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts" / "training"))

from agent_cache import (  # noqa: E402
    AgentCacheStore,
    MatchupTrainingRecord,
    PlayerCacheContext,
    deck_content_fingerprint,
    deck_matchup_key,
)


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


def test_tier1_key_uses_content_fingerprints() -> None:
    ctx = PlayerCacheContext(
        player_deck="runtime_uuid",
        player_hero="briar",
        opponent_deck="DorintheSAGEPrecon",
        opponent_hero="dorinthea",
        player_deck_fingerprint="p1fp",
        opponent_deck_fingerprint="p2fp",
    )
    assert ctx.tier_keys()[0] == "p1fp__vs__p2fp"


def test_convergence_registry_skips_when_agent_exists(tmp_path: Path) -> None:
    cache_root = tmp_path / "agent_cache"
    store = AgentCacheStore(cache_root, "silver_age")

    p1_fp = deck_content_fingerprint({"strike_red": 6})
    p2_fp = "opponent_precon_fp"
    matchup_key = deck_matchup_key(p1_fp, p2_fp)

    ctx = PlayerCacheContext(
        player_deck="cached",
        player_hero="briar",
        opponent_deck="cached",
        opponent_hero="dorinthea",
        player_deck_fingerprint=p1_fp,
        opponent_deck_fingerprint=p2_fp,
    )
    agent = store.bootstrap_player(ctx, lambda: __import__("rl_agents.ppo", fromlist=["PPOAgent"]).PPOAgent()).agents[0]
    agent.obs_dim = 4
    agent.n_actions = 8
    agent._init_nets(4)
    store.save(1, ctx.tier_keys()[0], agent)

    store.mark_matchup_converged(
        p1_fingerprint=p1_fp,
        p2_fingerprint=p2_fp,
        p1_hero="briar",
        p2_hero="dorinthea",
        episodes_completed=1000,
        target_episodes=1000,
        p1_win_rate=0.55,
    )

    record = store.should_skip_training(
        p1_fingerprint=p1_fp,
        p2_fingerprint=p2_fp,
        target_episodes=1000,
    )
    assert record is not None
    assert record.matchup_key == matchup_key
    assert record.converged is True

    assert store.should_skip_training(
        p1_fingerprint=p1_fp,
        p2_fingerprint=p2_fp,
        target_episodes=2000,
    ) is None

    registry_path = cache_root / "silver_age" / "deck_matchup_registry.json"
    assert registry_path.is_file()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert matchup_key in payload
    assert MatchupTrainingRecord.from_dict(payload[matchup_key]).p1_win_rate == 0.55
