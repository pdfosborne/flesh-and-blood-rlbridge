"""Tests for unified agent manifest sync and path helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fab_bridge.agents import (
    PLAYER_OBS_SCHEMA_VERSION,
    agent_status,
    list_local_agents,
    load_manifest,
    release_asset_names,
    save_manifest,
    suggest_next_release_tag,
    sync_agents,
    unified_agent_cache_format,
    unified_agent_weights_path,
)
from flesh_and_blood_rlbridge.player_observation import PLAYER_OBS_DIM
from rl_agents.ppo import PPOAgent, UNIFIED_AGENT_WEIGHT_VERSION


def _init_agent(agent: PPOAgent, *, n_actions: int = 128) -> None:
    agent.n_actions = n_actions
    agent.obs_dim = PLAYER_OBS_DIM
    agent._init_nets(PLAYER_OBS_DIM)


def test_unified_agent_cache_format_maps_sage() -> None:
    assert unified_agent_cache_format("sage") == "silver_age"
    assert unified_agent_cache_format("silver_age") == "silver_age"


def test_unified_agent_weights_path(tmp_path: Path) -> None:
    path = unified_agent_weights_path(tmp_path / "cache", "silver_age")
    assert path.name == f"unified_agent_v{UNIFIED_AGENT_WEIGHT_VERSION}.json"
    assert path.parent.name == "silver_age"


def test_release_asset_names() -> None:
    weights, meta = release_asset_names("silver_age")
    assert weights == f"silver_age-unified_agent_v{UNIFIED_AGENT_WEIGHT_VERSION}.json"
    assert meta == f"silver_age-unified_agent_v{UNIFIED_AGENT_WEIGHT_VERSION}.meta.json"


def test_list_local_agents_empty(tmp_path: Path) -> None:
    assert list_local_agents(tmp_path / "missing") == []


def test_list_local_agents_finds_weights(tmp_path: Path) -> None:
    cache = tmp_path / "agent_cache"
    fmt_dir = cache / "silver_age"
    fmt_dir.mkdir(parents=True)
    agent = PPOAgent()
    _init_agent(agent, n_actions=128)
    weights = fmt_dir / f"unified_agent_v{UNIFIED_AGENT_WEIGHT_VERSION}.json"
    agent.save(weights)
    meta = {
        "obs_dim": PLAYER_OBS_DIM,
        "obs_schema_version": PLAYER_OBS_SCHEMA_VERSION,
        "architecture": "attention_v1",
        "weight_version": UNIFIED_AGENT_WEIGHT_VERSION,
        "total_episodes_trained": 42,
        "last_updated": "2026-01-01T00:00:00+00:00",
    }
    (fmt_dir / "unified_agent.meta.json").write_text(json.dumps(meta), encoding="utf-8")

    found = list_local_agents(cache)
    assert len(found) == 1
    assert found[0].format == "silver_age"
    assert found[0].total_episodes_trained == 42


def test_agent_status_missing(tmp_path: Path) -> None:
    status = agent_status(tmp_path / "cache", "silver_age")
    assert status["exists"] is False
    assert status["cache_format"] == "silver_age"


def test_load_and_save_manifest_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    data = load_manifest(path)
    assert data["agents"] == []
    data["agents"].append({"format": "silver_age", "release": "agents-2026.06.1"})
    save_manifest(data, path)
    loaded = load_manifest(path)
    assert loaded["agents"][0]["format"] == "silver_age"


def test_suggest_next_release_tag_increments_patch() -> None:
    manifest = {
        "agents": [
            {"release": "agents-2026.06.1"},
            {"release": "agents-2026.06.3"},
        ]
    }
    tag = suggest_next_release_tag(manifest)
    assert tag.startswith("agents-2026.")
    assert tag.endswith(".4") or tag.endswith(".1")


def test_summarize_public_agent_sync_states(tmp_path: Path) -> None:
    from fab_bridge.agents import summarize_public_agent_sync

    cache = tmp_path / "cache"
    fmt_dir = cache / "silver_age"
    fmt_dir.mkdir(parents=True)
    agent = PPOAgent()
    _init_agent(agent, n_actions=128)
    weights = unified_agent_weights_path(cache, "silver_age")
    agent.save(weights)
    from fab_bridge import agents as agents_mod

    sha = agents_mod._sha256_file(weights)
    manifest = {
        "agents": [
            {
                "format": "silver_age",
                "release": "agents-2026.06.1",
                "sha256": sha,
            }
        ]
    }
    rows = summarize_public_agent_sync(cache_dir=cache, manifest=manifest)
    assert rows[0]["state"] == "up to date"

    manifest["agents"][0]["release"] = "agents-2026.06.2"
    manifest["agents"][0]["sha256"] = "0" * 64
    rows = summarize_public_agent_sync(cache_dir=cache, manifest=manifest)
    assert rows[0]["state"] == "outdated"


def test_sync_agents_downloads_and_installs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "cache"
    fmt_dir = cache / "silver_age"
    fmt_dir.mkdir(parents=True)
    agent = PPOAgent()
    _init_agent(agent, n_actions=128)
    dest = unified_agent_weights_path(cache, "silver_age")
    agent.save(dest)
    from fab_bridge import agents as agents_mod

    sha = agents_mod._sha256_file(dest)
    manifest_path = tmp_path / "manifest.json"
    save_manifest(
        {
            "agents": [
                {
                    "format": "silver_age",
                    "weights_url": "http://example/w.json",
                    "sha256": sha,
                }
            ]
        },
        manifest_path,
    )

    def fail_get(*_a: object, **_k: object) -> None:
        raise AssertionError("should not download")

    monkeypatch.setattr(agents_mod.requests, "get", fail_get)
    results = sync_agents(manifest_url=str(manifest_path), cache_dir=cache)
    assert results[0].action == "unchanged"
