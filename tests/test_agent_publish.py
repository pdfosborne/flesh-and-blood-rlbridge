"""Tests for unified agent publish bundle and GitHub release helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fab_bridge.agents import (
    prepare_publish_bundle,
    publish_to_github_release,
    update_manifest_entry,
)
from flesh_and_blood_rlbridge.player_observation import PLAYER_OBS_DIM, PLAYER_OBS_SCHEMA_VERSION
from rl_agents.ppo import ARCHITECTURE, PPOAgent, UNIFIED_AGENT_WEIGHT_VERSION


def _write_local_agent(cache: Path, fmt: str = "silver_age") -> None:
    fmt_dir = cache / fmt
    fmt_dir.mkdir(parents=True)
    agent = PPOAgent()
    agent.n_actions = 128
    agent.obs_dim = PLAYER_OBS_DIM
    agent._init_nets(PLAYER_OBS_DIM)
    weights = fmt_dir / f"unified_agent_v{UNIFIED_AGENT_WEIGHT_VERSION}.json"
    agent.save(weights)
    meta = {
        "obs_dim": PLAYER_OBS_DIM,
        "obs_schema_version": PLAYER_OBS_SCHEMA_VERSION,
        "architecture": ARCHITECTURE,
        "weight_version": UNIFIED_AGENT_WEIGHT_VERSION,
        "total_episodes_trained": 500,
        "game_format": fmt,
    }
    (fmt_dir / "unified_agent.meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_prepare_publish_bundle(tmp_path: Path) -> None:
    cache = tmp_path / "agent_cache"
    _write_local_agent(cache)
    bundle = prepare_publish_bundle("silver_age", cache_dir=cache, release_id="agents-2026.06.9")
    try:
        assert bundle.format == "silver_age"
        assert bundle.obs_dim == PLAYER_OBS_DIM
        assert bundle.weights_staging_path.is_file()
        assert bundle.meta_staging_path.is_file()
        meta = json.loads(bundle.meta_staging_path.read_text(encoding="utf-8"))
        assert meta["release_id"] == "agents-2026.06.9"
        assert meta["source"] == "fab-rlbridge-official"
        assert meta["architecture"] == ARCHITECTURE
        assert meta["weight_version"] == UNIFIED_AGENT_WEIGHT_VERSION
        assert meta["text_embed_version"] == "v1"
        assert bundle.text_embed_staging_path is not None
        assert bundle.text_embed_staging_path.is_file()
    finally:
        from fab_bridge.agents import cleanup_publish_bundle

        cleanup_publish_bundle(bundle)


def test_prepare_publish_bundle_missing_weights(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prepare_publish_bundle("silver_age", cache_dir=tmp_path / "empty")


def test_update_manifest_entry_replaces_format() -> None:
    manifest = {"agents": [{"format": "silver_age", "release": "old"}]}
    bundle = MagicMock()
    bundle.format = "silver_age"
    bundle.release_id = "agents-2026.06.2"
    bundle.obs_schema_version = PLAYER_OBS_SCHEMA_VERSION
    bundle.obs_dim = PLAYER_OBS_DIM
    bundle.weights_asset_name = f"silver_age-unified_agent_v{UNIFIED_AGENT_WEIGHT_VERSION}.json"
    bundle.meta_asset_name = f"silver_age-unified_agent_v{UNIFIED_AGENT_WEIGHT_VERSION}.meta.json"
    bundle.sha256 = "abc"
    urls = {
        "weights_url": "https://example/w.json",
        "meta_url": "https://example/m.json",
        "repository": "org/repo",
    }
    updated = update_manifest_entry(manifest, bundle, urls)
    assert len(updated["agents"]) == 1
    assert updated["agents"][0]["release"] == "agents-2026.06.2"


def test_publish_to_github_release_invokes_gh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "agent_cache"
    _write_local_agent(cache)
    bundle = prepare_publish_bundle("silver_age", cache_dir=cache, release_id="agents-test-1")
    try:
        monkeypatch.setattr(
            "fab_bridge.agents.gh_auth_ok",
            lambda: (True, "ok"),
        )
        monkeypatch.setattr(
            "fab_bridge.agents.shutil.which",
            lambda name: "gh" if name == "gh" else None,
        )

        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""
        monkeypatch.setattr("fab_bridge.agents.subprocess.run", lambda *a, **k: proc)

        urls = publish_to_github_release(
            bundle,
            notes="test",
            repository="pdfosborne/flesh-and-blood-rlbridge",
        )
        assert "releases/download/agents-test-1" in urls["weights_url"]
        assert urls["meta_url"].endswith(".meta.json")
    finally:
        from fab_bridge.agents import cleanup_publish_bundle

        cleanup_publish_bundle(bundle)
