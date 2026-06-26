"""Tests for unified agent publish bundle and GitHub release helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fab_bridge.agents import (
    PLAYER_OBS_SCHEMA_VERSION,
    prepare_publish_bundle,
    publish_to_github_release,
    update_manifest_entry,
)
from rl_agents.ppo import PPOAgent


def _write_local_agent(cache: Path, fmt: str = "silver_age") -> None:
    fmt_dir = cache / fmt
    fmt_dir.mkdir(parents=True)
    agent = PPOAgent()
    agent.obs_dim = 16
    agent.n_actions = 4
    agent._init_nets(16)
    weights = fmt_dir / f"unified_agent_v{PLAYER_OBS_SCHEMA_VERSION}.json"
    agent.save(weights)
    meta = {
        "obs_dim": 16,
        "obs_schema_version": PLAYER_OBS_SCHEMA_VERSION,
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
        assert bundle.obs_dim == 16
        assert bundle.weights_staging_path.is_file()
        assert bundle.meta_staging_path.is_file()
        meta = json.loads(bundle.meta_staging_path.read_text(encoding="utf-8"))
        assert meta["release_id"] == "agents-2026.06.9"
        assert meta["source"] == "fab-rlbridge-official"
        assert len(bundle.sha256) == 64
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
    bundle.obs_schema_version = 1
    bundle.obs_dim = 16
    bundle.weights_asset_name = "silver_age-unified_agent_v1.json"
    bundle.meta_asset_name = "silver_age-unified_agent_v1.meta.json"
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
