"""Public unified-agent manifest, sync, and GitHub release publishing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from fab_bridge.paths import configure_import_paths, repo_root

configure_import_paths()

from flesh_and_blood_rlbridge.player_observation import PLAYER_OBS_SCHEMA_VERSION  # noqa: E402

DEFAULT_REPOSITORY = "pdfosborne/flesh-and-blood-rlbridge"
MANIFEST_REL_PATH = Path("agents") / "manifest.json"
UNIFIED_META_FILENAME = "unified_agent.meta.json"
SUPPORTED_FORMATS = ("silver_age", "classic_constructed", "blitz", "upf")
OFFICIAL_SOURCE = "fab-rlbridge-official"


@dataclass
class LocalAgentInfo:
    format: str
    weights_path: Path
    meta_path: Path
    obs_schema_version: int
    obs_dim: int
    total_episodes_trained: int
    last_updated: str
    release_id: str


@dataclass
class PublishBundle:
    format: str
    release_id: str
    obs_schema_version: int
    obs_dim: int
    weights_asset_name: str
    meta_asset_name: str
    weights_staging_path: Path
    meta_staging_path: Path
    sha256: str
    meta: dict[str, Any]
    staging_dir: Path = field(repr=False)


@dataclass
class SyncResult:
    format: str
    action: str
    detail: str


def agent_cache_dir() -> Path:
    override = os.environ.get("FAB_AGENT_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "results" / "agent_cache"


def manifest_path() -> Path:
    return repo_root() / MANIFEST_REL_PATH


def default_manifest_url() -> str:
    custom = os.environ.get("FAB_AGENTS_MANIFEST_URL", "").strip()
    if custom:
        return custom
    local = manifest_path()
    if local.is_file():
        return local.as_uri()
    return (
        f"https://raw.githubusercontent.com/{DEFAULT_REPOSITORY}/main/"
        f"{MANIFEST_REL_PATH.as_posix()}"
    )


def unified_agent_cache_format(game_format: str) -> str:
    from fab_tui.config import normalize_pipeline_format  # noqa: PLC0415

    return normalize_pipeline_format(game_format)


def unified_agent_weights_path(cache_dir: Path, game_format: str) -> Path:
    cache_format = unified_agent_cache_format(game_format)
    store_root = Path(cache_dir) / cache_format
    return store_root / f"unified_agent_v{PLAYER_OBS_SCHEMA_VERSION}.json"


def unified_agent_meta_path(cache_dir: Path, game_format: str) -> Path:
    cache_format = unified_agent_cache_format(game_format)
    return Path(cache_dir) / cache_format / UNIFIED_META_FILENAME


def release_asset_names(game_format: str, obs_schema_version: int) -> tuple[str, str]:
    fmt = unified_agent_cache_format(game_format)
    weights = f"{fmt}-unified_agent_v{obs_schema_version}.json"
    meta = f"{fmt}-unified_agent_v{obs_schema_version}.meta.json"
    return weights, meta


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_file(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return raw


def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    if path is None:
        path = manifest_path()
    source = str(path)
    if source.startswith("file://"):
        from urllib.request import url2pathname
        from urllib.parse import urlparse

        parsed = urlparse(source)
        file_path = Path(url2pathname(parsed.path))
        if not file_path.is_file():
            return _empty_manifest()
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    elif source.startswith(("http://", "https://")):
        response = requests.get(source, timeout=60)
        response.raise_for_status()
        raw = response.json()
    else:
        file_path = Path(path)
        if not file_path.is_file():
            return _empty_manifest()
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Agent manifest must be a JSON object")
    raw.setdefault("manifest_version", 1)
    raw.setdefault("obs_schema_version", PLAYER_OBS_SCHEMA_VERSION)
    raw.setdefault("repository", DEFAULT_REPOSITORY)
    raw.setdefault("default_format", "silver_age")
    raw.setdefault("agents", [])
    return raw


def _empty_manifest() -> dict[str, Any]:
    return {
        "manifest_version": 1,
        "obs_schema_version": PLAYER_OBS_SCHEMA_VERSION,
        "repository": DEFAULT_REPOSITORY,
        "default_format": "silver_age",
        "agents": [],
    }


def save_manifest(data: dict[str, Any], path: Path | None = None) -> Path:
    target = path or manifest_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return target


def _read_meta(meta_path: Path) -> dict[str, Any]:
    if not meta_path.is_file():
        return {}
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def list_local_agents(cache_dir: Path | None = None) -> list[LocalAgentInfo]:
    root = Path(cache_dir or agent_cache_dir())
    found: list[LocalAgentInfo] = []
    if not root.is_dir():
        return found
    for fmt_dir in sorted(root.iterdir()):
        if not fmt_dir.is_dir():
            continue
        weights = fmt_dir / f"unified_agent_v{PLAYER_OBS_SCHEMA_VERSION}.json"
        if not weights.is_file():
            continue
        meta_path = fmt_dir / UNIFIED_META_FILENAME
        meta = _read_meta(meta_path)
        try:
            agent_data = _load_json_file(weights)
            obs_dim = int(meta.get("obs_dim", agent_data.get("obs_dim", 0)))
        except (ValueError, KeyError, TypeError):
            obs_dim = int(meta.get("obs_dim", 0))
        found.append(
            LocalAgentInfo(
                format=fmt_dir.name,
                weights_path=weights,
                meta_path=meta_path,
                obs_schema_version=int(
                    meta.get("obs_schema_version", PLAYER_OBS_SCHEMA_VERSION)
                ),
                obs_dim=obs_dim,
                total_episodes_trained=int(meta.get("total_episodes_trained", 0)),
                last_updated=str(meta.get("last_updated", "")),
                release_id=str(meta.get("release_id", "")),
            )
        )
    return found


def agent_status(cache_dir: Path | None, game_format: str) -> dict[str, Any]:
    cache = Path(cache_dir or agent_cache_dir())
    cache_format = unified_agent_cache_format(game_format)
    weights_path = unified_agent_weights_path(cache, game_format)
    meta_path = unified_agent_meta_path(cache, game_format)
    meta = _read_meta(meta_path)
    exists = weights_path.is_file()
    return {
        "format": game_format,
        "cache_format": cache_format,
        "exists": exists,
        "weights_path": str(weights_path),
        "meta_path": str(meta_path),
        "obs_schema_version": int(
            meta.get("obs_schema_version", PLAYER_OBS_SCHEMA_VERSION)
        ),
        "obs_dim": int(meta.get("obs_dim", 0)) if meta else None,
        "release_id": str(meta.get("release_id", "")) if meta else "",
        "source": str(meta.get("source", "")) if meta else "",
        "total_episodes_trained": int(meta.get("total_episodes_trained", 0)) if meta else 0,
        "last_updated": str(meta.get("last_updated", "")) if meta else "",
        "sha256": str(meta.get("sha256", "")) if meta else "",
    }


def _manifest_entry_for_format(manifest: dict[str, Any], game_format: str) -> Optional[dict[str, Any]]:
    fmt = unified_agent_cache_format(game_format)
    for row in manifest.get("agents", []):
        if isinstance(row, dict) and str(row.get("format", "")) == fmt:
            return row
    return None


def summarize_public_agent_sync(
    *,
    cache_dir: Path | None = None,
    manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compare local cache against manifest entries for sync UI."""
    data = manifest if manifest is not None else load_manifest()
    cache = Path(cache_dir or agent_cache_dir())
    rows: list[dict[str, Any]] = []
    for entry in data.get("agents", []):
        if not isinstance(entry, dict):
            continue
        fmt = str(entry.get("format", ""))
        if not fmt:
            continue
        local = agent_status(cache, fmt)
        public_release = str(entry.get("release", ""))
        public_sha = str(entry.get("sha256", "")).lower()
        weights_path = Path(str(local.get("weights_path", "")))
        file_sha = _sha256_file(weights_path).lower() if weights_path.is_file() else ""
        local_sha = str(local.get("sha256", "")).lower() or file_sha
        local_release = str(local.get("release_id", ""))
        if not local.get("exists"):
            state = "missing"
        elif public_release and local_release and public_release != local_release:
            state = "outdated"
        elif public_sha and local_sha and public_sha == local_sha:
            state = "up to date"
        else:
            state = "outdated"
        rows.append(
            {
                "format": fmt,
                "public_release": public_release,
                "local_release": local_release,
                "state": state,
                "weights_path": str(local.get("weights_path", "")),
            }
        )
    return rows


def _download_file(url: str, dest: Path) -> None:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)


def sync_agents(
    *,
    manifest_url: str | None = None,
    cache_dir: Path | None = None,
    formats: list[str] | None = None,
    force: bool = False,
) -> list[SyncResult]:
    manifest = load_manifest(manifest_url or default_manifest_url())
    cache = Path(cache_dir or agent_cache_dir())
    cache.mkdir(parents=True, exist_ok=True)
    results: list[SyncResult] = []

    entries = manifest.get("agents", [])
    if formats:
        wanted = {unified_agent_cache_format(f) for f in formats}
        entries = [
            row for row in entries if isinstance(row, dict) and row.get("format") in wanted
        ]
    elif not entries:
        default_fmt = str(manifest.get("default_format", "silver_age"))
        results.append(
            SyncResult(
                format=default_fmt,
                action="skipped",
                detail="Manifest has no agent entries to sync",
            )
        )
        return results

    for row in entries:
        if not isinstance(row, dict):
            continue
        fmt = str(row.get("format", ""))
        weights_url = str(row.get("weights_url", "")).strip()
        meta_url = str(row.get("meta_url", "")).strip()
        expected_sha = str(row.get("sha256", "")).strip().lower()
        if not fmt or not weights_url:
            results.append(
                SyncResult(format=fmt or "?", action="skipped", detail="Incomplete manifest row")
            )
            continue

        dest_weights = cache / fmt / f"unified_agent_v{PLAYER_OBS_SCHEMA_VERSION}.json"
        dest_meta = cache / fmt / UNIFIED_META_FILENAME

        if dest_weights.is_file() and not force:
            local_sha = _sha256_file(dest_weights)
            if expected_sha and local_sha == expected_sha:
                results.append(
                    SyncResult(
                        format=fmt,
                        action="unchanged",
                        detail=f"Already synced ({dest_weights})",
                    )
                )
                continue

        _download_file(weights_url, dest_weights)
        actual_sha = _sha256_file(dest_weights)
        if expected_sha and actual_sha != expected_sha:
            dest_weights.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA256 mismatch for {fmt}: expected {expected_sha}, got {actual_sha}"
            )
        if meta_url:
            _download_file(meta_url, dest_meta)
            meta = _read_meta(dest_meta)
        elif dest_meta.is_file():
            meta = _read_meta(dest_meta)
        else:
            meta = {}
        meta.setdefault("obs_schema_version", row.get("obs_schema_version", PLAYER_OBS_SCHEMA_VERSION))
        meta.setdefault("obs_dim", row.get("obs_dim"))
        meta.setdefault("game_format", fmt)
        meta["release_id"] = str(row.get("release", ""))
        meta["source"] = OFFICIAL_SOURCE
        meta["sha256"] = actual_sha
        meta["synced_at"] = datetime.now(timezone.utc).isoformat()
        dest_meta.parent.mkdir(parents=True, exist_ok=True)
        dest_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        results.append(
            SyncResult(
                format=fmt,
                action="downloaded",
                detail=f"Installed to {dest_weights}",
            )
        )
    return results


def suggest_next_release_tag(manifest: dict[str, Any] | None = None) -> str:
    data = manifest or load_manifest()
    prefix = datetime.now(timezone.utc).strftime("agents-%Y.%m")
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)$")
    max_patch = 0
    for row in data.get("agents", []):
        if not isinstance(row, dict):
            continue
        match = pattern.match(str(row.get("release", "")))
        if match:
            max_patch = max(max_patch, int(match.group(1)))
    return f"{prefix}.{max_patch + 1}"


def prepare_publish_bundle(
    game_format: str,
    cache_dir: Path | None = None,
    release_id: str = "",
) -> PublishBundle:
    cache = Path(cache_dir or agent_cache_dir())
    fmt = unified_agent_cache_format(game_format)
    weights_path = unified_agent_weights_path(cache, game_format)
    meta_path = unified_agent_meta_path(cache, game_format)
    if not weights_path.is_file():
        raise FileNotFoundError(f"No unified agent weights at {weights_path}")

    meta = _read_meta(meta_path)
    agent_data = _load_json_file(weights_path)
    obs_dim = int(meta.get("obs_dim", agent_data.get("obs_dim", 0)))
    if obs_dim <= 0:
        raise ValueError(f"Cannot publish {fmt}: obs_dim missing from weights/meta")

    sha256 = _sha256_file(weights_path)
    tag = release_id or suggest_next_release_tag()
    weights_asset, meta_asset = release_asset_names(fmt, PLAYER_OBS_SCHEMA_VERSION)

    staging_dir = Path(tempfile.mkdtemp(prefix="fab-agent-publish-"))
    weights_staging = staging_dir / weights_asset
    meta_staging = staging_dir / meta_asset
    shutil.copy2(weights_path, weights_staging)

    enriched = dict(meta)
    enriched.update(
        {
            "obs_schema_version": PLAYER_OBS_SCHEMA_VERSION,
            "obs_dim": obs_dim,
            "n_actions": int(agent_data.get("n_actions", meta.get("n_actions", 0))),
            "game_format": fmt,
            "release_id": tag,
            "source": OFFICIAL_SOURCE,
            "sha256": sha256,
            "published_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    meta_staging.write_text(json.dumps(enriched, indent=2) + "\n", encoding="utf-8")

    return PublishBundle(
        format=fmt,
        release_id=tag,
        obs_schema_version=PLAYER_OBS_SCHEMA_VERSION,
        obs_dim=obs_dim,
        weights_asset_name=weights_asset,
        meta_asset_name=meta_asset,
        weights_staging_path=weights_staging,
        meta_staging_path=meta_staging,
        sha256=sha256,
        meta=enriched,
        staging_dir=staging_dir,
    )


def gh_auth_ok() -> tuple[bool, str]:
    gh = shutil.which("gh")
    if not gh:
        return False, "GitHub CLI (gh) not found on PATH"
    proc = subprocess.run(  # noqa: S603
        [gh, "auth", "status"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, detail[0] if detail else "gh auth status failed"
    return True, "authenticated"


def _resolve_repository(manifest: dict[str, Any]) -> str:
    repo = str(manifest.get("repository", "")).strip()
    if repo:
        return repo
    gh = shutil.which("gh")
    if gh:
        proc = subprocess.run(  # noqa: S603
            [gh, "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    return DEFAULT_REPOSITORY


def publish_to_github_release(
    bundle: PublishBundle,
    *,
    notes: str = "",
    repository: str = "",
    draft: bool = False,
    manifest: dict[str, Any] | None = None,
) -> dict[str, str]:
    ok, detail = gh_auth_ok()
    if not ok:
        raise RuntimeError(detail)

    gh = shutil.which("gh")
    assert gh is not None

    manifest_data = manifest or load_manifest()
    repo = repository or _resolve_repository(manifest_data)
    title = f"Official unified agents — {bundle.release_id}"
    body = notes or (
        f"Published unified agent for format `{bundle.format}` "
        f"(obs_schema v{bundle.obs_schema_version}, obs_dim={bundle.obs_dim})."
    )

    cmd = [
        gh,
        "release",
        "create",
        bundle.release_id,
        str(bundle.weights_staging_path),
        str(bundle.meta_staging_path),
        "--repo",
        repo,
        "--title",
        title,
        "--notes",
        body,
    ]
    if draft:
        cmd.append("--draft")

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"gh release create failed: {err}")

    base = f"https://github.com/{repo}/releases/download/{bundle.release_id}"
    return {
        "weights_url": f"{base}/{bundle.weights_asset_name}",
        "meta_url": f"{base}/{bundle.meta_asset_name}",
        "repository": repo,
        "release_id": bundle.release_id,
    }


def update_manifest_entry(
    manifest: dict[str, Any],
    bundle: PublishBundle,
    urls: dict[str, str],
    *,
    min_package_version: str = "0.1.0",
) -> dict[str, Any]:
    entry = {
        "format": bundle.format,
        "release": bundle.release_id,
        "obs_schema_version": bundle.obs_schema_version,
        "obs_dim": bundle.obs_dim,
        "weights_filename": bundle.weights_asset_name,
        "meta_filename": bundle.meta_asset_name,
        "weights_url": urls["weights_url"],
        "meta_url": urls["meta_url"],
        "sha256": bundle.sha256,
        "min_package_version": min_package_version,
    }
    agents = [row for row in manifest.get("agents", []) if row.get("format") != bundle.format]
    agents.append(entry)
    manifest["agents"] = agents
    manifest["repository"] = urls.get("repository", manifest.get("repository", DEFAULT_REPOSITORY))
    return manifest


def cleanup_publish_bundle(bundle: PublishBundle) -> None:
    shutil.rmtree(bundle.staging_dir, ignore_errors=True)


def publish_local_agent(
    game_format: str,
    *,
    release_id: str = "",
    notes: str = "",
    draft: bool = False,
    cache_dir: Path | None = None,
    update_local_cache_meta: bool = True,
) -> tuple[dict[str, Any], PublishBundle]:
    bundle = prepare_publish_bundle(game_format, cache_dir=cache_dir, release_id=release_id)
    manifest = load_manifest()
    try:
        urls = publish_to_github_release(
            bundle,
            notes=notes,
            draft=draft,
            manifest=manifest,
        )
        manifest = update_manifest_entry(manifest, bundle, urls)
        save_manifest(manifest)

        if update_local_cache_meta:
            meta_path = unified_agent_meta_path(cache_dir or agent_cache_dir(), game_format)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(bundle.meta, indent=2) + "\n", encoding="utf-8")

        return manifest, bundle
    finally:
        cleanup_publish_bundle(bundle)
