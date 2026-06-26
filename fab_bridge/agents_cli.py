"""CLI handlers for ``fab-bridge agents``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fab_bridge.agents import (
    SUPPORTED_FORMATS,
    agent_cache_dir,
    agent_status,
    default_manifest_url,
    ensure_agents_available,
    load_manifest,
    publish_local_agent,
    sync_agents,
)


def run_agents_command(args: argparse.Namespace) -> int:
    if args.agents_command == "sync":
        return _run_sync(args)
    if args.agents_command == "ensure":
        return _run_ensure(args)
    if args.agents_command == "status":
        return _run_status(args)
    if args.agents_command == "publish":
        return _run_publish(args)
    print(f"Unknown agents command: {args.agents_command}", file=sys.stderr)
    return 1


def _run_sync(args: argparse.Namespace) -> int:
    try:
        results = sync_agents(
            manifest_url=args.manifest,
            cache_dir=agent_cache_dir(),
            formats=args.formats,
            force=args.force,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Agent sync failed: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("No agents synced.")
        return 0

    for row in results:
        print(f"  [{row.action}] {row.format}: {row.detail}")
    downloaded = sum(1 for row in results if row.action == "downloaded")
    if downloaded:
        print(f"\nSynced {downloaded} agent(s) to {agent_cache_dir()}")
    return 0


def _run_ensure(args: argparse.Namespace) -> int:
    try:
        results = ensure_agents_available(
            manifest_url=args.manifest,
            cache_dir=agent_cache_dir(),
            formats=args.formats,
            force=args.force,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Agent ensure failed: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("No agent formats to ensure.")
        return 0

    for row in results:
        print(f"  [{row.action}] {row.format}: {row.detail}")
    ready = sum(
        1
        for row in results
        if row.action in {"downloaded", "unchanged", "bootstrapped"}
    )
    if ready:
        print(f"\nAgent cache ready at {agent_cache_dir()}")
    return 0


def _run_status(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    fmt = args.format or str(manifest.get("default_format", "silver_age"))
    status = agent_status(agent_cache_dir(), fmt)
    print(json.dumps(status, indent=2))
    return 0 if status.get("exists") else 1


def _run_publish(args: argparse.Namespace) -> int:
    notes = args.notes or ""
    if notes and Path(notes).is_file():
        notes = Path(notes).read_text(encoding="utf-8")

    try:
        manifest, bundle = publish_local_agent(
            args.format,
            release_id=args.tag,
            notes=notes,
            draft=args.draft,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Publish failed: {exc}", file=sys.stderr)
        return 1

    print(f"Published {bundle.format} as {bundle.release_id}")
    print(f"  SHA256: {bundle.sha256}")
    print(f"  Manifest updated ({len(manifest.get('agents', []))} format(s) listed)")
    print("\nNext: commit agents/manifest.json and push.")
    return 0


def print_supported_formats() -> None:
    print("Supported formats:", ", ".join(SUPPORTED_FORMATS))
