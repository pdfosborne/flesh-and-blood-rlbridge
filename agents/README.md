# Official unified agent releases

Public unified PPO agent weights for GUI evaluation and live play are listed in
[`manifest.json`](manifest.json) and hosted on [GitHub Releases](https://github.com/pdfosborne/flesh-and-blood-rlbridge/releases).

## On-disk layout (after sync)

```
results/agent_cache/<format>/unified_agent_v2.json
results/agent_cache/<format>/unified_agent.meta.json
```

Supported format keys: `silver_age`, `classic_constructed`, `blitz`, `upf` (GUI `sage` maps to `silver_age`).

**Note:** Legacy MLP weights (`unified_agent_v1.json` on GitHub release `agents-2026.06.1`) are incompatible with the attention policy trunk (`attention_v1`). `fab-bridge agents ensure` rejects them and installs an untrained v2 bootstrap placeholder until a v2 release is published.

## For users

Download official weights (or install a bootstrap placeholder when the public release is legacy/incompatible):

```bash
fab-bridge agents ensure
```

Strict manifest sync only (no bootstrap fallback):

```bash
fab-bridge agents sync
```

Or use TUI menu **9 — Sync unified agent from public release** to compare local vs public versions and download updates.

`fab-bridge init` runs sync automatically (best-effort; use `--no-agents` to skip offline).

Check status:

```bash
fab-bridge agents status
fab-bridge agents status --format silver_age
```

## For maintainers

1. Train a unified agent (TUI menu **6** or `train_unified_random_matchups.py`).
2. Publish from the TUI (**8. Publish unified agent**) or CLI:

```bash
fab-bridge agents publish --format silver_age --tag agents-2026.06.1
```

This creates a GitHub Release via `gh`, uploads weight + meta assets, and updates `agents/manifest.json`.

3. Commit and push the manifest update so `fab-bridge agents sync` picks up new URLs.

### Requirements for publish

- [GitHub CLI](https://cli.github.com/) (`gh`) installed and authenticated (`gh auth login`)
- Local weights at `results/agent_cache/<format>/unified_agent_v2.json`

### Manifest fields

Each `agents[]` entry includes `weights_url`, `meta_url`, `sha256`, `obs_dim`, `architecture`, and `release` tag.
Release asset filenames use the pattern `{format}-unified_agent_v2.json` (distinct from cache filenames).
