# Official unified agent releases

Public unified PPO agent weights for GUI evaluation and live play are listed in
[`manifest.json`](manifest.json) and hosted on [GitHub Releases](https://github.com/pdfosborne/flesh-and-blood-rlbridge/releases).

## On-disk layout (after sync)

```
results/agent_cache/shared/card_text_embeddings_v1.npz
results/agent_cache/shared/card_text_embeddings_v1.meta.json
results/agent_cache/<format>/unified_agent_v3.json
results/agent_cache/<format>/unified_agent.meta.json
```

Supported format keys: `silver_age`, `classic_constructed`, `blitz`, `upf` (GUI `sage` maps to `silver_age`).

**Note:** Legacy weights (`attention_v1`, schema v1/v2) are incompatible with the current `attention_v2_text` policy and obs schema v3 (818-dim). Run `fab-bridge agents ensure` to install a bootstrap placeholder or sync an official release.

Card text embeddings (`card_text_embeddings_v1.npz`) ship in the Python package and on GitHub Releases. `fab-bridge agents sync/ensure` installs them into `results/agent_cache/shared/` so GUI eval works with custom decks via frozen MiniLM embeddings.

## For users

Download official weights and embeddings:

```bash
fab-bridge agents ensure
```

Strict manifest sync only (no bootstrap fallback):

```bash
fab-bridge agents sync
```

Check status:

```bash
fab-bridge agents status
fab-bridge agents status --format silver_age
```

## For maintainers

1. Train a unified agent (TUI menu **6** or `train_unified_random_matchups.py`).
2. Publish from the TUI (**8. Publish unified agent**) or CLI:

```bash
fab-bridge agents publish --format silver_age --tag agents-2026.07.1
```

This creates a GitHub Release via `gh`, uploads weight + meta + text embedding assets, and updates `agents/manifest.json`.

3. Commit and push the manifest update so `fab-bridge agents sync` picks up new URLs.

### Requirements for publish

- [GitHub CLI](https://cli.github.com/) (`gh`) installed and authenticated (`gh auth login`)
- Local weights at `results/agent_cache/<format>/unified_agent_v3.json`
- Bundled embeddings at `src/flesh_and_blood_rlbridge/card_db/card_text_embeddings_v1.npz`

### Manifest fields

Top-level `text_embeddings` block: shared embedding asset URLs and SHA256.

Each `agents[]` entry includes `weights_url`, `meta_url`, `sha256`, `obs_dim` (818), `obs_schema_version` (3), `architecture` (`attention_v2_text`), `requires_text_embed_version`, and `release` tag.

Release asset filenames use the pattern `{format}-unified_agent_v3.json`.
