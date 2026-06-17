# Flesh and Blood rlbridge

Reinforcement-learning bridge for [Flesh and Blood](https://fabtcg.com/) TCG, built on [Talishar](https://talishar.net/) as the game engine.

This package is **not** bundled with rlbridge. Install it separately when you want FaB training, evaluation, or MCP tooling.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Talishar Setup](#talishar-setup)
- [FaBrary deck fetching](#fabrary-deck-fetching)
- [C++ Engine](#c-engine)
- [MCP Tools](#mcp-tools)
- [PowerShell Scripts](#powershell-scripts)
- [Python Scripts](#python-scripts)
- [Environments](#environments)
- [Card Database](#card-database)
- [Repository](#repository)

---

## Overview

This package wraps the [Talishar](https://talishar.net/) server into Gym-compatible RL environments and provides a full training/evaluation/simulation stack:

- **Self-play training** with PPO dual-agent pipelines
- **Three-phase RL** — Phase 1 builds a deck, Phase 2 sideboards before each match, Phase 3 plays games
- **Format-rule enforcement** — deck-size caps (e.g. Silver Age = 40), hero + full equipment required, token cards excluded from counts
- **FaBrary integration** — fetch real decks by URL or slug; auto-sideboard to meet format rules
- **C++ fast simulation** — compiled pybind11 engine replaces the HTTP game server for training (up to 8× faster, fully thread-safe)
- **MCP tools** — expose training and simulation tasks to AI assistants via the rlbridge MCP plugin

---

## Prerequisites

| Dependency | Minimum version | Notes |
|------------|----------------|-------|
| Python | 3.10 | 3.12 recommended |
| PowerShell | 7.0 (`pwsh`) | `powershell.exe` v5.1 accepted as fallback |
| Docker Desktop | Latest | Talishar backend container |
| Node.js / npm | 18 | Talishar-FE Vite dev server |
| CMake | 3.21 | C++ engine build |
| C++ compiler | MSVC 2022 / GCC 11 | C++ engine build |

---

## Installation

### From GitHub

```bash
pip install rlbridge
pip install git+https://github.com/pdfosborne/flesh-and-blood-rlbridge.git
```

### Development checkout

```bash
pip install -e /path/to/rlbridge
pip install -e /path/to/flesh-and-blood-rlbridge
```

rlbridge discovers environments via the `rlbridge.environments` entry-point and MCP tools via `rlbridge.environment_mcp_tools`.

### Update Talishar source subtree

```bash
git subtree pull --prefix=Talishar talishar-upstream main --squash
```

---

## Talishar Setup

The RL pipeline requires the Talishar game server running locally. Two components are involved:

| Component | Directory | Default URL |
|-----------|-----------|-------------|
| Talishar backend (PHP + Docker) | `Talishar/` | `http://localhost:8080` |
| Talishar-FE (Vite, optional) | `Talishar-FE/` | `http://localhost:5173` |

> **The FE is only needed for GIF rendering.** All training and simulation work with the backend alone.

### First-time setup

1. Ensure `Talishar/` is populated (git subtree pull or manual clone of [Talishar](https://github.com/Talishar/Talishar)).
2. Ensure `Talishar-FE/` is populated (clone [Talishar-FE](https://github.com/Talishar/Talishar-FE)).
3. Install Node dependencies for the front end:
   ```bash
   cd Talishar-FE
   npm install
   ```
4. (Optional) Copy `.env.example` → `.env` in `Talishar-FE/` and set:
   ```
   VITE_BACKEND_URL=http://localhost:8080
   ```

### Starting and stopping

```powershell
# Start backend + FE
.\start_talishar.ps1

# Backend only (recommended for training — saves RAM, no browser needed)
.\start_talishar.ps1 -BackendOnly

# FE only (backend already running)
.\start_talishar.ps1 -FeOnly

# Stop everything
.\start_talishar.ps1 -Down
```

The script polls both services until they respond (backend ≤ 30 s, FE ≤ 20 s) and prints readiness status.

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TALISHAR_URL` | `http://localhost:8080` | Game API endpoint (HTTP environments) |
| `TALISHAR_FE_URL` | `http://localhost:5173` | FE URL for GIF screenshot render |
| `TALISHAR_ASSETS_PATH` | `Talishar/` | Card image asset root |

```powershell
$env:TALISHAR_URL     = "http://localhost:8080"
$env:TALISHAR_FE_URL  = "http://localhost:5173"
```

---

## FaBrary deck fetching

Several scripts accept FaBrary deck URLs or 26-character slugs and convert them to rlbridge JSON (`hero_id`, `hero_class`, `equipment_header`, `deck`, `sideboard`). The fetch logic lives in `scripts/deck/fetch_fabrary_deck.py` and is used by `runscripts/simulate_deck_matchup.py`, `scripts/cpp/build_cpp_engine_for_matchup.py`, and the training pipeline scripts.

### Do I need an API key?

**Usually no.** For public decks, the fetcher falls back to FaBrary's AppSync GraphQL API (unauthenticated Cognito IAM credentials). That path works without any setup.

An API key is only needed if:

- You want the legacy REST endpoint (`x-api-key` header) instead of the GraphQL fallback.
- A deck is private or AppSync returns no data (you must export it yourself or use a local JSON file).

If fetch fails, you can always pass a local deck JSON path instead of a URL:

```powershell
python runscripts/simulate_deck_matchup.py `
    "C:\Decks\briar.json" `
    "C:\Decks\riptide.json"
```

### Setting the API key

The fetcher resolves credentials in this order:

1. `--api-key` on `fetch_fabrary_deck.py`
2. `FABRARY_API_KEY` environment variable (alias: `FABRARY_KEY`)
3. Resolved `$FaBraryKey` in `Talishar/APIKeys/APIKeys.php` (Talishar contributors with 1Password access)

#### Option A — environment variable (recommended)

Set the key for the current PowerShell session:

```powershell
$env:FABRARY_API_KEY = "your-fabrary-api-key"
.\simulate_deck_matchup.ps1
```

Persist it for your user account (PowerShell 7):

```powershell
[Environment]::SetEnvironmentVariable("FABRARY_API_KEY", "your-fabrary-api-key", "User")
```

On Linux/macOS:

```bash
export FABRARY_API_KEY="your-fabrary-api-key"
```

#### Option B — CLI flag (one-off fetches)

```bash
python scripts/deck/fetch_fabrary_deck.py \
    "https://fabrary.net/decks/01KTBBVEZE0TPDAZ74Z4D787G4" \
    --api-key "your-fabrary-api-key" \
    --out results/matchup_sims/decks/briar.json \
    --pretty
```

#### Option C — Talishar `APIKeys.php` (contributors)

Talishar stores the FaBrary server key in `Talishar/APIKeys/APIKeys.php`. That file is gitignored; copy from the template and inject secrets with 1Password:

```bash
cp Talishar/APIKeys/APIKeys.php.template Talishar/APIKeys/APIKeys.php
op inject -i Talishar/APIKeys/APIKeys.php -o Talishar/APIKeys/APIKeys.php
```

`fetch_fabrary_deck.py` reads `$FaBraryKey` only when the value is a literal string (not an `op://` placeholder). If you have Talishar contributor access, this avoids setting a separate env var.

### Fetching a deck manually

```bash
# Print JSON to stdout
python scripts/deck/fetch_fabrary_deck.py https://fabrary.net/decks/01KST88R7JVEQ73M82ZA0PJ9RN

# Save to a file (cached decks are reused by simulate_deck_matchup.ps1)
python scripts/deck/fetch_fabrary_deck.py https://fabrary.net/decks/01KST88R7JVEQ73M82ZA0PJ9RN \
    --out results/matchup_sims/decks/aurora.json --pretty

# Append to the static deck database for warm-start training
python scripts/deck/fetch_fabrary_deck.py https://fabrary.net/decks/01KST88R7JVEQ73M82ZA0PJ9RN \
    --append-to src/flesh_and_blood_rlbridge/card_db/fabrary_decks.json \
    --deck-id fab_aurora_sa_starter
```

Fetched decks are cached under `results/matchup_sims/decks/` (or the script-specific deck dir). Delete a cached `*_deck.json` file to force a re-fetch.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `HTTP 403` on REST, then success via AppSync | Normal without API key | Ignore the REST warning, or set `FABRARY_API_KEY` |
| `WAFForbiddenException` on AppSync | WAF blocked the request | Update to the latest `fetch_fabrary_deck.py`; ensure outbound HTTPS to `*.appsync-api.us-east-2.amazonaws.com` is allowed |
| `AppSync returned no deck data` | Private or invalid deck slug | Open the URL in a browser; export JSON or add to `fabrary_decks.json` |
| `Deck JSON not available` in a PS script | Fetch failed and no cache | Run `fetch_fabrary_deck.py` manually to see the error, or pass a local JSON path |

For low-level API inspection, inspect `scripts/deck/fetch_fabrary_deck.py` or run it with `--help`.

---

## C++ Engine

### What it does

`scripts/cpp/generate_cpp_engine.py` scans ≈50 Talishar PHP source files and auto-generates a self-contained C++ game engine:

| Generated file | Contents |
|----------------|----------|
| `gamestate.h/.cpp` | Full game-state structs and turn-loop logic |
| `cards.h` | Card-effect dispatch table |
| `bindings.cpp` | pybind11 bindings (identical API to the HTTP environments) |
| `CMakeLists.txt` | Stand-alone CMake build configuration |

The compiled output is a Python extension module (`fab_engine.cp312-win_amd64.pyd` on Windows / `.so` on Linux) placed in a **content-hashed** directory under `results/cpp_engines/{matchup}-{hash}/`.

### Why use it

| | HTTP environments | C++ engine |
|--|--|--|
| Game call cost | One HTTP round-trip per step | In-process function call |
| Thread safety | No (single PHP process) | Yes (independent state per env) |
| Parallel workers | 1 | 4–8 |
| Typical speed | 1× | 6–8× |

### Building

```powershell
# Build engine for a hero matchup
python scripts/cpp/build_cpp_engine_for_matchup.py --deck1 aurora --deck2 briar

# With optional FaBrary deck JSON files for full card-pool coverage
python scripts/cpp/build_cpp_engine_for_matchup.py --deck1 aurora --deck2 briar --deck1-json path/to/deck.json
```

Pipeline scripts (`train_full_pipeline.py`, `simulate_deck_matchup.ps1`) **auto-discover** the hashed engine directory — no manual path configuration is needed after building.

---

## MCP Tools

When this package is installed, rlbridge's MCP plugin exposes **4 tools**:

### `fab_start_talishar`

Start, stop, or check the status of the local Talishar stack.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | `"start"/"stop"/"status"` | `"start"` | Operation to perform |
| `backend_only` | bool | `false` | Start only the Docker backend, skip the Vite FE |
| `fe_only` | bool | `false` | Start only the Vite FE |
| `talishar_url` | str | `http://localhost:8080` | Override backend URL |
| `fe_url` | str | `http://localhost:5173` | Override FE URL |
| `timeout_seconds` | int | `60` | Maximum wait for services to become ready |

---

### `fab_simulate_matchup`

Simulate two fixed decks head-to-head and return win percentages. Automatically sideboards either deck if it does not meet format rules. Decks may be FaBrary URLs, FaBrary slugs, or paths to local deck JSON files.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deck1_source` | str | — | Deck 1 (URL / slug / local path) |
| `deck2_source` | str | — | Deck 2 (URL / slug / local path) |
| `play_episodes` | int | `200` | Games per in-training evaluation |
| `final_eval_episodes` | int | `500` | Games in final head-to-head evaluation |
| `game_format` | str | `"silver_age"` | `"silver_age"` / `"blitz"` / `"cc"` / `"ll"` / `"upf"` |

---

### `fab_simulate_vs_fixed_opponent`

Train one player's deck and sideboard against a **fixed opponent** deck. The training player runs through all three phases (deckbuild → sideboard → play); the opponent's deck never changes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `training_deck_source` | str | — | The deck that will be optimised |
| `fixed_deck_source` | str | — | The opponent's fixed deck |
| `deckbuild_episodes` | int | `500` | Phase 1 (deck construction) training games |
| `sideboard_episodes` | int | `300` | Phase 2 (sideboard) training games |
| `play_episodes` | int | `1000` | Phase 3 (play) training games |
| `game_format` | str | `"silver_age"` | Format |

---

### `fab_run_full_pipeline`

Full co-training pipeline where **both players** simultaneously build decks, sideboard, and train. Both players can be seeded from FaBrary decks or just a hero name.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `p1_source` | str | — | P1 deck / hero (URL / slug / hero name) |
| `p2_source` | str | — | P2 deck / hero |
| `deckbuild_episodes` | int | `500` | Phase 1 games per player |
| `sideboard_episodes` | int | `300` | Phase 2 games per player |
| `play_episodes` | int | `2000` | Phase 3 games |
| `iterations` | int | `3` | Number of complete pipeline repeats |
| `game_format` | str | `"silver_age"` | Format |
| `gif_fps` | int | `0` | Render a GIF at N fps after training; `0` = no render (FE not required when `0`) |

---

## PowerShell Scripts

| Script | Purpose |
|--------|---------|
| `start_talishar.ps1` | Start/stop the Talishar Docker backend and Vite FE. Flags: `-BackendOnly`, `-FeOnly`, `-Down`. Polls both services for readiness. |
| `scripts/cpp/build_cpp_engine_for_matchup.py` | Generate C++ source from Talishar PHP and CMake-build the pybind11 engine for a matchup. Output lands in `results/cpp_engines/{matchup}-{hash}/`. |
| `runscripts/simulate_deck_matchup.py` | Fetch two decks (FaBrary or local JSON), sideboard each to format rules, build the C++ engine, run simulated games, and print win percentages. |
| `runscripts/aurora_vs_briar_fixed_opponent.py` | Train an Aurora deck through all three phases (deckbuild → sideboard → play) against a fixed Briar opponent. |
| `runscripts/sage_aurora_vs_briar_deckbuild.py` | Full dual co-training pipeline for Aurora vs Briar: both players deckbuild, sideboard, and train simultaneously. |
| `runscripts/sage_briar_vs_dorinthea_play.py` | Play-phase-only training for Briar vs Dorinthea (skips deckbuild and sideboard). |

---

## Python Scripts

Utility scripts live under `scripts/` in four categories. See `scripts/README.md`
for layout. High-level orchestration lives in `runscripts/` at the repo root.

### `scripts/training/`

| Script | Purpose |
|--------|---------|
| `train_full_pipeline.py` | Three-phase pipeline: deckbuild → sideboard → play. Supports `preset`, `mirror`, and `dual` modes. Main entry-point for runscripts. |
| `train_eval_render_pipeline.py` | Train → evaluate → render the optimal-policy rollout as images/GIF. |
| `train_sage_precons.py` | Dual-agent PPO training across all 45 SAGE precon cross-matchups (C(10,2)). |
| `train_silver_age_decks.py` | Dual-agent PPO for Silver Age FaBrary deck cross-matchups. |
| `train_classic_constructed_decks.py` | Dual-agent PPO for Classic Constructed FaBrary deck cross-matchups. |
| `train_dual_agent_common.py` | Shared PPO training loop, `Matchup` dataclass, and `make_env()` factory. |
| `agent_cache.py` | Four-tier PPO agent cache for warm-starting training. |
| `episode_cache.py` | Persistent episode replay buffer keyed by matchup. |

### `scripts/eval/`

| Script | Purpose |
|--------|---------|
| `eval_phase3_checkpoint.py` | Live dashboard for phase-3 checkpoint evaluation and GIF render. |
| `eval_agent_vs_agent.py` | Evaluate two saved PPO agents head-to-head across N games. |

### `scripts/cpp/`

| Script | Purpose |
|--------|---------|
| `build_cpp_engine_for_matchup.py` | Generate and CMake-build the pybind11 engine for a matchup. |
| `generate_cpp_engine.py` | Scan Talishar PHP source and emit C++ engine sources. |
| `check_cpp_vs_talishar_parity.py` | Core C++ vs HTTP Talishar parity checker library. |
| `run_parity_check.py` | CLI wrapper for single-matchup parity checks. |
| `run_random_parity_sweep.py` | Randomized parity sweeps across Talishar deck assets. |

### `scripts/deck/`

| Script | Purpose |
|--------|---------|
| `fetch_fabrary_deck.py` | Fetch a deck from the FaBrary API by URL or slug. |

### Tests

| Script | Purpose |
|--------|---------|
| `tests/test_talishar_env_smoke.py` | Manual smoke test against a live Talishar server (random actions). |

---

## Environments

| Env ID | Description |
|--------|-------------|
| `FleshAndBlood-Talishar-v0` | Self-play: one policy controls both players (live Talishar HTTP server) |
| `FleshAndBlood-Talishar-SelfPlay-v0` | Alias for `FleshAndBlood-Talishar-v0` |
| `FleshAndBlood-Talishar-VsAI-v0` | Agent vs CombatDummy AI (live Talishar HTTP server) |
| `FleshAndBlood-SelfPlay-v0` | Single policy controls both heroes (local C++ simulator) |
| `FleshAndBlood-DeckBuild-v0` | Two-phase deck selection before play |

```python
from rlbridge.environments.registry import registry

env = registry.create("FleshAndBlood-Talishar-v0", format="silver_age")
result = env.reset(seed=0)
print(result.observation)

# List all registered FaB environments
registry.list_environments(namespace="flesh_and_blood")
```

---

## Card Database

Card and hero metadata live in `src/flesh_and_blood_rlbridge/card_db/`. The database (`cards.json`, ~6 900 cards) is the authoritative source for:

- **Equipment slots** — `type_line` suffix e.g. `"Generic Equipment - Chest"` → slot `chest`
- **Weapon types and hand count** — e.g. `"Ninja Weapon - Dagger (1H)"`, `"Runeblade Weapon - Sword (2H)"`
- **Hero variants** — e.g. `"Ninja Hero - Young"`
- **Token identification** — `card_types` list contains `"token"`

To refresh from an upstream export:

```bash
cd src/flesh_and_blood_rlbridge/card_db
python import_from_talishar.py --source /path/to/upstream_cards.json --out cards.json
```

---

## Repository

https://github.com/pdfosborne/flesh-and-blood-rlbridge
