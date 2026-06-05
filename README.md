# Flesh and Blood rlbridge

Reinforcement-learning bridge for [Flesh and Blood](https://fabtcg.com/) TCG, built on [Talishar](https://talishar.net/) as the game engine.

This package is **not** bundled with rlbridge. Install it separately when you want FaB training, evaluation, or MCP tooling.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Talishar Setup](#talishar-setup)
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

## C++ Engine

### What it does

`scripts/generate_cpp_engine.py` scans ≈50 Talishar PHP source files and auto-generates a self-contained C++ game engine:

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
# Build engine for a hero matchup (names, slugs, or FaBrary URLs accepted)
.\build_cpp_engine_for_matchup.ps1 -Deck1 aurora -Deck2 briar

# With FaBrary deck URLs
.\build_cpp_engine_for_matchup.ps1 -Deck1 "https://fabrary.net/decks/..." -Deck2 briar
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
| `build_cpp_engine_for_matchup.ps1` | Generate C++ source from Talishar PHP and CMake-build the pybind11 engine for a matchup. Output lands in `results/cpp_engines/{matchup}-{hash}/`. |
| `simulate_deck_matchup.ps1` | Fetch two decks (FaBrary or local JSON), sideboard each to format rules, build the C++ engine, run simulated games, and print win percentages. |
| `run_aurora_vs_briar_fixed_opponent.ps1` | Train an Aurora deck through all three phases (deckbuild → sideboard → play) against a fixed Briar opponent. |
| `run_sage_aurora_vs_briar_deckbuild.ps1` | Full dual co-training pipeline for Aurora vs Briar: both players deckbuild, sideboard, and train simultaneously. |
| `run_sage_briar_vs_dorinthea_play.ps1` | Play-phase-only training for Briar vs Dorinthea (skips deckbuild and sideboard). |

---

## Python Scripts

All utility scripts live in `scripts/`.

### Training pipelines

| Script | Purpose |
|--------|---------|
| `train_full_pipeline.py` | Three-phase pipeline: deckbuild → sideboard → play. Supports `preset`, `mirror`, and `dual` modes. Accepts `--p1-fixed-deck`, `--p2-fixed-deck`, `--cpp-engine-dir`. Main entry-point for PS scripts. |
| `train_three_phase_pipeline.py` | Earlier three-phase pipeline (superseded by `train_full_pipeline.py`). |
| `train_eval_render_pipeline.py` | Train → evaluate → render the optimal-policy rollout as images/GIF. |
| `train_sage_precons.py` | Dual-agent PPO training across all 45 SAGE precon cross-matchups (C(10,2)). |
| `train_silver_age_decks.py` | Dual-agent PPO for Silver Age FaBrary deck cross-matchups. |
| `train_classic_constructed_decks.py` | Dual-agent PPO for Classic Constructed FaBrary deck cross-matchups. |
| `train_dual_agent_common.py` | Shared PPO training loop, `Matchup` dataclass, and `make_env()` factory used by all pipeline scripts. |

### Evaluation & diagnostics

| Script | Purpose |
|--------|---------|
| `eval_agent_vs_agent.py` | Evaluate two saved PPO agents head-to-head across N games; returns win rates and confidence intervals. |
| `check_oracle.py` | Run games and validate combat resolutions against the rules oracle to catch engine regressions. |
| `test_talishar_env.py` | Environment smoke test — plays one full episode with random actions against the live server and prints a summary. |
| `_debug_game_flow.py` | Internal game-flow debugger; prints step-by-step state transitions. |

### Deck & data utilities

| Script | Purpose |
|--------|---------|
| `fetch_fabrary_deck.py` | Fetch a deck from the FaBrary API by URL or slug. Saves JSON with `hero_id`, `hero_class`, `equipment_header`, `deck`, and `sideboard` fields. |
| `generate_cpp_engine.py` | Scan Talishar PHP source and emit C++ + pybind11 engine sources ready for CMake. |
| `cli_talishar.py` | Interactive human-vs-CombatDummy CLI. Optionally load a trained agent checkpoint to watch it play. |
| `_probe_fabrary.py` | FaBrary API probe / diagnostics — inspect raw API responses. |

### Cache helpers

| Script | Purpose |
|--------|---------|
| `agent_cache.py` | Four-tier PPO agent cache (deck×deck → deck×hero → hero×hero → hero). Warm-starts training from the best available prior run. |
| `episode_cache.py` | Persistent episode replay buffer keyed by matchup. Stores complete episodes to skip the default-policy warmup phase on re-runs. |

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
