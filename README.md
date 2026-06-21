# Flesh and Blood rlbridge

Reinforcement-learning simulation for [Flesh and Blood](https://fabtcg.com/) TCG, built on [Talishar](https://talishar.net/) as the game engine.

This package is **not** bundled with rlbridge. Install it separately when you want FaB training, evaluation, or MCP tooling.

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


---

## Card Database

Card and hero metadata live in `src/flesh_and_blood_rlbridge/card_db/`. The database (`cards.json`, ~6,900 cards) is the authoritative source for:

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
