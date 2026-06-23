# Flesh and Blood rlbridge

Reinforcement-learning simulation for [Flesh and Blood](https://fabtcg.com/) TCG, built on [Talishar](https://talishar.net/) as the game engine.

This package is **not** bundled with rlbridge. Install it separately when you want FaB training, evaluation, or MCP tooling.

---

## Overview

This package wraps the [Talishar](https://talishar.net/) server into Gym-compatible RL environments and provides a full training/evaluation/simulation stack:

- **Self-play training** with PPO dual-agent pipelines
- **Three-phase workflow** - tune decks via targeted swaps (Phase 1), auto sideboard per matchup with manual override (Phase 2), train play agents (Phase 3)
- **Format-rule enforcement** - deck-size caps (e.g. Silver Age = 40), hero + full equipment required, token cards excluded from counts
- **FaBrary integration** - fetch real decks by URL or slug; auto-sideboard to meet format rules
- **C++ fast simulation** - compiled pybind11 engine replaces the HTTP game server for training (up to 8× faster, fully thread-safe)
- **MCP tools** - expose training and simulation tasks to AI assistants via the rlbridge MCP plugin

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

# Backend only (recommended for training - saves RAM, no browser needed)
.\start_talishar.ps1 -BackendOnly

# FE only (backend already running)
.\start_talishar.ps1 -FeOnly

# Stop everything
.\start_talishar.ps1 -Down
```

The script polls both services until they respond (backend ≤ 30 s, FE ≤ 20 s) and prints readiness status.

---

## Interactive launcher (`main.py`)

The default entry point is a Rich-based terminal UI for running experiments without memorizing script paths:

```bash
python main.py          # launch the TUI (default)
```


### Main menu

| # | Option | What it does |
|---|--------|----------------|
| **1** | Sideboard comparison | Full sideboard-tuning pipeline: pick your deck and a SAGE precon opponent, generate swap variants, train play agents in parallel, and compare candidates. |
| **2** | Fixed deck simulation | Train play agents on two fixed decks (FaBrary URL/slug or local JSON) with optional C++ engine acceleration. |
| **3** | Evaluate checkpoints | Win-rate evaluation or GIF render-only replay from saved phase-3 checkpoints; can watch for new checkpoints. |
| **4** | Evaluate trained agent | Run additional Talishar eval games against a **completed** training run (latest checkpoint). |
| **5** | Real-time Talishar play | Open the live Talishar frontend in Chromium to **watch** the agent or **play against it**. In human mode, choose trained vs opponent deck; optional agent-coach overlay shows policy % and C++ win estimates on your turns. |
| **6** | Settings | Talishar backend/FE URLs, Assets path, FaBrary API key; rescan `cards.json` from the FAB Card Vault; normalize card IDs for Talishar. |

Deck sources throughout the TUI include Talishar SAGE precons, FaBrary links/slugs, and saved sideboard lists.

**Talishar requirements by menu option:**

- Options **3–5** need the Talishar backend running (`.\start_talishar.ps1` or `-BackendOnly`).
- Option **5** also needs Talishar-FE (`.\start_talishar.ps1` without `-BackendOnly`, or `-FeOnly` if the backend is already up).

---

## Implementation status

### Three-phase pipeline

| Step | Status | Notes |
|------|--------|-------|
| **Phase 1 - Deck tuning** | ✓ | Starting from a deck, swap specific cards and evaluate variants by win rate (e.g. sideboard comparison). |
| **Phase 2 - Sideboard** | ✓ | A default policy auto-selects the game deck for each matchup; you can override manually when needed. |
| **Phase 3 - Play** | ✓ | Dual-agent PPO self-play. Training uses the **C++ engine** when built; checkpoints and final eval can use Talishar HTTP. |
| **Final evaluation** | ✓ | Win-rate games on the Talishar backend; optional GIF replay via Talishar-FE + Playwright. |

### TUI & launcher (`main.py`)

| Step | Status | Notes |
|------|--------|-------|
| **1 - Sideboard comparison** | ✓ | Swap variants, parallel Phase 3 training, HTML dashboard. |
| **2 - Fixed deck simulation** | ✓ | Two fixed decks, optional C++ engine, play training + final eval. |
| **3 - Evaluate checkpoints** | ✓ | Win-rate eval or GIF render-only; can watch for new checkpoints. |
| **4 - Evaluate trained agent** | ✓ | Extra Talishar eval games against a completed run. |
| **5 - Real-time Talishar play** | ✓ | Watch agent or play against it in Chromium; optional agent-coach overlay. |
| **6 - Settings** | ✓ | Talishar URLs, Assets path, FaBrary key, card DB rescan / ID normalization. |
| **Agent coach overlay** | ✓ | Policy % and C++ win estimates during human vs agent play. |


### Reinforcement Learning Agents

| Step | Status | Notes |
|------|--------|-------|
| **PPO** | ✓ | Simple PPO agent implementation. |
| **Warm-start Training** | ✓ | Logic based policy (best net attack for current hand) used for early exploration. |
| **Agent Cache** | ✓ | Store trained agents, re-use for future runs. |


### Simulation backends

| Backend | Status | Notes |
|---------|--------|-------|
| **HTTP Talishar** | ✓ | Full FaB rules via local Docker backend. Required for live play, GIF rendering, and authoritative final eval. |
| **C++ engine** | Partial | Auto-generated per matchup (`scripts/cpp/generate_cpp_engine.py`). Fast and thread-safe for training, but uses a simplified turn loop and **per-card stubs** that must be translated from Talishar PHP. Stalemate detection ends games when stubs are incomplete. |
| **C++ ↔ Talishar parity** | Partial | Checker at `scripts/cpp/check_cpp_vs_talishar_parity.py`; expect discrepancies until card stubs are finished. |
| **Legacy Python simulator** | ✓ | `FleshAndBloodGameplayEnvironment` - lightweight scripted opponent, not full Talishar rules. |

### Integrations & tooling

| Feature | Status | Notes |
|---------|--------|-------|
| **FaBrary deck fetch** | ✓ | URL, slug, or local JSON throughout TUI and training scripts. |
| **Format rules** | ✓ | Silver Age, Classic Constructed, Blitz, UPF (SAGE maps to Silver Age). |
| **MCP tools** | ✓ | `fab_simulate_matchup`, `fab_simulate_vs_fixed_opponent`, `fab_run_full_pipeline`, `fab_start_talishar`. |
| **Card database** | ✓ | ~6,900 cards; rescan from FAB Card Vault; Talishar ID normalization. |

### Not yet implemented or intentionally limited

- **Full Talishar fidelity in C++ training** - training win rates reflect the compiled C++ engine, not Talishar due to runtime costs .
- **Complete card effect coverage** - ~447 cards still have partial, unparsed, or missing effect logic (see `card_db/unimplemented_cards.md`).
- **Equip during default play policy** - `mode=3` is skipped to avoid pitch-window infinite loops (`talishar_default_policy.py`).
- **Arbitrary-deck C++ engine without build step** - each matchup needs generation + compile under `results/cpp_engines/`.
- **Cloud / remote Talishar** - workflows assume a local backend; no intergration into actual Talishar website.
- **LLM Agent** - would be slower interaction but powerful method for guided actions for user.

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
| Typical speed | 0.05 games/s | 5-10 games/s  |


---

## Card Database

Card and hero metadata live in `src/flesh_and_blood_rlbridge/card_db/`. The database (`cards.json`, ~6,900 cards) is the authoritative source for:

- **Equipment slots** - `type_line` suffix e.g. `"Generic Equipment - Chest"` → slot `chest`
- **Weapon types and hand count** - e.g. `"Ninja Weapon - Dagger (1H)"`, `"Runeblade Weapon - Sword (2H)"`
- **Hero variants** - e.g. `"Ninja Hero - Young"`
- **Token identification** - `card_types` list contains `"token"`

To refresh from an upstream export:

```bash
cd src/flesh_and_blood_rlbridge/card_db
python import_from_talishar.py --source /path/to/upstream_cards.json --out cards.json
```

---

## Repository

https://github.com/pdfosborne/flesh-and-blood-rlbridge
