# Flesh and Blood rlbridge 
Reinforcement learning simulation for [Flesh and Blood](https://fabtcg.com/) TCG, built on [Talishar](https://talishar.net/) as the game engine.

![FrontEnd-with-AgentGuide](./docs/_images/fab-FE-demo-agentguide.gif)

*Example usage: play against AI agents with guidance for optimal play based on millions of simulated matches.*


**This project is in alpha development and will likely have bugs.**

---

## Quick start

**Option A** needs [Docker Desktop](https://www.docker.com/products/docker-desktop/) only. 

### Option A - Docker GUI + Talishar-FE (live play)

Recommended if you want the web GUI, pre-trained unified agents, sideboard evaluation, replay GIFs, and **live play** in Talishar-FE (watch or play against the agent).

```bash
git clone https://github.com/pdfosborne/flesh-and-blood-rlbridge.git
cd flesh-and-blood-rlbridge
./scripts/docker-setup.sh
```

```powershell
.\scripts\docker-setup.ps1
```

When setup finishes, open **http://localhost:8765** (GUI) and **http://localhost:5173** (Talishar-FE for live play). Unified agents are installed on first container start (public sync when available, otherwise a bootstrap placeholder). Refresh with `fab-bridge agents ensure`.

**Sideboard-only (no live play):** use `./scripts/docker-setup.sh --eval` (or `.\scripts\docker-setup.ps1 -Eval`) for a slimmer stack without Talishar-FE or Playwright.

![GUI-Demo-1](./docs/_images/GUI-demo-img-1.png)
*GUI demo - select player deck (precons or import from Fabrary), you can edit equipment as needed.*

![GUI-Demo-2](./docs/_images/GUI-demo-img-2.png)
*GUI demo - select opponent deck, auto sideboards, can edit equipment or any card and save for future runs.*

![GUI-Demo-3](./docs/_images/GUI-demo-img-3.png)
*GUI demo - create a deck variation by swapping cards, 'save for evaluation' confirms the selection.*


![GUI-Demo-4](./docs/_images/GUI-demo-img-4.png)
*GUI demo - agents are trained for both player and opponent, final evaluation on local Talishar server directly to ensure correctness (but can take a long time).*

![GUI-Demo-5](./docs/_images/GUI-demo-img-5.png)
*GUI demo - final results include win % summary, damage breakdown and example render of gameplay.*


To start the stack again later (inside the repo directory), use a compose wrapper so GPU is picked up automatically when available:

```bash
./scripts/docker-compose.sh --profile full up   # GUI + Talishar-FE (same as Option A)
./scripts/docker-setup.sh --eval                # sideboard eval only (no Talishar-FE)
```

```powershell
.\scripts\docker-compose.ps1 up   # Windows PowerShell
```

**Optional GPU in Docker:** `docker-setup` / `docker-compose` scripts probe for NVIDIA CUDA (`nvidia-smi` + `docker run --gpus all`). When a GPU is available, they merge `docker-compose.gpu.yml`, which gives the `fab-bridge` container CUDA PyTorch and `gpus: all`. Otherwise the stack stays on CPU PyTorch (default). Talishar PHP and C++ rollouts remain CPU-bound either way.

**CLI commands** (`fab-tui`, `fab-gui`, `fab-bridge`) are installed in the Docker image and exposed on the host via wrappers in `bin/` — no separate Python venv required:

```bash
source scripts/docker-env.sh   # once per shell
fab-tui                        # terminal UI (second terminal while compose is up)
```

Training output and saved sideboard lists are written to **`results/`** in the repo (e.g. `results/sideboard_compare/`, `results/tui_decks/saved/`). Generated deck files for final evaluation are written to **`Talishar/Assets/`** (shared with the Talishar container; `Talishar/` itself is gitignored and cloned from upstream on first setup). Bundled training decks ship in **`assets/talishar_decks/`** and are synced into `Talishar/Assets/` automatically. Replay GIFs on the Results tab use the bundled Talishar-FE container.

If you also run Talishar separately (e.g. `cd Talishar && docker compose up`), stop that stack first so port **8080** and **`Talishar/Assets`** are not split across two instances. The compose stack auto-creates **`Talishar/HostFiles/Redirector.php`**, **`GameIDCounter.txt`**, and **`Talishar/APIKeys/APIKeys.php`** (stub with empty secrets — required by Talishar but not in git).

Stop everything with `Ctrl+C`, then:

```bash
./scripts/docker-compose.sh down
```

```powershell
.\scripts\docker-compose.ps1 down
```

### Option B - Local Python full build (train custom agents)

Use this when you want to **train your own agents** (`fab-tui`, PPO pipelines, C++ fast simulation). **Option B** needs [Docker Desktop](https://www.docker.com/products/docker-desktop/) and **Python 3.10+** (and CMake plus a C++ compiler for fast C++ training). Heavier setup than Option A; recommended for development and long training runs.

NOTE: Training AI agents (reinforcement learning) by playing thousands of games requires powerful hardware. By sharing this work I hope we can pool resources to collaboratively train AI agents to improve Flesh & Blood resources for players.

**Windows (PowerShell):**

```powershell
git clone https://github.com/pdfosborne/flesh-and-blood-rlbridge.git
cd flesh-and-blood-rlbridge
.\scripts\setup.ps1
fab-gui
```

**macOS / Linux:**

```bash
git clone https://github.com/pdfosborne/flesh-and-blood-rlbridge.git
cd flesh-and-blood-rlbridge
chmod +x scripts/setup.sh
sudo chown -R "$USER:$USER" results Talishar .venv
./scripts/setup.sh
source .venv/bin/activate && fab-gui
```

`setup` creates a venv, installs the package, verifies `Talishar/Assets`, and starts the Talishar backend via Docker.


#### Starting and stopping

```bash
# Backend only (recommended for training)
python start_talishar.py --backend-only
fab-bridge talishar --backend-only

# Backend + FE together
python start_talishar.py

# Stop backend containers
python start_talishar.py --down
fab-bridge talishar --down
```

Or use Docker Compose from the repo root (starts backend + GUI + host CLI wrappers):

```bash
./scripts/docker-setup.sh --foreground
source scripts/docker-env.sh   # once per shell; then fab-tui / fab-gui / fab-bridge
```

#### Talishar-FE

If you use [Option B](#option-b---local-python-full-build-train-custom-agents) without the root Docker stack, clone and start Talishar-FE from your `flesh-and-blood-rlbridge` checkout:

```bash
git clone https://github.com/Talishar/Talishar-FE
python start_talishar.py --fe-only
```

### Commands

| Command | What it does |
|---------|----------------|
| `fab-gui` | Web GUI for sideboard comparison (http://localhost:8765) |
| `fab-tui` | Interactive terminal menu |
| `fab-bridge init` | First-time setup: dirs + Talishar backend + agent ensure |
| `fab-bridge agents ensure` | Sync official weights (when compatible) or install bootstrap placeholders |
| `fab-bridge agents sync` | Download official unified agent weights from manifest only |
| `fab-bridge agents publish` | Publish local weights to GitHub Releases (maintainers) |
| `fab-bridge doctor` | Check Python, Docker, Assets, card DB |
| `fab-bridge talishar --down` | Stop Talishar Docker containers |
| `python start_talishar.py --backend-only` | Start Talishar only (cross-platform) |

Legacy entry points `python main.py` and `python main.py gui` still work.

---

## Overview

This package wraps the [Talishar](https://talishar.net/) server into Gym-compatible RL environments and provides a full training/evaluation/simulation stack:

- **Self-play training** with PPO dual-agent pipelines
- **Three-phase workflow** - tune decks via targeted swaps (Phase 1), auto sideboard per matchup with manual override (Phase 2), train play agents (Phase 3)
- **Format-rule enforcement** - deck-size caps (e.g. Silver Age = 40), hero + full equipment required, token cards excluded from counts
- **FaBrary integration** - fetch real decks by URL or slug; auto-sideboard to meet format rules
- **C++ fast simulation** - compiled pybind11 engine replaces the HTTP game server for training (up to 50× faster, fully thread-safe)
- **MCP tools** - expose training and simulation tasks to AI assistants via the rlbridge MCP plugin


---

## Interactive launcher

| Entry point | Description |
|-------------|-------------|
| `fab-gui` | Web GUI for sideboard comparison |
| `fab-tui` | Rich-based terminal UI (default menu) |
| `fab-bridge sage` | Non-interactive SAGE deckbuilder pipeline |
| `python main.py` | Same as `fab-tui` (compatibility) |

### TUI main menu

| # | Option | What it does |
|---|--------|----------------|
| **1** | Sideboard comparison | Full sideboard-tuning pipeline: pick your deck and a SAGE precon opponent, generate swap variants, train play agents in parallel, and compare candidates. |
| **2** | Fixed deck simulation | Train play agents on two fixed decks (FaBrary URL/slug or local JSON) with optional C++ engine acceleration. |
| **3** | Evaluate checkpoints | Win-rate evaluation or GIF render-only replay from saved phase-3 checkpoints; can watch for new checkpoints. |
| **4** | Evaluate trained agent | Run additional Talishar eval games against a **completed** training run (latest checkpoint). |
| **5** | Real-time Talishar play | Open the live Talishar frontend in Chromium to **watch** the agent or **play against it**. |
| **6** | Settings | Talishar backend/FE URLs, Assets path, FaBrary API key; rescan `cards.json`; normalize card IDs for Talishar. |

**Talishar requirements by menu option:**

- Options **3–5** need the Talishar backend (`fab-bridge init` or `python start_talishar.py --backend-only`).
- Option **5** also needs Talishar-FE (`http://localhost:5173`; started automatically with Docker).

---

## Prerequisites

| Dependency | Minimum version | Notes |
|------------|----------------|-------|
| Python | 3.10 | 3.12 recommended |
| Docker Desktop | Latest | Talishar backend (required for GUI training eval) |
| Node.js / npm | 18 | Talishar-FE only (live play / GIF) |
| CMake | 3.21 | C++ engine build (optional, faster training) |
| C++ compiler | MSVC 2022 / GCC 11 | C++ engine build |

Run `fab-bridge doctor` to verify your environment.

---

## Training engine backends

Training defaults to the **Talishar fast backend** (`talishar_backend=fast`): optimized HTTP with optional `RLStep.php` overlay installed from `docker/talishar/rl-bridge/` at container start. No edits to upstream `Talishar/` are required.

| Backend | Default? | Fidelity | Typical speed | Use case |
|---------|----------|----------|---------------|----------|
| **Fast (Talishar)** | Yes | Full PHP rules | ~12–25 steps/s | Training rollouts |
| HTTP (legacy) | No | Full PHP rules | ~2 steps/s | Debugging |
| C++ (opt-in) | No | Stub approximate | 5–10 games/s | Future fast rollouts when parity complete |

Benchmark backends:

```bash
python scripts/benchmark_talishar_backends.py --base-url http://localhost:8080/game
python scripts/benchmark_talishar_backends.py --profile-rlstep --steps 30
```

After updating `docker/talishar/rl-bridge/` overlays, restart the backend so copies land in the container: `python start_talishar.py --backend-only`. Training RLStep requests set `trainingMode` automatically and use the minimal `BuildRLGameState.php` response builder.

Set `TALISHAR_URL=http://localhost:8080/game` (Docker compose default).


---

## Card Database

Card and hero metadata live in `src/flesh_and_blood_rlbridge/card_db/`. The database (`cards.json`, ~6,900 cards) is the authoritative source for equipment slots, weapon types, hero variants, and token identification.

To refresh from an upstream export:

```bash
cd src/flesh_and_blood_rlbridge/card_db
python import_from_talishar.py --source /path/to/upstream_cards.json --out cards.json
```

---

## Repository

https://github.com/pdfosborne/flesh-and-blood-rlbridge
