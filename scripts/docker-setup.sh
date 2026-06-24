#!/usr/bin/env bash
# Build and start the Docker stack; enable host fab-* wrappers in bin/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FOREGROUND=false
if [[ "${1:-}" == "--foreground" ]] || [[ "${1:-}" == "-f" ]]; then
  FOREGROUND=true
  shift
fi

chmod +x bin/fab-* 2>/dev/null || true

_ensure_talishar_fe() {
  if [[ ! -f Talishar-FE/package.json ]]; then
    echo "Cloning Talishar-FE..."
    git clone --depth 1 https://github.com/Talishar/Talishar-FE Talishar-FE
  fi
}

_wait_for_service() {
  local service="$1"
  local url="$2"
  local label="$3"
  local max_secs="${4:-120}"
  echo -n "  ${label}..."
  for _ in $(seq 1 "$max_secs"); do
    if docker compose ps "$service" --status running -q 2>/dev/null | grep -q .; then
      if [[ -z "$url" ]] || curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
        echo " ready."
        return 0
      fi
    fi
    sleep 1
    echo -n "."
  done
  echo " still starting (check: docker compose ps)."
  return 1
}

_wait_for_stack() {
  echo ""
  echo "Waiting for services (Talishar-FE may take a few minutes on first run)..."
  docker compose wait talishar-fe-clone 2>/dev/null || true
  _wait_for_service web-server "http://localhost:8080/" "Talishar backend" 120 || true
  _wait_for_service talishar-fe "http://localhost:5173/" "Talishar-FE" 300 || true
  _wait_for_service fab-bridge "http://localhost:8765/" "web GUI" 120 || true
  docker compose wait fab-cli-setup 2>/dev/null || true
}

_print_ready_banner() {
  cat <<'EOF'

================================================================
  FAB RL Bridge — ready to use
================================================================

  WEB GUI — open in your browser:

    http://localhost:8765

  Pick your deck on the Decks tab, tune sideboard swaps on the Editor
  tab, then start training. Progress and final rankings appear on the
  Monitor tab; replay GIFs on the Results tab.

  Also running:

    Talishar backend   http://localhost:8080
    Talishar-FE        http://localhost:5173

----------------------------------------------------------------
  TERMINAL UI (fab-tui) — second terminal, same repo checkout
----------------------------------------------------------------

    source scripts/docker-env.sh
    fab-tui

  Suggested menu options:

    1  Sideboard comparison
       Full pipeline: your deck vs a SAGE precon, swap variants,
       parallel training, and ranked results.

    2  Fixed deck simulation
       Train agents on two fixed decks (FaBrary URL/slug or local JSON).

    3  Evaluate checkpoints
       Win-rate eval or GIF replay from saved training checkpoints.

    5  Real-time Talishar play
       Watch the agent or play against it in Chromium (uses Talishar-FE).

    6  Settings
       Talishar URLs, Assets path, FaBrary API key, card DB tools.

  Quick commands:

    fab-gui              open the web GUI in your browser
    fab-bridge doctor    check Docker, Assets, and card DB
    docker compose down  stop everything

================================================================

EOF
}

_ensure_talishar_fe

echo "Building and starting Docker stack..."
docker compose up --build -d "$@"

_wait_for_stack
_print_ready_banner

if $FOREGROUND; then
  echo "Following container logs (Ctrl+C exits log view; run 'docker compose down' to stop)."
  echo ""
  exec docker compose logs -f --tail=100
fi
