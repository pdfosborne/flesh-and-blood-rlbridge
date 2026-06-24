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

_ensure_talishar_fe

_print_cli_hint() {
  echo ""
  echo "FAB CLI (no host venv):"
  echo "  source scripts/docker-env.sh"
  echo "  fab-tui          # terminal UI"
  echo "  fab-gui          # open web GUI"
  echo "  fab-bridge ...   # other CLI subcommands"
  echo ""
}

if $FOREGROUND; then
  _print_cli_hint
  exec docker compose up --build "$@"
fi

docker compose up --build -d "$@"

echo "Waiting for fab-bridge..."
for _ in $(seq 1 90); do
  if docker compose ps fab-bridge --status running -q 2>/dev/null | grep -q .; then
    break
  fi
  sleep 1
done

echo "Web GUI: http://localhost:8765"
echo "Talishar-FE: http://localhost:5173"
_print_cli_hint
