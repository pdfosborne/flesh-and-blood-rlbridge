#!/usr/bin/env bash
# First-time local setup: venv, editable install, Talishar backend, then launch GUI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON not found" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment in .venv"
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[gui]"

fab-bridge init

echo ""
echo "Setup complete. Start the web GUI with:"
echo "  source .venv/bin/activate && fab-gui"
echo ""
echo "Or use Docker for Talishar + GUI together:"
echo "  docker compose up --build"
