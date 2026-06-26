#!/usr/bin/env bash
# Configure COMPOSE_FILE for eval-only (slim) or full stack overlays.
set -euo pipefail

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

fab_docker_eval_compose_init() {
  local mode="${FAB_DOCKER_STACK:-}"
  if [[ "${mode}" == "eval" ]]; then
    export COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}:docker-compose.eval.yml"
  fi
}

fab_docker_eval_compose_note() {
  if [[ "${FAB_DOCKER_STACK:-}" == "eval" ]]; then
    echo "[docker] Eval stack — Talishar backend + GUI (no Talishar-FE / Playwright)"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  FAB_DOCKER_STACK="${1:-eval}"
  fab_docker_eval_compose_init
  echo "COMPOSE_FILE=${COMPOSE_FILE}"
fi
