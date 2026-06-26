#!/bin/bash
# Prepare bind-mounted results/ and ensure unified agents before the main command.
set -euo pipefail

RESULTS_DIR="${FAB_RESULTS_DIR:-/app/results}"
CACHE_DIR="${RESULTS_DIR}/agent_cache"

mkdir -p "${CACHE_DIR}"

if [ "$(id -u)" = "0" ] && [ -n "${HOST_UID:-}" ] && [ "${HOST_UID}" != "0" ]; then
  chown -R "${HOST_UID}:${HOST_GID:-${HOST_UID}}" "${RESULTS_DIR}" 2>/dev/null || true
fi

if [ "${FAB_SKIP_AGENT_ENSURE:-0}" != "1" ]; then
  fab-bridge agents ensure --manifest /app/agents/manifest.json || true
  if [ "$(id -u)" = "0" ] && [ -n "${HOST_UID:-}" ] && [ "${HOST_UID}" != "0" ]; then
    chown -R "${HOST_UID}:${HOST_GID:-${HOST_UID}}" "${RESULTS_DIR}" 2>/dev/null || true
  fi
fi

exec "$@"
