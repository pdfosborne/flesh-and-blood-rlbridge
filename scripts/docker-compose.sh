#!/usr/bin/env bash
# docker compose wrapper — auto-enables GPU overlay when CUDA is available.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/docker-gpu-compose.sh
source "$ROOT/scripts/docker-gpu-compose.sh"
fab_docker_gpu_compose_init
exec docker compose "$@"
