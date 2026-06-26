#!/usr/bin/env bash
# Configure COMPOSE_FILE for optional GPU support on fab-bridge.
set -euo pipefail

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/docker-gpu-detect.sh
source "$_ROOT/scripts/docker-gpu-detect.sh"

fab_docker_gpu_compose_init() {
  fab_docker_gpu_detect
  export FAB_DOCKER_GPU
  if [[ "${FAB_DOCKER_GPU}" == "1" ]]; then
    export COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}:docker-compose.gpu.yml"
  else
    export COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
  fi
}

fab_docker_gpu_compose_note() {
  if [[ "${FAB_DOCKER_GPU:-0}" == "1" ]]; then
    echo "[docker] GPU detected — fab-bridge will use CUDA PyTorch (gpus: all)"
  else
    echo "[docker] No Docker GPU — fab-bridge training uses CPU PyTorch"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  fab_docker_gpu_compose_init
  fab_docker_gpu_compose_note
  echo "COMPOSE_FILE=${COMPOSE_FILE}"
fi
