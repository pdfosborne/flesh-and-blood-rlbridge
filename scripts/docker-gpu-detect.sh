#!/usr/bin/env bash
# Detect whether Docker can pass an NVIDIA GPU into containers.
# Sets FAB_DOCKER_GPU=1 when a quick `docker run --gpus all` smoke test succeeds.
set -euo pipefail

fab_docker_gpu_detect() {
  FAB_DOCKER_GPU=0

  if ! command -v docker >/dev/null 2>&1; then
    return 0
  fi
  if ! docker info >/dev/null 2>&1; then
    return 0
  fi

  # Fast reject when the host has no NVIDIA tooling at all.
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  # Confirm Docker Desktop / nvidia-container-toolkit can expose the GPU.
  if docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    FAB_DOCKER_GPU=1
  fi
  return 0
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  fab_docker_gpu_detect
  echo "${FAB_DOCKER_GPU}"
fi
