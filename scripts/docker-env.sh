#!/usr/bin/env bash
# Source from your shell to put Docker-backed fab-* commands on PATH.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$ROOT/bin:$PATH"
export FAB_DOCKER=1
