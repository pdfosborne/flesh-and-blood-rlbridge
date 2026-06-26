#!/usr/bin/env bash
# Build and start the Docker stack; enable host fab-* wrappers in bin/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FOLLOW_LOGS=false
EVAL_ONLY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--foreground)
      shift
      ;;
    --logs)
      FOLLOW_LOGS=true
      shift
      ;;
    --eval)
      EVAL_ONLY=true
      shift
      ;;
    *)
      break
      ;;
  esac
done

chmod +x bin/fab-* 2>/dev/null || true

_ensure_talishar_fe() {
  if $EVAL_ONLY; then
    return 0
  fi
  if [[ ! -f Talishar-FE/package.json ]]; then
    echo "[setup] Cloning Talishar-FE..."
    git clone --depth 1 https://github.com/Talishar/Talishar-FE Talishar-FE
  fi
}

_prepare_results_dir() {
  mkdir -p results/agent_cache
  export HOST_UID="$(id -u)"
  export HOST_GID="$(id -g)"
  if [[ -d results ]] && [[ ! -w results ]]; then
    echo "[setup] Fixing ownership of results/ (often created by Docker as root)..."
    if command -v sudo >/dev/null 2>&1; then
      sudo chown -R "${HOST_UID}:${HOST_GID}" results 2>/dev/null || true
    fi
    if [[ ! -w results ]]; then
      echo "[setup] WARNING: results/ is not writable. Run: sudo chown -R \"\$USER:\$USER\" results"
    fi
  fi
}

_wait_for_url() {
  local url="$1"
  local label="$2"
  local max_secs="${3:-120}"
  echo -n "[setup] ${label}..."
  for _ in $(seq 1 "$max_secs"); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      echo " OK"
      return 0
    fi
    sleep 1
    echo -n "."
  done
  echo " TIMEOUT (may still be starting — check: docker compose ps)"
  return 1
}

_wait_for_stack() {
  local step=1
  local total=5

  echo ""
  echo "[setup] Waiting for all services (Talishar-FE can take several minutes on first run)..."
  echo ""

  if $EVAL_ONLY; then
    total=3
    _wait_for_url "http://localhost:8080/" "Step 1/${total}: Talishar backend" 120 || true
    _wait_for_url "http://localhost:8765/" "Step 2/${total}: Web GUI (fab-bridge)" 120 || true
    echo -n "[setup] Step 3/${total}: CLI wrappers..."
    docker compose wait fab-cli-setup 2>/dev/null || true
    echo " OK"
    echo ""
    return
  fi

  echo "[setup] Step ${step}/${total}: Talishar-FE clone"
  step=$((step + 1))
  docker compose wait talishar-fe-clone 2>/dev/null || true
  echo "[setup]   done."

  _wait_for_url "http://localhost:8080/" "Step ${step}/${total}: Talishar backend" 120 || true
  step=$((step + 1))

  _wait_for_url "http://localhost:5173/" "Step ${step}/${total}: Talishar-FE (Vite)" 300 || true
  step=$((step + 1))

  _wait_for_url "http://localhost:8765/" "Step ${step}/${total}: Web GUI (fab-bridge)" 120 || true
  step=$((step + 1))

  echo -n "[setup] Step ${step}/${total}: CLI wrappers..."
  docker compose wait fab-cli-setup 2>/dev/null || true
  echo " OK"
  echo ""
}

_print_ready_banner() {
  if $EVAL_ONLY && [[ -f "$ROOT/docker/ready-message-eval.txt" ]]; then
    cat "$ROOT/docker/ready-message-eval.txt"
    return
  fi
  if [[ -f "$ROOT/docker/ready-message.txt" ]]; then
    cat "$ROOT/docker/ready-message.txt"
  else
    echo ""
    echo "Setup complete. Open http://localhost:8765"
    echo ""
  fi
}

_ensure_talishar_fe

_prepare_results_dir

# shellcheck source=scripts/docker-gpu-compose.sh
source "$ROOT/scripts/docker-gpu-compose.sh"
fab_docker_gpu_compose_init
fab_docker_gpu_compose_note

if $EVAL_ONLY; then
  export FAB_DOCKER_STACK=eval
  # shellcheck source=scripts/docker-eval-compose.sh
  source "$ROOT/scripts/docker-eval-compose.sh"
  fab_docker_eval_compose_init
  fab_docker_eval_compose_note
  COMPOSE_PROFILES="eval"
else
  export FAB_DOCKER_STACK=full
  COMPOSE_PROFILES="full"
fi

echo "[setup] Building and starting Docker stack..."
docker compose ${COMPOSE_PROFILES:+--profile "$COMPOSE_PROFILES"} up --build -d "$@"

_wait_for_stack
_print_ready_banner

if $FOLLOW_LOGS; then
  echo "[setup] Following container logs (Ctrl+C exits log view only; run 'docker compose down' to stop)."
  echo ""
  exec docker compose logs -f --tail=50
fi
