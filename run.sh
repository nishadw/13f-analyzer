#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="$ROOT/.venv/bin/python"
BACKEND_PORT="${BACKEND_PORT:-7779}"
FRONTEND_PORT="${FRONTEND_PORT:-3001}"
RELOAD=1
DO_BOOTSTRAP=0
DO_INGEST=0
DO_TRAIN=0

kill_listeners() {
  local port="$1"
  local pids
  pids="$(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "Port ${port} in use by PID(s): ${pids}. Stopping..."
    # shellcheck disable=SC2086
    kill $pids >/dev/null 2>&1 || true
    sleep 1
    pids="$(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
      echo "Force stopping PID(s) on port ${port}: ${pids}"
      # shellcheck disable=SC2086
      kill -9 $pids >/dev/null 2>&1 || true
      sleep 1
    fi
  fi
}

assert_port_free() {
  local port="$1"
  local pids
  pids="$(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "ERROR: Port ${port} is still in use by PID(s): ${pids}"
    exit 1
  fi
}

usage() {
  cat <<USAGE
Usage: ./run.sh [options]

Options:
  --bootstrap    Install Python and frontend dependencies first
  --ingest       Run ingestion before starting servers
  --train        Run model training before starting servers
  --no-reload    Start backend without auto-reload
  --help         Show this message

Env overrides:
  BACKEND_PORT=7779
  FRONTEND_PORT=3001
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bootstrap) DO_BOOTSTRAP=1 ;;
    --ingest) DO_INGEST=1 ;;
    --train) DO_TRAIN=1 ;;
    --no-reload) RELOAD=0 ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtualenv python at: $PYTHON_BIN"
  echo "Create it first: python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required but not found in PATH"
  exit 1
fi

CERT="$ROOT/api_backend/cert.pem"
KEY="$ROOT/api_backend/key.pem"
if [[ ! -f "$CERT" || ! -f "$KEY" ]]; then
  echo "Generating local TLS cert for backend..."
  openssl req -x509 -newkey rsa:2048 -keyout "$KEY" -out "$CERT" -days 825 -nodes -subj '/CN=localhost' >/dev/null 2>&1
fi

if [[ "$DO_BOOTSTRAP" -eq 1 ]]; then
  echo "Installing Python deps..."
  "$PYTHON_BIN" -m pip install -r requirements.txt
  echo "Installing frontend deps..."
  npm --prefix frontend install
fi

if [[ "$DO_INGEST" -eq 1 ]]; then
  echo "Running ingestion..."
  "$PYTHON_BIN" scripts/ingest.py
fi

if [[ "$DO_TRAIN" -eq 1 ]]; then
  echo "Running training..."
  "$PYTHON_BIN" scripts/train.py
fi

if [[ ! -f "$ROOT/data/13f_holdings.parquet" ]]; then
  echo "Missing data/13f_holdings.parquet — running ingestion automatically..."
  "$PYTHON_BIN" scripts/ingest.py
fi

echo "Stopping any previous local servers on ports ${BACKEND_PORT}/${FRONTEND_PORT}..."
kill_listeners "$BACKEND_PORT"
kill_listeners "$FRONTEND_PORT"
assert_port_free "$BACKEND_PORT"
assert_port_free "$FRONTEND_PORT"

BACKEND_CMD=("$PYTHON_BIN" scripts/serve.py --host 0.0.0.0 --port "$BACKEND_PORT")
if [[ "$RELOAD" -eq 1 ]]; then
  BACKEND_CMD+=(--reload)
fi

FRONTEND_CMD=(npm --prefix frontend run dev -- --port "$FRONTEND_PORT")

echo "Starting backend: https://localhost:${BACKEND_PORT}"
"${BACKEND_CMD[@]}" &
BACKEND_PID=$!

echo "Starting frontend: http://localhost:${FRONTEND_PORT}/stocks"
"${FRONTEND_CMD[@]}" &
FRONTEND_PID=$!

cleanup() {
  printf "\nStopping services...\n"
  kill "$BACKEND_PID" "$FRONTEND_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo ""
echo "App URLs:"
echo "  Frontend: http://localhost:${FRONTEND_PORT}/stocks"
echo "  Backend:  https://localhost:${BACKEND_PORT}"
echo ""
echo "Press Ctrl+C to stop both services."

while true; do
  if ! kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    wait "$BACKEND_PID" || true
    echo "Backend process exited. Stopping frontend."
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
    exit 1
  fi

  if ! kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
    wait "$FRONTEND_PID" || true
    echo "Frontend process exited. Stopping backend."
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
    exit 1
  fi

  sleep 1
done
