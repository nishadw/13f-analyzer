#!/usr/bin/env sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "[start] .venv missing. Creating virtualenv..."
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv "$ROOT/.venv"
  else
    echo "[start] ERROR: python3 not found in PATH"
    exit 1
  fi
fi

if [ ! -d "$ROOT/frontend/node_modules" ]; then
  echo "[start] frontend/node_modules missing. Installing frontend deps..."
  npm --prefix "$ROOT/frontend" install
fi

if [ ! -f "$ROOT/data/signals_cache.json" ]; then
  echo "[start] signals cache missing. Building cache..."
  "$ROOT/.venv/bin/python" "$ROOT/scripts/refresh_signals.py"
fi

echo "[start] Launching app (backend + frontend)..."
exec bash "$ROOT/run.sh" --no-reload
