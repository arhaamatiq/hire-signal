#!/usr/bin/env bash
# Start the HireSignal API. Frees port 8000 if it's in use so you don't get "Address already in use".
set -e
cd "$(dirname "$0")/.."
PORT=8000
if lsof -i :$PORT >/dev/null 2>&1; then
  echo "Freeing port $PORT..."
  lsof -ti :$PORT | xargs kill 2>/dev/null || true
  sleep 2
fi
echo "Starting backend at http://localhost:$PORT"
.venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port $PORT
