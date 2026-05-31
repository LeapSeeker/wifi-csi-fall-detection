#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.codex-test-venv/Scripts/python.exe"
RCLONE="$ROOT/.local/rclone/rclone-v1.74.1-windows-amd64/rclone.exe"

if [[ ! -x "$PY" ]]; then
  echo "Python venv not found: $PY" >&2
  exit 1
fi

cd "$ROOT"

export SAFESIGNAL_DISABLE_INFERENCE=1
export SAFESIGNAL_DRIVE_UPLOAD=1
export SAFESIGNAL_DRIVE_REMOTE="gdrive:SafeSignal_Dataset"
export SAFESIGNAL_RCLONE_BIN="$RCLONE"

"$SAFESIGNAL_RCLONE_BIN" lsd gdrive:
"$PY" server/main.py
