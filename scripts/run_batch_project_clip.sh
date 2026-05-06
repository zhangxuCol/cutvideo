#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PY_BIN="/Users/zhangxu/.pyenv/versions/3.12.10/bin/python3"
CONTROLLER="$SCRIPT_DIR/batch_clip_controller.py"

if [[ ! -x "$CONTROLLER" ]]; then
  echo "batch controller not executable: $CONTROLLER"
  exit 4
fi

exec "$PY_BIN" "$CONTROLLER" "$@"
