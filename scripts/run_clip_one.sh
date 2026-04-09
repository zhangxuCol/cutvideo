#!/usr/bin/env zsh
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <target_video_path>"
  exit 2
fi

TARGET="$1"
CFG="/Users/zhangxu/.codex/worktrees/ce87/cutvideo/configurations/ai_pipeline.defaults.json"
PY_BIN="/Users/zhangxu/.pyenv/versions/3.12.10/bin/python3"

if [[ ! -f "$TARGET" ]]; then
  echo "target not found: $TARGET"
  exit 3
fi

base=$(basename "$TARGET")
stem="${base%.*}"
target_dir=$(cd "$(dirname "$TARGET")" && pwd)

# Auto-detect project layout by target parent directory.
# Supported:
#   <project>/素材/<file>
#   <project>/adx原/<file>
if [[ "$(basename "$target_dir")" == "素材" || "$(basename "$target_dir")" == "adx原" ]]; then
  PROJECT_DIR=$(cd "$target_dir/.." && pwd)
else
  echo "unsupported target path: $TARGET"
  echo "expect target under <project>/素材 or <project>/adx原"
  exit 6
fi

SRC="$PROJECT_DIR/剧集"
OUT_DIR="$PROJECT_DIR/output"
CACHE_DIR="$OUT_DIR/.cache_fast_v7"
out="$OUT_DIR/${stem}_V3_FAST.mp4"
cache="$CACHE_DIR/${stem}"

if [[ ! -d "$SRC" ]]; then
  echo "source dir not found: $SRC"
  exit 5
fi

mkdir -p "$OUT_DIR" "$cache"
echo "[clip-one] target=$TARGET"
echo "[clip-one] project=$PROJECT_DIR"
echo "[clip-one] source=$SRC"
echo "[clip-one] out=$out"
echo "[clip-one] python=$PY_BIN"

"$PY_BIN" /Users/zhangxu/.codex/worktrees/ce87/cutvideo/fast_v7.py \
  --config "$CFG" \
  --target "$TARGET" \
  --source-dir "$SRC" \
  --output "$out" \
  --cache "$cache" \
  --no-run-evidence-validation \
  --no-run-ai-verify-snapshots

if [[ ! -f "$out" ]]; then
  echo "[clip-one] output missing: $out"
  exit 4
fi

echo "[clip-one] done: $out"
