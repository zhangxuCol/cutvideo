#!/usr/bin/env zsh
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <target_video_path>"
  exit 2
fi

TARGET="$1"
SRC="/Users/zhangxu/work/项目/南城以北/剧集"
OUT_DIR="/Users/zhangxu/work/项目/南城以北/output"
CACHE_DIR="$OUT_DIR/.cache_fast_v7"
CFG="/Users/zhangxu/.codex/worktrees/ce87/cutvideo/configurations/ai_pipeline.defaults.json"

if [[ ! -f "$TARGET" ]]; then
  echo "target not found: $TARGET"
  exit 3
fi

base=$(basename "$TARGET")
stem="${base%.*}"
out="$OUT_DIR/${stem}_V3_FAST.mp4"
cache="$CACHE_DIR/${stem}"

mkdir -p "$OUT_DIR" "$cache"
echo "[clip-one] target=$TARGET"
echo "[clip-one] out=$out"

python3 /Users/zhangxu/.codex/worktrees/ce87/cutvideo/fast_v7.py \
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
