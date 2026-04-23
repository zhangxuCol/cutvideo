#!/usr/bin/env zsh
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <target_video_path> [extra fast_v7 args...]"
  exit 2
fi

TARGET="$1"
shift
EXTRA_ARGS=("$@")
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
CFG="$ROOT_DIR/configurations/ai_pipeline.defaults.json"
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
PROJECT_CACHE_DIR="$PROJECT_DIR/.cache_fast_v7"
FRAME_INDEX_CACHE_DIR="$PROJECT_CACHE_DIR/frame_index_shared"
out="$OUT_DIR/${stem}_V3_FAST.mp4"
cache="$CACHE_DIR/${stem}"

if [[ ! -d "$SRC" ]]; then
  echo "source dir not found: $SRC"
  exit 5
fi

mkdir -p "$OUT_DIR" "$cache" "$FRAME_INDEX_CACHE_DIR"
echo "[clip-one] target=$TARGET"
echo "[clip-one] project=$PROJECT_DIR"
echo "[clip-one] source=$SRC"
echo "[clip-one] out=$out"
echo "[clip-one] frame_index_cache=$FRAME_INDEX_CACHE_DIR"
echo "[clip-one] python=$PY_BIN"

report="${out%.mp4}.quality_report.json"
rm -f "$out" "$report"

"$PY_BIN" "$ROOT_DIR/fast_v7.py" \
  --config "$CFG" \
  --target "$TARGET" \
  --source-dir "$SRC" \
  --output "$out" \
  --cache "$cache" \
  --frame-index-cache-dir "$FRAME_INDEX_CACHE_DIR" \
  --no-run-evidence-validation \
  --no-run-ai-verify-snapshots \
  "${EXTRA_ARGS[@]}"

if [[ ! -f "$out" ]]; then
  echo "[clip-one] output missing: $out"
  exit 4
fi

if [[ -f "$report" ]]; then
  render_status=$(jq -r '.render_metrics.status // empty' "$report" 2>/dev/null || true)
  if [[ "$render_status" != "ok" ]]; then
    echo "[clip-one] render status is not ok: ${render_status:-unknown}"
    echo "[clip-one] report=$report"
    exit 7
  fi
fi

echo "[clip-one] done: $out"
