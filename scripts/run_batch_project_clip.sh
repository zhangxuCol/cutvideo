#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <project_dir> [workers]"
  exit 2
fi

PROJECT_DIR="$1"
WORKERS="${2:-2}"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RUN_ONE="$SCRIPT_DIR/run_clip_one.sh"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "project dir not found: $PROJECT_DIR"
  exit 3
fi
if [[ ! -x "$RUN_ONE" ]]; then
  echo "runner not executable: $RUN_ONE"
  exit 4
fi

if [[ -d "$PROJECT_DIR/素材" ]]; then
  MAT_DIR="$PROJECT_DIR/素材"
elif [[ -d "$PROJECT_DIR/adx原" ]]; then
  MAT_DIR="$PROJECT_DIR/adx原"
else
  echo "material dir not found under: $PROJECT_DIR (expect 素材 or adx原)"
  exit 5
fi

OUT_DIR="$PROJECT_DIR/output"
mkdir -p "$OUT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
TODO_LIST="$OUT_DIR/batch_clip_${STAMP}.todo.txt"
RUNTIME_LOG="$OUT_DIR/batch_clip_${STAMP}.runtime.log"
PROGRESS_LOG="$OUT_DIR/batch_clip_${STAMP}.progress.log"

find "$MAT_DIR" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' \) | sort | while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  b="$(basename "$f")"
  stem="${b%.*}"
  out="$OUT_DIR/${stem}_V3_FAST.mp4"
  [[ -f "$out" ]] || echo "$f"
done > "$TODO_LIST"

TOTAL="$(wc -l < "$TODO_LIST" | tr -d ' ')"
if [[ "$TOTAL" == "0" ]]; then
  echo "[batch] project=$PROJECT_DIR"
  echo "[batch] material_dir=$MAT_DIR"
  echo "[batch] output_dir=$OUT_DIR"
  echo "[batch] workers=$WORKERS"
  echo "[batch] nothing to do"
  exit 0
fi

echo "[batch] project=$PROJECT_DIR"
echo "[batch] material_dir=$MAT_DIR"
echo "[batch] output_dir=$OUT_DIR"
echo "[batch] workers=$WORKERS"
echo "[batch] todo=$TOTAL"
echo "[batch] todo_list=$TODO_LIST"
echo "[batch] runtime_log=$RUNTIME_LOG"
echo "[batch] progress_log=$PROGRESS_LOG"

batch_start="$(date +%s)"

cat "$TODO_LIST" | xargs -I{} -P"$WORKERS" bash -lc '
  f="$1"
  run_one="$2"
  runtime_log="$3"
  progress_log="$4"
  stem="$(basename "${f%.*}")"
  t0="$(date +%s)"
  echo "[start] $stem"
  if zsh "$run_one" "$f" >>"$runtime_log" 2>&1; then
    t1="$(date +%s)"
    echo "OK|$stem|$((t1-t0))s" >>"$progress_log"
    echo "[done]  OK   $stem ($((t1-t0))s)"
  else
    t1="$(date +%s)"
    echo "FAIL|$stem|$((t1-t0))s" >>"$progress_log"
    echo "[done]  FAIL $stem ($((t1-t0))s)"
  fi
' _ {} "$RUN_ONE" "$RUNTIME_LOG" "$PROGRESS_LOG"

OK_COUNT="$(grep -c '^OK|' "$PROGRESS_LOG" 2>/dev/null || true)"
FAIL_COUNT="$(grep -c '^FAIL|' "$PROGRESS_LOG" 2>/dev/null || true)"
BATCH_END="$(date +%s)"
BATCH_ELAPSED="$((BATCH_END-batch_start))"

echo "[summary] total=$TOTAL ok=$OK_COUNT fail=$FAIL_COUNT elapsed=${BATCH_ELAPSED}s"
echo "[summary] runtime_log=$RUNTIME_LOG"
echo "[summary] progress_log=$PROGRESS_LOG"
