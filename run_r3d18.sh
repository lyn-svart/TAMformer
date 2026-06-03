#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_r3d18.sh - R3D-18 baseline aligned with configs_custom_json_motion_location.yaml
#
# Usage:
#   bash run_r3d18.sh                        # uses DEFAULT_SOURCE below
#   bash run_r3d18.sh /path/to/dataset       # override source path only
#   EPOCHS=10 bash run_r3d18.sh              # override any env variable
#
# Compare motion test metrics to:
#   python run.py --config_file configs/configs_custom_json_motion_location.yaml
#
# All variables can be overridden via environment:
#   SOURCE=/data/ds BATCH=64 EPOCHS=20 LR=1e-4 bash run_r3d18.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Same dataset root as TAMformer data_opts.path_to_frames_root (override on server)
SOURCE="${1:-${SOURCE:-/path/to/PreventionData}}"

SAVE_DIR="${SAVE_DIR:-}"
RESULTS_DIR="${RESULTS_DIR:-}"                  # empty → TAMformer/ (folder with r3d18.py)
FRAMES_ROOT="${FRAMES_ROOT:-}"                  # empty → same as SOURCE

TRAIN_JSON="${TRAIN_JSON:-}"
VAL_JSON="${VAL_JSON:-}"
TEST_JSON="${TEST_JSON:-}"

# TAMformer: obs_length=10, chunk_dt=10, obs_seconds=1, interval=10, fstride=1
CLIP_LEN="${CLIP_LEN:-10}"
CHUNK_STRIDE="${CHUNK_STRIDE:-1}"
CROP_PAD="${CROP_PAD:-0.10}"
INPUT_SIZE="${INPUT_SIZE:-112}"
SKIP_CROP_RESIZE="${SKIP_CROP_RESIZE:-}"   # set to 1 for full-frame resize benchmark (no bbox crop)
BENCHMARK_DUMMY_READ="${BENCHMARK_DUMMY_READ:-}"  # set to 1: read frames, feed zeros to model

# TAMformer motion_location config: batch 64, lr 1e-4, epochs 20, no class weights
BATCH="${BATCH:-64}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-20}"
FRACTION="${FRACTION:-1.0}"
WORKERS="${WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
CACHE_SIZE="${CACHE_SIZE:-50000}"
COMPILE="${COMPILE:-}"
LOG_EVERY_N_BATCHES="${LOG_EVERY_N_BATCHES:-50}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
SEED="${SEED:-42}"
FRAME_KEEP_MOD="${FRAME_KEEP_MOD:-1}"           # legacy CLI; sliding windows use consecutive frames

WEIGHTED_SAMPLER="${WEIGHTED_SAMPLER:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

echo "============================================================"
echo " R3D-18 Motion Classification (TAMformer-aligned)"
echo "============================================================"
echo "  SOURCE         : $SOURCE"
echo "  FRAMES_ROOT    : ${FRAMES_ROOT:-<same as SOURCE>}"
echo "  SAVE_DIR       : ${SAVE_DIR:-<auto>}"
echo "  RESULTS_DIR    : ${RESULTS_DIR:-<repo folder with r3d18.py>}"
echo "  CLIP_LEN (T)   : $CLIP_LEN"
echo "  CHUNK_STRIDE   : $CHUNK_STRIDE"
echo "  INPUT_SIZE     : $INPUT_SIZE"
echo "  SKIP_CROP      : ${SKIP_CROP_RESIZE:-off (bbox crop)}"
echo "  DUMMY_READ     : ${BENCHMARK_DUMMY_READ:-off}"
echo "  BATCH_SIZE     : $BATCH"
echo "  LR             : $LR"
echo "  EPOCHS         : $EPOCHS"
echo "  PATIENCE       : $PATIENCE"
echo "  FRACTION       : $FRACTION"
echo "  WORKERS        : $WORKERS"
echo "  PREFETCH       : $PREFETCH_FACTOR"
echo "  CACHE_SIZE     : $CACHE_SIZE"
echo "  COMPILE        : ${COMPILE:-off}"
echo "  LOG_EVERY      : $LOG_EVERY_N_BATCHES batches"
echo "  MAX_BATCHES    : ${MAX_TRAIN_BATCHES:-0 (full epoch)}"
echo "  SEED           : $SEED"
echo "============================================================"
echo ""

EXTRA_FLAGS=""
[ -n "$SAVE_DIR"       ] && EXTRA_FLAGS="$EXTRA_FLAGS --save-dir       $SAVE_DIR"
[ -n "$RESULTS_DIR"    ] && EXTRA_FLAGS="$EXTRA_FLAGS --results-dir    $RESULTS_DIR"
[ -n "$FRAMES_ROOT"    ] && EXTRA_FLAGS="$EXTRA_FLAGS --frames-root    $FRAMES_ROOT"
[ -n "$TRAIN_JSON"     ] && EXTRA_FLAGS="$EXTRA_FLAGS --train-json     $TRAIN_JSON"
[ -n "$VAL_JSON"       ] && EXTRA_FLAGS="$EXTRA_FLAGS --val-json       $VAL_JSON"
[ -n "$TEST_JSON"      ] && EXTRA_FLAGS="$EXTRA_FLAGS --test-json      $TEST_JSON"
[ -n "$WEIGHTED_SAMPLER" ] && EXTRA_FLAGS="$EXTRA_FLAGS --weighted-sampler"
[ -n "$COMPILE"        ] && EXTRA_FLAGS="$EXTRA_FLAGS --compile"
[ -n "$SKIP_CROP_RESIZE" ] && EXTRA_FLAGS="$EXTRA_FLAGS --skip-crop-resize"
[ -n "$BENCHMARK_DUMMY_READ" ] && EXTRA_FLAGS="$EXTRA_FLAGS --benchmark-dummy-read"

$PYTHON "$SCRIPT_DIR/r3d18.py" \
    --source         "$SOURCE"        \
    --clip-len       "$CLIP_LEN"      \
    --chunk-stride   "$CHUNK_STRIDE"  \
    --crop-pad       "$CROP_PAD"      \
    --input-size     "$INPUT_SIZE"    \
    --batch-size     "$BATCH"         \
    --lr             "$LR"            \
    --weight-decay   "$WEIGHT_DECAY"  \
    --epochs         "$EPOCHS"        \
    --patience       "$PATIENCE"      \
    --use-fraction   "$FRACTION"      \
    --num-workers    "$WORKERS"       \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --cache-size     "$CACHE_SIZE"    \
    --log-every-n-batches "$LOG_EVERY_N_BATCHES" \
    --max-train-batches   "$MAX_TRAIN_BATCHES" \
    --seed           "$SEED"          \
    --frame-keep-mod "$FRAME_KEEP_MOD" \
    $EXTRA_FLAGS

echo ""
echo "Done."
