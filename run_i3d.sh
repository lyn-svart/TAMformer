#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_i3d.sh - Easy launcher for I3D motion classification training
#
# Usage:
#   bash run_i3d.sh                        # uses DEFAULT_SOURCE below
#   bash run_i3d.sh /path/to/dataset       # override source path only
#   EPOCHS=10 bash run_i3d.sh              # override any env variable
#
# All variables can be overridden via environment:
#   SOURCE=/data/ds BATCH=32 EPOCHS=50 LR=1e-4 bash run_i3d.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Configurable variables (override via env or positional arg)
SOURCE="${1:-${SOURCE:-/home/bdemirkan/MotionDetection/02-Source/PreventionDataset}}"
SAVE_DIR="${SAVE_DIR:-}"                        # empty → auto (<source>/../checkpoints)

TRAIN_JSON="${TRAIN_JSON:-}"                    # empty → <source>/Train_bahar.json
VAL_JSON="${VAL_JSON:-}"                        # empty → <source>/Validation_bahar.json
TEST_JSON="${TEST_JSON:-}"                      # empty → <source>/Test_bahar.json

CLIP_LEN="${CLIP_LEN:-10}"                      # -T
CROP_PAD="${CROP_PAD:-0.10}"
INPUT_SIZE="${INPUT_SIZE:-112}"

BATCH="${BATCH:-16}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
FRACTION="${FRACTION:-1.0}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-42}"
FRAME_KEEP_MOD="${FRAME_KEEP_MOD:-5}"

# Set to "--weighted-sampler" to enable WeightedRandomSampler
WEIGHTED_SAMPLER="${WEIGHTED_SAMPLER:-}"

# Script location (so it can be run from any directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

# Print config
echo "============================================================"
echo " I3D Motion Classification — Training Run"
echo "============================================================"
echo "  SOURCE       : $SOURCE"
echo "  SAVE_DIR     : ${SAVE_DIR:-<auto>}"
echo "  CLIP_LEN (T) : $CLIP_LEN"
echo "  INPUT_SIZE   : $INPUT_SIZE"
echo "  BATCH_SIZE   : $BATCH"
echo "  LR           : $LR"
echo "  EPOCHS       : $EPOCHS"
echo "  PATIENCE     : $PATIENCE"
echo "  FRACTION     : $FRACTION"
echo "  WORKERS      : $WORKERS"
echo "  SEED         : $SEED"
echo "============================================================"
echo ""

# Build optional flags
EXTRA_FLAGS=""
[ -n "$SAVE_DIR"    ] && EXTRA_FLAGS="$EXTRA_FLAGS --save-dir    $SAVE_DIR"
[ -n "$TRAIN_JSON"  ] && EXTRA_FLAGS="$EXTRA_FLAGS --train-json  $TRAIN_JSON"
[ -n "$VAL_JSON"    ] && EXTRA_FLAGS="$EXTRA_FLAGS --val-json    $VAL_JSON"
[ -n "$TEST_JSON"   ] && EXTRA_FLAGS="$EXTRA_FLAGS --test-json   $TEST_JSON"
[ -n "$WEIGHTED_SAMPLER" ] && EXTRA_FLAGS="$EXTRA_FLAGS --weighted-sampler"

# Run
$PYTHON "$SCRIPT_DIR/i3d.py" \
    --source        "$SOURCE"       \
    --clip-len      "$CLIP_LEN"     \
    --crop-pad      "$CROP_PAD"     \
    --input-size    "$INPUT_SIZE"   \
    --batch-size    "$BATCH"        \
    --lr            "$LR"           \
    --weight-decay  "$WEIGHT_DECAY" \
    --epochs        "$EPOCHS"       \
    --patience      "$PATIENCE"     \
    --use-fraction  "$FRACTION"     \
    --num-workers   "$WORKERS"      \
    --seed          "$SEED"         \
    --frame-keep-mod "$FRAME_KEEP_MOD" \
    $EXTRA_FLAGS

echo ""
echo "Done."