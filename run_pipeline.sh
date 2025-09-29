#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# config 
# ----------------------------
CFG="${CFG:-./config.yaml}"       
ROOT_PATH="${ROOT_PATH:-./dataset}"
LIST_DIR="${LIST_DIR:-./lists}"
OUTPUT_DIR="${OUTPUT_DIR:-./model_out}" 
IMG_SIZE="${IMG_SIZE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_EPOCHS_TRAIN="${MAX_EPOCHS:-30000}"
BASE_LR="${BASE_LR:-0.01}"
NUM_CLASSES="${NUM_CLASSES:-1}"
SEED="${SEED:-1234}"
FREEZE_ENCODER="${FREEZE_ENCODER:-1}"
SIG_THRESHOLD="${SIG_THRESHOLD:-0.5}"
EARLY_STOP="${EARLY_STOP:-15}"

# test parameter
TEST_SPLIT="${TEST_SPLIT:-test}"        # "test" or "val"
TEST_BATCH="${TEST_BATCH:-24}"
MAX_EPOCHS_TEST="${MAX_EPOCHS:-150}"

# ----------------------------
# helper
# ----------------------------
echo_info() { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
echo_warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
echo_err()  { printf "\033[1;31m[ERR ]\033[0m %s\n" "$*"; }

# ----------------------------
# 1) training
# ----------------------------
echo_info "Start training …"
TIMESTAMP=$(python train.py \
  --cfg "${CFG}" \
  --root_path "${ROOT_PATH}" \
  --list_dir "${LIST_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --img_size "${IMG_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_epochs "${MAX_EPOCHS_TRAIN}" \
  --base_lr "${BASE_LR}" \
  --num_classes "${NUM_CLASSES}" \
  --seed "${SEED}" \
  --freeze_encoder "${FREEZE_ENCODER}" \
  --sig_threshold "${SIG_THRESHOLD}" \
  --early_stopping_patience "${EARLY_STOP}")

echo_info "training completed"


# ----------------------------
# 3) run the test
# ----------------------------

echo_info "Start test with timestamp='${TIMESTAMP}' (Split=${TEST_SPLIT}) …"

python test.py \
  --dataset_path "${ROOT_PATH}" \
  --dataset "SegArtifact" \
  --num_classes "${NUM_CLASSES}" \
  --list_dir "${LIST_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_epochs "${MAX_EPOCHS_TEST}" \
  --batch_size "${TEST_BATCH}" \
  --img_size "${IMG_SIZE}" \
  --sig_threshold "${SIG_THRESHOLD}" \
  --split "${TEST_SPLIT}" \
  --timestamp "${TIMESTAMP}"

echo_info "Done. Results are located at: ${OUTPUT_DIR}"
