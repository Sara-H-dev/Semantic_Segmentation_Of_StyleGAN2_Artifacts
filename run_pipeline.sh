#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# config
# ----------------------------
CFG="${CFG:-./config.yaml}"
ROOT_PATH="${ROOT_PATH:-./dataset}"
LIST_DIR="${LIST_DIR:-./lists}"
OUTPUT_DIR="${OUTPUT_DIR:-./model_out}"

N_GPU="${N_GPU:-1}"
DETERMINISTIC="${DETERMINISTIC:-True}"
USE_CHECKPOINT="${USE_CHECKPOINT:-False}"

IMG_SIZE="${IMG_SIZE:-1024}"
NUM_CLASSES="${NUM_CLASSES:-1}"
SEED="${SEED:-1234}"
FREEZE_ENCODER="${FREEZE_ENCODER:-1}"
# hyperparameter
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}"
MAX_EPOCHS_TRAIN="${MAX_EPOCHS_TRAIN:-30000}"
BASE_LR="${BASE_LR:-0.01}"
SIG_THRESHOLD="${SIG_THRESHOLD:-0.5}"
EARLY_STOP="${EARLY_STOP:-100}"
LOSS_BETA="${LOSS_BETA:-0.6}"
LOSS_ALPHA="${LOSS_ALPHA:-0.4}"
UNFREEZE_STAGE3="${UNFREEZE_STAGE3:-0.4}"
UNFREEZE_STAGE2="${UNFREEZE_STAGE2:-0.7}"
UNFREEZE_STAGE1="${UNFREEZE_STAGE1:-0.9}"
UNFREEZE_STAGE0="${UNFREEZE_STAGE0:-0.98}"

# test parameter
TEST_SPLIT="${TEST_SPLIT:-test}"   # "test" or "val"
IS_SAVENII="${IS_SAVENII:-True}"
TEST_SEED="${TEST_SEED:-100}"
TEST_USE_CHECKPOINT="${TEST_USE_CHECKPOINT:-False}"

# ----------------------------
# helpers
# ----------------------------
echo_info() { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
echo_warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
echo_err()  { printf "\033[1;31m[ERR ]\033[0m %s\n" "$*"; }

to_bool() {
  # case-insensitive: True/true/1/yes -> 0 (success), else 1
  case "${1,,}" in
    true|1|yes|y) return 0 ;;
    *)            return 1 ;;
  esac
}

# ----------------------------
# flags 
# ----------------------------
DETERMINISTIC_FLAG=""
IS_SAVENII_FLAG=""
TEST_USE_CHECKPOINT_FLAG=""
USE_CHECKPOINT_FLAG=""
ACCUM_FLAG=""

to_bool "${DETERMINISTIC}" && DETERMINISTIC_FLAG="--deterministic"
to_bool "${IS_SAVENII}" && IS_SAVENII_FLAG="--is_savenii"
to_bool "${TEST_USE_CHECKPOINT}" && TEST_USE_CHECKPOINT_FLAG="--use_checkpoint"
to_bool "${USE_CHECKPOINT}" && USE_CHECKPOINT_FLAG="--use_checkpoint"

if [ -n "${ACCUMULATION_STEPS}" ]; then
  ACCUM_FLAG="--accumulation_steps ${ACCUMULATION_STEPS}"
fi

# ----------------------------
# 1) training
# ----------------------------
echo_info "Start training …"

TIMESTAMP="$(python train.py \
  --cfg "${CFG}" \
  --root_path "${ROOT_PATH}" \
  --list_dir "${LIST_DIR}" \
  --n_gpu "${N_GPU}" \
  --output_dir "${OUTPUT_DIR}" \
  --img_size "${IMG_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_epochs "${MAX_EPOCHS_TRAIN}" \
  --base_lr "${BASE_LR}" \
  --num_classes "${NUM_CLASSES}" \
  --seed "${SEED}" \
  --freeze_encoder "${FREEZE_ENCODER}" \
  --sig_threshold "${SIG_THRESHOLD}" \
  --early_stopping_patience "${EARLY_STOP}" \
  --loss_alpha "${LOSS_ALPHA}" \
  --loss_beta "${LOSS_BETA}" \
  --unfreeze_stage3 "${UNFREEZE_STAGE3}" \
  --unfreeze_stage2 "${UNFREEZE_STAGE2}" \
  --unfreeze_stage1 "${UNFREEZE_STAGE1}" \
  --unfreeze_stage0 "${UNFREEZE_STAGE0}" \
  ${DETERMINISTIC_FLAG} \
  ${ACCUM_FLAG} \
  ${USE_CHECKPOINT_FLAG} \
  | tail -n 1)"

echo_info "Training completed."

# ----------------------------
# 2) test
# ----------------------------
echo_info "Start test with timestamp='${TIMESTAMP}' (Split=${TEST_SPLIT}) …"

python test.py \
  --dataset_path "${ROOT_PATH}" \
  --num_classes "${NUM_CLASSES}" \
  --list_dir "${LIST_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_epochs "${MAX_EPOCHS_TRAIN}" \
  --img_size "${IMG_SIZE}" \
  --sig_threshold "${SIG_THRESHOLD}" \
  --split "${TEST_SPLIT}" \
  --timestamp "${TIMESTAMP}" \
  ${IS_SAVENII_FLAG} \
  ${DETERMINISTIC_FLAG} \
  --seed "${TEST_SEED}" \
  ${TEST_USE_CHECKPOINT_FLAG}

echo_info "Done. Results are in: ${OUTPUT_DIR}"
