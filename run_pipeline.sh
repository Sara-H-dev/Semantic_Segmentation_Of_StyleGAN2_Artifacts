#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# config
# ----------------------------
# ----------------------------
# helpers
# ----------------------------
echo_info() { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
echo_warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
echo_err()  { printf "\033[1;31m[ERR ]\033[0m %s\n" "$*"; }


# ----------------------------
# 1) training
# ----------------------------
echo_info "Start training …"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

TIMESTAMP="$(
python3 train.py \
    --cfg "./config.yaml" \
  | tail -n 1
)"

echo_info "Training completed."

# ----------------------------
# 2) test
# ----------------------------
echo_info "Start test with timestamp='${TIMESTAMP}' (Split=${TEST_SPLIT}) …"

python test.py \
  --timestamp "${TIMESTAMP}" \

echo_info "Done. Results are in: ${OUTPUT_DIR}"
