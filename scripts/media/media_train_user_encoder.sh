#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd )"

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_PROCS=${NUM_PROCS:-4}
GPU_IDS="${GPU_IDS:-0,1,2,3}"
BATCH_SIZE=${BATCH_SIZE:-128}
SEED=${SEED:-42}
WANDB_PROJECT="${WANDB_PROJECT}"

LEARNING_RATES=(1e-4)

TRAIN_PY="${SCRIPT_DIR}/media_train_user_encoder.py"

LORA_FLAGS="--use_lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.05"
BASE_FLAGS="--prefix_pos after_bos --bert_pool mean --gt_path embeds"

run_exp () {
  local lr="$1"
  local run_name="$2"
  local extra_flags="$3"
  local k_val="$4"

  accelerate launch --num_processes="${NUM_PROCS}" --gpu_ids="${GPU_IDS}" "${TRAIN_PY}" \
    --train_file "${PROJECT_ROOT}/dataset/user_split/media/train_w_detail.json" \
    --val_file "${PROJECT_ROOT}/dataset/user_split/media/val_w_detail.json" \
    --output_dir "${PROJECT_ROOT}/checkpoints/bert_mean/${BATCH_SIZE}/${lr}" \
    --epochs 10 \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${lr}" \
    --seed "${SEED}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${run_name}" \
    ${LORA_FLAGS} \
    ${BASE_FLAGS} \
    ${extra_flags}
}

for k in "${K_VALUES[@]}"; do
  for lr in "${LEARNING_RATES[@]}"; do
    run_name="lr${lr}_b${BATCH_SIZE}_ep10"
    run_exp "${lr}" "${run_name}" "${extra_flags}" 
  done
done
