#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$( cd -- "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd )"

PYTHON_BIN="${PYTHON_BIN:-python}"

LEARNING_RATES=(1e-3)
BATCH_SIZES=(256)

SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-game}"

NUM_PROCS="${NUM_PROCS:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

TRAIN_PY="${SCRIPT_DIR}/game_train_user_encoder.py"

LORA_FLAGS="--use_lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.05"
BASE_FLAGS="--prefix_pos after_bos --bert_pool mean --gt_path embeds --topk 1,5,10"

TRAIN_FILE="${PROJECT_ROOT}/dataset/user_split/game/train_w_detail.json"
VAL_FILE="${PROJECT_ROOT}/dataset/user_split/game/val_w_detail.json"
CKPT_BASE="${PROJECT_ROOT}/checkpoints/game"

for bs in "${BATCH_SIZES[@]}"; do
  for lr in "${LEARNING_RATES[@]}"; do
    RUN_NAME="bs${bs}_lr${lr}"
    OUT_DIR="${CKPT_BASE}/bs_${bs}_lr_${lr}"

    "${PYTHON_BIN}" -m accelerate.commands.launch \
      --num_processes="${NUM_PROCS}" \
      --gpu_ids="${GPU_IDS}" \
      "${TRAIN_PY}" \
      --train_file "${TRAIN_FILE}" \
      --val_file "${VAL_FILE}" \
      --output_dir "${OUT_DIR}" \
      --epochs 10 \
      --batch_size "${bs}" \
      --learning_rate "${lr}" \
      --seed "${SEED}" \
      --wandb_project "${WANDB_PROJECT}" \
      --wandb_run_name "${RUN_NAME}" \
      ${LORA_FLAGS} \
      ${BASE_FLAGS}
  done
done
