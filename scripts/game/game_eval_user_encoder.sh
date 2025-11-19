#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$( cd -- "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd )"

PYTHON_BIN="${PYTHON_BIN:-python}"

TEST_JSON="${TEST_JSON:-${PROJECT_ROOT}/dataset/user_split/game/test.json}"
IMG_EMB_ROOT="${IMG_EMB_ROOT:-${PROJECT_ROOT}/data/game/image_embeddings_e5v}"
IMG_SRC_ROOT="${IMG_SRC_ROOT:-${PROJECT_ROOT}/data/game/images}"
IMG_OUT_DIR="${IMG_OUT_DIR:-${PROJECT_ROOT}/inference/game/topk_images}"

CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints/game}"

OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/inference/game/results_game}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TOP_K="${TOP_K:-10}"
SEED="${SEED:-42}"

MODEL_NAME="${MODEL_NAME:-royokong/e5-v}"
BERT_NAME="${BERT_NAME:-bert-base-uncased}"
BERT_POOL="${BERT_POOL:-mean}"
PREFIX_POS="${PREFIX_POS:-after_bos}"
PROJ_TYPE="${PROJ_TYPE:-linear}"
TEMP_INIT="${TEMP_INIT:-2.6592}"
GT_PATH="${GT_PATH:-embeds}"

mkdir -p "${OUT_DIR}" "${IMG_OUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/game_eval_user_encoder.py" \
  --test_file "${TEST_JSON}" \
  --image_embedding_dir "${IMG_EMB_ROOT}" \
  --checkpoint_dir "${CKPT_DIR}" \
  --output_dir "${OUT_DIR}" \
  --image_source_dir "${IMG_SRC_ROOT}" \
  --image_output_dir "${IMG_OUT_DIR}" \
  --top_k "${TOP_K}" \
  --batch_size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --model_name "${MODEL_NAME}" \
  --bert_model_name "${BERT_NAME}" \
  --bert_pool "${BERT_POOL}" \
  --prefix_pos "${PREFIX_POS}" \
  --proj_type "${PROJ_TYPE}" \
  --temp_init "${TEMP_INIT}" \
  --gt_path "${GT_PATH}" \
  --target_policy "all" \
  --skip_on_mismatch \
  --allow_game_names_in_context
