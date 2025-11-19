#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd )"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"


for k in "${K_VALUES[@]}"; do
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" "${SCRIPT_DIR}/media_eval_user_encoder.py" \
    --model_name "royokong/e5-v" \
    --bert_model_name "bert-base-uncased" \
    --test_file "${PROJECT_ROOT}/dataset/user_split/media/test.json" \
    --image_embedding_dir "${PROJECT_ROOT}/data/media/image_embeddings_e5v" \
    --checkpoint_dir "${PROJECT_ROOT}/checkpoints/" \
    --output_dir "${PROJECT_ROOT}/results/" \
    --image_source_dir "${PROJECT_ROOT}/data/media/images" \
    --image_output_dir "${PROJECT_ROOT}/image_results/" \
    --top_k 50 \
    --batch_size 128 \
    --gt_path embeds \
    --prefix_pos after_bos \
    --target_policy filter_last \
    --allowed_subs "favorite media;favourite media;favorite actors and directors;favourite actors and directors" \
    --skip_on_mismatch \
    --eval_gt_similarity
done
